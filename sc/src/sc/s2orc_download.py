#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s2orc_download.py
Download S2ORC JSONL(.gz) shards via the Semantic Scholar Datasets API.

- Uses endpoint: /datasets/v1/release/{release_id}/dataset/{dataset_name}
- Honors API rate limit (~1 req/sec) for metadata calls
- Parallel S3 downloads (not API-limited)
- Resume via HTTP Range; retries with backoff
- Auto-refresh expired presigned URLs (HTTP 403)
- Works whether 'files' are URL strings OR objects

Usage:
  export S2_API_KEY="YOUR_KEY"
  python s2orc_download.py --out_dir ./data/s2orc/full --num-shards 3 --workers 8
  python s2orc_download.py --out_dir ./data/s2orc/full --slice 100:150 --workers 8
  # ALL shards (HUGE):
  python download_s2orc.py --out_dir ./data/s2orc/full --workers 8
"""
import os, sys, time, json, requests, threading
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, unquote

BASE = "https://api.semanticscholar.org/datasets/v1"
API_MIN_INTERVAL = float(os.environ.get("S2_API_MIN_INTERVAL", "1.1"))  # seconds
_api_lock = threading.Lock()
_last_api_call = 0.0

def _api_get(url: str, headers: Dict[str, str] | None = None) -> requests.Response:
    """GET with global ~1 rps gate for the Datasets API."""
    global _last_api_call
    with _api_lock:
        wait = _last_api_call + API_MIN_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        r = requests.get(url, headers=headers or {}, timeout=60)
        _last_api_call = time.time()
    r.raise_for_status()
    return r

def die(msg: str):
    print(msg, file=sys.stderr); sys.exit(1)

def get_api_key(cli_key: Optional[str]) -> str:
    key = cli_key or os.environ.get("S2_API_KEY")
    if not key:
        die("Set S2_API_KEY env var or pass --api-key.")
    return key

def human(n: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.1f}{u}"
        x /= 1024.0

def parse_slice(s: str) -> Tuple[int,int]:
    if ":" not in s: die("--slice must look like START:END (END not included)")
    a,b = s.split(":",1)
    start = int(a) if a else 0
    end   = int(b) if b else 0
    if end and end <= start: die("--slice END must be > START")
    return start, end

# ---------- list datasets & fetch file links ----------
def get_release_id() -> str:
    # Prefer explicit latest date from /release (returns list), else fallback "latest"
    try:
        data = _api_get(f"{BASE}/release").json()
        if isinstance(data, list) and data:
            return data[-1]
    except Exception:
        pass
    return "latest"

def list_datasets_for_release(release_id: str) -> List[Dict[str, Any]]:
    resp = _api_get(f"{BASE}/release/{release_id}").json()
    if isinstance(resp, dict) and "datasets" in resp:
        return resp["datasets"]
    if isinstance(resp, list):
        return resp
    die(f"Unexpected datasets response for release {release_id}: {type(resp)}")

def find_dataset_name(datasets: List[Dict[str,Any]], prefer: str = "s2orc") -> str:
    names = []
    for d in datasets:
        name = d.get("name") or d.get("datasetName") or ""
        if name: names.append(name)
    for n in names:
        if n.lower() == prefer:
            return n
    for n in names:
        if prefer in n.lower():
            return n
    die(f"Could not find an S2ORC dataset in: {names}")

def get_dataset_meta(release_id: str, dataset_name: str, api_key: str) -> Dict[str, Any]:
    H = {"x-api-key": api_key}
    url = f"{BASE}/release/{release_id}/dataset/{dataset_name}"
    return _api_get(url, headers=H).json()

def _basename_from_url(u: str) -> str:
    p = urlparse(u)
    base = Path(unquote(p.path)).name
    # if URL path ends with '/', fall back to query param hint (rare)
    return base or "shard.jsonl.gz"

def normalize_files(raw_files: Any) -> List[Dict[str,Any]]:
    out: List[Dict[str,Any]] = []
    if isinstance(raw_files, list):
        for i, it in enumerate(raw_files):
            if isinstance(it, str):
                out.append({"index": i, "url": it, "filename": _basename_from_url(it)})
            elif isinstance(it, dict):
                u = it.get("url") or it.get("s3Url") or it.get("href")
                fn = it.get("filename") or it.get("fileName") or ( _basename_from_url(u) if u else None )
                if not u or not fn:
                    # skip malformed entries
                    continue
                out.append({"index": i, "url": u, "filename": fn})
    return out

def head_content_length(url: str) -> Optional[int]:
    try:
        r = requests.head(url, timeout=30, allow_redirects=True)
        r.raise_for_status()
        cl = r.headers.get("Content-Length")
        return int(cl) if cl is not None else None
    except Exception:
        return None

# ---------- download one shard (resume + refresh) ----------
def refresh_single_file_url(release_id: str, dataset_name: str, filename: str, index: int, api_key: str) -> Optional[str]:
    """Refetch the dataset file list and return a fresh URL (match by filename, fallback by index)."""
    meta = get_dataset_meta(release_id, dataset_name, api_key)
    files = normalize_files(meta.get("files") or [])
    # try exact filename match first
    for f in files:
        if f.get("filename") == filename and f.get("url"):
            return f["url"]
    # fallback: same index
    for f in files:
        if f.get("index") == index and f.get("url"):
            return f["url"]
    return None

def download_one(file_item: Dict[str,Any], out_dir: Path, *,
                 release_id: str, dataset_name: str, api_key: str,
                 timeout: int, chunk_mb: int, retries: int) -> str:
    url, fname, fidx = file_item["url"], file_item["filename"], file_item["index"]
    dst = out_dir / fname
    if dst.exists() and dst.stat().st_size > 0:
        return f"exists: {fname}"

    tmp = dst.with_suffix(dst.suffix + ".part")
    chunk = 1024 * 1024 * max(1, chunk_mb)
    attempt = 0
    expected = head_content_length(url)

    while attempt <= retries:
        attempt += 1
        try:
            start = tmp.stat().st_size if tmp.exists() else 0
            headers = {"Range": f"bytes={start}-"} if start > 0 else None

            with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                if r.status_code == 403:
                    # URL expired → refresh and retry
                    new_url = refresh_single_file_url(release_id, dataset_name, fname, fidx, api_key)
                    if not new_url:
                        raise RuntimeError(f"Cannot refresh URL for {fname}")
                    url = new_url
                    continue
                if r.status_code == 416:
                    # Range not satisfiable: if tmp already complete, finalize; else restart
                    if expected is not None and tmp.exists() and tmp.stat().st_size == expected:
                        tmp.replace(dst)
                        return f"done:   {fname} ({human(dst.stat().st_size)})"
                    if tmp.exists(): tmp.unlink(missing_ok=True)
                    continue
                r.raise_for_status()
                mode = "ab" if start > 0 else "wb"
                with open(tmp, mode) as w:
                    for chunk_bytes in r.iter_content(chunk_size=chunk):
                        if chunk_bytes:
                            w.write(chunk_bytes)

            if expected is not None:
                got = tmp.stat().st_size
                if got != expected:
                    raise IOError(f"size mismatch (got {got}, want {expected})")

            tmp.replace(dst)
            sz = dst.stat().st_size
            return f"done:   {fname} ({human(sz)})"

        except Exception as e:
            if attempt > retries:
                return f"FAIL:   {fname} ({e})"
            time.sleep(min(60, 2 ** attempt))  # backoff

# ---------- main ----------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Download S2ORC JSONL(.gz) shards only.")
    ap.add_argument("--api-key", default=None, help="Semantic Scholar Datasets API key (or set S2_API_KEY).")
    ap.add_argument("--out_dir", required=True, help="Where to save .jsonl.gz shards")
    ap.add_argument("--num-shards", type=int, default=0, help="Download first N shards (0 = all or --slice)")
    ap.add_argument("--slice", type=str, default=None, help="Shard index slice START:END (END not included)")
    ap.add_argument("--workers", type=int, default=4, help="Parallel downloads for S3 URLs")
    ap.add_argument("--retries", type=int, default=3, help="Retries per file")
    ap.add_argument("--timeout", type=int, default=600, help="Per-request timeout (seconds)")
    ap.add_argument("--chunk-mb", type=int, default=1, help="Download chunk size in MB")
    ap.add_argument("--list-only", action="store_true", help="List shard filenames and exit (no download)")
    ap.add_argument("--save-urls", default=None, help="Also write selected URLs to this file")
    ap.add_argument("--dataset", default="s2orc", help="Dataset name (default: s2orc). If unsure, leave as is.")
    args = ap.parse_args()

    api_key = get_api_key(args.api_key)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    rid = get_release_id()
    datasets = list_datasets_for_release(rid)
    ds_name = args.dataset or "s2orc"
    chosen = ds_name if ds_name.lower() != "s2orc" else find_dataset_name(datasets, prefer="s2orc")
    print(f"[INFO] Release: {rid} | dataset: {chosen}")

    meta = get_dataset_meta(rid, chosen, api_key)
    files = normalize_files(meta.get("files") or [])
    total = len(files)
    print(f"[INFO] available shards: {total}")

    # Select subset
    if args.slice:
        start, end = parse_slice(args.slice)
        end = end or total
        sel = files[start:end]
        print(f"[INFO] using slice {start}:{end} → {len(sel)} shards")
    elif args.num_shards and args.num_shards > 0:
        sel = files[:args.num_shards]
        print(f"[INFO] using first {len(sel)} shards")
    else:
        sel = files
        print(f"[WARN] no --num-shards/--slice provided → ALL {len(sel)} shards will be downloaded")

    if args.save_urls:
        with open(args.save_urls, "w", encoding="utf-8") as f:
            for it in sel:
                f.write(it["url"] + "\n")
        print(f"[INFO] wrote URLs to {args.save_urls}")

    if args.list_only:
        for i, it in enumerate(sel):
            print(f"{i:05d} {it['filename']}")
        return

    print(f"[INFO] downloading → {out_dir} | workers={args.workers}")
    t0 = time.time()
    jobs = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for it in sel:
            jobs.append(ex.submit(
                download_one, it, out_dir,
                release_id=rid, dataset_name=chosen, api_key=api_key,
                timeout=args.timeout, chunk_mb=args.chunk_mb, retries=args.retries
            ))
        for fut in as_completed(jobs):
            print(fut.result(), flush=True)
    dt = time.time() - t0
    print(f"[INFO] completed in {dt/60:.1f} min → files in {out_dir}")

if __name__ == "__main__":
    main()
