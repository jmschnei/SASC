#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2ORC → 10-class CSV (text,level1,level2)
- 输入：S2ORC JSONL 或 JSONL.GZ（支持通配符）
- 输出：train.csv / eval.csv（带表头；level2 留空）
- 兼容旧/新 schema：
    * 顶层 abstract 列表
    * grobid_parse.body_text / pdf_parse.body_text / 顶层 body_text
    * 每个段目标取字段：text (+ section/header)

usage：
python s2orc_to_csv.py \
  --input_glob sample_s2orc_long.jsonl.gz \
  --out_dir ./data/s2orc/csv/ \
  --val_ratio 0.2 \
  --seed 42 \
  --min_chars 20 \
  --max_samples 0

"""
import os, re, csv, json, gzip, glob, random, argparse
from typing import Iterable, Dict, Any

SECTION_MAP = {
  "abstract":        [r"\babstract\b", r"\bsummary\b"],
  "introduction":    [r"\bintroduction\b", r"\bintro\b", r"\bbackground\b"],
  "background":    [r"\brelated work\b", r"\bliterature review\b", r"\bprior work\b", r"\bstate of the art\b"],
  "methods":         [r"\bmethods?\b", r"\bmaterials? and methods?\b", r"\bmethodology\b", r"\bexperimental (setup|design)\b", r"\bimplementation\b"],
  "results":         [r"\bresults?\b", r"\bfindings\b", r"\bevaluation\b", r"\bexperiments?\b"],
  "discussion":      [r"\bdiscussion\b", r"\banalysis\b", r"\bdiscussion and.*results\b", r"\berror analysis\b"],
  "conclusion":      [r"\bconclusions?\b", r"\bconcluding remarks\b", r"\bsummary and (future work|conclusions?)\b"],
  "references":      [r"\breferences\b", r"\bbibliograph(y|ies)\b", r"\bworks cited\b"],
  "acknowledgments": [r"\backnowledg(e)?ments?\b", r"\bthanks\b"],
  "appendix":        [r"\bappendix\b", r"\bappendices\b", r"\bsupplement(ary|al)\b"]
}

SECTION_REGEX = {k: [re.compile(p, flags=re.I) for p in v] for k, v in SECTION_MAP.items()}
TEN_LABELS = list(SECTION_MAP.keys())

def norm_header(h: str) -> str:
    if not h: return ""
    h = h.strip().lower()
    # normalize punctuation/whitespace
    h = re.sub(r"[_\-:]+", " ", h)
    h = re.sub(r"\s+", " ", h)
    return h

def map_to_label(header: str) -> str:
    h = norm_header(header)
    for label, plist in SECTION_REGEX.items():
        if any(p.search(h) for p in plist):
            return label
    return ""  # unknown

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def iter_sections(rec: Dict[str, Any]) -> Iterable[Dict[str, str]]:
    # abstracts
    for a in rec.get("abstract", []) or []:
        txt = (a.get("text") or "").strip()
        if txt:
            yield {"text": txt, "header": "abstract"}

    # body candidates
    def body_items(obj):
        if not obj: return []
        return obj.get("body_text") or []

    candidates = []
    # new/old schema possibilities
    for key in ["grobid_parse", "pdf_parse"]:
        if key in rec: candidates.extend(body_items(rec.get(key)))
    if rec.get("body_text"): candidates.extend(rec["body_text"])

    for it in candidates:
        txt = (it.get("text") or it.get("paragraph") or "").strip()
        hdr = (it.get("section") or it.get("header") or "").strip()
        if txt:
            yield {"text": txt, "header": hdr}

def collect_rows(files: Iterable[str], min_chars:int, max_samples:int) -> Iterable[Dict[str,str]]:
    rng = random.Random(1234)
    kept = 0
    for fp in files:
        for rec in read_jsonl(fp):
            for seg in iter_sections(rec):
                txt = seg["text"]
                if len(txt) < min_chars: continue
                label = map_to_label(seg["header"])
                if not label: continue
                yield {"text": txt, "level1": label, "level2": ""}
                kept += 1
                if max_samples and kept >= max_samples:
                    return

def write_csv(rows: Iterable[Dict[str,str]], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text","level1","level2"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", required=True, help="Glob for S2ORC jsonl/jsonl.gz files, e.g. '/data/s2orc/*/*.jsonl.gz'")
    ap.add_argument("--out_dir", default=".", help="Where to write train.csv / eval.csv")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_chars", type=int, default=60, help="drop very short snippets")
    ap.add_argument("--max_samples", type=int, default=0, help="cap total samples (0 means all)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise SystemExit(f"No files matched: {args.input_glob}")

    rows = list(collect_rows(files, args.min_chars, args.max_samples))
    if not rows:
        raise SystemExit("No rows collected. Check mapping rules or min_chars.")

    # stratified-ish split by label
    from collections import defaultdict
    rng = random.Random(args.seed)
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["level1"]].append(r)
    train, dev = [], []
    for lbl, lst in buckets.items():
        rng.shuffle(lst)
        k = int(len(lst) * (1.0 - args.val_ratio))
        train.extend(lst[:k]); dev.extend(lst[k:])

    rng.shuffle(train); rng.shuffle(dev)

    out_train = os.path.join(args.out_dir, "train.csv")
    out_eval  = os.path.join(args.out_dir, "eval.csv")
    write_csv(train, out_train)
    write_csv(dev,   out_eval)

    # quick stats
    from collections import Counter
    def stats(name, data):
        c = Counter([r["level1"] for r in data])
        tot = len(data)
        top = ", ".join([f"{k}:{v}" for k,v in c.most_common(10)])
        return f"{name}: {tot} rows | {top}"
    print(stats("train", train))
    print(stats("eval",  dev))
    print("labels:", ", ".join(TEN_LABELS))
    print("Wrote:", out_train, "and", out_eval)

if __name__ == "__main__":
    main()
