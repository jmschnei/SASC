#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s2orc_sections_minimal.py

Output ONLY two columns: text,section_name
- Supports:
  (1) grobid_parse/pdf_parse/body_text[].{text, section}
  (2) top-level body_text[].{text/paragraph, section/header}
  (3) content.text + annotations.{paragraph,sectionheader} (offsets)
      * expand numbered headers ("1.") to whole line / next non-empty line
      * if still numeric-only, auto-title from the first paragraph
      * group multiple paragraphs under the same header
  (4) plain full text fallback: split by heading lines via regex

Writes:
  <out_dir>/train.csv  (text,section_name)
  <out_dir>/eval.csv   (text,section_name)
"""

"""
python s2orc_sections_with_names.py \
  --input_glob "./data/s2orc/full/*.jsonl.gz" \
  --out_dir ./data/clear/ \
  --val_ratio 0.2 --seed 42 \
  --min_chars 300 --min_words 60 \
  --group_offsets \
  --drop_no_header

"""

import re, csv, gzip, json, glob, bisect, argparse, hashlib
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple, Optional

# Fallback: detect heading lines in plain text
HEADING_LINE_RE = re.compile(
    r"(?im)^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)?\s*[.)-]?\s*"
    r"(abstract|introduction|related\s+work|literature\s+review|background|"
    r"materials\s+and\s+methods|methods?|methodology|experimental\s+(?:setup|design)|implementation|"
    r"results?|evaluation|experiments?|discussion|analysis|error\s+analysis|"
    r"conclusions?|concluding\s+remarks|summary(?:\s+and\s+(?:future\s+work|conclusions?))?|"
    r"acknowledg(?:e)?ments?|references|bibliograph(?:y|ies)|works\s+cited|appendix|appendices|supplement(?:ary|al))"
    r".*$"
)

def _json_loads_maybe(v):
    if v is None: return None
    if isinstance(v, (list, dict)): return v
    if isinstance(v, str):
        try: return json.loads(v)
        except Exception: return None
    return None

def _parse_spans(v) -> List[Tuple[int, int]]:
    v = _json_loads_maybe(v)
    out: List[Tuple[int,int]] = []
    if isinstance(v, list):
        for it in v:
            try:
                s = int(it.get("start")); e = int(it.get("end"))
                if 0 <= s < e: out.append((s, e))
            except Exception: pass
    return sorted(out, key=lambda x: x[0])

def _walk(obj):
    if isinstance(obj, dict):
        yield obj
        for val in obj.values():
            yield from _walk(val)
    elif isinstance(obj, list):
        for val in obj:
            yield from _walk(val)

def _long_text_in_node(d: Dict[str,Any]) -> Optional[str]:
    for k in ("content","full_text","body","text"):
        v = d.get(k)
        if isinstance(v, str) and len(v) > 1000:
            return v
    return None

def _get_node_or_ann(d: Dict[str,Any], key: str):
    if key in d: return d.get(key)
    ann = d.get("annotations") or {}
    if isinstance(ann, dict) and key in ann: return ann.get(key)
    return None

def normalize_header(h: str) -> str:
    if not h: return ""
    x = h.strip()
    x = re.sub(r'^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)\s*[.)-]?\s*', '', x, flags=re.I)  # strip leading numbering
    x = re.sub(r'[:\-–—\s]+$', '', x)                                            # trailing delimiters
    x = re.sub(r'\s+', ' ', x).lower()
    return x

def iter_bodytext(rec: Dict[str,Any]) -> Iterable[Dict[str,str]]:
    # optional abstract
    for a in rec.get("abstract", []) or []:
        txt = (a.get("text") or "").strip()
        if txt:
            yield {"text": txt, "header": "Abstract"}

    def _bt(obj): return (obj or {}).get("body_text") or []
    cands=[]
    if "grobid_parse" in rec: cands += _bt(rec.get("grobid_parse"))
    if "pdf_parse"   in rec: cands += _bt(rec.get("pdf_parse"))
    if isinstance(rec.get("body_text"), list): cands += rec["body_text"]

    for it in cands:
        txt = (it.get("text") or it.get("paragraph") or it.get("content") or it.get("body") or "").strip()
        hdr = (it.get("section") or it.get("header") or it.get("heading") or
               it.get("title") or it.get("section_title") or it.get("sec_title") or "").strip()
        if txt:
            yield {"text": txt, "header": hdr}

def iter_content_with_headers(rec: Dict[str,Any], group_offsets: bool=True) -> Iterable[Dict[str,str]]:
    """Use offsets if present; else split by heading regex."""
    for node in _walk(rec):
        whole = _long_text_in_node(node)
        if not whole: 
            continue

        para_spans = _parse_spans(_get_node_or_ann(node,"paragraph") or _get_node_or_ann(node,"paragraphs"))
        hdr_spans  = _parse_spans(
            _get_node_or_ann(node,"sectionheader") or _get_node_or_ann(node,"section_header") or
            _get_node_or_ann(node,"headers") or _get_node_or_ann(node,"heading_spans")
        )

        if para_spans:
            hdr_starts = sorted([s for s, _ in hdr_spans]) if hdr_spans else []
            NUMERIC_HDR_RE = re.compile(r"^[\s\.\)\(0-9IVXLC\-–—]+$")

            def next_nonempty_line(start_pos: int) -> str:
                i, N = start_pos, len(whole)
                while i < N and whole[i] in ("\n","\r"," ","\t"):
                    i += 1
                if i >= N: return ""
                j = whole.find("\n", i)
                if j == -1: j = min(N, i+300)
                return whole[i:j].strip()

            def line_at(pos: int) -> str:
                prev_nl = whole.rfind("\n", 0, max(0, pos))
                line_start = prev_nl + 1 if prev_nl != -1 else 0
                nl = whole.find("\n", pos)
                line_end = nl if nl != -1 else min(len(whole), pos+300)
                line = whole[line_start:line_end].strip()
                if len(line) < 5 or NUMERIC_HDR_RE.fullmatch(line):
                    nxt = next_nonempty_line(line_end + 1)
                    if nxt: line = (line + " " + nxt).strip()
                return line

            def header_for(pos: int) -> str:
                if not hdr_starts: return ""
                i = bisect.bisect_right(hdr_starts, pos) - 1
                if i < 0: return ""
                return line_at(hdr_starts[i])

            def auto_title_if_numeric(header_text: str, first_para_text: str) -> str:
                h = (header_text or "").strip()
                if not h or len(h) < 5 or NUMERIC_HDR_RE.fullmatch(h):
                    s = (first_para_text or "").strip()
                    m = re.search(r"([.?!])\s", s)
                    return (s[:m.end()].strip() if m else " ".join(s.split()[:12])) or "unknown"
                return h

            if group_offsets:
                cur_h, buf, cur_first_para = None, [], ""
                for ps, pe in para_spans:
                    try:
                        seg = whole[ps:pe].strip()
                    except Exception:
                        continue
                    if not seg: continue
                    htxt = header_for(ps)
                    if cur_h is None:
                        cur_h, cur_first_para = htxt, seg
                    if htxt != cur_h and buf:
                        final_h = auto_title_if_numeric(cur_h, cur_first_para)
                        yield {"text": " ".join(buf).strip(), "header": final_h}
                        buf, cur_h, cur_first_para = [], htxt, seg
                    buf.append(seg)
                if buf:
                    final_h = auto_title_if_numeric(cur_h, cur_first_para)
                    yield {"text": " ".join(buf).strip(), "header": final_h}
            else:
                for ps, pe in para_spans:
                    try:
                        seg = whole[ps:pe].strip()
                    except Exception:
                        continue
                    if not seg: continue
                    htxt = header_for(ps)
                    final_h = auto_title_if_numeric(htxt, seg)
                    yield {"text": seg, "header": final_h}
            return

        # fallback: heading regex in plain content
        heads = [m for m in HEADING_LINE_RE.finditer(whole)]
        if heads:
            heads.sort(key=lambda m:m.start())
            for i,m in enumerate(heads):
                seg_s = m.end()
                seg_e = heads[i+1].start() if i+1 < len(heads) else len(whole)
                seg = whole[seg_s:seg_e].strip()
                if seg:
                    yield {"text": seg, "header": m.group(0).strip()}
            return

        # no headings → skip this node
        return

def stable_eval_split(paper_key: str, val_ratio: float, seed: int=42) -> bool:
    h = hashlib.md5((paper_key + str(seed)).encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 1000
    return bucket < int(val_ratio * 1000)   # True → eval

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_chars", type=int, default=300)
    ap.add_argument("--min_words", type=int, default=60)
    ap.add_argument("--group_offsets", action="store_true",
                    help="Group multiple paragraphs under the same header")
    ap.add_argument("--drop_no_header", action="store_true",
                    help="Drop samples with empty header after expansion/auto-title")
    ap.add_argument("--use_norm", action="store_true",
                    help="If set, write normalized section name; else write raw/auto title")
    ap.add_argument("--max_samples", type=int, default=0, help="0 = no cap")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    p_tr, p_ev = out/"train_2.csv", out/"eval_2.csv"

    ftr = open(p_tr, "w", encoding="utf-8", newline="")
    fev = open(p_ev, "w", encoding="utf-8", newline="")
    wtr = csv.DictWriter(ftr, fieldnames=["text","section_name"])
    wte = csv.DictWriter(fev, fieldnames=["text","section_name"])
    wtr.writeheader(); wte.writeheader()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        print(f"[ERROR] No files matched: {args.input_glob}")
        return

    n_kept = 0

    def write_row(text: str, header_raw: str, paper_key: str):
        nonlocal n_kept
        if args.max_samples and n_kept >= args.max_samples:
            return False
        sec = normalize_header(header_raw) if args.use_norm else header_raw
        if args.drop_no_header and not sec.strip():
            return True  # skip silently but continue
        row = {"text": text, "section_name": sec.strip()}
        (wte if stable_eval_split(paper_key, args.val_ratio, args.seed) else wtr).writerow(row)
        n_kept += 1
        return True

    for p in files:
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8", errors="ignore") as fin:
            for line in fin:
                if not line.strip(): continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                paper_id = str(rec.get("paper_id") or rec.get("corpusid") or rec.get("id") or "").strip()
                if not paper_id:
                    title = str(rec.get("title") or "")
                    pdfsha = str(((rec.get("content") or {}).get("source") or {}).get("pdfsha") or "")
                    paper_id = hashlib.md5((title + "|" + pdfsha).encode("utf-8")).hexdigest()

                used = False
                # 1) body_text first
                for seg in iter_bodytext(rec):
                    txt = (seg.get("text") or "").strip()
                    hdr = (seg.get("header") or "").strip()
                    if len(txt) < args.min_chars or len(txt.split()) < args.min_words:
                        continue
                    if not write_row(txt, hdr, paper_id):
                        break
                    used = True
                if args.max_samples and n_kept >= args.max_samples:
                    break

                # 2) fallback: offsets/plain headings
                if not used:
                    for seg in iter_content_with_headers(rec, group_offsets=args.group_offsets):
                        txt = (seg.get("text") or "").strip()
                        hdr = (seg.get("header") or "").strip()
                        if args.drop_no_header and not hdr:
                            continue
                        if len(txt) < args.min_chars or len(txt.split()) < args.min_words:
                            continue
                        if not write_row(txt, hdr, paper_id):
                            break
                if args.max_samples and n_kept >= args.max_samples:
                    break

    ftr.close(); fev.close()
    print(f"[OK] wrote:\n  {p_tr}\n  {p_ev}\n[STATS] samples: {n_kept}")

if __name__ == "__main__":
    main()
