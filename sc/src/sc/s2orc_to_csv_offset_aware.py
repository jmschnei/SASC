#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s2orc_to_csv_offset_aware.py
Convert S2ORC JSONL(.gz) shards to train/eval CSV with 10 section labels.

- Supports three schemas:
  (1) grobid_parse/pdf_parse/body_text[].{text,section}
  (2) top-level body_text[].{text/paragraph, section/header}
  (3) content.text (+ annotations.paragraph/sectionheader) or plain full text

- Maps headings to 10 coarse classes with robust regex + fallbacks.
- Never writes an empty label: uses heuristics when no heading is present.
- CLI stays the same as your original converter.

Outputs:
  <out_dir>/train.csv
  <out_dir>/eval.csv
Columns:
  text,level1,level2
"""

"""
python s2orc_to_csv_offset_aware.py \
  --input_glob "./data/s2orc/full/20250923_032917_00215_4nkum_00183017-e5aa-4a3d-93cd-cb99d0f49047.jsonl.gz" \
  --out_dir ./data/processed/ \
  --val_ratio 0.2 --seed 42 \
  --min_chars 20 --max_samples 100000
"""

import os, re, csv, gzip, json, glob, random, argparse
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple, Optional

# ---------------- Section mapping (10 classes, robust variants) ----------------
SECTION_MAP = {
  "abstract":        [r"\babstracts?\b", r"\bsummary\b", r"\bsynopsis\b"],
  "introduction":    [r"\bintroductions?\b", r"\bintro\b", r"\bbackground\b"],
  # Related work grouped into background (adjust if you want a separate class)
  "background":      [r"\brelated\s+works?\b", r"\bliterature\s+reviews?\b",
                      r"\bprior\s+work\b", r"\bstate\s+of\s+the\s+art\b"],
  "methods":         [r"\bmethods?\b",
                      r"\bmaterials?\s*(?:and|&)\s*methods?\b",
                      r"\bmethodolog(?:y|ies)\b",
                      r"\bexperimental\s+(setup|design)\b",
                      r"\bimplementation\b",
                      r"\bstatistical\s+(analysis|methods?)\b",
                      r"\bstudy\s+design\b",
                      r"\bdata\s+collection\b",
                      r"\bparticipants?\b",
                      r"\bpatients?\s+and\s+methods?\b"],
  "results":         [r"\bresults?\b",
                      r"\bmain\s+findings?\b",
                      r"\bevaluations?\b",
                      r"\bexperiments?\b",
                      r"\bresults?\s+and\s+discussion\b",
                      r"\bfindings?\b"],
  "discussion":      [r"\bdiscussions?\b",
                      r"\banalysis\b",
                      r"\berror\s+analysis\b",
                      r"\blimitations?\b",
                      r"\bdiscussion\s+and\s+results?\b"],
  "conclusion":      [r"\bconclusions?\b",
                      r"\bconcluding\s+remarks?\b",
                      r"\bsummary\s+and\s+(future\s+work|conclusions?)\b"],
  "references":      [r"\breferences\b",
                      r"\bbibliograph(?:y|ies)\b",
                      r"\bworks\s+cited\b"],
  "acknowledgments": [r"\backnowledg(?:e)?ments?\b",
                      r"\bthanks\b",
                      r"\bfunding\b"],
  "appendix":        [r"\bappendix\b",
                      r"\bappendices\b",
                      r"\bsupplement(?:ary|al)\b"],
}
SECTION_REGEX = {k: [re.compile(p, re.I) for p in v] for k, v in SECTION_MAP.items()}

# Heading-at-line-start detector used for plain full text
HEADING_RE = re.compile(
    r"(?im)^\s*(?:\d+(?:\.\d+)*[.)]?\s+)?"
    r"(abstract|introduction|related\s+work|literature\s+review|background|"
    r"materials\s+and\s+methods|methods?|methodology|experimental\s+(?:setup|design)|implementation|"
    r"results?|evaluation|experiments?|discussion|analysis|error\s+analysis|"
    r"conclusions?|concluding\s+remarks|summary(?:\s+and\s+(?:future\s+work|conclusions?))?|"
    r"acknowledg(?:e)?ments?|references|bibliograph(?:y|ies)|works\s+cited|appendix|appendices|supplement(?:ary|al))"
    r"\s*:?\s*$"
)

# ---------------- Utilities ----------------
def _json_loads_maybe(s):
    if s is None:
        return None
    if isinstance(s, (list, dict)):
        return s
    if isinstance(s, str):
        try:
            return json.loads(s)
        except Exception:
            return None
    return None

def _parse_spans(v) -> List[Tuple[int, int]]:
    """Accepts a JSON string or list of {'start','end'} dicts. Returns sorted (s,e) spans."""
    v = _json_loads_maybe(v)
    if not isinstance(v, list):
        return []
    out: List[Tuple[int, int]] = []
    for it in v:
        try:
            s = int(it.get("start")); e = int(it.get("end"))
            if 0 <= s < e:
                out.append((s, e))
        except Exception:
            pass
    return sorted(out, key=lambda x: x[0])

def _walk(obj):
    """Yield every dict node in a nested JSON structure (depth-first)."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)

def _long_text_in_node(d: Dict[str, Any]) -> Optional[str]:
    """Find a long plain-text field within a dict node."""
    for k in ("content", "full_text", "body", "text"):
        v = d.get(k)
        if isinstance(v, str) and len(v) > 1000:
            return v
    return None

def _get_node_or_ann(d: Dict[str, Any], key: str):
    """Return d[key] or d['annotations'][key] if present."""
    if key in d:
        return d.get(key)
    ann = d.get("annotations") or {}
    if isinstance(ann, dict) and key in ann:
        return ann.get(key)
    return None

# ---------------- Extraction paths ----------------
def iter_primary_sections(rec: Dict[str, Any]) -> Iterable[Dict[str, str]]:
    """Yield {'text','header'} using body_text from grobid/pdf/top-level if available."""
    # abstracts (optional)
    for a in rec.get("abstract", []) or []:
        txt = (a.get("text") or "").strip()
        if txt:
            yield {"text": txt, "header": "abstract"}

    def _bt(obj): return (obj or {}).get("body_text") or []
    cands = []
    if "grobid_parse" in rec: cands.extend(_bt(rec.get("grobid_parse")))
    if "pdf_parse"   in rec: cands.extend(_bt(rec.get("pdf_parse")))
    if isinstance(rec.get("body_text"), list): cands.extend(rec["body_text"])

    for it in cands:
        txt = (it.get("text") or it.get("paragraph") or it.get("content") or it.get("body") or "").strip()
        hdr = (it.get("section") or it.get("header") or it.get("heading") or
               it.get("title") or it.get("section_title") or it.get("sec_title") or "").strip()
        if txt:
            yield {"text": txt, "header": hdr}

def iter_offsets_or_plain_content(rec: Dict[str, Any]) -> Iterable[Dict[str, str]]:
    """
    Fallback extractor:
    - content.text + (annotations.paragraph / sectionheader) via offsets
    - else: content.text split by HEADING_RE
    - else: whole text → 'introduction'
    """
    for node in _walk(rec):
        whole = _long_text_in_node(node)
        if not whole:
            continue

        # Get paragraph/header spans from node OR node['annotations']
        para_spans = _parse_spans(
            _get_node_or_ann(node, "paragraph") or _get_node_or_ann(node, "paragraphs")
        )
        header_spans = _parse_spans(
            _get_node_or_ann(node, "sectionheader") or
            _get_node_or_ann(node, "section_header") or
            _get_node_or_ann(node, "headers") or
            _get_node_or_ann(node, "heading_spans")
        )

        # A) offsets variant
        if para_spans:
            headers: List[Tuple[int,int,str]] = []
            if header_spans:
                for s, e in header_spans:
                    try:
                        headers.append((s, e, whole[s:e].strip()))
                    except Exception:
                        pass
                headers.sort(key=lambda x: x[0])

            for ps, pe in para_spans:
                try:
                    txt = whole[ps:pe].strip()
                except Exception:
                    continue
                if not txt:
                    continue
                htxt = ""
                if headers:
                    prev = [h for h in headers if h[0] <= ps]
                    if prev:
                        htxt = prev[-1][2]
                yield {"text": txt, "header": htxt}
            return  # done for this record

        # B) plain content: split by heading lines
        heads = [(m.start(), m.end(), m.group(1)) for m in HEADING_RE.finditer(whole)]
        if heads:
            heads.sort(key=lambda x: x[0])
            for i, (s, e, label) in enumerate(heads):
                seg_s = e
                seg_e = heads[i + 1][0] if i + 1 < len(heads) else len(whole)
                seg = whole[seg_s:seg_e].strip()
                if seg:
                    yield {"text": seg, "header": label}
            return

        # C) no headings at all → whole as 'introduction'
        t = whole.strip()
        if t:
            yield {"text": t, "header": "introduction"}
        return

def iter_sections(rec: Dict[str, Any]) -> Iterable[Dict[str, str]]:
    """Unified generator that tries primary, then fallback."""
    yielded = False
    for seg in iter_primary_sections(rec):
        yielded = True
        yield seg
    if not yielded:
        for seg in iter_offsets_or_plain_content(rec):
            yield seg

# ---------------- Labeling with fallbacks (never empty) ----------------
def map_label(header: str, text: str) -> str:
    h = (header or "").strip().lower()
    # 1) direct heading mapping
    for lbl, pats in SECTION_REGEX.items():
        if any(p.search(h) for p in pats):
            return lbl

    # 2) content-based heuristics (first ~200 chars)
    t = (text or "")[:200].lower()
    # references likely contain DOIs, years, journal cues
    if "reference" in t or "bibliograph" in t or "doi" in t or "vol." in t:
        return "references"
    if "acknowledg" in t or "funding" in t or "grant" in t:
        return "acknowledgments"
    if re.search(r"\bwe (propose|present|introduce)\b", t):
        return "introduction"
    if "method" in t or "we use" in t or "we used" in t or "materials" in t or "participants" in t:
        return "methods"
    if "results" in t or "we found" in t or "significant" in t or "p=" in t or "p <" in t:
        return "results"
    if "discussion" in t or "implications" in t or "limitations" in t:
        return "discussion"
    if "conclusion" in t or "in summary" in t or "in conclusion" in t:
        return "conclusion"

    # 3) final fallback
    return "introduction"

# ---------------- Main: JSONL(.gz) → CSV ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", required=True, help="Glob for .jsonl or .jsonl.gz files")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_chars", type=int, default=60)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train_2.csv"
    eval_path  = out_dir / "eval_2.csv"

    ftr = open(train_path, "w", encoding="utf-8", newline="")
    fev = open(eval_path,  "w", encoding="utf-8", newline="")
    wtr = csv.DictWriter(ftr, fieldnames=["text", "level1", "level2"])
    wte = csv.DictWriter(fev, fieldnames=["text", "level1", "level2"])
    wtr.writeheader(); wte.writeheader()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        print(f"[ERROR] No files matched: {args.input_glob}")
        return

    kept_total = 0
    kept_by_label = {k: 0 for k in SECTION_MAP}

    def write_row(text: str, label: str):
        nonlocal kept_total
        if args.max_samples and kept_total >= args.max_samples:
            return False
        row = {"text": text, "level1": label, "level2": ""}
        (wte if rng.random() < args.val_ratio else wtr).writerow(row)
        kept_total += 1
        kept_by_label[label] = kept_by_label.get(label, 0) + 1
        return True

    # process
    for p in files:
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8", errors="ignore") as fin:
            for line in fin:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                for seg in iter_sections(rec):
                    txt = (seg.get("text") or "").strip()
                    if len(txt) < args.min_chars:
                        continue
                    lbl = map_label(seg.get("header", ""), txt)
                    if not write_row(txt, lbl):
                        break

    ftr.close(); fev.close()
    print(f"[OK] Wrote {train_path} and {eval_path}")
    nonzero = {k: v for k, v in kept_by_label.items() if v > 0}
    print("[Stats] Label counts:", nonzero)
    if kept_total == 0:
        print("[WARN] No rows collected. Try lowering --min_chars or check schema.", flush=True)

if __name__ == "__main__":
    main()
