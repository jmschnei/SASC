import os
import re
import json
import gzip
from pathlib import Path
from difflib import SequenceMatcher

import tensorflow_datasets as tfds
import getpass
import pathlib

# =========================
# 1) Put TFDS data on a large disk
# =========================
user = os.environ.get("USER") or getpass.getuser()   # often 'root' in containers
if user == "root":
    user = os.environ.get("DATA_USER", "xshuiai")        # allow override via DATA_USER

BASE = f"/netscratch/{user}"
for d in [f"{BASE}/tfds_data", f"{BASE}/tfds_download", f"{BASE}/tfds_tmp"]:
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)

os.environ["TFDS_DATA_DIR"] = f"{BASE}/tfds_data"
os.environ["TFDS_DOWNLOAD_DIR"] = f"{BASE}/tfds_download"
os.environ["TFDS_TMP_DIR"] = f"{BASE}/tfds_tmp"

# =========================
# 2) Configuration
# =========================
DATASET = "scientific_papers/arxiv"   # or "scientific_papers/pubmed"
SPLIT_BASE = "train"
# Start small to validate; scale up later to "train"
SPLIT = os.environ.get("TFDS_SPLIT", f"{SPLIT_BASE}[:2%]")

SOURCE_NAME = "arxiv" if "arxiv" in DATASET else "pubmed"
OUT = Path(f"sections_10labels_{SOURCE_NAME}_{SPLIT_BASE}.jsonl.gz")

# Only limits how many papers process after loading (does not reduce TFDS build cost)
MAX_PAPERS = 10  # set to None for no cap

LABEL2ID = {
    'abstract':0,'introduction':1,'background':2,'methods':3,'results':4,
    'discussion':5,'conclusion':6,'references':7,'acknowledgments':8,'supplementary':9,
}
RULES = [
    (r"^(abstract|summary)$",'abstract'),
    (r"^(introduction|intro|overview)$",'introduction'),
    (r"^(background|related work|literature review|prior work|preliminaries)$",'background'),
    (r"^(methods?|materials and methods|methodology|experimental setup|experiments?|implementation|dataset)$",'methods'),
    (r"^(results?|findings|evaluation|performance)$",'results'),
    (r"^(discussion|analysis|interpretation)$",'discussion'),
    (r"^(conclusion|conclusions|future work|limitations?)$",'conclusion'),
    (r"^(references|bibliography|works cited)$",'references'),
    (r"^(acknowledg(e)?ments?|acknowledgment)$",'acknowledgments'),
    (r"^(supplementary|appendix|supporting information)$",'supplementary'),
]
RULES = [(re.compile(p, re.I), lab) for p, lab in RULES]

# =========================
# 3) Helpers
# =========================
def normalize_title(s: str) -> str:
    """Strip numbering/punctuation and lowercase the title."""
    s = s.strip()
    # Remove leading section numbering like "1.", "I.", "A)" etc.
    s = re.sub(r"^[\s\.\-\(\[]*([0-9]+|[IVXLC]+|[A-Z])([\.:\-\)\]\(])+\s*", "", s)
    # Trim trailing separators
    s = re.sub(r"[\s\:\-\–\—]+$", "", s)
    return s.lower()

def title_to_label(title_raw):
    """Map a section title to a canonical label (id, name) using regex rules."""
    t = normalize_title(title_raw)
    for rgx, lab in RULES:
        if rgx.search(t):
            return LABEL2ID[lab], lab
    return None

def split_lines(text: str):
    """Split into non-empty lines; handle both LF and CRLF."""
    return [p.strip() for p in re.split(r"\r?\n+", text) if p.strip()]

def fuzzy_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()

def find_heading_indices(pars, titles):
    """
    For each given title, find its index in the paragraph list.
    Try exact match first (from last position forward), then fuzzy match (>= 0.85).
    Returns a list of indices (or None) aligned to 'titles'.
    """
    norm_pars = [normalize_title(p) for p in pars]
    idxs, last = [], -1
    for raw_t in titles:
        t = normalize_title(raw_t)
        best_i, best_s = None, -1.0

        # Exact search from last+1 forward
        for i in range(last + 1, len(norm_pars)):
            if norm_pars[i] == t:
                best_i, best_s = i, 1.0
                break

        # Fallback: fuzzy search
        if best_i is None:
            for i in range(last + 1, len(norm_pars)):
                sc = fuzzy_ratio(norm_pars[i], t)
                if sc > best_s and sc >= 0.85:
                    best_i, best_s = i, sc

        idxs.append(best_i)
        if best_i is not None:
            last = best_i
    return idxs

def slice_sections(pars, heading_idxs):
    """
    Using heading indices, slice content spans for each detected title.
    Returns list of (k, start, end): k is title index, [start:end) are paragraph indices.
    """
    spans = []
    det = [(k, i) for k, i in enumerate(heading_idxs) if i is not None]
    for j, (k, idx) in enumerate(det):
        start = idx + 1
        end = det[j + 1][1] if j < len(det) - 1 else len(pars)
        if start < end:
            spans.append((k, start, end))
    return spans

# =========================
# 4) Load dataset (TFDS)
# =========================
print(f"Loading {DATASET}:{SPLIT} into {os.environ['TFDS_DATA_DIR']} ...")
ds = tfds.load(
    DATASET,
    split=SPLIT,                       # small subset first to validate
    data_dir=os.environ["TFDS_DATA_DIR"],
    with_info=False,
    try_gcs=False,                     # avoid GCS if not configured
)

# =========================
# 5) Process and export
# =========================
count_papers = count_sections = kept = 0
OUT.parent.mkdir(parents=True, exist_ok=True)

with gzip.open(OUT, "wt", encoding="utf-8") as fout:
    for i, ex in enumerate(tfds.as_numpy(ds)):
        if MAX_PAPERS is not None and count_papers >= MAX_PAPERS:
            break

        # Byte/string compatibility
        article = ex.get(b"article") if b"article" in ex else ex.get("article")
        section_names = ex.get(b"section_names") if b"section_names" in ex else ex.get("section_names")
        if article is None or section_names is None:
            continue

        if isinstance(article, (bytes, bytearray)):
            article = article.decode("utf-8", errors="ignore")
        else:
            article = str(article)

        if isinstance(section_names, (bytes, bytearray)):
            section_names = section_names.decode("utf-8", errors="ignore")
        else:
            section_names = str(section_names)

        pars, titles = split_lines(article), split_lines(section_names)
        if not pars or not titles:
            continue

        heading_idxs = find_heading_indices(pars, titles)
        spans = slice_sections(pars, heading_idxs)
        if not spans:
            continue

        count_papers += 1
        for k, s, e in spans:
            raw_title = titles[k]
            mapped = title_to_label(raw_title)
            count_sections += 1
            if mapped is None:
                continue

            label_id, label_name = mapped
            text = "\n".join(pars[s:e]).strip()
            if not text:
                continue

            obj = {
                "source": SOURCE_NAME,
                "paper_id": f"{SPLIT_BASE}_{i}",  # keep base split name (no [:2%])
                "section_title": raw_title.strip(),
                "label_id": label_id,
                "label_name": label_name,
                "section_text": text,
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

print(f"[DONE] papers={count_papers}, sections={count_sections}, kept={kept}")
print(f"Saved -> {OUT.resolve()}")
