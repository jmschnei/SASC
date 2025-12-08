import os
import re
import ast
import json
from typing import List, Dict, Any, Optional

import pandas as pd
from datasets import load_dataset, get_dataset_split_names

from HSSC import (
    HierarchicalSectionClassifier,
    HierarchicalLabel,
    extract_features_for_classification,
)

# ---------------------------
# Global settings and paths
# ---------------------------

RAW_DIR = "./raw_csv"
PREPROC_DIR = "./preprocessed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PREPROC_DIR, exist_ok=True)

# Length constraints for section_content in the flattened dataset
MIN_LEN = 30
MAX_LEN = 8000

# Use the label hierarchy defined in HSSC
HSSC_HIERARCHY: Dict[str, List[str]] = HierarchicalSectionClassifier.SECTION_HIERARCHY
HSSC_LEVEL1_LABELS: List[str] = list(HSSC_HIERARCHY.keys())
HSSC_LEVEL2_LABELS: Dict[str, set] = {
    parent: set(children) for parent, children in HSSC_HIERARCHY.items()
}

# Optional priority order for level1 labels (still useful for feature-based mapping)
LEVEL1_PRIORITY = [
    "abstract",
    "introduction",
    "related_work",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "references",
    "appendix",
    "acknowledgments",
]


# ---------------------------
# 1. English detection
# ---------------------------
# ======================================================
# English Filter using langdetect (Option A)
# ======================================================

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # make langdetect deterministic


def is_english_text(text: str) -> bool:
    """
    Return True if text is detected as English using langdetect.
    - Very short text (<20 chars) automatically accepted.
    - Detect errors fallback to ASCII heuristic.
    """
    if not text:
        return False

    text = text.strip()
    if len(text) < 20:
        # Very short content is unreliable for langdetect
        return True

    try:
        lang = detect(text)
        return lang == "en"
    except Exception:
        # Fallback: simple ASCII proportion check (safe fallback)
        ascii_ratio = sum(ord(c) < 128 for c in text) / max(1, len(text))
        return ascii_ratio > 0.95


def row_is_english(row: pd.Series, threshold: float = 0.7) -> bool:
    """
    Decide if a document row is English using langdetect on:
      title + all section_content (fulltext_sections + fulltext_additional).

    Note:
      - 'threshold' is kept only for backward compatibility with previous code
        but it is not used in the decision.
    """
    title = str(row.get("title", "") or "")
    fulltext_sections_raw = row.get("fulltext_sections", "")
    fulltext_additional_raw = row.get("fulltext_additional", "")

    sections_main = parse_sections_field(fulltext_sections_raw)
    sections_add = parse_sections_field(fulltext_additional_raw)

    contents = [title]
    for s in sections_main:
        c = s.get("section_content", "")
        if isinstance(c, str):
            contents.append(c)
    for s in sections_add:
        c = s.get("section_content", "")
        if isinstance(c, str):
            contents.append(c)

    big_text = ("\n".join(contents))[:5000]
    return is_english_text(big_text)


# This helper is currently unused in the pipeline, kept for flexibility.
def load_csv_and_filter_english(csv_path: str, text_column="fulltext"):
    """
    Load SciLake CSV, remove non-English rows using langdetect.
    Return DataFrame with English rows only.

    This function is not used in the main preprocessing pipeline,
    but is kept as an alternative helper.
    """
    print(f"📂 Loading {csv_path}")

    df = pd.read_csv(csv_path)

    possible_fields = ["title", "abstract", "fulltext", "content"]

    def get_text(row):
        parts = []
        for c in possible_fields:
            if c in row and isinstance(row[c], str):
                parts.append(row[c])
        return " ".join(parts)

    english_mask = []
    print("🌐 Filtering non-English rows using langdetect...")

    for _, row in df.iterrows():
        text = get_text(row)
        english_mask.append(is_english_text(text))

    df["is_english"] = english_mask
    english_df = df[df["is_english"]].drop(columns=["is_english"])

    print(f"✅ English rows: {len(english_df)}/{len(df)}")

    return english_df


# ---------------------------
# 2. Robust parsing of sections fields
# ---------------------------

def parse_sections_field(raw_val) -> List[Dict[str, Any]]:
    """
    Robust parser for fulltext_sections / fulltext_additional.

    Handles cases like:
      - "[{'section_content': '...', ...}, {'section_content': '...', ...}]"
      - "{'section_content': '...', ...}" (single dict, no list)
      - Broken dumps where dicts appear back-to-back without commas:
        "[{'...'} {'...'} {'...'}]"

    Returns:
        list[dict], or [] if parsing fails.
    """
    if isinstance(raw_val, list):
        return raw_val

    if not isinstance(raw_val, str) or not raw_val.strip():
        return []

    s = raw_val.strip()

    # Single dict wrapped as list
    if s.startswith("{") and s.endswith("}"):
        s = "[" + s + "]"

    # Basic sanity: should look like a list
    if not (s.startswith("[") and s.endswith("]")):
        return []

    # Fix missing commas between dicts: "} {" or "}\n{"
    s_fixed = re.sub(r"}\s*{", "}, {", s)

    try:
        data = ast.literal_eval(s_fixed)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except Exception:
        return []

    return []


# ---------------------------
# 3. Cleaning of fulltext_additional
# ---------------------------

def clean_additional_sections(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean 'fulltext_additional' for ONE document.

    Rules:
      1) For each group of entries that share the same section_name:
         - If the group contains at least one non-empty section_content:
             keep all NON-empty entries in that group;
             drop entries whose content is empty (the whole dict is dropped).
         - If the group contains only empty section_content:
             drop all entries in that group.
      2) After grouping, remove exact duplicates by (section_name, section_content).

    IMPORTANT:
      - We do NOT modify 'section_num': if an entry is kept, its original
        section_num is preserved; if it is dropped, the whole dict
        (name + content + num) is removed.
    """
    norm_sections = []
    for s in sections:
        name = s.get("section_name", None)
        content = s.get("section_content", None)
        num = s.get("section_num", None)
        norm_sections.append({
            "section_name": name if isinstance(name, str) or name is None else str(name),
            "section_content": content if isinstance(content, str) or content is None else str(content),
            "section_num": num if isinstance(num, str) or num is None else str(num),
        })

    from collections import defaultdict
    by_name = defaultdict(list)
    for idx, s in enumerate(norm_sections):
        name = s["section_name"]
        key = name if name is not None else ""
        by_name[key].append((idx, s))

    to_keep_indices = set()

    for _, items in by_name.items():
        first_non_empty_idx = None

        for idx, s in items:
            content = s["section_content"] or ""
            if content.strip():
                first_non_empty_idx = idx
                break

        if first_non_empty_idx is not None:
            # Keep all non-empty entries in this group
            for idx, s in items:
                content = s["section_content"] or ""
                if content.strip():
                    to_keep_indices.add(idx)
        else:
            # All entries in this group are empty: drop them all
            continue

    cleaned = [norm_sections[i] for i in sorted(to_keep_indices)]

    # Deduplicate by (section_name, section_content)
    seen = set()
    unique_cleaned = []
    for s in cleaned:
        key = ((s["section_name"] or "").strip(), (s["section_content"] or "").strip())
        if key in seen:
            continue
        seen.add(key)
        unique_cleaned.append(s)

    return unique_cleaned


def clean_fulltext_additional_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply 'clean_additional_sections' to the 'fulltext_additional' column
    for all rows in a DataFrame, and print before/after stats.

    This function operates at corpus level (across all documents),
    while 'clean_additional_sections' operates at single-document level.
    """
    total_before = 0
    total_after = 0

    cleaned_col = []

    for _, row in df.iterrows():
        raw = row.get("fulltext_additional", "")
        sections = parse_sections_field(raw)
        total_before += len(sections)
        cleaned = clean_additional_sections(sections)
        total_after += len(cleaned)
        cleaned_col.append(json.dumps(cleaned, ensure_ascii=False))

    print("🧹 Cleaning fulltext_additional (drop empty / keep non-empty duplicates)...")
    print(f"   fulltext_additional sections: before={total_before}, after={total_after}")

    df = df.copy()
    df["fulltext_additional"] = cleaned_col
    return df


# ---------------------------
# 4. Mapping to HSSC labels (level1 / level2)
# ---------------------------

def normalize_text(t: Optional[str]) -> str:
    """
    Normalize whitespace and strip leading/trailing spaces.
    """
    if not isinstance(t, str):
        return ""
    return re.sub(r"\s+", " ", t).strip()


def find_level1_by_alias(aliases: List[str]) -> Optional[str]:
    """
    Given a list of alias substrings (e.g. ["methods", "materials and methods"]),
    try to find a level1 label in HSSC_LEVEL1_LABELS whose name contains
    any of these substrings (case-insensitive).
    """
    aliases_low = [a.lower() for a in aliases]
    for lab in HSSC_LEVEL1_LABELS:
        lab_low = lab.lower()
        for alias in aliases_low:
            if alias in lab_low:
                return lab
    return None


def map_level1_label(section_name: str, content: str) -> Optional[str]:
    """
    Map (section_name, content) to a level1 HSSC label.

    Strategy:
      1) Strong name-based rules for canonical headings.
      2) Strong name-based rules for typical methods subheadings.
      3) Use HSSC's 'extract_features_for_classification' and 'has_keywords'
         as secondary evidence.
      4) Light content-based fallback for generic patterns.
      5) Constrain predictions to labels defined in SECTION_HIERARCHY.
    """
    name = (section_name or "").strip()
    name_low = name.lower()
    text = content or ""
    if not text.strip():
        return None

    # Helper: find canonical labels for major rhetorical roles
    introduction_label = find_level1_by_alias(["introduction", "background", "overview"])
    related_work_label = find_level1_by_alias(["related work", "literature review", "state of the art"])
    methods_label      = find_level1_by_alias(["methods", "materials and methods", "patients and methods"])
    results_label      = find_level1_by_alias(["results", "findings", "experiments"])
    discussion_label   = find_level1_by_alias(["discussion"])
    conclusion_label   = find_level1_by_alias(["conclusion", "summary", "final remarks"])
    references_label   = find_level1_by_alias(["references", "bibliography"])
    appendix_label     = find_level1_by_alias(["appendix", "supplementary", "supporting information", "additional file"])
    ack_label          = find_level1_by_alias(["acknowledgments", "acknowledgements", "funding"])

    # 1) Strong canonical name-based rules

    # Abstract
    if "abstract" in name_low:
        abstract_label = find_level1_by_alias(["abstract"])
        if abstract_label is not None:
            return abstract_label

    # Introduction / Background / Overview
    if any(k in name_low for k in ["introduction", "background", "overview"]):
        if introduction_label is not None:
            return introduction_label

    # Related work / literature review
    if any(k in name_low for k in ["related work", "literature review", "state of the art"]):
        if related_work_label is not None:
            return related_work_label

    # Methods / Materials / Experimental
    if any(k in name_low for k in [
        "methods", "methodology", "materials",
        "experimental", "patients and methods",
        "material and methods", "materials and methods"
    ]):
        if methods_label is not None:
            return methods_label

    # Results / Findings / Experiments
    if any(k in name_low for k in [
        "results", "findings", "experiments",
        "numerical results", "simulation results"
    ]):
        if results_label is not None:
            return results_label

    # Discussion
    if "discussion" in name_low and discussion_label is not None:
        return discussion_label

    # Conclusion / Summary
    if any(k in name_low for k in [
        "conclusion", "conclusions", "summary",
        "concluding remarks", "final remarks", "closing remarks",
        "considerações finais"  # Portuguese: "final considerations"
    ]):
        if conclusion_label is not None:
            return conclusion_label

    # References / bibliography
    if any(k in name_low for k in ["references", "reference", "bibliography"]):
        if references_label is not None:
            return references_label

    # Appendix / supplementary
    if any(k in name_low for k in [
        "appendix", "supplementary", "supporting information", "additional file"
    ]):
        if appendix_label is not None:
            return appendix_label

    # Acknowledgments / funding / conflicts (name-based)
    if any(k in name_low for k in [
        "acknowledg", "funding", "conflict of interest", "competing interest",
        "author contributions", "data availability"
    ]):
        if ack_label is not None:
            return ack_label

    # 2) Strong name-based rules for METHODS-style subheadings

    methods_name_patterns = [
        # general methods / materials
        "materials and methods",
        "material and methods",
        "patients and methods",
        "methods and materials",
        "general methods",
        "experimental procedures",
        "experimental procedure",
        "experimental design",
        "study design",
        "study population",
        "study eligibility",
        "inclusion and exclusion criteria",
        "inclusion criteria",
        "exclusion criteria",
        "selection of studies",
        "research strategy",
        "data extraction",
        "data sources",
        "sample collection",
        "sampling",
        "participants",
        "patients",
        "subjects",

        # lab / molecular methods
        "dna extraction",
        "rna extraction",
        "genomic studies",
        "genotyping",
        "pcr",
        "polymerase chain reaction",
        "sequencing",
        "rna-seq",
        "rna-sequencing",
        "western blot",
        "flow cytometry",
        "immunostaining",
        "immunochemistry",
        "immunohistochemistry",
        "in situ hybridization",
        "histology",
        "histological",
        "microscopy",
        "electron microscopy",
        "antibodies",

        # physiology / recording / measurements
        "current measurements",
        "electrophysiology",
        "kcnt1 current",
        "inap current",
        "voltage clamp",
        "patch clamp",
        "recordings",

        # cell / animal preparation
        "primary cortical neuron culture",
        "neuron culture",
        "neuron modeling",
        "brain preparation",
        "ovarian histology",
        "hormonal stimulation",

        # imaging / quantification
        "photo-documentation",
        "image analysis",
        "quantitative analysis",
        "imaging and quantification",

        # field / sampling in energy/transport domains
        "onboard sampling",
        "adcp data analysis",
        "calculation of mesozooplankton",
        "calculation of poc content",
        "case ascertainment",  # epidemiological sampling definition

        # extra patterns from recent logs (cancer / neuro / energy)
        "quality of papers included in this metaanalysis",
        "proteomic data analysis",
        "sample preparation and 2-dimension electrophoresis",
        "sample preparation and 2-dimension electrophoresis (2-de)",
        "protein identification by ms",
        "primary cells and cell lines cultures",
        "cells and serum samples",
        "cell culture and viability assay",
        "generation of fluorescent ns/pcs",
        "chromosome analysis",
        "macroarray analysis",
        "mouse cortex slice preparation",
        "habituation to the orofacial stimulation test",
        "design of the cfos experiment",
        "quantification of p-erk-positive cells",
        "quantification of p-cofilin staining",
        "tms-induced eeg oscillations",
        "tms-evoked eeg potentials",
        "learning and verbal memory",
        "visual-motor coordination and executive function",
        "clinical examination of psychological state and dysfunction",
        "neuroticism",
        "anxiety and depression",
        "prr for other outcomes",
        "procedure",
    ]

    if methods_label is not None and any(pat in name_low for pat in methods_name_patterns):
        return methods_label

    # Common "Statistical analysis" subheading → usually grouped under methods
    if "statistical analysis" in name_low and methods_label is not None:
        return methods_label

    # Publication bias in meta-analyses is usually under results/discussion,
    # but we map it to results as a pragmatic choice.
    if "publication bias" in name_low and results_label is not None:
        return results_label

    # 3) HSSC feature-based rules (has_keywords)
    try:
        feats = extract_features_for_classification(
            text=text,
            position_in_doc=0.5  # approximate position
        )
        kw = feats.get("has_keywords", {}) or {}
    except Exception:
        kw = {}

    # Priority order uses canonical labels if they exist
    priority_order = [
        lab for lab in [
            introduction_label,
            methods_label,
            results_label,
            discussion_label,
            conclusion_label,
            references_label,
            appendix_label,
            ack_label,
        ] if lab is not None
    ]

    for label in priority_order:
        if kw.get(label):
            return label

    # Fallback: any HSSC level1 label with has_keywords=True
    for label in HSSC_LEVEL1_LABELS:
        if kw.get(label):
            return label

    # 4) Content-based fallback (only if nothing above fired)
    text_low = text.lower()

    # Introduction-like phrases
    if any(phrase in text_low for phrase in [
        "in this paper we", "in this article we", "in this study we",
        "this paper presents", "the aim of this study", "the objective of this study"
    ]):
        if introduction_label is not None:
            return introduction_label

    # Methods-like phrases
    if any(phrase in text_low for phrase in [
        "we collected data", "we conducted", "we performed",
        "we recruited", "we enrolled", "we used a questionnaire",
        "study design", "randomized", "double-blind",
        "cohort study", "case-control", "experiment was carried out"
    ]):
        if methods_label is not None:
            return methods_label

    # Results-like phrases
    if any(phrase in text_low for phrase in [
        "the results show", "our results show",
        "we found that", "we find that", "the findings indicate"
    ]):
        if results_label is not None:
            return results_label

    # Discussion-like phrases
    if any(phrase in text_low for phrase in [
        "these results suggest", "our findings suggest",
        "this suggests that", "this indicates that", "we speculate"
    ]):
        if discussion_label is not None:
            return discussion_label

    # Conclusion-like phrases
    if text_low.startswith("in conclusion") or text_low.startswith("to conclude") or text_low.startswith("in summary"):
        if conclusion_label is not None:
            return conclusion_label

    # Content-based fallback for acknowledgments when section_name is empty
    if ack_label is not None and any(
        phrase in text_low for phrase in [
            "acknowledgments", "acknowledgements",
            "this research has been made possible thanks to",
            "we thank", "we are grateful to"
        ]
    ):
        return ack_label

    # If nothing matched, return None (conservative)
    return None


def map_level2_label(section_name: str, content: str, level1: Optional[str]) -> Optional[str]:
    """
    Map (section_name, content, level1) to a level2 HSSC label.

    Strategy:
      1) Use HSSC's 'extract_features_for_classification' and read 'has_keywords'.
      2) Restrict candidate level2 labels to the children under 'level1'
         in SECTION_HIERARCHY.
      3) If multiple candidate children are True, pick the first in
         SECTION_HIERARCHY's order.
      4) Heuristic fallback:
         - If there is exactly ONE child: use that child as default level2.
         - If there are multiple children:
             * first try has_keywords
             * then try "tables_figures" pattern
             * finally fallback to the first child (e.g. "results_main").
    """
    if level1 is None:
        return None

    text = content or ""
    if not text.strip():
        return None

    children = list(HSSC_HIERARCHY.get(level1, []))
    if not children:
        return None

    # 0) Simple case: only one child → use it as default level2.
    if len(children) == 1:
        return children[0]

    # 1) Use HSSC features / has_keywords for multi-child cases
    try:
        feats = extract_features_for_classification(
            text=text,
            position_in_doc=0.5
        )
        kw = feats.get("has_keywords", {}) or {}
    except Exception:
        kw = {}

    # Prefer any child with has_keywords[child] == True
    for child in children:
        if kw.get(child):
            return child

    # 2) Optional fallback for "tables_figures"
    lowered = text.lower()
    if "tables_figures" in children:
        if any(tok in lowered for tok in ["table ", "table:", "fig.", "figure ", "see table", "see figure"]):
            return "tables_figures"

    # 3) Final fallback for multi-child cases:
    #    if nothing matched, fallback to the first child
    #    (e.g. "results_main" for the "results" branch).
    return children[0]


# ---------------------------
# 5. HuggingFace: download & save as CSV (if needed)
# ---------------------------

def get_save_data_from_huggingface():
    """
    Download the SciLake fulltext corpus from HuggingFace (if needed)
    and save each split as a CSV in ./raw_csv.
    """
    dataset_name = "SIRIS-Lab/scilake-fulltext-corpus"
    config_name = "default"

    print("Checking existing CSV files in ./raw_csv ...")
    splits = get_dataset_split_names(dataset_name, config_name)
    for split in splits:
        out_file = os.path.join(RAW_DIR, f"scilake_{split}.csv")
        if os.path.exists(out_file):
            print(f"⚠️  {out_file} already exists, skip downloading split '{split}'.")
            continue

        print(f"🔄 Loading split: {split}")
        ds = load_dataset(dataset_name, config_name, split=split)
        df = ds.to_pandas()

        keep_cols = ["doi", "title", "fulltext_sections", "fulltext_additional"]
        for c in keep_cols:
            if c not in df.columns:
                df[c] = None
        df = df[keep_cols]

        df.to_csv(out_file, index=False)
        print(f"✅ Saved {out_file} ({len(df)} rows)")


# ---------------------------
# 6. Flatten & preprocess one split
# ---------------------------

def inspect_sections_distribution(df: pd.DataFrame):
    """
    Print basic distribution info of fulltext_sections / fulltext_additional
    for the current DataFrame (after English filtering).
    """
    n_rows = len(df)
    print(f"   Docs after English filter          : {n_rows}")

    docs_with_main = 0
    docs_with_add = 0
    total_main_sections = 0
    total_add_sections = 0

    for _, row in df.iterrows():
        sections_main = parse_sections_field(row.get("fulltext_sections", ""))
        sections_add = parse_sections_field(row.get("fulltext_additional", ""))

        if len(sections_main) > 0:
            docs_with_main += 1
            total_main_sections += len(sections_main)
        if len(sections_add) > 0:
            docs_with_add += 1
            total_add_sections += len(sections_add)

    print(f"   Docs with fulltext_sections    : {docs_with_main}")
    print(f"   Total fulltext_sections        : {total_main_sections}")
    print(f"   Docs with fulltext_additional  : {docs_with_add}")
    print(f"   Total fulltext_additional      : {total_add_sections}")


def flatten_sections(df: pd.DataFrame, split_name: str,
                     min_len: int = MIN_LEN, max_len: int = MAX_LEN) -> pd.DataFrame:
    """
    Flatten fulltext_sections and fulltext_additional into a row-per-section DataFrame,
    with heuristic HSSC level1/level2 labels (based on HSSC features and hierarchy).

    Final output columns:
      - doi
      - title
      - section_name
      - section_content
      - hssc_level1
      - hssc_level2
    """
    records = []

    for _, row in df.iterrows():
        doi = row.get("doi", None)
        title = row.get("title", None)

        sections_main = parse_sections_field(row.get("fulltext_sections", ""))
        sections_add = parse_sections_field(row.get("fulltext_additional", ""))

        def add_sections(sections: List[Dict[str, Any]], is_additional_flag: int):
            for s in sections:
                name = normalize_text(s.get("section_name"))
                content = normalize_text(s.get("section_content"))

                if not content:
                    continue

                if len(content) < min_len or len(content) > max_len:
                    continue

                lvl1 = map_level1_label(name, content)
                lvl2 = map_level2_label(name, content, lvl1)

                # NOTE: We no longer store hssc_label_path, is_additional, or source_split
                rec = {
                    "doi": doi,
                    "title": title,
                    "section_name": name if name else None,
                    "section_content": content,
                    "hssc_level1": lvl1,
                    "hssc_level2": lvl2,
                }
                records.append(rec)

        add_sections(sections_main, is_additional_flag=0)
        add_sections(sections_add, is_additional_flag=1)

    flat_df = pd.DataFrame(records)
    return flat_df


def preprocess_one_split(split_name: str,
                         english_threshold: float = 0.7):
    """
    Full pipeline for one split:

      1. Load raw CSV from ./raw_csv/scilake_{split_name}.csv
      2. Apply English filtering at document level (title + sections).
      3. Inspect distribution of fulltext_sections / fulltext_additional.
      4. Clean 'fulltext_additional' on the English subset.
      5. Flatten to section-level dataset, map to HSSC level1/level2.
      6. Apply paper-level filter for level2:
           - Drop papers (dois) that have no level2 anywhere.
           - Keep all rows for papers that have at least one level2.
      7. Print statistics about unique level names for hssc_level1 / hssc_level2.
      8. Save to:
           ./preprocessed/scilake_{split_name}_level2_hssc.csv
    """
    in_path = os.path.join(RAW_DIR, f"scilake_{split_name}.csv")
    out_path = os.path.join(PREPROC_DIR, f"scilake_{split_name}_level2_hssc.csv")

    print(f"\n📂 Loading {in_path}")
    if not os.path.exists(in_path):
        print(f"❌ File not found: {in_path}. Skipping this split.")
        return

    df = pd.read_csv(in_path)

    # 1) English filtering BEFORE any cleaning of additional sections
    print("🌐 Filtering non-English rows using langdetect (document-level)...")
    english_mask = df.apply(lambda r: row_is_english(r, threshold=english_threshold), axis=1)
    n_total = len(df)
    n_english = int(english_mask.sum())
    print(f"✅ English rows: {n_english}/{n_total}")

    df_eng = df[english_mask].reset_index(drop=True)

    # 2) Inspect distribution for English-only docs (before cleaning additional)
    inspect_sections_distribution(df_eng)

    # 3) Clean fulltext_additional only on English docs
    df_eng = clean_fulltext_additional_for_df(df_eng)

    # 4) Flatten to section-level dataset
    print("🧱 Extracting sections with HSSC feature-based mapping...")
    flat_df = flatten_sections(df_eng, split_name=split_name,
                               min_len=MIN_LEN, max_len=MAX_LEN)

    total_sections = len(flat_df)
    kept_level1 = flat_df["hssc_level1"].notna().sum()
    kept_level2 = flat_df["hssc_level2"].notna().sum()

    print(f"   Total sections (fulltext_sections + additional): {total_sections}")
    print(
        "   NOTE: Total sections is the number of rows AFTER applying "
        "flatten_sections() filters:\n"
        f"         - section_content is non-empty\n"
        f"         - {MIN_LEN} <= len(section_content) <= {MAX_LEN}\n"
        "         - content is normalized (whitespace) and must remain non-empty\n"
        "         Therefore Total sections is generally smaller than:\n"
        "           Total fulltext_sections + cleaned fulltext_additional."
    )
    print(f"   With hssc_level1 (non-empty level1 label) : {kept_level1}")
    print(f"   With hssc_level2 (non-empty level2 label) : {kept_level2}")

    # ---------- Paper-level filter based on level2 ----------
    # If a paper (same doi) has no non-empty hssc_level2 in any section,
    # we drop ALL rows for that doi.
    # If a paper has at least one section with non-empty hssc_level2,
    # we keep ALL rows for that doi (even if some rows have hssc_level2 = NaN).
    if not flat_df.empty:
        has_l2 = flat_df.groupby("doi")["hssc_level2"].transform(
            lambda s: s.notna().any()
        )

        n_docs_before = flat_df["doi"].nunique()
        flat_df = flat_df[has_l2].copy()
        n_docs_after = flat_df["doi"].nunique()

        print(
            f"   Docs with at least one non-empty level2: "
            f"{n_docs_after}/{n_docs_before} "
            f"(dropped {n_docs_before - n_docs_after} docs with no level2 at all)"
        )
    else:
        print("   WARNING: flat_df is empty after flatten_sections().")
    # --------------------------------------------------------

    # ---------- Statistics about unique level names ----------
    if not flat_df.empty:
        l1_nonnull = flat_df["hssc_level1"].dropna()
        l2_nonnull = flat_df["hssc_level2"].dropna()

        num_l1 = l1_nonnull.nunique()
        num_l2 = l2_nonnull.nunique()

        print(f"   Unique hssc_level1 labels (non-empty): {num_l1}")
        print(f"   Unique hssc_level2 labels (non-empty): {num_l2}")

        print("   hssc_level1 label names:")
        print("     " + ", ".join(sorted(map(str, l1_nonnull.unique()))))

        print("   hssc_level2 label names (first 50, if many):")
        unique_l2 = sorted(map(str, l2_nonnull.unique()))
        if len(unique_l2) > 50:
            print("     " + ", ".join(unique_l2[:50]) + ", ...")
        else:
            print("     " + ", ".join(unique_l2))

        print("   hssc_level1 distribution (top 20):")
        print(l1_nonnull.value_counts().head(20))

        print("   hssc_level2 distribution (top 20):")
        print(l2_nonnull.value_counts().head(20))
    else:
        print("   No rows left after paper-level filtering; skipping label stats.")
    # --------------------------------------------------------

    # Show some examples without level1 mapping (for debugging coverage)
    no_level1 = flat_df[flat_df["hssc_level1"].isna()]
    if not no_level1.empty:
        print("   Examples of sections WITHOUT level1 mapping:")
        for _, r in no_level1.head(20).iterrows():
            name = (r["section_name"] or "")[:80]
            preview = (r["section_content"] or "")[:120].replace("\n", " ")
            print(f"     [name] {repr(name)} | [preview] {repr(preview)}")
    else:
        print("   All sections received a level1 label (no NaNs).")

    flat_df.to_csv(out_path, index=False)
    print(f"💾 Saved level2 dataset: {out_path} ({len(flat_df)} rows)")


# ---------------------------
# 7. Main entry point
# ---------------------------

def main():
    # Step 1: download from HuggingFace if needed
    get_save_data_from_huggingface()

    # Step 2: preprocess each split
    splits = ["cancer", "neuroscience", "energy", "transport", "general"]
    for split in splits:
        preprocess_one_split(split_name=split, english_threshold=0.7)


if __name__ == "__main__":
    main()
