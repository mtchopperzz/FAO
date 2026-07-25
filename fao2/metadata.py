# -*- coding: utf-8 -*-
"""
metadata.py
===========

Parse sample metadata from file or directory names.

Recommended FASTQ/sample naming style:

    ATP1B3__lib-scFv__round-R3__cond-pos.fastq.gz
    ATP1B3__lib-VHH__round-R5__cond-pos.fastq.gz
    ATP1B3__lib-scFv__round-R3__cond-neg__negtype-tag.fastq.gz

The parser is deliberately permissive.  It also recognizes VHH/scFv and R3/R4
patterns in older sample names such as ``VHH_2A_R5``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

FASTQ_SUFFIXES = [".fastq.gz", ".fq.gz", ".fastq", ".fq"]
CSV_SUFFIXES = [".csv"]


def strip_known_suffixes(name: str) -> str:
    base = name
    for suffix in FASTQ_SUFFIXES + CSV_SUFFIXES:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    for suffix in ["_annotated", "_pep_counts", "_counts"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


def parse_sample_name(name_or_path: str | Path) -> Dict[str, object]:
    """Parse metadata from a file path, folder name, or sample ID string."""
    path = Path(name_or_path)
    raw_name = path.name
    sample_id = strip_known_suffixes(raw_name)

    # If the file name itself is generic, the parent folder often carries the
    # original FASTQ sample name in FAO parser outputs.
    if sample_id in {"", "annotated"} or sample_id.endswith("_annotated"):
        sample_id = strip_known_suffixes(path.parent.name)

    info: Dict[str, object] = {
        "sample_id": sample_id,
        "library_type": "unknown",
        "round": pd.NA,
        "condition": "unknown",
        "negative_type": "",
        "target": "",
        "replicate": "",
        "source_name": raw_name,
    }

    tokens = re.split(r"__+", sample_id)
    for tok in tokens:
        low = tok.lower()
        if low.startswith("lib-"):
            lib = tok.split("-", 1)[1]
            info["library_type"] = normalize_library_type(lib)
        elif low.startswith("round-"):
            info["round"] = parse_round(tok.split("-", 1)[1])
        elif low.startswith("cond-"):
            info["condition"] = normalize_condition(tok.split("-", 1)[1])
        elif low.startswith("negtype-"):
            info["negative_type"] = tok.split("-", 1)[1]
        elif low.startswith("target-"):
            info["target"] = tok.split("-", 1)[1]
        elif low.startswith("rep-"):
            info["replicate"] = tok.split("-", 1)[1]

    # Fallbacks for older names.
    low_all = sample_id.lower()
    if info["library_type"] == "unknown":
        if "vhh" in low_all or "nanobody" in low_all:
            info["library_type"] = "VHH"
        elif "scfv" in low_all or "sc_fv" in low_all:
            info["library_type"] = "scFv"

    if pd.isna(info["round"]):
        m = re.search(r"(?:^|[_\-])R(\d+)(?:$|[_\-])", sample_id, flags=re.IGNORECASE)
        if m:
            info["round"] = int(m.group(1))

    if info["condition"] == "unknown":
        if re.search(r"(?:^|[_\-])neg(?:$|[_\-])", low_all) or "negative" in low_all:
            info["condition"] = "neg"
        elif re.search(r"(?:^|[_\-])input(?:$|[_\-])", low_all):
            info["condition"] = "input"
        elif re.search(r"(?:^|[_\-])pos(?:$|[_\-])", low_all) or "round" in low_all or re.search(r"(?:^|[_\-])R\d+", sample_id, flags=re.IGNORECASE):
            # Most round-numbered samples in screening are positive selection
            # outputs unless explicitly named neg/input.
            info["condition"] = "pos"

    return info


def normalize_library_type(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"vhh", "vh", "nanobody", "sdab"}:
        return "VHH"
    if text in {"scfv", "sc-fv", "sc_fv", "paired", "paired_hl", "fab"}:
        return "scFv"
    return str(value).strip() or "unknown"


def normalize_condition(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"pos", "positive", "target"}:
        return "pos"
    if text in {"neg", "negative", "control"}:
        return "neg"
    if text in {"input", "pre", "presort", "r0"}:
        return "input"
    return text or "unknown"


def parse_round(value: object) -> object:
    text = str(value).strip()
    m = re.search(r"(\d+)", text)
    if not m:
        return pd.NA
    return int(m.group(1))


def discover_annotated_files(parser_out: str | Path) -> List[Path]:
    root = Path(parser_out)
    return sorted(root.glob("**/*_annotated.csv"))


def build_sample_metadata(annotated_files: Iterable[str | Path]) -> pd.DataFrame:
    records = []
    for f in annotated_files:
        path = Path(f)
        # Prefer the parent folder, because FAO parser output is usually
        # parser_outputs/<sample>/<sample>_annotated.csv.
        parent_info = parse_sample_name(path.parent.name)
        file_info = parse_sample_name(path.name)
        info = parent_info
        # If parent is generic, fall back to file name.
        if info["sample_id"] in {"", "."} or info["library_type"] == "unknown" and file_info["library_type"] != "unknown":
            info = file_info
        info = dict(info)
        info["annotated_file"] = str(path)
        records.append(info)
    if not records:
        return pd.DataFrame(columns=["sample_id", "library_type", "round", "condition", "negative_type", "annotated_file"])
    df = pd.DataFrame(records)
    # Ensure sample IDs are unique.  If duplicates occur, append a suffix based
    # on row number; this prevents count-column collisions.
    seen = {}
    unique_ids = []
    for sid in df["sample_id"].astype(str):
        n = seen.get(sid, 0)
        seen[sid] = n + 1
        unique_ids.append(sid if n == 0 else f"{sid}__dup{n+1}")
    df["sample_id"] = unique_ids
    return df
