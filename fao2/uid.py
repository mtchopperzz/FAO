# -*- coding: utf-8 -*-
"""
uid.py
======

Small utilities for assigning stable full-length sequence IDs.

Design decision
---------------
seq_uid is assigned ONLY to a complete full-length candidate identity:

    H_FL:<H_FL_PEP>|L_FL:<L_FL_PEP>

For VHH libraries, L_FL_PEP is intentionally empty:

    H_FL:<H_FL_PEP>|L_FL:

No UID is assigned to H_CDR3 or CDR-combination layers.  Region-level tables use
representative_seq_uid to point back to a full-length representative sequence.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import pandas as pd

EMPTY_LIKE = {"", "nan", "none", "null", "na", "n/a", "<na>", "pd.na"}


def clean_str(value: Any) -> str:
    """Normalize missing-looking values to an empty string.

    Pandas may represent an empty light chain as NaN/None/<NA>.  UID generation
    must treat all of these identically, otherwise the same VHH sequence may
    receive multiple IDs.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in EMPTY_LIKE:
        return ""
    return text


def make_seq_uid(
    h_fl: Any,
    l_fl: Any = "",
    *,
    prefix: str = "SEQ",
    digest_len: int = 12,
) -> str:
    """Return a deterministic UID for a full-length H/L amino acid sequence.

    The function itself only requires H_FL to be non-empty.  Topology filtering
    for scFv/VHH is handled by :func:`make_seq_uid_from_row` because it requires
    library_type metadata.
    """
    h = clean_str(h_fl)
    l = clean_str(l_fl)
    if not h:
        return ""
    key = f"H_FL:{h}|L_FL:{l}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:digest_len].upper()
    return f"{prefix}_{digest}"


def is_topology_valid(h_fl: Any, l_fl: Any, library_type: str = "unknown") -> bool:
    """Check whether a row has a valid FL topology for UID assignment.

    Rules agreed for FAO2 demo:
      - scFv requires both H_FL and L_FL.
      - VHH requires H_FL; L_FL is allowed to be empty.
      - unknown library type falls back to H_FL-only validity.
    """
    h = clean_str(h_fl)
    l = clean_str(l_fl)
    lib = clean_str(library_type).lower()
    if not h:
        return False
    if lib in {"scfv", "scfv-like", "paired", "paired_hl", "fab"}:
        return bool(l)
    if lib in {"vhh", "vh", "vh_only", "nanobody", "sdab"}:
        return True
    # Conservative fallback: if library type cannot be parsed, still generate
    # a UID when H_FL exists; downstream code can filter if needed.
    return True


def make_seq_uid_from_row(
    row: Mapping[str, Any],
    *,
    h_col: str = "H_FL_PEP",
    l_col: str = "L_FL_PEP",
    library_type: str = "unknown",
    strict_topology: bool = True,
    prefix: str = "SEQ",
    digest_len: int = 12,
) -> str:
    """Generate seq_uid from a pandas row-like object.

    If strict_topology is True, invalid scFv H-only/L-only rows get an empty UID.
    """
    h = row.get(h_col, "")
    l = row.get(l_col, "")
    if strict_topology and not is_topology_valid(h, l, library_type):
        return ""
    return make_seq_uid(h, l, prefix=prefix, digest_len=digest_len)


def append_seq_uid_column(
    df: pd.DataFrame,
    *,
    library_type: str = "unknown",
    h_col: str = "H_FL_PEP",
    l_col: str = "L_FL_PEP",
    seq_uid_col: str = "seq_uid",
    strict_topology: bool = True,
) -> pd.DataFrame:
    """Return a copy of df with seq_uid appended as the final column.

    Existing columns are not altered.  If seq_uid already exists, it is replaced
    and moved to the end so the parser output remains visually stable.
    """
    out = df.copy()
    if h_col not in out.columns:
        out[seq_uid_col] = ""
    else:
        if l_col not in out.columns:
            out[l_col] = ""
        out[seq_uid_col] = out.apply(
            lambda row: make_seq_uid_from_row(
                row,
                h_col=h_col,
                l_col=l_col,
                library_type=library_type,
                strict_topology=strict_topology,
            ),
            axis=1,
        )
    cols = [c for c in out.columns if c != seq_uid_col] + [seq_uid_col]
    return out[cols]
