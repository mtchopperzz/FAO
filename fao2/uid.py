# -*- coding: utf-8 -*-
"""
Stable full-length antibody sequence UID utilities.

The UID is determined only by the normalized VH and VL amino-acid sequences.
Library, target, round, condition, count, germline and annotation metadata are
not included.

Canonical order:
    VH <unit-separator> VL

For VHH, VL is an empty string.
"""

from __future__ import annotations

import base64
import hashlib
import re
from typing import Any, Mapping

import pandas as pd


EMPTY_LIKE = {"", "nan", "none", "null", "na", "n/a", "<na>", "pd.na"}
STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
_SEQUENCE_SEPARATOR = "\x1f"


def clean_str(value: Any) -> str:
    """Normalize missing-looking values to an empty string."""
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


def normalize_variable_peptide(value: Any) -> str:
    """
    Normalize a VH/VL amino-acid sequence for UID generation.

    Only whitespace and letter case are normalized. A sequence containing a
    non-standard residue is rejected by returning an empty string.
    """
    sequence = re.sub(r"\s+", "", clean_str(value)).upper()
    if not sequence:
        return ""
    if any(residue not in STANDARD_AA for residue in sequence):
        return ""
    return sequence


def make_seq_uid(
    vh: Any,
    vl: Any = "",
    *,
    prefix: str = "SEQ",
    digest_bytes: int = 16,
) -> str:
    """
    Generate a deterministic UID from VH and VL amino-acid sequences only.

    SHA-256 is calculated over the canonical VH/VL payload. The first 16 bytes
    provide a 128-bit identifier, represented by 22 Base64url characters.

    VHH:
        VH=<sequence>, VL=""
    """
    vh_sequence = normalize_variable_peptide(vh)
    vl_raw = clean_str(vl)
    vl_sequence = normalize_variable_peptide(vl_raw) if vl_raw else ""

    if not vh_sequence:
        return ""
    if vl_raw and not vl_sequence:
        return ""

    payload = f"{vh_sequence}{_SEQUENCE_SEPARATOR}{vl_sequence}".encode("ascii")
    digest = hashlib.sha256(payload).digest()[:digest_bytes]
    token = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"{prefix}_{token}"


def is_topology_valid(vh: Any, vl: Any, library_type: str = "unknown") -> bool:
    """Validate scFv/VHH topology before UID assignment."""
    vh_sequence = normalize_variable_peptide(vh)
    vl_raw = clean_str(vl)
    vl_sequence = normalize_variable_peptide(vl_raw) if vl_raw else ""
    library = clean_str(library_type).lower()

    if not vh_sequence:
        return False
    if vl_raw and not vl_sequence:
        return False

    if library in {"scfv", "scfv-like", "paired", "paired_hl", "fab"}:
        return bool(vl_sequence)
    if library in {"vhh", "vh", "vh_only", "nanobody", "sdab"}:
        return True

    return True


def make_seq_uid_from_row(
    row: Mapping[str, Any],
    *,
    h_col: str = "H_FL_PEP",
    l_col: str = "L_FL_PEP",
    library_type: str = "unknown",
    strict_topology: bool = True,
    prefix: str = "SEQ",
    digest_bytes: int = 16,
) -> str:
    """Generate seq_uid from a pandas row-like object."""
    vh = row.get(h_col, "")
    vl = row.get(l_col, "")

    if strict_topology and not is_topology_valid(vh, vl, library_type):
        return ""

    return make_seq_uid(
        vh,
        vl,
        prefix=prefix,
        digest_bytes=digest_bytes,
    )


def append_seq_uid_column(
    df: pd.DataFrame,
    *,
    library_type: str = "unknown",
    h_col: str = "H_FL_PEP",
    l_col: str = "L_FL_PEP",
    seq_uid_col: str = "seq_uid",
    strict_topology: bool = True,
) -> pd.DataFrame:
    """
    Return a copy with seq_uid as the final column.

    Existing parser columns are not modified.
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

    columns = [column for column in out.columns if column != seq_uid_col]
    return out[columns + [seq_uid_col]]
