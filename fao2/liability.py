# -*- coding: utf-8 -*-
"""Batch liability screening for FAO2 candidate tables.

The module adapts the standalone liability scanner to the one-row-per-seq_uid
candidate table. H_FL_PEP and L_FL_PEP are numbered separately with ANARCI,
then the configured regex rules are applied to IMGT CDR1, CDR2 and CDR3.

Four columns are appended, matching the standalone report after its ID column:

    Status, Chain, Liabilities, Details

Because a candidate row may contain both H and L, ``Chain`` contains the
recognized chain types joined with ``;`` and ``Details`` uses chain-qualified
region labels such as HCDR1, KCDR2 and LCDR3.
"""

from __future__ import annotations

import json
import multiprocessing
import re
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

try:
    from anarci import anarci
    try:
        from anarci import run_anarci
    except ImportError:
        run_anarci = None
    ANARCI_AVAILABLE = True
except ImportError:
    anarci = None
    run_anarci = None
    ANARCI_AVAILABLE = False


IMGT_REGIONS: Dict[str, Tuple[int, int]] = {
    "CDR1": (27, 38),
    "CDR2": (56, 65),
    "CDR3": (105, 117),
}

VALID_CHAINS = {"H", "K", "L"}
LIABILITY_OUTPUT_COLUMNS = ["Status", "Chain", "Liabilities", "Details"]

# Rules are intentionally preserved from the supplied standalone scanner.
# The hydropathy/charge rule in that script is commented out, so this module
# currently applies only the active regex rules.
LIABILITY_RULES: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    (
        "N-linked_glycosylation",
        r"N[^P][ST]",
        ("CDR1", "CDR2", "CDR3"),
    ),
    (
        "Asn_Deamidation",
        r"N[GSTN]|GN[FGTY]",
        ("CDR1", "CDR2", "CDR3"),
    ),
    (
        "Asp_Isomerization",
        r"D[GSN]",
        ("CDR1", "CDR2", "CDR3"),
    ),
    (
        "Cysteine",
        r"C",
        ("CDR1", "CDR2", "CDR3"),
    ),
    (
        "Hydrolysis",
        r"DP",
        ("CDR1", "CDR2", "CDR3"),
    ),
    (
        "Gln_Deamidation",
        r"QG",
        ("CDR1", "CDR2", "CDR3"),
    ),
    (
        "Methionine",
        r"M",
        ("CDR1", "CDR2", "CDR3"),
    ),
    (
        "Lysine",
        r"K",
        ("CDR1", "CDR2", "CDR3"),
    ),
)

COMPILED_LIABILITY_RULES = tuple(
    (name, re.compile(pattern), regions)
    for name, pattern, regions in LIABILITY_RULES
)


def _clean_sequence(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    sequence = re.sub(r"\s+", "", str(value)).upper()
    if sequence.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""
    return sequence


def _domain_parts(domain_obj: Any) -> Tuple[Optional[List[Any]], Optional[int], Optional[int]]:
    if isinstance(domain_obj, tuple):
        residues = (
            domain_obj[0]
            if len(domain_obj) > 0 and isinstance(domain_obj[0], list)
            else None
        )
        start = (
            int(domain_obj[1])
            if len(domain_obj) > 1 and domain_obj[1] is not None
            else None
        )
        end = (
            int(domain_obj[2])
            if len(domain_obj) > 2 and domain_obj[2] is not None
            else None
        )
        return residues, start, end
    if isinstance(domain_obj, list):
        # Some ANARCI releases wrap the actual numbering list once more.
        if domain_obj and isinstance(domain_obj[0], list):
            return domain_obj[0], None, None
        return domain_obj, None, None
    return None, None, None


def _extract_region(numbering: Sequence[Any], span: Tuple[int, int]) -> str:
    start, end = span
    residues: List[str] = []
    for item in numbering:
        try:
            position = int(item[0][0])
            aa = str(item[1])
        except (IndexError, TypeError, ValueError):
            continue
        if start <= position <= end and aa != "-":
            residues.append(aa)
    return "".join(residues)


def _scan_numbering(
    chain_type: str,
    numbering: Sequence[Any],
) -> Dict[str, List[str]]:
    motifs_found: Dict[str, List[str]] = defaultdict(list)
    for liability_name, regex, regions in COMPILED_LIABILITY_RULES:
        for region_name in regions:
            sequence = _extract_region(numbering, IMGT_REGIONS[region_name])
            if sequence and regex.search(sequence):
                label = f"{chain_type}{region_name}"
                if label not in motifs_found[liability_name]:
                    motifs_found[liability_name].append(label)
    return dict(motifs_found)


def _expected_chain_matches(expected_role: str, anarci_chain: str) -> bool:
    if expected_role == "H":
        return anarci_chain == "H"
    return anarci_chain in {"K", "L"}


def _parse_numbering_result(
    expected_role: str,
    numbering_entry: Any,
    alignment_entry: Any,
) -> Dict[str, Any]:
    if not numbering_entry or not alignment_entry:
        return {
            "status": "No antibody recognized",
            "chain": "-",
            "motifs": {},
        }

    best: Optional[Dict[str, Any]] = None
    for domain_index, domain_obj in enumerate(numbering_entry):
        if domain_index >= len(alignment_entry):
            continue
        meta = alignment_entry[domain_index]
        chain_type = str(meta.get("chain_type", ""))
        if chain_type not in VALID_CHAINS:
            continue
        if not _expected_chain_matches(expected_role, chain_type):
            continue

        numbering, _, _ = _domain_parts(domain_obj)
        if not numbering:
            continue

        bitscore = float(meta.get("bitscore", 0.0) or 0.0)
        parsed = {
            "status": "Pass",
            "chain": chain_type,
            "motifs": _scan_numbering(chain_type, numbering),
            "bitscore": bitscore,
        }
        if best is None or parsed["bitscore"] > best["bitscore"]:
            best = parsed

    if best is None:
        return {
            "status": "No antibody recognized",
            "chain": "-",
            "motifs": {},
        }
    return best


def _run_unique_anarci(
    requests: Mapping[Tuple[str, str], List[Tuple[int, str]]],
    *,
    ncpu: int,
    batch_size: int,
    bit_score_threshold: float,
    allowed_species: Optional[Sequence[str]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not ANARCI_AVAILABLE:
        raise ImportError(
            "ANARCI is required for liability screening but could not be imported."
        )

    unique_keys = list(requests.keys())
    parsed_results: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for batch_start in range(0, len(unique_keys), batch_size):
        batch_keys = unique_keys[batch_start : batch_start + batch_size]
        sequences = [
            (f"liability_{batch_start + index}", sequence)
            for index, (_, sequence) in enumerate(batch_keys)
        ]

        try:
            if run_anarci is not None:
                _, numbering, alignment_details, _ = run_anarci(
                    sequences,
                    ncpu=min(ncpu, max(1, len(sequences))),
                    scheme="imgt",
                    output=False,
                    assign_germline=False,
                    allowed_species=allowed_species,
                    allow={"H", "K", "L"},
                    bit_score_threshold=bit_score_threshold,
                )
            else:
                numbering, alignment_details, _ = anarci(
                    sequences,
                    scheme="imgt",
                    output=False,
                    ncpu=min(ncpu, max(1, len(sequences))),
                    assign_germline=False,
                    allowed_species=allowed_species,
                    allow={"H", "K", "L"},
                    bit_score_threshold=bit_score_threshold,
                )
        except Exception as exc:
            message = f"ANARCI Error: {exc}"
            for key in batch_keys:
                parsed_results[key] = {
                    "status": message,
                    "chain": "-",
                    "motifs": {},
                }
            continue

        for index, key in enumerate(batch_keys):
            numbering_entry = (
                numbering[index] if numbering is not None else None
            )
            alignment_entry = (
                alignment_details[index]
                if alignment_details is not None
                else None
            )
            parsed_results[key] = _parse_numbering_result(
                key[0],
                numbering_entry,
                alignment_entry,
            )

    return parsed_results


def _aggregate_candidate_result(
    requested_roles: Sequence[str],
    role_results: Mapping[str, Dict[str, Any]],
) -> Dict[str, str]:
    if not requested_roles:
        return {
            "Status": "Empty Sequence",
            "Chain": "-",
            "Liabilities": "",
            "Details": "",
        }

    successful_roles = [
        role
        for role in requested_roles
        if role_results.get(role, {}).get("status") == "Pass"
    ]

    if len(successful_roles) == len(requested_roles):
        status = "Pass"
    elif successful_roles:
        status = "Partial"
    else:
        statuses = [
            str(role_results.get(role, {}).get("status", "No antibody recognized"))
            for role in requested_roles
        ]
        error_statuses = [value for value in statuses if value.startswith("ANARCI Error:")]
        status = error_statuses[0] if error_statuses else "No antibody recognized"

    chains: List[str] = []
    combined_details: Dict[str, List[str]] = defaultdict(list)
    errors: List[str] = []

    for role in requested_roles:
        result = role_results.get(
            role,
            {"status": "No antibody recognized", "chain": "-", "motifs": {}},
        )
        if result.get("status") == "Pass":
            chain = str(result.get("chain", "-"))
            if chain != "-" and chain not in chains:
                chains.append(chain)
            for liability_name, labels in result.get("motifs", {}).items():
                for label in labels:
                    if label not in combined_details[liability_name]:
                        combined_details[liability_name].append(label)
        else:
            errors.append(f"{role}:{result.get('status', 'No antibody recognized')}")

    details_payload: Dict[str, Any] = {
        name: labels
        for name, labels in sorted(combined_details.items())
    }
    if errors:
        details_payload["_Errors"] = errors

    liability_names = sorted(combined_details.keys())
    liabilities = ";".join(liability_names) if liability_names else "None"

    return {
        "Status": status,
        "Chain": ";".join(chains) if chains else "-",
        "Liabilities": liabilities,
        "Details": json.dumps(
            details_payload,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
    }


def add_liability_columns(
    candidates: pd.DataFrame,
    *,
    h_col: str = "H_FL_PEP",
    l_col: str = "L_FL_PEP",
    ncpu: Optional[int] = None,
    batch_size: int = 5000,
    bit_score_threshold: float = 80.0,
    allowed_species: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Append Status/Chain/Liabilities/Details to a candidate table.

    Unique ``(expected role, full-length peptide)`` inputs are numbered once,
    then mapped back to all candidate rows. VHH rows naturally contain only the
    H request because L_FL_PEP is empty.
    """
    output = candidates.copy()
    for column in LIABILITY_OUTPUT_COLUMNS:
        output[column] = ""

    if output.empty:
        return output

    available_cpus = max(1, multiprocessing.cpu_count())
    requested_cpus = available_cpus if ncpu is None else max(1, int(ncpu))
    worker_count = min(requested_cpus, available_cpus)
    batch_size = max(1, int(batch_size))

    requests: Dict[Tuple[str, str], List[Tuple[int, str]]] = defaultdict(list)
    row_roles: Dict[int, List[str]] = {}

    for row_index, row in output.iterrows():
        roles: List[str] = []
        h_sequence = _clean_sequence(row.get(h_col, ""))
        l_sequence = _clean_sequence(row.get(l_col, ""))

        if h_sequence:
            roles.append("H")
            requests[("H", h_sequence)].append((row_index, "H"))
        if l_sequence:
            roles.append("L")
            requests[("L", l_sequence)].append((row_index, "L"))
        row_roles[row_index] = roles

    parsed_unique = _run_unique_anarci(
        requests,
        ncpu=worker_count,
        batch_size=batch_size,
        bit_score_threshold=float(bit_score_threshold),
        allowed_species=allowed_species,
    )

    row_results: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for key, consumers in requests.items():
        result = parsed_unique.get(
            key,
            {"status": "No antibody recognized", "chain": "-", "motifs": {}},
        )
        for row_index, role in consumers:
            row_results[row_index][role] = result

    for row_index in output.index:
        aggregated = _aggregate_candidate_result(
            row_roles.get(row_index, []),
            row_results.get(row_index, {}),
        )
        for column, value in aggregated.items():
            output.at[row_index, column] = value

    return output
