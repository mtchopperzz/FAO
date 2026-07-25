# -*- coding: utf-8 -*-
"""
prioritizer.py
==============

FAO2 post-processing module.

This file is designed to replace ``fao2/prioritizer.py``.
Compared with the previous demo version, this version does NOT load all
annotated.csv files into one global dataframe before splitting by library.
It first builds sample metadata, groups samples by library_key, and then reads
only one library's annotated.csv files at a time.

Inputs
------
- Existing FAO parser outputs: parser_outputs/**/**_annotated.csv
- Optional LLM clustering CSV/XLSX

Outputs
-------
- prioritization_outputs/sample_metadata.csv
- prioritization_outputs/library_manifest.csv
- prioritization_outputs/by_library/<library_key>/candidate_prioritization_table.csv
- prioritization_outputs/by_library/<library_key>/recommended_candidates.csv
- prioritization_outputs/by_library/<library_key>/region_support/*.csv

Core rules
----------
1. seq_uid only refers to a full-length sequence identity:
       H_FL:<H_FL_PEP>|L_FL:<L_FL_PEP>
2. VHH is treated as an antibody with an empty light chain.
3. scFv candidates must have both H_FL_PEP and L_FL_PEP when strict_topology=True.
4. Region counts are summed across all topology-valid full-length sequences
   owning the region.
5. Region representatives are chosen from positive samples only: latest positive
   round first, highest read count second.
6. Negative samples never choose representatives. They only mark positive
   candidates/representatives as negative-associated when the same seq_uid is
   observed in a negative sample.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .metadata import build_sample_metadata, discover_annotated_files
from .uid import append_seq_uid_column, clean_str, make_seq_uid


DEFAULT_REGION_SPECS = [
    "H_CDR3_PEP",
    "H_CDRs_PEP-L_CDRs_PEP",
    "H_FL_PEP-L_FL_PEP",
]

CORE_SEQUENCE_COLUMNS = [
    "Seq",
    "Count",
    "DNA",
    "H_CDR3_PEP",
    "L_CDR3_PEP",
    "H_CDRs_PEP",
    "L_CDRs_PEP",
    "H_FL_PEP",
    "L_FL_PEP",
    "H_V_Gene",
    "H_J_Gene",
    "L_V_Gene",
    "L_J_Gene",
    "H_DNA",
    "L_DNA",
]


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def find_count_col(df: pd.DataFrame) -> str:
    for col in ["Count", "count", "Read_Count", "read_count", "Reads", "reads"]:
        if col in df.columns:
            return col
    raise ValueError("Could not find a read-count column. Expected Count/count/Read_Count/reads.")


def make_display_id(row: pd.Series, fallback: str) -> str:
    for col in ["display_id", "ID", "Seq_ID", "seq_id", "sequence_id", "Name", "name"]:
        if col in row.index and clean_str(row[col]):
            return clean_str(row[col])
    return fallback


def region_columns(region_spec: str) -> List[str]:
    return [x.strip() for x in str(region_spec).split("-") if x.strip()]


def build_region_key(row: pd.Series, region_spec: str) -> str:
    parts = [clean_str(row.get(col, "")) for col in region_columns(region_spec)]
    if not any(parts):
        return ""
    return "-".join(parts)


def build_region_key_series(df: pd.DataFrame, region_spec: str) -> pd.Series:
    cols = region_columns(region_spec)
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    values = []
    for col in cols:
        if col in df.columns:
            s = df[col].map(clean_str)
        else:
            s = pd.Series([""] * len(df), index=df.index)
        values.append(s)
    if len(values) == 1:
        key = values[0]
    else:
        key = values[0].astype(str)
        for s in values[1:]:
            key = key + "-" + s.astype(str)
    all_empty = pd.concat(values, axis=1).apply(lambda r: not any(r.astype(str)), axis=1)
    key = key.mask(all_empty, "")
    return key


def safe_filename(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("|", "_")
        .replace(":", "_")
    )


def region_prefix(spec: str) -> str:
    if spec == "H_CDR3_PEP":
        return "H_CDR3"
    if spec == "H_CDRs_PEP-L_CDRs_PEP":
        return "HL_CDRs"
    if spec == "H_FL_PEP-L_FL_PEP":
        return "HL_FL"
    return safe_filename(spec)


def make_library_key_from_sample_id(sample_id: object) -> str:
    text = str(sample_id).strip()
    parts = re.split(r"__+", text)
    kept: List[str] = []

    for part in parts:
        token = part.strip()
        if not token:
            continue

        # Old style: <LIB>_R2 -> <LIB>
        token = re.sub(r"_R\d+(?=_|$)", "", token, flags=re.IGNORECASE)

        # New FAO2 metadata tokens should not define the library.
        if re.match(r"^round-?R?\d+$", token, flags=re.IGNORECASE):
            continue
        if re.match(r"^cond-", token, flags=re.IGNORECASE):
            continue
        if re.match(r"^negtype-", token, flags=re.IGNORECASE):
            continue
        if re.match(r"^rep-", token, flags=re.IGNORECASE):
            continue
        if re.match(r"^dup\d+$", token, flags=re.IGNORECASE):
            continue

        if token:
            kept.append(token)

    return "__".join(kept) if kept else text


def add_library_keys(metadata: pd.DataFrame) -> pd.DataFrame:
    meta = metadata.copy()
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta["library_key"] = meta["sample_id"].apply(make_library_key_from_sample_id)
    meta["round_num"] = pd.to_numeric(meta.get("round", pd.NA), errors="coerce")
    return meta


def positive_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["condition"].astype(str).str.lower().eq("pos")].copy()


def negative_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["condition"].astype(str).str.lower().eq("neg")].copy()


# -----------------------------------------------------------------------------
# Metadata discovery
# -----------------------------------------------------------------------------


def prepare_sample_metadata(parser_out: str | Path, output_dir: str | Path) -> pd.DataFrame:
    files = discover_annotated_files(parser_out)
    if not files:
        raise FileNotFoundError(f"No *_annotated.csv files found under {parser_out}")
    metadata = build_sample_metadata(files)
    metadata = add_library_keys(metadata)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(out_dir / "sample_metadata.csv", index=False)
    return metadata


# -----------------------------------------------------------------------------
# Per-library loading
# -----------------------------------------------------------------------------


def load_library_annotated(
    lib_meta: pd.DataFrame,
    *,
    write_annotated_uid: bool = False,
    strict_topology: bool = True,
) -> pd.DataFrame:
    tables: List[pd.DataFrame] = []

    for _, meta in lib_meta.iterrows():
        f = Path(meta["annotated_file"])
        df = pd.read_csv(f, keep_default_na=False)
        library_type = str(meta.get("library_type", "unknown"))

        df = append_seq_uid_column(
            df,
            library_type=library_type,
            strict_topology=strict_topology,
        )
        if write_annotated_uid:
            df.to_csv(f, index=False)

        count_col = find_count_col(df)
        df["__count"] = pd.to_numeric(df[count_col], errors="coerce").fillna(0).astype(int)
        df["sample_id"] = str(meta["sample_id"])
        df["library_key"] = str(meta["library_key"])
        df["library_type"] = library_type
        df["round"] = meta.get("round", pd.NA)
        df["round_num"] = pd.to_numeric(meta.get("round", pd.NA), errors="coerce")
        df["condition"] = meta.get("condition", "unknown")
        df["negative_type"] = meta.get("negative_type", "")
        df["source_annotated_file"] = str(f)
        df["display_id"] = [
            make_display_id(row, f"{meta['sample_id']}__row{i + 1}")
            for i, row in df.iterrows()
        ]
        tables.append(df)

    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True, sort=False)


# -----------------------------------------------------------------------------
# Region support
# -----------------------------------------------------------------------------


def build_negative_uid_map(df: pd.DataFrame) -> Dict[str, List[str]]:
    neg = negative_rows(df)
    neg = neg[(neg["seq_uid"].astype(str).str.len() > 0) & (neg["__count"] > 0)]
    out: Dict[str, List[str]] = {}
    for uid, sub in neg.groupby("seq_uid"):
        out[str(uid)] = sorted(set(sub["sample_id"].astype(str)))
    return out


def build_region_support_tables(
    annotated: pd.DataFrame,
    metadata: pd.DataFrame,
    region_specs: Sequence[str],
    output_dir: str | Path,
) -> Dict[str, pd.DataFrame]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    region_tables: Dict[str, pd.DataFrame] = {}

    valid = annotated[annotated["seq_uid"].astype(str).str.len() > 0].copy()
    neg_uid_to_samples = build_negative_uid_map(valid)
    sample_order = list(metadata["sample_id"].astype(str))

    for spec in region_specs:
        df = valid.copy()
        df["region_spec"] = spec
        df["region_key"] = build_region_key_series(df, spec)
        df = df[df["region_key"].astype(str).str.len() > 0].copy()

        out_path = out_dir / safe_filename(f"region_support__{spec}.csv")
        if df.empty:
            empty = pd.DataFrame(columns=["region_spec", "region_key"])
            empty.to_csv(out_path, index=False)
            region_tables[spec] = empty
            continue

        grouped = (
            df.groupby(["region_spec", "region_key", "sample_id"], as_index=False)["__count"]
            .sum()
            .rename(columns={"__count": "region_sample_count"})
        )

        count_wide = grouped.pivot_table(
            index=["region_spec", "region_key"],
            columns="sample_id",
            values="region_sample_count",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()
        count_wide.columns = [str(c) for c in count_wide.columns]

        pos = positive_rows(df)
        rep_records: List[Dict[str, object]] = []

        for (region_spec, region_key), sub in pos.groupby(["region_spec", "region_key"]):
            sub_valid = sub[sub["seq_uid"].astype(str).str.len() > 0].copy()
            if sub_valid.empty:
                continue

            sub_valid["__round_sort"] = pd.to_numeric(sub_valid["round_num"], errors="coerce").fillna(-1)
            latest_round = sub_valid["__round_sort"].max()
            latest = sub_valid[sub_valid["__round_sort"].eq(latest_round)].copy()
            latest = latest.sort_values(["__count", "seq_uid"], ascending=[False, True])
            rep = latest.iloc[0]

            source_sample = str(rep["sample_id"])
            region_count_same_sample = int(
                grouped[
                    grouped["region_spec"].eq(region_spec)
                    & grouped["region_key"].eq(region_key)
                    & grouped["sample_id"].astype(str).eq(source_sample)
                ]["region_sample_count"].sum()
            )
            rep_count = int(rep["__count"])
            fraction = rep_count / region_count_same_sample if region_count_same_sample > 0 else np.nan
            rep_uid = str(rep["seq_uid"])
            neg_samples = neg_uid_to_samples.get(rep_uid, [])

            rep_records.append(
                {
                    "region_spec": region_spec,
                    "region_key": region_key,
                    "representative_seq_uid": rep_uid,
                    "representative_display_id": rep.get("display_id", ""),
                    "representative_source_sample": source_sample,
                    "representative_count": rep_count,
                    "representative_region_count": region_count_same_sample,
                    "representative_region_fraction": fraction,
                    "negative_flag": "negative_hit" if neg_samples else "clean_or_unknown",
                    "negative_sample_hit": ";".join(neg_samples),
                }
            )

        reps = pd.DataFrame(rep_records)
        if reps.empty:
            merged = count_wide
        else:
            merged = count_wide.merge(reps, on=["region_spec", "region_key"], how="left")

        front = ["region_spec", "region_key"]
        sample_cols = [s for s in sample_order if s in merged.columns]
        extra_cols = [c for c in merged.columns if c not in front + sample_cols]
        merged = merged[front + sample_cols + extra_cols]
        merged.to_csv(out_path, index=False)
        region_tables[spec] = merged

    return region_tables


# -----------------------------------------------------------------------------
# Candidate table
# -----------------------------------------------------------------------------


def build_candidate_table(
    annotated: pd.DataFrame,
    metadata: pd.DataFrame,
    region_tables: Dict[str, pd.DataFrame],
    llm_clusters: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    valid = annotated[annotated["seq_uid"].astype(str).str.len() > 0].copy()
    if valid.empty:
        return pd.DataFrame()

    sample_order = list(metadata["sample_id"].astype(str))
    count_wide = valid.pivot_table(
        index="seq_uid",
        columns="sample_id",
        values="__count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    count_wide.columns = [str(c) for c in count_wide.columns]
    for sid in sample_order:
        if sid not in count_wide.columns:
            count_wide[sid] = 0

    rows: List[Dict[str, object]] = []
    pos = positive_rows(valid)

    for uid, sub_all in valid.groupby("seq_uid"):
        sub_pos = pos[pos["seq_uid"].eq(uid)].copy()
        if not sub_pos.empty:
            sub_pos["__round_sort"] = pd.to_numeric(sub_pos["round_num"], errors="coerce").fillna(-1)
            latest_round = sub_pos["__round_sort"].max()
            choices = sub_pos[sub_pos["__round_sort"].eq(latest_round)].copy()
        else:
            choices = sub_all.copy()
            choices["__round_sort"] = pd.to_numeric(choices["round_num"], errors="coerce").fillna(-1)

        choices = choices.sort_values(["__count", "sample_id"], ascending=[False, True])
        rep = choices.iloc[0]

        rec: Dict[str, object] = {"seq_uid": uid}
        for col in CORE_SEQUENCE_COLUMNS:
            if col in rep.index:
                rec[col] = rep[col]
        rec.update(
            {
                "display_id": rep.get("display_id", ""),
                "source_sample_id": rep.get("sample_id", ""),
                "source_annotated_file": rep.get("source_annotated_file", ""),
                "library_key": rep.get("library_key", ""),
                "library_type": rep.get("library_type", "unknown"),
                "representative_count_in_source": int(rep.get("__count", 0)),
            }
        )
        rows.append(rec)

    base = pd.DataFrame(rows)
    out = base.merge(count_wide, on="seq_uid", how="left")

    pos_samples = metadata[metadata["condition"].astype(str).str.lower().eq("pos")]["sample_id"].astype(str).tolist()
    neg_samples = metadata[metadata["condition"].astype(str).str.lower().eq("neg")]["sample_id"].astype(str).tolist()

    for sid in pos_samples + neg_samples:
        if sid not in out.columns:
            out[sid] = 0

    out["total_positive_count"] = out[pos_samples].sum(axis=1) if pos_samples else 0
    out["total_negative_count"] = out[neg_samples].sum(axis=1) if neg_samples else 0
    out["negative_flag"] = np.where(out["total_negative_count"] > 0, "negative_hit", "clean_or_unknown")
    out["negative_sample_hit"] = out.apply(
        lambda r: ";".join([sid for sid in neg_samples if int(r.get(sid, 0)) > 0]), axis=1
    )
    out["detected_positive_samples"] = out.apply(
        lambda r: ";".join([sid for sid in pos_samples if int(r.get(sid, 0)) > 0]), axis=1
    )
    out["detected_positive_sample_count"] = out.apply(
        lambda r: sum(int(r.get(sid, 0)) > 0 for sid in pos_samples), axis=1
    )
    out["trajectory_class"] = out.apply(lambda r: classify_trajectory(r, metadata), axis=1)

    for spec, table in region_tables.items():
        if table.empty:
            continue

        prefix = region_prefix(spec)
        out[f"{prefix}_region_key"] = out.apply(lambda r: build_region_key(r, spec), axis=1)

        keep_cols = [
            "region_key",
            "representative_seq_uid",
            "representative_region_fraction",
            "representative_count",
            "representative_region_count",
            "negative_flag",
            "negative_sample_hit",
        ]
        keep_cols = [c for c in keep_cols if c in table.columns]
        tmp = table[keep_cols].copy()
        tmp = tmp.rename(
            columns={
                "representative_seq_uid": f"{prefix}_representative_seq_uid",
                "representative_region_fraction": f"{prefix}_representative_region_fraction",
                "representative_count": f"{prefix}_representative_count",
                "representative_region_count": f"{prefix}_representative_region_count",
                "negative_flag": f"{prefix}_region_negative_flag",
                "negative_sample_hit": f"{prefix}_region_negative_sample_hit",
            }
        )
        out = out.merge(tmp, left_on=f"{prefix}_region_key", right_on="region_key", how="left")
        out = out.drop(columns=["region_key"], errors="ignore")

        rep_col = f"{prefix}_representative_seq_uid"
        if rep_col in out.columns:
            out[f"{prefix}_is_region_representative"] = out["seq_uid"].astype(str).eq(out[rep_col].astype(str))

    if llm_clusters is not None and not llm_clusters.empty:
        out = merge_llm_clusters(out, llm_clusters)

    priority_result = out.apply(assign_priority, axis=1)
    out["priority_tier"] = [x[0] for x in priority_result]
    out["priority_class"] = [x[1] for x in priority_result]
    out["decision_reason"] = [x[2] for x in priority_result]

    front = [
        "seq_uid",
        "priority_tier",
        "priority_class",
        "decision_reason",
        "negative_flag",
        "display_id",
        "source_sample_id",
        "library_key",
        "library_type",
        "trajectory_class",
    ]
    front = [c for c in front if c in out.columns]
    out = out[front + [c for c in out.columns if c not in front]]
    return out


# -----------------------------------------------------------------------------
# Priority / trajectory
# -----------------------------------------------------------------------------


def classify_trajectory(row: pd.Series, metadata: pd.DataFrame) -> str:
    pos_meta = metadata[metadata["condition"].astype(str).str.lower().eq("pos")].copy()
    if pos_meta.empty:
        return "positive_unknown"

    pos_meta["round_sort"] = pd.to_numeric(pos_meta["round"], errors="coerce").fillna(-1)
    pos_meta = pos_meta.sort_values(["round_sort", "sample_id"])
    counts = [int(row.get(str(sid), 0)) for sid in pos_meta["sample_id"].astype(str)]
    nonzero = [(i, c) for i, c in enumerate(counts) if c > 0]

    if not nonzero:
        return "not_in_positive"
    if len(nonzero) == 1:
        i, _ = nonzero[0]
        if i == len(counts) - 1:
            return "final_only"
        return "one_positive_sample_only"

    final = counts[-1]
    prev = counts[-2] if len(counts) >= 2 else 0
    first_nonzero = nonzero[0][1]

    if final == 0:
        return "lost_before_final"
    if final < prev:
        return "declining_late"

    start_idx = nonzero[0][0]
    tail = counts[start_idx:]
    if all(tail[i] <= tail[i + 1] for i in range(len(tail) - 1)):
        if first_nonzero <= 50 and final >= max(500, first_nonzero * 10):
            return "rare_fast"
        return "steady_rising"

    if final >= max(counts):
        return "late_rising"
    return "mixed"


def assign_priority(row: pd.Series) -> Tuple[str, str, str]:
    if str(row.get("negative_flag", "")).lower() == "negative_hit":
        return "Reject_dirty", "E_negative_dirty", "seq_uid observed in negative sample"

    traj = str(row.get("trajectory_class", ""))
    total_pos = int(row.get("total_positive_count", 0))
    detected = int(row.get("detected_positive_sample_count", 0))

    if traj in {"steady_rising", "late_rising"} and total_pos > 0:
        return "Pick", "A_rising_candidate", f"{traj}; positive count={total_pos}"
    if traj == "rare_fast":
        return "Diversity_pick", "B_rare_fast_supported", "rare-fast positive trajectory"
    if detected >= 2 and total_pos > 0:
        return "Pick", "A_multi_round_candidate", f"detected in {detected} positive samples"
    if traj in {"final_only", "one_positive_sample_only"} and total_pos >= 100:
        return "Backup", "C_single_round_candidate", f"single positive sample; count={total_pos}"
    if total_pos > 0:
        return "Backup", "D_low_or_mixed_evidence", f"positive count={total_pos}; trajectory={traj}"
    return "Deprioritize", "F_no_positive_evidence", "not detected in positive samples"


# -----------------------------------------------------------------------------
# LLM clustering
# -----------------------------------------------------------------------------


def load_llm_clusters(path: Optional[str | Path]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"LLM clustering file not found: {p}")
    if p.suffix.lower() in {".xlsx", ".xls"}:
        llm = pd.read_excel(p)
    else:
        llm = pd.read_csv(p, keep_default_na=False)

    if "seq_uid" not in llm.columns:
        if "H_FL_PEP" in llm.columns:
            if "L_FL_PEP" not in llm.columns:
                llm["L_FL_PEP"] = ""
            llm["seq_uid"] = [make_seq_uid(h, l) for h, l in zip(llm["H_FL_PEP"], llm["L_FL_PEP"])]
        else:
            raise ValueError("LLM clustering file must contain seq_uid or H_FL_PEP/L_FL_PEP columns.")

    drop_cols = [c for c in ["H_FL_PEP", "L_FL_PEP", "H_CDR3_PEP", "L_CDR3_PEP"] if c in llm.columns]
    llm = llm.drop(columns=drop_cols, errors="ignore")
    llm = llm.drop_duplicates("seq_uid")
    return llm


def merge_llm_clusters(candidates: pd.DataFrame, llm: pd.DataFrame) -> pd.DataFrame:
    return candidates.merge(llm, on="seq_uid", how="left")


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------


def write_recommended_candidates(candidates: pd.DataFrame, output_path: Path) -> None:
    if candidates.empty:
        candidates.to_csv(output_path, index=False)
        return

    keep_tiers = {"Pick", "Diversity_pick", "Backup", "Control"}
    out = candidates[
        candidates["priority_tier"].astype(str).isin(keep_tiers)
        & candidates["negative_flag"].astype(str).ne("negative_hit")
    ].copy()
    out.to_csv(output_path, index=False)


# -----------------------------------------------------------------------------
# Main API
# -----------------------------------------------------------------------------


def run_prioritization(
    parser_out: str | Path,
    output_dir: str | Path,
    *,
    llm_clusters: Optional[str | Path] = None,
    region_specs: Sequence[str] = DEFAULT_REGION_SPECS,
    write_annotated_uid: bool = False,
    strict_topology: bool = True,
    split_by_library: bool = True,
    write_global_table: bool = False,
    library_key_filter: Optional[str] = None,
) -> Dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = prepare_sample_metadata(parser_out, out_dir)
    llm = load_llm_clusters(llm_clusters)

    outputs: Dict[str, Path] = {
        "sample_metadata": out_dir / "sample_metadata.csv",
    }
    manifest_records: List[Dict[str, object]] = []

    if not split_by_library:
        # Backward-compatible global mode. This intentionally loads all samples.
        metadata_global = metadata.copy()
        metadata_global["library_key"] = "ALL_LIBRARIES"
        annotated = load_library_annotated(
            metadata_global,
            write_annotated_uid=write_annotated_uid,
            strict_topology=strict_topology,
        )
        region_dir = out_dir / "region_support"
        region_tables = build_region_support_tables(annotated, metadata_global, region_specs, region_dir)
        candidates = build_candidate_table(annotated, metadata_global, region_tables, llm)
        candidate_path = out_dir / "candidate_prioritization_table.csv"
        candidates.to_csv(candidate_path, index=False)
        write_recommended_candidates(candidates, out_dir / "recommended_candidates.csv")
        outputs["candidate_table"] = candidate_path
        outputs["region_support_dir"] = region_dir
        manifest_records.append(
            {
                "library_key": "ALL_LIBRARIES",
                "library_dir": str(out_dir),
                "n_samples": int(len(metadata_global)),
                "n_positive_samples": int((metadata_global["condition"].astype(str).str.lower() == "pos").sum()),
                "n_negative_samples": int((metadata_global["condition"].astype(str).str.lower() == "neg").sum()),
                "n_candidates": int(len(candidates)),
                "candidate_table": str(candidate_path),
            }
        )
    else:
        library_root = out_dir / "by_library"
        library_root.mkdir(parents=True, exist_ok=True)

        library_keys = sorted(metadata["library_key"].dropna().astype(str).unique())
        if library_key_filter:
            pattern = str(library_key_filter)
            library_keys = [k for k in library_keys if pattern in k]

        for library_key in library_keys:
            lib_meta = metadata[metadata["library_key"].astype(str).eq(library_key)].copy()
            if lib_meta.empty:
                continue

            lib_dir = library_root / safe_filename(library_key)
            lib_dir.mkdir(parents=True, exist_ok=True)
            lib_meta.to_csv(lib_dir / "sample_metadata.csv", index=False)

            print(f"Processing library: {library_key} ({len(lib_meta)} samples)")

            lib_annotated = load_library_annotated(
                lib_meta,
                write_annotated_uid=write_annotated_uid,
                strict_topology=strict_topology,
            )

            region_dir = lib_dir / "region_support"
            region_tables = build_region_support_tables(lib_annotated, lib_meta, region_specs, region_dir)
            candidates = build_candidate_table(lib_annotated, lib_meta, region_tables, llm)

            candidate_path = lib_dir / "candidate_prioritization_table.csv"
            candidates.to_csv(candidate_path, index=False)
            write_recommended_candidates(candidates, lib_dir / "recommended_candidates.csv")

            n_pos = int((lib_meta["condition"].astype(str).str.lower() == "pos").sum())
            n_neg = int((lib_meta["condition"].astype(str).str.lower() == "neg").sum())
            manifest_records.append(
                {
                    "library_key": library_key,
                    "library_dir": str(lib_dir),
                    "n_samples": int(len(lib_meta)),
                    "n_positive_samples": n_pos,
                    "n_negative_samples": n_neg,
                    "n_candidates": int(len(candidates)),
                    "candidate_table": str(candidate_path),
                }
            )

            del lib_annotated, region_tables, candidates
            gc.collect()

        library_manifest = pd.DataFrame(manifest_records)
        library_manifest_path = out_dir / "library_manifest.csv"
        library_manifest.to_csv(library_manifest_path, index=False)
        outputs["library_manifest"] = library_manifest_path
        outputs["library_output_root"] = library_root

        if write_global_table:
            candidate_paths = [Path(r["candidate_table"]) for r in manifest_records if Path(r["candidate_table"]).exists()]
            if candidate_paths:
                global_candidates = pd.concat(
                    [pd.read_csv(p, keep_default_na=False) for p in candidate_paths],
                    ignore_index=True,
                    sort=False,
                )
                global_path = out_dir / "candidate_prioritization_table.global.csv"
                global_candidates.to_csv(global_path, index=False)
                outputs["global_candidate_table"] = global_path

    manifest = {
        "parser_out": str(parser_out),
        "output_dir": str(output_dir),
        "llm_clusters": str(llm_clusters) if llm_clusters else "",
        "region_specs": list(region_specs),
        "write_annotated_uid": write_annotated_uid,
        "strict_topology": strict_topology,
        "split_by_library": split_by_library,
        "write_global_table": write_global_table,
        "library_key_filter": library_key_filter or "",
        "libraries": manifest_records,
    }
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    outputs["manifest"] = manifest_path

    return outputs


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FAO2 candidate prioritization post-processor")
    p.add_argument("--parser-out", required=True, help="Existing FAO parser_outputs directory")
    p.add_argument("--out", required=True, help="Output directory for prioritization outputs")
    p.add_argument("--llm-clusters", default=None, help="Optional CSV/XLSX file with seq_uid or H_FL_PEP/L_FL_PEP and LLM cluster columns")
    p.add_argument(
        "--region-specs",
        nargs="+",
        default=DEFAULT_REGION_SPECS,
        help="Region specs to summarize",
    )
    p.add_argument(
        "--write-annotated-uid",
        action="store_true",
        help="Append seq_uid to each existing *_annotated.csv in place",
    )
    p.add_argument(
        "--no-strict-topology",
        action="store_true",
        help="Generate seq_uid whenever H_FL_PEP exists, without scFv H/L completeness filtering",
    )
    p.add_argument(
        "--no-split-by-library",
        action="store_true",
        help="Write one global candidate table instead of per-library output folders; this loads all samples",
    )
    p.add_argument(
        "--write-global-table",
        action="store_true",
        help="After per-library processing, concatenate candidate tables into a global table",
    )
    p.add_argument(
        "--library-key",
        default=None,
        help="Optional substring filter; process only libraries whose library_key contains this text",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    outputs = run_prioritization(
        parser_out=args.parser_out,
        output_dir=args.out,
        llm_clusters=args.llm_clusters,
        region_specs=args.region_specs,
        write_annotated_uid=args.write_annotated_uid,
        strict_topology=not args.no_strict_topology,
        split_by_library=not args.no_split_by_library,
        write_global_table=args.write_global_table,
        library_key_filter=args.library_key,
    )
    print("FAO2 prioritization complete.")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
