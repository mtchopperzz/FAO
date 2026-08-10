# -*- coding: utf-8 -*-
"""
FAO2 per-library candidate prioritization.

The module reads one library at a time. LLM integration is intentionally
disabled in this version.
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
from .uid import append_seq_uid_column, clean_str


DEFAULT_REGION_SPECS = [
    "H_CDR3_PEP",
    "H_CDRs_PEP-L_CDRs_PEP",
    "H_FL_PEP-L_FL_PEP",
]

SAMPLE_METADATA_OUTPUT_COLUMNS = [
    "sample_id",
    "library_key",
    "library_type",
    "round",
    "condition",
    "negative_type",
    "annotated_file",
]

LIBRARY_MANIFEST_COLUMNS = [
    "library_key",
    "library_dir",
    "n_samples",
    "n_positive_samples",
    "n_negative_samples",
    "n_candidates",
    "candidate_table",
]

CORE_SEQUENCE_COLUMNS = [
    "H_CDR3_PEP",
    "H_CDR3_DEGENERATED_PEP",
    "H_CDRs_PEP",
    "L_CDRs_PEP",
    "HL_CDRs_DEGENERATED_PEP",
    "H_FL_PEP",
    "L_FL_PEP",
    "L_CDR3_PEP",
    "H_V_Gene",
    "H_J_Gene",
    "L_V_Gene",
    "L_J_Gene",
    "H_FL_DNA",
    "L_FL_DNA",
]

# Reduced 11-class amino-acid alphabet agreed for similarity inspection:
# DE / ILV / A / NQ / RHK / ST / C / P / G / M / F / W / Y 

AA_DEGENERATION_MAP = {
    "D": "B",
    "E": "B",
    "I": "J",
    "L": "J",
    "V": "J",
    "A": "A",
    "N": "O",
    "Q": "O",
    "R": "U",
    "H": "U",
    "K": "U",
    "S": "X",
    "T": "X",
    "C": "C",
    "P": "P",
    "G": "G",
    "M": "M",
    "F": "F",
    "W": "W",
    "Y": "Y",
}


def find_count_col(df: pd.DataFrame) -> str:
    for column in ["Count", "count", "Read_Count", "read_count", "Reads", "reads"]:
        if column in df.columns:
            return column
    raise ValueError(
        "Could not find a read-count column. "
        "Expected Count/count/Read_Count/read_count/Reads/reads."
    )


def make_display_id(row: pd.Series, fallback: str) -> str:
    for column in [
        "display_id",
        "ID",
        "Seq_ID",
        "seq_id",
        "sequence_id",
        "Name",
        "name",
    ]:
        if column in row.index and clean_str(row[column]):
            return clean_str(row[column])
    return fallback


def degenerate_sequence(value: object) -> str:
    """Convert an amino-acid sequence to the reduced 11-class alphabet."""
    sequence = re.sub(r"\s+", "", clean_str(value)).upper()
    if not sequence:
        return ""
    return "".join(AA_DEGENERATION_MAP.get(residue, "X") for residue in sequence)


def add_degenerated_sequence_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the two reduced-alphabet columns used by the current analysis layers.

    H_CDR3_DEGENERATED_PEP:
        reduced H_CDR3 sequence

    HL_CDRs_DEGENERATED_PEP:
        reduced H_CDRs and L_CDRs joined in canonical H-L order
    """
    out = df.copy()

    if "H_CDR3_PEP" in out.columns:
        out["H_CDR3_DEGENERATED_PEP"] = out["H_CDR3_PEP"].map(
            degenerate_sequence
        )
    else:
        out["H_CDR3_DEGENERATED_PEP"] = ""

    h_cdrs = (
        out["H_CDRs_PEP"].map(degenerate_sequence)
        if "H_CDRs_PEP" in out.columns
        else pd.Series([""] * len(out), index=out.index)
    )
    l_cdrs = (
        out["L_CDRs_PEP"].map(degenerate_sequence)
        if "L_CDRs_PEP" in out.columns
        else pd.Series([""] * len(out), index=out.index)
    )

    out["HL_CDRs_DEGENERATED_PEP"] = (
        "H:" + h_cdrs.astype(str) + "|L:" + l_cdrs.astype(str)
    )
    return out


def region_columns(region_spec: str) -> List[str]:
    return [item.strip() for item in str(region_spec).split("-") if item.strip()]


def build_region_key(row: pd.Series, region_spec: str) -> str:
    parts = [clean_str(row.get(column, "")) for column in region_columns(region_spec)]
    if not any(parts):
        return ""
    return "-".join(parts)


def build_region_key_series(df: pd.DataFrame, region_spec: str) -> pd.Series:
    columns = region_columns(region_spec)
    if not columns:
        return pd.Series([""] * len(df), index=df.index)

    values: List[pd.Series] = []
    for column in columns:
        if column in df.columns:
            values.append(df[column].map(clean_str))
        else:
            values.append(pd.Series([""] * len(df), index=df.index))

    key = values[0].astype(str)
    for series in values[1:]:
        key = key + "-" + series.astype(str)

    all_empty = pd.concat(values, axis=1).apply(
        lambda row: not any(row.astype(str)),
        axis=1,
    )
    return key.mask(all_empty, "")


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

        token = re.sub(r"_R\d+(?=_|$)", "", token, flags=re.IGNORECASE)

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

        kept.append(token)

    return "__".join(kept) if kept else text


def add_library_keys(metadata: pd.DataFrame) -> pd.DataFrame:
    out = metadata.copy()
    out["sample_id"] = out["sample_id"].astype(str)
    out["library_key"] = out["sample_id"].apply(make_library_key_from_sample_id)
    out["round_num"] = pd.to_numeric(out.get("round", pd.NA), errors="coerce")
    return out


def metadata_output_view(metadata: pd.DataFrame) -> pd.DataFrame:
    out = metadata.copy()
    for column in SAMPLE_METADATA_OUTPUT_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    return out[SAMPLE_METADATA_OUTPUT_COLUMNS]


def positive_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["condition"].astype(str).str.lower().eq("pos")].copy()


def negative_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["condition"].astype(str).str.lower().eq("neg")].copy()


def prepare_sample_metadata(
    parser_out: str | Path,
    output_dir: str | Path,
) -> pd.DataFrame:
    files = discover_annotated_files(parser_out)
    if not files:
        raise FileNotFoundError(f"No *_annotated.csv files found under {parser_out}")

    metadata = add_library_keys(build_sample_metadata(files))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata_output_view(metadata).to_csv(
        output / "sample_metadata.csv",
        index=False,
    )
    return metadata


def load_library_annotated(
    lib_meta: pd.DataFrame,
    *,
    write_annotated_uid: bool = False,
    strict_topology: bool = True,
) -> pd.DataFrame:
    tables: List[pd.DataFrame] = []

    for _, meta in lib_meta.iterrows():
        annotated_file = Path(meta["annotated_file"])
        df = pd.read_csv(annotated_file, keep_default_na=False)
        library_type = str(meta.get("library_type", "unknown"))

        df = append_seq_uid_column(
            df,
            library_type=library_type,
            strict_topology=strict_topology,
        )

        if write_annotated_uid:
            # Parser output remains unchanged except for the final seq_uid column.
            df.to_csv(annotated_file, index=False)

        df = add_degenerated_sequence_columns(df)

        count_column = find_count_col(df)
        df["__count"] = (
            pd.to_numeric(df[count_column], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        df["sample_id"] = str(meta["sample_id"])
        df["library_key"] = str(meta["library_key"])
        df["library_type"] = library_type
        df["round"] = meta.get("round", pd.NA)
        df["round_num"] = pd.to_numeric(
            meta.get("round", pd.NA),
            errors="coerce",
        )
        df["condition"] = meta.get("condition", "unknown")
        df["negative_type"] = meta.get("negative_type", "")
        df["display_id"] = [
            make_display_id(row, f"{meta['sample_id']}__row{index + 1}")
            for index, (_, row) in enumerate(df.iterrows())
        ]

        tables.append(df)

    if not tables:
        return pd.DataFrame()

    return pd.concat(tables, ignore_index=True, sort=False)


def build_negative_uid_map(df: pd.DataFrame) -> Dict[str, List[str]]:
    negative = negative_rows(df)
    negative = negative[
        (negative["seq_uid"].astype(str).str.len() > 0)
        & (negative["__count"] > 0)
    ]

    out: Dict[str, List[str]] = {}
    for uid, group in negative.groupby("seq_uid"):
        out[str(uid)] = sorted(set(group["sample_id"].astype(str)))
    return out


def build_region_support_tables(
    annotated: pd.DataFrame,
    metadata: pd.DataFrame,
    region_specs: Sequence[str],
    output_dir: str | Path,
) -> Dict[str, pd.DataFrame]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    region_tables: Dict[str, pd.DataFrame] = {}
    valid = annotated[annotated["seq_uid"].astype(str).str.len() > 0].copy()
    negative_uid_to_samples = build_negative_uid_map(valid)
    sample_order = list(metadata["sample_id"].astype(str))

    for spec in region_specs:
        df = valid.copy()
        df["region_spec"] = spec
        df["region_key"] = build_region_key_series(df, spec)
        df = df[df["region_key"].astype(str).str.len() > 0].copy()

        output_path = output / safe_filename(f"region_support__{spec}.csv")

        if df.empty:
            empty = pd.DataFrame(columns=["region_spec", "region_key"])
            empty.to_csv(output_path, index=False)
            region_tables[spec] = empty
            continue

        grouped = (
            df.groupby(
                ["region_spec", "region_key", "sample_id"],
                as_index=False,
            )["__count"]
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
        count_wide.columns = [str(column) for column in count_wide.columns]

        positive = positive_rows(df)
        representative_records: List[Dict[str, object]] = []

        for (region_spec, region_key), group in positive.groupby(
            ["region_spec", "region_key"]
        ):
            group = group[group["seq_uid"].astype(str).str.len() > 0].copy()
            if group.empty:
                continue

            group["__round_sort"] = (
                pd.to_numeric(group["round_num"], errors="coerce")
                .fillna(-1)
            )
            latest_round = group["__round_sort"].max()
            latest = group[group["__round_sort"].eq(latest_round)].copy()
            latest = latest.sort_values(
                ["__count", "seq_uid"],
                ascending=[False, True],
            )
            representative = latest.iloc[0]

            source_sample = str(representative["sample_id"])
            region_count_same_sample = int(
                grouped[
                    grouped["region_spec"].eq(region_spec)
                    & grouped["region_key"].eq(region_key)
                    & grouped["sample_id"].astype(str).eq(source_sample)
                ]["region_sample_count"].sum()
            )
            representative_count = int(representative["__count"])
            fraction = (
                representative_count / region_count_same_sample
                if region_count_same_sample > 0
                else np.nan
            )

            representative_uid = str(representative["seq_uid"])
            negative_samples = negative_uid_to_samples.get(
                representative_uid,
                [],
            )

            representative_records.append(
                {
                    "region_spec": region_spec,
                    "region_key": region_key,
                    "representative_seq_uid": representative_uid,
                    "representative_display_id": representative.get(
                        "display_id",
                        "",
                    ),
                    "representative_source_sample": source_sample,
                    "representative_count": representative_count,
                    "representative_region_count": region_count_same_sample,
                    "representative_region_fraction": fraction,
                    "negative_flag": (
                        "negative_hit"
                        if negative_samples
                        else "clean_or_unknown"
                    ),
                    "negative_sample_hit": ";".join(negative_samples),
                }
            )

        representatives = pd.DataFrame(representative_records)
        merged = (
            count_wide
            if representatives.empty
            else count_wide.merge(
                representatives,
                on=["region_spec", "region_key"],
                how="left",
            )
        )

        front = ["region_spec", "region_key"]
        sample_columns = [
            sample for sample in sample_order if sample in merged.columns
        ]
        other_columns = [
            column
            for column in merged.columns
            if column not in front + sample_columns
        ]
        merged = merged[front + sample_columns + other_columns]
        merged.to_csv(output_path, index=False)
        region_tables[spec] = merged

    return region_tables


def build_candidate_table(
    annotated: pd.DataFrame,
    metadata: pd.DataFrame,
    region_tables: Dict[str, pd.DataFrame],
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
    count_wide.columns = [str(column) for column in count_wide.columns]

    for sample_id in sample_order:
        if sample_id not in count_wide.columns:
            count_wide[sample_id] = 0

    positive = positive_rows(valid)
    records: List[Dict[str, object]] = []

    for uid, all_rows in valid.groupby("seq_uid"):
        positive_rows_for_uid = positive[positive["seq_uid"].eq(uid)].copy()

        if not positive_rows_for_uid.empty:
            positive_rows_for_uid["__round_sort"] = (
                pd.to_numeric(
                    positive_rows_for_uid["round_num"],
                    errors="coerce",
                ).fillna(-1)
            )
            latest_round = positive_rows_for_uid["__round_sort"].max()
            choices = positive_rows_for_uid[
                positive_rows_for_uid["__round_sort"].eq(latest_round)
            ].copy()
        else:
            choices = all_rows.copy()
            choices["__round_sort"] = (
                pd.to_numeric(choices["round_num"], errors="coerce")
                .fillna(-1)
            )

        choices = choices.sort_values(
            ["__count", "sample_id"],
            ascending=[False, True],
        )
        representative = choices.iloc[0]

        record: Dict[str, object] = {"seq_uid": uid}
        for column in CORE_SEQUENCE_COLUMNS:
            if column in representative.index:
                record[column] = representative[column]

        record.update(
            {
                "display_id": representative.get("display_id", ""),
                "source_sample_id": representative.get("sample_id", ""),
                "library_key": representative.get("library_key", ""),
                "library_type": representative.get(
                    "library_type",
                    "unknown",
                ),
                "representative_count_in_source": int(
                    representative.get("__count", 0)
                ),
            }
        )
        records.append(record)

    base = pd.DataFrame(records)
    out = base.merge(count_wide, on="seq_uid", how="left")

    positive_samples = metadata[
        metadata["condition"].astype(str).str.lower().eq("pos")
    ]["sample_id"].astype(str).tolist()
    negative_samples = metadata[
        metadata["condition"].astype(str).str.lower().eq("neg")
    ]["sample_id"].astype(str).tolist()

    for sample_id in positive_samples + negative_samples:
        if sample_id not in out.columns:
            out[sample_id] = 0

    out["total_positive_count"] = (
        out[positive_samples].sum(axis=1)
        if positive_samples
        else 0
    )
    out["total_negative_count"] = (
        out[negative_samples].sum(axis=1)
        if negative_samples
        else 0
    )
    out["negative_flag"] = np.where(
        out["total_negative_count"] > 0,
        "negative_hit",
        "clean_or_unknown",
    )
    out["negative_sample_hit"] = out.apply(
        lambda row: ";".join(
            [
                sample_id
                for sample_id in negative_samples
                if int(row.get(sample_id, 0)) > 0
            ]
        ),
        axis=1,
    )
    out["detected_positive_samples"] = out.apply(
        lambda row: ";".join(
            [
                sample_id
                for sample_id in positive_samples
                if int(row.get(sample_id, 0)) > 0
            ]
        ),
        axis=1,
    )
    out["detected_positive_sample_count"] = out.apply(
        lambda row: sum(
            int(row.get(sample_id, 0)) > 0
            for sample_id in positive_samples
        ),
        axis=1,
    )
    out["trajectory_class"] = out.apply(
        lambda row: classify_trajectory(row, metadata),
        axis=1,
    )

    for spec, table in region_tables.items():
        if table.empty:
            continue

        prefix = region_prefix(spec)
        out[f"{prefix}_region_key"] = out.apply(
            lambda row: build_region_key(row, spec),
            axis=1,
        )

        keep_columns = [
            "region_key",
            "representative_seq_uid",
            "representative_region_fraction",
            "representative_count",
            "representative_region_count",
            "negative_flag",
            "negative_sample_hit",
        ]
        keep_columns = [
            column for column in keep_columns if column in table.columns
        ]
        temporary = table[keep_columns].copy()
        temporary = temporary.rename(
            columns={
                "representative_seq_uid": (
                    f"{prefix}_representative_seq_uid"
                ),
                "representative_region_fraction": (
                    f"{prefix}_representative_region_fraction"
                ),
                "representative_count": (
                    f"{prefix}_representative_count"
                ),
                "representative_region_count": (
                    f"{prefix}_representative_region_count"
                ),
                "negative_flag": (
                    f"{prefix}_region_negative_flag"
                ),
                "negative_sample_hit": (
                    f"{prefix}_region_negative_sample_hit"
                ),
            }
        )
        out = out.merge(
            temporary,
            left_on=f"{prefix}_region_key",
            right_on="region_key",
            how="left",
        )
        out = out.drop(columns=["region_key"], errors="ignore")

        representative_column = f"{prefix}_representative_seq_uid"
        if representative_column in out.columns:
            out[f"{prefix}_is_region_representative"] = (
                out["seq_uid"].astype(str).eq(
                    out[representative_column].astype(str)
                )
            )

    # LLM integration is intentionally disabled in this version.
    # The old implementation loaded an external clustering table and merged all
    # non-sequence columns by seq_uid:
    #
    # llm = load_llm_clusters(llm_clusters)
    # if llm is not None and not llm.empty:
    #     out = merge_llm_clusters(out, llm)

    priority_results = out.apply(assign_priority, axis=1)
    out["priority_tier"] = [result[0] for result in priority_results]
    out["priority_class"] = [result[1] for result in priority_results]
    out["decision_reason"] = [result[2] for result in priority_results]

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
    front = [column for column in front if column in out.columns]
    return out[front + [column for column in out.columns if column not in front]]


def classify_trajectory(
    row: pd.Series,
    metadata: pd.DataFrame,
) -> str:
    positive_metadata = metadata[
        metadata["condition"].astype(str).str.lower().eq("pos")
    ].copy()

    if positive_metadata.empty:
        return "positive_unknown"

    positive_metadata["round_sort"] = (
        pd.to_numeric(positive_metadata["round"], errors="coerce")
        .fillna(-1)
    )
    positive_metadata = positive_metadata.sort_values(
        ["round_sort", "sample_id"]
    )

    counts = [
        int(row.get(str(sample_id), 0))
        for sample_id in positive_metadata["sample_id"].astype(str)
    ]
    nonzero = [
        (index, count)
        for index, count in enumerate(counts)
        if count > 0
    ]

    if not nonzero:
        return "not_in_positive"
    if len(nonzero) == 1:
        index, _ = nonzero[0]
        return (
            "final_only"
            if index == len(counts) - 1
            else "one_positive_sample_only"
        )

    final = counts[-1]
    previous = counts[-2] if len(counts) >= 2 else 0
    first_nonzero = nonzero[0][1]

    if final == 0:
        return "lost_before_final"
    if final < previous:
        return "declining_late"

    start_index = nonzero[0][0]
    tail = counts[start_index:]

    if all(
        tail[index] <= tail[index + 1]
        for index in range(len(tail) - 1)
    ):
        if (
            first_nonzero <= 50
            and final >= max(500, first_nonzero * 10)
        ):
            return "rare_fast"
        return "steady_rising"

    if final >= max(counts):
        return "late_rising"

    return "mixed"


def assign_priority(row: pd.Series) -> Tuple[str, str, str]:
    if str(row.get("negative_flag", "")).lower() == "negative_hit":
        return (
            "Reject_dirty",
            "E_negative_dirty",
            "seq_uid observed in negative sample",
        )

    trajectory = str(row.get("trajectory_class", ""))
    total_positive = int(row.get("total_positive_count", 0))
    detected = int(row.get("detected_positive_sample_count", 0))

    if trajectory in {"steady_rising", "late_rising"} and total_positive > 0:
        return (
            "Pick",
            "A_rising_candidate",
            f"{trajectory}; positive count={total_positive}",
        )
    if trajectory == "rare_fast":
        return (
            "Diversity_pick",
            "B_rare_fast_supported",
            "rare-fast positive trajectory",
        )
    if detected >= 2 and total_positive > 0:
        return (
            "Pick",
            "A_multi_round_candidate",
            f"detected in {detected} positive samples",
        )
    if (
        trajectory in {"final_only", "one_positive_sample_only"}
        and total_positive >= 100
    ):
        return (
            "Backup",
            "C_single_round_candidate",
            f"single positive sample; count={total_positive}",
        )
    if total_positive > 0:
        return (
            "Backup",
            "D_low_or_mixed_evidence",
            f"positive count={total_positive}; trajectory={trajectory}",
        )

    return (
        "Deprioritize",
        "F_no_positive_evidence",
        "not detected in positive samples",
    )


def write_recommended_candidates(
    candidates: pd.DataFrame,
    output_path: Path,
) -> None:
    if candidates.empty:
        candidates.to_csv(output_path, index=False)
        return

    keep_tiers = {"Pick", "Diversity_pick", "Backup", "Control"}
    output = candidates[
        candidates["priority_tier"].astype(str).isin(keep_tiers)
        & candidates["negative_flag"].astype(str).ne("negative_hit")
    ].copy()
    output.to_csv(output_path, index=False)


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
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    metadata = prepare_sample_metadata(parser_out, output)

    # LLM integration is intentionally paused. The argument is retained to keep
    # existing analysis_clean.py calls backward-compatible.
    if llm_clusters:
        print(
            "Warning: llm_clusters was provided, but LLM integration is "
            "disabled in this version and the file will be ignored."
        )

    outputs: Dict[str, Path] = {
        "sample_metadata": output / "sample_metadata.csv",
    }
    manifest_records: List[Dict[str, object]] = []

    if not split_by_library:
        global_metadata = metadata.copy()
        global_metadata["library_key"] = "ALL_LIBRARIES"

        annotated = load_library_annotated(
            global_metadata,
            write_annotated_uid=write_annotated_uid,
            strict_topology=strict_topology,
        )
        region_directory = output / "region_support"
        region_tables = build_region_support_tables(
            annotated,
            global_metadata,
            region_specs,
            region_directory,
        )
        candidates = build_candidate_table(
            annotated,
            global_metadata,
            region_tables,
        )

        candidate_path = output / "candidate_prioritization_table.csv"
        candidates.to_csv(candidate_path, index=False)
        write_recommended_candidates(
            candidates,
            output / "recommended_candidates.csv",
        )

        outputs["candidate_table"] = candidate_path
        outputs["region_support_dir"] = region_directory
        manifest_records.append(
            {
                "library_key": "ALL_LIBRARIES",
                "library_dir": str(output),
                "n_samples": int(len(global_metadata)),
                "n_positive_samples": int(
                    global_metadata["condition"]
                    .astype(str)
                    .str.lower()
                    .eq("pos")
                    .sum()
                ),
                "n_negative_samples": int(
                    global_metadata["condition"]
                    .astype(str)
                    .str.lower()
                    .eq("neg")
                    .sum()
                ),
                "n_candidates": int(len(candidates)),
                "candidate_table": str(candidate_path),
            }
        )
    else:
        library_root = output / "by_library"
        library_root.mkdir(parents=True, exist_ok=True)

        library_keys = sorted(
            metadata["library_key"].dropna().astype(str).unique()
        )
        if library_key_filter:
            pattern = str(library_key_filter)
            library_keys = [
                key for key in library_keys if pattern in key
            ]

        for library_key in library_keys:
            library_metadata = metadata[
                metadata["library_key"].astype(str).eq(library_key)
            ].copy()
            if library_metadata.empty:
                continue

            library_directory = library_root / safe_filename(library_key)
            library_directory.mkdir(parents=True, exist_ok=True)

            metadata_output_view(library_metadata).to_csv(
                library_directory / "sample_metadata.csv",
                index=False,
            )

            print(
                f"Processing library: {library_key} "
                f"({len(library_metadata)} samples)"
            )

            annotated = load_library_annotated(
                library_metadata,
                write_annotated_uid=write_annotated_uid,
                strict_topology=strict_topology,
            )
            region_directory = library_directory / "region_support"
            region_tables = build_region_support_tables(
                annotated,
                library_metadata,
                region_specs,
                region_directory,
            )
            candidates = build_candidate_table(
                annotated,
                library_metadata,
                region_tables,
            )

            candidate_path = (
                library_directory / "candidate_prioritization_table.csv"
            )
            candidates.to_csv(candidate_path, index=False)
            write_recommended_candidates(
                candidates,
                library_directory / "recommended_candidates.csv",
            )

            manifest_records.append(
                {
                    "library_key": library_key,
                    "library_dir": str(library_directory),
                    "n_samples": int(len(library_metadata)),
                    "n_positive_samples": int(
                        library_metadata["condition"]
                        .astype(str)
                        .str.lower()
                        .eq("pos")
                        .sum()
                    ),
                    "n_negative_samples": int(
                        library_metadata["condition"]
                        .astype(str)
                        .str.lower()
                        .eq("neg")
                        .sum()
                    ),
                    "n_candidates": int(len(candidates)),
                    "candidate_table": str(candidate_path),
                }
            )

            del annotated, region_tables, candidates
            gc.collect()

        library_manifest = pd.DataFrame(
            manifest_records,
            columns=LIBRARY_MANIFEST_COLUMNS,
        )
        library_manifest_path = output / "library_manifest.csv"
        library_manifest.to_csv(library_manifest_path, index=False)

        outputs["library_manifest"] = library_manifest_path
        outputs["library_output_root"] = library_root

        if write_global_table:
            candidate_paths = [
                Path(record["candidate_table"])
                for record in manifest_records
                if Path(record["candidate_table"]).exists()
            ]
            if candidate_paths:
                global_candidates = pd.concat(
                    [
                        pd.read_csv(path, keep_default_na=False)
                        for path in candidate_paths
                    ],
                    ignore_index=True,
                    sort=False,
                )
                global_path = (
                    output / "candidate_prioritization_table.global.csv"
                )
                global_candidates.to_csv(global_path, index=False)
                outputs["global_candidate_table"] = global_path

    manifest = {
        "parser_out": str(parser_out),
        "output_dir": str(output_dir),
        "uid_algorithm": (
            "SHA-256, first 128 bits, Base64url; "
            "payload is normalized VH<US>VL only"
        ),
        "llm_integration_enabled": False,
        "llm_clusters_argument_ignored": (
            str(llm_clusters) if llm_clusters else ""
        ),
        "region_specs": list(region_specs),
        "write_annotated_uid": write_annotated_uid,
        "strict_topology": strict_topology,
        "split_by_library": split_by_library,
        "write_global_table": bool(write_global_table),
        "library_key_filter": library_key_filter or "",
        "libraries": manifest_records,
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    outputs["manifest"] = manifest_path

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FAO2 candidate prioritization post-processor"
    )
    parser.add_argument(
        "--parser-out",
        required=True,
        help="Existing FAO parser_outputs directory",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for prioritization outputs",
    )
    parser.add_argument(
        "--llm-clusters",
        default=None,
        help="Reserved for future use; currently ignored",
    )
    parser.add_argument(
        "--region-specs",
        nargs="+",
        default=DEFAULT_REGION_SPECS,
        help="Region specs to summarize",
    )
    parser.add_argument(
        "--write-annotated-uid",
        action="store_true",
        help="Append seq_uid to each existing *_annotated.csv in place",
    )
    parser.add_argument(
        "--no-strict-topology",
        action="store_true",
        help=(
            "Generate seq_uid whenever H_FL_PEP exists, without scFv H/L "
            "completeness filtering"
        ),
    )
    parser.add_argument(
        "--no-split-by-library",
        action="store_true",
        help=(
            "Write one global candidate table instead of per-library output; "
            "this loads all samples"
        ),
    )
    parser.add_argument(
        "--write-global-table",
        action="store_true",
        help=(
            "After per-library processing, concatenate candidate tables "
            "into a global table"
        ),
    )
    parser.add_argument(
        "--library-key",
        default=None,
        help=(
            "Optional substring filter; process only libraries whose "
            "library_key contains this text"
        ),
    )
    return parser


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
