#!/usr/bin/env python3
"""Validate that FL DNA columns translate exactly to FL peptide columns."""

import argparse
from pathlib import Path

import pandas as pd
from Bio.Seq import Seq


def clean(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    return "" if text in {"", "NAN", "NONE"} else text


def check_chain(df: pd.DataFrame, chain: str) -> pd.DataFrame:
    dna_col = f"{chain}_FL_DNA"
    pep_col = f"{chain}_FL_PEP"
    if dna_col not in df.columns or pep_col not in df.columns:
        return pd.DataFrame()

    rows = []
    for idx, row in df.iterrows():
        dna = clean(row[dna_col])
        peptide = clean(row[pep_col])
        if not dna and not peptide:
            continue

        translated = ""
        if dna and len(dna) % 3 == 0:
            translated = str(Seq(dna).translate())

        rows.append({
            "row_index": idx,
            "ID": row.get("ID", ""),
            "chain": chain,
            "dna_length": len(dna),
            "dna_length_mod3": len(dna) % 3 if dna else "",
            "translated_peptide": translated,
            "expected_peptide": peptide,
            "matches": bool(dna and peptide and translated == peptide),
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotated_csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_path = Path(args.annotated_csv)
    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + "_translation_QA.csv"
    )

    df = pd.read_csv(input_path, keep_default_na=False)
    qa = pd.concat(
        [check_chain(df, "H"), check_chain(df, "L")],
        ignore_index=True,
    )
    qa.to_csv(output_path, index=False)

    if qa.empty:
        print("No FL DNA/peptide pairs were found.")
        return

    summary = qa.groupby("chain")["matches"].agg(["count", "sum"])
    summary["match_fraction"] = summary["sum"] / summary["count"]
    print(summary)
    print(f"QA table saved to: {output_path}")


if __name__ == "__main__":
    main()
