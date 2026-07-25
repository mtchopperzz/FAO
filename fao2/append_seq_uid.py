#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Append seq_uid to existing FAO *_annotated.csv files.

This is a small convenience wrapper.  It does not change Count/Seq/DNA or any
annotation columns.  It only appends seq_uid as the final column.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fao2.metadata import build_sample_metadata, discover_annotated_files
from fao2.uid import append_seq_uid_column


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parser-out", required=True, help="FAO parser_outputs directory")
    ap.add_argument("--inplace", action="store_true", help="Update existing annotated.csv files in place")
    ap.add_argument("--out", default=None, help="Optional output directory for copied annotated files with UID")
    ap.add_argument("--no-strict-topology", action="store_true")
    args = ap.parse_args()

    files = discover_annotated_files(args.parser_out)
    meta = build_sample_metadata(files)
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
    for _, row in meta.iterrows():
        f = Path(row["annotated_file"])
        df = pd.read_csv(f, keep_default_na=False)
        df = append_seq_uid_column(
            df,
            library_type=row.get("library_type", "unknown"),
            strict_topology=not args.no_strict_topology,
        )
        if args.inplace:
            dest = f
        elif args.out:
            dest = Path(args.out) / f.name
        else:
            dest = f.with_name(f.stem + "__with_uid.csv")
        df.to_csv(dest, index=False)
        print(f"wrote {dest}")


if __name__ == "__main__":
    main()
