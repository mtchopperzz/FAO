#!/usr/bin/env python3
"""
split_by_prefix.py  --  Split a merged FASTA into per-library files
based on sequence prefixes defined in a 2-column config file.

Header format in the input FASTA (one line):
    >seq_<rank>_count_<int>

Immediately followed by the amino-acid sequence.

Each output file will contain headers re-ranked *within that library*:
    >seq_<rank>_count_<same_total_count>

Unassigned sequences go into  '<outdir>/UNASSIGNED.fasta'.
"""

import argparse
import gzip
import os
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------

def open_maybe_gzip(path, mode="rt"):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)

def load_config(path):
    """Return list of (lib_name, prefix) sorted by decreasing prefix length."""
    libs = []
    with open(path) as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                name, prefix = line.split(None, 1)        # split on 1st whitespace/tab
            except ValueError:
                sys.exit(f"Config parse error line {ln} in {path!r}")
            libs.append((name, prefix.strip()))
    # Longest prefix first so we take the most specific match
    libs.sort(key=lambda x: len(x[1]), reverse=True)
    return libs

def parse_merged_fasta(path, header_re):
    """Yield (count:int, seq:str) from merged FASTA."""
    with open_maybe_gzip(path) as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            seq = fh.readline().rstrip()
            m = header_re.match(header)
            if not m:
                raise ValueError(f"Bad header: {header.strip()}")
            yield int(m.group("count")), seq

# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Split merged FASTA by sequence prefix.")
    p.add_argument("-i", "--infile",   required=True, help="merged FASTA (from previous step)")
    p.add_argument("-c", "--config",   required=True, help="two-column config file (library, prefix)")
    p.add_argument("-o", "--outdir",   default="split_libs", help="output directory (default: split_libs)")
    args = p.parse_args()

    libs = load_config(args.config)
    if not libs:
        sys.exit("Config file has no usable entries.")

    # Add a catch-all entry for unassigned sequences (name, prefix)
    libs_with_unassigned = libs + [("UNASSIGNED", None)]

    # Dict: lib_name -> dict(seq -> count)
    lib2seq_counts = defaultdict(lambda: defaultdict(int))

    header_re = re.compile(r"^>seq_\d+_count_(?P<count>\d+)\s*$")

    # --------------------------------------------------------------------
    for count, seq in parse_merged_fasta(args.infile, header_re):
        assigned = False
        for name, prefix in libs:
            if seq.startswith(prefix):
                lib2seq_counts[name][seq] += count
                assigned = True
                break
        if not assigned:
            lib2seq_counts["UNASSIGNED"][seq] += count

    # --------------------------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)

    for name, _ in libs_with_unassigned:
        seq_counts = lib2seq_counts.get(name)
        if not seq_counts:
            continue   # nothing for this library

        # Sort within the library by descending count
        sorted_items = sorted(seq_counts.items(), key=lambda x: x[1], reverse=True)
        out_path = os.path.join(args.outdir, f"{name}.fasta")
        with open(out_path, "w") as out_f:
            for rank, (seq, cnt) in enumerate(sorted_items, start=1):
                out_f.write(f">seq_{rank}_count_{cnt}\n{seq}\n")

        print(f"Wrote {len(sorted_items):>6} sequences to {out_path}")

    print("Done.")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#python3 ../../code/split_by_prefix.py -i merged.fasta -c ../../code/Library_design.tsv -o split_libs