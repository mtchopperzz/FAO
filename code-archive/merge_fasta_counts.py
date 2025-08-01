#!/usr/bin/env python3
"""
merge_fasta_counts.py  --  Merge FASTA-like files that store counts in headers.

Input  header format (one line):
    >seq_<anything>_count_<int>

Immediately followed by a single line containing the amino-acid sequence.

Output file (specified with -o / --out) is rewritten so that:
    • Each unique sequence appears once.
    • Its count is the sum of counts in all input files.
    • Sequences are re-ranked by descending count.
      Header pattern:  >seq_<rank>_count_<total_count>
"""

import argparse
import gzip
import re
from collections import defaultdict

# ---------------------------------------------------------------------------

def open_maybe_gzip(path, mode="rt"):
    """Transparent handler for .gz or plain text files."""
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)

def parse_file(path, seq2count, header_re):
    with open_maybe_gzip(path) as fh:
        while True:
            header = fh.readline()
            if not header:
                break  # EOF
            seq    = fh.readline().rstrip()
            m = header_re.match(header)
            if not m:
                raise ValueError(f"Header not recognized in {path!r}: {header.strip()}")
            count  = int(m.group("count"))
            seq2count[seq] += count

def main():
    p = argparse.ArgumentParser(description="Merge FASTA-like files with counts in headers.")
    p.add_argument("infiles", nargs="+", help="20 input FASTA-like files (.gz allowed)")
    p.add_argument("-o", "--out", required=True, help="merged output file")
    args = p.parse_args()

    # Dict that will accumulate total counts per unique sequence
    seq2count = defaultdict(int)

    # Header parser  (tolerates anything between 'seq_' and '_count_')
    header_re = re.compile(r"^>seq_[^_]*_count_(?P<count>\d+)\s*$")

    # ------------------------------------------------------------
    for f in args.infiles:
        parse_file(f, seq2count, header_re)

    # ------------------------------------------------------------
    # Re-rank by descending count (ties keep arbitrary order)
    sorted_items = sorted(seq2count.items(), key=lambda x: x[1], reverse=True)

    with open(args.out, "w") as out_f:
        for rank, (seq, total_count) in enumerate(sorted_items, start=1):
            out_f.write(f">seq_{rank}_count_{total_count}\n{seq}\n")

    print(f"Merged {len(args.infiles)} files; wrote {len(sorted_items)} unique sequences to {args.out}")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# python3 python ../../code/merge_fasta_couts.py *.fasta -o merged.fasta