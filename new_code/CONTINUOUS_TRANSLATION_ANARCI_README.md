# Continuous translation + parallel ANARCI demo

## Files

- `ContinuousIgblastExtractor.py`: place under `utils/`.
- `analysis_clean_continuous.py`: run this instead of the existing `analysis_clean.py` after updating paths.
- `validate_continuous_output.py`: optional QA check for DNA/peptide agreement.

## What changes

The existing FASTQ filters, linker splitting, counting, chunk merging, downstream CSV format, convergence analysis and enrichment analysis are retained.

For each IgBLAST chain hit, the new extractor:

1. Uses IgBLAST for chain type, V/J calls, FR1 start and approximate FR4 end.
2. Translates DNA continuously from the inferred FR1 start.
3. Uses frame offsets `0, +1, +2` only when the primary frame cannot be numbered.
4. Uses IgBLAST FR4 end when available; otherwise uses the existing CDR3/FR4 sequence motif rescue.
5. Runs ANARCI with IMGT numbering to obtain FL, CDR1, CDR2 and CDR3.
6. Crops DNA to the exact ANARCI-numbered peptide domain.

The resulting fields satisfy:

```text
translate(H_FL_DNA) == H_FL_PEP
translate(L_FL_DNA) == L_FL_PEP
```

for successfully emitted chains.

## CPU configuration

`analysis_clean_continuous.py` exposes:

```python
igblast_ncpu = multiprocessing.cpu_count()
anarci_ncpu = max(1, multiprocessing.cpu_count() - 1)
anarci_batch_size = 5000
```

The extractor uses ANARCI's `run_anarci(..., ncpu=N)` wrapper when available. This distributes sequence batches over multiple worker processes. If an older ANARCI installation does not expose `run_anarci`, it falls back to `anarci(..., ncpu=N)`.

## First test

Run on a small FASTQ first. Then validate an annotated CSV:

```bash
python validate_continuous_output.py \
  /path/to/sample_pep_counts_annotated.csv
```

The expected match fraction for emitted H and L chains is `1.0`.
