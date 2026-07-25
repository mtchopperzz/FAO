"""FAO2: post-processing tools for diversity-aware antibody NGS prioritization.

This package is intentionally designed as a light add-on to the existing FAO
parser.  It does not change the parser's counting logic.  It appends a stable
FL-level seq_uid and builds a candidate prioritization table from annotated.csv,
region-level enrichment/support tables, and optional LLM clustering results.
"""

__version__ = "0.1.0"
