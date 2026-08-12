# -*- coding: utf-8 -*-
"""Continuous-frame IgBLAST + ANARCI antibody extractor.

IgBLAST is used for chain assignment, germline calls and approximate N/C
boundaries. Antibody peptide sequences are translated continuously from the
IgBLAST-defined N terminus and annotated with ANARCI/IMGT.
"""

import glob
import multiprocessing
import os
import re
import subprocess
import tempfile
from io import StringIO
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
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

from utils.ProcessHandlers import IgblastExtractor, _nb_dna_to_pep




IMGT_REGION_RANGES = {
    "CDR1": (27, 38),
    "CDR2": (56, 65),
    "CDR3": (105, 117),
    "FR4": (118, 128),
}


class ContinuousIgblastExtractor(IgblastExtractor):
    """Drop-in replacement for ``IgblastExtractor``.

    The output string format is unchanged, so the existing counting,
    chunk-merging and ``format_for_downstream`` methods continue to work.
    """

    _C_TERM_PATTERN = re.compile(
        r"(YYC|Y[CFHNWDSL]C|[HCDFIS]YC|NYC)"
        r"(.{1,30}?)"
        r"(WGQ[GVLE]|WGR[GV]|WGK[GV]|WG[PHLE]G|"
        r"WV(?:EG|Q[GVL])|CG(?:Q[GVL]|RG)|"
        r"[LGRVS]GQG|FG[GQ]G|W[DSA]QG)"
        r"(.*)"
    )

    def __init__(self, config_meta):
        super().__init__(config_meta)
        if not ANARCI_AVAILABLE:
            raise ImportError(
                "ANARCI is required for ContinuousIgblastExtractor but could not be imported."
            )

        available_cpus = max(1, multiprocessing.cpu_count())
        requested_anarci = int(config_meta.get("anarci_ncpu", max(1, available_cpus - 1)))
        requested_igblast = int(config_meta.get("igblast_ncpu", available_cpus))

        self.anarci_ncpu = max(1, min(requested_anarci, available_cpus))
        self.igblast_ncpu = max(1, min(requested_igblast, available_cpus))
        self.anarci_batch_size = max(1, int(config_meta.get("anarci_batch_size", 5000)))
        self.frame_rescue_offsets = tuple(config_meta.get("frame_rescue_offsets", (0, 1, 2)))
        self.anarci_allowed_species = config_meta.get("anarci_allowed_species", None)
        self.anarci_bit_score_threshold = float(
            config_meta.get("anarci_bit_score_threshold", 80)
        )
        self.require_anarci_cdr3 = bool(
            config_meta.get("require_anarci_cdr3", True)
        )
        self.min_anarci_fr4_residues = max(
            0,
            int(config_meta.get("min_anarci_fr4_residues", 1)),
        )
        self.deduplicate_anarci_inputs = bool(
            config_meta.get("deduplicate_anarci_inputs", True)
        )

        self.debug_igblast = bool(config_meta.get("debug_igblast", False))
        default_debug_dir = os.path.join(
            getattr(self.dirs, "logs", "."),
            "igblast_debug",
        )
        self.debug_igblast_dir = str(
            config_meta.get("debug_igblast_dir", default_debug_dir)
        )
        self.debug_write_query_fasta = bool(
            config_meta.get("debug_write_query_fasta", True)
        )
        self.debug_write_airr_tsv = bool(
            config_meta.get("debug_write_airr_tsv", True)
        )
        self.debug_write_hit_table = bool(
            config_meta.get("debug_write_hit_table", True)
        )
        self.debug_max_rows_per_chunk = int(
            config_meta.get("debug_max_rows_per_chunk", 0)
        )

        if self.debug_igblast:
            os.makedirs(self.debug_igblast_dir, exist_ok=True)

    @staticmethod
    def _debug_safe_name(value: Any) -> str:
        text = str(value)
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "sample"

    def _debug_path(self, sample_name: Any, suffix: str) -> str:
        stem = self._debug_safe_name(sample_name)
        return os.path.join(self.debug_igblast_dir, f"{stem}{suffix}")

    def _write_debug_stage_summary(
        self,
        sample_name: Any,
        summary: Dict[str, Any],
    ) -> None:
        if not self.debug_igblast:
            return
        path = os.path.join(self.debug_igblast_dir, "stage_summary.csv")
        row = {"sample_chunk": str(sample_name), **summary}
        frame = pd.DataFrame([row])
        frame.to_csv(
            path,
            mode="a",
            header=not os.path.exists(path),
            index=False,
        )

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        if value is None or pd.isna(value):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_airr_boolean(value: Any) -> bool:
        """Parse AIRR boolean fields such as T/F, True/False, or 1/0."""
        if value is None:
            return False
        try:
            if pd.isna(value):
                return False
        except (TypeError, ValueError):
            pass
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return str(value).strip().lower() in {"t", "true", "1", "yes", "y"}


    @staticmethod
    def _oriented_query_sequence(hit: pd.Series) -> str:
        """
        Return the IgBLAST AIRR ``sequence`` used for coordinate-based translation.

        In the current IgBLAST 1.22.0 outfmt 19 output, ``sequence`` is already
        oriented consistently with ``sequence_alignment`` and the reported
        FR/CDR coordinates. ``rev_comp`` is retained as QC metadata only and is
        not applied to the sequence again.
        """
        sequence = str(hit.get("sequence", "")).upper().replace(" ", "")
        if not sequence or sequence == "NAN":
            return ""
        return sequence

    @staticmethod
    def _translate(dna: str) -> str:
        dna = str(dna).upper().replace(" ", "")
        usable = len(dna) - (len(dna) % 3)
        if usable <= 0:
            return ""
        arr = np.frombuffer(dna[:usable].encode("ascii"), dtype=np.uint8)
        return _nb_dna_to_pep(arr, stop_readthrough=True).tobytes().decode("ascii")

    @staticmethod
    def _expected_chain(hit: pd.Series) -> Optional[str]:
        locus = str(hit.get("locus", "")).upper()
        if locus.startswith("IGH"):
            return "H"
        if locus.startswith("IGK") or locus.startswith("IGL"):
            return "L"
        return None

    @classmethod
    def _rescue_c_terminal_end(cls, peptide: str) -> Optional[int]:
        match = cls._C_TERM_PATTERN.search(peptide)
        if not match:
            return None
        fr4_start = match.start(3)
        fr4_length = min(11, len(match.group(3) + match.group(4)))
        if fr4_length <= 0:
            return None
        return fr4_start + fr4_length

    def _n_terminal_start(self, hit: pd.Series) -> Optional[int]:
        fwr1_start = self._safe_int(hit.get("fwr1_start"))
        if fwr1_start is None:
            return None

        start = max(0, fwr1_start - 1)
        v_germline_start = self._safe_int(hit.get("v_germline_start"))
        if v_germline_start is not None and v_germline_start > 1:
            start = max(0, start - (v_germline_start - 1))
        return start

    def _build_translation_candidate(
        self,
        hit: pd.Series,
        candidate_id: str,
        frame_offset: int,
    ) -> Optional[Dict[str, Any]]:
        """Build an untrimmed protein candidate for ANARCI.

        IgBLAST FR4 coordinates and the empirical C-terminal motif are retained
        as hints only. Neither is required and neither is used to trim the
        candidate before ANARCI. ANARCI determines the final domain boundaries.
        """
        raw_seq = self._oriented_query_sequence(hit)
        if not raw_seq:
            return None

        expected_chain = self._expected_chain(hit)
        n_start = self._n_terminal_start(hit)
        if expected_chain is None or n_start is None:
            return None

        start = n_start + int(frame_offset)
        if start >= len(raw_seq):
            return None

        dna = raw_seq[start:]
        dna = dna[: len(dna) - (len(dna) % 3)]
        peptide = self._translate(dna)

        # A stop after the antibody domain is a natural boundary for the input
        # candidate. A stop before CDR3/FR4 will cause ANARCI/post-ANARCI gates
        # to reject the incomplete domain.
        if "*" in peptide:
            peptide = peptide.split("*", 1)[0]
            dna = dna[: len(peptide) * 3]

        if not peptide or len(peptide) < 70:
            return None

        fwr4_end = self._safe_int(hit.get("fwr4_end"))
        motif_rescue_end = self._rescue_c_terminal_end(peptide)

        if fwr4_end is not None and fwr4_end > start:
            c_end_method = "igblast_fwr4_hint_untrimmed"
        elif motif_rescue_end is not None:
            c_end_method = "sequence_motif_hint_untrimmed"
        else:
            c_end_method = "anarci_untrimmed_no_c_terminal_hint"

        return {
            "candidate_id": candidate_id,
            "orig_idx": int(hit["orig_idx"]),
            "expected_chain": expected_chain,
            "frame_offset": int(frame_offset),
            "peptide": peptide,
            "dna": dna,
            "v_call": str(hit.get("v_call", "")).split(",")[0]
            if pd.notna(hit.get("v_call"))
            else "",
            "j_call": str(hit.get("j_call", "")).split(",")[0]
            if pd.notna(hit.get("j_call"))
            else "",
            "c_end_method": c_end_method,
            "igblast_fwr4_end": fwr4_end,
            "motif_rescue_end": motif_rescue_end,
        }

    @staticmethod
    def _domain_parts(domain_obj: Any) -> Tuple[Optional[List[Any]], Optional[int], Optional[int]]:
        if isinstance(domain_obj, tuple):
            residues = domain_obj[0] if len(domain_obj) > 0 and isinstance(domain_obj[0], list) else None
            start = int(domain_obj[1]) if len(domain_obj) > 1 and domain_obj[1] is not None else None
            end = int(domain_obj[2]) if len(domain_obj) > 2 and domain_obj[2] is not None else None
            return residues, start, end
        if isinstance(domain_obj, list):
            return domain_obj, None, None
        return None, None, None

    @staticmethod
    def _numbered_sequence(residues: Sequence[Any]) -> str:
        chars: List[str] = []
        for entry in residues:
            try:
                aa = str(entry[1])
            except (IndexError, TypeError):
                continue
            if aa != "-":
                chars.append(aa)
        return "".join(chars)

    @staticmethod
    def _numbered_region(residues: Sequence[Any], start: int, end: int) -> str:
        chars: List[str] = []
        for entry in residues:
            try:
                position = int(entry[0][0])
                aa = str(entry[1])
            except (IndexError, TypeError, ValueError):
                continue
            if start <= position <= end and aa != "-":
                chars.append(aa)
        return "".join(chars)

    @staticmethod
    def _numbered_position_bounds(
        residues: Sequence[Any],
    ) -> Tuple[Optional[int], Optional[int]]:
        positions: List[int] = []
        for entry in residues:
            try:
                position = int(entry[0][0])
                aa = str(entry[1])
            except (IndexError, TypeError, ValueError):
                continue
            if aa != "-":
                positions.append(position)
        if not positions:
            return None, None
        return min(positions), max(positions)

    @staticmethod
    def _find_domain_start(peptide: str, numbered_seq: str, start_hint: Optional[int]) -> Optional[int]:
        if not numbered_seq:
            return None
        matches = [m.start() for m in re.finditer(re.escape(numbered_seq), peptide)]
        if not matches:
            return None
        if start_hint is None:
            return matches[0]
        return min(matches, key=lambda x: abs(x - start_hint))

    def _parse_anarci_result(
        self,
        candidate: Dict[str, Any],
        numbering_entry: Any,
        alignment_entry: Any,
    ) -> Optional[Dict[str, Any]]:
        if not numbering_entry or not alignment_entry:
            return None

        best: Optional[Dict[str, Any]] = None

        for domain_index, domain_obj in enumerate(numbering_entry):
            if domain_index >= len(alignment_entry):
                continue
            meta = alignment_entry[domain_index]
            anarci_chain = str(meta.get("chain_type", ""))
            normalized_chain = "L" if anarci_chain in {"K", "L"} else anarci_chain
            if normalized_chain != candidate["expected_chain"]:
                continue

            residues, start_hint, _ = self._domain_parts(domain_obj)
            if not residues:
                continue

            numbered_seq = self._numbered_sequence(residues)
            peptide_start = self._find_domain_start(
                candidate["peptide"], numbered_seq, start_hint
            )
            if peptide_start is None:
                continue

            peptide_end = peptide_start + len(numbered_seq)
            coding_dna = candidate["dna"][peptide_start * 3 : peptide_end * 3]
            if len(coding_dna) != len(numbered_seq) * 3:
                continue
            if self._translate(coding_dna) != numbered_seq:
                continue

            cdr1 = self._numbered_region(residues, *IMGT_REGION_RANGES["CDR1"])
            cdr2 = self._numbered_region(residues, *IMGT_REGION_RANGES["CDR2"])
            cdr3 = self._numbered_region(residues, *IMGT_REGION_RANGES["CDR3"])
            fr4 = self._numbered_region(residues, *IMGT_REGION_RANGES["FR4"])

            if self.require_anarci_cdr3 and not cdr3:
                continue
            if len(fr4) < self.min_anarci_fr4_residues:
                continue

            first_imgt, last_imgt = self._numbered_position_bounds(residues)
            bitscore = float(meta.get("bitscore", 0.0) or 0.0)
            completeness = len(numbered_seq)
            parsed = {
                **candidate,
                "FL_PEP": numbered_seq,
                "FL_DNA": coding_dna,
                "CDR3_PEP": cdr3,
                "CDRs_PEP": cdr1 + cdr2 + cdr3,
                "FR4_PEP": fr4,
                "anarci_bitscore": bitscore,
                "anarci_completeness": completeness,
                "anarci_first_imgt_position": first_imgt,
                "anarci_last_imgt_position": last_imgt,
                "anarci_fr4_length": len(fr4),
                "anarci_fr4_complete": len(fr4) == 11,
            }

            if best is None or (
                parsed["anarci_bitscore"],
                parsed["anarci_fr4_length"],
                parsed["anarci_completeness"],
                -parsed["frame_offset"],
            ) > (
                best["anarci_bitscore"],
                best["anarci_fr4_length"],
                best["anarci_completeness"],
                -best["frame_offset"],
            ):
                best = parsed

        return best

    def _run_anarci(
        self,
        candidates: Sequence[Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
        """Run ANARCI once per unique ``(expected_chain, peptide)`` pair.

        ANARCI/HMMER work is shared across duplicate read candidates. The raw
        numbering result is then parsed against every original candidate so
        that read index, DNA, frame, germline calls and downstream pairing are
        preserved.
        """
        results: Dict[str, Dict[str, Any]] = {}
        stats = {
            "input_candidates": len(candidates),
            "unique_inputs": 0,
            "deduplicated_candidates": 0,
        }
        if not candidates:
            return results, stats

        grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
        for candidate in candidates:
            expected_chain = str(candidate.get("expected_chain", ""))
            peptide = str(candidate.get("peptide", ""))

            if self.deduplicate_anarci_inputs:
                key: Tuple[str, ...] = (expected_chain, peptide)
            else:
                key = (
                    str(candidate.get("candidate_id", "")),
                    expected_chain,
                    peptide,
                )

            grouped.setdefault(key, []).append(candidate)

        unique_items = list(grouped.items())
        stats["unique_inputs"] = len(unique_items)
        stats["deduplicated_candidates"] = len(candidates) - len(unique_items)

        for batch_start in range(0, len(unique_items), self.anarci_batch_size):
            batch_items = unique_items[
                batch_start : batch_start + self.anarci_batch_size
            ]

            sequences: List[Tuple[str, str]] = []
            unique_id_to_candidates: Dict[str, List[Dict[str, Any]]] = {}

            for local_index, (key, candidate_group) in enumerate(batch_items):
                unique_id = f"anarci_unique_{batch_start + local_index}"
                peptide = key[-1]
                sequences.append((unique_id, peptide))
                unique_id_to_candidates[unique_id] = candidate_group

            if run_anarci is not None:
                _, numbering, alignment_details, _ = run_anarci(
                    sequences,
                    ncpu=min(self.anarci_ncpu, len(sequences)),
                    scheme="imgt",
                    output=False,
                    assign_germline=False,
                    allowed_species=self.anarci_allowed_species,
                    allow={"H", "K", "L"},
                    bit_score_threshold=self.anarci_bit_score_threshold,
                )
            else:
                numbering, alignment_details, _ = anarci(
                    sequences,
                    scheme="imgt",
                    output=False,
                    ncpu=min(self.anarci_ncpu, len(sequences)),
                    assign_germline=False,
                    allowed_species=self.anarci_allowed_species,
                    allow={"H", "K", "L"},
                    bit_score_threshold=self.anarci_bit_score_threshold,
                )

            for index, (unique_id, _) in enumerate(sequences):
                numbering_entry = (
                    numbering[index] if numbering is not None else None
                )
                alignment_entry = (
                    alignment_details[index]
                    if alignment_details is not None
                    else None
                )

                for candidate in unique_id_to_candidates[unique_id]:
                    parsed = self._parse_anarci_result(
                        candidate,
                        numbering_entry,
                        alignment_entry,
                    )
                    if parsed is not None:
                        results[candidate["candidate_id"]] = parsed

        return results, stats

    @staticmethod
    def _best_by_read_and_chain(
        annotations: Iterable[Dict[str, Any]],
    ) -> Dict[Tuple[int, str], Dict[str, Any]]:
        selected: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for item in annotations:
            key = (item["orig_idx"], item["expected_chain"])
            old = selected.get(key)
            if old is None or (
                item["anarci_bitscore"],
                item.get("anarci_fr4_length", 0),
                item["anarci_completeness"],
                -item["frame_offset"],
            ) > (
                old["anarci_bitscore"],
                old.get("anarci_fr4_length", 0),
                old["anarci_completeness"],
                -old["frame_offset"],
            ):
                selected[key] = item
        return selected

    def run_igblast_and_translate(self):
        def _op(data):
            for sample in data:
                new_D, new_P, new_Q = [], [], []
                fasta_entries: List[str] = []
                n_input_reads = len(sample.D)
                sample_name = getattr(sample, "name", "sample")

                for i, row_data in enumerate(sample.D):
                    row_bytes = (
                        row_data.tobytes().strip(b"\x00")
                        if row_data.dtype.kind == "S"
                        else "".join(row_data).encode("ascii")
                    )
                    p1, p2 = self._split_linker(row_bytes)
                    fasta_entries.append(f">seq{i}_P1\n{p1.decode('ascii')}")
                    if p2 and len(p2) > 50:
                        fasta_entries.append(f">seq{i}_P2\n{p2.decode('ascii')}")

                if not fasta_entries:
                    continue

                fasta_text = "\n".join(fasta_entries)
                if self.debug_igblast and self.debug_write_query_fasta:
                    with open(
                        self._debug_path(sample_name, "__igblast_query.fasta"),
                        "w",
                        encoding="utf-8",
                    ) as handle:
                        handle.write(fasta_text)

                with tempfile.NamedTemporaryFile(
                    mode="w", dir=self.temp_dir, suffix=".fasta", delete=False
                ) as tmp_fasta:
                    tmp_fasta.write(fasta_text)
                    fasta_path = tmp_fasta.name

                env = os.environ.copy()
                env["IGDATA"] = self.igdata_path
                cmd = [
                    self.igblast_exec,
                    "-query",
                    fasta_path,
                    "-organism",
                    self.species,
                    "-ig_seqtype",
                    "Ig",
                    "-domain_system",
                    "imgt",
                    "-outfmt",
                    "19",
                    "-num_threads",
                    str(self.igblast_ncpu),
                ]

                aux_file = os.path.join(
                    self.igdata_path, "optional_file", f"{self.species}_gl.aux"
                )
                if os.path.exists(aux_file):
                    cmd.extend(["-auxiliary_data", aux_file])
                else:
                    self.logger.warning(
                        f"No auxiliary data found for {self.species}. CDR3 end boundaries may be incomplete."
                    )

                for locus in ("V", "D", "J"):
                    cmd.extend(
                        [
                            f"-germline_db_{locus}",
                            os.path.join(
                                self.db_dir, f"{self.species}_{locus}_igblast.fasta"
                            ),
                        ]
                    )

                result = subprocess.run(cmd, env=env, capture_output=True, text=True)

                if self.debug_igblast and self.debug_write_airr_tsv:
                    with open(
                        self._debug_path(sample_name, "__igblast_airr.tsv"),
                        "w",
                        encoding="utf-8",
                    ) as handle:
                        handle.write(result.stdout)

                    if result.stderr:
                        with open(
                            self._debug_path(sample_name, "__igblast_stderr.txt"),
                            "w",
                            encoding="utf-8",
                        ) as handle:
                            handle.write(result.stderr)

                    with open(
                        self._debug_path(sample_name, "__igblast_command.txt"),
                        "w",
                        encoding="utf-8",
                    ) as handle:
                        handle.write(" ".join(cmd))

                os.remove(fasta_path)

                if result.returncode != 0:
                    self.logger.error(f"IgBLAST failed: {result.stderr.strip()}")
                    continue

                try:
                    hits_df = pd.read_csv(StringIO(result.stdout), sep="\t")
                    hits_df["orig_idx"] = hits_df["sequence_id"].apply(
                        lambda x: int(str(x).split("_")[0].replace("seq", ""))
                    )
                    if self.debug_igblast:
                        hits_df["hit_key"] = [
                            f"hit{idx}" for idx in hits_df.index
                        ]
                        hits_df["rev_comp_parsed"] = (
                            hits_df["rev_comp"].apply(self._parse_airr_boolean)
                            if "rev_comp" in hits_df.columns
                            else False
                        )
                        hits_df["oriented_sequence"] = hits_df.apply(
                            self._oriented_query_sequence,
                            axis=1,
                        )
                        hits_df["n_terminal_start"] = hits_df.apply(
                            self._n_terminal_start,
                            axis=1,
                        )
                        hits_df["expected_chain"] = hits_df.apply(
                            self._expected_chain,
                            axis=1,
                        )
                except Exception as exc:
                    self.logger.error(f"Parsing AIRR failed: {exc}")
                    continue

                primary_candidates: List[Dict[str, Any]] = []
                hit_rows: Dict[str, pd.Series] = {}
                frame0_built: Dict[str, bool] = {} if self.debug_igblast else {}

                for hit_index, hit in hits_df.iterrows():
                    hit_key = f"hit{hit_index}"
                    hit_rows[hit_key] = hit
                    candidate = self._build_translation_candidate(
                        hit, f"{hit_key}_f0", frame_offset=0
                    )
                    if self.debug_igblast:
                        frame0_built[hit_key] = candidate is not None
                    if candidate is not None:
                        primary_candidates.append(candidate)

                annotations, primary_anarci_stats = self._run_anarci(
                    primary_candidates
                )
                n_primary_anarci_success = len(annotations)
                successful_hits = {
                    candidate_id.rsplit("_f", 1)[0] for candidate_id in annotations
                }

                rescue_candidates: List[Dict[str, Any]] = []
                rescue_offsets_built: Dict[str, List[int]] = (
                    {} if self.debug_igblast else {}
                )
                for hit_key, hit in hit_rows.items():
                    if hit_key in successful_hits:
                        continue
                    for offset in self.frame_rescue_offsets:
                        if int(offset) == 0:
                            continue
                        candidate = self._build_translation_candidate(
                            hit, f"{hit_key}_f{int(offset)}", frame_offset=int(offset)
                        )
                        if candidate is not None:
                            rescue_candidates.append(candidate)
                            if self.debug_igblast:
                                rescue_offsets_built.setdefault(
                                    hit_key, []
                                ).append(int(offset))

                rescue_annotations, rescue_anarci_stats = self._run_anarci(
                    rescue_candidates
                )
                annotations.update(rescue_annotations)
                selected = self._best_by_read_and_chain(annotations.values())

                if self.debug_igblast and self.debug_write_hit_table:
                    primary_success_hits = {
                        candidate_id.rsplit("_f", 1)[0]
                        for candidate_id in annotations
                        if candidate_id.endswith("_f0")
                    }
                    rescue_success_hits = {
                        candidate_id.rsplit("_f", 1)[0]
                        for candidate_id in rescue_annotations
                    }
                    selected_candidate_ids = {
                        item["candidate_id"] for item in selected.values()
                    }
                    selected_hits = {
                        candidate_id.rsplit("_f", 1)[0]
                        for candidate_id in selected_candidate_ids
                    }
                    selected_by_hit = {
                        item["candidate_id"].rsplit("_f", 1)[0]: item
                        for item in selected.values()
                    }

                    debug_hits = hits_df.copy()
                    debug_hits["frame0_candidate_built"] = debug_hits[
                        "hit_key"
                    ].map(frame0_built).fillna(False)
                    debug_hits["frame0_anarci_success"] = debug_hits[
                        "hit_key"
                    ].isin(primary_success_hits)
                    debug_hits["rescue_offsets_built"] = debug_hits[
                        "hit_key"
                    ].map(
                        lambda key: ",".join(
                            str(x) for x in rescue_offsets_built.get(key, [])
                        )
                    )
                    debug_hits["rescue_anarci_success"] = debug_hits[
                        "hit_key"
                    ].isin(rescue_success_hits)
                    debug_hits["selected_for_output"] = debug_hits[
                        "hit_key"
                    ].isin(selected_hits)
                    debug_hits["selected_frame_offset"] = debug_hits[
                        "hit_key"
                    ].map(
                        lambda key: selected_by_hit.get(key, {}).get(
                            "frame_offset", ""
                        )
                    )
                    debug_hits["selected_c_end_method"] = debug_hits[
                        "hit_key"
                    ].map(
                        lambda key: selected_by_hit.get(key, {}).get(
                            "c_end_method", ""
                        )
                    )
                    debug_hits["selected_anarci_bitscore"] = debug_hits[
                        "hit_key"
                    ].map(
                        lambda key: selected_by_hit.get(key, {}).get(
                            "anarci_bitscore", ""
                        )
                    )
                    debug_hits["selected_anarci_last_imgt_position"] = debug_hits[
                        "hit_key"
                    ].map(
                        lambda key: selected_by_hit.get(key, {}).get(
                            "anarci_last_imgt_position", ""
                        )
                    )
                    debug_hits["selected_anarci_fr4_length"] = debug_hits[
                        "hit_key"
                    ].map(
                        lambda key: selected_by_hit.get(key, {}).get(
                            "anarci_fr4_length", ""
                        )
                    )
                    debug_hits["selected_anarci_fr4_complete"] = debug_hits[
                        "hit_key"
                    ].map(
                        lambda key: selected_by_hit.get(key, {}).get(
                            "anarci_fr4_complete", ""
                        )
                    )

                    preferred_columns = [
                        "hit_key",
                        "sequence_id",
                        "orig_idx",
                        "sequence",
                        "rev_comp",
                        "rev_comp_parsed",
                        "oriented_sequence",
                        "locus",
                        "expected_chain",
                        "productive",
                        "vj_in_frame",
                        "stop_codon",
                        "v_call",
                        "d_call",
                        "j_call",
                        "fwr1_start",
                        "v_germline_start",
                        "fwr4_start",
                        "fwr4_end",
                        "sequence_aa",
                        "sequence_alignment",
                        "sequence_alignment_aa",
                        "n_terminal_start",
                        "frame0_candidate_built",
                        "frame0_anarci_success",
                        "rescue_offsets_built",
                        "rescue_anarci_success",
                        "selected_for_output",
                        "selected_frame_offset",
                        "selected_c_end_method",
                        "selected_anarci_bitscore",
                        "selected_anarci_last_imgt_position",
                        "selected_anarci_fr4_length",
                        "selected_anarci_fr4_complete",
                    ]
                    preferred_columns = [
                        column
                        for column in preferred_columns
                        if column in debug_hits.columns
                    ]
                    remaining_columns = [
                        column
                        for column in debug_hits.columns
                        if column not in preferred_columns
                    ]
                    debug_hits = debug_hits[
                        preferred_columns + remaining_columns
                    ]
                    if self.debug_max_rows_per_chunk > 0:
                        debug_hits = debug_hits.head(
                            self.debug_max_rows_per_chunk
                        )
                    debug_hits.to_csv(
                        self._debug_path(
                            sample_name,
                            "__igblast_hits_debug.csv",
                        ),
                        index=False,
                    )

                for i in range(len(sample.D)):
                    h_ann = selected.get((i, "H"))
                    l_ann = selected.get((i, "L"))
                    if h_ann is None and l_ann is None:
                        continue

                    h_data: Dict[str, str] = {}
                    l_data: Dict[str, str] = {}

                    if h_ann is not None:
                        h_data = {
                            "H_CDR3_PEP": h_ann["CDR3_PEP"],
                            "H_CDRs_PEP": h_ann["CDRs_PEP"],
                            "H_FL_PEP": h_ann["FL_PEP"],
                            "H_FL_DNA": h_ann["FL_DNA"],
                            "V_Gene": h_ann["v_call"],
                            "J_Gene": h_ann["j_call"],
                            "Species": self.species,
                        }

                    if l_ann is not None:
                        l_data = {
                            "L_CDR3_PEP": l_ann["CDR3_PEP"],
                            "L_CDRs_PEP": l_ann["CDRs_PEP"],
                            "L_FL_PEP": l_ann["FL_PEP"],
                            "L_FL_DNA": l_ann["FL_DNA"],
                            "V_Gene": l_ann["v_call"],
                            "J_Gene": l_ann["j_call"],
                            "Species": self.species,
                        }

                    pep_parts: List[str] = []
                    for reg in self.extract_regions:
                        hp = h_data.get(f"H_{reg}_PEP", "")
                        lp = l_data.get(f"L_{reg}_PEP", "")
                        pep_parts.append(f"H_{reg}:{hp}|L_{reg}:{lp}")

                    # Count identity is peptide-only. IgBLAST germline/species
                    # annotations are stored with the representative DNA payload
                    # and therefore do not split otherwise identical sequences.
                    annotation_parts: List[str] = []
                    if h_ann is not None:
                        annotation_parts.append(
                            f"H_V_Gene:{h_data['V_Gene']}|"
                            f"H_J_Gene:{h_data['J_Gene']}|"
                            f"H_Species:{h_data['Species']}"
                        )
                    if l_ann is not None:
                        annotation_parts.append(
                            f"L_V_Gene:{l_data['V_Gene']}|"
                            f"L_J_Gene:{l_data['J_Gene']}|"
                            f"L_Species:{l_data['Species']}"
                        )

                    full_pep_str = "||".join(pep_parts)

                    hd = h_data.get("H_FL_DNA", "")
                    ld = l_data.get("L_FL_DNA", "")
                    companion_parts = [f"H_FL:{hd}|L_FL:{ld}"]
                    companion_parts.extend(annotation_parts)
                    dna_str = "||".join(companion_parts)

                    new_P.append(full_pep_str)
                    new_D.append(dna_str)
                    if sample.Q is not None:
                        new_Q.append(sample.Q[i])

                if self.debug_igblast:
                    self._write_debug_stage_summary(
                        sample_name,
                        {
                            "input_reads": n_input_reads,
                            "igblast_query_fragments": len(fasta_entries),
                            "igblast_hit_rows": len(hits_df),
                            "igblast_rev_comp_hits": int(
                                hits_df["rev_comp_parsed"].sum()
                            ),
                            "igblast_hits_with_chain": int(
                                hits_df["expected_chain"].notna().sum()
                            ),
                            "igblast_hits_with_fwr1": int(
                                pd.to_numeric(
                                    hits_df.get("fwr1_start"),
                                    errors="coerce",
                                ).notna().sum()
                            ),
                            "igblast_hits_with_fwr4": int(
                                pd.to_numeric(
                                    hits_df.get("fwr4_end"),
                                    errors="coerce",
                                ).notna().sum()
                            ),
                            "frame0_candidates_built": len(primary_candidates),
                            "frame0_anarci_unique_inputs": primary_anarci_stats[
                                "unique_inputs"
                            ],
                            "frame0_anarci_deduplicated_candidates": (
                                primary_anarci_stats[
                                    "deduplicated_candidates"
                                ]
                            ),
                            "frame0_anarci_success": n_primary_anarci_success,
                            "rescue_candidates_built": len(rescue_candidates),
                            "rescue_anarci_unique_inputs": rescue_anarci_stats[
                                "unique_inputs"
                            ],
                            "rescue_anarci_deduplicated_candidates": (
                                rescue_anarci_stats[
                                    "deduplicated_candidates"
                                ]
                            ),
                            "rescue_anarci_success": len(rescue_annotations),
                            "selected_chains": len(selected),
                            "selected_reads": len(
                                {item["orig_idx"] for item in selected.values()}
                            ),
                            "selected_igblast_fwr4_hint": sum(
                                item.get("c_end_method")
                                == "igblast_fwr4_hint_untrimmed"
                                for item in selected.values()
                            ),
                            "selected_sequence_motif_hint": sum(
                                item.get("c_end_method")
                                == "sequence_motif_hint_untrimmed"
                                for item in selected.values()
                            ),
                            "selected_no_c_terminal_hint": sum(
                                item.get("c_end_method")
                                == "anarci_untrimmed_no_c_terminal_hint"
                                for item in selected.values()
                            ),
                            "selected_complete_fr4": sum(
                                bool(item.get("anarci_fr4_complete", False))
                                for item in selected.values()
                            ),
                            "output_rows": len(new_P),
                        },
                    )

                sample.D = np.array(new_D, dtype=object)
                sample.P = np.array(new_P, dtype=object)
                if sample.Q is not None:
                    sample.Q = np.array(new_Q, dtype=object)
                sample.transform()

            return data

        _op.__name__ = "run_igblast_continuous_anarci"
        return _op

    def format_for_downstream(self):
        """Write the established annotated.csv schema.

        New parser outputs keep peptide regions in ``Seq`` and keep one
        representative DNA plus IgBLAST V/J/species annotations in the
        companion ``DNA`` payload. Older outputs with metadata embedded in
        ``Seq`` remain readable.
        """
        search_pattern = os.path.join(
            self.dirs.parser_out,
            "*",
            "*_pep_counts.csv",
        )
        merged_files = glob.glob(search_pattern)

        exact_columns = [
            "ID",
            "H_CDR3_PEP",
            "H_CDRs_PEP",
            "H_FL_PEP",
            "L_CDRs_PEP",
            "L_FL_PEP",
            "L_CDR3_PEP",
            "H_V_Gene",
            "H_J_Gene",
            "H_Species",
            "L_V_Gene",
            "L_Species",
            "L_J_Gene",
            "Count",
            "H_FL_DNA",
            "L_FL_DNA",
        ]

        for file_path in merged_files:
            try:
                df = pd.read_csv(file_path, keep_default_na=False)
            except Exception as exc:
                self.logger.warning(
                    f"Could not read merged count file {file_path}: {exc}"
                )
                continue

            if "Seq" not in df.columns:
                continue

            records: List[Dict[str, Any]] = []

            for _, row in df.iterrows():
                record: Dict[str, Any] = {
                    "Count": row.get("Count", 0),
                }

                # Peptide regions. This also keeps backward compatibility with
                # older files where IgBLAST metadata was embedded in Seq.
                for packed_region in str(row.get("Seq", "")).split("||"):
                    for chain_part in packed_region.split("|"):
                        if ":" not in chain_part:
                            continue
                        key, value = chain_part.split(":", 1)
                        if key.endswith("_Gene") or key.endswith("_Species"):
                            record[key] = value
                        else:
                            record[f"{key}_PEP"] = value

                # Representative DNA and current-format IgBLAST annotations.
                for packed_item in str(row.get("DNA", "")).split("||"):
                    for chain_part in packed_item.split("|"):
                        if ":" not in chain_part:
                            continue
                        key, value = chain_part.split(":", 1)
                        if key.endswith("_Gene") or key.endswith("_Species"):
                            record[key] = value
                        else:
                            record[f"{key}_DNA"] = value

                records.append(record)

            new_df = pd.DataFrame(records)
            base = os.path.basename(file_path).replace(
                "_pep_counts.csv",
                "",
            )
            new_df.insert(
                0,
                "ID",
                [
                    f"{base}_{index:06d}"
                    for index in range(1, len(new_df) + 1)
                ],
            )

            for column in exact_columns:
                if column not in new_df.columns:
                    new_df[column] = ""

            new_df = new_df[exact_columns]
            output_path = file_path.replace(
                "_pep_counts.csv",
                "_pep_counts_annotated.csv",
            )
            new_df.to_csv(output_path, index=False)
            self.logger.info(
                f"Formatted for downstream analysis: {output_path}"
            )
