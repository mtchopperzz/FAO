# -*- coding: utf-8 -*-
"""
ProcessHandlers.py
==================

Core processing logic for the sequence analysis pipeline.
This module handles:
1. Low-level Sequence Manipulation: High-performance DNA translation and pattern matching using Numba.
2. Pipeline Orchestration: Managing streaming data, logging, and directory structures.
3. FASTQ Processing: Filtering, translation, and fuzzy region extraction.
4. Annotation: Parallelized antibody numbering using ANARCI.
5. Enrichment Analysis: Calculating fold changes and scoring candidates across multiple rounds.

Dependencies:
- ANARCI (for antibody numbering)
- Numba (for JIT optimization of tight loops)
- Pandas/Numpy (for data management)
"""

#  --- IMPORTS --- 
import time, os, logging, gzip, re, inspect, copy
import multiprocessing 
import matplotlib.pyplot as plt
from fao2 import Plotter
from fao2.datatypes import Data, SequencingSample
from fao2.constants import constants 
import glob

import numpy as np
import pandas as pd
import numba as nb
from collections import Counter
import shutil

import subprocess
import tempfile
from io import StringIO

# Check for ANARCI availability (Critical for antibody numbering)
try:
    from anarci import anarci
    ANARCI_AVAILABLE = True
except ImportError:
    ANARCI_AVAILABLE = False

# --- CONSTANTS FOR IMGT NUMBERING ---
# Defines the antibody regions according to IMGT scheme.
IMGT_DEFINITIONS = {
    "FR1":  (1, 26),
    "CDR1": (27, 38),
    "FR2":  (39, 55),
    "CDR2": (56, 65),
    "FR3":  (66, 104),
    "CDR3": (105, 117),
    "FR4":  (118, 128),
}

# --- NUMBA LOOKUP TABLES & HELPERS ---
# Numba JIT compilation
# 1. DNA Mapping Table: Maps ASCII characters A, C, G, T to integers 0, 1, 2, 3.
_DNA_TO_INT = np.full(128, 4, dtype=np.int8) 
_DNA_TO_INT[ord('A')] = 0
_DNA_TO_INT[ord('C')] = 1
_DNA_TO_INT[ord('G')] = 2
_DNA_TO_INT[ord('T')] = 3

# 2. Complement Table: Maps bases to their complement (A->T, C->G, etc.)
_COMP_LUT = np.full(128, 0, dtype=np.uint8)
for a, b in zip(b"ACGTN", b"TGCAN"):
    _COMP_LUT[a] = b

# 3. Codon Table: 3D array [4][4][4] of translation codon to look up amino acids from integer triplet inputs.
_CODON_LUT = np.full((4, 4, 4), ord('X'), dtype=np.uint8) 

def _fill_codon_table_from_constants():
    """Populates the Numba-optimized codon lookup table from the constants file."""
    table = constants.codon_table
    for codon, aa in table.items():
        i1 = _DNA_TO_INT[ord(codon[0])]
        i2 = _DNA_TO_INT[ord(codon[1])]
        i3 = _DNA_TO_INT[ord(codon[2])]
        _CODON_LUT[i1, i2, i3] = ord(aa)

_fill_codon_table_from_constants()

@nb.njit(fastmath=True)
def _nb_dna_to_pep(dna_bytes, stop_readthrough):
    """
    High-performance DNA to Protein translation.
    
    Args:
        dna_bytes (uint8 array): ASCII bytes of DNA sequence.
        stop_readthrough (bool): If True, continues translating past stop codons (*).
                                 If False, truncates sequence at the first stop.
    Returns:
        uint8 array: Translated protein sequence (ASCII bytes).
    """
    n = len(dna_bytes)
    n_codons = n // 3
    out = np.empty(n_codons, dtype=np.uint8)
    
    out_idx = 0
    for i in range(n_codons):
        base_idx = i * 3
        b1 = dna_bytes[base_idx]
        b2 = dna_bytes[base_idx+1]
        b3 = dna_bytes[base_idx+2]
        
        # Verify valid bases (ACGT)
        if b1 < 128 and b2 < 128 and b3 < 128:
            i1 = _DNA_TO_INT[b1]
            i2 = _DNA_TO_INT[b2]
            i3 = _DNA_TO_INT[b3]
            if i1 > 3 or i2 > 3 or i3 > 3:
                aa = 43 # '+' represents ambiguous codon
            else:
                aa = _CODON_LUT[i1, i2, i3]
        else:
            aa = 43 

        # Handle Stop Codons
        if aa == 42: # '*' is 42 in ASCII
            if stop_readthrough:
                out[out_idx] = aa
                out_idx += 1
            else:
                return out[:out_idx]
        else:
            out[out_idx] = aa
            out_idx += 1
            
    return out[:out_idx]

@nb.njit(fastmath=True)
def _nb_revcom(seq):
    """Computes Reverse Complement of a DNA byte array."""
    n = len(seq)
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        val = seq[n - 1 - i]
        if val < 128:
            out[i] = _COMP_LUT[val]
        else:
            out[i] = val 
    return out

@nb.njit(fastmath=True)
def _match_block_fuzzy(seq, motif, tol):
    """
    Finds the first occurrence of 'motif' in 'seq' allowing for 'tol' mismatches.
    Returns: Index of match start, or -1 if not found.
    """
    n = len(seq)
    m = len(motif)
    if m > n: return -1
    
    for i in range(n - m + 1):
        err = 0
        match = True
        for j in range(m):
            if seq[i + j] != motif[j]:
                err += 1
                if err > tol:
                    match = False
                    break
        if match: return i
    return -1

@nb.njit(parallel=True)
def _nb_q_filter(Q, min_q, frac):
    """
    Filters sequences based on Phred Quality Scores (Q-scores).
    Logic: Keep sequence if (frac)% of bases have Q-score >= min_q.
    """
    rows, cols = Q.shape
    keep = np.zeros(rows, dtype=np.bool_)
    for i in nb.prange(rows):
        total = 0
        passed = 0
        for j in range(cols):
            val = Q[i, j]
            if val != 0: 
                total += 1
                if val >= min_q: passed += 1
        if total == 0: keep[i] = False
        else:
            if (passed / total) >= frac: keep[i] = True
    return keep

# --- WORKER FUNCTION FOR MULTIPROCESSING ---
def _anarci_worker(args):
    """
    Independent worker function for ANARCI parallel processing.
    
    Args:
        args (tuple): Contains chunk of sequences and configuration parameters.
    
    Returns:
        list: A list of dictionaries containing annotated domains (CDRs, FRs, Germlines).
    """
    sequences, extract_regions, allowed_species, assign_germline, output_mode = args
    
    # Run ANARCI on this chunk (single core mode per chunk to avoid thrashing)
    kwargs = {'scheme': 'imgt', 'output': False, 'ncpu': 1, 'assign_germline': assign_germline}
    if allowed_species:
        kwargs['allowed_species'] = allowed_species
        
    try:
        numbering, alignment_details, hit_tables = anarci(sequences, **kwargs)
    except Exception as e:
        return []

    chunk_results = []
    
    # Iterate through results
    for i, (num, align, hits) in enumerate(zip(numbering, alignment_details, hit_tables)):
        if not num: continue
        
        original_idx = int(sequences[i][0])
        
        row_data = {'_Original_Index': original_idx}
        has_valid_domain = False
        
        # Helper to extract specific region sequence from the numbered list
        def get_region_seq(residue_list, region_def, query_start_idx):
            valid_query_indices = []
            pep_res = []
            current_query_pos = query_start_idx
            
            for entry in residue_list:
                try: imgt_idx = int(entry[0][0])
                except: continue
                
                residue = entry[1]
                is_gap = (residue == '-')
                
                # Track position in original query string (for DNA extraction later)
                this_pos_in_query = -1
                if not is_gap:
                    this_pos_in_query = current_query_pos
                    current_query_pos += 1
                    
                r_start, r_end = region_def
                if r_start <= imgt_idx <= r_end:
                    if not is_gap:
                        pep_res.append(residue)
                        valid_query_indices.append(this_pos_in_query)
                    elif output_mode == 'aligned':
                        pep_res.append(residue)
                        
            return "".join(pep_res), valid_query_indices

        if not align: continue

        # Extract domains
        for domain_idx, domain_meta in enumerate(align):
            chain_type = domain_meta.get('chain_type', 'X')
            if chain_type == 'K': chain_type = 'L' # Normalize Kappa to Light
            domain_start_idx = domain_meta['query_start']
            
            # Germline Extraction
            if assign_germline and 'germlines' in domain_meta:
                germs = domain_meta['germlines']
                if 'v_gene' in germs and germs['v_gene']:
                    row_data[f"{chain_type}_V_Gene"] = germs['v_gene'][0][1]
                if 'j_gene' in germs and germs['j_gene']:
                    row_data[f"{chain_type}_J_Gene"] = germs['j_gene'][0][1]
                if 'species' in domain_meta:
                    row_data[f"{chain_type}_Species"] = domain_meta['species']

            if domain_idx < len(num):
                domain_obj = num[domain_idx]
                domain_residues = None
                
                # Handle ANARCI output format variations (list vs tuple)
                if isinstance(domain_obj, list): domain_residues = domain_obj
                elif isinstance(domain_obj, tuple):
                    if isinstance(domain_obj[0], list): domain_residues = domain_obj[0]
                    else:
                        for item in domain_obj:
                            if isinstance(item, list): domain_residues = item; break
                
                if domain_residues is None: continue
                
                # Extract requested regions (CDR3, FR1, etc.)
                for req in extract_regions:
                    p_seq, valid_indices = "", []
                    
                    if req == 'FL': 
                        p_seq, valid_indices = get_region_seq(domain_residues, (1, 128), domain_start_idx)
                    elif req == 'CDRs':
                        p1, v1 = get_region_seq(domain_residues, IMGT_DEFINITIONS['CDR1'], domain_start_idx)
                        p2, v2 = get_region_seq(domain_residues, IMGT_DEFINITIONS['CDR2'], domain_start_idx)
                        p3, v3 = get_region_seq(domain_residues, IMGT_DEFINITIONS['CDR3'], domain_start_idx)
                        p_seq = p1+p2+p3
                        valid_indices = v1+v2+v3
                    elif req in IMGT_DEFINITIONS: 
                        p_seq, valid_indices = get_region_seq(domain_residues, IMGT_DEFINITIONS[req], domain_start_idx)
                    elif '-' in req: # Range support e.g. "105-117"
                        try:
                            s, e = map(int, req.split('-'))
                            p_seq, valid_indices = get_region_seq(domain_residues, (s, e), domain_start_idx)
                        except: pass
                    
                    if p_seq:
                        row_data[f"{chain_type}_{req}_PEP"] = p_seq
                        row_data[f"{chain_type}_{req}_INDICES"] = valid_indices 
                        has_valid_domain = True
        
        if has_valid_domain:
            chunk_results.append(row_data)
            
    return chunk_results

# --- CLASS DEFINITIONS ---

class Logger:
    """Handles logging configuration (Console + File)."""
    def __init__(self, config=None):
        self.conf = config
        self.__fallback()
        self.__configure_logger()

    def __fallback(self):
        """Sets default values if config is missing."""
        if self.conf is None:
            self.name = 'unnamed ' + str(time.time())
            self.verbose = False
            self.log_to_file = False
            self.log_fname = None
        else:
            self.name = self.conf.name
            self.verbose = self.conf.verbose
            self.log_to_file = self.conf.log_to_file
            self.log_fname = self.conf.log_fname

    def __configure_logger(self):
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s]: %(message)s")
        self.logger.handlers.clear()

        # Console Handler
        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File Handler
        if self.log_to_file and self.log_fname:
            log_dir = os.path.dirname(self.log_fname)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            filehandler = logging.FileHandler(self.log_fname)      
            filehandler.setFormatter(formatter)
            self.logger.addHandler(filehandler)       

class DirectoryTracker:
    """Manages input/output directory paths."""
    def __init__(self, config=None):
        self._conf = config
        self.__fallback()
        self.__setup_dirs()

    def __fallback(self):
        cwd = os.getcwd()
        if self._conf is None:
            self.seq_data = cwd
            self.logs = cwd
            self.parser_out = cwd
        else:
            self.seq_data = self._conf.seq_data
            self.logs = self._conf.logs
            self.parser_out = self._conf.parser_out

    def __setup_dirs(self):
        """Creates directories if they don't exist."""
        for d in [x for x in dir(self) if not x.startswith('_')]:
            path = getattr(self, d)
            if isinstance(path, str) and not os.path.isdir(path):
                os.makedirs(path)

class Handler:
    """Base class for all processing handlers."""
    def __init__(self, *args):
        self.__dict__.update(*args)
        if not hasattr(self, 'logger') or self.logger is None:
            self.logger = Logger().logger
        if not hasattr(self, 'dirs'):
            self.dirs = DirectoryTracker()

    def _on_completion(self):
        msg = f'The following handler was succesfully initialized: {self}'
        self.logger.info(msg)

class Pipeline(Handler):
    """
    Manages the execution flow of data processing routines.
    Supports streaming large datasets in chunks to minimize memory usage.
    """
    def __init__(self, *args):
        super(Pipeline, self).__init__(*args)
        self._on_startup()
        super(Pipeline, self)._on_completion()

    def _on_startup(self):
        self.que = []
        if not hasattr(self, 'exp_name'):
            self.exp_name = 'unnamed'

    def _describe_data(self, data=None):
        """Logs statistics about the current data chunk (e.g., number of reads, lengths)."""
        if data is None: return 0
        total_count = 0
        for sample in data:
            count = len(sample)
            total_count += count
            
            def get_stats(arr, name):
                if arr is None: return "None"
                if hasattr(arr, 'size') and arr.size == 0: return "Empty (0)"
                if isinstance(arr, list) and len(arr) == 0: return "Empty (0)"
                # Handle 2D arrays (like Q-scores or padded strings)
                if hasattr(arr, 'ndim') and arr.ndim == 2:
                    if arr.dtype.kind in ('S', 'U'): 
                        pad = b'' if arr.dtype.kind == 'S' else ''
                        lens = (arr != pad).sum(axis=1)
                        if len(lens) > 0: return f"MaxLen={lens.max()}"
                        return "MaxLen=0"
                    elif arr.dtype == np.uint8:
                         lens = (arr != 0).sum(axis=1)
                         if len(lens) > 0: return f"MaxLen={lens.max()}"
                         return "MaxLen=0"
                    else: return f"Shape={arr.shape}"
                # Handle 1D arrays (list of strings)
                elif hasattr(arr, 'ndim') and arr.ndim == 1:
                    try:
                        max_l = max(len(str(x)) for x in arr)
                        return f"MaxLen={max_l}"
                    except: return "Unknown"
                return "Unknown"

            d_stats = get_stats(sample.D, "DNA")
            p_stats = get_stats(sample.P, "PEP")
            q_stats = get_stats(sample.Q, "Q")
            msg = f"[INFO] {sample.name}: Count={count} | DNA: {d_stats} | PEP: {p_stats} | Q: {q_stats}"
            self.logger.info(msg)
        return total_count

    def enque(self, routines):
        """Adds processing functions to the execution queue."""
        for func in routines:
            self.que.append(func)
        self.logger.info(f'{len(routines)} routines appended to pipeline.')

    def run_over_stream(self, data_iter_factory, save_summary=True):
        """
        Executes the queued routines over a data stream (chunk by chunk).
        """
        summary = []
        chunk_idx = 0
        iterator = data_iter_factory()
        
        total_input_reads = 0
        total_final_reads = 0
        file_start_time = time.time()
        
        for data in iterator:
            chunk_idx += 1
            sample_name = data.samples[0].name if data.samples else "Unknown"
            self.logger.info(f"--- Processing Chunk {chunk_idx} ({sample_name}) ---")
            chunk_input = self._describe_data(data)
            total_input_reads += chunk_input
            current_count = chunk_input
            
            # Execute pipeline steps
            for func in self.que:
                t0 = time.time()
                op_name = func.__name__
                self.logger.info(f"> Running {op_name}...")
                data = func(data)
                op_time = time.time() - t0
                new_count = self._describe_data(data)
                dropped = current_count - new_count
                
                summary.append({
                    'Chunk_ID': chunk_idx, 'Sample': sample_name, 'Operation': op_name,
                    'Time(s)': round(op_time, 3), 'Input_Sequences': current_count,
                    'Dropped_Sequences': dropped, 'Remaining_Sequences': new_count
                })
                current_count = new_count
            total_final_reads += current_count

        total_elapsed = time.time() - file_start_time
        total_dropped = total_input_reads - total_final_reads
        self.logger.info("="*60)
        self.logger.info(f"PROCESSING COMPLETE")
        self.logger.info(f"Total Time: {total_elapsed:.2f}s")
        self.logger.info(f"Total Input:  {total_input_reads}")
        self.logger.info(f"Total Output: {total_final_reads}")
        self.logger.info("="*60)

        if save_summary and summary:
            df = pd.DataFrame(summary)
            # Add summary row
            total_row = {
                'Chunk_ID': 'TOTAL', 'Sample': 'ALL_FILES', 'Operation': 'ALL_STEPS',
                'Time(s)': round(total_elapsed, 3), 'Input_Sequences': total_input_reads,
                'Dropped_Sequences': total_dropped, 'Remaining_Sequences': total_final_reads
            }
            df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
            fname = f'{self.exp_name}_streaming_summary.csv'
            path = os.path.join(self.dirs.logs, fname)
            df.to_csv(path, index=False)
            self.logger.info(f"Summary saved to {path}")
        return None

    def merge_chunk_outputs(self, delete_chunks=True):
        """
        Combines fragmented chunk outputs (CSV/FASTA) into single files per library.
        Useful when data was processed in parallel chunks.
        """
        root = self.dirs.parser_out
        if not os.path.exists(root): return
        chunk_dirs = [d for d in os.listdir(root) if "__chunk" in d and os.path.isdir(os.path.join(root, d))]
        if not chunk_dirs: return
        
        # Group chunks by their base library name
        groups = {}
        for d in chunk_dirs:
            try:
                base = d.split('__chunk')[0]
                if base not in groups: groups[base] = []
                groups[base].append(d)
            except: pass
            
        for base_name, c_dirs in groups.items():
            dest_dir = os.path.join(root, base_name)
            if not os.path.exists(dest_dir): os.makedirs(dest_dir)
            first_chunk_path = os.path.join(root, c_dirs[0])
            if not os.path.isdir(first_chunk_path): continue
            try: files = [f for f in os.listdir(first_chunk_path) if os.path.isfile(os.path.join(first_chunk_path, f))]
            except: continue
            if not files: continue

            suffix_map = {}
            for fname in files:
                if fname.startswith(c_dirs[0]) and fname.endswith('.csv'):
                    suffix = fname[len(c_dirs[0]):]
                    suffix_map[suffix] = fname 
            
            # Merge logic: Sum counts for identical sequences
            for suffix, example_fname in suffix_map.items():
                self.logger.info(f"Merging {suffix} for {base_name}...")
                merged_counts = Counter()
                merged_dna = {} 
                for c_dir in c_dirs:
                    chunk_fname = c_dir + suffix
                    src_path = os.path.join(root, c_dir, chunk_fname)
                    if not os.path.exists(src_path): continue
                    try:
                        df = pd.read_csv(src_path, keep_default_na=False, na_values=[''])
                        if 'Seq' in df.columns and 'Count' in df.columns:
                            for _, row in df.iterrows():
                                seq = row['Seq']; count = int(row['Count'])
                                if seq: 
                                    merged_counts[seq] += count
                                    if 'DNA' in df.columns and row['DNA']:
                                        if seq not in merged_dna: merged_dna[seq] = row['DNA']
                    except: pass
                if not merged_counts: continue
                
                # Save merged output
                final_csv_name = base_name + suffix
                final_fasta_name = final_csv_name.replace('.csv', '.fasta')
                path_csv = os.path.join(dest_dir, final_csv_name)
                path_fasta = os.path.join(dest_dir, final_fasta_name)
                
                sorted_items = sorted(merged_counts.items(), key=lambda x: (-x[1], x[0]))
                final_data = [{'Seq': seq, 'Count': count, 'DNA': merged_dna.get(seq, "")} for seq, count in sorted_items]
                pd.DataFrame(final_data).to_csv(path_csv, index=False)
                self.logger.info(f"Saved merged CSV to {path_csv}")
                with open(path_fasta, 'w') as f:
                    for rank, (seq, count) in enumerate(sorted_items, start=1):
                        f.write(f">seq_{rank}_count_{count}\n{seq}\n")
            
            if delete_chunks:
                for c_dir in c_dirs:
                    try: shutil.rmtree(os.path.join(root, c_dir))
                    except: pass


class IgblastExtractor(Handler):
    """
    Splits scFv via linker, runs IgBLAST, extracts specific variable regions, 
    and translates them while strictly preserving Heavy-Light pairing.
    """
    def __init__(self, config_meta):
        super(IgblastExtractor, self).__init__(config_meta)
        self.igblast_exec = config_meta.get('igblast_exec', 'igblastn')
        self.igdata_path = config_meta.get('igdata_path', '')
        self.db_dir = config_meta.get('database_dir', '')
        self.species = config_meta.get('species', 'human')
        self.extract_regions = config_meta.get('extract_regions', ['FL'])
        
        self.linker_seq = config_meta.get('linker_seq', '').encode('ascii')
        
        # Pre-calculate the reverse complement of the linker
        if self.linker_seq:
            linker_arr = np.frombuffer(self.linker_seq, dtype=np.uint8)
            self.linker_rc = _nb_revcom(linker_arr).tobytes()
            self.linker_tol = int(len(self.linker_seq) * config_meta.get('linker_tol_ratio', 0.1))
        else:
            self.linker_rc = b''
            self.linker_tol = 0
        
        self.temp_dir = '/dev/shm' if os.path.exists('/dev/shm') else None

    @staticmethod
    def rescue_cdr3_and_fr4(translated_sequence):
        """
        Regex safety net to extract BOTH CDR3 and FR4 from highly mutated 
        clones when IgBLAST drops the J-gene alignment.
        """
        if pd.isna(translated_sequence) or not isinstance(translated_sequence, str):
            return pd.NA, pd.NA
            
        # The explicit list of 17 biologically validated FR3 anchors
        left_anchors = r'(YYC|HYC|YHC|YGC|CYC|HGC|YCC|FNC|YFC|YSC|YLC|HFC|KFC|YDC|FYC|YIC|SYC)'
        
        pattern = f'{left_anchors}(.{{1,30}}?)(WG.G|FG.G|WVG|FSDG|LG.G|IG.G|$)(.*)'
        match = re.search(pattern, translated_sequence)
        
        if match:
            cdr3 = match.group(2)
            anchor = match.group(3)
            tail = match.group(4)
            
            # If the sequence physically ended before FR4, return empty FR4
            if not anchor:
                fr4 = ""
            else:
                # FR4 is the anchor + the tail. Capped at 11 amino acids.
                fr4 = (anchor + tail)[:11]
                
            return cdr3, fr4
            
        return pd.NA, pd.NA

    def _split_linker(self, dna_bytes):
        if not self.linker_seq: return dna_bytes, None
        
        # 1. Try matching the forward linker
        idx = _match_block_fuzzy(dna_bytes, self.linker_seq, self.linker_tol)
        if idx != -1:
            return dna_bytes[:idx], dna_bytes[idx + len(self.linker_seq):]
            
        # 2. Try matching the reverse complement linker
        idx_rc = _match_block_fuzzy(dna_bytes, self.linker_rc, self.linker_tol)
        if idx_rc != -1:
            return dna_bytes[:idx_rc], dna_bytes[idx_rc + len(self.linker_rc):]
            
        return dna_bytes, None

    def run_igblast_and_translate(self):
        
        def _extract_and_translate_domains(hit):
            """Extracts and translates all 7 IMGT domains individually with 5' and 3' physical rescue."""
            seq = str(hit.get('sequence', ''))
            if not seq or seq == 'nan': return {}
            
            domain_map = {
                'fr1': 'fwr1', 'cdr1': 'cdr1', 
                'fr2': 'fwr2', 'cdr2': 'cdr2', 
                'fr3': 'fwr3', 'cdr3': 'cdr3', 
                'fr4': 'fwr4'
            }
            
            results = {}
            
            for internal_name, airr_prefix in domain_map.items():
                start = hit.get(f'{airr_prefix}_start')
                end = hit.get(f'{airr_prefix}_end')
                
                if pd.notna(start):
                    start_idx = int(float(start)) - 1
                    end_idx = int(float(end)) if pd.notna(end) else start_idx
                    
                    # --- 5' PHYSICAL RESCUE (FR1) ---
                    if internal_name == 'fr1':
                        # CRITICAL FIX: Use germline start, not sequence start
                        v_germ_start = hit.get('v_germline_start')
                        if pd.notna(v_germ_start):
                            # Calculate exactly how many germline bases are missing (e.g., 4 - 1 = 3)
                            missing_nt = int(float(v_germ_start)) - 1
                            if missing_nt > 0:
                                # Step backward exactly that many bases into the raw read
                                start_idx = max(0, start_idx - missing_nt)
                                
                    # --- 3' ANCHORED BASE COUNTING (FR4) ---
                    elif internal_name == 'fr4':
                        end_idx = min(len(seq), start_idx + 33)
                        
                    if start_idx < end_idx:
                        dna_slice = seq[start_idx:end_idx].upper()
                        dna_arr = np.frombuffer(dna_slice.encode('ascii'), dtype=np.uint8)
                        pep_slice = _nb_dna_to_pep(dna_arr, stop_readthrough=True).tobytes().decode('ascii')
                        
                        results[f'{internal_name}_DNA'] = dna_slice
                        results[f'{internal_name}_PEP'] = pep_slice
                        continue
                        
                results[f'{internal_name}_DNA'] = ""
                results[f'{internal_name}_PEP'] = ""
                
            return results

        def _op(data):
            for sample in data:
                new_D, new_P, new_Q = [], [], []
                fasta_entries = []
                
                # 1. Split by Linker
                for i, row_data in enumerate(sample.D):
                    row_bytes = row_data.tobytes().strip(b'\x00') if row_data.dtype.kind == 'S' else "".join(row_data).encode('ascii')
                    p1, p2 = self._split_linker(row_bytes)
                    fasta_entries.append(f">seq{i}_P1\n{p1.decode('ascii')}")
                    if p2 and len(p2) > 50: fasta_entries.append(f">seq{i}_P2\n{p2.decode('ascii')}")

                if fasta_entries:
                    with tempfile.NamedTemporaryFile(mode='w', dir=self.temp_dir, suffix='.fasta', delete=False) as tmp_fasta:
                        tmp_fasta.write("\n".join(fasta_entries))
                        fasta_path = tmp_fasta.name

                    # 2. Run IgBLAST
                    env = os.environ.copy()
                    env["IGDATA"] = self.igdata_path
                    
                    cmd = [
                        self.igblast_exec, '-query', fasta_path, '-organism', self.species,
                        '-ig_seqtype', 'Ig', '-domain_system', 'imgt',
                        '-outfmt', '19', '-num_threads', str(multiprocessing.cpu_count())
                    ]
                    
                    aux_file = os.path.join(self.igdata_path, 'optional_file', f'{self.species}_gl.aux')
                    if os.path.exists(aux_file):
                        cmd.extend(['-auxiliary_data', aux_file])
                    else:
                        self.logger.warning(f"No auxiliary data found for {self.species}. CDR3 end boundaries may be incomplete.")

                    for locus in ['V', 'D', 'J']:
                        cmd.extend([f'-germline_db_{locus}', os.path.join(self.db_dir, f'{self.species}_{locus}_igblast.fasta')])


                    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
                    os.remove(fasta_path) 

                    if result.returncode == 0:
                        try:
                            # 3. Parse AIRR & Group by Original Read ID
                            df = pd.read_csv(StringIO(result.stdout), sep='\t')
                            df['orig_idx'] = df['sequence_id'].apply(lambda x: int(x.split('_')[0].replace('seq', '')))
                            grouped = df.groupby('orig_idx')
                            
                            for i in range(len(sample.D)):
                                if i not in grouped.groups: continue
                                
                                hits = grouped.get_group(i)
                                h_data, l_data = {}, {}
                                
                                for _, hit in hits.iterrows():
                                    chain = 'H' if str(hit.get('locus', '')).startswith('IGH') else 'L'
                                    
                                    # 1. Grab Continuous Untrimmed DNA starting exactly from FR1
                                    raw_seq = str(hit.get('sequence', ''))
                                    start_idx = 0
                                    if pd.notna(hit.get('fwr1_start')):
                                        start_idx = max(0, int(float(hit.get('fwr1_start'))) - 1)
                                        if pd.notna(hit.get('v_germline_start')):
                                            start_idx = max(0, start_idx - (int(float(hit.get('v_germline_start'))) - 1))
                                            
                                    continuous_dna = raw_seq[start_idx:].upper() if start_idx < len(raw_seq) else ""
                                    
                                    # 2. Fetch all individually translated domains
                                    doms = _extract_and_translate_domains(hit)
                                    if not doms: continue

                                    # 3. Cleaned-Up Rescue Block (No more DNA slicing math!)
                                    if not doms.get('cdr3_PEP'):
                                        if continuous_dna:
                                            dna_arr = np.frombuffer(continuous_dna.encode('ascii'), dtype=np.uint8)
                                            full_pep = _nb_dna_to_pep(dna_arr, stop_readthrough=True).tobytes().decode('ascii')
                                            
                                            rescued_cdr3, rescued_fr4 = self.rescue_cdr3_and_fr4(full_pep)
                                            
                                            if pd.notna(rescued_cdr3):
                                                doms['cdr3_PEP'] = rescued_cdr3
                                                doms['fr4_PEP'] = rescued_fr4 if rescued_fr4 else ""

                                    # 4. Assemble the requested Peptides
                                    for reg in self.extract_regions:
                                        pep_seq = ""
                                        
                                        if reg == 'FL': keys = ['fr1', 'cdr1', 'fr2', 'cdr2', 'fr3', 'cdr3', 'fr4']
                                        elif reg == 'CDRs': keys = ['cdr1', 'cdr2', 'cdr3']
                                        elif reg.lower() in ['fr1', 'cdr1', 'fr2', 'cdr2', 'fr3', 'cdr3', 'fr4']: keys = [reg.lower()]
                                        else: continue 
                                            
                                        for k in keys:
                                            pep_seq += doms.get(f'{k}_PEP', '')
                                            
                                        if pep_seq:
                                            # Grab metadata only upon successful sequence assembly
                                            v_call = str(hit.get('v_call', '')).split(',')[0] if pd.notna(hit.get('v_call')) else ""
                                            j_call = str(hit.get('j_call', '')).split(',')[0] if pd.notna(hit.get('j_call')) else ""

                                            if chain == 'H':
                                                h_data[f"H_{reg}_PEP"] = pep_seq
                                                h_data['H_FL_DNA'] = continuous_dna  # Store untrimmed DNA once!
                                                h_data['V_Gene'] = v_call
                                                h_data['J_Gene'] = j_call
                                                h_data['Species'] = self.species
                                            else:
                                                l_data[f"L_{reg}_PEP"] = pep_seq
                                                l_data['L_FL_DNA'] = continuous_dna  # Store untrimmed DNA once!
                                                l_data['V_Gene'] = v_call
                                                l_data['J_Gene'] = j_call
                                                l_data['Species'] = self.species
                                # 5. Tightly couple the H and L chains AND attach Germline Metadata
                                pep_parts = []
                                valid_read = False
                                has_stop_codon = False
                                
                                for reg in self.extract_regions:
                                    hp, lp = h_data.get(f"H_{reg}_PEP", ""), l_data.get(f"L_{reg}_PEP", "")

                                    if '*' in hp or '*' in lp:
                                        has_stop_codon = True
                                        break
                                        
                                    if hp or lp: valid_read = True
                                    pep_parts.append(f"H_{reg}:{hp}|L_{reg}:{lp}")
                                    
                                # Grab the single, continuous DNA string for each chain
                                hd, ld = h_data.get("H_FL_DNA", ""), l_data.get("L_FL_DNA", "")
                                dna_str = f"H_FL:{hd}|L_FL:{ld}"
                                
                                # Skip packaging and counting this sequence entirely
                                if has_stop_codon:
                                    continue
                                    
                                if valid_read:
                                    # Append the germline metadata so it persists through the counting phase
                                    meta_parts = []
                                    if 'V_Gene' in h_data:
                                        meta_parts.append(f"H_V_Gene:{h_data['V_Gene']}|H_J_Gene:{h_data['J_Gene']}|H_Species:{h_data['Species']}")
                                    if 'V_Gene' in l_data:
                                        meta_parts.append(f"L_V_Gene:{l_data['V_Gene']}|L_J_Gene:{l_data['J_Gene']}|L_Species:{l_data['Species']}")
                                        
                                    full_pep_str = "||".join(pep_parts)
                                    if meta_parts:
                                        full_pep_str += "||" + "||".join(meta_parts)
                                        
                                    new_P.append(full_pep_str)
                                    new_D.append(dna_str)  # Send only the clean, full-length DNA pair!
                                    if sample.Q is not None: new_Q.append(sample.Q[i])
                                    
                        except Exception as e:
                            self.logger.error(f"Parsing AIRR failed: {e}")

                # Save the successfully translated sequences back to the sample object
                sample.D = np.array(new_D, dtype=object)
                sample.P = np.array(new_P, dtype=object)
                if sample.Q is not None: sample.Q = np.array(new_Q, dtype=object)
                sample.transform()

            return data
        
        _op.__name__ = "run_igblast_and_translate"
        return _op

    def format_for_downstream(self):
        """
        Unpacks the tightly paired strings and germline metadata into 
        the explicitly requested exact column format for downstream analysis.
        """
        search_pattern = os.path.join(self.dirs.parser_out, "*", "*_pep_counts.csv")
        merged_files = glob.glob(search_pattern)
        
        # The exact column layout requested
        exact_columns = [
            'ID', 'H_CDR3_PEP', 'H_CDRs_PEP', 'H_FL_PEP', 
            'L_CDRs_PEP', 'L_FL_PEP', 'L_CDR3_PEP', 
            'H_V_Gene', 'H_J_Gene', 'H_Species', 
            'L_V_Gene', 'L_Species', 'L_J_Gene', 'Count', 
            'H_FL_DNA', 'L_FL_DNA'
        ]
        
        for file_path in merged_files:
            try: df = pd.read_csv(file_path, keep_default_na=False)
            except: continue
            if 'Seq' not in df.columns: continue

            records = []
            for _, row in df.iterrows():
                record = {'Count': row['Count']}
                
                # Unpack Peptides and Metadata
                for pr in str(row['Seq']).split('||'):
                    for chain_part in pr.split('|'):
                        if ':' in chain_part:
                            k, v = chain_part.split(':', 1)
                            # Do not add '_PEP' to the gene/species metadata keys
                            if k.endswith('_Gene') or k.endswith('_Species'):
                                record[k] = v
                            else:
                                record[f"{k}_PEP"] = v
                            
                # Unpack DNAs
                if 'DNA' in row and pd.notna(row['DNA']):
                    for dr in str(row['DNA']).split('||'):
                        for chain_part in dr.split('|'):
                            if ':' in chain_part:
                                k, v = chain_part.split(':', 1)
                                record[f"{k}_DNA"] = v
                                
                records.append(record)
                
            new_df = pd.DataFrame(records)
            base = os.path.basename(file_path).replace('_pep_counts.csv', '')
            new_df.insert(0, 'ID', [f"{base}_{i:06d}" for i in range(1, len(new_df) + 1)])
            
            # Guarantee every requested column exists (fill missing with empty strings)
            for col in exact_columns:
                if col not in new_df.columns:
                    new_df[col] = ""
                    
            # Enforce the exact requested column order and drop any extra unexpected columns
            new_df = new_df[exact_columns]

            out_name = file_path.replace('_pep_counts.csv', '_pep_counts_annotated.csv')
            new_df.to_csv(out_name, index=False)
            self.logger.info(f"Formatted for downstream analysis: {out_name}")

class FastqParser(Handler):
    """
    Handles core FASTQ processing operations:
    Length filtering, Quality filtering, Translation, and Pattern Extraction.
    """
    def __init__(self, *args):
        super(FastqParser, self).__init__(*args)
        self._validate() 
        super(FastqParser, self)._on_completion()

    def _validate(self):
        if not (hasattr(self, 'P_design') and hasattr(self, 'D_design')): pass

    def _transform_check(self, sample, func):
        """Ensures data is in the correct format (numpy array) before processing."""
        if not sample.get_ndims() == 2:
            try: sample.transform()
            except: pass

    def len_filter(self, where='dna', len_range=None):
        """Factory for length filtering operation."""
        if len_range is None: raise ValueError("len_range must be specified [min, max]")
        min_l, max_l = len_range[0], len_range[1]
        def _op(data):
            for sample in data:
                self._transform_check(sample, inspect.stack()[0][3])
                arr = sample[where]
                # Calculate lengths
                if arr.dtype.kind in ('S', 'U'):
                    pad = b'' if arr.dtype.kind == 'S' else ''
                    lengths = (arr != pad).sum(axis=1)
                else: lengths = np.array([len(str(x)) for x in arr])
                # Filter
                mask = (lengths >= min_l) & (lengths < max_l)
                sample(mask)
            return data
        _op.__name__ = f"len_filter_{where}"
        return _op
    
    def q_score_filt(self, minQ=30, frac=0.9):
        """Factory for Quality Score filtering."""
        def _op(data):
            for sample in data:
                self._transform_check(sample, inspect.stack()[0][3])
                if sample.Q is None: continue
                if sample.Q.dtype == object: sample.transform_Q()
                # Apply Numba-accelerated filter
                mask = _nb_q_filter(sample.Q, minQ, frac)
                sample(mask)
            return data
        _op.__name__ = f"q_score_filt_Q{minQ}"
        return _op
    
    def translate_both_strands(self, *, force_at_frame=None, stop_readthrough=False, utr5_offset=0, tol=0):
        """
        Translates DNA in both forward and reverse complement directions.
        Uses fuzzy matching of Barcode/UTR sequences to find the correct Frame.
        """
        bc_bytes = self.barcode[0].encode('ascii')
        utr_bytes = self.utr5_seq[0].encode('ascii')
        bc_arr = np.frombuffer(bc_bytes, dtype=np.uint8)
        utr_arr = np.frombuffer(utr_bytes, dtype=np.uint8)
        
        def _op(data):
            for sample in data:
                new_D, new_Q, new_P = [], [], []
                for i in range(len(sample.D)):
                    d_raw = sample.D[i]
                    q_raw = sample.Q[i] if sample.Q is not None and len(sample.Q) > i else None
                    if isinstance(d_raw, str): seq_b = d_raw.encode('ascii')
                    elif isinstance(d_raw, bytes): seq_b = d_raw
                    elif isinstance(d_raw, np.ndarray): seq_b = d_raw.tobytes()
                    else: seq_b = str(d_raw).encode('ascii')
                    seq_arr = np.frombuffer(seq_b, dtype=np.uint8)
                    
                    # 1. Try Forward Match
                    bc_idx = _match_block_fuzzy(seq_arr, bc_arr, tol)
                    found_fwd = False
                    if bc_idx != -1:
                        utr_idx = _match_block_fuzzy(seq_arr, utr_arr, tol)
                        if utr_idx != -1:
                            found_fwd = True; start = utr_idx + utr5_offset
                            if start < 0: start = 0
                            coding_seq = seq_b[start:]; q_slice = q_raw[start:] if q_raw is not None else None
                            pep_arr = _nb_dna_to_pep(np.frombuffer(coding_seq, dtype=np.uint8), stop_readthrough)
                            new_D.append(coding_seq.decode('ascii')); new_Q.append(q_slice); new_P.append(pep_arr.tobytes().decode('ascii'))
                    if found_fwd: continue
                    
                    # 2. Try Reverse Complement Match
                    rc_arr = _nb_revcom(seq_arr); rc_bytes = rc_arr.tobytes()
                    bc_idx_rc = _match_block_fuzzy(rc_arr, bc_arr, tol)
                    if bc_idx_rc != -1:
                        utr_idx_rc = _match_block_fuzzy(rc_arr, utr_arr, tol)
                        if utr_idx_rc != -1:
                            start = utr_idx_rc + utr5_offset
                            if start < 0: start = 0
                            coding_seq = rc_bytes[start:]; q_slice = q_raw[::-1][start:] if q_raw is not None else None
                            pep_arr = _nb_dna_to_pep(np.frombuffer(coding_seq, dtype=np.uint8), stop_readthrough)
                            new_D.append(coding_seq.decode('ascii')); new_Q.append(q_slice); new_P.append(pep_arr.tobytes().decode('ascii'))
                            
                sample.D = np.array(new_D, dtype=object)
                sample.P = np.array(new_P, dtype=object)
                if sample.Q is not None: sample.Q = np.array(new_Q, dtype=object)
                sample.transform()
            return data
        _op.__name__ = f"translate_both_strands_tol{tol}"
        return _op
    
    def translate_all_6_frames(self, stop_readthrough=False):
        """Blind translation of all 3 forward frames and 3 reverse frames."""
        def _op(data):
            new_samples = []
            for sample in data:
                D_all, P_all, Q_all = [], [], []
                def get_bytes_arr(seq_obj):
                    if isinstance(seq_obj, bytes): return np.frombuffer(seq_obj, dtype=np.uint8)
                    if isinstance(seq_obj, str): return np.frombuffer(seq_obj.encode('ascii'), dtype=np.uint8)
                    if isinstance(seq_obj, np.ndarray): return seq_obj.view(np.uint8)
                    return np.frombuffer(str(seq_obj).encode('ascii'), dtype=np.uint8)
                for i in range(len(sample.D)):
                    d_arr = get_bytes_arr(sample.D[i])
                    # Forward 3 frames
                    for f in range(3):
                        if len(d_arr) > f:
                            pep = _nb_dna_to_pep(d_arr[f:], stop_readthrough)
                            P_all.append(pep.tobytes().decode('ascii')); D_all.append(d_arr.tobytes().decode('ascii')); Q_all.append(None) 
                    # Reverse 3 frames
                    rc_arr = _nb_revcom(d_arr)
                    for f in range(3):
                        if len(rc_arr) > f:
                            pep = _nb_dna_to_pep(rc_arr[f:], stop_readthrough)
                            P_all.append(pep.tobytes().decode('ascii')); D_all.append(rc_arr.tobytes().decode('ascii')); Q_all.append(None)
                new_s = SequencingSample(name=f"{sample.name}_6frames",D=D_all, P=P_all, Q=Q_all if any(x is not None for x in Q_all) else [])
                new_s.D = np.array(new_s.D, dtype=object); new_s.P = np.array(new_s.P, dtype=object); new_s.transform()
                new_samples.append(new_s)
            return Data(samples=new_samples)
        _op.__name__ = "translate_all_6_frames"
        return _op
    
    def extract_fuzzy_regions(self, *, where='pep', loc=None, tol=0, pad=''):
        """
        Extracts variable regions based on constant flanking anchors.
        
        Args:
            where (str): 'pep' (protein) or 'dna'.
            loc (list): List of region indices to extract (e.g. [0, 1, 2]).
            tol (int or dict): 
                - If int: Applies this tolerance to ALL anchors.
                - If dict: Maps region index to specific tolerance. e.g. {0: 2, 2: 0}.
                           Keys must match the indices in your design.
        """
        design = self.P_design if where == 'pep' else self.D_design
        if loc is None: raise ValueError("loc must be specified")
        
        # Precompute anchors from design template
        anchors_list = []
        for tpl in design:
            t_anchors = {}
            for r_idx in tpl.loc:
                if not tpl.is_vr[r_idx]:
                    seq = "".join(tpl([r_idx]))
                    t_anchors[r_idx] = seq.encode('ascii')
            anchors_list.append(t_anchors)
            
        def _op(data):
            for sample in data:
                self._transform_check(sample, inspect.stack()[0][3])
                arr = sample[where]; n_rows = arr.shape[0]
                new_D, new_P, new_Q = [], [], []
                
                for i in range(n_rows):
                    row_data = arr[i]
                    if row_data.dtype.kind == 'S': row_bytes = row_data.tobytes().strip(b'\x00').strip(b'')
                    else: row_bytes = "".join(row_data).encode('ascii')
                    
                    best_match_seq = None; extract_start_index = 0; extract_len = 0
                    
                    # Try to match against provided templates
                    for t_idx, anchors in enumerate(anchors_list):
                        cursor = 0; valid_template = True; temp_extract_parts = []; current_parts_len = 0; first_vr_start = -1 
                        
                        for target_idx in loc:
                            
                            # --- DETERMINE TOLERANCE FOR THIS STEP ---
                            # Logic: If 'tol' is a dict, look up the specific region index we are trying to match.
                            # If we are matching a Constant Region (Anchor) at target_idx, use tol[target_idx]
                            # If we are matching the RIGHT anchor of a Variable Region, use tol[target_idx + 1]
                            
                            current_tol = 0
                            anchor_idx_being_checked = -1
                            
                            is_vr = design.templates[t_idx].is_vr[target_idx]
                            if not is_vr:
                                anchor_idx_being_checked = target_idx
                            else:
                                anchor_idx_being_checked = target_idx + 1 # The anchor is the next block
                            
                            if isinstance(tol, dict):
                                current_tol = tol.get(anchor_idx_being_checked, 0) # Default to 0 if not specified in dict
                            else:
                                current_tol = tol # Use global int
                                
                            # ------------------------------------------

                            # Case A: target_idx is a Constant Region (Anchor)
                            if not is_vr:
                                motif = anchors[target_idx]
                                pos = _match_block_fuzzy(row_bytes[cursor:], motif, current_tol)
                                
                                if pos != -1:
                                    abs_pos = cursor + pos
                                    if first_vr_start == -1: first_vr_start = abs_pos
                                    part = row_bytes[abs_pos : abs_pos+len(motif)]; temp_extract_parts.append(part); current_parts_len += len(part); cursor = abs_pos + len(motif)
                                else: valid_template = False; break
                            
                            # Case B: target_idx is a Variable Region (Extraction Target)
                            else:
                                if first_vr_start == -1: first_vr_start = cursor
                                next_cr_idx = target_idx + 1
                                # Look ahead for the next anchor to define the boundary
                                if next_cr_idx in anchors:
                                    right_motif = anchors[next_cr_idx]
                                    pos = _match_block_fuzzy(row_bytes[cursor:], right_motif, current_tol)
                                    
                                    if pos != -1:
                                        abs_pos_right = cursor + pos; part = row_bytes[cursor : abs_pos_right]; temp_extract_parts.append(part); current_parts_len += len(part); cursor = abs_pos_right 
                                    else: valid_template = False; break
                                else:
                                    part = row_bytes[cursor:]; temp_extract_parts.append(part); current_parts_len += len(part); cursor = len(row_bytes)
                        
                        if valid_template:
                            best_match_seq = b"".join(temp_extract_parts); extract_start_index = first_vr_start; extract_len = current_parts_len; break 
                    
                    if best_match_seq is not None:
                        decoded_seq = best_match_seq.decode('ascii')
                        p_start = extract_start_index; p_end = extract_start_index + extract_len
                        if sample.D[i].dtype.kind == 'S': d_full = sample.D[i].tobytes().strip(b'\x00').decode('ascii')
                        else: d_full = "".join(sample.D[i])
                        d_slice = d_full[p_start*3 : p_end*3]
                        if where == 'pep':
                            new_P.append(decoded_seq); new_D.append(d_slice)
                            if sample.Q is not None: new_Q.append(sample.Q[i]) 
                        else: new_D.append(decoded_seq); new_P.append(sample.P[i])
                sample.D = np.array(new_D, dtype=object); sample.P = np.array(new_P, dtype=object)
                if sample.Q is not None:
                    if len(new_Q) == len(new_P): sample.Q = np.array(new_Q, dtype=object)
                    else: sample.Q = np.empty(len(new_P), dtype=object)
                sample.transform()
            return data
        _op.__name__ = f"extract_fuzzy_regions_{where}"
        return _op
        
    def unpad(self):
        def _op(data):
            for sample in data: sample.unpad()
            return data
        _op.__name__ = "unpad"
        return _op
    
    def count_summary(self, where='pep', fmt='csv'):
        """Counts unique sequences and saves them to disk."""
        def _op(data):
            for sample in data:
                arr = sample[where]
                rows = []; dnas = []
                for x in arr:
                    if x.dtype.kind == 'S': rows.append(x.tobytes().decode('ascii').strip('\x00'))
                    else: rows.append("".join(x))
                for x in sample.D:
                    if x.dtype.kind == 'S': dnas.append(x.tobytes().decode('ascii').strip('\x00'))
                    else: dnas.append("".join(x))
                counter = Counter(); key_to_dna = {}
                for k, d in zip(rows, dnas):
                    counter[k] += 1
                    if k not in key_to_dna: key_to_dna[k] = d
                dest = os.path.join(self.dirs.parser_out, sample.name, sample.name + f'_{where}_counts.{fmt}')
                chunk_output_dir = os.path.join(self.dirs.parser_out, sample.name)
                if not os.path.exists(chunk_output_dir): os.makedirs(chunk_output_dir)
                if fmt == 'csv':
                    items = []
                    for k, cnt in counter.most_common(): items.append({'Seq': k, 'Count': cnt, 'DNA': key_to_dna[k]})
                    pd.DataFrame(items).to_csv(dest, index=False)
                elif fmt == 'fasta':
                    with open(dest, 'w') as f:
                        for i, (seq, cnt) in enumerate(counter.most_common()): f.write(f">seq_{i}_count_{cnt}\n{seq}\n")
            return data
        _op.__name__ = f"count_summary_{where}_{fmt}"
        return _op
    
    def stream_chunks_from_gz_dir(self, chunk_lines=100000):
        """Generator to read large GZIP files in chunks."""
        fnames = [os.path.join(self.dirs.seq_data, x) for x in os.listdir(self.dirs.seq_data) if x.endswith(".gz")]
        def _iter():
            for f in fnames:
                basename = os.path.basename(f).split('.')[0]
                with gzip.open(f, 'rt') as fh:
                    lines = []; chunk_id = 0
                    for line in fh:
                        lines.append(line)
                        if len(lines) >= chunk_lines:
                            d = [x.strip() for x in lines[1::4]]; q = [x.strip() for x in lines[3::4]]
                            chunk_id += 1; name = f"{basename}__chunk{chunk_id}"
                            s = SequencingSample(D=d, Q=q, P=None, name=name); yield Data(samples=[s]); lines = []
                    if lines:
                         d = [x.strip() for x in lines[1::4]]; q = [x.strip() for x in lines[3::4]]
                         chunk_id += 1; name = f"{basename}__chunk{chunk_id}"
                         s = SequencingSample(D=d, Q=q, P=None, name=name); yield Data(samples=[s])
        return _iter
    
    
class AnarciAnnotator:
    """
    Wrapper for ANARCI to perform parallelized antibody numbering and region extraction.
    """
    def __init__(self, output_dir, logger):
        self.output_dir = output_dir
        self.logger = logger
        if not ANARCI_AVAILABLE:
            self.logger.warning("ANARCI not installed.")

    def process_merged_csv(self, file_path, extract_regions=None, output_mode='clean', n_cpu=1, allowed_species=None, assign_germline=False):
        """
        Annotates a CSV of sequences using ANARCI.
        
        UPDATED: Automatically enforces that H and L columns exist for all detected regions.
        If a VHH library is processed, L columns will be created and filled with blanks.
        """
        t_start_load = time.time()
        
        if extract_regions is None: extract_regions = ['FL']
        if not os.path.exists(file_path): return

        self.logger.info(f"Starting Optimized ANARCI (External Pool) for {file_path}...")
        
        try:
            df = pd.read_csv(file_path, keep_default_na=False, na_values=[''])
        except: return
        input_rows = len(df)
        
        sequences = [(str(i), row['Seq']) for i, row in df.iterrows()]
        
        if n_cpu <= 1: n_cpu = multiprocessing.cpu_count()
        self.logger.info(f"Splitting {input_rows} sequences into {n_cpu} chunks for {n_cpu} cores.")
        
        chunk_size = int(np.ceil(len(sequences) / n_cpu))
        seq_chunks = [sequences[i:i + chunk_size] for i in range(0, len(sequences), chunk_size)]
        
        worker_args = []
        for chunk in seq_chunks:
            worker_args.append((chunk, extract_regions, allowed_species, assign_germline, output_mode))
            
        all_results = []
        try:
            with multiprocessing.Pool(processes=n_cpu) as pool:
                chunk_outputs = pool.map(_anarci_worker, worker_args)
                for chunk_res in chunk_outputs:
                    all_results.extend(chunk_res)
        except Exception as e:
            self.logger.error(f"Multiprocessing failed: {e}")
            return

        # Re-assemble results and map back to original DNA
        final_rows = []
        for res in all_results:
            original_idx = res.pop('_Original_Index') 
            full_dna = df.at[original_idx, 'DNA']
            count = df.at[original_idx, 'Count']
            
            # Map protein indices back to DNA indices
            for key in list(res.keys()):
                if key.endswith('_INDICES'):
                    indices = res.pop(key)
                    prefix = key.replace('_INDICES', '')
                    dna_parts = []
                    for q_idx in indices:
                        s = q_idx * 3; e = s + 3
                        if e <= len(full_dna): dna_parts.append(full_dna[s:e])
                    res[f"{prefix}_DNA"] = "".join(dna_parts)
            
            res['Count'] = count
            final_rows.append(res)

        res_df = pd.DataFrame(final_rows)
        res_df.fillna('', inplace=True)
        
        # --- NEW LOGIC: ENFORCE H/L COLUMN CONSISTENCY ---
        # 1. Identify all unique suffixes (e.g. '_CDR3_PEP', '_V_Gene') ignoring the H/L prefix
        # 2. Ensure both H_{Suffix} and L_{Suffix} exist.
        
        existing_cols = set(res_df.columns)
        suffixes = set()
        
        # Regex to capture "H_CDR3_PEP" -> group(2) is "CDR3_PEP"
        # We consider H, L, K (normalized to L)
        col_pattern = re.compile(r"^[HLK]_(.+)$")
        
        for col in existing_cols:
            match = col_pattern.match(col)
            if match:
                suffixes.add(match.group(1))
        
        # Force H and L versions for every found suffix
        for suffix in suffixes:
            for chain in ['H', 'L']:
                col_name = f"{chain}_{suffix}"
                if col_name not in res_df.columns:
                    # Fill missing column with empty strings
                    res_df[col_name] = ""
                    # self.logger.debug(f"Created missing column: {col_name}")

        # --------------------------------------------------

        # Aggregate duplicates after extraction
        pep_cols = [c for c in res_df.columns if c.endswith('_PEP')]
        germ_cols = [c for c in res_df.columns if c.endswith('_Gene') or c.endswith('_Species')]
        group_keys = pep_cols + germ_cols
        dna_cols = [c for c in res_df.columns if c.endswith('_DNA')]
        
        output_rows = 0
        if group_keys:
            agg_dict = {'Count': 'sum'}
            for dc in dna_cols: agg_dict[dc] = 'first'
            
            # Groupby will now safely include the empty L columns if present
            grouped = res_df.groupby(group_keys).agg(agg_dict).reset_index()
            grouped.sort_values(by='Count', ascending=False, inplace=True)
            
            base = os.path.basename(file_path).replace('.csv', '').replace('_pep_counts', '')
            grouped.insert(0, 'ID', [f"{base}_{i:06d}" for i in range(1, len(grouped) + 1)])
            
            out_name = file_path.replace('.csv', '_annotated.csv')
            grouped.to_csv(out_name, index=False)
            self.logger.info(f"Saved: {out_name}")
            output_rows = len(grouped)
        
        total = time.time() - t_start_load
        
        self.logger.info("="*60)
        self.logger.info(f"ANARCI ANNOTATION SUMMARY: {os.path.basename(file_path)}")
        self.logger.info(f"Total Time:       {total:.2f}s")
        self.logger.info(f"Total Input:      {input_rows} sequences")
        self.logger.info(f"Total Output:     {output_rows} annotated groups")
        self.logger.info("="*60)

    def visualize_convergence(self, file_path, region_specs, output_dir=None):
        """
        Generates Rarefaction/Convergence plots to assess library diversity.
        """
        if not os.path.exists(file_path):
            self.logger.error(f"File not found for visualization: {file_path}")
            return

        try:
            df = pd.read_csv(file_path, keep_default_na=False)
        except Exception as e:
            self.logger.error(f"Failed to read CSV for visualization: {e}")
            return

        if 'Count' not in df.columns:
            self.logger.error("Input CSV missing 'Count' column, cannot calculate convergence.")
            return

        save_dir = output_dir if output_dir else os.path.dirname(file_path)
        base_name = os.path.basename(file_path).replace('_annotated.csv', '')

        for spec in region_specs:
            target_cols = spec.split('-')
            missing_cols = [col for col in target_cols if col not in df.columns]
            
            if missing_cols:
                self.logger.warning(f"Skipping region spec '{spec}': Missing columns {missing_cols}")
                continue
            
            if len(target_cols) > 1:
                combined_seqs = df[target_cols[0]].astype(str)
                for col in target_cols[1:]:
                    combined_seqs = combined_seqs + "-" + df[col].astype(str)
                group_series = combined_seqs
            else:
                group_series = df[target_cols[0]]

            grouped_df = df.groupby(group_series)['Count'].sum().reset_index()
            grouped_df.sort_values(by='Count', ascending=False, inplace=True)

            total_reads = grouped_df['Count'].sum()
            if total_reads == 0:
                 self.logger.warning(f"Total reads is 0 for spec '{spec}', skipping plot.")
                 continue
                 
            grouped_df['Cumulative_Count'] = grouped_df['Count'].cumsum()
            grouped_df['Cumulative_Percentage'] = (grouped_df['Cumulative_Count'] / total_reads) * 100
            grouped_df['Rank'] = range(1, len(grouped_df) + 1)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(grouped_df['Rank'], grouped_df['Cumulative_Percentage'], linewidth=2, color='#1f77b4')
            ax.set_xscale('log')
            
            markers = [1, 10, 100]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

            for i, rank in enumerate(markers):
                if rank <= len(grouped_df):
                    row = grouped_df.iloc[rank-1]
                    perc = row['Cumulative_Percentage']
                    ax.scatter(rank, perc, color=colors[i], s=50, zorder=5)
                    ax.axvline(x=rank, color=colors[i], linestyle=':', linewidth=1.2)
                    ax.axhline(y=perc, color=colors[i], linestyle=':', linewidth=1.2)
                    label_text = f"top {rank}: {perc:.2f}%"
                    xytext = (5, 5) if rank == 1 else (5, -10) if rank == 10 else (5, 5)
                    ax.annotate(label_text, xy=(rank, perc), xytext=xytext, 
                                textcoords='offset points', fontsize=10)

            ax.set_xlabel('Number of sequences (sorted by count ↓)', fontsize=12)
            ax.set_ylabel('Cumulative % of reads', fontsize=12)
            plot_title = f"Cumulative sequencing curve\nRegion: {spec}"
            ax.set_title(plot_title, fontsize=14)
            ax.set_ylim(0, 105)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xticks([1, 10, 100, 1000, 10000])
            plt.tight_layout()

            safe_spec = spec.replace('-', '_')
            plot_filename = f"{base_name}_{safe_spec}_convergence.png"
            plot_path = os.path.join(save_dir, plot_filename)
            plt.savefig(plot_path, dpi=300)
            plt.close(fig) 
            self.logger.info(f"Saved convergence plot to {plot_path}")

class EnrichmentAnalyzer:
    """
    Automatically groups libraries by name, detects rounds (R1, R2...),
    and tracks sequence enrichment across selection rounds.
    """
    def __init__(self, parser_out_path, logger):
        self.root = parser_out_path
        self.logger = logger

    def visualize_convergence(self, file_path, region_specs, output_dir=None):
        """
        Reads the annotated CSV and plots the cumulative fraction of reads 
        (rarefaction curve) to visualize library convergence, explicitly 
        marking the Top 1, Top 10, and Top 100 clones.
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        if output_dir is None:
            output_dir = os.path.dirname(file_path)
            
        try:
            df = pd.read_csv(file_path, keep_default_na=False)
        except Exception as e:
            self.logger.error(f"Could not read {file_path} for visualization: {e}")
            return
            
        base_name = os.path.basename(file_path).replace('_annotated.csv', '').replace('.csv', '')
        
        for region in region_specs:
            cols = region.split('-')
            
            missing_cols = [c for c in cols if c not in df.columns]
            if missing_cols:
                self.logger.warning(f"Skipping {region} visualization. Missing columns: {missing_cols}")
                continue
                
            valid_df = df[df[cols[0]] != ""]
            
            grouped = valid_df.groupby(cols)['Count'].sum().reset_index()
            grouped = grouped.sort_values(by='Count', ascending=False)
            
            total_reads = grouped['Count'].sum()
            if total_reads == 0:
                self.logger.warning(f"No valid reads found for {region} in {base_name}.")
                continue
                
            grouped['Fraction'] = grouped['Count'] / total_reads
            grouped['Cumulative'] = grouped['Fraction'].cumsum()
            
            plt.figure(figsize=(8, 6))
            plt.plot(np.arange(1, len(grouped) + 1), grouped['Cumulative'], color='#2ca02c', linewidth=2)
            
            plt.title(f"Clonal Convergence: {base_name}\nTarget: {region.replace('_PEP', '')}")
            plt.xlabel("Clone Rank (Log Scale)")
            plt.ylabel("Cumulative Fraction of Total Reads")
            plt.xscale('log')
            
            # Set y-axis to start exactly at 0 to make the drop-lines look clean
            plt.ylim(bottom=0, top=1.05) 
            plt.grid(True, which="both", ls="--", alpha=0.4)
            
            # Mark out Top 1, Top 10, and Top 100 with points and text
            for rank in [1, 10, 100]:
                if len(grouped) >= rank:
                    cum_val = grouped['Cumulative'].iloc[rank-1]
                    
                    # 1. Draw the actual point on the line
                    plt.plot(rank, cum_val, marker='o', color='red', markersize=6)
                    
                    # 2. Draw a vertical dotted line down to the x-axis for visual reference
                    plt.hlines(y=cum_val, xmin=0, xmax=rank, color='red', linestyle=':', alpha=0.5)
                    
                    # 3. Anchored Text Annotation (Offset so it doesn't cover the point)
                    plt.annotate(
                        f"Top {rank}: {cum_val:.1%}",
                        xy=(rank, cum_val),
                        xytext=(10, -15), # Offset 10px right, 15px down
                        textcoords='offset points',
                        color='red',
                        fontsize=9,
                        fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7) # Slight white background so the text pops over the grid
                    )

            out_plot = os.path.join(output_dir, f"{base_name}_{region.replace('-','_')}_convergence.png")
            plt.savefig(out_plot, dpi=300, bbox_inches='tight')
            plt.close()
            
        self.logger.info(f"Saved convergence plots for {base_name}")

    def auto_discover_and_analyze(self, region_specs, epsilon=1e-5, power=1.0, retention_power=1.0):
        self.logger.info("Scanning for multi-round libraries...")

        def _get_parser_root():
            for attr in (
                "parser_out",
                "parser_output",
                "parser_output_dir",
                "root_dir",
                "root",
                "output_dir",
                "out_dir",
                "base_dir",
                "data_dir",
            ):
                value = getattr(self, attr, None)
                if isinstance(value, str) and value:
                    return value
            return "."

        def _clean_sample_stem(file_path):
            stem = os.path.splitext(os.path.basename(file_path))[0]

            changed = True
            while changed:
                changed = False
                for suffix in ("_annotated", "_pep_counts"):
                    if stem.endswith(suffix):
                        stem = stem[: -len(suffix)]
                        changed = True

            return stem

        def _parse_round(sample_stem):
            m = re.search(
                r"(?:^|__)round-?R?(\d+)(?=__|_|$)",
                sample_stem,
                flags=re.IGNORECASE,
            )
            if m:
                return int(m.group(1))

            m = re.search(
                r"(?:^|_)R(\d+)(?=__|_|$)",
                sample_stem,
                flags=re.IGNORECASE,
            )
            if m:
                return int(m.group(1))

            return None

        def _parse_condition(sample_stem):
            if re.search(
                r"(?:^|__)cond-neg(?:ative)?(?=__|_|$)",
                sample_stem,
                flags=re.IGNORECASE,
            ):
                return "neg"

            if re.search(
                r"(?:^|__)cond-input(?=__|_|$)",
                sample_stem,
                flags=re.IGNORECASE,
            ):
                return "input"

            if re.search(
                r"(?:^|__)cond-pos(?:itive)?(?=__|_|$)",
                sample_stem,
                flags=re.IGNORECASE,
            ):
                return "pos"

            if _parse_round(sample_stem) is not None:
                return "pos"

            return "unknown"

        def _library_key(sample_stem):
            parts = sample_stem.split("__")
            kept = []

            for part in parts:
                p = part.strip()
                if not p:
                    continue

                p = re.sub(r"_R\d+(?=_|$)", "", p, flags=re.IGNORECASE)
                p = re.sub(r"^R\d+$", "", p, flags=re.IGNORECASE)

                if re.match(r"^round-?R?\d+$", p, flags=re.IGNORECASE):
                    continue
                if re.match(r"^cond-", p, flags=re.IGNORECASE):
                    continue
                if re.match(r"^negtype-", p, flags=re.IGNORECASE):
                    continue
                if re.match(r"^rep-", p, flags=re.IGNORECASE):
                    continue

                if p:
                    kept.append(p)

            return "__".join(kept) if kept else sample_stem

        def _round_sort_key(round_key):
            return int(str(round_key).lstrip("Rr"))

        parser_root = _get_parser_root()

        annotated_files = glob.glob(
            os.path.join(parser_root, "**", "*_annotated.csv"),
            recursive=True,
        )
        annotated_files = list(dict.fromkeys(annotated_files))

        self.logger.info(f"Found {len(annotated_files)} annotated CSV files under: {parser_root}")

        discovered_libraries = {}

        for file_path in annotated_files:
            sample_stem = _clean_sample_stem(file_path)
            round_num = _parse_round(sample_stem)
            condition = _parse_condition(sample_stem)
            lib_key = _library_key(sample_stem)

            self.logger.info(
                "Round discovery candidate | "
                f"sample='{sample_stem}' | "
                f"condition='{condition}' | "
                f"round='{round_num}' | "
                f"library_key='{lib_key}'"
            )

            if condition != "pos":
                self.logger.info(f"Skipping non-positive sample for enrichment: {sample_stem}")
                continue

            if round_num is None:
                self.logger.warning(f"Could not parse round number from annotated file: {sample_stem}")
                continue

            round_key = f"R{round_num}"

            if lib_key not in discovered_libraries:
                discovered_libraries[lib_key] = {}

            if round_key in discovered_libraries[lib_key]:
                old_path = discovered_libraries[lib_key][round_key]
                self.logger.warning(
                    f"Duplicate positive round detected for library '{lib_key}', {round_key}. "
                    f"Old file: {old_path} | New file: {file_path}. Keeping the later file."
                )

            discovered_libraries[lib_key][round_key] = file_path

        if not discovered_libraries:
            self.logger.warning(
                "No positive round files matched either the old <LIB>_R# pattern "
                "or the new FAO2 __round-R#__cond-pos pattern."
            )
            return

        libraries = {}

        for lib_key, round_map in discovered_libraries.items():
            round_map = dict(sorted(round_map.items(), key=lambda x: _round_sort_key(x[0])))
            detected = ", ".join(round_map.keys())

            if len(round_map) < 2:
                self.logger.warning(
                    f"Skipping library with fewer than 2 positive rounds: "
                    f"{lib_key} | rounds: {detected}"
                )
                continue

            libraries[lib_key] = round_map
            self.logger.info(f"Detected multi-round library: {lib_key} | rounds: {detected}")

        if not libraries:
            self.logger.warning(
                "Annotated files were found, but no library had at least two positive rounds. "
                "Enrichment analysis requires at least two positive rounds per library."
            )
            return

        # 3. Process each Library
        for lib, rounds_dict in libraries.items():
            # Sort rounds naturally: R1, R2, R10
            sorted_rounds = sorted(rounds_dict.keys(), key=lambda x: int(str(x).lstrip("Rr")))
            
            if len(sorted_rounds) < 2:
                self.logger.info(f"Skipping {lib}: Found {len(sorted_rounds)} rounds (Need >= 2).")
                continue
            
            self.logger.info(f"Analyzing Enrichment for {lib}: {sorted_rounds}")
            self._analyze_single_group(lib, sorted_rounds, rounds_dict, region_specs, epsilon, power, retention_power)

    def _analyze_single_group(self, library_name, sorted_rounds, file_map, region_specs, epsilon, power, retention_power):
        # 1. Load DataFrames
        round_dfs = {}
        for r in sorted_rounds:
            path = file_map[r]
            try:
                df = pd.read_csv(path, keep_default_na=False)
                total_reads = df['Count'].sum()
                df['_Freq'] = df['Count'] / max(total_reads, 1)
                round_dfs[r] = df
            except Exception as e:
                self.logger.error(f"Error reading {path}: {e}")
                return

        out_dir = os.path.join(self.root, "enrichment_analysis")
        if not os.path.exists(out_dir): os.makedirs(out_dir)

        # 2. Process Specs (e.g. CDR3, CDR3-FR4)
        for spec in region_specs:
            merged_data = None
            
            # Helper to handle concatenated columns
            def get_group_series(df, s):
                cols = s.split('-')
                missing = [c for c in cols if c not in df.columns]
                if missing: return None
                if len(cols) == 1: return df[cols[0]]
                combined = df[cols[0]].astype(str)
                for c in cols[1:]: combined = combined + "-" + df[c].astype(str)
                return combined

            # Aggregate Counts per Round
            freq_cols = []
            for r in sorted_rounds:
                df = round_dfs[r]
                seq_series = get_group_series(df, spec)
                if seq_series is None: continue
                
                # Recount logic
                grp = df.groupby(seq_series)['Count'].sum().reset_index()
                grp.columns = ['Sequence', f'Count_{r}']
                
                total_r = grp[f'Count_{r}'].sum()
                col_freq = f'Freq_{r}'
                grp[col_freq] = grp[f'Count_{r}'] / max(total_r, 1)
                freq_cols.append(col_freq)
                
                if merged_data is None: merged_data = grp
                else: merged_data = pd.merge(merged_data, grp, on='Sequence', how='outer')
            
            if merged_data is None: continue
            merged_data.fillna(0, inplace=True)
            
            # Retrieve Representative ID (from the latest possible round)
            seq_to_id = {}
            for r in reversed(sorted_rounds):
                df = round_dfs[r]
                s_series = get_group_series(df, spec)
                if 'ID' not in df.columns: continue
                temp = pd.DataFrame({'Seq': s_series, 'ID': df['ID']})
                temp = temp.drop_duplicates(subset='Seq')
                current_dict = dict(zip(temp['Seq'], temp['ID']))
                for k, v in current_dict.items():
                    if k not in seq_to_id: seq_to_id[k] = v
            
            merged_data['ID_Final_Round'] = merged_data['Sequence'].map(seq_to_id)
            
            # --- CALCULATE METRICS ---
            
            freq_matrix = merged_data[freq_cols].values
            
            adjusted_fcs = []
            non_zero_counts = []
            
            for i in range(len(freq_matrix)):
                row = freq_matrix[i]
                
                # 1. Find Non-Zero Rounds
                nonzero_indices = np.where(row > 0)[0]
                n_nonzero = len(nonzero_indices)
                non_zero_counts.append(n_nonzero)
                
                if n_nonzero == 0:
                    adjusted_fcs.append(0.0)
                    continue
                    
                first_nonzero_idx = nonzero_indices[0]
                last_idx = len(row) - 1 
                
                # 2. Determine Start Point (1 round before appearance, or index 0)
                if first_nonzero_idx == 0:
                    calc_start_idx = 0
                else:
                    calc_start_idx = first_nonzero_idx - 1
                
                start_freq = row[calc_start_idx]
                end_freq = row[last_idx]
                
                # 3. Calculate Enrichment (Adjusted Fold Change)
                adj_fc = (end_freq + epsilon) / (start_freq + epsilon)
                adjusted_fcs.append(adj_fc)
                
            merged_data['Enrichment_Adjusted'] = adjusted_fcs
            merged_data['Non_Zero_Rounds'] = non_zero_counts
            
            # --- FINAL SCORE CALCULATION ---
            
            # 1. Ratio in Final Round (Magnitude)
            last_r = sorted_rounds[-1]
            ratio_final = merged_data[f'Freq_{last_r}']
            
            # 2. Retention Punishment (Consistency)
            # Logic: Compare final freq to the maximum freq ever seen for this clone.
            # If it peaked in R2 and crashed in R3, this factor will be < 1.0.
            max_freq_across_rounds = merged_data[freq_cols].max(axis=1)
            retention_factor = (ratio_final + epsilon) / (max_freq_across_rounds + epsilon)
            
            merged_data['Retention_Factor'] = retention_factor
            
            # 3. Apply Formula
            merged_data['Enrichment_Score'] = (
                ratio_final * (retention_factor ** retention_power) * (merged_data['Enrichment_Adjusted'] ** power)
            )
            
            merged_data.sort_values(by='Enrichment_Score', ascending=False, inplace=True)
            
            # Reorder columns for output
            cols = [
                'Sequence', 'ID_Final_Round', 'Enrichment_Score', 
                'Enrichment_Adjusted', 'Retention_Factor', 'Non_Zero_Rounds'
            ]
            for r in sorted_rounds:
                cols.append(f'Count_{r}')
                cols.append(f'Freq_{r}')
                
            merged_data = merged_data[cols]
            
            safe_spec = spec.replace('-', '_')
            out_name = f"{library_name}_{safe_spec}_enrichment.csv"
            out_path = os.path.join(out_dir, out_name)
            merged_data.to_csv(out_path, index=False)
            self.logger.info(f"Saved enrichment (Ratio-Weighted Mode): {out_name}")