# -*- coding: utf-8 -*-
"""
@author: Alex Vinogradov
Modified by Jinxuan ZHAO
"""
"""
Core code for sequence processing parsers
"""

import time, os, logging, gzip, re, inspect, copy
import multiprocessing 
from utils import Plotter
from utils.datatypes import Data, SequencingSample
from utils.constants import constants 

import numpy as np
import pandas as pd
import numba as nb
from collections import Counter
import shutil

# Import ANARCI
try:
    from anarci import anarci
    ANARCI_AVAILABLE = True
except ImportError:
    ANARCI_AVAILABLE = False

# --- CONSTANTS FOR IMGT NUMBERING ---
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

_DNA_TO_INT = np.full(128, 4, dtype=np.int8) 
_DNA_TO_INT[ord('A')] = 0
_DNA_TO_INT[ord('C')] = 1
_DNA_TO_INT[ord('G')] = 2
_DNA_TO_INT[ord('T')] = 3

_COMP_LUT = np.full(128, 0, dtype=np.uint8)
for a, b in zip(b"ACGTN", b"TGCAN"):
    _COMP_LUT[a] = b

_CODON_LUT = np.full((4, 4, 4), ord('X'), dtype=np.uint8) 

def _fill_codon_table_from_constants():
    table = constants.codon_table
    for codon, aa in table.items():
        i1 = _DNA_TO_INT[ord(codon[0])]
        i2 = _DNA_TO_INT[ord(codon[1])]
        i3 = _DNA_TO_INT[ord(codon[2])]
        _CODON_LUT[i1, i2, i3] = ord(aa)

_fill_codon_table_from_constants()

@nb.njit(fastmath=True)
def _nb_dna_to_pep(dna_bytes, stop_readthrough):
    n = len(dna_bytes)
    n_codons = n // 3
    out = np.empty(n_codons, dtype=np.uint8)
    
    out_idx = 0
    for i in range(n_codons):
        base_idx = i * 3
        b1 = dna_bytes[base_idx]
        b2 = dna_bytes[base_idx+1]
        b3 = dna_bytes[base_idx+2]
        
        if b1 < 128 and b2 < 128 and b3 < 128:
            i1 = _DNA_TO_INT[b1]
            i2 = _DNA_TO_INT[b2]
            i3 = _DNA_TO_INT[b3]
            if i1 > 3 or i2 > 3 or i3 > 3:
                aa = 43
            else:
                aa = _CODON_LUT[i1, i2, i3]
        else:
            aa = 43 

        if aa == 42:
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

# --- CLASS DEFINITIONS ---

class Logger:
    def __init__(self, config=None):
        self.conf = config
        self.__fallback()
        self.__configure_logger()

    def __fallback(self):
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

        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        if self.log_to_file and self.log_fname:
            log_dir = os.path.dirname(self.log_fname)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            filehandler = logging.FileHandler(self.log_fname)      
            filehandler.setFormatter(formatter)
            self.logger.addHandler(filehandler)       

class DirectoryTracker:
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
        for d in [x for x in dir(self) if not x.startswith('_')]:
            path = getattr(self, d)
            if isinstance(path, str) and not os.path.isdir(path):
                os.makedirs(path)

class Handler:
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
    def __init__(self, *args):
        super(Pipeline, self).__init__(*args)
        self._on_startup()
        super(Pipeline, self)._on_completion()

    def _on_startup(self):
        self.que = []
        if not hasattr(self, 'exp_name'):
            self.exp_name = 'unnamed'

    def _describe_data(self, data=None):
        """
        Detailed logging of dataset statistics for debugging.
        Logs Count and Max Length for DNA, PEP, and Q.
        Returns just the count for quick tracking.
        """
        if data is None: 
            return 0
        
        total_count = 0
        for sample in data:
            count = len(sample)
            total_count += count
            
            def get_stats(arr, name):
                if arr is None: return "None"
                if hasattr(arr, 'size') and arr.size == 0: return "Empty (0)"
                if isinstance(arr, list) and len(arr) == 0: return "Empty (0)"

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
        for func in routines:
            self.que.append(func)
        self.logger.info(f'{len(routines)} routines appended to pipeline.')

    def run_over_stream(self, data_iter_factory, save_summary=True):
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
            
            for func in self.que:
                t0 = time.time()
                op_name = func.__name__
                self.logger.info(f"> Running {op_name}...")
                data = func(data)
                op_time = time.time() - t0
                new_count = self._describe_data(data)
                dropped = current_count - new_count
                
                summary.append({
                    'Chunk_ID': chunk_idx, 
                    'Sample': sample_name,
                    'Operation': op_name,
                    'Time(s)': round(op_time, 3), 
                    'Input_Sequences': current_count,
                    'Dropped_Sequences': dropped,
                    'Remaining_Sequences': new_count
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
            total_row = {
                'Chunk_ID': 'TOTAL',
                'Sample': 'ALL_FILES',
                'Operation': 'ALL_STEPS',
                'Time(s)': round(total_elapsed, 3),
                'Input_Sequences': total_input_reads,
                'Dropped_Sequences': total_dropped,
                'Remaining_Sequences': total_final_reads
            }
            df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
            fname = f'{self.exp_name}_streaming_summary.csv'
            path = os.path.join(self.dirs.logs, fname)
            df.to_csv(path, index=False)
            self.logger.info(f"Summary saved to {path}")
            
        return None

    def merge_chunk_outputs(self, delete_chunks=True):
        root = self.dirs.parser_out
        if not os.path.exists(root): 
            self.logger.warning(f"Parser output directory not found: {root}")
            return
        
        chunk_dirs = [d for d in os.listdir(root) if "__chunk" in d and os.path.isdir(os.path.join(root, d))]
        if not chunk_dirs: 
            self.logger.info("No chunk directories found to merge.")
            return
        
        groups = {}
        for d in chunk_dirs:
            try:
                base = d.split('__chunk')[0]
                if base not in groups: groups[base] = []
                groups[base].append(d)
            except Exception as e:
                self.logger.error(f"Failed to parse chunk directory name '{d}'. Skipping. Error: {e}")
            
        for base_name, c_dirs in groups.items():
            dest_dir = os.path.join(root, base_name)
            if not os.path.exists(dest_dir): os.makedirs(dest_dir)
            
            first_chunk_path = os.path.join(root, c_dirs[0])
            if not os.path.isdir(first_chunk_path): continue

            try:
                files = [f for f in os.listdir(first_chunk_path) if os.path.isfile(os.path.join(first_chunk_path, f))]
            except: continue
                
            if not files: continue

            suffix_map = {}
            for fname in files:
                if fname.startswith(c_dirs[0]) and fname.endswith('.csv'):
                    suffix = fname[len(c_dirs[0]):]
                    suffix_map[suffix] = fname 
            
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
                                seq = row['Seq']
                                count = int(row['Count'])
                                if seq: 
                                    merged_counts[seq] += count
                                    if 'DNA' in df.columns and row['DNA']:
                                        if seq not in merged_dna:
                                            merged_dna[seq] = row['DNA']
                    except Exception as e: 
                        self.logger.error(f"Error merging CSV {src_path}: {e}")
                
                if not merged_counts: continue
                    
                final_csv_name = base_name + suffix
                final_fasta_name = final_csv_name.replace('.csv', '.fasta')
                
                path_csv = os.path.join(dest_dir, final_csv_name)
                path_fasta = os.path.join(dest_dir, final_fasta_name)
                
                sorted_items = sorted(merged_counts.items(), key=lambda x: (-x[1], x[0]))
                
                final_data = []
                for rank, (seq, count) in enumerate(sorted_items, start=1):
                    dna = merged_dna.get(seq, "")
                    final_data.append({
                        'Seq': seq,
                        'Count': count,
                        'DNA': dna
                    })
                df_final = pd.DataFrame(final_data)
                df_final.to_csv(path_csv, index=False)
                self.logger.info(f"Saved merged CSV to {path_csv}")
                
                with open(path_fasta, 'w') as f:
                    for rank, (seq, count) in enumerate(sorted_items, start=1):
                        f.write(f">seq_{rank}_count_{count}\n{seq}\n")
                self.logger.info(f"Saved merged FASTA to {path_fasta}")
                            
            self.logger.info(f"Merged {base_name}")

            if delete_chunks:
                for c_dir in c_dirs:
                    try: shutil.rmtree(os.path.join(root, c_dir))
                    except: pass

class FastqParser(Handler):
    
    def __init__(self, *args):
        super(FastqParser, self).__init__(*args)
        self._validate() 
        super(FastqParser, self)._on_completion()

    def _validate(self):
        if not (hasattr(self, 'P_design') and hasattr(self, 'D_design')):
             pass

    def _transform_check(self, sample, func):
        if not sample.get_ndims() == 2:
            try: sample.transform()
            except: pass
        return

    # --- INPUT OPS ---
    
    def stream_chunks_from_gz_dir(self, chunk_lines=100000):
        """
        Pass specified lines of fastq data to the pipeline to save resource.DEFAULT=100000 (25000)
        """
        fnames = [os.path.join(self.dirs.seq_data, x) for x in os.listdir(self.dirs.seq_data) if x.endswith(".gz")]
        def _iter():
            for f in fnames:
                basename = os.path.basename(f).split('.')[0]
                with gzip.open(f, 'rt') as fh:
                    lines = []
                    chunk_id = 0
                    for line in fh:
                        lines.append(line)
                        if len(lines) >= chunk_lines:
                            d = [x.strip() for x in lines[1::4]]
                            q = [x.strip() for x in lines[3::4]]
                            chunk_id += 1
                            name = f"{basename}__chunk{chunk_id}"
                            s = SequencingSample(D=d, Q=q, P=None, name=name)
                            yield Data(samples=[s])
                            lines = []
                    if lines:
                         d = [x.strip() for x in lines[1::4]]
                         q = [x.strip() for x in lines[3::4]]
                         chunk_id += 1
                         name = f"{basename}__chunk{chunk_id}"
                         s = SequencingSample(D=d, Q=q, P=None, name=name)
                         yield Data(samples=[s])
        return _iter
    
    # --- PROCESSING OPS ---
    def translate_both_strands(self, *, stop_readthrough=False, utr5_offset=0, tol=0):
        """
        Correct Sequencing direction and find transation starting position. Also trim the DNA to UTR5'-3'.
        stop_readthrooug: [Boolean] False to stop the translation when encounter stop codon, True when force the translation continue to the 3' end. stop codon → *; non-trimer oligo → _.
        utr5_offset: [int] 0 for start transation at 5' of UTR5. ("Untranslated" region 5') Set to len(UTR5)to start translation right after it. 
        tol: [int] Tolerable mutation number in UTR5 and BARCODE for matching. 
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

                    bc_idx = _match_block_fuzzy(seq_arr, bc_arr, tol)
                    found_fwd = False
                    if bc_idx != -1:
                        utr_idx = _match_block_fuzzy(seq_arr, utr_arr, tol)
                        if utr_idx != -1:
                            found_fwd = True
                            start = utr_idx + utr5_offset
                            if start < 0: start = 0
                            coding_seq = seq_b[start:]
                            q_slice = q_raw[start:] if q_raw is not None else None
                            pep_arr = _nb_dna_to_pep(np.frombuffer(coding_seq, dtype=np.uint8), stop_readthrough)
                            new_D.append(coding_seq.decode('ascii'))
                            new_Q.append(q_slice)
                            new_P.append(pep_arr.tobytes().decode('ascii'))
                            
                    if found_fwd: continue
                    
                    rc_arr = _nb_revcom(seq_arr)
                    rc_bytes = rc_arr.tobytes()
                    bc_idx_rc = _match_block_fuzzy(rc_arr, bc_arr, tol)
                    if bc_idx_rc != -1:
                        utr_idx_rc = _match_block_fuzzy(rc_arr, utr_arr, tol)
                        if utr_idx_rc != -1:
                            start = utr_idx_rc + utr5_offset
                            if start < 0: start = 0
                            coding_seq = rc_bytes[start:]
                            q_slice = q_raw[::-1][start:] if q_raw is not None else None
                            pep_arr = _nb_dna_to_pep(np.frombuffer(coding_seq, dtype=np.uint8), stop_readthrough)
                            new_D.append(coding_seq.decode('ascii'))
                            new_Q.append(q_slice)
                            new_P.append(pep_arr.tobytes().decode('ascii'))

                sample.D = np.array(new_D, dtype=object)
                sample.P = np.array(new_P, dtype=object)
                if sample.Q is not None:
                    sample.Q = np.array(new_Q, dtype=object)
                
                sample.transform()
            return data
        
        _op.__name__ = f"translate_both_strands_tol{tol}"
        return _op

    def translate_all_6_frames(self, stop_readthrough=False):
        """
        Seldomly used parser for sequences with no fixed UTR5 or BARCODE.
        stop_readthrooug: [Boolean] False to stop the translation when encounter stop codon, True when force the translation continue to the 3' end. stop codon → *; non-trimer oligo → _.
        """
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
                    
                    for f in range(3):
                        if len(d_arr) > f:
                            pep = _nb_dna_to_pep(d_arr[f:], stop_readthrough)
                            P_all.append(pep.tobytes().decode('ascii'))
                            D_all.append(d_arr.tobytes().decode('ascii'))
                            Q_all.append(None) 
                    
                    rc_arr = _nb_revcom(d_arr)
                    for f in range(3):
                        if len(rc_arr) > f:
                            pep = _nb_dna_to_pep(rc_arr[f:], stop_readthrough)
                            P_all.append(pep.tobytes().decode('ascii'))
                            D_all.append(rc_arr.tobytes().decode('ascii'))
                            Q_all.append(None)

                new_s = SequencingSample(
                    name=f"{sample.name}_6frames",
                    D=D_all, P=P_all, Q=Q_all if any(x is not None for x in Q_all) else []
                )
                new_s.D = np.array(new_s.D, dtype=object)
                new_s.P = np.array(new_s.P, dtype=object)
                new_s.transform()
                new_samples.append(new_s)
            
            return Data(samples=new_samples)
        
        _op.__name__ = "translate_all_6_frames"
        return _op

    def len_filter(self, where='dna', len_range=None):
        """
        Filter out DNAs that extremely short (empty vector) or long (more than two virable regions).
        len_range :[int - int] range of allowed DNA length.
        """
        if len_range is None: raise ValueError("len_range must be specified [min, max]")
        min_l, max_l = len_range[0], len_range[1]
        def _op(data):
            for sample in data:
                self._transform_check(sample, inspect.stack()[0][3])
                arr = sample[where]
                if arr.dtype.kind in ('S', 'U'):
                    pad = b'' if arr.dtype.kind == 'S' else ''
                    lengths = (arr != pad).sum(axis=1)
                else:
                    lengths = np.array([len(str(x)) for x in arr])
                mask = (lengths >= min_l) & (lengths < max_l)
                sample(mask)
            return data
        
        _op.__name__ = f"len_filter_{where}"
        return _op

    def q_score_filt(self, minQ=30, frac=0.9):
        """
        Filter out DNAs that contains low-quality base.
        minQ:[int] minimum of Q score. DEFAULT = 30 (99.9% accuracy).
        frac: [double] fraction of bases need to exceed minQ. DEFAULT = 0.9 (90%)
        """
        def _op(data):
            for sample in data:
                self._transform_check(sample, inspect.stack()[0][3])
                if sample.Q is None: continue
                if sample.Q.dtype == object: sample.transform_Q()
                mask = _nb_q_filter(sample.Q, minQ, frac)
                sample(mask)
            return data
        
        _op.__name__ = f"q_score_filt_Q{minQ}"
        return _op

    def extract_fuzzy_regions(self, *, where='pep', loc=None, tol=0):
        """
        Extract wanted region based on peptide design. 
        loc: [int, list] location to be extracted.
        tol: [int] tolerable AA mutation during motif matching. 
        """
        design = self.P_design if where == 'pep' else self.D_design
        if loc is None: raise ValueError("loc must be specified")        
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
                arr = sample[where]
                n_rows = arr.shape[0]
                
                new_D, new_P, new_Q = [], [], []
                
                for i in range(n_rows):
                    row_data = arr[i]
                    if row_data.dtype.kind == 'S':
                        row_bytes = row_data.tobytes().strip(b'\x00').strip(b'')
                    else:
                        row_bytes = "".join(row_data).encode('ascii')

                    best_match_seq = None
                    extract_start_index = 0
                    extract_len = 0
                    
                    for t_idx, anchors in enumerate(anchors_list):
                        cursor = 0
                        valid_template = True
                        temp_extract_parts = []
                        current_parts_len = 0
                        first_vr_start = -1 
                        
                        for target_idx in loc:
                            if not design.templates[t_idx].is_vr[target_idx]:
                                motif = anchors[target_idx]
                                pos = _match_block_fuzzy(row_bytes[cursor:], motif, tol)
                                if pos != -1:
                                    abs_pos = cursor + pos
                                    if first_vr_start == -1: first_vr_start = abs_pos
                                    part = row_bytes[abs_pos : abs_pos+len(motif)]
                                    temp_extract_parts.append(part)
                                    current_parts_len += len(part)
                                    cursor = abs_pos + len(motif)
                                else:
                                    valid_template = False; break
                            else:
                                if first_vr_start == -1: first_vr_start = cursor
                                next_cr_idx = target_idx + 1
                                if next_cr_idx in anchors:
                                    right_motif = anchors[next_cr_idx]
                                    pos = _match_block_fuzzy(row_bytes[cursor:], right_motif, tol)
                                    if pos != -1:
                                        abs_pos_right = cursor + pos
                                        part = row_bytes[cursor : abs_pos_right]
                                        temp_extract_parts.append(part)
                                        current_parts_len += len(part)
                                        cursor = abs_pos_right 
                                    else:
                                        valid_template = False; break
                                else:
                                    part = row_bytes[cursor:]
                                    temp_extract_parts.append(part)
                                    current_parts_len += len(part)
                                    cursor = len(row_bytes)
                        
                        if valid_template:
                            best_match_seq = b"".join(temp_extract_parts)
                            extract_start_index = first_vr_start
                            extract_len = current_parts_len
                            break 
                            
                    if best_match_seq is not None:
                        decoded_seq = best_match_seq.decode('ascii')
                        p_start = extract_start_index
                        p_end = extract_start_index + extract_len
                        
                        if sample.D[i].dtype.kind == 'S':
                            d_full = sample.D[i].tobytes().strip(b'\x00').decode('ascii')
                        else:
                            d_full = "".join(sample.D[i])
                            
                        d_slice = d_full[p_start*3 : p_end*3]
                        
                        if where == 'pep':
                            new_P.append(decoded_seq)
                            new_D.append(d_slice)
                            if sample.Q is not None: new_Q.append(sample.Q[i]) 
                        else:
                            new_D.append(decoded_seq)
                            new_P.append(sample.P[i])
                            
                sample.D = np.array(new_D, dtype=object)
                sample.P = np.array(new_P, dtype=object)
                
                if sample.Q is not None:
                    if len(new_Q) == len(new_P):
                         sample.Q = np.array(new_Q, dtype=object)
                    else:
                         sample.Q = np.empty(len(new_P), dtype=object)
                
                sample.transform()
            return data
        
        _op.__name__ = f"extract_fuzzy_regions_{where}"
        return _op

    # --- OUTPUT---

    def unpad(self):
        """
        Unpad data to readable format.
        """        
        def _op(data):
            for sample in data: sample.unpad()
            return data
        _op.__name__ = "unpad"
        return _op
    
    def count_summary(self, where='pep', fmt='csv'):
        """
        Combine identical peptides, synonym mutation DNAs are ignored and unified to most enriched one.
        """ 
        def _op(data):
            for sample in data:
                arr = sample[where]
                rows = []
                dnas = []
                for x in arr:
                    if x.dtype.kind == 'S': rows.append(x.tobytes().decode('ascii').strip('\x00'))
                    else: rows.append("".join(x))
                for x in sample.D:
                    if x.dtype.kind == 'S': dnas.append(x.tobytes().decode('ascii').strip('\x00'))
                    else: dnas.append("".join(x))
                
                counter = Counter()
                key_to_dna = {}
                for k, d in zip(rows, dnas):
                    counter[k] += 1
                    if k not in key_to_dna: key_to_dna[k] = d
                
                dest = os.path.join(self.dirs.parser_out, sample.name, sample.name + f'_{where}_counts.{fmt}')
                chunk_output_dir = os.path.join(self.dirs.parser_out, sample.name)
                if not os.path.exists(chunk_output_dir): os.makedirs(chunk_output_dir)
                    
                if fmt == 'csv':
                    items = []
                    for k, cnt in counter.most_common():
                        items.append({'Seq': k, 'Count': cnt, 'DNA': key_to_dna[k]})
                    pd.DataFrame(items).to_csv(dest, index=False)
                elif fmt == 'fasta':
                    with open(dest, 'w') as f:
                        for i, (seq, cnt) in enumerate(counter.most_common()):
                            f.write(f">seq_{i}_count_{cnt}\n{seq}\n")
            return data
        _op.__name__ = f"count_summary_{where}_{fmt}"
        return _op
         

class AnarciAnnotator:
    """
    Post-processing class to annotate, split, and re-rank sequences using ANARCI.
    """
    def __init__(self, output_dir, logger):
        self.output_dir = output_dir
        self.logger = logger
        if not ANARCI_AVAILABLE:
            self.logger.warning("ANARCI is not installed or importable. AnarciAnnotator will fail if run.")

    def process_merged_csv(self, file_path, extract_regions=None, output_mode='clean', n_cpu=1, allowed_species=None, assign_germline=False):
        """
        Reads a merged CSV, runs ANARCI, extracts requested regions (IMBGT), 
        creates new columns, re-counts, and saves output.
        """
        if extract_regions is None:
            self.logger.info(f"No region specified, extract full-length VL and VH.")
            extract_regions = ['FL']

        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return

        # Start timer for stats logging
        start_time = time.time()
        self.logger.info(f"Starting ANARCI annotation for {file_path}...")
        
        try:
            df = pd.read_csv(file_path, keep_default_na=False, na_values=[''])
        except Exception as e:
            self.logger.error(f"Failed to read CSV: {e}")
            return

        input_rows = len(df)
        if 'Seq' not in df.columns or 'DNA' not in df.columns or 'Count' not in df.columns:
            self.logger.error("CSV missing required columns (Seq, DNA, Count).")
            return

        sequences = [(str(i), row['Seq']) for i, row in df.iterrows()]
        
        if n_cpu <= 1: n_cpu = multiprocessing.cpu_count()
        self.logger.info(f"ANARCI running on {n_cpu} cores.")
        
        try:
            # call ANARCI
            kwargs = {'scheme': 'imgt', 'output': False, 'ncpu': n_cpu, 'assign_germline': assign_germline}
            if allowed_species:
                kwargs['allowed_species'] = allowed_species
                
            numbering, alignment_details, hit_tables = anarci(sequences, **kwargs)
        except Exception as e:
            self.logger.error(f"ANARCI run failed: {e}")
            return
                        
        # DEBUG LINES TO PRINT OUT ANARCI OUTPUTS
        #if alignment_details and len(alignment_details) > 0:
        #    first_align = alignment_details[0]
        #    if first_align:
        #        # first_align is a list of dicts (one per domain)
        #        if len(first_align) > 0:
        #            domain_meta = first_align[0]
        #            self.logger.info(f"DEBUG: alignment_details[0] keys: {domain_meta.keys()}")
        #            if 'germlines' in domain_meta:
        #                self.logger.info(f"DEBUG: germlines content: {domain_meta['germlines']}")

        processed_rows = []
        
        for i, (num, align, hits) in enumerate(zip(numbering, alignment_details, hit_tables)):
            original_row = df.iloc[i]
            count = original_row['Count']
            full_dna = original_row['DNA'] 
            full_pep = original_row['Seq'] 
            
            if not num: continue
            
            # Extract regions
            def get_region_seq(residue_list, region_def, query_start_idx):
                valid_query_indices = []
                pep_res = []
                current_query_pos = query_start_idx
                
                for entry in residue_list:
                    # Get residue from ANARCI annotation.
                    # entry: ((1, ' '), 'D')
                    try:
                        # Index [0][0] is the IMGT number (integer)
                        imgt_idx = int(entry[0][0])
                    except: continue
                        
                    residue = entry[1]
                    is_gap = (residue == '-')
                    
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
                            
                pep_str = "".join(pep_res)
                
                dna_str = ""
                if valid_query_indices:
                    dna_parts = []
                    for q_idx in valid_query_indices:
                        start_bp = q_idx * 3
                        end_bp = start_bp + 3
                        if end_bp <= len(full_dna):
                            dna_parts.append(full_dna[start_bp:end_bp])
                    dna_str = "".join(dna_parts)
                    
                return pep_str, dna_str

            if not align: continue
            
            row_dict = {'Count': count}
            has_valid_domain = False
            
            for domain_idx, domain_meta in enumerate(align):
                chain_type = domain_meta.get('chain_type', 'X')
                # Unify light chain types, K "kappa", L "lambda" to "L"
                if chain_type == 'K': chain_type = 'L'
                
                domain_start_idx = domain_meta['query_start']
                
                # --- GERMLINE EXTRACTION ---
                # Extract 'v_gene' and 'identity_species' if assign_germline is True.
                if assign_germline and 'germlines' in domain_meta:
                    germs = domain_meta['germlines']
                    
                    # 1. Extract V Gene Name
                    if 'v_gene' in germs and germs['v_gene']:
                        # Expected: e.g., 'v_gene': [('mouse', 'IGKV3-4*01'), 0.9895833333333334]
                        v_hit = germs['v_gene'][0]
                        row_dict[f"{chain_type}_V_Gene"] = v_hit[1]
                    
                    # 2. Extract J Gene Name
                    if 'j_gene' in germs and germs['j_gene']:
                        # Expected: e.g., 'j_gene': [('mouse', 'IGKJ2*02'), 0.9166666666666666]
                        j_hit = germs['j_gene'][0]
                        row_dict[f"{chain_type}_J_Gene"] = j_hit[1]
                        
                    # 3. Extract Species
                    if 'species' in domain_meta:
                         row_dict[f"{chain_type}_Species"] = domain_meta['species']

                if domain_idx < len(num):
                    domain_obj = num[domain_idx]
                    domain_residues = None
                    
                    if isinstance(domain_obj, list):
                        domain_residues = domain_obj
                    elif isinstance(domain_obj, tuple):
                        if isinstance(domain_obj[0], list):
                             domain_residues = domain_obj[0]
                        else:
                             for item in domain_obj:
                                if isinstance(item, list):
                                    domain_residues = item
                                    break
                    
                    if domain_residues is None: continue
                    
                    for req in extract_regions:
                        p_seq = ""
                        d_seq = ""
                        
                        if req == 'FL':
                            p_seq, d_seq = get_region_seq(domain_residues, (1, 128), domain_start_idx)
                        elif req == 'CDRs':
                            p1, d1 = get_region_seq(domain_residues, IMGT_DEFINITIONS['CDR1'], domain_start_idx)
                            p2, d2 = get_region_seq(domain_residues, IMGT_DEFINITIONS['CDR2'], domain_start_idx)
                            p3, d3 = get_region_seq(domain_residues, IMGT_DEFINITIONS['CDR3'], domain_start_idx)
                            p_seq = p1 + p2 + p3
                            d_seq = d1 + d2 + d3
                        elif req in IMGT_DEFINITIONS:
                            p_seq, d_seq = get_region_seq(domain_residues, IMGT_DEFINITIONS[req], domain_start_idx)
                        elif '-' in req:
                            try:
                                s, e = map(int, req.split('-'))
                                p_seq, d_seq = get_region_seq(domain_residues, (s, e), domain_start_idx)
                            except: pass
                        
                        if p_seq:
                            row_dict[f"{chain_type}_{req}_PEP"] = p_seq
                            row_dict[f"{chain_type}_{req}_DNA"] = d_seq
                            has_valid_domain = True
            
            if has_valid_domain:
                processed_rows.append(row_dict)

        if not processed_rows:
            self.logger.warning("No domains annotated by ANARCI.")
            return

        res_df = pd.DataFrame(processed_rows)
        res_df.fillna('', inplace=True)
        
        # --- GROUPING BASED ON PEP AND GERMLINE (IF ASSIGN_GERMLINE = TRUE)---
        # 1. Group logic: All extracted _PEP columns AND Germline columns
        # (Ignore DNA columns from grouping to combine synonymous, only the most abundant DNA is kept)
        
        pep_cols = [c for c in res_df.columns if c.endswith('_PEP')]
        germ_cols = [c for c in res_df.columns if c.endswith('_Gene') or c.endswith('_Species')]
        group_keys = pep_cols + germ_cols
        
        dna_cols = [c for c in res_df.columns if c.endswith('_DNA')]
        
        if not group_keys:
            self.logger.warning("No peptide/germline columns to group by.")
            return
        
        agg_dict = {'Count': 'sum'}
        for dc in dna_cols:
            agg_dict[dc] = 'first' 
            
        grouped = res_df.groupby(group_keys).agg(agg_dict).reset_index()
        
        grouped.sort_values(by='Count', ascending=False, inplace=True)
        
        base_name = os.path.basename(file_path).replace('.csv', '')
        if '_pep_counts' in base_name: base_name = base_name.replace('_pep_counts', '')
            
        grouped.insert(0, 'ID', [f"{base_name}_{i:06d}" for i in range(1, len(grouped) + 1)])
        
        out_name = file_path.replace('.csv', '_annotated.csv')
        grouped.to_csv(out_name, index=False)
        self.logger.info(f"Saved annotated file to {out_name}")
        
        # --- LOG SUMMARY ---
        elapsed = time.time() - start_time
        output_rows = len(grouped)
        self.logger.info("="*60)
        self.logger.info(f"ANARCI ANNOTATION COMPLETE: {os.path.basename(file_path)}")
        self.logger.info(f"Total Time:   {elapsed:.2f}s")
        self.logger.info(f"Total Input:  {input_rows} sequences")
        self.logger.info(f"Total Output: {output_rows} annotated groups")
        self.logger.info("="*60)