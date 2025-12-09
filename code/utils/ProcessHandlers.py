# -*- coding: utf-8 -*-
"""
Created on Fri May 21 20:28:10 2021
@author: Alex Vinogradov
"""

import time, os, logging, gzip, re, inspect, copy
from utils import Plotter
from utils.datatypes import Data, SequencingSample

import numpy as np
import pandas as pd

import numba as nb
# Build global lookup table (Numba friendly)

_AA_ALPHABET = np.asarray(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ+*_"), dtype="<U1")

_AA2INT_LUT = np.full(128, -1, dtype=np.int8)  # for ASCII 0-127

for i, c in enumerate(_AA_ALPHABET):

    _AA2INT_LUT[ord(c)] = i



@nb.njit

def encode_string(s):

    out = np.empty(len(s), dtype=np.int8)

    for i in range(len(s)):

        ch = s[i]

        code = ord(ch)

        if code >= 128 or _AA2INT_LUT[code] == -1:

            raise ValueError(f"Unknown character {ch}")

        out[i] = _AA2INT_LUT[code]

    return out



@nb.njit

def match_block(read, block, tol):

    n = read.size

    m = block.size

    for i in range(n - m + 1):

        err = 0

        for j in range(m):

            if read[i + j] != block[j]:

                err += 1

                if err > tol:

                    break

        if err <= tol:

            return True

    return False
    

class Logger:
    '''
    A decorated version of the standard python logger object.
    Can be setup from a config file. Two main customizations
    are implemented: verbosity (whether logger messages should be
    printed to the running stream) and log_to_file, which
    if set, will setup a dedicated handler to dump log info
    to a file.
    '''
    
    def __init__(self, config=None):
        self.conf = config
    
        self.__fallback()
        self.__configure_logger()
        return

    def __repr__(self):
        return f'<Logger {self.name}; verbose: {self.verbose}; log_to_file: {self.log_to_file}>'

    def __fallback(self):
        '''
        If no config is passed fallback to some innocuous defaults,
        which is basically a silent logger.
        '''
        
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
        
        return

    def __configure_logger(self):

        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s]: %(message)s")

        #clear any preexisting handlers to avoid stream duplication
        self.logger.handlers.clear()

        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(logging.INFO)

            self.logger.addHandler(console_handler)

        if self.log_to_file:
               
            filehandler = logging.FileHandler(self.log_fname)      
            filehandler.setFormatter(formatter)
            filehandler.setLevel(logging.INFO)
                    
            self.logger.addHandler(filehandler)       
        
        return
        
class DirectoryTracker:
    '''
    A simple object to keep track where the data,  
    logs, etc should belooked for. Config should be
    passed to customize the directories. Otherwise,
    everything will be looked up and dumped into cwd.
    '''
    
    def __init__(self, config=None):

        self._conf = config
        self.__fallback()
        self.__setup_dirs()

    def __repr__(self):
        return '<DirectoryTracker object>'

    def __fallback(self):
        '''
        If no config is passed, fallback to some preset defaults.
        At present, all directories will be set to cwd if config
        is not specified.
        '''
        cwd = os.getcwd()
        if self._conf is None:
            
            self.seq_data = cwd
            self.logs = cwd
            self.parser_out = cwd
            
        else:
            self.seq_data = self._conf.seq_data
            self.logs = self._conf.logs
            self.parser_out = self._conf.parser_out
            
        return
            
    def __setup_dirs(self):
                   
        for d in [x for x in dir(self) if not x.startswith('_')]:
            if not os.path.isdir(getattr(self, d)):
                os.makedirs(getattr(self, d))
        return        
    
class Handler:
    '''
    Base handler class. Should not be invoked directly.
    '''
    
    def __init__(self, *args):
        self.__dict__.update(*args)
        self.__logger_fallback()
        self.__tracker_fallback()
        return

    def __logger_fallback(self):
        '''
        If no logger is passed to a data handler, a default Logger 
        object will be invoked. The default logger is silent. 
        '''
        
        if not hasattr(self, 'logger'):
            self.logger = Logger().logger
    
        if self.logger is None:
            self.logger = Logger().logger
            
        return
    
    def __tracker_fallback(self):
        '''
        If no DirectorTracker object was passed to a handler, a default
        tracker will be invoked (everything in the cwd)
        '''        
        
        if not hasattr(self, 'dirs'):
            self.dirs = DirectoryTracker()
            pass
        
        return
        
    def _on_completion(self):
        msg = f'The following handler was succesfully initialized: {self}'
        self.logger.info(msg)
        return

class Pipeline(Handler):
    
    def __init__(self, *args):
        super(Pipeline, self).__init__(*args)
        self._on_startup()
        
        super(Pipeline, self)._on_completion()
        return

    def __repr__(self):
        return f'<Pipeline object; current queue size: {len(self.que)} routine(s)>'

    def _on_startup(self):
        self.que = []
        if not hasattr(self, 'exp_name'):
            self.exp_name = 'unnamed'
        return

    def _describe_data(self, data=None):
        '''
        Go over every dataset for every sample and
        log all array shapes. Used during dequeing
        to keep track of data flows.
        '''
        data_descr = []
        
        if data is None:
            return data_descr
        
        for sample in data:
            
            data_descr.append((sample.name, len(sample)))
            
            for tup in sample:
                
                if tup[0].shape:
                    shape = tup[0].shape
                else:
                    shape = None
                
                msg = f'{sample.name} {tup[1]} dataset shape: {shape}'
                self.logger.info(msg)
                
        msg = 65 * '-'
        self.logger.info(msg)
    
        return data_descr

    def _reassemble_summary(self, summary):
        
        ops = []
        times = []
        samples = []
        
        #code below is a mess, but the task is trivial, 
        #so whatever; fix if nothing better to do
        for x in summary:
            ops.append(x['op'])
            times.append(x['op_time'])
            for j in x['data_description']:
                samples.append(j[0])
        
        samples = list(set(samples))
        sizes = np.zeros((len(summary), len(samples)))
        for i,entry in enumerate(summary):
            for tup in entry['data_description']:
                for j, name in enumerate(samples):
                    if tup[0] == name:
                        sizes[i,j] = tup[1]
        
        df = pd.DataFrame(columns=['time'] + samples, index=ops)
        df['time'] = times
        for i,name in enumerate(samples):
            df[name] = sizes[:,i]
        
        return df
    
    def enque(self, routines):
        '''
        Takes a list of functions and adds them to the pipeline queue.
        self.deque will take some data as an argument and apply dump
        the queue on it, i.e. sequentially transform the data by applying
        the queued up routines. 

        Parameters
        ----------
        routines : a list of functions capable of acting on data.
                   every routine should take data as the only argument
                   and return transformed data in the same format (Data object)

        Returns
        -------
        None.

        '''
        
        for func in routines:
            self.que.append(func)
            
        msg = f'{len(routines)} routines appended to pipeline; current queue size: {len(self.que)}'
        self.logger.info(msg)
        
        return        

    def run(self, data=None, save_summary=True):
        '''
        Chainlinks the list of routines one by one to 
        sequentially transform the data. The method will
        basically execute the queued up experiment.

        Parameters
        ----------
        data : Data object or None
               if None, the first func in the que
               has to load the data

        save_summary: save a .csv summary file containing
                      the progress of the experiment and
                      the basic description of data at
                      every stage. location: logs

        Returns
        -------
        transformed data as a Data object

        '''
        summary = list()
        data_descr = self._describe_data(data)
        summary.append({'op': None, 'op_time': None, 'data_description': data_descr})
        
        for _ in range(len(self.que)):
        
            func = self.que.pop(0)
            msg = f'Queuing <{func.__name__}> routine. . .'
            self.logger.info(msg)
            
            t = time.time()
            data = func(data)
            op_time = np.round(time.time() - t, decimals=3)
            
            msg = f'The operation took {op_time} s'
            self.logger.info(msg)
            data_descr = self._describe_data(data)
            
            summary.append({'op': func.__name__, 'op_time': op_time, 'data_description': data_descr})
        
        if save_summary:
            summary = self._reassemble_summary(summary)
            fname = f'{self.exp_name}_pipeline_summary.csv'
            path = os.path.join(self.dirs.logs, fname)
            summary.to_csv(path)
        
        return data   


    def run_over_stream(self, data_iter_factory, save_summary=True):
        """
        Like .run(), but consumes a stream (generator factory) of chunked Data.
        Apply the queued ops to each chunk sequentially and log a single summary.
        """
        summary = []
        # Build prettified text for chunk ID
        chunk_idx = 0
        for data in data_iter_factory():
            chunk_idx += 1
            for func in self.que:
                t0 = time.time()
                # collect pre-op description
                data_descr = self._describe_data(data)
                # apply op
                data = func(data)
                op_time = time.time() - t0
                summary.append({'chunk': chunk_idx,
                                'op': func.__name__,
                                'op_time': op_time,
                                'data_description': data_descr})
        if save_summary and summary:
            summary = self._reassemble_summary(summary)
            fname = f'{self.exp_name}_streaming_pipeline_summary.csv'
            path = os.path.join(self.dirs.logs, fname)
            summary.to_csv(path)
        return None

    # --- Post-processing: merge chunked outputs -------------------------------- 
    def merge_chunk_outputs(self, root_dirs=None, delete_chunks: bool = True):
        """
        Simplified merger for layouts like:
          <parent>/<base>__chunk1/<base>__chunk1_pep_count_summary.fasta
          <parent>/<base>__chunk2/<base>__chunk2_pep_count_summary.fasta
        It merges files across all chunk dirs by stripping '__chunkN' from the filename.

        FASTA behavior (fixed header '...count_<N>'):
          - Parse each record as:
              > ... count_<N>
              SEQUENCE
            Sum counts by SEQUENCE (whitespace removed, uppercased).
          - Write merged FASTA to <parent>/<base>/<base>_... with headers:
              >seq_<rank>_count_<SUM>
              SEQUENCE (wrapped as-is, no wrapping added)

        CSV behavior (if present):
          - Sum 'count' by sequence column (pep/dna/sequence/seq), re-rank globally.

        NPY: vertical concat (axis=0) of compatible shapes.

        Args:
            root_dirs: list of directories to scan. Defaults to [self.dirs.parser_out, self.dirs.logs].
            delete_chunks: remove per-chunk files and empty chunk dirs after merge.
        """
        import os, re, csv
        from collections import defaultdict, OrderedDict
        import numpy as np

        # ---------- resolve roots ----------
        roots = []
        if root_dirs:
            roots = list(root_dirs)
        else:
            for attr in ("parser_out", "logs"):
                if hasattr(self.dirs, attr):
                    d = getattr(self.dirs, attr)
                    if isinstance(d, str) and os.path.isdir(d):
                        roots.append(d)
        if not roots:
            self.logger.warning("merge_chunk_outputs: no directories to scan.")
            return

        chunk_dir_re = re.compile(r"^(?P<base>.+)__chunk\d+$")
        strip_chunk_in_name = re.compile(r"__chunk\d+")

        def _norm_seq(s: str) -> str:
            return "".join(str(s).split()).upper()

        # count_<N> only (your fixed format)
        COUNT_RE = re.compile(r"count_(\d+)", re.IGNORECASE)

        def _read_fasta_counts(path: str):
            """
            Read FASTA with headers containing 'count_<N>'.
            Returns list[(sequence_str, count_int)].
            """
            items, count, seq_lines = [], None, []
            with open(path, "r") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    if line.startswith(">"):
                        # flush previous record
                        if count is not None and seq_lines:
                            seq = _norm_seq("".join(seq_lines))
                            if seq:
                                items.append((seq, count))
                        # parse new header
                        m = COUNT_RE.search(line)
                        count = int(m.group(1)) if m else 1
                        seq_lines = []
                    else:
                        seq_lines.append(line)
            if count is not None and seq_lines:
                seq = _norm_seq("".join(seq_lines))
                if seq:
                    items.append((seq, count))
            return items

        def _write_merged_fasta(counts: dict[str, int], outpath: str):
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            # sort by descending count, then lexicographically
            items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            with open(outpath, "w") as out:
                for rank, (seq, cnt) in enumerate(items, 1):
                    out.write(f">seq_{rank}_count_{cnt}\n")
                    out.write(seq + "\n")

        def _read_csv_rows(fp: str):
            with open(fp, "r", newline="") as fh:
                rdr = csv.DictReader(fh)
                return rdr.fieldnames, list(rdr)

        def _merge_csv_files(part_files: list[str], target: str):
            # collect rows and header union
            all_rows, all_fields = [], OrderedDict()
            key_candidates = ("pep", "peptide", "dna", "sequence", "seq")
            for fp in part_files:
                try:
                    fields, rows = _read_csv_rows(fp)
                except Exception as e:
                    self.logger.warning(f"CSV read failed for {fp}: {e}; skipping.")
                    continue
                if not fields:
                    continue
                for f in fields:
                    all_fields.setdefault(f, None)
                all_rows.extend(rows)
            if not all_rows:
                self.logger.warning(f"No CSV rows to merge for {target}")
                return False

            header = list(all_fields.keys())
            keycol = next((k for k in key_candidates if k in header), None)
            if keycol is None:
                # fallback: concat
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "w", newline="") as outfh:
                    written = False
                    for fp in part_files:
                        with open(fp, "r", newline="") as fh:
                            first = True
                            for line in fh:
                                if first:
                                    if not written:
                                        outfh.write(line)
                                        written = True
                                    first = False
                                    continue
                                outfh.write(line)
                return True

            # sum counts by normalized sequence key
            agg = {}
            for r in all_rows:
                k = _norm_seq(r.get(keycol, ""))
                if not k:
                    continue
                rec = agg.get(k)
                if rec is None:
                    rec = {c: r.get(c, "") for c in header}
                    # initialize count as float if present
                    if "count" in header:
                        try:
                            rec["count"] = float(rec["count"])
                        except Exception:
                            rec["count"] = 0.0
                    rec[keycol] = k
                    agg[k] = rec
                else:
                    # sum count if present
                    if "count" in header:
                        try:
                            v = r.get("count", "0")
                            rec["count"] += float(v)
                        except Exception:
                            pass

            items = list(agg.values())
            has_count = "count" in header
            if has_count:
                # convert to int when exact & sort
                for rec in items:
                    v = rec.get("count", 0.0)
                    try:
                        iv = int(v)
                        rec["count"] = iv if abs(iv - float(v)) < 1e-9 else float(v)
                    except Exception:
                        pass
                items.sort(key=lambda rec: (-float(rec.get("count", 0.0)), rec.get(keycol, "")))
                if "rank" not in header:
                    header.append("rank")
            else:
                items.sort(key=lambda rec: rec.get(keycol, ""))

            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", newline="") as outfh:
                w = csv.DictWriter(outfh, fieldnames=header)
                w.writeheader()
                for rank, rec in enumerate(items, 1):
                    row = {}
                    for col in header:
                        if col == "rank" and has_count:
                            row[col] = rank
                        elif col == "count" and has_count:
                            v = rec.get("count", 0.0)
                            try:
                                iv = int(v)
                                row[col] = iv if abs(iv - float(v)) < 1e-9 else float(v)
                            except Exception:
                                row[col] = v
                        else:
                            row[col] = rec.get(col, "")
                    w.writerow(row)
            return True

        def _merge_npy(part_files: list[str], target: str):
            try:
                arrays, trailing = [], None
                for fp in part_files:
                    arr = np.load(fp, allow_pickle=True)
                    if trailing is None:
                        trailing = arr.shape[1:] if arr.ndim >= 1 else ()
                    elif arr.shape[1:] != trailing:
                        self.logger.warning(f"Skipping {fp} in NPY merge due to shape mismatch {arr.shape} vs {trailing}")
                        continue
                    arrays.append(arr)
                if not arrays:
                    self.logger.warning(f"No compatible arrays to merge for {target}")
                    return False
                merged = np.concatenate(arrays, axis=0)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                np.save(target, merged, allow_pickle=True)
                return True
            except Exception as e:
                self.logger.error(f"NPY merge failed for {target}: {e}")
                return False

        # ---------- scan roots for '<base>__chunkN/' groups ----------
        for root in roots:
            names = os.listdir(root)
            # group chunk dirs by base
            groups = {}
            for name in names:
                m = chunk_dir_re.match(name)
                if not m:
                    continue
                base = m.group("base")
                groups.setdefault((root, base), []).append(os.path.join(root, name))

            if not groups:
                continue

            for (parent, base), chunk_dirs in groups.items():
                chunk_dirs = sorted(chunk_dirs)
                out_dir = os.path.join(parent, base)
                self.logger.info(f"Merging {len(chunk_dirs)} chunk dir(s) -> {out_dir}/")
                os.makedirs(out_dir, exist_ok=True)

                # Build relname -> [paths] map by stripping '__chunkN' in file names
                relmap = {}
                for cdir in chunk_dirs:
                    try:
                        files = os.listdir(cdir)
                    except Exception as e:
                        self.logger.warning(f"Cannot list {cdir}: {e}")
                        continue
                    for fname in files:
                        rel = strip_chunk_in_name.sub("", fname)  # e.g., EM_2_merged_pep_count_summary.fasta
                        relmap.setdefault(rel, []).append(os.path.join(cdir, fname))

                # Merge each relname
                for rel, part_files in relmap.items():
                    part_files = sorted(part_files)
                    target = os.path.join(out_dir, rel)
                    ext = os.path.splitext(rel)[1].lower()
                    ok = False

                    if ext in (".fasta", ".fa", ".faa"):
                        # Sum counts by SEQUENCE, write as '>seq_<rank>_count_<SUM>'
                        total = defaultdict(int)
                        try:
                            for fp in part_files:
                                for seq, cnt in _read_fasta_counts(fp):
                                    total[seq] += int(cnt)
                            _write_merged_fasta(total, target)
                            ok = True
                        except Exception as e:
                            self.logger.error(f"FASTA merge failed for {target}: {e}")
                            ok = False

                    elif ext == ".csv":
                        ok = _merge_csv_files(part_files, target)

                    elif ext == ".npy":
                        ok = _merge_npy(part_files, target)

                    else:
                        # generic text concat
                        try:
                            with open(target, "w") as outfh:
                                for fp in part_files:
                                    with open(fp, "r") as fh:
                                        outfh.write(fh.read())
                            ok = True
                        except Exception as e:
                            self.logger.error(f"Text merge failed for {target}: {e}")
                            ok = False

                    if ok and delete_chunks:
                        for fp in part_files:
                            try:
                                os.remove(fp)
                            except Exception as e:
                                self.logger.warning(f"Could not delete chunk file {fp}: {e}")
                    self.logger.info("Merge " + ("succeeded" if ok else "failed") + f" for {target}")

                # remove empty chunk dirs
                if delete_chunks:
                    for cdir in chunk_dirs:
                        try:
                            if not os.listdir(cdir):
                                os.rmdir(cdir)
                        except Exception as e:
                            self.logger.debug(f"Could not remove dir {cdir}: {e}")
    # --- end merge chunked outputs --------------------------------------------



class FastqParser(Handler):
    '''
    A processor for fastq/fastq.gz data. Primary parser for the sequencing
    data. The class holds methods for applying sequential filters to DNA
    sequencing data to eliminate noise, etc, and to convert raw NGS output
    to a list of peptides for the downstream applications. 
    
    Most public routines act on Data objects (except IO data fetchers) to
    return a transformed instance of Data.

    The class also holds a number of ops for basic statistics gathering.
    These also take Data as input, describe it in some way, write an out
    file (.png or txt or both) and return Data as-is.
    '''
    
    def __init__(self, *args):
        super(FastqParser, self).__init__(*args)
    
        self._validate()
        super(FastqParser, self)._on_completion()
        return    

    def _init_internal_state(self, sample):
        n_cols = max(len(self.P_design), len(self.D_design))
        sample._internal_state = np.ones((len(sample), n_cols), dtype=bool)

    def __repr__(self):
        return '<FastqParser object>'

    def _validate(self):
        
        if not (hasattr(self, 'P_design') and
                hasattr(self, 'D_design')
               ):
            msg = 'FastqParser requires peptide and DNA library design objects for setup. . .'
            self.logger.error(msg)
            raise ValueError(msg)
        #---Removed in order to fit the situation of multiple DNA tempaltes code same peptide sequence.    
        #if not len(self.P_design) == len(self.D_design):
            #msg = 'Peptide and DNA library designs must contains the same number of templates; cannot inialize FastqProcessor. . .'
            #self.logger.error(msg)
            #raise ValueError(msg)
            
        if not hasattr(self, 'constants'):
            msg = 'FastqParser requires constants for setup. . .'
            self.logger.error(msg)
            raise ValueError(msg)
        return

    def _transform_check(self, sample, func):
        if not sample.get_ndims() == 2:
            raise ValueError(f'Sample {sample.name} holds arrays of unsupported dimensionality for {func} op. Expected: arrays of ndims=2, got: ndims={sample.get_ndims()}')
        
        return

    def _decode_q_ascii(self, q_ascii, *, offset:int = 33):
        return np.frombuffer(q_ascii.encode("ascii"), dtype=np.uint8, count=-1) - offset

    def _dna_to_pep(self, seq, force_at_frame=None, stop_readthrough=False):       
             
        def find_orf(seq):
            loc = re.search(self.utr5_seq, seq)
            if loc is not None:
                return seq[loc.start():]
            else:
                return None
        
        def find_stop(peptide):
            if stop_readthrough:
                return peptide
            
            else:
                ind = peptide.find('*')
                if ind == -1:
                    return peptide + '+'
                else:
                    return peptide[:ind]

        #figure out what to use as orf
        if force_at_frame is None:                
            orf = find_orf(seq)
        else:
            orf = seq[force_at_frame:]
            
        #throughout, '+' is a reserved symbol to denote messed up sequences
        #no stop codon, weird codons, etc
        pep = ''
        if orf is not None: 
            for i in range(0, len(orf), 3):
                try:
                    pep += self.constants.codon_table[orf[i:i+3]]
                except:
                    if len(orf[i:i+3]) != 3:
                        pep += '_'
                    else:
                        pep += '+'

        return find_stop(pep)

    def _L_summary(self, arr):

        #infer what the pad token is
        pad = np.zeros(1, dtype=arr.dtype)[0]
        
        #fetch the indexes where dna/pep length == designed
        return np.sum(arr != pad, axis=1)

    def _find_max_len(self, design, loc):
        '''
        When trying to get a column-wise view of the array,
        The views for different designs can have a different
        shape (for example, different vr size). This will 
        find the largest possible column-wise view.
        Output m is used to as a shape parameter during
        array creation.
        '''
        m = 0 
        for template in design:
            if len(template(loc)) > m:     
                m = len(template(loc))
        return m

    def _where_check(self, where):
        
        if where == 'pep':
            if not hasattr(self, 'P_design'):
                msg = "Cannot run peptide filtration routines with unspecified library design."
                self.logger.error(msg)
                raise ValueError(msg)
            
        elif where == 'dna':
            if not hasattr(self, 'D_design'):
                msg = "Cannot run dna filtration routines with unspecified library design."
                self.logger.error(msg)
                raise ValueError(msg)                       
         
        else:
            msg = f'The parser did not understand which dataset it should operate on. Passed value: {where}; allowed values: pep/dna.'
            self.logger.error(msg)
            raise ValueError(msg)
        return

    def _loc_check(self, loc, design):
        
        if not isinstance(loc, list):
            msg = f'The Parser expected to receive a list of region indexes to parse; received: {type(loc)}'
            self.logger.error(msg)
            raise ValueError(msg)

        if max(loc) > design.loc.max():
            msg = f'{design.lib_type} library design does not contain enough regions. Library design contains {design.loc.max() + 1} regions; specified: up to {max(loc) + 1}'
            self.logger.error(msg)
            raise AssertionError(msg)
        return

    def _prepare_destinations(self, data):
        
        for sample in data:
                destination = os.path.join(self.dirs.parser_out, sample.name)
                if not os.path.isdir(destination):
                    os.makedirs(destination)        
        return    
    
    #--------------------------------------------
    #The methods below are public data transformers.
    #All of them modify data in some way.
    #--------------------------------------------
    def translate(self, force_at_frame=None, stop_readthrough=False):
        '''
    	For each sample in Data, perform in silico translation for DNA sequencing data. 
    	The op will return data containing translated peptide lists. The op is 
        intended for one-ORF-per-read NGS data, but not for long, multiple-ORFs-per-read
        samples.
             
        This op should be called after fetching the data and (optionally) running
        the FastqParser.revcom(), prior to any filtration routines.
        
        On top of running translation, this op will also transform the data 
        to a reprensentation suitable for downstream ops.
        
        Parameters:
                force_at_frame: if None, a regular ORF search will be performed. Regular ORF
                                search entails looking for a Shine-Dalgarno sequence upstream 
                                of an ATG codon (the exact 5’-UTR sequence signalling an 
                                ORF is specified in config.py).
                                								
                                if not None, can take values of 0, 1 or 2. This will force-start
                                the translation at the specified frame regardless of the 
                                presence or absence of the SD sequence.
                                
                                For example:
                                DNA: TACGACTCACTATAGGGTTAACTTTAAGAAGGA
                   force_at_frame=0  ----------> 
                    force_at_frame=1  ---------->
                     force_at_frame=2  ---------->
                                 
              stop_readthrough:	bool (True/False; default: False). if True, translation will
                                continue even after encountering a stop codon until the 3'-end
                                of the corresponding read. Note, that an "_" amino acid will
                                be appended to the peptide sequence at the C-terminus if the 
                                last encountred codon is missing 1 or 2 bases.
                                
                                if False, the op will return true ORF sequences. In this case,
                                peptide sequences coming from ORFs which miss a stop codon will
                                be labelled with a "+" amino acid at the C-terminus.
                                
                                Should be flagged True for ORFs with no stop codon inside the read.
				 
        Returns:
                Data object containing peptide sequence information
        '''
        
        if force_at_frame is not None:
            if force_at_frame not in (0, 1, 2):
                msg = f'<translate> routine expected to receive param "force_at_frame" as any of (0, 1, 2); received: {force_at_frame}'
                self.logger.error(msg)
                raise ValueError(msg)
        else:     
            if not hasattr(self, 'utr5_seq'):
                msg = "5' UTR sequence is not set for the <translation> routine. Can not perform ORF search. Aborting. . ."
                self.logger.error(msg)
                raise ValueError(msg)   
            
            if isinstance(self.utr5_seq, (list, tuple, set)):
                # join them with | so re.search() will match any of them
                self.utr5_seq = "(?:" + ")|(?:".join(self.utr5_seq) + ")"
                
        if type(stop_readthrough) != bool:
            msg = f'<translate> routine expected to receive param "stop_readthrough" as type=bool; received: {type(stop_readthrough)}'
            self.logger.error(msg)
            raise ValueError(msg)   
                                
        def translate_dna(data):
            for sample in data:
                
                sample.P = np.array([self._dna_to_pep(
                                                      x, 
                                                      force_at_frame=force_at_frame,
                                                      stop_readthrough=stop_readthrough
                                                     ) 
                                     
                                     for x in sample.D])

                #this transformation is not declared publicly; may be it should
                sample.transform()
                self._init_internal_state(sample)

                #set the internal state for the first time, removed for DNA template 
                #shape = (len(sample), len(self.P_design))
                #sample._internal_state = np.ones(shape, dtype=np.bool)

            return data
        return translate_dna
    
    def translate_both_strands(self, *, force_at_frame=None, stop_readthrough=False, utr5_offset: int = 0):
        """
        Barcode-gated translation:
          • Search BARCODE on forward strand.
              - If found: search UTR5 on forward. If found -> slice from UTR5.end() and translate.
              - If UTR5 not found -> drop.
          • Else reverse-complement and search BARCODE on RC.
              - If found: search UTR5 on RC. If found -> slice from UTR5.end() and translate.
              - If UTR5 not found -> drop.
          • If BARCODE not found in either orientation -> drop.

        Outputs one row per kept read. Adds/updates `sample.strand_id` (0=fwd, 1=rc).
        """

        # ---------------- sanity -------------------------------------------------
        if force_at_frame is not None and force_at_frame not in (0, 1, 2):
            raise ValueError("force_at_frame must be 0 / 1 / 2 or None")
        if not hasattr(self, "utr5_seq"):
            raise ValueError("5'-UTR regex `utr5_seq` is missing in config")
        if not hasattr(self, "barcode"):
            raise ValueError("`barcode` is missing in config")
        if not isinstance(stop_readthrough, bool):
            raise TypeError("stop_readthrough must be bool")
        if not isinstance(utr5_offset, int) or utr5_offset < 0:
            raise ValueError("utr5_offset must be a non-negative int")

        # ---------------- setup --------------------------------------------------
        import re
        import numpy as np

        # Compile UTR (supports str or list/tuple of alternatives)
        utr5_cfg = getattr(self, "utr5_seq", None)
        if isinstance(utr5_cfg, (list, tuple)):
            utr5_re = re.compile("(?:" + "|".join(map(re.escape, utr5_cfg)) + ")")
        elif isinstance(utr5_cfg, str):
            utr5_re = re.compile(utr5_cfg)
        else:
            raise TypeError("utr5_seq must be str or list/tuple of str")

        # Compile BARCODE (supports str or list/tuple; list treated as literal alts)
        bc_cfg = getattr(self, "barcode", None)
        if isinstance(bc_cfg, (list, tuple)):
            bc_re = re.compile("(?:" + "|".join(map(re.escape, bc_cfg)) + ")")
        elif isinstance(bc_cfg, str):
            bc_re = re.compile(bc_cfg)
        else:
            raise TypeError("barcode must be str or list/tuple of str")

        def _rc_str(seq: str) -> str:
            return seq.translate(self.constants.complement_table)[::-1]

        # DNA row -> trimmed string (remove right pad)
        def _dna_row_to_str(row) -> str:
            if isinstance(row, np.ndarray):
                if row.dtype.kind in ("U", "S"):
                    return "".join(row.tolist()).rstrip(" ")
                return "".join(map(str, row.tolist())).rstrip(" ")
            return str(row).rstrip(" ")

        # Trim Q to DNA length
        def _q_row_trim_to_len(row, L: int):
            if row is None:
                return None
            if isinstance(row, np.ndarray):
                return row[:L]
            s = str(row)
            return s[:L]

        # Slice Q object from index
        def _q_slice(q_obj, start: int):
            if q_obj is None:
                return None
            try:
                return q_obj[start:]
            except Exception:
                if isinstance(q_obj, (list, tuple)):
                    return q_obj[start:]
                return q_obj

        # ---------------- core op ------------------------------------------------
        def translate_DNA(data):
            for sample in data:
                has_Q = hasattr(sample, "Q") and sample.Q is not None

                # Determine iteration count from D (1-D list or 2-D padded)
                if isinstance(sample.D, np.ndarray) and getattr(sample.D, "ndim", 1) == 2:
                    N = sample.D.shape[0]
                else:
                    N = len(sample.D)

                D_out, Q_out, P_out, strand_out = [], [], [], []

                for i in range(N):
                    # Fetch DNA & Q as strings/arrays trimmed to effective length
                    dna_row = sample.D[i]
                    dna = _dna_row_to_str(dna_row)
                    if not dna:
                        continue
                    if has_Q:
                        q_row = sample.Q[i]
                        q_trim = _q_row_trim_to_len(q_row, len(dna))
                    else:
                        q_trim = None

                    # ---------- 1) Forward: barcode -> utr5 ----------
                    if bc_re.search(dna) is not None:
                        m_utr = utr5_re.search(dna)
                        if m_utr is None:
                            # barcode present but no UTR5 in forward → drop
                            continue
                        start_idx = m_utr.start() + utr5_offset         # .start() AT UTR5, change to .end() when start AFTER UTR5
                        dna_use = dna[start_idx:]
                        q_use = _q_slice(q_trim, start_idx) if has_Q else None
                        pep = self._dna_to_pep(
                            dna_use,
                            force_at_frame=0,
                            stop_readthrough=stop_readthrough
                        )
                        D_out.append(dna_use)
                        if has_Q:
                            Q_out.append(q_use)
                        P_out.append(pep)
                        strand_out.append(0)             # forward
                        continue

                    # ---------- 2) Reverse: barcode (RC) -> utr5 (RC) ----------
                    dna_rc = _rc_str(dna)
                    if bc_re.search(dna_rc) is not None:
                        m_utr_rc = utr5_re.search(dna_rc)
                        if m_utr_rc is None:
                            # barcode in RC but no UTR5 in RC → drop
                            continue
                        start2 = m_utr_rc.start() + utr5_offset          # '.start()' AT UTR5 in RC, change to .end() when start AFTER UTR5 in RC
                        if has_Q:
                            q_rc = q_trim[::-1] if not isinstance(q_trim, np.ndarray) else q_trim[::-1]
                            q_use = _q_slice(q_rc, start2)
                        else:
                            q_use = None
                        dna_use = dna_rc[start2:]
                        pep = self._dna_to_pep(
                            dna_use,
                            force_at_frame=0,
                            stop_readthrough=stop_readthrough
                        )
                        D_out.append(dna_use)
                        if has_Q:
                            Q_out.append(q_use)
                        P_out.append(pep)
                        strand_out.append(1)             # reverse-complement
                        continue

                    # ---------- 3) No barcode in either orientation -> drop ----------
                    # (do nothing)

                # Assign back as object arrays (ragged-safe) and normalize
                sample.D = np.asarray(D_out, dtype=object)
                if has_Q:
                    sample.Q = np.asarray(Q_out, dtype=object)
                sample.P = np.asarray(P_out, dtype=object)
                sample.strand_id = np.asarray(strand_out, dtype=np.int8)

                sample.transform()
                self._init_internal_state(sample)

            return data

        return translate_DNA

    
    def translate_all_frames_both_strands(self, stop_readthrough: bool = False):
        """
        For every read create SIX peptide sequences:

            ┌─ forward strand ────────────────────────────────┐
            │  frame 0 , frame 1 , frame 2                    │
            └──────────────────────────────────────────────────┘
            ┌─ reverse-complement strand ─────────────────────┐
            │  frame 0 , frame 1 , frame 2                    │
            └──────────────────────────────────────────────────┘

        The returned Data object keeps **one sample per input sample**
        but each sample is 6 × longer.  
        Extra per-row annotations:
            • .frame_id   → 0,1,2   (translation frame)
            • .strand_id  → 0=fwd , 1=rev-comp
        """
        # helpers – SAME as in your old revcom()
        @np.vectorize
        def _rc(seq: str) -> str:
            return seq.translate(self.constants.complement_table)[::-1]

        @np.vectorize
        def _r(seq: str) -> str:
            return seq[::-1]

        def _make_6x_sample(sample):
            # -------------------------------------------------------------
            # 1)  forward-strand peptides (frames 0/1/2) ------------------
            pep_fwd = []
            for frame in (0, 1, 2):
                pep_fwd.append([
                    self._dna_to_pep(d,
                                     force_at_frame=frame,
                                     stop_readthrough=stop_readthrough)
                    for d in sample.D
                ])
            P_fwd = np.concatenate([np.asarray(x) for x in pep_fwd], axis=0)
            D_fwd = np.concatenate([sample.D] * 3, axis=0)
            Q_fwd = np.concatenate([sample.Q] * 3, axis=0)
            frame_fwd  = np.repeat([0, 1, 2], len(sample))
            strand_fwd = np.zeros(len(P_fwd), dtype=np.int8)          # 0 = forward

            # -------------------------------------------------------------
            # 2)  reverse-strand peptides (frames 0/1/2) ------------------
            D_rc  = _rc(sample.D)
            Q_rc  = _r(sample.Q)
            pep_rev = []
            for frame in (0, 1, 2):
                pep_rev.append([
                    self._dna_to_pep(d,
                                     force_at_frame=frame,
                                     stop_readthrough=stop_readthrough)
                    for d in D_rc
                ])
            P_rev = np.concatenate([np.asarray(x) for x in pep_rev], axis=0)
            D_rev = np.concatenate([D_rc] * 3, axis=0)
            Q_rev = np.concatenate([Q_rc] * 3, axis=0)
            frame_rev  = np.repeat([0, 1, 2], len(sample))
            strand_rev = np.ones(len(P_rev), dtype=np.int8)           # 1 = reverse

            # -------------------------------------------------------------
            # 3)  concat forward + reverse -------------------------------
            P_cat = np.concatenate([P_fwd, P_rev], axis=0)
            D_cat = np.concatenate([D_fwd, D_rev], axis=0)
            Q_cat = np.concatenate([Q_fwd, Q_rev], axis=0)
            frame_id  = np.concatenate([frame_fwd,  frame_rev],  axis=0)
            strand_id = np.concatenate([strand_fwd, strand_rev], axis=0)

            # build fresh SequencingSample ---------------------------------
            from utils.datatypes import SequencingSample
            big = SequencingSample(
                name=f"{sample.name}_6frames",
                D=D_cat,
                Q=Q_cat,
                P=P_cat
            )
            big.frame_id  = frame_id    # 0/1/2
            big.strand_id = strand_id   # 0=fwd 1=rev

            # 1-D → 2-D (pads with empty '' char)
            big.transform()
            self._init_internal_state(big)
            return big

        # the closure actually executed by Pipeline -----------------------
        def _translate(data):
            new_samples = [_make_6x_sample(sample) for sample in data]
            from utils.datatypes import Data
            return Data(samples=new_samples)

        return _translate
    
    def translate_from_current_D(self, *, force_at_frame: int = 0, stop_readthrough: bool = False):
        """
        Recompute peptides from the CURRENT DNA (e.g., after trimming) and replace sample.P.
        Translation starts at `force_at_frame` (default 0 = first base of current D) and runs to end.
    """
        import numpy as np
        import inspect

        def _row_to_str(row) -> str:
            # join a 1-char-wide padded row or pass through string/object
            if isinstance(row, np.ndarray):
                if row.dtype.kind in ("U", "S"):
                    return "".join(row.tolist()).rstrip(" ")
                else:
                    return "".join(map(str, row.tolist())).rstrip(" ")
            return str(row).rstrip(" ")

        def _op(data):
            for sample in data:
                self._transform_check(sample, inspect.stack()[0][3])
                # Build peptide list from current DNA rows
                if hasattr(sample.D, "ndim") and sample.D.ndim == 2:
                    N = sample.D.shape[0]
                    peps = [
                        self._dna_to_pep(_row_to_str(sample.D[i]),
                                         force_at_frame=force_at_frame,
                                         stop_readthrough=stop_readthrough)
                        for i in range(N)
                    ]
                else:
                    peps = [
                        self._dna_to_pep(_row_to_str(d),
                                         force_at_frame=force_at_frame,
                                         stop_readthrough=stop_readthrough)
                        for d in sample.D
                    ]
                sample.P = np.asarray(peps, dtype=object)
                # normalize representation and internal state
                sample.transform()
                self._init_internal_state(sample)
            return data

        _op.__name__ = "translate_from_current_D"
        return _op


    def revcom(self):
        '''
        For each sample in Data, get reverse complement of DNA sequences and 
        reverse sequences of the corresponding Q score. If used, should enqueued 
        right after the fetching op, and before any downstream ops.
        
        Parameters:
                None
    
        Returns:
                Transformed Data object holding reverse-complemented DNA
                and reversed Q score information
        '''        

        @np.vectorize
        def _rc(seq):
            return seq.translate(self.constants.complement_table)[::-1]
        
        def _rev_q_arr(arr):
            """
            Reverse each element of the *object-dtype* Q array produced for PacBio.
            Falls back to the old behaviour for plain strings.
            """
            if arr.dtype == object:                   # PacBio / decoded Q
                return np.array([q[::-1] for q in arr], dtype=object)
            else:                                     # classic Illumina strings
                return np.array([q[::-1] for q in arr], dtype=arr.dtype)

        def revcom_data(data):
            for sample in data:
    
                if sample.D.ndim != 1 or sample.Q.ndim != 1:
                    msg = f'<revcom> can only be called on samples holding 1D-represented DNA. Ignoring the routine for {sample.name} sample. . .'
                    self.logger.warning(msg)
                    continue
                
                if sample.P:
                    msg = 'Attempting to to revcom a sample holding a P dataset. P dataset will be ignored. . .'
                    self.logger.warning(msg)
                    
                sample.D = _rc(sample.D)
                sample.Q = _rev_q_arr(sample.Q)
    
            return data
        return revcom_data

    def len_filter(self, where=None, len_range=None):
        '''
        For each sample in Data, filter out sequences longer/shorter than the specified 
        library designs. Alternatively, a length range of sequences to take can be optionally 
        specified to filter out the entries (NGS reads) outside of this range.
        
        Parameters:
                   where: 'dna' or 'pep' to specify which dataset the op 
                          should work on.
						  
               len_range: either None (filtration will be done according to
                          the library design rules), or a list of two ints 
                          that specifies the length range to fetch.						  
					 
        Returns:
                Transformed Data object containing length-filtered data
        '''        
        self._where_check(where)

        if where == 'pep':
            design = self.P_design
            
        elif where == 'dna':
            design = self.D_design
        
        if len_range is not None:
            if not isinstance(len_range, list):
                msg = f'<len_filter> routine expected to receive len_range argument as a list; received: {type(len_range)}'
                self.logger.error(msg)
                raise ValueError(msg)
            
            if len(len_range) != 2:
                msg = f'<len_filter> routine expected to receive len_range as a list with two values; received: len={len(len_range)}'
                self.logger.error(msg)
                raise ValueError(msg)                

        def length_filter(data):
            for sample in data:
                
                self._transform_check(sample, inspect.stack()[0][3])
                arr = sample[where]  
   
                #L is a length summary array
                L = self._L_summary(arr)
                
                #change the sample internal state
                for i,template in enumerate(design):
                    row_mask = sample._internal_state[:,i]
                    
                    if len_range is None:
                        sample._internal_state[row_mask, i] = L[row_mask] == template.L
                    else:
                        sample._internal_state[row_mask, i] = (L[row_mask] > len_range[0]) & (L[row_mask] < len_range[1])
                    
                #keep every entry that has at least one positive
                #value in the internal state array
                ind = np.any(sample._internal_state, axis=-1)
                sample(ind)
            
            return data
        return length_filter

    def DNA_vote_filter(self, *, templates=None, names=None, k: int = 9,
                        min_votes: int = 150, annotate: bool = True,
                        trim_to_template: bool = True):
        """
        Filter reads by k-mer voting against DNA templates (constant regions only).
        - templates: list[str or Template-like] or None (auto-discover from D_design)
        - names: optional list[str]; if None, auto-uses template.name or the template DNA string
        - k: k-mer size (default 9)
        - min_votes: keep a read only if max votes across templates >= min_votes
        - annotate: attach scaffold_id, scaffold_name (if names available), and kmer_votes
        - trim_to_template: if True, trim D (and Q) to the assigned template span using
                            the dominant k-mer offset (mode of read_i - tmpl_j).

        Behavior:
          * Variable positions in templates (non-ACGT like '1','N', regex tokens) are IGNORED for voting.
          * Works with 1-D or 2-D padded D/Q/P; output is normalized via transform().
          * P (peptide) is not altered by trimming.
        """
        import numpy as np
        from collections import Counter

        # ---------------- helpers ----------------
        def _coerce_templates_to_str_list(tmpls):
            out = []
            for t in tmpls:
                if isinstance(t, str):
                    out.append(t)
                else:
                    v = getattr(t, "lib_seq", None)
                    if isinstance(v, str):
                        out.append(v)
                    else:
                        for attr in ("dna", "DNA", "template"):
                            v2 = getattr(t, attr, None)
                            if isinstance(v2, str):
                                out.append(v2)
                                break
                        else:
                            out.append(str(t))
            return [str(x) for x in out]

        def _autodiscover_templates_and_names():
            # Prefer config merged into parser
            for attr in ("D_design", "D_library", "Dlib", "library_design", "lib_design"):
                obj = getattr(self, attr, None)
                if obj is not None and hasattr(obj, "templates"):
                    tmpls = list(obj.templates)
                    names_local = []
                    for t in tmpls:
                        nm = getattr(t, "name", None)
                        if isinstance(nm, str) and nm:
                            names_local.append(nm)
                        else:
                            nm2 = getattr(t, "lib_seq", None)
                            names_local.append(nm2 if isinstance(nm2, str) else str(t))
                    return _coerce_templates_to_str_list(tmpls), names_local
            # Fallback to loaded config modules
            import sys
            for mod in list(sys.modules.values()):
                try:
                    PC = getattr(mod, "ParserConfig", None)
                    if PC is None:
                        continue
                    Dlib = getattr(PC, "D_design", None)
                    if Dlib is None or not hasattr(Dlib, "templates"):
                        continue
                    tmpls = list(Dlib.templates)
                    names_local = []
                    for t in tmpls:
                        nm = getattr(t, "name", None)
                        if isinstance(nm, str) and nm:
                            names_local.append(nm)
                        else:
                            nm2 = getattr(t, "lib_seq", None)
                            names_local.append(nm2 if isinstance(nm2, str) else str(t))
                    return _coerce_templates_to_str_list(tmpls), names_local
                except Exception:
                    continue
            raise ValueError("DNA_vote_filter: cannot find D_design.templates; pass templates=[...] explicitly.")

        def _constant_mask(tdna: str) -> np.ndarray:
            arr = np.frombuffer(tdna.upper().encode("ascii"), dtype="S1")
            return np.isin(arr, np.array([b"A", b"C", b"G", b"T"]))

        def _template_kmers_with_pos(tdna: str, k: int):
            """
            Return:
              kset: set of constant-only kmers
              kpos: dict kmer -> list of positions (start indices in template)
            """
            mask = _constant_mask(tdna)
            L = len(tdna)
            kset = set()
            kpos = {}
            if L >= k:
                m = mask.astype(np.uint8)
                win_ok = (np.convolve(m, np.ones(k, dtype=np.uint8), mode="valid") == k)
                tdnaU = tdna.upper()
                for i, ok in enumerate(win_ok):
                    if ok:
                        km = tdnaU[i:i+k]
                        kset.add(km)
                        kpos.setdefault(km, []).append(i)
            return kset, kpos

        def _template_kmers(tdna: str, k: int):
            return _template_kmers_with_pos(tdna, k)[0]

        # ---------------- prepare templates ----------------
        if templates is None:
            templates, auto_names = _autodiscover_templates_and_names()
            if names is None:
                names = auto_names
        else:
            templates = _coerce_templates_to_str_list(list(templates))
            if names is None:
                names = templates

        T = len(templates)
        if T == 0:
            raise ValueError("DNA_vote_filter: no templates available")
        if names is not None and len(names) != T:
            raise ValueError("DNA_vote_filter: names length must match templates")

        t_dnas = templates  # already strings
        if trim_to_template:
            # need positions for offset estimation
            tk = [_template_kmers_with_pos(t, k) for t in t_dnas]
            t_ksets = [x[0] for x in tk]
            t_kpos  = [x[1] for x in tk]
        else:
            t_ksets = [_template_kmers(t, k) for t in t_dnas]
            t_kpos  = None

        def _calc_votes(seq: str):
            # Use all windows from the read; only constant template kmers can match anyway.
            seqU = seq.upper()
            Ls = len(seqU)
            if Ls < k:
                return np.zeros(T, dtype=np.int32)
            v = np.zeros(T, dtype=np.int32)
            for i in range(Ls - k + 1):
                km = seqU[i:i+k]
                for t_idx, kset in enumerate(t_ksets):
                    if km in kset:
                        v[t_idx] += 1
            return v

        def _estimate_offset_and_span(seqU: str, t_idx: int) -> tuple[int | None, int | None, int | None, int]:
            """
            Estimate alignment offset AND the span of consistent matches in READ coords.
            Returns (offset, min_i, max_i, support). Only k-mers whose (i - j) equals the
            mode offset contribute to min_i/max_i. If no matches, returns (None, None, None, 0).
            """
            if t_kpos is None:
                return None, None, None, 0
            kpos_map = t_kpos[t_idx]
            Ls = len(seqU)
            if Ls < k:
                return None, None, None, 0

            offsets = Counter()
            # First pass: gather all offsets from any match
            for i in range(Ls - k + 1):
                km = seqU[i:i+k]
                pos_list = kpos_map.get(km)
                if pos_list:
                    for j in pos_list:
                        offsets[i - j] += 1

            if not offsets:
                return None, None, None, 0

            # Mode offset (break ties by larger support)
            off, support = max(offsets.items(), key=lambda kv: kv[1])

            # Second pass: compute span using only matches consistent with the mode
            min_i, max_i = None, None
            for i in range(Ls - k + 1):
                km = seqU[i:i+k]
                pos_list = kpos_map.get(km)
                if not pos_list:
                    continue
                # any template position j giving the mode offset?
                if any((i - j) == off for j in pos_list):
                    if min_i is None or i < min_i:
                        min_i = i
                    if max_i is None or i > max_i:
                        max_i = i

            if min_i is None or max_i is None:
                return None, None, None, 0

            return int(off), int(min_i), int(max_i), int(support)

        def _row_to_string(row):
            # Convert a row that might be a '<U1' char array, bytes, or string → clean DNA string
            if isinstance(row, np.ndarray):
                if row.dtype.kind in ("U", "S"):
                    return "".join(row.tolist()).rstrip(" ")
                else:
                    return "".join(map(str, row.tolist())).rstrip(" ")
            s = str(row)
            return s.strip()

        def _slice_q_like(q_src, i_row, start, end):
            """
            Slice the Q row i_row from [start:end] and return a 1-D object suitable for transform().
            Handles 2-D numpy matrices, 1-D numpy arrays, and strings.
            """
            if q_src is None:
                return None
            if hasattr(q_src, "ndim"):
                if q_src.ndim == 2:
                    return q_src[i_row, start:end]
                elif q_src.ndim == 1:
                    return q_src[start:end]
            # object / string list
            row = q_src[i_row] if isinstance(q_src, (list, tuple, np.ndarray)) else q_src
            if isinstance(row, np.ndarray):
                return row[start:end]
            return str(row)[start:end]

        # ---------------- core op ----------------
        def DNA_vote_filter(d):
            for sample in d:
                # Build list of DNA strings and remember original lengths for default slicing
                D_is_2d = hasattr(sample.D, "ndim") and sample.D.ndim == 2
                if D_is_2d:
                    D_strs = [_row_to_string(r) for r in sample.D]
                else:
                    D_strs = [_row_to_string(r) for r in sample.D]

                Q_src = getattr(sample, "Q", None)
                P_src = getattr(sample, "P", None)

                keep_idx = []
                assign_ids = []
                vote_vals = []

                D_out, Q_out = [], []
                P_out = [] if P_src is not None else None
                trim_starts, trim_ends = [], []

                for i, seq in enumerate(D_strs):
                    if not seq:
                        continue

                    v = _calc_votes(seq)
                    t_idx = int(np.argmax(v))
                    vmax = int(v[t_idx])
                    if vmax < int(min_votes):
                        continue  # drop

                    # keep this row
                    keep_idx.append(i)
                    assign_ids.append(t_idx)
                    vote_vals.append(vmax)

                    # 2) In the main loop (inside DNA_vote_filter), replace the old trimming block:
                    if trim_to_template:
                        # Tail-trim to the last constant-region k-mer hit of the assigned template
                        seqU = seq.upper()
                        Ls = len(seqU)
                        kset = t_ksets[t_idx]  # constant-only kmers for assigned template

                        last_i = -1
                        if Ls >= k and kset:
                            # scan once to locate the last hit
                            for j in range(Ls - k + 1):
                                if seqU[j:j+k] in kset:
                                    last_i = j

                        if last_i >= 0:
                            start = 0                      # 5' already trimmed by translator
                            end   = last_i + k             # keep through the LAST anchor k-mer
                        else:
                            # no anchor found (should be rare if votes >= min_votes): keep as-is
                            start, end = 0, len(seq)

                        D_out.append(seq[start:end])
                        if Q_src is not None:
                            Q_out.append(_slice_q_like(Q_src, i, start, end))
                        if P_out is not None:
                            P_i = P_src[i] if hasattr(P_src, "__getitem__") else P_src
                            if isinstance(P_i, np.ndarray) and P_i.dtype.kind in ("U", "S"):
                                P_out.append("".join(P_i.tolist()).rstrip(" "))
                            else:
                                P_out.append(P_i)
                        trim_starts.append(start)
                        trim_ends.append(end)
                    else:
                        # No trimming: keep effective sequence string (right pads removed)
                        D_out.append(seq)
                        if Q_src is not None:
                            L = len(seq)
                            if hasattr(Q_src, "ndim") and getattr(Q_src, "ndim", 1) == 2:
                                Q_out.append(Q_src[i, :L])
                            else:
                                qi = Q_src[i] if hasattr(Q_src, "__getitem__") else Q_src
                                if isinstance(qi, np.ndarray):
                                    Q_out.append(qi[:L])
                                else:
                                    Q_out.append(str(qi)[:L])
                        if P_out is not None:
                            P_i = P_src[i] if hasattr(P_src, "__getitem__") else P_src
                            if isinstance(P_i, np.ndarray) and P_i.dtype.kind in ("U", "S"):
                                P_out.append("".join(P_i.tolist()).rstrip(" "))
                            else:
                                P_out.append(P_i)

                # Apply to sample: convert to object arrays and re-pad via transform()
                sample.D = np.asarray(D_out, dtype=object)
                if Q_src is not None:
                    sample.Q = np.asarray(Q_out, dtype=object)
                if P_out is not None:
                    sample.P = np.asarray(P_out, dtype=object)

                # Annotations
                if annotate:
                    sample.scaffold_id = np.asarray(assign_ids, dtype=np.int32)
                    if names is not None:
                        sample.scaffold_name = np.asarray([names[i] for i in assign_ids], dtype=object)
                    sample.kmer_votes = np.asarray(vote_vals, dtype=np.int32)
                    if trim_to_template:
                        sample.trim_start = np.asarray(trim_starts, dtype=np.int32)
                        sample.trim_end   = np.asarray(trim_ends,   dtype=np.int32)
                        sample.template_len = np.asarray([len(t_dnas[i]) for i in assign_ids], dtype=np.int32)

                # Rebuild padded representation & internal state
                sample.transform()
                self._init_internal_state(sample)

            return d

        return DNA_vote_filter


    def cr_filter_fuzzy(self, *, where="pep", loc=None, tol=0):
        """
        Fuzzy constant-region filter (works with multiple templates).

        Parameters
        ----------
        where : {"pep", "dna"}
            Dataset to operate on.
        loc : list[int]
            Constant-region indices to check.
        tol : int
            Max substitutions allowed inside a constant block.

        Returns
        -------
        callable   –  ready for Pipeline.enque()
        """
        # ------------------------------------------------ sanity ----------
        self._where_check(where)
        design = self.P_design if where == "pep" else self.D_design
        self._loc_check(loc, design)
        if not isinstance(tol, int) or tol < 0:
            raise ValueError("tol must be a non-negative int")

        # ------------------------------------------------ ref blocks ------
        anchors = [ {r: "".join(tpl([r])) for r in loc}  # per-template dict
                    for tpl in design ]

        def _ham(a, b):     # tiny helper
            return sum(x != y for x, y in zip(a, b))

        # ------------------------------------------------ core op ---------
        def cr_filter_fuzzy(data):

            for sample in data:
                self._transform_check(sample, inspect.stack()[0][3])

                arr   = sample[where]
                seqs  = np.asarray(["".join(row) for row in arr])

                # NB: DO NOT collapse – we want all template columns intact

                for tpl_idx, tpl in enumerate(design):

                    ref_blocks = anchors[tpl_idx]

                    # evaluate ALL rows for this template --------------
                    keep_vec = np.zeros(len(seqs), dtype=bool)

                    for ridx, seq in enumerate(seqs):
                        ok = True
                        for block in ref_blocks.values():
                            L = len(block)
                            hit = False
                            for p in range(len(seq) - L + 1):
                                if _ham(seq[p:p+L], block) <= tol:
                                    hit = True
                                    break
                            if not hit:
                                ok = False
                                break
                        keep_vec[ridx] = ok

                    # combine with previous state (logical AND) --------
                    sample._internal_state[:, tpl_idx] &= keep_vec

                # drop reads that failed for *every* template ----------
                sample(np.any(sample._internal_state, axis=1))
            return data

        return cr_filter_fuzzy

    def filt_ambiguous(self, where=None):
        '''
        For each sample in Data, filter out sequences not containing intact ambiguous 
        tokens. For DNA, these are "N" nucleotides, which Illumina NGS routines occasionally
        assign during base calling. For peptides, these are any sequences containing
        amino acids outside of the translation table specification.	
    
        Parameters:
                   where: 'dna' or 'pep' to specify which dataset the op 
                          should work on.
						  
        Returns:
                Transformed Data object containg entries without ambiguous
                tokens
        '''
        self._where_check(where)  
        
        #fetch the relevant monomer sets
        if where == 'pep':
            allowed_monomers = self.constants.aas
            
        elif where == 'dna':
            allowed_monomers = self.constants.bases
            
        def filter_ambiguous(data):      
            for sample in data:
                
                self._transform_check(sample, inspect.stack()[0][3])
                arr = sample[where]
                
                #perform the check; a little annoying because pads are also technically not allowed
                ind = np.in1d(arr, allowed_monomers).reshape(arr.shape)
                ind = np.sum(ind, axis=1) == self._L_summary(arr)
            
                #filter the sample
                sample(ind)
            
            return data      
        return filter_ambiguous

    def drop_data(self, where=None):
        '''
        For each sample in Data, delete datasets specified in 'where'. See documentation 
        on Data objects above for more information.
    
        Parameters:
                   where: 'dna', 'pep' or 'q' to specify which datasets 
                          should be dropped. 				
						  
        Returns:
                Transformed Data object without dropped datasets
        '''
        if where not in ('pep', 'dna', 'q'):
            msg = f"Invalid argument passed to <drop_dataset> routine. Expected where = any of ('pep', 'dna', 'q'); got: {where}"
            self.logger.error(msg)
            raise ValueError(msg)
        
        def drop_dataset(data):
            
            for sample in data:
                sample.drop(where)
                
            return data
        return drop_dataset

    def q_score_report(self, minQ: int | None = None, per_read_limit: int = 20,
                       basewise: bool = False, show_dna: bool = False,
                       dna_max_len: int = 300, dna_wrap: int = 0):
        """
        Print Q-score diagnostics to the terminal (logger):
          • Normalizes Q to numeric Phred.
          • Ignores padding columns (spaces/zeros), checks only real bases.
          • Logs global stats and per-read summaries (first `per_read_limit` reads).
          • If `minQ` is given (e.g., 75), uses numeric Phred; if data are numeric and
            minQ looks ASCII-like (>60), automatically interprets as minQ-33.
          • If `show_dna=True`, also logs the DNA sequence under review (unpadded).
        """
        import numpy as np
        import inspect

        def _to_phred_numeric(q2d: np.ndarray) -> np.ndarray:
            if not hasattr(q2d, "ndim") or q2d.ndim != 2:
                raise ValueError("q_score_report expects sample.Q as 2-D matrix after transform()")
            kind = q2d.dtype.kind
            if kind in ("u", "i", "f"):
                qmin = int(np.nanmin(q2d)) if q2d.size else 0
                qmax = int(np.nanmax(q2d)) if q2d.size else 0
                if 33 <= qmin <= 126 and 33 <= qmax <= 126:  # ASCII codes -> Phred
                    return (q2d.astype(np.int16) - 33)
                return q2d.astype(np.int16)
            if kind in ("U", "S"):
                to_ord = np.frompyfunc(lambda ch: ord(ch) - 33, 1, 1)
                return to_ord(q2d).astype(np.int16)
            out = np.empty(q2d.shape, dtype=np.int16)
            it = np.nditer(q2d, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
            while not it.finished:
                v = it[0].item()
                if isinstance(v, (int, float, np.integer, np.floating)):
                    ival = int(v)
                    out[it.multi_index] = ival - 33 if (33 <= ival <= 126) else ival
                else:
                    s = str(v)
                    out[it.multi_index] = (ord(s[0]) - 33) if s else 0
                it.iternext()
            return out

        def _valid_base_mask(D2d: np.ndarray) -> np.ndarray:
            # True where D contains a real base; handles char and numeric ASCII.
            kind = D2d.dtype.kind
            if kind in ("U", "S"):
                return (D2d != " ") & (D2d != "")
            if kind in ("u", "i"):
                v = D2d
                upper = (v >= 65) & (v <= 90)     # 'A'..'Z'
                lower = (v >= 97) & (v <= 122)    # 'a'..'z'
                return upper | lower
            return (D2d.astype("U") != " ")

        def _row_dna_string(D2d: np.ndarray, row_idx: int, valid_row_mask: np.ndarray) -> str:
            """Return unpadded DNA string for row_idx restricted to valid bases."""
            row = D2d[row_idx]
            if D2d.dtype.kind in ("U", "S"):
                vals = row[valid_row_mask]
                return "".join(vals.tolist())
            if D2d.dtype.kind in ("u", "i"):
                vals = row[valid_row_mask]
                try:
                    return "".join(chr(int(x)) for x in vals)
                except Exception:
                    return "".join(str(int(x)) for x in vals)
            # fallback
            vals = row[valid_row_mask].astype("U")
            return "".join(vals.tolist())

        def _wrap(s: str, w: int) -> str:
            if not w or w <= 0:
                return s
            return "\n".join(s[i:i+w] for i in range(0, len(s), w))

        def _op(data):
            for sample in data:
                self._transform_check(sample, inspect.stack()[0][3])
                D, Q = sample.D, sample.Q
                vb = _valid_base_mask(D)
                q = _to_phred_numeric(Q)

                flat = q[vb]
                if flat.size == 0:
                    self.logger.info("[q_score_report] no valid bases in this sample.")
                    continue

                gmin = int(flat.min())
                gmed = float(np.median(flat))
                gmean = float(flat.mean())
                g95 = float(np.quantile(flat, 0.95))
                self.logger.info(f"[q_score_report] bases={flat.size}, rows={D.shape[0]}, "
                                 f"global Phred: min={gmin}, median={gmed:.1f}, mean={gmean:.1f}, p95={g95:.1f}")

                thr = None
                if minQ is not None:
                    thr = int(minQ)
                    if thr > 60 and flat.max() <= 60:
                        self.logger.info(f"[q_score_report] Interpreting minQ={minQ} as ASCII; using {minQ-33} on numeric Phred.")
                        thr = minQ - 33
                    frac_bases = float((flat >= thr).sum()) / flat.size
                    # per-read fractions
                    lens = vb.sum(axis=1)
                    pr_frac = []
                    reads_all = 0
                    for r in range(D.shape[0]):
                        if lens[r] == 0:
                            continue
                        vals = q[r, vb[r]]
                        ok = (vals >= thr)
                        pr_frac.append(ok.mean())
                        if ok.all():
                            reads_all += 1
                    pr_frac = np.array(pr_frac) if pr_frac else np.array([0.0])
                    p50, p90, p95, p99 = np.quantile(pr_frac, [0.5, 0.9, 0.95, 0.99])
                    self.logger.info(f"[q_score_report] threshold={thr} (numeric Phred); "
                                     f"bases>=thr={frac_bases*100:.1f}%; "
                                     f"reads all>=thr={reads_all}/{(lens>0).sum()}; "
                                     f"per-read frac>=thr: 50%={p50:.2f}, 90%={p90:.2f}, 95%={p95:.2f}, 99%={p99:.2f}")

                # Per-read details (first per_read_limit)
                limit = min(int(per_read_limit), D.shape[0])
                lens = vb.sum(axis=1)
                for i in range(limit):
                    if lens[i] == 0:
                        continue
                    vals = q[i, vb[i]]
                    mu = float(vals.mean())
                    mn = int(vals.min())
                    p5 = float(np.quantile(vals, 0.05))
                    p95i = float(np.quantile(vals, 0.95))
                    if thr is None:
                        self.logger.info(f"[q_score_report] read#{i}: len={int(lens[i])}, "
                                         f"mean={mu:.1f}, min={mn}, p5={p5:.1f}, p95={p95i:.1f}")
                    else:
                        frac_ok = float((vals >= thr).mean())
                        self.logger.info(f"[q_score_report] read#{i}: len={int(lens[i])}, "
                                         f"mean={mu:.1f}, min={mn}, p5={p5:.1f}, p95={p95i:.1f}, "
                                         f"frac>=thr={frac_ok:.2f}")

                    if show_dna:
                        seq = _row_dna_string(D, i, vb[i])
                        seq_len = len(seq)
                        if seq_len > dna_max_len > 0:
                            disp = seq[:dna_max_len] + f"... (total {seq_len} nt)"
                        else:
                            disp = seq
                        disp = _wrap(disp, dna_wrap)
                        self.logger.info(f"[q_score_report] read#{i} DNA ({seq_len} nt):\n{disp}")

                    if basewise:
                        # show up to 1000 base-wise values to avoid spam
                        L_show = min(1000, vals.size)
                        preview = " ".join(str(int(x)) for x in vals[:L_show])
                        self.logger.info(f"[q_score_report] read#{i} first {L_show} Q: {preview}")

            return data

        _op.__name__ = "q_score_report"
        return _op



    def q_score_filt(self, minQ: int, loc: list[int] | None = None, frac: float = 1.0):
        """
        Filter reads by per-base quality (numeric Phred after normalization).
        - Normalize Q exactly once based on valid (non-pad) bases.
        - Ignores padding columns in D (spaces OR zeros); only real bases are checked.
        - If minQ looks like ASCII (e.g., 75) and data are numeric, uses (minQ - 33).
        """
        import numpy as np
        import inspect

        if not isinstance(minQ, int):
            raise ValueError("<q_score_filt> expects minQ as int")

        use_full_read = (loc is None) or (loc == "all")
        if not use_full_read:
            self._loc_check(loc, self.D_design)

     # ---- helpers ---------------------------------------------------------
        def _to_ascii_codes(q2d: np.ndarray) -> np.ndarray:
            """
            Return a numeric matrix of ASCII codes (33..126) if Q is char-like,
            or the original numeric values if Q is already numeric. NO -33 HERE.
            """
            if not hasattr(q2d, "ndim") or q2d.ndim != 2:
                raise ValueError("q_score_filt expects sample.Q as 2-D matrix after transform()")
            kind = q2d.dtype.kind
            if kind in ("u", "i", "f"):
                return q2d.astype(np.int16)                     # keep numbers as-is
            if kind in ("U", "S"):
                to_ord = np.frompyfunc(lambda ch: ord(ch) if ch else 0, 1, 1)
                return to_ord(q2d).astype(np.int16)            # ASCII codes
            # object / mixed: numbers stay; strings → ord(char)
            out = np.empty(q2d.shape, dtype=np.int16)
            it = np.nditer(q2d, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
            while not it.finished:
                v = it[0].item()
                if isinstance(v, (int, float, np.integer, np.floating)):
                    out[it.multi_index] = int(v)
                else:
                    s = str(v)
                    out[it.multi_index] = (ord(s[0]) if s else 0)
                it.iternext()
            return out

        def _valid_base_mask(D2d: np.ndarray) -> np.ndarray:
            """
            True where D contains a real base (A/C/G/T/N/letters).
            Handles char matrices (U/S) and numeric (ASCII code) with zero/space padding.
            """
            kind = D2d.dtype.kind
            if kind in ("U", "S"):
                return (D2d != " ") & (D2d != "")
            if kind in ("u", "i"):
                v = D2d
                upper = (v >= 65) & (v <= 90)    # 'A'..'Z'
                lower = (v >= 97) & (v <= 122)   # 'a'..'z'
                return upper | lower
            return (D2d.astype("U") != " ")

        def q_score_filter(data):
            for sample in data:
                # Ensure transformed 2-D matrices
                self._transform_check(sample, inspect.stack()[0][3])

                D = sample.D
                Q = sample.Q

                vb_all = _valid_base_mask(D)          # 2-D boolean: real bases only

                # Step 1: get a numeric matrix (ASCII codes if chars; raw if numeric)
                q_raw = _to_ascii_codes(Q)

                # Step 2: normalize ONCE to numeric Phred using ONLY valid cells
                flat_raw_valid = q_raw[vb_all]
                if flat_raw_valid.size and 33 <= int(flat_raw_valid.min()) <= 126 and 33 <= int(flat_raw_valid.max()) <= 126:
                    # Looks like ASCII-coded qualities → convert to Phred
                    q_phred = (q_raw - 33).astype(np.int16)
                else:
                    # Already numeric Phred (or something else outside ASCII range)
                    q_phred = q_raw.astype(np.int16)

                # Step 3: threshold (ASCII→numeric) decision also from valid cells
                flat = q_phred[vb_all]
                thr = int(minQ)
                if thr > 60 and flat.size and int(flat.max()) <= 60:
                    self.logger.info(f"[q_score_filt] Interpreting minQ={minQ} as ASCII; using {minQ-33} on numeric Phred.")
                    thr = minQ - 33

                # Per-template decisions, intersecting with vb_all (and optional loc mask)
                for i, template in enumerate(self.D_design):
                    row_mask = sample._internal_state[:, i]
                    rows = np.where(row_mask)[0]
                    if rows.size == 0:
                        continue

                    tmask = None if use_full_read else template(loc, return_mask=True)  # 1-D across columns
                    keep_vec = np.zeros(rows.size, dtype=bool)

                    for idx, r in enumerate(rows):
                        vb = vb_all[r]                       # valid columns for this row
                        if tmask is None:
                            eff = vb
                        else:
                            L = min(vb.shape[0], tmask.shape[0])
                            eff = vb[:L] & tmask[:L]

                        cols = np.nonzero(eff)[0]
                        if cols.size == 0:
                            keep_vec[idx] = False
                            continue

                        qvals = q_phred[r, cols]
                        if frac >= 1.0:
                            keep_vec[idx] = bool(np.all(qvals >= thr))
                        else:
                            need = int(np.ceil(qvals.size * float(frac)))
                            keep_vec[idx] = int(np.sum(qvals >= thr)) >= need

                    sample._internal_state[row_mask, i] = keep_vec

                # Drop reads rejected for all templates
                sample(np.any(sample._internal_state, axis=-1))
            return data

        return q_score_filter




    def fetch_at(self, where=None, loc=None):
        '''
        For each sample in Data, for a dataset specified by 'where', fetch the regions
        specified by 'loc' and discard other sequence regions.
        
        Collapses sample's internal state.
        See documentation on Data objects for more information.
    
        Parameters:
                   where: 'dna' or 'pep' to specify which dataset the op 
                          should work on.
						  
                     loc: a list of ints to specify regions to be fetched 
						  
        Returns:
                Transformed Data object		
        '''
        self._where_check(where)               
        if where == 'pep':
            design = self.P_design
        
        elif where == 'dna':
            design = self.D_design
        
        self._loc_check(loc, design)        
        
        def fetch_region(data):
            for sample in data:
                
                self._transform_check(sample, inspect.stack()[0][3])
                arr = sample[where]
                
                if not sample._is_collapsed:
                    msg = f"<fetch_region> routine will collapse sample {sample.name}'s internal state"
                    self.logger.info(msg)
                    sample._collapse_internal_state()
                
                #initialize the array to hold the results
                max_len = self._find_max_len(design, loc)
                result = np.zeros((arr.shape[0], max_len), dtype=arr.dtype)
                
                for i, template in enumerate(design):
                    
                    col_mask = template(loc, return_mask=True)                        
                    row_mask = sample._internal_state[:,i]
                    
                    result[row_mask, :len(col_mask)] = arr[row_mask][:,col_mask]
                    sample[where] = result
                    
            #reindex the library design accordingly so that the downstream ops
            #can still be called with originally defined loc pointers
            design.truncate_and_reindex(loc)
                    
            return data
        return fetch_region
    
    def fetch_at_fuzzy(self, *, where: str = "pep", loc: list[int], tol: int = 0,
                       pad: str = "", keep_design: bool = True):
        """
        Fuzzy version of fetch_at:
          • works even when the variable regions (VRs) are longer / shorter
            in different reads;
          • constant blocks may differ from the template by ≤ `tol`
            substitutions.
        It keeps ONLY the regions in `loc` and discards everything else
        (DNA and Q arrays are NOT touched).

        Parameters
        ----------
        where : {"pep", "dna"}
            Dataset on which to operate.
        loc : list[int]
            Region indices (0-based) from LibraryDesign.loc to retain.
        tol : int, default 0
            Maximum per-block Hamming distance allowed when identifying
            constant regions.
        pad : str, default ""
            Character used to pad the right side so that the result is still
            a rectangular ndarray.
        keep_design : bool, default False
            If True the LibraryDesign object is NOT truncated; otherwise it
            is shrunk to the retained regions so that downstream loc values
            stay valid.

        Returns
        -------
        callable  – ready to `pip.enque()`
        """
        self._where_check(where)
        design = self.P_design if where == "pep" else self.D_design
        self._loc_check(loc, design)
        if tol < 0 or not isinstance(tol, int):
            raise ValueError("tol must be a non-negative int")

        def _ham(a: str, b: str) -> int:
            return sum(x != y for x, y in zip(a, b))

        refs = []
        n_regions = design.loc.max() + 1
        anchors = []
        for tpl in design:
            anchors.append({i: "".join(tpl([i])) 
                            for i in range(n_regions)
                            if not design.is_vr[i]}) 

        def fetch_fuzzy(data):
            for sample in data:
                self._transform_check(sample, inspect.stack()[0][3])
                if not sample._is_collapsed:
                    sample._collapse_internal_state()

                arr   = sample[where]
                seqs  = ["".join(r).rstrip(pad) for r in arr]
                out_snippets = []
                max_len = 0

        # ------------------------------------------------------------------
        # iterate read-by-read
        # ------------------------------------------------------------------
                for ridx, seq in enumerate(seqs):
                    flags = sample._internal_state[ridx, :len(design)]
                    if not np.any(flags):               # nothing matched → keep empty
                        out_snippets.append("")
                        continue

                    tpl_idx = flags.argmax()            # library template that matched
                    anchor  = anchors[tpl_idx]

                    piece, pos = [], 0                  # ← moving cursor

            # go through the requested blocks *in order*
                    for r in loc:
                        if design.is_vr[r]:
                            if (r+1) not in anchor:
                                continue
                    # left boundary is current cursor -----------------------
                            left = pos

                    # try finding the right-hand anchor (CR r+1) ------------
                            right = None
                            if (r + 1) in anchor:
                                refR = anchor[r + 1]
                                Lr   = len(refR)
                                for p in range(left, len(seq) - Lr + 1):
                                    if _ham(seq[p:p+Lr], refR) <= tol:
                                        right = p
                                        break

                    # no right anchor? – cut at read end --------------------
                            if right is None:
                                continue

                            piece.append(seq[left:right])
                            pos = right                      # advance the cursor

                        else:                               # constant region
                            ref = anchor[r]; Lc = len(ref)
                            found = False                   # search *after* the cursor
                            for p in range(pos, len(seq) - Lc + 1):
                                if _ham(seq[p:p+Lc], ref) <= tol:
                                    piece.append(seq[p:p+Lc])
                                    pos   = p + Lc
                                    found = True
                                    break
                            if not found:
                                # skip the block if the anchor cannot be located
                                continue

                    frag = "".join(piece)
                    max_len = max(max_len, len(frag))
                    out_snippets.append(frag)

        # build a padded ndarray -------------------------------------------
                out = np.full((len(out_snippets), max_len), pad, dtype=arr.dtype)
                for i, frag in enumerate(out_snippets):
                    out[i, :len(frag)] = list(frag)

                sample[where] = out

            if not keep_design:
                design.truncate_and_reindex(loc)
            return data
        return fetch_fuzzy

    def unpad(self):
        '''
        For each sample in Data, unpads the D, Q, P arrays. For each array, removes 
    	the columns where every value is a padding token. See documentation on Data 
        objects for more information.

        Parameters:
                None	
						  
        Returns:
                Transformed Data object		
        '''        
        def unpad_data(data):
            for sample in data:
                sample.unpad()            
                                           
            return data            
        return unpad_data

    #--------------------------------------------
    #The methods below do not transform the data.
    #They are only used to assemble statistics, 
    #plots the results, etc.
    #--------------------------------------------
    def len_summary(self, where=None, save_txt=False):
        '''
        For each sample in Data, compute the distribution of peptide/DNA sequence lengths
        (specified by 'where') and plot the resulting histogram in the parser output folder
        as specified by config.py. Optionally, the data can also be written to a txt file.
    
        Parameters:
                   where: 'dna' or 'pep' to specify which dataset the op 
                          should work on.
                          						  
                save_txt: if True, the data will be written to a txt file saved
                          in the same folder as the .png and .svg plots				
						  
        Returns:
                Data object (no transformation)
        '''
        self._where_check(where)
        def length_summary(data):
            
            self._prepare_destinations(data)        
            for sample in data:
                
                self._transform_check(sample, inspect.stack()[0][3])
                arr = sample[where]
                
                L = self._L_summary(arr)            
                L, counts = np.unique(L, return_counts=True)
                
                destination = os.path.join(self.dirs.parser_out, sample.name)
                fname = f'{sample.name}_{where}_L_distribution'
                basename = os.path.join(destination, fname)
                Plotter.SequencingData.L_distribution(L, counts, where, basename)
                
                if save_txt:
                    np.savetxt(basename + '.csv',
                               np.array((L, counts)).T,
                               delimiter=',', 
                               header='Seq length,Count')
            
            return data
        return length_summary

    def convergence_summary(self, where=None):
        '''
        For each sample in Data, perform basic library convergence analysis on a sequence 
        level. Computes normalized Shannon entropy, and postition-wise sequence conservation. 
        Plots the results in the parser output folder as specified by config.py.
    
        Parameters:
                   where: 'dna' or 'pep' to specify which dataset the op 
                          should work on.
                          						  							  
        Returns:
                Data object (no transformation)
        '''
        self._where_check(where)
        
        if where == 'pep':
            tokens = self.constants.aas
                    
        elif where == 'dna':               
            tokens = self.constants.bases
        
        from utils.misc import shannon_entropy, get_freqs
        def _seq_conservation(freq):
            '''
            NOTE: this computation doesn't really make sense for
            arrays containing sequences of uneven length. Rather,
            the meaning becomes somewhat counterintuitive, but
            what to do about it?
            '''
            with np.errstate(divide='ignore', invalid='ignore'):
                em = np.nan_to_num(np.multiply(freq, np.log2(freq)))

            return np.sum(em, axis=0) + np.log2(freq.shape[0])    
            
        def library_convergence_summary(data):

            self._prepare_destinations(data)
            for sample in data:

                self._transform_check(sample, inspect.stack()[0][3])                
                arr = sample[where]
                
                shannon, counts = shannon_entropy(arr, norm=True)
                freq = get_freqs(arr, tokens)
                seq_conservation = _seq_conservation(freq)
                
                destination = os.path.join(self.dirs.parser_out, sample.name)
                fname = f'{sample.name}_{where}_library_convergence'
                basename = os.path.join(destination, fname)
                Plotter.SequencingData.dataset_convergence(counts, shannon, where, basename)                
                
                fname = f'{sample.name}_{where}_sequence_conservation'
                basename = os.path.join(destination, fname)                
                Plotter.SequencingData.conservation(seq_conservation, where, basename)
                
            return data
        return library_convergence_summary

    def freq_summary(self, where=None, loc=None, save_txt=False):
        '''
        Perform basic library convergence analysis at a token level. For each sample in Data, 
        computes the frequency of each token in the dataset. Plots the results in the parser 
        output folder as specified by config.py. Optionally, the data can also be written to
        a txt file.

        Parameters:
                   where: 'dna' or 'pep' to specify which dataset the op 
                          should work on.
						  
                     loc: a list of ints to specify regions to be analyzed;
                          in this case, the op will collapse sample's internal
                          state (see explanation for Data objects)
                          
                          OR
                          
                          'all': to get the same statistics over the entire sequence;
                                 in this case, the op will NOT collapse sample's 
                                 internal state

                save_txt: if True, the data will be written to a txt file saved
                          in the same folder as the .png and .svg plots						 
                          						  							  
        Returns:
                Data object (no transformation)
        '''
        self._where_check(where)
        if where == 'pep':
            design = self.P_design
            tokens = self.constants.aas
            
        elif where == 'dna':
            design = self.D_design        
            tokens = self.constants.bases
            
        if loc != 'all':
            self._loc_check(loc, design)
            
        from utils.misc import get_freqs
        
        def frequency_summary(data):
            self._prepare_destinations(data)
            for sample in data:
                
                self._transform_check(sample, inspect.stack()[0][3])
                arr = sample[where]
                
                if loc == 'all':
                    freq = get_freqs(arr, tokens)
                    
                else:
                    #array internal state has to be collapsed for this calculation
                    if not sample._is_collapsed:
                        msg = f"<frequency_summary> routine will collapse sample {sample.name}'s internal state"
                        self.logger.info(msg)
                        sample._collapse_internal_state()
                    
                    #initialize the frequency array: 3D array to be reduced along axis 0 at the end
                    maxlen = self._find_max_len(design, loc)
                    freq = np.zeros((len(design), len(tokens), maxlen), dtype=np.float32)
                    
                    for i,template in enumerate(design):                    
                        
                        row_mask = sample._internal_state[:,i]
                        col_mask = template(loc, return_mask=True)
    
                        #calculated weighed contributions of each design
                        #to the overall frequency array
                        norm = np.divide(np.sum(row_mask), arr.shape[0])
                        freq[i,:,:len(col_mask)] = norm * np.nan_to_num(get_freqs(arr[row_mask][:,col_mask], tokens))
    
                    #reduce back to a 2D array and plot/save
                    freq = np.sum(freq, axis=0)
                
                if loc == 'all':
                    nloc = 'overall'
                    fname =f'{sample.name}_{where}_overall_tokenwise_frequency'
                else:
                    nloc =  ', '.join(str(x + 1) for x in loc)
                    fname = f'{sample.name}_{where}_reg{nloc}_tokenwise_frequency'
                
                destination = os.path.join(self.dirs.parser_out, sample.name)
                basename = os.path.join(destination, fname)
                Plotter.SequencingData.tokenwise_frequency(freq, tokens, where, nloc, basename)  

                if save_txt:
                    
                    np.savetxt(basename + '.csv',
                               freq,
                               delimiter=',')                
                    
            return data
        return frequency_summary

    def q_summary(self, loc=None, save_txt=False):
        '''
        For each sample in Data, compute some basic Q score statistics.
    	For each position in regions specified by 'loc', computes the mean and standard deviation
        of Q scores. Plots the results in the parser output folder as specified by config.py.
        Optionally, the data can also be written to a txt file.
        	    	
        Parameters:					  
                     loc: a list of ints to specify regions to be analyzed;
                          in this case, the op will collapse sample's internal
                          state (see explanation for Data objects)
                          
                          OR
                          
                          'all': to get the same statistics over the entire Q 
                                 score arrays. in this case, the op will NOT 
                                 collapse sample's internal state
                          
                save_txt: if True, the data will be written to a txt file saved
                          in the same folder as the .png and .svg plots						 
                          						  							  
        Returns:
                Data object (no transformation)	
        '''
        
        if loc != 'all':
            self._loc_check(loc, self.D_design)
            
        def q_score_summary(data):
            
            self._prepare_destinations(data)
            for sample in data:
                self._transform_check(sample, inspect.stack()[0][3])
                
                if loc == 'all':
                    relevant_arr = sample.Q.astype(np.float32)
                    
                else:
                    arr = sample.Q
                    if not sample._is_collapsed:
                        msg = f"<q_score_summary> routine will collapse sample {sample.name}'s internal state"
                        self.logger.info(msg)
                        sample._collapse_internal_state()
                                    
                    maxlen = self._find_max_len(self.D_design, loc)
                    #iterate over templates and append all of the relevant arr views to this array
                    #relevant view: masked (row/columnwise) arr
                    relevant_arr = []
                    
                    for i,template in enumerate(self.D_design):
                        
                        row_mask = sample._internal_state[:,i]
                        col_mask = template(loc, return_mask=True)                
                        
                        arr_view = np.zeros((np.sum(row_mask), maxlen), dtype=np.float32)
                        arr_view[:,:len(col_mask)] = arr[row_mask][:,col_mask]
                        relevant_arr.append(arr_view)
    
                    #assemble into a single array
                    relevant_arr = np.vstack(relevant_arr)
                    
                #mask out pads (0) as nans for nanmean/nanstd statistics
                relevant_arr[relevant_arr == 0] = np.nan
                
                #get the stats; plot
                q_mean = np.nanmean(relevant_arr, axis=0)
                q_std = np.nanstd(relevant_arr, axis=0)

                destination = os.path.join(self.dirs.parser_out, sample.name)
                if loc == 'all':
                    nloc = 'overall'
                    fname = f'{sample.name}_overall_q_score_summary'
                else:
                    nloc =  ', '.join(str(x + 1) for x in loc)
                    fname = f'{sample.name}_reg{nloc}_q_score_summary'
                    
                basename = os.path.join(destination, fname)
                Plotter.SequencingData.Q_score_summary(q_mean, q_std, nloc, basename)  

                if save_txt:
                    q = np.vstack((q_mean, q_std))
                    np.savetxt(basename + '.csv',
                               q.T,
                               delimiter=',',
                               header='Q mean, Q std')  
                
            return data
        return q_score_summary

    def count_summary(self, where=None, top_n=None, fmt=None):
        '''
        For each sample in Data, counts the number of times each unique sequence is found in the
        dataset specified by 'where'. The results are written to a file in the parser output  
        folder as specified by config.py.
        
        Parameters:					  
                   where: 'dna' or 'pep' to specify which dataset the op 
                          should work on.

                   top_n: if None, full summary will be created. If
                          an int is passed, only top_n sequences (by count)
                          will be written to a file.

                     fmt: the format of the output file. Supported values are
                          'csv' and 'fasta'.					 
                          						  							  
        Returns:
                Data object (no transformation)
        '''
        self._where_check(where)
        
        if fmt not in ('csv', 'fasta'):
            msg = f"<count_summary> routine received invalid fmt argument. Acceted any of ('csv', 'fasta'); received: {fmt}"
            self.logger.error(msg)
            raise ValueError(msg)
            
        if top_n is not None:
            if not isinstance(top_n, int):
                msg = f'<count_summary> routine expected to receive parameter top_n as as int; received: {type(top_n)}'
                self.logger.error(msg)  
                raise ValueError(msg)

        def _writer(sample, og_ind, counts, fmt, path):
            if fmt == 'csv':           
                df = pd.DataFrame(columns=['Peptide', f'{where} count', 'DNA'])
                df['Peptide'] = [''.join(x) for x in sample.P[og_ind]]
                df['DNA'] = [''.join(x) for x in sample.D[og_ind]]
                df[f'{where} count'] = counts
                df.to_csv(path + '.csv', sep=',')
                
            if fmt == 'fasta':
                
                arr = sample[where][og_ind]
                arr_1d = [''.join(x) for x in arr]    
                
                with open(path + '.fasta', 'w') as f:
                    for i,seq in enumerate(arr_1d):
                        f.write(f'>seq_{i+1}_count_{counts[i]}\n')  
                        f.write(f'{seq}\n')
                return 
        
        def full_count_summary(data):
            self._prepare_destinations(data)
            for sample in data:
                
                self._transform_check(sample, inspect.stack()[0][3])
                arr = sample[where]            
                
                #count entries in the array
                unique, og_ind, counts = np.unique(arr, axis=0, 
                                                   return_counts=True,
                                                   return_index=True)                
                
                #if top_n is unset, ind array will index every entry in the sample
                ind = np.argsort(counts)[::-1][:top_n]

                og_ind = og_ind[ind]
                counts = counts[ind]
                
                destination = os.path.join(self.dirs.parser_out, sample.name)
                fname = f'{sample.name}_{where}_count_summary'
                path = os.path.join(destination, fname)
                
                _writer(sample, og_ind, counts, fmt, path)
                                
            return data
        return full_count_summary
    
    def template_summary(self, where=None):
        '''
        For each sample in Data, compute the number of matches between the dataset 
        specified by 'where' and the corresponding library templates. The results 
        are written to a file in the parser output folder as specified by config.py.
        
        In other words, summarize where dataset sequences come from (from which
        libraries). The op could also be called "_internal_state_summary"
        
        Parameters:					  
                   where: 'dna' or 'pep' to specify which dataset the op 
                          should work on                        						  							  
        Returns:
                Data object (no transformation)
        '''        
        
        self._where_check(where)
        if where == 'pep':
            design = self.P_design
        
        elif where == 'dna':
            design = self.D_design
            
        def template_breakdown(data):
            self._prepare_destinations(data)
       
            #summarize straight into a pandas dataframe
            sample_names = [sample.name for sample in data]
            templates = [template.lib_seq for template in design]
        
            #all this op is: axis=0-wide sum of the internal states
            import pandas as pd
            df = pd.DataFrame(index=sample_names, columns=templates)
            
            for sample in data:
                self._transform_check(sample, inspect.stack()[0][3])
                #df.loc[sample.name] = np.sum(sample._internal_state, axis=0)
                counts = np.sum(sample._internal_state[:, :len(design)], axis = 0)
                df.loc[sample.name] = counts
                
            fname = f'{self.exp_name}_by_template_breakdown.csv'
            path = os.path.join(self.dirs.logs, fname)            
            df.to_csv(path + '.csv', sep=',')
    
            return data
        return template_breakdown

    def tSNE_analysis(self, where=None, top_n=1000, cluster_fasta=False):
        '''
        For each sample in Data, compute tSNE embeddings for the dataset 
        specified by 'where'. Cluster the results (HDBSCAN) and summarize. 
        
        Parameters:					  
                   where: 'dna' or 'pep' to specify which dataset the op 
                          should work on

                   top_n: the number of top entries (by count) to analyze   

           cluster_fasta: True/False. if True, create a .fasta file for
                          sequences comprising each cluster. 
          						  						   
        Returns:
                Data object (no transformation)
        '''        

        self._where_check(where)
        if not isinstance(top_n, int):
            msg = f'<tSNE_analysis> routine expected to receive parameter top_n as as int; received: {type(top_n)}'
            self.logger.error(msg)  
            raise ValueError(msg)        

        if not isinstance(cluster_fasta, bool):
            msg = f'<tSNE_analysis> routine expected to receive parameter cluster_fasta as as bool; received: {type(cluster_fasta)}'
            self.logger.error(msg)  
            raise ValueError(msg)        

        try:
            from sklearn.manifold import TSNE
            from sklearn.preprocessing import StandardScaler
            import hdbscan
        except:
            msg = 'Failed to import the libraries necessary for <tSNE_analysis> op. . .'
            self.logger.error(msg)
            raise ImportError(msg)

        def _cluster_writer(df, destination, sname):
            clusters = np.unique(df['cluster'])
            for cluster in clusters:
                fname = f'{sname}_{where}_HDBSCAN_cluster_{cluster}.fasta'
                path = os.path.join(destination, fname)
                
                with open(path, 'w') as f:
                    for entry in df[df['cluster'] == cluster].iterrows():
                        f.write(f'>cluster_{cluster}_count_{entry[1]["counts"]}\n')  
                        f.write(f'{entry[1]["sequence"]}\n')
            return        

        def tSNE_embedding(data):
            for sample in data:
                
                self._transform_check(sample, inspect.stack()[0][3])
                arr = sample[where]
                
                #get an array of top_n entries, X; C - counts array
                X, C = np.unique(arr, return_counts=True, axis=0)
                ind = np.argsort(C)[::-1][:top_n]
                X = X[ind]
                C = C[ind]
                
                #get one-hot encodings of the datasets
                tokens = np.unique(arr)
                nX = np.vectorize(lambda x: np.where(tokens == x)[0][0])(X)
                nX = nX.astype(np.int8).ravel()
                
                fX = np.zeros((X.size, tokens.size))
                fX[np.arange(nX.size), nX] = 1
                fX = np.reshape(fX, (X.shape[0], -1))
                
                #embed the representations and scale
                embedder = TSNE(n_components=2,
                                perplexity=30, 
                                learning_rate=200,
                                init='random')
                
                Y = embedder.fit_transform(fX.astype(np.float32))
                Y = StandardScaler().fit_transform(Y)
                
                #run HDBSCAN clustering
                hdb = hdbscan.HDBSCAN(cluster_selection_epsilon=0.02,
                                      cluster_selection_method = 'eom')
                hdb.fit(Y)
                
                #plot the results
                destination = os.path.join(self.dirs.parser_out, sample.name, 'tSNE_analysis')
                if not os.path.isdir(destination):
                    os.makedirs(destination)       
                    
                fname = f'{sample.name}_{where}_tSNE'
                basename = os.path.join(destination, fname)         
                sizes = 8000 * np.power(np.divide(C, arr.shape[0]), 0.6)
                Plotter.Analysis.tSNE(Y, sizes, hdb.labels_, basename)
                
                #dump analysis into a .csv
                entries = np.array([''.join(x) for x in X])
                clusters = hdb.labels_ + 1
                
                d = {'sequence': entries,
                     'counts': C,
                     'cluster': clusters}
                
                df = pd.DataFrame.from_dict(d)
                df = df.sort_values(['cluster', 'counts'], ascending=False)
                fname = f'{sample.name}_{where}_HDBSCAN_clustering.csv'
                full_name = os.path.join(destination, fname)
                df.to_csv(full_name, sep=',', index=False)
                
                if cluster_fasta:
                    _cluster_writer(df, destination, sample.name)
                
            return data
        return tSNE_embedding

    #--------------------------------------------
    #Below are IO readers/writers
    #--------------------------------------------    
    def _fetch_fastq_file(self, reader):
        '''
        Fetch DNA and Q score sequence lists from a .fastq file.
        .fastq files are base call .fastqs from single pair reads
        on Illumina's MiSeq instrument.
        
        in:            
            reader: a buffered reader with a preloaded file
        
        out:            
            DNA: a list of strings each containing a single read DNA sequence
            Q:   Q-scores corresponding to individual base calls, in the same format            
        '''        
        basename = os.path.basename(reader.name)
        sample_name = os.path.splitext(basename)[0]        
 
        with reader as f:
            msg = f'Fetching {basename}. . .'
            self.logger.info(msg)
            content = f.readlines()
            
            DNA = content[1::4]
            DNA = np.array([x.rstrip('\n') for x in DNA])
            
            Q_ascii = content[3::4] 
            Q_ascii = [x.rstrip('\n') for x in Q_ascii]
            min_char = min(min(map(ord, s)) for s in Q_ascii)
            offset   = 33 if min_char < 59 else 64           # auto-detect
            Q        = np.array(
                [self._decode_q_ascii(s, offset=offset) for s in Q_ascii],
                dtype=object,
            )
            f.close()
        
        sample = SequencingSample(
                                  name=sample_name,
                                  D=DNA,
                                  Q=Q,
                                  P=None
                                 )
        return sample
 
    

    def stream_chunks_from_gz_dir(self, *, chunk_lines: int = 4_000_000, chunk_bytes: int | None = None):
        """
        A generator factory that yields Data objects chunk-by-chunk from all .fastq.gz files
        in the configured sequencing_data directory. Each yielded Data contains one SequencingSample
        named "<basename>__chunk<N>".
        Limits:
          - chunk_lines: max number of text lines per chunk (must be multiple of 4).
          - chunk_bytes: optional total byte cap per chunk across lines; if set, closes chunk
            when either lines or bytes threshold is exceeded.
        """
        if chunk_lines % 4 != 0:
            msg = f"<stream_chunks_from_gz_dir> chunk_lines must be multiple of 4; received: {chunk_lines}"
            self.logger.error(msg); raise ValueError(msg)

        fnames = [os.path.join(self.dirs.seq_data, x) for x in os.listdir(self.dirs.seq_data) if x.endswith(".fastq.gz")]
        if not fnames:
            msg = f'No .fastq.gz files were found in {self.dirs.seq_data}! Aborting.'
            self.logger.error(msg)
            raise IOError(msg)

        PHRED_OFFSET = 33
        def _make_sample(name, seqs, quals):
            # to 2D arrays
            import numpy as _np
            if not seqs:
                return SequencingSample(D=_np.empty((0,1), dtype='<U1'),
                                        Q=_np.empty((0,1), dtype=_np.uint8), P=None, name=name)
            maxL = max(len(s) for s in seqs)
            D = _np.full((len(seqs), maxL), '', dtype='<U1')
            for i, s in enumerate(seqs):
                D[i, :len(s)] = list(s)
            maxLq = max(len(q) for q in quals) if quals else 0
            Q = _np.zeros((len(quals), maxLq), dtype=_np.uint8)
            for i, q in enumerate(quals):
                qv = _np.asarray(q, dtype=_np.uint8)
                Q[i, :len(qv)] = qv
            return SequencingSample(D=D, Q=Q, P=None, name=name)

        def _iter():
            import gzip as _gzip
            for f in fnames:
                base = os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]
                with _gzip.open(f, 'rt') as fh:
                    buf_seq, buf_q = [], []
                    line_no = 0
                    chunk_id = 0
                    acc_bytes = 0
                    cur_seq, cur_q = None, None
                    for line in fh:
                        line_no += 1
                        acc_bytes += len(line.encode('utf-8', 'ignore'))
                        k = line_no % 4
                        if k == 2:
                            cur_seq = line.strip()
                        elif k == 0:
                            cur_q = line.strip()
                            buf_seq.append(cur_seq)
                            buf_q.append([ord(c) - PHRED_OFFSET for c in cur_q])
                            cur_seq, cur_q = None, None
                        # emit chunk if thresholds reached
                        if (line_no % chunk_lines == 0) or (chunk_bytes is not None and acc_bytes >= chunk_bytes):
                            chunk_id += 1
                            sample = _make_sample(f"{base}__chunk{chunk_id}", buf_seq, buf_q)
                            yield Data(samples=[sample])
                            buf_seq, buf_q = [], []
                            acc_bytes = 0
                    # tail
                    if buf_seq:
                        chunk_id += 1
                        sample = _make_sample(f"{base}__chunk{chunk_id}", buf_seq, buf_q)
                        yield Data(samples=[sample])
        return _iter
    def stream_from_fastq_dir(self, *args):
        '''
        A generator that yields data from self.fastq_dir sample by sample.
        Good when the entirety of the folder does not fit the memory.
        '''
        
        fnames = [os.path.join(self.dirs.seq_data, x) for x in os.listdir(self.dirs.seq_data) if x.endswith(".fastq")]
        if not fnames:
            msg = f'No .fastq files were found in {self.dirs.seq_data}! Aborting.'
            self.logger.error(msg)
            raise IOError(msg)
                
        for f in fnames:
            reader = open(f, 'r')
            sample = self._fetch_fastq_file(reader)
            yield sample
    
    def stream_from_gz_dir(self, *args):
        '''
        Fetch all .fastq.gz files from the sequencing_data directory 
        (as specified in config.py). Should be called as the first op in the workflow.
        
            Parameters:
                    None
        
            Returns:
                    Fetched Fastq data as an instance of Data
        '''
        fnames = [os.path.join(self.dirs.seq_data, x) for x in os.listdir(self.dirs.seq_data) if x.endswith(".gz")]
        if not fnames:
            msg = f'No .fastq.gz files were found in {self.dirs.seq_data}! Aborting.'
            self.logger.error(msg)
            raise IOError(msg)
                
        for f in fnames:
            reader = gzip.open(f, "rt")
            sample = self._fetch_fastq_file(reader)
            yield sample
                          
    def fetch_fastq_from_dir(self):
        '''
        Fetch all .fastq files from the sequencing_data directory 
        (as specified in config.py). Should be called as the first op in the workflow.
        
            Parameters:
                    None
        
            Returns:
                    Fetched Fastq data as an instance of Data
        '''
        def fetch_dir_fastq(*args):
            samples = list()
            for sample in self.stream_from_fastq_dir():
                samples.append(sample)
                
            return Data(samples=samples)
        return fetch_dir_fastq
       
    def fetch_gz_from_dir(self):
        '''
        Analogous to self.fetch_fastq_dir
        '''
        def fetch_dir_gz(*args):
            samples = list()
            for sample in self.stream_from_gz_dir():
                samples.append(sample)
                
            return Data(samples=samples)
        return fetch_dir_gz
    
    def save(self, where=None, fmt=None):
        '''
        For each sample in Data, save the dataset specified by 'where'. The results are written 
        to a file in the parser output folder as specified by config.py.
        
        Parameters:					  
                   where: 'dna' or 'pep' to specify which dataset the op 
                          should work on.

                     fmt: the format of the output file. Supported values are
                          'npy', 'fasta' and 'csv'					 
                          						  							  
        Returns:
                Data object (no transformation)
        '''
        
        if fmt not in ('npy', 'csv', 'fasta'):
            msg = f"<save_data> routine received invalid fmt argument. Acceted any of ('npy', 'csv', 'fasta'); received: {fmt}"
            self.logger.error(msg)
            raise ValueError(msg)
        
        self._where_check(where)
        def _writer(arr, fmt, path):
            
            if fmt == 'npy':
                np.save(path + '.npy', arr)    
                return
            
            arr_1d = [''.join(x) for x in arr]
            
            if fmt == 'csv':           
                with open(path + '.csv', 'w') as f:
                    for seq in arr_1d:
                        f.write(f'{seq},\n')    
                return
                
            if fmt == 'fasta':
                with open(path + '.fasta', 'w') as f:
                    for i,seq in enumerate(arr_1d):
                        f.write(f'>sequence_{i}\n')  
                        f.write(f'{seq}\n')
                return                
                
        def save_data(data):
                    
            self._prepare_destinations(data)
            for sample in data:
                
                self._transform_check(sample, inspect.stack()[0][3])
                arr = sample[where]  

                destination = os.path.join(self.dirs.parser_out, sample.name)
                fname = f'{sample.name}_{where}'
                path = os.path.join(destination, fname)
                
                _writer(arr, fmt, path)
                
            return data
        return save_data    
    
