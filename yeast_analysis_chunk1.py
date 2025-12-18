# -*- coding: utf-8 -*-
"""
Integrated Analysis Script for NGS
"""

import os
import multiprocessing
from utils.ProcessHandlers import Pipeline, FastqParser, AnarciAnnotator
from utils.lib_design import LibraryDesign
from utils.ProcessHandlers import Logger, DirectoryTracker

SEQ_DATA_PATH = '/home/zhao/NGS/test/yeast_test'
LOGS_PATH = '/home/zhao/NGS/test/yeast_test/logs'
PARSER_OUT_PATH = '/home/zhao/NGS/test/yeast_test/parser_outputs'
DELETE_INTERMEDIATE_CHUNKS = True 


# --- CONFIGURATION SECTION ---------------------------------------------------

class Config:
    experiment = 'Specifica_naiveFL_test'
    
    class TrackerConfig:
        seq_data = SEQ_DATA_PATH
        logs = LOGS_PATH
        parser_out = PARSER_OUT_PATH

    class LoggerConfig:
        name = 'Specifica_naiveFL_test_logger'
        verbose = True
        log_to_file = True
        log_fname = os.path.join(LOGS_PATH, name + '.log')

    class ParserConfig:
        barcode = ['CGCTTACATTCACGCCCT'] 
        utr5_seq = ['AAGCTTCTGCAGGCT']
        
        # DNA library design
        D_design = LibraryDesign(
            templates=['111ATCG111'],
            monomers={1: ('A', 'G', 'T', 'C')},
            lib_type='dna'
        )
        
        # Peptide library design: KLLQA + Variable
        P_design = LibraryDesign(
            templates=[
                'KLLQA111EQKLISEEDL111'
            ],
            monomers={
                1: ('A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'),
                2: ('M'),
            },
            lib_type='pep'
        )

# --- DISPATCHER OVERRIDE -----------------------------------------------------

class LocalDispatcher:
    def __init__(self, config_obj):
        self.cfg = config_obj
        
    def dispatch_handlers(self, handler_classes):
        l_conf = self.cfg.LoggerConfig
        t_conf = self.cfg.TrackerConfig
        
        shared_logger = Logger(l_conf).logger
        shared_dirs = DirectoryTracker(t_conf)
        
        instances = []
        
        for h_cls in handler_classes:
            if h_cls == Pipeline:
                meta = {
                    'exp_name': self.cfg.experiment,
                    'dirs': shared_dirs,
                    'logger': shared_logger,
                    'conf': None
                }
                inst = Pipeline(meta)
                instances.append(inst)
                
            elif h_cls == FastqParser:
                p_conf = self.cfg.ParserConfig
                meta = {
                    'dirs': shared_dirs,
                    'logger': shared_logger,
                    'barcode': p_conf.barcode,
                    'utr5_seq': p_conf.utr5_seq,
                    'D_design': p_conf.D_design,
                    'P_design': p_conf.P_design
                }
                inst = FastqParser(meta)
                instances.append(inst)
                
            elif h_cls == AnarciAnnotator:
                # Pass output dir for file finding
                inst = AnarciAnnotator(shared_dirs.parser_out, shared_logger)
                instances.append(inst)
                
        return tuple(instances)


# --- MAIN EXECUTION ----------------------------------------------------------

if __name__ == '__main__':
    
    # Initialize Dispatcher
    dispatcher = LocalDispatcher(Config)
    pip, par, annotator = dispatcher.dispatch_handlers((Pipeline, FastqParser, AnarciAnnotator))
    
    # 1. PHASE 1: Process FASTQ -> Merged CSV/FASTA (Full Extracted Region)
    pip.enque([
        par.translate_both_strands(stop_readthrough=False, utr5_offset=0, tol=2),
        par.len_filter(where='dna', len_range=[900, 1500]), 
        par.q_score_filt(minQ=30, frac=0.9),
        par.extract_fuzzy_regions(where='pep', loc=[0, 1, 2], tol=3),
        par.count_summary(where='pep', fmt='csv'),
        par.count_summary(where='pep', fmt='fasta'),
        par.unpad(),
    ])

    data_iter = par.stream_chunks_from_gz_dir(chunk_lines=200000)
    pip.run_over_stream(data_iter, save_summary=True)
    pip.merge_chunk_outputs(delete_chunks=DELETE_INTERMEDIATE_CHUNKS)
    
    # 2. PHASE 2: ANARCI Annotation
    import glob
    root_out = Config.TrackerConfig.parser_out
    csv_files = glob.glob(os.path.join(root_out, "*", "*_pep_counts.csv"))

    # Define what regions you want to extract
    # Options: 'CDR1', 'CDR2', 'CDR3', 'FR1', 'FR2', 'FR3', 'FR4', 'CDRs', 'FL', '100-110'
    REGIONS = [ 'CDR1', 'CDR2', 'CDR3', 'FR1', 'FR2', 'FR3', 'FR4', 'CDRs', 'FL', '100-110'] 
    ALLOWED_SPECIES = ['mouse'] # 'all'
    #ALLOWED_SPECIES = None # 'all'
    N_CPU = 50
    #N_CPU = multiprocessing.cpu_count()
    ASSIGN_GERMLINE = True
    
    for csv_path in csv_files:
        annotator.process_merged_csv(csv_path, extract_regions=REGIONS, output_mode='aligned', n_cpu=N_CPU, allowed_species=ALLOWED_SPECIES, assign_germline=ASSIGN_GERMLINE)