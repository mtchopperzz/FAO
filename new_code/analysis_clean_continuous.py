# -*- coding: utf-8 -*-
"""
Integrated Analysis Script for NGS
Continuous-frame translation + parallel ANARCI annotation version.
"""

import os
import multiprocessing
from utils.ProcessHandlers import Pipeline, FastqParser, EnrichmentAnalyzer
from utils.ProcessHandlers import Logger, DirectoryTracker
from utils.ContinuousIgblastExtractor import ContinuousIgblastExtractor

SEQ_DATA_PATH = '/home/zhao/data/DDB_NGS_archive/Display_Screening/Yeast/20260427/input'
LOGS_PATH = '/home/zhao/data/DDB_NGS_archive/Display_Screening/Yeast/20260427/logs'
PARSER_OUT_PATH = '/home/zhao/data/DDB_NGS_archive/Display_Screening/Yeast/20260427/parser_outputs'
DELETE_INTERMEDIATE_CHUNKS = True


class Config:
    experiment = '20260427_NGS'

    class TrackerConfig:
        seq_data = SEQ_DATA_PATH
        logs = LOGS_PATH
        parser_out = PARSER_OUT_PATH

    class LoggerConfig:
        name = 'IgBLAST_ANARCI_screening_logger'
        verbose = True
        log_to_file = True
        log_fname = os.path.join(LOGS_PATH, name + '.log')

    class ParserConfig:
        igblast_exec = 'igblastn'
        igdata_path = '/home/zhao/NGS/ncbi-igblast-1.22.0'
        database_dir = '/home/zhao/NGS/ncbi-igblast-1.22.0/database'
        species = 'human'

        linker_seq = 'TCCGGAGGGTCGACCATAACTTCGTATAATGTATACTATACGAAGTTATCCTCGAGCGGTACC'
        linker_tol_ratio = 0.1
        extract_regions = ['CDR3', 'CDRs', 'FL']

        igblast_ncpu = multiprocessing.cpu_count()
        anarci_ncpu = max(1, multiprocessing.cpu_count() - 1)
        anarci_batch_size = 5000
        frame_rescue_offsets = (0, 1, 2)
        anarci_allowed_species = None
        anarci_bit_score_threshold = 80

    class FilterConfig:
        len_range = [200, 1200]
        min_q_score = 30
        q_pass_fraction = 0.9
        chunk_lines = 100000

    class AnalysisConfig:
        visualization_specs = [
            'H_CDR3_PEP',
            'H_CDRs_PEP-L_CDRs_PEP',
            'H_FL_PEP-L_FL_PEP'
        ]

        enrichment_specs = [
            'H_CDR3_PEP',
            'H_CDRs_PEP-L_CDRs_PEP',
            'H_FL_PEP-L_FL_PEP'
        ]
        enrichment_power = 2.0
        retention_power = 1.0


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
                meta = {
                    'dirs': shared_dirs,
                    'logger': shared_logger
                }
                inst = FastqParser(meta)
                instances.append(inst)

            elif h_cls == ContinuousIgblastExtractor:
                p_conf = self.cfg.ParserConfig
                meta = {
                    'dirs': shared_dirs,
                    'logger': shared_logger,
                    'igblast_exec': getattr(p_conf, 'igblast_exec', 'igblastn'),
                    'igdata_path': getattr(p_conf, 'igdata_path', ''),
                    'database_dir': getattr(p_conf, 'database_dir', ''),
                    'species': getattr(p_conf, 'species', 'human'),
                    'linker_seq': getattr(p_conf, 'linker_seq', ''),
                    'linker_tol_ratio': getattr(p_conf, 'linker_tol_ratio', 0.1),
                    'extract_regions': getattr(p_conf, 'extract_regions', ['FL', 'CDR3', 'CDRs']),
                    'igblast_ncpu': getattr(p_conf, 'igblast_ncpu', multiprocessing.cpu_count()),
                    'anarci_ncpu': getattr(p_conf, 'anarci_ncpu', max(1, multiprocessing.cpu_count() - 1)),
                    'anarci_batch_size': getattr(p_conf, 'anarci_batch_size', 5000),
                    'frame_rescue_offsets': getattr(p_conf, 'frame_rescue_offsets', (0, 1, 2)),
                    'anarci_allowed_species': getattr(p_conf, 'anarci_allowed_species', None),
                    'anarci_bit_score_threshold': getattr(p_conf, 'anarci_bit_score_threshold', 80),
                }
                inst = ContinuousIgblastExtractor(meta)
                instances.append(inst)

            elif h_cls == EnrichmentAnalyzer:
                inst = EnrichmentAnalyzer(shared_dirs.parser_out, shared_logger)
                instances.append(inst)

        return tuple(instances)


if __name__ == '__main__':
    dispatcher = LocalDispatcher(Config)
    pip, par, ig_extractor, enricher = dispatcher.dispatch_handlers(
        (Pipeline, FastqParser, ContinuousIgblastExtractor, EnrichmentAnalyzer)
    )

    pip.enque([
        par.len_filter(where='dna', len_range=Config.FilterConfig.len_range),
        par.q_score_filt(
            minQ=Config.FilterConfig.min_q_score,
            frac=Config.FilterConfig.q_pass_fraction,
        ),
        ig_extractor.run_igblast_and_translate(),
        par.count_summary(where='pep', fmt='csv'),
        par.unpad(),
    ])

    data_iter = par.stream_chunks_from_gz_dir(
        chunk_lines=Config.FilterConfig.chunk_lines
    )
    pip.run_over_stream(data_iter, save_summary=True)
    pip.merge_chunk_outputs(delete_chunks=DELETE_INTERMEDIATE_CHUNKS)

    ig_extractor.format_for_downstream()

    import glob

    root_out = Config.TrackerConfig.parser_out
    annotated_files = glob.glob(os.path.join(root_out, '*', '*_annotated.csv'))

    print(f'Starting Visualization for {len(annotated_files)} files...')
    for f_path in annotated_files:
        enricher.visualize_convergence(
            f_path,
            region_specs=Config.AnalysisConfig.visualization_specs,
        )

    print('Starting Automatic Enrichment Analysis...')
    enricher.auto_discover_and_analyze(
        region_specs=Config.AnalysisConfig.enrichment_specs,
        power=Config.AnalysisConfig.enrichment_power,
        retention_power=Config.AnalysisConfig.retention_power,
    )

    print('Pipeline Complete.')
