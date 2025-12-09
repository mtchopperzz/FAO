# -*- coding: utf-8 -*-
"""
Created on Sat Aug 8 20:28:10 2021
@author: Alex Vinogradov
"""
"""
Modified by Zhao Jinxuan
"""

if __name__ == '__main__':
    
    #import prerequisities
    import argparse, importlib, importlib.util, sys, os, gc
    from utils.ProcessHandlers import Pipeline, FastqParser
    from utils.Dispatcher import Dispatcher
    from utils.datatypes import Data
    
    #config file holds the information about library designs and
    #other parser instructions (where to look for data, where to save results etc)
    def load_config(cfg_arg: str):
      if os.path.isfile(cfg_arg):
          spec = importlib.util.spec_from_file_location("config", cfg_arg)
          module = importlib.util.module_from_spec(spec)
          sys.modules["config"] = module      # <-- critical line
          spec.loader.exec_module(module)
      else:
          module = importlib.import_module(cfg_arg)
          sys.modules["config"] = module
      return module
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs.ab_lib_A",
                    help="module or path of config to load "
                         "(default: configs.ab_lib_A)")
    args = ap.parse_args()

    config = load_config(args.config)
    
    #initialize a dispatcher object; dispatcher is strictly speaking
    #not necessary, but it simplifies initialization of data handlers
    dispatcher = Dispatcher(config)
    pip, par  = dispatcher.dispatch_handlers((Pipeline, FastqParser))
  
    #a list of handlers to initialize; pipeline should always be included
    #if NGS data parsing is the goal, FastqParser will do most of the work
    #handlers = (Pipeline, FastqParser)
    
    #initialize the handlers
    #pip, par = dispatcher.dispatch_handlers(handlers)
        
    #enqueue the list of ops to run
    #note that at this stage no data processing will take place,
    #but the validity of specifications will be asserted.

    #generall work flow
    #fetch_gz (must be called)
    #revcom (if necessary, must be called before translation)
    #translate or translate_all_frames (must be called to create pep/DNA/q_score data format)
    #filters and summaries
    #note that only the filtered data will be passed to the next operation, think twice.
    pip.enque([
                par.translate_both_strands(stop_readthrough=False, utr5_offset=0), 
                par.len_filter(where='dna', len_range=[800,1500]),
                par.q_score_filt(minQ=30, frac=0.9),        
                par.cr_filter_fuzzy(where='pep', loc=[2], tol=0),
                par.fetch_at_fuzzy(where='pep', loc=[0,1,2], tol=0),
                #par.convergence_summary(where='dna'),
                #par.convergence_summary(where='pep'),
                #par.freq_summary(where='dna', loc='all', save_txt=True),
                #par.freq_summary(where='pep', loc='all', save_txt=True),
                #par.count_summary(where='dna', top_n=500, fmt='csv'),
                par.count_summary(where='pep', fmt='csv'),
                #par.count_summary(where='dna', top_n=500, fmt='fasta'),
                par.count_summary(where='pep', fmt='fasta'),
                par.unpad(),
                #par.save(where='dna', fmt='npy'),
                par.save(where='pep', fmt='npy'),
                #par.template_summary(where='dna'),
                #par.template_summary(where='pep'),
                #par.tSNE_analysis(where='pep', top_n=1000) #uncomment this line to run tSNE
              ])


    data_iter = par.stream_chunks_from_gz_dir(chunk_lines=100000) 

    pip.run_over_stream(data_iter, save_summary=True)
    pip.merge_chunk_outputs(delete_chunks=True) 


        
        
    
    
    
    
    

















