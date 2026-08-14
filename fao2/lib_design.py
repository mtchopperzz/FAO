# -*- coding: utf-8 -*-
"""
@author: Alex Vinogradov
Modified by Jinxuan ZHAO
"""
"""
Definitions of library design based on constants.
"""

import numpy as np
from fao2.constants import constants 

class Template:

    def __init__(self, lib_seq='', monomers={}, wt=None, lib_type=None):
        self.lib_seq = lib_seq
        self.monomers = monomers
        self.lib_type = lib_type
        
        self._typecheck()
        self._build()
        return
    
    def _typecheck(self):
    
        if not isinstance(self.lib_seq, str):
            raise ValueError("Library design must be specified as a string (dtype=str). . .")

        lib_seq_monomer_types = set(int(x) for x in self.lib_seq if x.isdigit())
        specified_monomer_types = set(self.monomers.keys())
        
        if not lib_seq_monomer_types.issubset(specified_monomer_types):
            raise ValueError("Variable region monomer names must match what's specified in monomers. . .")

        if self.lib_type != 'dna' and self.lib_type != 'pep':
            raise ValueError('Library type can only accept either "dna" or "pep" as valid values')
    
        if self.lib_type == 'dna':
            lookup_monomers = constants.bases
        else:
            lookup_monomers = constants.aas
        
        for m in self.lib_seq:
            if not m.isdigit():
                if not m in lookup_monomers:
                    raise ValueError('All library design monomers must be specified in the lookup tables. . .') 
                
        for key in self.monomers:
            for m in self.monomers[key]:
                 if not m in lookup_monomers:
                    raise ValueError('All library design monomers must be specified in the lookup tables. . .') 
        
        return
    
    def _build(self):
        self.L = len(self.lib_seq)

        is_vr = []
        region = []
        mask = []
        
        current_region = list()
        current_mask = list()
        
        for i,m in enumerate(self.lib_seq):
            if i == 0:
                is_vr.append(m.isdigit())
                
            if m.isdigit():     
                if is_vr[-1] is True:        
                    current_region.append(int(m))
                    current_mask.append(i)
                    
                else:
                    region.append(current_region)
                    mask.append(current_mask)
                    
                    current_region = [int(m)]
                    current_mask = [i]
                    is_vr.append(True)
                
            else:
                if not is_vr[-1] is True:        
                    current_region.append(m)
                    current_mask.append(i)
                    
                else:
                    region.append(current_region)
                    mask.append(current_mask)
                    
                    current_region = [m]
                    current_mask = [i]
                    is_vr.append(False)        
                
        region.append(current_region)
        mask.append(current_mask)
        
        self.is_vr = np.array(is_vr, dtype=bool) 
        self.loc = np.arange(self.is_vr.size)
        self.mask = mask
        self.region = region
        return

    def __repr__(self):
        seq = ''.join(self.lib_seq)    
        return f'<Template container for {seq}  lib_type={self.lib_type}>'    

    def _fancy_index(self, arr, loc):
        out = []
        for x in loc:
            out.extend(arr[x])
            
        return out

    def __call__(self, loc, return_mask=False):
        if not np.all(np.in1d(loc, self.loc)):
            raise ValueError('Library design: a call to non-existent region was made. . .')
        
        if return_mask:
            arr = self.mask
        else:
            arr = self.region
        
        return self._fancy_index(arr, loc)

    def truncate_and_reindex(self, loc):
        def remask():
            mask = list()
            ind = 0
            for reg in self.region:
                current = list()
                for x in reg:    
                    current.append(ind)
                    ind += 1
                mask.append(current)
            
            self.mask = mask
            self.L = ind + 1
            return
        
        self.is_vr = self.is_vr[loc]
        self.loc = np.array(loc)
        
        reg = list()
        for ind in self.loc:
            reg.append(self.region[ind])

        self.region = reg
        remask()
        return

class LibraryDesign:
    def __init__(self, templates=[], monomers={}, lib_type=None):
        self.monomers = monomers
        self.lib_type = lib_type
        self.templates = tuple(
                               Template(lib_seq=x, 
                                        monomers=self.monomers, 
                                        lib_type=self.lib_type) 
                               for x in templates
                              )

        self.L = list(set([x.L for x in self.templates]))
        self._topology_check()
        
        self.loc = self.templates[0].loc
        self.is_vr = self.templates[0].is_vr
        return        
        
    def __repr__(self):
        return f'<Library design container for {len(self.templates)} templates. lib_type={self.lib_type}>'
        
    def __iter__(self):
        for template in self.templates:
            yield template
    
    def __len__(self):
        return len(self.templates)

    def __getitem__(self, item):
        return self.templates[item]

    def _topology_check(self):
        topologies = []
        for t in self.templates:
            topologies.append(tuple(t.is_vr))
        
        if len(set(topologies)) != 1:
            raise ValueError('All library templates should have the same topology. . .') 
        
        return
        
    def truncate_and_reindex(self, loc):
        for template in self.templates:
            template.truncate_and_reindex(loc)
        return