# -*- coding: utf-8 -*-
"""
@author: Alex Vinogradov
Modified by Jinxuan ZHAO
"""
"""
Fundamental biological facts
"""
import string

_codon_table_data = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
}

_reserved = ('_', '+', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0')
_bases = ('T', 'C', 'A', 'G')
_comp_table = str.maketrans('ACTGN', 'TGACN')

_aas = tuple(sorted(set(x for x in _codon_table_data.values() if x not in _reserved)))
_codons = tuple(sorted(set(x for x in _codon_table_data.keys())))
_aa_dict = {aa: i for i, aa in enumerate(_aas)}

class constants:
    codon_table = _codon_table_data
    bases = _bases
    complement_table = _comp_table
    _reserved_aa_names = _reserved
    
    aas = _aas
    codons = _codons
    aa_dict = _aa_dict