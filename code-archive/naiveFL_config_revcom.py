import os
from utils.lib_design import LibraryDesign

experiment = 'Specifica_naiveFL_test'

class constants:
    '''
    Star symbol (*) is reserved for stop codons.
    Plus and underscore symbols (+ and _) are internally reserved tokens.
    Numerals (1234567890) are internally reserved for library design specification. 
    These symbols (123456790+_) should not be used to encode amino acids. 
    Other symbols are OK.
    
    Although the class holds multiple attributes, only
    the codon table should be edited. Everything else
    is inferred automatically from it.
    '''
    codon_table = {
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


    bases = ('T', 'C', 'A', 'G')
    complement_table = str.maketrans('ACTGN', 'TGACN')

    global _reserved_aa_names
    _reserved_aa_names = ('_', '+', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0')    
    
    aas = tuple(sorted(set(x for x in codon_table.values() if x not in _reserved_aa_names)))
    codons = tuple(sorted(set(x for x in codon_table.keys())))
    aa_dict = {aa: i for i,aa in enumerate(aas)}
          
class ParserConfig:

    utr5_seq = ['GATATCCAGATGACCCAGTCTCCATCTTCTGTT','GAAATCGTTATG','GAATCTGCTCTG','GATATCCAGATGACCCAGTCTCCATCTTCTCTG']
    #DNA library design
    #"1" is used to refer random monomers. the number of 1, in another word the length of variable region, does not need to be exact when calling _fuzzy operations
    #therefore, 111 can be any non-constant region sequence with any length
    #Germline database can be fitted into D and P_design when using Beacon
    # For synthesized library, D and P_design need to be entered manually
    # of course, the D and P_design need to be trimmed to fit the actual NGS sample design
    D_design = LibraryDesign(
        
                    templates=[
                                '111ACCGTCGATATGAAAATTACCTCTCCGACCACTGAGGACACTGCGACGTACTTCTGTGCGCGT111TGGGGTCCGGGTACTTTGGTTACCGTCTCCTCC111',
                                '111ACCGTCGATATGAAAATTACCTCTCCGACCACTGAGGACACTGCGACGTACTTCTGTGCGCGT111TGGGGTCCGGGTACTTTGGTTACCGTCTCCTCC111',
                              ],
            
                    monomers={
                              1: ('A', 'G', 'T', 'C'),
                              },
                    
                    lib_type='dna'
                        
                            )
    
    #peptide library design
    P_design = LibraryDesign(
        
                    templates=[
                                '111DIQMTQSPSSVSASVGDRVTITCRAS111LAWYQQKPGKAPKLLIY111GVPSRFSGSGSGTDFTLTISSLQPEDFANYYC111FGGGTKVEIKSGGSTITSYNVYYTKLSSSGTQVQLVQSGAEVKKPGASVKVSCKVS111IHWVRQAPGKGLEWMGG111IYAQKFQGRVTMTEDTSTDTAYMELSSLKSEDTAVYYC111WGKGTTVTVSS111',
                                '111EIVMTQSPATLSLSPGERATLSCRAS111LAWYQQKPGQAPRLLIY111GIPARFSGSGSGTDFTLTISSLEPEDFAVYYC111FGGGTKVEIKSGGSTITSYNVYYTKLSSSGTQVQLQESGPGLVKPSQTLSLTCTVS111WSWIRQPPGKGLEWIGY111DYNPSLKSRVTMSVDTSKNKFSLKVNSVTAADTAVYYC111WGQGTLVTVSS111',
                                '111ESALTQPASVSGSPGQSITISCTGT111VSWYQQHPGKAPKLMIY111GVSNRFSGSKSGNTASLTISGLQAEDEADYYC111FGGGTKLTVLSGGSTITSYNVYYTKLSSSGTEVQLVQSGAEVKKPGASVKVSCKAS111ISWVRQAPGQGLEWMGW111NYAQKLQGRGTMTTDTSTSTAYMELRSLRSDDTAVYYC111WGQGTTVTVSS111',
                                '111DIQMTQSPSSLSASVGDRVTITCRAS111LAWYQQKPGKAPKLLIY111GVPSRFSGSGSGTDFTLTISSLQPEDVATYYC111FGGGTKVEIKSGGSTITSYNVYYTKLSSSGTEVQLVESGGGLVQPGRSLRLSCAAS111MHWVRQAPGKGLEWVSA111DYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYC111WGQGTLVTVSS111',

                              ],
            
                    monomers={
                              1: ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'),
                              2: ('M'),
                              },
                    
                    lib_type='pep'
                        
                            )
    
  
class TrackerConfig:
    
    #directory holding sequencing data files (fastq or fastq.gz)
    seq_data = '../../data/DDB_NGS_archive/Specifica_Library_QC/scfv_FL_Pacbio/splitted2'
        
    #directory for writing logs to
    logs = '../../data/DDB_NGS_archive/Specifica_Library_QC/scfv_FL_Pacbio/logs/all-Q-revcom'
    
    #directory that stores fastqparser outputs
    parser_out = '../../data/DDB_NGS_archive/Specifica_Library_QC/scfv_FL_Pacbio/parser_outputs/all-Q-revcom'

class LoggerConfig:
    
    #logger name
    name = experiment + '_logger'
    
    #verbose loggers print to the console
    verbose = True
    
    #write logs to file
    log_to_file = True
    
    #log filename; when None, the name will be inferred by the Logger itself
    log_fname = os.path.join(TrackerConfig.logs, experiment + ' logs')    