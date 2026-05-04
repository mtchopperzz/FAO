import os
import gzip
import tempfile
import subprocess
import pandas as pd
import numpy as np
from io import StringIO

# Import the high-performance tools directly from your existing pipeline
from utils.ProcessHandlers import _match_block_fuzzy, _nb_revcom, _nb_dna_to_pep

def debug_first_chunk(fastq_path, config_meta, reads_to_process=100):
    print(f"[*] Starting Debug Analysis on: {fastq_path}")
    
    # 1. Setup Parameters
    linker_seq = config_meta.get('linker_seq', '').encode('ascii')
    linker_tol = int(len(linker_seq) * config_meta.get('linker_tol_ratio', 0.1)) if linker_seq else 0
    linker_rc = _nb_revcom(np.frombuffer(linker_seq, dtype=np.uint8)).tobytes() if linker_seq else b''
    
    igdata_path = config_meta.get('igdata_path', '')
    db_dir = config_meta.get('database_dir', '')
    species = config_meta.get('species', 'human')
    igblast_exec = config_meta.get('igblast_exec', 'igblastn')
    
    # 2. Extract the first N reads from the FASTQ
    raw_reads = []
    open_func = gzip.open if fastq_path.endswith('.gz') else open
    
    with open_func(fastq_path, 'rt') as f:
        lines = []
        for line in f:
            lines.append(line.strip())
            if len(lines) == 4:
                raw_reads.append(lines[1]) # Index 1 is the DNA sequence
                lines = []
                if len(raw_reads) >= reads_to_process:
                    break

    print(f"[*] Extracted {len(raw_reads)} reads. Analyzing linker direction...")

    # 3. Analyze Linker Direction & Split the Sequence
    fasta_entries = []
    debug_records = []
    
    for i, dna_str in enumerate(raw_reads):
        dna_bytes = dna_str.encode('ascii')
        dna_arr = np.frombuffer(dna_bytes, dtype=np.uint8)
        
        direction = "Unknown"
        linker_found = False
        seq_oriented = dna_str
        p1 = seq_oriented
        p2 = ""
        
        if linker_seq:
            # Check Forward
            idx = _match_block_fuzzy(dna_arr, linker_seq, linker_tol)
            if idx != -1:
                linker_found = True
                direction = "Forward"
                p1 = seq_oriented[:idx]
                p2 = seq_oriented[idx + len(linker_seq):]
            else:
                # Check Reverse Complement
                idx_rc = _match_block_fuzzy(dna_arr, linker_rc, linker_tol)
                if idx_rc != -1:
                    linker_found = True
                    direction = "Reverse_Complement"
                    # Orient the sequence to the forward strand
                    seq_oriented = _nb_revcom(dna_arr).tobytes().decode('ascii')
                    
                    # Find the forward linker in the newly oriented sequence to split it
                    arr_oriented = np.frombuffer(seq_oriented.encode('ascii'), dtype=np.uint8)
                    idx_ori = _match_block_fuzzy(arr_oriented, linker_seq, linker_tol)
                    if idx_ori != -1:
                        p1 = seq_oriented[:idx_ori]
                        p2 = seq_oriented[idx_ori + len(linker_seq):]
                    else:
                        p1 = seq_oriented # Fallback

        # Add BOTH parts to the FASTA file as separate queries
        fasta_entries.append(f">read_{i}_P1\n{p1}")
        if len(p2) > 20:  # Minimum length to bother running through IgBLAST
            fasta_entries.append(f">read_{i}_P2\n{p2}")
            
        debug_records.append({
            "Read_ID": f"read_{i}",
            "Raw_DNA": dna_str,
            "Linker_Found": linker_found,
            "Direction": direction,
            "Oriented_DNA": seq_oriented,
            "P1_DNA": p1,
            "P2_DNA": p2
        })

    # 4. Run IgBLAST
    print("[*] Running IgBLAST Annotation...")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fasta') as tmp_fasta:
        tmp_fasta.write("\n".join(fasta_entries))
        fasta_path = tmp_fasta.name

    env = os.environ.copy()
    env["IGDATA"] = igdata_path
    
    cmd = [
        igblast_exec, '-query', fasta_path, '-organism', species,
        '-ig_seqtype', 'Ig', '-domain_system', 'imgt', '-outfmt', '19'
    ]
    for locus in ['V', 'D', 'J']:
        cmd.extend([f'-germline_db_{locus}', os.path.join(db_dir, f'{species}_{locus}_igblast.fasta')])

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    os.remove(fasta_path)

    # 5. Parse IgBLAST and Translate from FR1
    print("[*] Parsing IgBLAST output and translating from FR1...")
    df_igblast = pd.read_csv(StringIO(result.stdout), sep='\t') if result.returncode == 0 else pd.DataFrame()
    
    # Map IgBLAST results back to our debug records
    for record in debug_records:
        read_id = record["Read_ID"]
        
        # Initialize default columns for both chains
        for chain in ['H', 'L']:
            record[f"{chain}_V_Gene"] = "N/A"
            record[f"{chain}_FR1_Start_Index"] = "Not Found"
            record[f"{chain}_Translated_FL"] = "N/A"
            record[f"{chain}_Status"] = "Missing"
            
        if df_igblast.empty:
            record["H_Status"] = "IgBLAST Failed"
            record["L_Status"] = "IgBLAST Failed"
            continue
            
        # Check BOTH halves of the split sequence (P1 and P2)
        for part in ["_P1", "_P2"]:
            if part == "_P2" and not record["P2_DNA"]:
                continue # Skip if no linker was found to create a P2
                
            hits = df_igblast[df_igblast['sequence_id'] == f"{read_id}{part}"]
            if hits.empty: 
                continue
                
            # Take the top hit for this specific half of the read
            hit = hits.iloc[0]
            locus = str(hit.get('locus', ''))
            
            # Let IgBLAST tell us if this half is Heavy or Light!
            if locus.startswith('IGH'):
                chain = 'H'
            elif locus.startswith('IGK') or locus.startswith('IGL'):
                chain = 'L'
            else:
                continue 
                
            v_call_raw = hit.get('v_call')
            record[f"{chain}_V_Gene"] = str(v_call_raw).split(',')[0] if pd.notna(v_call_raw) else "Unknown"
            
            fwr1_start = hit.get('fwr1_start')
            v_germ_start = hit.get('v_germline_start')
            
            # Fetch the specific DNA half we are translating
            part_seq = record["P1_DNA"] if part == "_P1" else record["P2_DNA"]
            
            if pd.notna(fwr1_start):
                start_idx = max(0, int(float(fwr1_start)) - 1)
                if pd.notna(v_germ_start):
                    start_idx = max(0, start_idx - (int(float(v_germ_start)) - 1))
                    
                record[f"{chain}_FR1_Start_Index"] = start_idx
                
                # Translate ONLY this half of the DNA
                dna_slice = part_seq[start_idx:].upper()
                dna_arr = np.frombuffer(dna_slice.encode('ascii'), dtype=np.uint8)
                full_pep = _nb_dna_to_pep(dna_arr, stop_readthrough=True).tobytes().decode('ascii')
                
                record[f"{chain}_Translated_FL"] = full_pep
                record[f"{chain}_Status"] = "Stop Codon Detected" if '*' in full_pep else "Success"

    # 6. Save Debug Output
    output_df = pd.DataFrame(debug_records)
    out_file = "debug_first_chunk_analysis.csv"
    output_df.to_csv(out_file, index=False)
    print(f"[*] Debug complete! Results saved to: {out_file}")

# =====================================================================
# Execution Block
# =====================================================================
if __name__ == "__main__":
    # Example Config (Update these paths to match your actual environment)
    DEBUG_CONFIG = {
        'igblast_exec': '/home/zhao/NGS/ncbi-igblast-1.22.0/bin/igblastn',
        'igdata_path': '/home/zhao/NGS/ncbi-igblast-1.22.0',
        'database_dir': '/home/zhao/NGS/ncbi-igblast-1.22.0/database',
        'species': 'human',
        'linker_seq': 'TCCGGAGGGTCGACCATAACTTCGTATAATGTATACTATACGAAGTTATCCTCGAGCGGTACC', # Update with your exact linker
        'linker_tol_ratio': 0.1
    }
    
    # Path to one of your FASTQ files
    TARGET_FASTQ = '/home/zhao/data/DDB_NGS_archive/Display_Screening/Yeast/20260406/input/ATP1B3_chicken_library_screening_R4.fastq.gz'
    
    debug_first_chunk(TARGET_FASTQ, DEBUG_CONFIG, reads_to_process=25000)