#!/usr/bin/env python3
"""
Stahování trénovacích dat pro SQBCP.

Stáhne:
  1. PDB struktury s navázaným NAD (pozitivní příklady)
  2. PDB struktury BEZ NAD (negativní příklady)
  3. Sekvence z UniProt s/bez NAD anotace (pro sequence branch)

Používá RCSB PDB Search API a UniProt REST API.
"""

import os
import json
import time
import urllib.request
import urllib.error
import sys
from pathlib import Path


# ============================================================
# 1. Stažení PDB struktur z RCSB
# ============================================================

def search_rcsb_for_ligand(ligand_id='NAD', max_results=200):
    """
    Vyhledá PDB struktury obsahující daný ligand přes RCSB Search API.
    
    Returns:
        list of PDB IDs
    """
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_nonpolymer_instance_annotation.comp_id",
                        "operator": "exact_match",
                        "value": ligand_id
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": 3.0
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": max_results
            },
            "results_content_type": ["experimental"],
            "sort": [
                {
                    "sort_by": "rcsb_entry_info.resolution_combined",
                    "direction": "asc"
                }
            ]
        }
    }
    
    data = json.dumps(query).encode('utf-8')
    req = urllib.request.Request(url, data=data, 
                                 headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read())
        
        pdb_ids = [hit['identifier'] for hit in result.get('result_set', [])]
        total = result.get('total_count', 0)
        print(f"Nalezeno {total} PDB struktur s ligandem {ligand_id}, "
              f"stahujeme {len(pdb_ids)}")
        return pdb_ids
    
    except urllib.error.URLError as e:
        print(f"Chyba při vyhledávání RCSB: {e}")
        return []


def search_rcsb_negative(exclude_ligand='NAD', max_results=200):
    """
    Vyhledá PDB struktury enzymů BEZ daného ligandu (negativní příklady).
    Hledá oxidoreduktázy (EC 1.*) bez NAD.
    """
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity.rcsb_ec_lineage.id",
                        "operator": "starts_with",
                        "value": "1"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": 2.5
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_results * 3},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "score", "direction": "desc"}]
        }
    }
    
    data = json.dumps(query).encode('utf-8')
    req = urllib.request.Request(url, data=data,
                                 headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read())
        
        all_ids = [hit['identifier'] for hit in result.get('result_set', [])]
        print(f"Nalezeno {len(all_ids)} oxidoreduktáz, filtruji...")
        return all_ids[:max_results]
    
    except urllib.error.URLError as e:
        print(f"Chyba při vyhledávání RCSB: {e}")
        return []


def download_pdb(pdb_id, output_dir, format='pdb'):
    """Stáhne jeden PDB soubor z RCSB."""
    output_file = os.path.join(output_dir, f"{pdb_id}.pdb")
    
    if os.path.exists(output_file):
        return True
    
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    try:
        urllib.request.urlretrieve(url, output_file)
        return True
    except urllib.error.URLError:
        return False


def download_all_pdbs(pdb_ids, output_dir, delay=0.2):
    """Stáhne PDB soubory s pauzou mezi požadavky."""
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded = 0
    failed = 0
    
    for i, pdb_id in enumerate(pdb_ids):
        success = download_pdb(pdb_id, output_dir)
        if success:
            downloaded += 1
        else:
            failed += 1
        
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(pdb_ids)}] Staženo: {downloaded}, "
                  f"Selhalo: {failed}")
        
        time.sleep(delay)
    
    print(f"Hotovo: {downloaded} staženo, {failed} selhalo")
    return downloaded


# ============================================================
# 2. Stažení sekvencí z UniProt
# ============================================================

def download_uniprot_sequences(query, output_file, max_results=5000):
    """
    Stáhne sekvence z UniProt REST API.
    
    Příklady query:
      - NAD-binding: (cc_cofactor:NAD) AND (reviewed:true)
      - Non-NAD:     (reviewed:true) NOT (cc_cofactor:NAD) AND (ec:1.*)
    """
    import urllib.parse
    
    encoded_query = urllib.parse.quote(query)
    url = (f"https://rest.uniprot.org/uniprotkb/search?"
           f"query={encoded_query}"
           f"&format=fasta"
           f"&size={max_results}")
    
    print(f"Stahuji z UniProt: {query}")
    
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            content = response.read().decode('utf-8')
        
        with open(output_file, 'w') as f:
            f.write(content)
        
        # Spočítej sekvence
        n_seqs = content.count('>')
        print(f"  Staženo {n_seqs} sekvencí → {output_file}")
        return n_seqs
    
    except urllib.error.URLError as e:
        print(f"  Chyba: {e}")
        return 0


def parse_fasta_to_csv(positive_fasta, negative_fasta, output_csv):
    """
    Převede FASTA soubory na CSV s labely.
    
    Output formát: uniprot_id,sequence,label,cofactor
    """
    import csv
    
    def read_fasta(fasta_file):
        """Jednoduché FASTA čtení bez BioPython."""
        sequences = []
        current_id = None
        current_seq = []
        
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id is not None:
                        sequences.append((current_id, ''.join(current_seq)))
                    # Parse UniProt ID: >sp|P12345|NAME ...
                    parts = line[1:].split('|')
                    current_id = parts[1] if len(parts) > 1 else parts[0].split()[0]
                    current_seq = []
                elif line:
                    current_seq.append(line)
        
        if current_id is not None:
            sequences.append((current_id, ''.join(current_seq)))
        
        return sequences
    
    positive = read_fasta(positive_fasta)
    negative = read_fasta(negative_fasta)
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['uniprot_id', 'sequence', 'label', 'cofactor'])
        
        for uid, seq in positive:
            writer.writerow([uid, seq, 1, 'NAD'])
        
        for uid, seq in negative:
            writer.writerow([uid, seq, 0, ''])
    
    print(f"CSV vytvořeno: {output_csv}")
    print(f"  Pozitivní: {len(positive)}, Negativní: {len(negative)}")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    PDB_POS_DIR = os.path.join(DATA_DIR, 'pdb_positive')  # NAD-binding
    PDB_NEG_DIR = os.path.join(DATA_DIR, 'pdb_negative')  # non-NAD
    SEQ_DIR = os.path.join(DATA_DIR, 'sequences')
    
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(SEQ_DIR, exist_ok=True)
    
    print("=" * 60)
    print("SQBCP – Stahování trénovacích dat")
    print("=" * 60)
    
    # ---- Argumenty ----
    n_positive = 100  # PDB s NAD
    n_negative = 100  # PDB bez NAD
    n_seq_pos = 2000  # UniProt sekvence s NAD
    n_seq_neg = 2000  # UniProt sekvence bez NAD
    
    if len(sys.argv) > 1:
        n_positive = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_negative = int(sys.argv[2])
    
    # ---- 1. PDB struktury s NAD ----
    print(f"\n[1/4] Vyhledávám PDB struktury s NAD (max {n_positive})...")
    pos_ids = search_rcsb_for_ligand('NAD', max_results=n_positive)
    
    if pos_ids:
        print(f"  Stahuji {len(pos_ids)} PDB souborů...")
        download_all_pdbs(pos_ids, PDB_POS_DIR)
    
    # ---- 2. PDB struktury BEZ NAD (negativní) ----
    print(f"\n[2/4] Vyhledávám PDB struktury BEZ NAD (max {n_negative})...")
    neg_ids = search_rcsb_negative('NAD', max_results=n_negative)
    
    # Vyfiltruj ty, co mají NAD
    neg_ids = [pid for pid in neg_ids if pid not in set(pos_ids)]
    neg_ids = neg_ids[:n_negative]
    
    if neg_ids:
        print(f"  Stahuji {len(neg_ids)} PDB souborů...")
        download_all_pdbs(neg_ids, PDB_NEG_DIR)
    
    # ---- 3. UniProt sekvence ----
    print(f"\n[3/4] Stahuji NAD-binding sekvence z UniProt...")
    pos_fasta = os.path.join(SEQ_DIR, 'nad_positive.fasta')
    download_uniprot_sequences(
        "(cc_cofactor:NAD) AND (reviewed:true)",
        pos_fasta,
        max_results=n_seq_pos
    )
    
    print(f"\n[4/4] Stahuji non-NAD sekvence z UniProt...")
    neg_fasta = os.path.join(SEQ_DIR, 'nad_negative.fasta')
    download_uniprot_sequences(
        "(reviewed:true) NOT (cc_cofactor:NAD) AND (ec:1.*)",
        neg_fasta,
        max_results=n_seq_neg
    )
    
    # ---- 4. Vytvořit CSV ----
    csv_path = os.path.join(SEQ_DIR, 'nad_sequences.csv')
    if os.path.exists(pos_fasta) and os.path.exists(neg_fasta):
        parse_fasta_to_csv(pos_fasta, neg_fasta, csv_path)
    
    # ---- Souhrn ----
    print("\n" + "=" * 60)
    print("SOUHRN")
    print("=" * 60)
    
    if os.path.exists(PDB_POS_DIR):
        n_pos = len([f for f in os.listdir(PDB_POS_DIR) if f.endswith('.pdb')])
        print(f"  PDB s NAD:     {n_pos} souborů v {PDB_POS_DIR}")
    
    if os.path.exists(PDB_NEG_DIR):
        n_neg = len([f for f in os.listdir(PDB_NEG_DIR) if f.endswith('.pdb')])
        print(f"  PDB bez NAD:   {n_neg} souborů v {PDB_NEG_DIR}")
    
    if os.path.exists(csv_path):
        print(f"  Sekvence CSV:  {csv_path}")
    
    print(f"\nDalší krok: python run_pipeline.py")
