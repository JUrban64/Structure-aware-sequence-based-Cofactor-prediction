#!/usr/bin/env python3
"""
SQBCP – Kompletní trénovací pipeline

Spuštění:
    # 1. Nainstalovat prostředí
    conda env create -f environment.yml
    conda activate sqbcp
    
    # 2. Stáhnout data
    python download_data.py
    
    # 3. Spustit trénink
    python run_pipeline.py
    
    # Nebo rychlý test na jednom PDB:
    python run_pipeline.py --test

Kroky pipeline:
    1. Načtení PDB souborů → extrakce binding sites
    2. ESM-2 extrakce embeddingů
    3. Sestavení grafového datasetu
    4. Načtení sekvenčních dat (UniProt)
    5. Inicializace DualBranchPredictor
    6. Trénink (GNN + Sequence branch)
"""

import os
import sys
import gc
import glob
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path


def log_memory(tag=""):
    """Vypíše aktuální spotřebu RAM (RSS) procesu."""
    try:
        import psutil
        proc = psutil.Process()
        rss_gb = proc.memory_info().rss / 1024**3
        print(f"  [MEM] {tag}: {rss_gb:.2f} GB RSS")
    except ImportError:
        pass  # psutil není dostupný – tiše přeskočíme

# ============================================================
# KONFIGURACE
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = {
    # Cesty (dynamicky sestaveny v main() podle ligandu)
    'data_root': os.path.join(BASE_DIR, 'data'),
    'pdb_positive_dir': os.path.join(BASE_DIR, 'data', 'NAD', 'PDB', 'positive', 'vycisteno_protonated'),
    'pdb_negative_dir': os.path.join(BASE_DIR, 'data', 'NAD', 'PDB', 'negative', 'boltz_negatives_protonated'),
    'seq_positive_csv': os.path.join(BASE_DIR, 'data', 'NAD', 'sequences', 'positive', 'NAD_only_dataset.csv'),
    'seq_negative_csv': os.path.join(BASE_DIR, 'data', 'NAD', 'sequences', 'negative', 'NO_cofa_15000_id0.csv'),
    'cache_dir': os.path.join(BASE_DIR, 'cache'),
    
    # Ligand
    'ligand_name': 'NAD',
    'distance_threshold': 6.0,
    
    # ESM model porotm tam vratit 650
    'esm_model': 'facebook/esm2_t33_650M_UR50D',
    'esm_dim': 1280,
    
    # Model
    'node_dim': 1310,  # 1280 + 20 + 7 + 3 (uloženo v grafu, ESM se kompresuje v modelu)
    'esm_compress_dim': 64,  # Varianta C: ESM 1280 → 64 v grafové větvi
    'ligand_dim': 36,  # LigandFeatures.LIGAND_FEAT_DIM
    'hidden_dim': 256,
    'num_gnn_layers': 3,
    'num_attention_heads': 4,
    'dropout': 0.6,
    'use_gat': True,
    'include_ligand': True,  # Přidat ligandové uzly a P-L hrany do grafu
    
    # Trénink
    'num_epochs': 30,           # best AUC typicky kolem epochy 11
    'lr': 0.0005,               # snížený LR pro stabilitu
    'dropout': 0.6,             # regularizace
    'consistency_weight': 0.5,  # alignment GNN ↔ Seq embeddingů
    'struct_weight': 5.0,       # zvýšená váha GNN loss (kompenzuje malý dataset)
    'seq_weight': 1.0,          # váha sequence loss
    'early_stopping_patience': 10,  # zastaví trénink po 10 epochách bez zlepšení
    'batch_size_graph': 32,     # batch size pro grafový loader
    'batch_size_seq': 32,       # batch size pro sekvenční loader
    
    # Cluster-based split (ochrana proti data leakage)
    'cluster_identity': 0.4,  # MMseqs2 identity threshold
                               # 0.3 = fold-level (přísné)
                               # 0.4 = superfamily-level (doporučené)
                               # 0.5 = family-level
    'mmseqs_threads': 4,       # MMseqs2 threads (zvyšte na ncpus z PBS)

    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def parse_args():
    parser = argparse.ArgumentParser(description='SQBCP Training Pipeline')
    parser.add_argument('--test', action='store_true',
                        help='Rychlý test na jednom PDB souboru')
    parser.add_argument('--pdb-dir', type=str, default=None,
                        help='Složka s PDB soubory (pozitivní)')
    parser.add_argument('--pdb-neg-dir', type=str, default=None,
                        help='Složka s PDB soubory (negativní)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--ligand', type=str, default='NAD')
    parser.add_argument('--no-seq', action='store_true',
                        help='Trénovat pouze na PDB (bez sequence branch)')
    parser.add_argument('--test-data', action='store_true',
                        help='Použít malý testovací dataset (data/NAD/test/) '
                             'pro ověření funkčnosti celého pipeline '
                             '(10 pozitivních + 5 negativních PDB a sekvencí)')
    parser.add_argument('--esm-model', type=str, 
                        default='facebook/esm2_t33_650M_UR50D')
    parser.add_argument('--cluster-identity', type=float, default=0.4,
                        help='MMseqs2 sequence identity threshold for '
                             'cluster-based split (default: 0.4 = superfamily)')
    parser.add_argument('--struct-weight', type=float, default=5.0,
                        help='Váha GNN struct loss (default: 5.0, kompenzuje '
                             'malý PDB dataset oproti sekvencím)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10 epoch)')
    parser.add_argument('--save-splits', type=str, default=None,
                        help='Cesta ke složce, kam se uloží rozdělené datasety '
                             '(train/val/test). Pokud není zadáno, splity se '
                             'neukládají.')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Pouze připraví datasety a uloží splity '
                             '(bez tréninku). Vyžaduje --save-splits.')
    parser.add_argument('--load-splits', type=str, default=None,
                        help='Cesta ke složce s dříve uloženými splity. '
                             'Přeskočí clusterování a použije existující '
                             'train/val/test rozdělení.')
    return parser.parse_args()


# ============================================================
# KROK 1: Extrakce binding sites z PDB
# ============================================================
def extract_binding_sites(pdb_dir, ligand_name, distance_threshold, 
                          label=1, recursive=False):
    """Extrahuje binding sites ze všech PDB souborů ve složce.
    
    Args:
        pdb_dir: cesta ke složce s PDB soubory
        ligand_name: název ligandu (NAD, FAD, ...)
        distance_threshold: práh vzdálenosti pro binding site (Å)
        label: 1 = pozitivní (nativní vazba), 0 = negativní (špatná vazba)
        recursive: hledat PDB i v podsložkách
    """
    from Binding_site_ex import BindingSiteExtractor
    
    extractor = BindingSiteExtractor(distance_threshold=distance_threshold)
    
    if recursive:
        pdb_files = glob.glob(os.path.join(pdb_dir, '**', '*.pdb'), recursive=True)
    else:
        pdb_files = glob.glob(os.path.join(pdb_dir, '*.pdb'))
    
    if not pdb_files:
        print(f"  ⚠ Žádné PDB soubory v {pdb_dir}")
        return []
    
    print(f"  Nalezeno {len(pdb_files)} PDB souborů v {pdb_dir}")
    
    binding_sites = []
    for i, pdb_file in enumerate(pdb_files):
        try:
            bs_info = extractor.extract_binding_site(
                pdb_file, ligand_name=ligand_name
            )
            
            # Přeskoč pokud je sekvence prázdná
            if not bs_info.get('full_sequence') or len(bs_info['full_sequence']) == 0:
                print(f"    ⚠ Prázdná sekvence v {os.path.basename(pdb_file)} – přeskakuji")
                continue
            
            bs_info['label'] = label
            binding_sites.append(bs_info)
            
            if (i + 1) % 20 == 0:
                print(f"    [{i+1}/{len(pdb_files)}] "
                      f"Zpracováno {len(binding_sites)} binding sites")
        except Exception as e:
            print(f"    ✗ {os.path.basename(pdb_file)}: {e}")
    
    print(f"  ✓ {len(binding_sites)}/{len(pdb_files)} úspěšně extrahováno")
    return binding_sites


# ============================================================
# KROK 2: ESM-2 embeddingy
# ============================================================
def compute_esm_embeddings(binding_sites, esm_model_name, cache_dir,
                           esm_extractor=None):
    """Extrahuje ESM embeddingy pro binding sites (s cachováním).
    
    Args:
        esm_extractor: volitelná existující ESM instance (pro sdílení mezi kroky)
    
    Returns:
        esm_extractor: ESM instance (pro znovupoužití v dalších krocích)
    """
    from esm2_feature_ex import ESMFeatureExtractor
    
    cache_file = os.path.join(cache_dir, 'esm_embeddings.pkl')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Odfiltruj binding sites s prázdnou sekvencí
    empty_count = 0
    for bs in binding_sites:
        if not bs.get('full_sequence') or len(bs['full_sequence'].strip()) == 0:
            empty_count += 1
    if empty_count > 0:
        print(f"  ⚠ {empty_count} binding sites má prázdnou sekvenci – budou přeskočeny")
        binding_sites[:] = [
            bs for bs in binding_sites 
            if bs.get('full_sequence') and len(bs['full_sequence'].strip()) > 0
        ]
    
    # Zkus načíst z cache
    if os.path.exists(cache_file):
        print("  Načítám ESM embeddingy z cache...")
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        
        if len(cached) == len(binding_sites):
            for i, bs in enumerate(binding_sites):
                bs['esm_embeddings'] = cached[i]['embeddings']
                # Aktualizuj binding site info podle valid indices
                if 'valid_indices' in cached[i]:
                    _update_bs_for_valid_indices(bs, cached[i]['valid_indices'])
            print(f"  ✓ {len(cached)} embeddingů načteno z cache")
            return esm_extractor  # vrátí None nebo existující instanci
        else:
            print(f"  Cache neodpovídá ({len(cached)} vs {len(binding_sites)}), "
                  "přepočítávám...")
    
    if esm_extractor is None:
        print(f"  Načítám ESM-2 model: {esm_model_name}")
        esm = ESMFeatureExtractor(model_name=esm_model_name)
    else:
        print(f"  Používám sdílenou ESM instanci (bez opakovaného načítání)")
        esm = esm_extractor
    
    embeddings_cache = []
    skipped = 0
    to_remove = []
    for i, bs in enumerate(binding_sites):
        try:
            emb, valid_indices = esm.extract_binding_site_embeddings(
                bs['full_sequence'],
                bs['binding_site_indices']
            )
            bs['esm_embeddings'] = emb
            
            # Aktualizuj binding site info aby odpovídal valid indices
            _update_bs_for_valid_indices(bs, valid_indices)
            
            embeddings_cache.append({
                'embeddings': emb,
                'valid_indices': valid_indices
            })
        except (ValueError, IndexError) as e:
            print(f"    ⚠ Přeskakuji binding site {i} "
                  f"({os.path.basename(bs.get('pdb_file', '?'))}): {e}")
            to_remove.append(i)
            skipped += 1
            continue
        
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(binding_sites)}] "
                  f"shape: {emb.shape}")
    
    # Odstraň problematické binding sites (odzadu)
    for i in sorted(to_remove, reverse=True):
        binding_sites.pop(i)
    
    if skipped > 0:
        print(f"  ⚠ Přeskočeno {skipped} binding sites kvůli chybám")
    
    # Ulož cache
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings_cache, f)
    print(f"  ✓ {len(embeddings_cache)} embeddingů uloženo do cache")
    
    # Vrať ESM instanci pro znovupoužití (neuvolněno!)
    return esm


def _update_bs_for_valid_indices(bs, valid_indices):
    """Aktualizuje binding site info tak, aby odpovídal pouze 
    residuím, pro které máme ESM embeddingy (po truncation).
    
    Toto zajistí že ESM [n, 1280], BLOSUM [n, 20], PhysChem [n, 7] 
    a Position [n, 3] mají STEJNÉ n.
    """
    original_indices = bs['binding_site_indices']
    
    # Pokud se nic nezměnilo, nic neděláme
    if len(valid_indices) == len(original_indices):
        return
    
    # Zjisti, které pozice v původním seznamu zůstaly
    valid_set = set(valid_indices)
    keep_mask = [idx in valid_set for idx in original_indices]
    
    # Aktualizuj indices
    bs['binding_site_indices'] = valid_indices
    
    # Aktualizuj sekvenci
    old_seq = bs['binding_site_sequence']
    bs['binding_site_sequence'] = ''.join(
        aa for aa, keep in zip(old_seq, keep_mask) if keep
    )
    
    # Aktualizuj residues (pokud existují)
    if 'binding_site_residues' in bs:
        bs['binding_site_residues'] = [
            res for res, keep in zip(bs['binding_site_residues'], keep_mask) 
            if keep
        ]
    
    # Aktualizuj n_binding_site
    bs['n_binding_site'] = len(valid_indices)
    
    # Aktualizuj kontaktní mapu (ořízni řádky/sloupce)
    if 'contact_map' in bs:
        old_cm = bs['contact_map']
        keep_idx = [i for i, k in enumerate(keep_mask) if k]
        bs['contact_map'] = old_cm[np.ix_(keep_idx, keep_idx)]
    
    # Aktualizuj protein-ligand kontakty (přeindexuj protein_idx)
    if 'protein_ligand_contacts' in bs:
        old_to_new = {}
        new_idx = 0
        for old_idx, keep in enumerate(keep_mask):
            if keep:
                old_to_new[old_idx] = new_idx
                new_idx += 1
        
        new_contacts = []
        for c in bs['protein_ligand_contacts']:
            if c['protein_idx'] in old_to_new:
                new_c = c.copy()
                new_c['protein_idx'] = old_to_new[c['protein_idx']]
                new_contacts.append(new_c)
        bs['protein_ligand_contacts'] = new_contacts
    
    print(f"    ℹ BS aktualizován: {len(original_indices)} → "
          f"{len(valid_indices)} residues (ESM truncation)")


# ============================================================
# KROK 3: Grafový dataset
# ============================================================
def build_graph_dataset(binding_sites, include_ligand=True):
    """Sestaví PyG grafový dataset (protein-ligand interakční graf)."""
    from binding_site_graph import BindingSiteGraphDataset
    
    # Labely nastavíme PŘED konstrukcí datasetu – grafy se staví lazy
    for bs in binding_sites:
        bs.setdefault('label', 1)
    
    dataset = BindingSiteGraphDataset(
        binding_sites,
        feature_config={
            'use_esm': True,
            'use_blosum': True,
            'use_physchem': True,
            'use_position': True
        },
        include_ligand=include_ligand
    )
    
    print(f"  ✓ {len(dataset)} grafů připraveno (lazy building)")
    if len(dataset) > 0:
        g = dataset[0]
        print(f"    Celkem uzlů: {g.x.shape[0]} "
              f"(protein: {g.n_protein_nodes}, ligand: {g.n_ligand_nodes})")
        print(f"    Hrany: {g.edge_index.shape[1]} "
              f"(PP + PL + LL)")
        print(f"    Protein features: {g.protein_dim}D, "
              f"Ligand features: {g.ligand_dim}D")
        if hasattr(g, 'cofactor_id'):
            print(f"    Cofactor: {g.cofactor_id}")
    
    return dataset


# ============================================================
# KROK 4: Sekvenční dataset
# ============================================================
def load_sequence_data(config, esm_extractor=None):
    """Načte a připraví sekvenční dataset z positive + negative CSV.
    
    Používá disk-based ESM embeddingy (lazy-loading) pro úsporu RAM:
    1. Embeddingy se uloží jako jednotlivé .npy soubory na disk
    2. SequenceDataset je čte on-demand v __getitem__
    3. ESM model se uvolní z paměti po extrakci (pokud nebyl předán zvenčí)
    
    Args:
        config: konfigurace pipeline
        esm_extractor: volitelná sdílená ESM instance (pro úsporu paměti)
    
    Returns:
        (dataset, esm_extractor) - dataset a ESM instance (pro další použití)
    """
    from sequence_dataset import (
        SequenceDataset, load_sequences_from_separate_csvs, 
        save_embeddings, load_embeddings
    )
    from esm2_feature_ex import ESMFeatureExtractor
    
    pos_csv = config['seq_positive_csv']
    neg_csv = config['seq_negative_csv']
    esm_model_name = config['esm_model']
    cache_dir = config['cache_dir']
    ligand = config['ligand_name']
    
    if not os.path.exists(pos_csv) and not os.path.exists(neg_csv):
        print(f"  ⚠ Žádné CSV soubory nenalezeny:")
        print(f"    Positive: {pos_csv}")
        print(f"    Negative: {neg_csv}")
        return None
    
    sequences, labels = load_sequences_from_separate_csvs(
        pos_csv, neg_csv, 
        cofactor_filter=ligand,
        max_negative=None  # použít všechny
    )
    
    if len(sequences) == 0:
        print("  ⚠ Žádné sekvence nenačteny")
        return None
    
    log_memory("před ESM seq embeddingy")
    
    # Složka pro per-sequence .npy embeddingy (disk-based cache)
    emb_dir = os.path.join(cache_dir, f'seq_emb_{ligand}')
    os.makedirs(emb_dir, exist_ok=True)
    
    # Generuj stabilní seq_ids (hash sekvence → deterministické jméno)
    import hashlib
    seq_ids = [hashlib.md5(seq.encode()).hexdigest()[:12] for seq in sequences]
    
    # Zkontroluj kolik embeddingů už existuje na disku
    existing = sum(1 for sid in seq_ids 
                   if os.path.exists(os.path.join(emb_dir, f"{sid}.npy")))
    
    esm_created_here = False
    if existing < len(sequences):
        missing = len(sequences) - existing
        print(f"  {existing}/{len(sequences)} embeddingů na disku, "
              f"chybí {missing} → spouštím ESM extrakci")
        
        if esm_extractor is None and esm_model_name:
            esm_extractor = ESMFeatureExtractor(model_name=esm_model_name)
            esm_created_here = True
        else:
            print(f"  Používám sdílenou ESM instanci")
        
        esm_extractor.extract_and_save_to_disk(
            list(zip(seq_ids, sequences)),
            output_dir=emb_dir,
            max_length=512
        )
    else:
        print(f"  ✓ Všech {existing} embeddingů nalezeno na disku")
    
    # Vytvoř dataset s lazy-loading z disku
    dataset = SequenceDataset(
        sequences, labels,
        emb_dir=emb_dir,
        seq_ids=seq_ids,
        max_length=512
    )
    
    log_memory("po vytvoření SequenceDataset (lazy)")
    print(f"  ✓ {len(dataset)} sekvencí připraveno (lazy-loading z disku)")
    return dataset, esm_extractor


# ============================================================
# KROK 5: TRÉNINK
# ============================================================
def save_splits(output_dir, train_data, val_data, test_data, 
                split_type='graph', config=None):
    """
    Uloží rozdělené datasety na disk.
    
    Args:
        output_dir: cílová složka
        train_data: train split (list of PyG Data nebo Subset)
        val_data: val split
        test_data: test split
        split_type: 'graph' nebo 'sequence'
        config: konfigurace (pro metadata)
    """
    split_dir = os.path.join(output_dir, split_type)
    os.makedirs(split_dir, exist_ok=True)
    
    splits = {'train': train_data, 'val': val_data, 'test': test_data}
    
    for name, data in splits.items():
        if data is None or len(data) == 0:
            print(f"  ℹ {split_type}/{name} je prázdný – přeskakuji")
            continue
        
        save_path = os.path.join(split_dir, f'{name}.pt')
        
        if split_type == 'graph':
            # PyG Data list
            torch.save(data, save_path)
        else:
            # Sequence Subset → uložíme indexy + metadata
            if hasattr(data, 'indices'):
                # torch.utils.data.Subset
                torch.save({
                    'indices': data.indices,
                    'sequences': [data.dataset.sequences[i] for i in data.indices],
                    'labels': [data.dataset.labels[i] for i in data.indices],
                }, save_path)
            else:
                torch.save(data, save_path)
        
        print(f"  ✓ Uloženo {len(data)} vzorků do {save_path}")
    
    # Metadata
    metadata = {
        'split_type': split_type,
        'n_train': len(train_data) if train_data else 0,
        'n_val': len(val_data) if val_data else 0,
        'n_test': len(test_data) if test_data else 0,
        'cluster_identity': config.get('cluster_identity', 0.4) if config else 0.4,
        'ligand': config.get('ligand_name', 'NAD') if config else 'NAD',
    }
    metadata_path = os.path.join(split_dir, 'split_metadata.pt')
    torch.save(metadata, metadata_path)
    print(f"  ✓ Metadata uložena do {metadata_path}")


def load_splits(input_dir, split_type='graph'):
    """
    Načte dříve uložené splity z disku.
    
    Args:
        input_dir: složka se splity (obsahuje graph/ a/nebo sequence/)
        split_type: 'graph' nebo 'sequence'
    
    Returns:
        (train_data, val_data, test_data, metadata) nebo None pokud neexistuje
    """
    split_dir = os.path.join(input_dir, split_type)
    
    if not os.path.isdir(split_dir):
        print(f"  ⚠ Složka {split_dir} neexistuje")
        return None
    
    # Metadata
    metadata_path = os.path.join(split_dir, 'split_metadata.pt')
    metadata = None
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path, weights_only=False)
        print(f"  ℹ Split metadata: {metadata}")
    
    splits = {}
    for name in ['train', 'val', 'test']:
        save_path = os.path.join(split_dir, f'{name}.pt')
        if os.path.exists(save_path):
            splits[name] = torch.load(save_path, weights_only=False)
            n = len(splits[name]) if not isinstance(splits[name], dict) else len(splits[name].get('sequences', []))
            print(f"  ✓ Načteno {n} vzorků z {save_path}")
        else:
            splits[name] = []
            print(f"  ℹ {save_path} nenalezen – prázdný split")
    
    return splits.get('train', []), splits.get('val', []), splits.get('test', []), metadata


def _build_seq_subsets_from_loaded(loaded_splits, esm_extractor=None, 
                                    esm_model_name=None, cache_dir=None,
                                    max_length=512):
    """
    Vytvoří SequenceDataset Subsety z načtených sekvenčních splitů.
    
    Každý split je dict s 'sequences' a 'labels' → vytvoří se
    samostatný SequenceDataset pro train/val/test.
    Používá disk-based lazy-loading pro úsporu RAM.
    
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    import hashlib
    from sequence_dataset import SequenceDataset
    from esm2_feature_ex import ESMFeatureExtractor
    
    train_data, val_data, test_data = loaded_splits[:3]
    datasets = []
    
    esm_loaded = False  # ESM se načte maximálně jednou
    
    for name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        if isinstance(data, dict) and 'sequences' in data:
            seqs = data['sequences']
            labs = data['labels']
        elif isinstance(data, list) and len(data) > 0:
            seqs = [d.get('sequence', '') if isinstance(d, dict) else '' for d in data]
            labs = [d.get('label', 0) if isinstance(d, dict) else 0 for d in data]
        else:
            datasets.append(None)
            continue
        
        if len(seqs) == 0:
            datasets.append(None)
            continue
        
        # Disk-based lazy-loading
        if cache_dir:
            emb_dir = os.path.join(cache_dir, f'seq_emb_{name}_split')
            os.makedirs(emb_dir, exist_ok=True)
            
            seq_ids = [hashlib.md5(s.encode()).hexdigest()[:12] for s in seqs]
            
            existing = sum(1 for sid in seq_ids 
                           if os.path.exists(os.path.join(emb_dir, f"{sid}.npy")))
            
            if existing < len(seqs):
                if esm_extractor is None and esm_model_name:
                    esm_extractor = ESMFeatureExtractor(model_name=esm_model_name)
                    esm_loaded = True
                if esm_extractor is not None:
                    esm_extractor.extract_and_save_to_disk(
                        list(zip(seq_ids, seqs)),
                        output_dir=emb_dir,
                        max_length=max_length
                    )
            
            ds = SequenceDataset(seqs, labs, emb_dir=emb_dir,
                               seq_ids=seq_ids, max_length=max_length)
        else:
            # Fallback: vytvořit cache_dir v aktuálním adresáři
            emb_dir = os.path.join('cache', f'seq_emb_{name}_split')
            os.makedirs(emb_dir, exist_ok=True)
            
            seq_ids = [hashlib.md5(s.encode()).hexdigest()[:12] for s in seqs]
            
            existing = sum(1 for sid in seq_ids 
                           if os.path.exists(os.path.join(emb_dir, f"{sid}.npy")))
            
            if existing < len(seqs):
                if esm_extractor is None and esm_model_name:
                    esm_extractor = ESMFeatureExtractor(model_name=esm_model_name)
                    esm_loaded = True
                if esm_extractor is not None:
                    esm_extractor.extract_and_save_to_disk(
                        list(zip(seq_ids, seqs)),
                        output_dir=emb_dir,
                        max_length=max_length
                    )
            ds = SequenceDataset(seqs, labs, emb_dir=emb_dir,
                               seq_ids=seq_ids, max_length=max_length)
        
        datasets.append(ds)
    
    # Uvolni ESM pokud byl načten
    if esm_loaded and esm_extractor is not None:
        del esm_extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        log_memory("po uvolnění ESM (loaded splits)")
    
    return tuple(datasets)


def train_dual(config, graph_dataset, seq_dataset=None):
    """Spustí dual-branch trénink."""
    from dual_predictor import DualBranchPredictor
    from dual_train import DualTrainer
    from sequence_dataset import collate_sequences
    from torch.utils.data import DataLoader, Subset
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from sequence_clustering import ClusterSplitter, cluster_and_split_graphs
    
    device = config['device']
    print(f"  Device: {device}")
    
    identity_threshold = config.get('cluster_identity', 0.4)
    
    # ---- Zkus načíst existující splity ----
    loaded_graph_splits = None
    loaded_seq_splits = None
    
    if config.get('load_splits_dir'):
        print(f"  Načítám splity z: {config['load_splits_dir']}")
        loaded_graph_splits = load_splits(config['load_splits_dir'], 'graph')
        loaded_seq_splits = load_splits(config['load_splits_dir'], 'sequence')
    
    # ---- Graph data split ----
    if loaded_graph_splits is not None:
        train_graphs, val_graphs, test_graphs, meta = loaded_graph_splits
        print(f"  ✓ Načtené graph splity: "
              f"{len(train_graphs)} train, {len(val_graphs)} val, "
              f"{len(test_graphs)} test")
    elif len(graph_dataset) >= 5:
        # Extrahuj sekvence a labely z datasetu (bez stavby grafů)
        sequences = graph_dataset.sequences
        labels = graph_dataset.labels
        
        identity_threshold = config.get('cluster_identity', 0.4)
        
        train_graphs, val_graphs, test_graphs = cluster_and_split_graphs(
            graph_dataset, sequences, labels,
            identity_threshold=identity_threshold,
            val_size=0.15,
            test_size=0.15,
            random_state=42
        )
        
        print(f"  ✓ Cluster-based split: "
              f"{len(train_graphs)} train, {len(val_graphs)} val, "
              f"{len(test_graphs)} test "
              f"(identity threshold: {identity_threshold})")
        
        # Ulož splity pokud je požadováno
        if config.get('save_splits_dir'):
            save_splits(config['save_splits_dir'], 
                       train_graphs, val_graphs, test_graphs,
                       split_type='graph', config=config)
    else:
        train_graphs = list(graph_dataset)
        val_graphs = list(graph_dataset)
        test_graphs = []
    
    # Pokud je prepare-only, netrénujeme
    if config.get('prepare_only'):
        if seq_dataset is not None and len(seq_dataset) > 0:
            # Ještě musíme zpracovat sekvenční split
            from sequence_clustering import cluster_and_split_sequences
            
            seq_sequences = [seq_dataset.sequences[i] for i in range(len(seq_dataset))]
            seq_labels = [seq_dataset.labels[i] for i in range(len(seq_dataset))]
            
            seq_train, seq_val, seq_test = cluster_and_split_sequences(
                seq_dataset, seq_sequences, seq_labels,
                identity_threshold=identity_threshold,
                val_size=0.15,
                test_size=0.15,
                random_state=42
            )
            
            print(f"  Sekvence: {len(seq_train)} train, {len(seq_val)} val, "
                  f"{len(seq_test)} test")
            
            if config.get('save_splits_dir'):
                save_splits(config['save_splits_dir'],
                           seq_train, seq_val, seq_test,
                           split_type='sequence', config=config)
        
        print("\n  ✓ Prepare-only režim – datasety připraveny, trénink přeskočen.")
        return None
    
    graph_train_loader = PyGDataLoader(
        train_graphs, batch_size=config['batch_size_graph'], shuffle=True
    )
    graph_val_loader = PyGDataLoader(
        val_graphs, batch_size=config['batch_size_graph']
    )
    
    print(f"  Grafy: {len(train_graphs)} train, {len(val_graphs)} val")
    
    # ---- Sequence data split ----
    seq_train_loader = None
    seq_val_loader = None
    
    if loaded_seq_splits is not None:
        # Načtené sekvenční splity
        seq_train_ds, seq_val_ds, seq_test_ds = _build_seq_subsets_from_loaded(
            loaded_seq_splits,
            esm_model_name=config['esm_model'],
            cache_dir=config['cache_dir'],
            max_length=512
        )
        
        if seq_train_ds is not None:
            seq_train_loader = DataLoader(
                seq_train_ds, batch_size=config['batch_size_seq'],
                shuffle=True, collate_fn=collate_sequences
            )
        if seq_val_ds is not None:
            seq_val_loader = DataLoader(
                seq_val_ds, batch_size=config['batch_size_seq'],
                collate_fn=collate_sequences
            )
        
        n_train = len(seq_train_ds) if seq_train_ds else 0
        n_val = len(seq_val_ds) if seq_val_ds else 0
        n_test = len(seq_test_ds) if seq_test_ds else 0
        print(f"  Sekvence (načtené): {n_train} train, {n_val} val, {n_test} test")
    
    elif seq_dataset is not None and len(seq_dataset) > 0:
        from sequence_clustering import cluster_and_split_sequences
        
        seq_sequences = [seq_dataset.sequences[i] for i in range(len(seq_dataset))]
        seq_labels = [seq_dataset.labels[i] for i in range(len(seq_dataset))]
        
        seq_train, seq_val, seq_test = cluster_and_split_sequences(
            seq_dataset, seq_sequences, seq_labels,
            identity_threshold=identity_threshold,
            val_size=0.15,
            test_size=0.15,
            random_state=42
        )
        
        seq_train_loader = DataLoader(
            seq_train, batch_size=config['batch_size_seq'],
            shuffle=True, collate_fn=collate_sequences
        )
        seq_val_loader = DataLoader(
            seq_val, batch_size=config['batch_size_seq'],
            collate_fn=collate_sequences
        )
        
        print(f"  Sekvence: {len(seq_train)} train, {len(seq_val)} val, "
              f"{len(seq_test)} test")
        
        # Ulož sekvenční splity pokud je požadováno
        if config.get('save_splits_dir'):
            save_splits(config['save_splits_dir'],
                       seq_train, seq_val, seq_test,
                       split_type='sequence', config=config)
    
    # ---- Model ----
    model = DualBranchPredictor(
        esm_dim=config['esm_dim'],
        node_dim=config['node_dim'],
        hidden_dim=config['hidden_dim'],
        num_gnn_layers=config['num_gnn_layers'],
        num_attention_heads=config['num_attention_heads'],
        dropout=config['dropout'],
        use_gat=config['use_gat'],
        ligand_dim=config.get('ligand_dim', 36),
        esm_compress_dim=config.get('esm_compress_dim', 64)
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params:,} parametrů")
    
    # ---- Trainer ----
    trainer = DualTrainer(
        model=model,
        graph_train_loader=graph_train_loader,
        graph_val_loader=graph_val_loader,
        seq_train_loader=seq_train_loader,
        seq_val_loader=seq_val_loader,
        device=device,
        lr=config['lr'],
        consistency_weight=config['consistency_weight'],
        struct_weight=config.get('struct_weight', 5.0),
        seq_weight=config.get('seq_weight', 1.0)
    )
    
    # ---- Trénink s early stopping ----
    trainer.train(
        num_epochs=config['num_epochs'],
        patience=config.get('early_stopping_patience', 10)
    )

    return model


def train_gnn_only(config, graph_dataset):
    """Trénink pouze GNN (bez sequence branch)."""
    from binding_site_predictor import BindingSiteNADPredictor
    from train import Trainer
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from sequence_clustering import cluster_and_split_graphs
    
    device = config['device']
    
    # ---- Zkus načíst existující splity ----
    loaded_graph_splits = None
    if config.get('load_splits_dir'):
        print(f"  Načítám splity z: {config['load_splits_dir']}")
        loaded_graph_splits = load_splits(config['load_splits_dir'], 'graph')
    
    if loaded_graph_splits is not None:
        train_graphs, val_graphs, test_graphs, meta = loaded_graph_splits
        print(f"  ✓ Načtené graph splity: "
              f"{len(train_graphs)} train, {len(val_graphs)} val, "
              f"{len(test_graphs)} test")
    elif len(graph_dataset) >= 5:
        # Extrahuj sekvence a labely z datasetu (bez stavby grafů)
        sequences = (graph_dataset.sequences if hasattr(graph_dataset, 'sequences')
                     else [g.sequence if hasattr(g, 'sequence') else '' for g in graph_dataset])
        labels = (graph_dataset.labels if hasattr(graph_dataset, 'labels')
                  else [g.y.item() for g in graph_dataset])
        
        identity_threshold = config.get('cluster_identity', 0.4)
        
        train_graphs, val_graphs, test_graphs = cluster_and_split_graphs(
            graph_dataset, sequences, labels,
            identity_threshold=identity_threshold,
            val_size=0.15,
            test_size=0.15,
            random_state=42
        )
        
        print(f"  ✓ Cluster-based split: "
              f"{len(train_graphs)} train, {len(val_graphs)} val, "
              f"{len(test_graphs)} test")
        
        # Ulož splity pokud je požadováno
        if config.get('save_splits_dir'):
            save_splits(config['save_splits_dir'],
                       train_graphs, val_graphs, test_graphs,
                       split_type='graph', config=config)
    else:
        train_graphs = list(graph_dataset)
        val_graphs = train_graphs
        test_graphs = []
    
    # Pokud je prepare-only, netrénujeme
    if config.get('prepare_only'):
        print("\n  ✓ Prepare-only režim – datasety připraveny, trénink přeskočen.")
        return None
    
    train_loader = PyGDataLoader(
        train_graphs, batch_size=config['batch_size_graph'], shuffle=True
    )
    val_loader = PyGDataLoader(
        val_graphs, batch_size=config['batch_size_graph']
    )
    
    model = BindingSiteNADPredictor(
        node_dim=config['node_dim'],
        hidden_dim=config['hidden_dim'],
        num_gnn_layers=config['num_gnn_layers'],
        num_attention_heads=config['num_attention_heads'],
        dropout=config['dropout'],
        use_gat=config['use_gat'],
        ligand_dim=config.get('ligand_dim', 36),
        esm_dim=config.get('esm_dim', 1280),
        esm_compress_dim=config.get('esm_compress_dim', 64)
    )
    
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} parametrů")
    
    trainer = Trainer(model, train_loader, val_loader, device=device)
    trainer.train(num_epochs=config['num_epochs'])
    
    return model


# ============================================================
# QUICK TEST
# ============================================================
def run_test(config):
    """Rychlý test na jednom PDB souboru bez ESM (random features)."""
    print("\n" + "=" * 60)
    print("RYCHLÝ TEST (bez ESM, random features)")
    print("=" * 60)
    
    # Najdi PDB soubor
    pdb_files = glob.glob(os.path.join(config.get('pdb_positive_dir', ''), '*.pdb'))
    if not pdb_files:
        pdb_files = glob.glob(os.path.join(BASE_DIR, '*.pdb'))
    if not pdb_files:
        pdb_files = glob.glob(os.path.join(BASE_DIR, 'data', '*', 'PDB', 'positive', '**', '*.pdb'), recursive=True)
    
    if not pdb_files:
        print("❌ Žádný PDB soubor nenalezen!")
        print("   Umístěte .pdb soubor do složky projektu nebo spusťte download_data.py")
        return
    
    pdb_file = pdb_files[0]
    print(f"\n[1] Extrakce binding site z {os.path.basename(pdb_file)}...")
    
    from Binding_site_ex import BindingSiteExtractor
    extractor = BindingSiteExtractor(distance_threshold=config['distance_threshold'])
    
    try:
        bs_info = extractor.extract_binding_site(pdb_file, config['ligand_name'])
    except ValueError as e:
        print(f"  ❌ {e}")
        print(f"  Tip: zkuste jiný ligand (--ligand FAD, --ligand ATP, ...)")
        return
    
    print(f"  ✓ Nalezeno {bs_info['n_binding_site']} residues v binding site")
    print(f"  Sekvence: {bs_info['binding_site_sequence']}")
    print(f"  Contact map shape: {bs_info['contact_map'].shape}")
    
    # Random ESM features pro test
    print("\n[2] Generuji random features (bez ESM)...")
    n_bs = bs_info['n_binding_site']
    bs_info['esm_embeddings'] = np.random.randn(n_bs, 1280).astype(np.float32)
    bs_info['label'] = 1
    
    print(f"  ✓ Node features: ESM({1280}) + BLOSUM(20) + Physchem(7) + Pos(3) = 1310D")
    
    # Duplicitní data pro test
    print("\n[3] Sestavuji testovací dataset (10 kopií)...")
    test_sites = [bs_info.copy() for _ in range(10)]
    # Polovina negativních (simuluje uměle dockovaný NAD se špatnými interakcemi)
    for i in range(5, 10):
        test_sites[i] = bs_info.copy()
        test_sites[i]['label'] = 0
    
    dataset = build_graph_dataset(test_sites, include_ligand=config.get('include_ligand', True))
    
    # Trénink
    print(f"\n[4] Trénuji GNN model (5 epoch)...")
    config_test = config.copy()
    config_test['num_epochs'] = 5
    config_test['batch_size_graph'] = 4
    
    model = train_gnn_only(config_test, dataset)
    
    print("\n✅ Test úspěšný! Pipeline funguje.")
    print("   Další krok: stáhněte data a spusťte plný trénink:")
    print("     python download_data.py")
    print("     python run_pipeline.py")


# ============================================================
# MAIN
# ============================================================
def _setup_test_data_paths(config):
    """Nastaví cesty na malý testovací dataset (data/<ligand>/test/).
    
    Testovací složka obsahuje:
      - 10 pozitivních PDB struktur
      - 5 negativních PDB struktur
      - 10 pozitivních sekvencí (CSV)
      - 5 negativních sekvencí (CSV)
    
    Slouží k ověření funkčnosti celého pipeline bez nutnosti
    zpracovávat stovky/tisíce souborů.
    """
    ligand = config['ligand_name']
    data_root = config['data_root']
    test_dir = os.path.join(data_root, ligand, 'test')
    
    if not os.path.isdir(test_dir):
        print(f"  ❌ Testovací složka neexistuje: {test_dir}")
        print(f"     Vytvořte ji pomocí: mkdir -p {test_dir}/PDB/{{positive,negative}}")
        print(f"     a zkopírujte do ní několik PDB souborů.")
        return config
    
    print(f"  Používám testovací data z: {test_dir}")
    
    # PDB cesty
    config['pdb_positive_dir'] = os.path.join(test_dir, 'PDB', 'positive')
    config['pdb_negative_dir'] = os.path.join(test_dir, 'PDB', 'negative')
    
    # Sekvenční cesty
    seq_pos = os.path.join(test_dir, 'sequences', 'positive')
    seq_neg = os.path.join(test_dir, 'sequences', 'negative')
    
    if os.path.isdir(seq_pos):
        csvs = glob.glob(os.path.join(seq_pos, '*.csv'))
        if csvs:
            config['seq_positive_csv'] = csvs[0]
    
    if os.path.isdir(seq_neg):
        csvs = glob.glob(os.path.join(seq_neg, '*.csv'))
        if csvs:
            config['seq_negative_csv'] = csvs[0]
    
    # Separátní cache pro test data
    config['cache_dir'] = os.path.join(data_root, '..', 'cache', f'{ligand}_test')
    
    # Menší počet epoch pro test
    config['num_epochs'] = min(config['num_epochs'], 10)
    config['batch_size_graph'] = min(config['batch_size_graph'], 8)
    config['batch_size_seq'] = min(config['batch_size_seq'], 8)
    
    return config


def _setup_cofactor_paths(config):
    """Nastaví cesty podle zvoleného kofaktoru a datové struktury."""
    ligand = config['ligand_name']
    data_root = config['data_root']
    cofactor_dir = os.path.join(data_root, ligand)
    
    if not os.path.isdir(cofactor_dir):
        print(f"  ⚠ Složka {cofactor_dir} neexistuje, zkouším alternativní cesty...")
        return config
    
    # PDB cesty
    pdb_base = os.path.join(cofactor_dir, 'PDB')
    pos_pdb = os.path.join(pdb_base, 'positive')
    
    # Podpora pro podsložky (vycisteno_protonated apod.)
    protonated = os.path.join(pos_pdb, 'vycisteno_protonated')
    if os.path.isdir(protonated) and glob.glob(os.path.join(protonated, '*.pdb')):
        config['pdb_positive_dir'] = protonated
    elif os.path.isdir(pos_pdb) and glob.glob(os.path.join(pos_pdb, '*.pdb')):
        config['pdb_positive_dir'] = pos_pdb
    
    config['pdb_negative_dir'] = os.path.join(pdb_base, 'negative')
    
    # Sekvenční cesty – hledej CSV soubory
    seq_base = os.path.join(cofactor_dir, 'sequences')
    pos_seq_dir = os.path.join(seq_base, 'positive')
    neg_seq_dir = os.path.join(seq_base, 'negative')
    
    if os.path.isdir(pos_seq_dir):
        csvs = glob.glob(os.path.join(pos_seq_dir, '*.csv'))
        if csvs:
            config['seq_positive_csv'] = csvs[0]
    
    if os.path.isdir(neg_seq_dir):
        csvs = glob.glob(os.path.join(neg_seq_dir, '*.csv'))
        if csvs:
            config['seq_negative_csv'] = csvs[0]
    
    # Cache per cofactor
    config['cache_dir'] = os.path.join(data_root, '..', 'cache', ligand)
    
    return config


def main():
    args = parse_args()
    config = DEFAULT_CONFIG.copy()
    
    # Override z argumentů
    if args.pdb_dir:
        config['pdb_positive_dir'] = args.pdb_dir
    if args.pdb_neg_dir:
        config['pdb_negative_dir'] = args.pdb_neg_dir
    config['num_epochs'] = args.epochs
    config['batch_size_graph'] = args.batch_size
    config['batch_size_seq'] = args.batch_size
    config['lr'] = args.lr
    config['ligand_name'] = args.ligand
    config['esm_model'] = args.esm_model
    config['cluster_identity'] = args.cluster_identity
    config['struct_weight'] = args.struct_weight
    config['early_stopping_patience'] = args.patience
    
    # Save splits / prepare-only
    if args.save_splits:
        config['save_splits_dir'] = args.save_splits
    if args.prepare_only:
        if not args.save_splits:
            print("❌ --prepare-only vyžaduje --save-splits <cesta>")
            return
        config['prepare_only'] = True
    if args.load_splits:
        config['load_splits_dir'] = args.load_splits
    
    # Nastav cesty dynamicky podle kofaktoru
    if args.test_data:
        config = _setup_test_data_paths(config)
    elif not args.pdb_dir:  # Pokud uživatel nezadal explicitní cestu
        config = _setup_cofactor_paths(config)
    
    # Quick test
    if args.test:
        run_test(config)
        return
    
    print("=" * 60)
    print("SQBCP – Trénovací Pipeline")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print(f"Ligand: {config['ligand_name']}")
    print(f"PDB positive: {config['pdb_positive_dir']}")
    print(f"PDB negative: {config['pdb_negative_dir']}")
    print(f"Koncept: pozitivní = nativní vazba, negativní = uměle dockovaný ligand")
    print(f"Seq positive: {config.get('seq_positive_csv', 'N/A')}")
    print(f"Seq negative: {config.get('seq_negative_csv', 'N/A')}")
    
    # ---- KROK 1: Extrakce binding sites ----
    print(f"\n{'='*60}")
    print("[KROK 1/5] Extrakce binding sites z PDB")
    print(f"{'='*60}")
    
    binding_sites = []
    
    # Pozitivní (nativní vazba kofaktoru – kvalitní interakce)
    if os.path.exists(config['pdb_positive_dir']):
        pdb_count = len(glob.glob(os.path.join(config['pdb_positive_dir'], '*.pdb')))
        if pdb_count > 0:
            print(f"\nPozitivní příklady (nativní {config['ligand_name']} vazba):")
            pos_sites = extract_binding_sites(
                config['pdb_positive_dir'],
                config['ligand_name'],
                config['distance_threshold'],
                label=1
            )
            binding_sites.extend(pos_sites)
        else:
            print(f"\n  ⚠ Žádné PDB soubory v {config['pdb_positive_dir']}")
    else:
        print(f"\n  ⚠ Složka neexistuje: {config['pdb_positive_dir']}")
    
    # Negativní (uměle dockovaný kofaktor – špatné protein-ligand interakce)
    # Negativní PDB obsahují stejný ligand (např. NAD), ale uměle umístěný
    # (např. Boltz docking), takže protein-ligand interakce jsou nekvalitní.
    # Model se učí rozlišovat kvalitní (nativní) vs špatné (dockované) interakce.
    neg_dir = config['pdb_negative_dir']
    if os.path.exists(neg_dir):
        # Hledej PDB i v podsložkách (boltz_negatives_protonated/ apod.)
        neg_pdb_files = glob.glob(os.path.join(neg_dir, '**', '*.pdb'), recursive=True)
        if not neg_pdb_files:
            neg_pdb_files = glob.glob(os.path.join(neg_dir, '*.pdb'))
        if neg_pdb_files:
            print(f"\nNegativní příklady (uměle dockovaný {config['ligand_name']}, "
                  f"špatné interakce):")
            neg_sites = extract_binding_sites(
                neg_dir,
                config['ligand_name'],
                config['distance_threshold'],
                label=0,
                recursive=True
            )
            binding_sites.extend(neg_sites)
        else:
            print(f"\n  ℹ Složka {neg_dir} je prázdná – pokračuji bez negativních PDB")
    
    # Fallback: zkus PDB soubory v root složce
    if len(binding_sites) == 0:
        print("\n⚠ Žádné PDB ve složkách data/. Zkouším root složku...")
        root_pdbs = glob.glob(os.path.join(BASE_DIR, '*.pdb'))
        if root_pdbs:
            pos_sites = extract_binding_sites(
                BASE_DIR, config['ligand_name'],
                config['distance_threshold'], label=1
            )
            binding_sites.extend(pos_sites)
    
    if len(binding_sites) == 0:
        print("\n❌ Žádné binding sites nalezeny!")
        print(f"   Umístěte PDB soubory do: data/{config['ligand_name']}/PDB/positive/")
        return
    
    n_pos = sum(1 for bs in binding_sites if bs['label'] == 1)
    n_neg = sum(1 for bs in binding_sites if bs['label'] == 0)
    print(f"\nCelkem: {len(binding_sites)} binding sites "
          f"(pozitivní: {n_pos}, negativní: {n_neg})")
    log_memory("po extrakci binding sites")
    
    # ---- KROK 2: ESM embeddingy ----
    print(f"\n{'='*60}")
    print("[KROK 2/5] ESM-2 embeddingy")
    print(f"{'='*60}")
    
    esm_extractor = compute_esm_embeddings(
        binding_sites, config['esm_model'], config['cache_dir']
    )
    
    # Uvolni BioPython Residue/Structure objekty – po ESM extrakci
    # a _update_bs_for_valid_indices už nejsou potřeba.
    # binding_site_graph.py používá jen contact_map, ligand_atoms, 
    # protein_ligand_contacts (numpy/dict), ne BioPython objekty.
    for bs in binding_sites:
        bs.pop('binding_site_residues', None)
    gc.collect()
    log_memory("po ESM embeddingách + cleanup residues")
    
    # ---- KROK 3: Grafový dataset ----
    print(f"\n{'='*60}")
    print("[KROK 3/5] Stavba grafového datasetu")
    print(f"{'='*60}")
    
    graph_dataset = build_graph_dataset(
        binding_sites, include_ligand=config.get('include_ligand', True)
    )
    
    # Binding sites zůstávají v graph_dataset.data (lazy building)
    del binding_sites
    gc.collect()
    log_memory("po uvolnění binding_sites")
    
    # ---- KROK 4: Sekvenční dataset (volitelné) ----
    seq_dataset = None
    if not args.no_seq:
        print(f"\n{'='*60}")
        print("[KROK 4/5] Sekvenční dataset")
        print(f"{'='*60}")
        
        # Sdílíme ESM instanci z kroku 2 (pokud existuje) → bez opakovaného načítání
        seq_dataset, esm_extractor = load_sequence_data(config, esm_extractor)
        log_memory("po načtení sekvenčního datasetu")
    
    # Uvolni ESM model – už není potřeba (oba kroky dokončeny)
    if esm_extractor is not None:
        print("  Uvolnění sdílené ESM instance z paměti...")
        del esm_extractor
        esm_extractor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        log_memory("po uvolnění sdílené ESM instance")
    
    # ---- KROK 5: Trénink ----
    print(f"\n{'='*60}")
    print("[KROK 5/5] Trénink modelu")
    print(f"{'='*60}")
    
    if args.no_seq or seq_dataset is None:
        print("  Režim: GNN-only (bez sequence branch)")
        model = train_gnn_only(config, graph_dataset)
    else:
        print("  Režim: Dual-branch (GNN + Sequence)")
        model = train_dual(config, graph_dataset, seq_dataset)
    
    print(f"\n{'='*60}")
    print("✅ TRÉNINK DOKONČEN")
    print(f"{'='*60}")
    print(f"Model uložen jako best_model.pth / best_dual_model.pth")


if __name__ == '__main__':
    main()
