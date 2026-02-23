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
import glob
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split

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
    'dropout': 0.5,
    'use_gat': True,
    'include_ligand': True,  # Přidat ligandové uzly a P-L hrany do grafu
    
    # Trénink
    'batch_size_graph': 32,
    'batch_size_seq': 16,
    'num_epochs': 100,
    'lr': 0.001,
    'consistency_weight': 0.3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Cluster-based split (ochrana proti data leakage)
    'cluster_identity': 0.4,  # CD-HIT identity threshold
                               # 0.3 = fold-level (přísné)
                               # 0.4 = superfamily-level (doporučené)
                               # 0.5 = family-level
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
                        help='CD-HIT sequence identity threshold for '
                             'cluster-based split (default: 0.4 = superfamily)')
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
def compute_esm_embeddings(binding_sites, esm_model_name, cache_dir):
    """Extrahuje ESM embeddingy pro binding sites (s cachováním)."""
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
            return
        else:
            print(f"  Cache neodpovídá ({len(cached)} vs {len(binding_sites)}), "
                  "přepočítávám...")
    
    print(f"  Načítám ESM-2 model: {esm_model_name}")
    esm = ESMFeatureExtractor(model_name=esm_model_name)
    
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
    
    # Nastavit správné labely
    for i, bs in enumerate(binding_sites):
        dataset.graphs[i].y = torch.LongTensor([bs.get('label', 1)])
    
    print(f"  ✓ {len(dataset)} grafů vytvořeno")
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
def load_sequence_data(config):
    """Načte a připraví sekvenční dataset z positive + negative CSV."""
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
    
    # Zkus načíst precomputed embeddingy
    os.makedirs(cache_dir, exist_ok=True)
    emb_cache = os.path.join(cache_dir, f'seq_embeddings_{ligand}.npz')
    precomputed = None
    if os.path.exists(emb_cache):
        print("  Načítám seq embeddingy z cache...")
        precomputed = load_embeddings(emb_cache)
    
    if precomputed and len(precomputed) == len(sequences):
        dataset = SequenceDataset(
            sequences, labels,
            precomputed_embeddings=precomputed,
            max_length=512
        )
    else:
        print(f"  Počítám ESM embeddingy pro {len(sequences)} sekvencí...")
        esm = ESMFeatureExtractor(model_name=esm_model_name)
        dataset = SequenceDataset(
            sequences, labels,
            esm_extractor=esm,
            max_length=512
        )
        # Ulož cache
        save_embeddings(dataset.precomputed, emb_cache)
    
    print(f"  ✓ {len(dataset)} sekvencí připraveno")
    return dataset


# ============================================================
# KROK 5: TRÉNINK
# ============================================================
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
    
    # ---- Graph data split (CLUSTER-BASED) ----
    if len(graph_dataset) >= 5:
        # Extrahuj sekvence a labely z datasetu
        sequences = []
        labels = []
        for g in graph_dataset.graphs:
            sequences.append(g.sequence if hasattr(g, 'sequence') else '')
            labels.append(g.y.item())
        
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
    else:
        train_graphs = graph_dataset.graphs
        val_graphs = graph_dataset.graphs
        test_graphs = []
    
    graph_train_loader = PyGDataLoader(
        train_graphs, batch_size=config['batch_size_graph'], shuffle=True
    )
    graph_val_loader = PyGDataLoader(
        val_graphs, batch_size=config['batch_size_graph']
    )
    
    print(f"  Grafy: {len(train_graphs)} train, {len(val_graphs)} val")
    
    # ---- Sequence data split (CLUSTER-BASED) ----
    seq_train_loader = None
    seq_val_loader = None
    
    if seq_dataset is not None and len(seq_dataset) > 0:
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
        consistency_weight=config['consistency_weight']
    )
    
    # ---- Trénink ----
    trainer.train(num_epochs=config['num_epochs'])
    
    return model


def train_gnn_only(config, graph_dataset):
    """Trénink pouze GNN (bez sequence branch)."""
    from binding_site_predictor import BindingSiteNADPredictor
    from train import Trainer
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from sequence_clustering import cluster_and_split_graphs
    
    device = config['device']
    
    if len(graph_dataset) >= 5:
        # Extrahuj sekvence a labely z datasetu
        sequences = []
        labels = []
        for g in (graph_dataset.graphs if hasattr(graph_dataset, 'graphs') 
                  else graph_dataset):
            sequences.append(g.sequence if hasattr(g, 'sequence') else '')
            labels.append(g.y.item())
        
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
    else:
        train_graphs = (graph_dataset.graphs if hasattr(graph_dataset, 'graphs') 
                       else list(graph_dataset))
        val_graphs = train_graphs
        test_graphs = []
    
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
    config['lr'] = args.lr
    config['ligand_name'] = args.ligand
    config['esm_model'] = args.esm_model
    config['cluster_identity'] = args.cluster_identity
    
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
    
    # ---- KROK 2: ESM embeddingy ----
    print(f"\n{'='*60}")
    print("[KROK 2/5] ESM-2 embeddingy")
    print(f"{'='*60}")
    
    compute_esm_embeddings(
        binding_sites, config['esm_model'], config['cache_dir']
    )
    
    # ---- KROK 3: Grafový dataset ----
    print(f"\n{'='*60}")
    print("[KROK 3/5] Stavba grafového datasetu")
    print(f"{'='*60}")
    
    graph_dataset = build_graph_dataset(
        binding_sites, include_ligand=config.get('include_ligand', True)
    )
    
    # ---- KROK 4: Sekvenční dataset (volitelné) ----
    seq_dataset = None
    if not args.no_seq:
        print(f"\n{'='*60}")
        print("[KROK 4/5] Sekvenční dataset")
        print(f"{'='*60}")
        
        seq_dataset = load_sequence_data(config)
    
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
