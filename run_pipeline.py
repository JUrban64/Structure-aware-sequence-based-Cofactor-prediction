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
    # Cesty
    'pdb_positive_dir': os.path.join(BASE_DIR, 'data', 'pdb_positive'),
    'pdb_negative_dir': os.path.join(BASE_DIR, 'data', 'pdb_negative'),
    'seq_csv': os.path.join(BASE_DIR, 'data', 'sequences', 'nad_sequences.csv'),
    'cache_dir': os.path.join(BASE_DIR, 'cache'),
    
    # Ligand
    'ligand_name': 'NAD',
    'distance_threshold': 6.0,
    
    # ESM model
    'esm_model': 'facebook/esm2_t33_650M_UR50D',
    'esm_dim': 1280,
    
    # Model
    'node_dim': 1310,  # 1280 + 20 + 7 + 3
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
}


def parse_args():
    parser = argparse.ArgumentParser(description='SQBCP Training Pipeline')
    parser.add_argument('--test', action='store_true',
                        help='Rychlý test na jednom PDB souboru')
    parser.add_argument('--pdb-dir', type=str, default=None,
                        help='Složka s PDB soubory (pozitivní)')
    parser.add_argument('--pdb-neg-dir', type=str, default=None,
                        help='Složka s PDB soubory (negativní)')
    parser.add_argument('--seq-csv', type=str, default=None,
                        help='CSV se sekvencemi')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--ligand', type=str, default='NAD')
    parser.add_argument('--no-seq', action='store_true',
                        help='Trénovat pouze na PDB (bez sequence branch)')
    parser.add_argument('--esm-model', type=str, 
                        default='facebook/esm2_t33_650M_UR50D')
    return parser.parse_args()


# ============================================================
# KROK 1: Extrakce binding sites z PDB
# ============================================================
def extract_binding_sites(pdb_dir, ligand_name, distance_threshold, 
                          label=1):
    """Extrahuje binding sites ze všech PDB souborů ve složce."""
    from Binding_site_ex import BindingSiteExtractor
    
    extractor = BindingSiteExtractor(distance_threshold=distance_threshold)
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
            bs_info['label'] = label
            binding_sites.append(bs_info)
            
            if (i + 1) % 20 == 0:
                print(f"    [{i+1}/{len(pdb_files)}] "
                      f"Zpracováno {len(binding_sites)} binding sites")
        except Exception as e:
            pass  # Tiché přeskočení problematických souborů
    
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
    
    # Zkus načíst z cache
    if os.path.exists(cache_file):
        print("  Načítám ESM embeddingy z cache...")
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        
        # Zkontroluj, jestli odpovídají
        if len(cached) == len(binding_sites):
            for i, bs in enumerate(binding_sites):
                bs['esm_embeddings'] = cached[i]
            print(f"  ✓ {len(cached)} embeddingů načteno z cache")
            return
        else:
            print(f"  Cache neodpovídá ({len(cached)} vs {len(binding_sites)}), "
                  "přepočítávám...")
    
    print(f"  Načítám ESM-2 model: {esm_model_name}")
    esm = ESMFeatureExtractor(model_name=esm_model_name)
    
    embeddings_cache = []
    for i, bs in enumerate(binding_sites):
        emb = esm.extract_binding_site_embeddings(
            bs['full_sequence'],
            bs['binding_site_indices']
        )
        bs['esm_embeddings'] = emb
        embeddings_cache.append(emb)
        
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(binding_sites)}] "
                  f"shape: {emb.shape}")
    
    # Ulož cache
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings_cache, f)
    print(f"  ✓ {len(embeddings_cache)} embeddingů uloženo do cache")


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
def load_sequence_data(csv_path, esm_model_name, cache_dir):
    """Načte a připraví sekvenční dataset."""
    from sequence_dataset import (
        SequenceDataset, load_sequences_from_csv, 
        save_embeddings, load_embeddings
    )
    from esm2_feature_ex import ESMFeatureExtractor
    
    if not os.path.exists(csv_path):
        print(f"  ⚠ CSV soubor nenalezen: {csv_path}")
        print("    Spusťte nejdřív: python download_data.py")
        return None
    
    sequences, labels = load_sequences_from_csv(csv_path)
    
    if len(sequences) == 0:
        print("  ⚠ Žádné sekvence nenačteny")
        return None
    
    # Zkus načíst precomputed embeddingy
    emb_cache = os.path.join(cache_dir, 'seq_embeddings.npz')
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
    
    device = config['device']
    print(f"  Device: {device}")
    
    # ---- Graph data split ----
    if len(graph_dataset) >= 5:
        train_graphs, val_graphs = train_test_split(
            graph_dataset.graphs, test_size=0.2, random_state=42
        )
    else:
        train_graphs = graph_dataset.graphs
        val_graphs = graph_dataset.graphs  # Malý dataset → stejná data
    
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
    
    if seq_dataset is not None and len(seq_dataset) > 0:
        indices = list(range(len(seq_dataset)))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, random_state=42
        )
        
        seq_train_loader = DataLoader(
            Subset(seq_dataset, train_idx),
            batch_size=config['batch_size_seq'],
            shuffle=True,
            collate_fn=collate_sequences
        )
        seq_val_loader = DataLoader(
            Subset(seq_dataset, val_idx),
            batch_size=config['batch_size_seq'],
            collate_fn=collate_sequences
        )
        
        print(f"  Sekvence: {len(train_idx)} train, {len(val_idx)} val")
    
    # ---- Model ----
    model = DualBranchPredictor(
        esm_dim=config['esm_dim'],
        node_dim=config['node_dim'],
        hidden_dim=config['hidden_dim'],
        num_gnn_layers=config['num_gnn_layers'],
        num_attention_heads=config['num_attention_heads'],
        dropout=config['dropout'],
        use_gat=config['use_gat'],
        ligand_dim=config.get('ligand_dim', 36)
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
    
    device = config['device']
    
    if len(graph_dataset) >= 5:
        train_graphs, val_graphs = train_test_split(
            graph_dataset.graphs, test_size=0.2, random_state=42
        )
    else:
        train_graphs = graph_dataset.graphs
        val_graphs = graph_dataset.graphs
    
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
        ligand_dim=config.get('ligand_dim', 36)
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
    pdb_files = glob.glob(os.path.join(BASE_DIR, '*.pdb'))
    if not pdb_files:
        pdb_files = glob.glob(os.path.join(BASE_DIR, 'data', 'pdb_positive', '*.pdb'))
    
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
    # Polovina negativních
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
def main():
    args = parse_args()
    config = DEFAULT_CONFIG.copy()
    
    # Override z argumentů
    if args.pdb_dir:
        config['pdb_positive_dir'] = args.pdb_dir
    if args.pdb_neg_dir:
        config['pdb_negative_dir'] = args.pdb_neg_dir
    if args.seq_csv:
        config['seq_csv'] = args.seq_csv
    config['num_epochs'] = args.epochs
    config['batch_size_graph'] = args.batch_size
    config['lr'] = args.lr
    config['ligand_name'] = args.ligand
    config['esm_model'] = args.esm_model
    
    # Quick test
    if args.test:
        run_test(config)
        return
    
    print("=" * 60)
    print("SQBCP – Trénovací Pipeline")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print(f"Ligand: {config['ligand_name']}")
    
    # ---- KROK 1: Extrakce binding sites ----
    print(f"\n{'='*60}")
    print("[KROK 1/5] Extrakce binding sites z PDB")
    print(f"{'='*60}")
    
    binding_sites = []
    
    # Pozitivní (s NAD)
    if os.path.exists(config['pdb_positive_dir']):
        print(f"\nPozitivní příklady (s {config['ligand_name']}):")
        pos_sites = extract_binding_sites(
            config['pdb_positive_dir'],
            config['ligand_name'],
            config['distance_threshold'],
            label=1
        )
        binding_sites.extend(pos_sites)
    
    # Negativní (bez NAD) – hledáme jiný ligand nebo žádný
    if os.path.exists(config['pdb_negative_dir']):
        print(f"\nNegativní příklady (bez {config['ligand_name']}):")
        neg_pdb_files = glob.glob(
            os.path.join(config['pdb_negative_dir'], '*.pdb')
        )
        # Pro negativní: předstíráme binding site = prvních 20 residues
        from Binding_site_ex import BindingSiteExtractor
        ext = BindingSiteExtractor(distance_threshold=config['distance_threshold'])
        
        neg_count = 0
        for pdb_file in neg_pdb_files:
            try:
                # Zkus najít binding site s jiným ligandem, nebo vezmi
                # náhodnou oblast
                structure = ext.parser.get_structure('prot', pdb_file)
                model = structure[0]
                chain = next(iter(model))
                seq = ext._get_sequence(chain)
                
                if len(seq) < 20:
                    continue
                
                # "Fake" binding site = centrální oblast sekvence
                center = len(seq) // 2
                bs_indices = list(range(max(0, center-10), min(len(seq), center+10)))
                bs_sequence = ''.join([seq[i] for i in bs_indices])
                
                # Kontaktní mapa z náhodné oblasti
                n = len(bs_indices)
                contact = np.eye(n)
                for k in range(n-1):
                    contact[k, k+1] = 1.0
                    contact[k+1, k] = 1.0
                
                bs_info = {
                    'full_sequence': seq,
                    'binding_site_sequence': bs_sequence,
                    'binding_site_indices': bs_indices,
                    'binding_site_residues': [],
                    'contact_map': contact,
                    'n_binding_site': n,
                    'ligand_name': 'NONE',
                    'pdb_file': pdb_file,
                    'label': 0
                }
                binding_sites.append(bs_info)
                neg_count += 1
            except Exception:
                pass
        
        print(f"  ✓ {neg_count} negativních příkladů")
    
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
        print("\n❌ Žádné binding sites! Spusťte nejdřív:")
        print("   python download_data.py")
        print("   # nebo umístěte PDB soubory do data/pdb_positive/")
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
    
    graph_dataset = build_graph_dataset(binding_sites, include_ligand=config.get('include_ligand', True))
    
    # ---- KROK 4: Sekvenční dataset (volitelné) ----
    seq_dataset = None
    if not args.no_seq:
        print(f"\n{'='*60}")
        print("[KROK 4/5] Sekvenční dataset (UniProt)")
        print(f"{'='*60}")
        
        seq_dataset = load_sequence_data(
            config['seq_csv'], config['esm_model'], config['cache_dir']
        )
    
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
