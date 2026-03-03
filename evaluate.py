#!/usr/bin/env python3
"""
SQBCP – Evaluace natrénovaného modelu na testovací sadě.

Načte uložený model (best_dual_model.pth) a vyhodnotí přesnost
na testovacích datech (z uložených splitů nebo přímo ze souborů).

Podporuje tři režimy:
  1. Sequence-only  – predikce čistě ze sekvence (bez struktury)
  2. Structure-only – predikce z PDB grafů (GNN větev)
  3. Both           – obě větve dohromady

Příklady spuštění:
  # Evaluace na sekvenčním testu z uložených splitů a uložení grafů:
  python evaluate.py --load-splits splits/ --mode sequence --plot-dir plots/

  # Evaluace na grafovém testu z uložených splitů:
  python evaluate.py --load-splits splits/ --mode structure --plot-dir plots/

  # Evaluace obou větví a uložení výsledků i grafů:
  python evaluate.py --load-splits splits/ --mode all --output results.json --plot-dir plots/

  # Evaluace z test dat (malý dataset):
  python evaluate.py --test-data --mode sequence

  # Predikce jedné sekvence:
  python evaluate.py --predict "MVLSPADKTNVKAAWGKVG..."

  # Predikce z FASTA souboru:
  python evaluate.py --fasta input.fasta
"""

import os
import sys
import argparse
import hashlib
import gc
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt  # PŘIDÁNO: import pro grafy
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
    precision_score, recall_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve
)

# ============================================================
# Konfigurace
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = {
    'esm_model': 'facebook/esm2_t33_650M_UR50D',
    'esm_dim': 1280,
    'node_dim': 1310,
    'esm_compress_dim': 64,
    'ligand_dim': 36,
    'hidden_dim': 256,
    'num_gnn_layers': 3,
    'num_attention_heads': 4,
    'dropout': 0.5,  # v eval módu se dropout vypne přes .eval()
    'use_gat': True,
    'max_length': 1024,
    'batch_size': 32,
    'cache_dir': os.path.join(BASE_DIR, 'cache'),
    'ligand_name': 'NAD',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='SQBCP – Evaluace modelu na testovací sadě',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Zdroj dat
    parser.add_argument('--load-splits', type=str, default=None,
                        help='Složka s uloženými splity (splits/)')
    parser.add_argument('--test-data', action='store_true',
                        help='Použít malý testovací dataset z data/NAD/test/')
    parser.add_argument('--seq-positive-csv', type=str, default=None,
                        help='CSV s pozitivními sekvencemi')
    parser.add_argument('--seq-negative-csv', type=str, default=None,
                        help='CSV s negativními sekvencemi')
    
    # Model
    parser.add_argument('--model-path', type=str, default='best_dual_model.pth',
                        help='Cesta k uloženému modelu (default: best_dual_model.pth)')
    parser.add_argument('--esm-model', type=str,
                        default='facebook/esm2_t33_650M_UR50D',
                        help='ESM-2 model (musí odpovídat tréninku)')
    
    # Režim evaluace
    parser.add_argument('--mode', type=str, default='all',
                        choices=['sequence', 'structure', 'both', 'all'],
                        help='Režim evaluace (default: all = obě větve zvlášť)')
    
    # Single prediction
    parser.add_argument('--predict', type=str, default=None,
                        help='Predikce pro jednu sekvenci (řetězec AA)')
    parser.add_argument('--fasta', type=str, default=None,
                        help='Predikce pro sekvence z FASTA souboru')
    
    # Parametry
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--ligand', type=str, default='NAD')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold pro binární klasifikaci (default: 0.5)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu, default: auto)')
    parser.add_argument('--output', type=str, default=None,
                        help='Uložit výsledky do JSON souboru')
    
    # CSV metadata
    parser.add_argument('--csv-output', type=str, default=None,
                        help='Uložit per-sample predikce do CSV souboru '
                             '(sloupce: id, true_label, pred_label, pred_prob)')
    
    # PŘIDÁNO: Složka pro uložení grafů
    parser.add_argument('--plot-dir', type=str, default=None,
                        help='Složka pro uložení vykreslených grafů (ROC a PR)')
    
    return parser.parse_args()


# ============================================================
# Načtení modelu
# ============================================================
def load_model(model_path, config, device):
    """Načte DualBranchPredictor z uloženého state_dict."""
    from dual_predictor import DualBranchPredictor
    
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
    
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model načten z {model_path}")
    print(f"  Parametry: {total_params:,}")
    print(f"  Device: {device}")
    
    return model


# ============================================================
# Načtení testovacích dat
# ============================================================
def load_test_sequences_from_splits(splits_dir, config, device):
    from run_pipeline import load_splits, _build_seq_subsets_from_loaded
    
    loaded = load_splits(splits_dir, split_type='sequence')
    if loaded is None:
        print("  ⚠ Sekvenční splity nenalezeny")
        return None
    
    train_data, val_data, test_data, metadata = loaded
    
    if isinstance(test_data, dict) and 'sequences' in test_data:
        n_test = len(test_data['sequences'])
    elif isinstance(test_data, list):
        n_test = len(test_data)
    else:
        n_test = 0
    
    if n_test == 0:
        print("  ⚠ Testovací sekvenční split je prázdný")
        return None
    
    print(f"  Test sekvencí: {n_test}")
    
    datasets = _build_seq_subsets_from_loaded(
        ([], [], test_data), 
        esm_model_name=config['esm_model'],
        cache_dir=config['cache_dir'],
        max_length=config['max_length']
    )
    
    return datasets[2]


def load_test_graphs_from_splits(splits_dir):
    from run_pipeline import load_splits
    
    loaded = load_splits(splits_dir, split_type='graph')
    if loaded is None:
        print("  ⚠ Grafové splity nenalezeny")
        return None
    
    train_data, val_data, test_data, metadata = loaded
    
    if len(test_data) == 0:
        print("  ⚠ Testovací grafový split je prázdný")
        return None
    
    print(f"  Test grafů: {len(test_data)}")
    return test_data


def load_test_data_from_dir(config, device):
    from run_pipeline import extract_binding_sites
    from esm2_feature_ex import ESMFeatureExtractor
    from binding_site_graph import BindingSiteGraphDataset
    from sequence_dataset import SequenceDataset, load_sequences_from_separate_csvs
    
    ligand = config['ligand_name']
    test_base = os.path.join(BASE_DIR, 'data', ligand, 'test')
    
    test_graphs = None
    test_seq_dataset = None
    
    # ---- Grafová data z test PDB ----
    pdb_pos = os.path.join(test_base, 'PDB', 'positive')
    pdb_neg = os.path.join(test_base, 'PDB', 'negative')
    
    if os.path.isdir(pdb_pos) or os.path.isdir(pdb_neg):
        binding_sites = []
        if os.path.isdir(pdb_pos):
            bs_pos = extract_binding_sites(
                pdb_pos, ligand, config.get('distance_threshold', 6.0),
                label=1, recursive=True
            )
            binding_sites.extend(bs_pos)
        if os.path.isdir(pdb_neg):
            bs_neg = extract_binding_sites(
                pdb_neg, ligand, config.get('distance_threshold', 6.0),
                label=0, recursive=True
            )
            binding_sites.extend(bs_neg)
        
        if binding_sites:
            esm_extractor = ESMFeatureExtractor(model_name=config['esm_model'])
            
            for bs in binding_sites:
                if 'esm_embeddings' not in bs or bs['esm_embeddings'] is None:
                    seq = bs.get('binding_site_sequence', '')
                    if seq:
                        bs['esm_embeddings'] = esm_extractor.extract_embeddings(seq)
            
            graph_dataset = BindingSiteGraphDataset(
                binding_sites,
                feature_config={
                    'use_esm': True,
                    'use_blosum': True,
                    'use_physchem': True,
                    'use_position': True
                },
                include_ligand=config.get('include_ligand', True)
            )
            test_graphs = [graph_dataset[i] for i in range(len(graph_dataset))]
            print(f"  Test grafů (z PDB): {len(test_graphs)}")
            
            del esm_extractor
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()
    
    # ---- Sekvenční data ----
    seq_pos = os.path.join(test_base, 'sequences', 'positive', 'NAD_only_dataset.csv')
    seq_neg = os.path.join(test_base, 'sequences', 'negative', 'NO_cofa_15000_id0.csv')
    
    if os.path.exists(seq_pos) or os.path.exists(seq_neg):
        if os.path.exists(seq_pos) and os.path.exists(seq_neg):
            sequences, labels = load_sequences_from_separate_csvs(seq_pos, seq_neg, cofactor_filter=ligand)
        elif os.path.exists(seq_pos):
            sequences, labels = load_sequences_from_separate_csvs(seq_pos, None, cofactor_filter=ligand)
        else:
            sequences, labels = [], []
        
        if sequences:
            emb_dir = os.path.join(config['cache_dir'], 'seq_emb_test_eval')
            os.makedirs(emb_dir, exist_ok=True)
            
            seq_ids = [hashlib.md5(s.encode()).hexdigest()[:12] for s in sequences]
            
            existing = sum(1 for sid in seq_ids if os.path.exists(os.path.join(emb_dir, f"{sid}.npy")))
            if existing < len(sequences):
                esm_extractor = ESMFeatureExtractor(model_name=config['esm_model'])
                esm_extractor.extract_and_save_to_disk(
                    list(zip(seq_ids, sequences)),
                    output_dir=emb_dir, max_length=config['max_length']
                )
                del esm_extractor
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                gc.collect()
            
            test_seq_dataset = SequenceDataset(
                sequences, labels, emb_dir=emb_dir,
                seq_ids=seq_ids, max_length=config['max_length']
            )
            print(f"  Test sekvencí (z CSV): {len(test_seq_dataset)}")
    
    return test_graphs, test_seq_dataset


def load_sequences_from_csvs(pos_csv, neg_csv, config):
    from sequence_dataset import SequenceDataset, load_sequences_from_separate_csvs
    from esm2_feature_ex import ESMFeatureExtractor
    
    sequences, labels = load_sequences_from_separate_csvs(pos_csv, neg_csv, cofactor_filter=config['ligand_name'])
    if not sequences: return None
    
    emb_dir = os.path.join(config['cache_dir'], 'seq_emb_eval_custom')
    os.makedirs(emb_dir, exist_ok=True)
    seq_ids = [hashlib.md5(s.encode()).hexdigest()[:12] for s in sequences]
    
    existing = sum(1 for sid in seq_ids if os.path.exists(os.path.join(emb_dir, f"{sid}.npy")))
    if existing < len(sequences):
        esm_extractor = ESMFeatureExtractor(model_name=config['esm_model'])
        esm_extractor.extract_and_save_to_disk(
            list(zip(seq_ids, sequences)), output_dir=emb_dir, max_length=config['max_length']
        )
        del esm_extractor
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
    
    return SequenceDataset(sequences, labels, emb_dir=emb_dir, seq_ids=seq_ids, max_length=config['max_length'])


# ============================================================
# Evaluace
# ============================================================
def evaluate_sequences(model, test_dataset, device, config, threshold=0.5):
    from torch.utils.data import DataLoader
    from sequence_dataset import collate_sequences
    
    loader = DataLoader(
        test_dataset, batch_size=config['batch_size'],
        shuffle=False, collate_fn=collate_sequences, num_workers=0
    )
    
    all_preds, all_labels, all_probs = [], [], []
    
    # Sesbírej IDs (seq_ids pokud existují, jinak indexy)
    all_ids = []
    if hasattr(test_dataset, 'seq_ids') and test_dataset.seq_ids:
        sample_ids = test_dataset.seq_ids
    elif hasattr(test_dataset, 'sequences'):
        sample_ids = [f"seq_{i}" for i in range(len(test_dataset))]
    else:
        sample_ids = [f"seq_{i}" for i in range(len(test_dataset))]
    
    idx_offset = 0
    
    model.eval()
    with torch.no_grad():
        for batch in loader:
            esm_emb = batch['embeddings'].to(device)
            seq_mask = batch['mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(mode='sequence', esm_embeddings=esm_emb, seq_mask=seq_mask)
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = (probs >= threshold).long()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # IDs pro tento batch
            batch_size = labels.size(0)
            for i in range(batch_size):
                if idx_offset + i < len(sample_ids):
                    all_ids.append(sample_ids[idx_offset + i])
                else:
                    all_ids.append(f"seq_{idx_offset + i}")
            idx_offset += batch_size
    
    results = _compute_metrics(all_labels, all_preds, all_probs, 'sequence')
    results['per_sample'] = _build_per_sample_records(all_ids, all_labels, all_preds, all_probs)
    return results


def evaluate_graphs(model, test_graphs, device, config, threshold=0.5):
    from torch_geometric.loader import DataLoader as PyGDataLoader
    
    loader = PyGDataLoader(test_graphs, batch_size=config['batch_size'], shuffle=False)
    
    all_preds, all_labels, all_probs = [], [], []
    all_ids = []
    
    # Sesbírej PDB IDs z grafů
    graph_ids = []
    for i, g in enumerate(test_graphs):
        if hasattr(g, 'pdb_file') and g.pdb_file:
            # Extrahuj basename bez přípony
            graph_ids.append(os.path.splitext(os.path.basename(g.pdb_file))[0])
        elif hasattr(g, 'name') and g.name:
            graph_ids.append(g.name)
        else:
            graph_ids.append(f"graph_{i}")
    
    idx_offset = 0
    
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, _ = model(mode='structure', graph_data=batch)
            
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = (probs >= threshold).long()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
            batch_size = batch.num_graphs
            for i in range(batch_size):
                if idx_offset + i < len(graph_ids):
                    all_ids.append(graph_ids[idx_offset + i])
                else:
                    all_ids.append(f"graph_{idx_offset + i}")
            idx_offset += batch_size
    
    results = _compute_metrics(all_labels, all_preds, all_probs, 'structure')
    results['per_sample'] = _build_per_sample_records(all_ids, all_labels, all_preds, all_probs)
    return results


def _build_per_sample_records(ids, labels, preds, probs):
    """Vytvoří seznam per-sample záznamů pro CSV výstup."""
    records = []
    for i in range(len(labels)):
        records.append({
            'id': ids[i] if i < len(ids) else f"sample_{i}",
            'true_label': int(labels[i]),
            'pred_label': int(preds[i]),
            'pred_prob': float(probs[i]),
        })
    return records


def save_predictions_csv(all_results, csv_path):
    """Uloží per-sample predikce do CSV souboru.
    
    Sloupce: branch, id, true_label, pred_label, pred_prob
    """
    import csv
    
    rows = []
    for branch_name, results in all_results.items():
        per_sample = results.get('per_sample', [])
        for record in per_sample:
            rows.append({
                'branch': branch_name,
                'id': record['id'],
                'true_label': record['true_label'],
                'pred_label': record['pred_label'],
                'pred_prob': f"{record['pred_prob']:.6f}",
            })
    
    if not rows:
        print(f"  ⚠ Žádné per-sample záznamy pro CSV")
        return
    
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['branch', 'id', 'true_label', 'pred_label', 'pred_prob'])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"  ✓ Per-sample predikce uloženy do {csv_path} ({len(rows)} záznamů)")


def _compute_metrics(labels, preds, probs, branch_name):
    """Spočítá kompletní metriky a uloží podklady pro křivky."""
    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)
    
    total = len(labels)
    correct = (preds == labels).sum()
    accuracy = correct / total if total > 0 else 0.0
    
    results = {
        'branch': branch_name,
        'n_samples': total,
        'n_positive': int((labels == 1).sum()),
        'n_negative': int((labels == 0).sum()),
        'accuracy': float(accuracy),
    }
    
    if len(set(labels)) > 1:
        results['auc_roc'] = float(roc_auc_score(labels, probs))
        results['average_precision'] = float(average_precision_score(labels, probs))
        results['f1'] = float(f1_score(labels, preds, zero_division=0))
        results['precision'] = float(precision_score(labels, preds, zero_division=0))
        results['recall'] = float(recall_score(labels, preds, zero_division=0))
        
        # PŘIDÁNO: Extrakce bodů pro křivky (uložíme si je jako pole k následnému vykreslení)
        fpr, tpr, thresholds_roc = roc_curve(labels, probs)
        prec_vals, rec_vals, thresholds_pr = precision_recall_curve(labels, probs)
        
        results['roc_curve_data'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        results['pr_curve_data'] = {'precision': prec_vals.tolist(), 'recall': rec_vals.tolist()}
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        results['confusion_matrix'] = cm.tolist()
        results['tn'] = int(cm[0, 0])
        results['fp'] = int(cm[0, 1])
        results['fn'] = int(cm[1, 0])
        results['tp'] = int(cm[1, 1])
        
        # Optimální threshold (Youden's J)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        results['optimal_threshold'] = float(thresholds_roc[best_idx])
        results['optimal_j_score'] = float(j_scores[best_idx])
    else:
        results['auc_roc'] = None
        results['note'] = 'Pouze jedna třída v testovacích datech'
    
    return results


# PŘIDÁNO: Vykreslovací funkce
def plot_evaluation_curves(all_results, plot_dir):
    """Vykreslí a uloží ROC a PR křivky ze všech vyhodnocených větví."""
    if not all_results: return
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. ROC Křivka
    plt.figure(figsize=(8, 6))
    for branch, res in all_results.items():
        if 'roc_curve_data' in res:
            fpr = res['roc_curve_data']['fpr']
            tpr = res['roc_curve_data']['tpr']
            auc = res.get('auc_roc', 0)
            plt.plot(fpr, tpr, lw=2, label=f"{branch.capitalize()} (AUC = {auc:.4f})")
            
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Náhodný model')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR / Recall)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    roc_path = os.path.join(plot_dir, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Precision-Recall Křivka
    plt.figure(figsize=(8, 6))
    for branch, res in all_results.items():
        if 'pr_curve_data' in res:
            prec = res['pr_curve_data']['precision']
            rec = res['pr_curve_data']['recall']
            ap = res.get('average_precision', 0)
            plt.plot(rec, prec, lw=2, label=f"{branch.capitalize()} (AP = {ap:.4f})")
            
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Křivka')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    
    pr_path = os.path.join(plot_dir, 'pr_curve.png')
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Grafy uloženy do: {plot_dir}")
    print(f"   - {roc_path}")
    print(f"   - {pr_path}")


def print_results(results, title=""):
    """Formátovaný výpis výsledků."""
    print(f"\n{'='*60}")
    if title:
        print(f"  {title}")
        print(f"{'='*60}")
    
    branch = results.get('branch', '?')
    print(f"  Větev:     {branch}")
    print(f"  Vzorků:    {results['n_samples']} "
          f"(pos: {results['n_positive']}, neg: {results['n_negative']})")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    
    if results.get('auc_roc') is not None:
        print(f"  AUC-ROC:   {results['auc_roc']:.4f}")
        print(f"  AP:        {results['average_precision']:.4f}")
        print(f"  F1:        {results['f1']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  Opt. threshold: {results['optimal_threshold']:.4f} "
              f"(J={results['optimal_j_score']:.4f})")
        
        print(f"\n  Zaměňovací matice:")
        print(f"              Pred=0   Pred=1")
        print(f"    True=0    {results['tn']:>6}   {results['fp']:>6}")
        print(f"    True=1    {results['fn']:>6}   {results['tp']:>6}")
    else:
        print(f"  ⚠ {results.get('note', 'AUC nelze spočítat')}")
    
    print(f"{'='*60}")


# ============================================================
# Predikce jedné sekvence
# ============================================================
def predict_single_sequence(model, sequence, device, config):
    from esm2_feature_ex import ESMFeatureExtractor
    print(f"  Sekvence: {sequence[:50]}... ({len(sequence)} AA)")
    
    esm_extractor = ESMFeatureExtractor(model_name=config['esm_model'])
    truncated = sequence[:config['max_length']]
    emb = esm_extractor.extract_embeddings(truncated)
    
    del esm_extractor
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    
    emb_tensor = torch.FloatTensor(emb).unsqueeze(0).to(device)
    mask = torch.zeros(1, emb_tensor.shape[1], dtype=torch.bool).to(device)
    
    model.eval()
    with torch.no_grad():
        logits, _ = model(mode='sequence', esm_embeddings=emb_tensor, seq_mask=mask)
        probs = F.softmax(logits, dim=1)
    
    prob_bind = probs[0, 1].item()
    prob_no_bind = probs[0, 0].item()
    
    return {
        'sequence_length': len(sequence),
        'probability_binds': prob_bind,
        'probability_no_bind': prob_no_bind,
        'prediction': 'BINDS' if prob_bind >= 0.5 else 'NO BIND',
    }


def predict_from_fasta(model, fasta_path, device, config):
    sequences, headers = [], []
    current_header, current_seq = None, []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    headers.append(current_header)
                    sequences.append(''.join(current_seq))
                current_header = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    
    if current_header is not None:
        headers.append(current_header)
        sequences.append(''.join(current_seq))
    
    if not sequences:
        print("  ⚠ FASTA soubor neobsahuje žádné sekvence")
        return []
    
    print(f"  Načteno {len(sequences)} sekvencí z {fasta_path}")
    
    from esm2_feature_ex import ESMFeatureExtractor
    esm_extractor = ESMFeatureExtractor(model_name=config['esm_model'])
    
    results = []
    model.eval()
    
    for i, (header, seq) in enumerate(zip(headers, sequences)):
        truncated = seq[:config['max_length']]
        emb = esm_extractor.extract_embeddings(truncated)
        
        emb_tensor = torch.FloatTensor(emb).unsqueeze(0).to(device)
        mask = torch.zeros(1, emb_tensor.shape[1], dtype=torch.bool).to(device)
        
        with torch.no_grad():
            logits, _ = model(mode='sequence', esm_embeddings=emb_tensor, seq_mask=mask)
            probs = F.softmax(logits, dim=1)
        
        prob_bind = probs[0, 1].item()
        
        results.append({
            'id': header,
            'length': len(seq),
            'probability_binds': prob_bind,
            'prediction': 'BINDS' if prob_bind >= 0.5 else 'NO BIND',
            "prediction_binary": 1 if prob_bind >= 0.5 else 0
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Zpracováno {i+1}/{len(sequences)}")
    
    del esm_extractor
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    
    return results


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = DEFAULT_CONFIG.copy()
    config['batch_size'] = args.batch_size
    config['ligand_name'] = args.ligand
    config['esm_model'] = args.esm_model
    
    print("=" * 60)
    print("  SQBCP – Evaluace modelu")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Model: {args.model_path}")
    print(f"  Ligand: {config['ligand_name']}")
    print(f"  Režim: {args.mode}")
    
    # ---- Načtení modelu ----
    if not os.path.exists(args.model_path):
        print(f"\n  ✗ Model nenalezen: {args.model_path}")
        print(f"    Spusťte nejdřív trénink (run_pipeline.py)")
        sys.exit(1)
    
    model = load_model(args.model_path, config, device)
    
    # ---- Režim: single prediction ----
    if args.predict:
        result = predict_single_sequence(model, args.predict, device, config)
        print(f"\n  Predikce: {result['prediction']}")
        print(f"  P(binds {config['ligand_name']}): {result['probability_binds']:.4f}")
        print(f"  P(no bind): {result['probability_no_bind']:.4f}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Výsledek uložen do {args.output}")
        return
    
    # ---- Režim: FASTA prediction ----
    if args.fasta:
        if not os.path.exists(args.fasta):
            print(f"\n  ✗ FASTA soubor nenalezen: {args.fasta}")
            sys.exit(1)
        
        results = predict_from_fasta(model, args.fasta, device, config)
        
        print(f"\n{'='*60}")
        print(f"  Výsledky predikce ({len(results)} sekvencí)")
        print(f"{'='*60}")
        print(f"  {'ID':<30} {'Délka':>6}  {'P(bind)':>8}  {'Predikce':<10}")
        print(f"  {'-'*30} {'-'*6}  {'-'*8}  {'-'*10}")
        
        for r in results:
            print(f"  {r['id']:<30} {r['length']:>6}  "
                  f"{r['probability_binds']:>8.4f}  {r['prediction']:<10}")
        
        n_bind = sum(1 for r in results if r['prediction'] == 'BINDS')
        print(f"\n  Celkem: {n_bind}/{len(results)} predikováno jako BINDS")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  Výsledky uloženy do {args.output}")
        return
    
    # ---- Načtení testovacích dat ----
    test_graphs = None
    test_seq_dataset = None
    all_results = {}
    
    if args.load_splits:
        print(f"\n  Načítám splity z: {args.load_splits}")
        if args.mode in ('sequence', 'all'):
            test_seq_dataset = load_test_sequences_from_splits(args.load_splits, config, device)
        if args.mode in ('structure', 'both', 'all'):
            test_graphs = load_test_graphs_from_splits(args.load_splits)
            
    elif args.test_data:
        print(f"\n  Načítám testovací data z data/{config['ligand_name']}/test/")
        test_graphs, test_seq_dataset = load_test_data_from_dir(config, device)
        
    elif args.seq_positive_csv or args.seq_negative_csv:
        print(f"\n  Načítám sekvence z CSV souborů")
        test_seq_dataset = load_sequences_from_csvs(args.seq_positive_csv, args.seq_negative_csv, config)
        
    else:
        print("\n  ⚠ Nebyl zadán zdroj testovacích dat.")
        print("    Použijte --load-splits, --test-data, --seq-positive-csv/--seq-negative-csv")
        print("    nebo --predict / --fasta pro predikci jednotlivých sekvencí.")
        sys.exit(1)
    
    # ---- Evaluace ----
    if args.mode in ('sequence', 'all') and test_seq_dataset is not None:
        print(f"\n  Evaluuji SEQUENCE branch...")
        seq_results = evaluate_sequences(model, test_seq_dataset, device, config, threshold=args.threshold)
        all_results['sequence'] = seq_results
        print_results(seq_results, "Sequence Branch – Test")
    
    if args.mode in ('structure', 'both', 'all') and test_graphs is not None:
        print(f"\n  Evaluuji STRUCTURE (GNN) branch...")
        gnn_results = evaluate_graphs(model, test_graphs, device, config, threshold=args.threshold)
        all_results['structure'] = gnn_results
        print_results(gnn_results, "Structure (GNN) Branch – Test")
    
    # PŘIDÁNO: Vykreslení a uložení grafů (pokud je zadán plot_dir a máme výsledky pro nějakou větev)
    if args.plot_dir and all_results:
        plot_evaluation_curves(all_results, args.plot_dir)
    
    # Souhrn
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"  SOUHRN")
        print(f"{'='*60}")
        print(f"  {'Větev':<15} {'AUC':>8} {'F1':>8} {'AP':>8} {'Acc':>8} {'N':>6}")
        print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
        for name, res in all_results.items():
            auc = f"{res['auc_roc']:.4f}" if res.get('auc_roc') else "N/A"
            f1 = f"{res['f1']:.4f}" if res.get('f1') else "N/A"
            ap = f"{res['average_precision']:.4f}" if res.get('average_precision') else "N/A"
            acc = f"{res['accuracy']:.4f}"
            n = res['n_samples']
            print(f"  {name:<15} {auc:>8} {f1:>8} {ap:>8} {acc:>8} {n:>6}")
    
    # Pokud nebyla žádná data
    if not all_results:
        print("\n  ⚠ Žádná testovací data nebyla načtena pro zadaný režim.")
        print(f"    Režim: {args.mode}")
        if test_graphs is None and args.mode in ('structure', 'both', 'all'):
            print("    → Grafový test split nenalezen")
        if test_seq_dataset is None and args.mode in ('sequence', 'all'):
            print("    → Sekvenční test split nenalezen")
    
    # Uložení výsledků
    if args.output and all_results:
        # Odstraníme data křivek a per-sample záznamy, aby JSON nebyl obrovský
        clean_results = {}
        for k, v in all_results.items():
            clean_results[k] = {k2: v2 for k2, v2 in v.items() 
                                if k2 not in ('roc_curve_data', 'pr_curve_data', 'per_sample')}
        
        with open(args.output, 'w') as f:
            json.dump(clean_results, f, indent=2, default=str)
        print(f"\n  Výsledky uloženy do {args.output}")
    
    # CSV metadata (per-sample predikce)
    if args.csv_output and all_results:
        save_predictions_csv(all_results, args.csv_output)
    
    print(f"\n  Hotovo.")

if __name__ == '__main__':
    main()