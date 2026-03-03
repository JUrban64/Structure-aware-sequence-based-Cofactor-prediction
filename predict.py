#!/usr/bin/env python3
"""
SQBCP – Predikce vazby kofaktoru pro nové sekvence.

Samostatný skript pro predikci – načte natrénovaný model a predikuje
pravděpodobnost vazby kofaktoru pro zadané proteinové sekvence.

Podporuje dva režimy:
  1. Jedna sekvence  – zadaná přímo na příkazové řádce (--predict)
  2. FASTA soubor    – predikce pro všechny sekvence v souboru (--fasta)

Příklady spuštění:
  # Predikce jedné sekvence:
  python predict.py --predict "MVLSPADKTNVKAAWGKVG..."

  # Predikce z FASTA souboru:
  python predict.py --fasta input.fasta --output results.json

  # Predikce s konkrétním modelem a ligandem:
  python predict.py --fasta input.fasta --model-path best_dual_model.pth --ligand NAD
"""

import os
import sys
import argparse
import gc
import json

import torch
import torch.nn.functional as F

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
    'dropout': 0.5,
    'use_gat': True,
    'max_length': 1024,
    'ligand_name': 'NAD',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='SQBCP – Predikce vazby kofaktoru',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Vstup
    parser.add_argument('--predict', type=str, default=None,
                        help='Predikce pro jednu sekvenci (řetězec aminokyselin)')
    parser.add_argument('--fasta', type=str, default=None,
                        help='Predikce pro sekvence z FASTA souboru')

    # Model
    parser.add_argument('--model-path', type=str, default='best_dual_model.pth',
                        help='Cesta k uloženému modelu (default: best_dual_model.pth)')
    parser.add_argument('--esm-model', type=str,
                        default='facebook/esm2_t33_650M_UR50D',
                        help='ESM-2 model (musí odpovídat tréninku)')

    # Parametry
    parser.add_argument('--ligand', type=str, default='NAD',
                        help='Název ligandu/kofaktoru (default: NAD)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold pro binární klasifikaci (default: 0.5)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu, default: auto)')
    parser.add_argument('--output', type=str, default=None,
                        help='Uložit výsledky do JSON souboru')

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
# Predikce jedné sekvence
# ============================================================
def predict_single_sequence(model, sequence, device, config):
    """Predikce pro jednu proteinovou sekvenci.

    Vrací dict s pravděpodobnostmi a binární predikcí.
    """
    from esm2_feature_ex import ESMFeatureExtractor

    print(f"  Sekvence: {sequence[:50]}... ({len(sequence)} AA)")

    esm_extractor = ESMFeatureExtractor(model_name=config['esm_model'])
    truncated = sequence[:config['max_length']]
    emb = esm_extractor.extract_embeddings(truncated)

    del esm_extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    emb_tensor = torch.FloatTensor(emb).unsqueeze(0).to(device)
    mask = torch.zeros(1, emb_tensor.shape[1], dtype=torch.bool).to(device)

    model.eval()
    with torch.no_grad():
        logits, _ = model(mode='sequence', esm_embeddings=emb_tensor, seq_mask=mask)
        probs = F.softmax(logits, dim=1)

    prob_bind = probs[0, 1].item()
    prob_no_bind = probs[0, 0].item()
    threshold = config.get('threshold', 0.5)

    return {
        'sequence_length': len(sequence),
        'probability_binds': prob_bind,
        'probability_no_bind': prob_no_bind,
        'prediction': 'BINDS' if prob_bind >= threshold else 'NO BIND',
        'prediction_binary': 1 if prob_bind >= threshold else 0,
    }


# ============================================================
# Predikce z FASTA souboru
# ============================================================
def predict_from_fasta(model, fasta_path, device, config):
    """Predikce pro všechny sekvence v FASTA souboru.

    Vrací list diktů s výsledky pro každou sekvenci.
    """
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
    threshold = config.get('threshold', 0.5)
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
            'prediction': 'BINDS' if prob_bind >= threshold else 'NO BIND',
            'prediction_binary': 1 if prob_bind >= threshold else 0,
        })

        if (i + 1) % 10 == 0:
            print(f"  Zpracováno {i + 1}/{len(sequences)}")

    del esm_extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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
    config['ligand_name'] = args.ligand
    config['esm_model'] = args.esm_model
    config['threshold'] = args.threshold

    # Musí být zadán alespoň jeden vstup
    if not args.predict and not args.fasta:
        print("  ✗ Zadejte --predict nebo --fasta")
        print("  Příklady:")
        print('    python predict.py --predict "MVLSPADKTNVKAAWGKVG..."')
        print('    python predict.py --fasta input.fasta')
        sys.exit(1)

    print("=" * 60)
    print("  SQBCP – Predikce vazby kofaktoru")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Model: {args.model_path}")
    print(f"  Ligand: {config['ligand_name']}")
    print(f"  Threshold: {args.threshold}")

    # ---- Načtení modelu ----
    if not os.path.exists(args.model_path):
        print(f"\n  ✗ Model nenalezen: {args.model_path}")
        print(f"    Spusťte nejdřív trénink (run_pipeline.py)")
        sys.exit(1)

    model = load_model(args.model_path, config, device)

    # ---- Predikce jedné sekvence ----
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

    # ---- Predikce z FASTA ----
    if args.fasta:
        if not os.path.exists(args.fasta):
            print(f"\n  ✗ FASTA soubor nenalezen: {args.fasta}")
            sys.exit(1)

        results = predict_from_fasta(model, args.fasta, device, config)

        print(f"\n{'=' * 60}")
        print(f"  Výsledky predikce ({len(results)} sekvencí)")
        print(f"{'=' * 60}")
        print(f"  {'ID':<30} {'Délka':>6}  {'P(bind)':>8}  {'Predikce':<10}")
        print(f"  {'-' * 30} {'-' * 6}  {'-' * 8}  {'-' * 10}")

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


if __name__ == '__main__':
    main()
