# SQBCP – Structure-aware Sequence-based Cofactor Binding Predictor

Binary classifier that predicts whether a protein **natively** binds a given cofactor (e.g. NAD). Uses a **dual-branch** architecture: a GNN over protein–ligand interaction graphs (when 3D structure is available) and a 1D-CNN + Self-Attention over ESM-2 embeddings (for sequences without structure).

---

## Problem

Both positive and negative examples contain the **same cofactor** in the structure. The model learns to distinguish **native** (experimentally determined) binding from **artificial** (Boltz-docked) binding based on protein–ligand interaction quality.

| Class | Source | Meaning |
|-------|--------|---------|
| Positive (1) | Experimental PDB | Native cofactor binding |
| Negative (0) | Boltz-docked PDB | Artificially placed cofactor |

---

## Architecture

```
  PDB structure                         Sequence (UniProt)
       |                                       |
       v                                       v
  +-----------------+                  +------------------+
  |   GNN Branch    |                  | Sequence Branch  |
  |                 |                  |                  |
  | Protein nodes   |                  | ESM-2 [L, 1280] |
  |  (1310D)        |                  | -> 1D-CNN (x3)  |
  | Ligand nodes    |                  | -> Self-Attn     |
  |  (36D)          |                  | -> Pooling       |
  | P-P, P-L, L-L   |                  |                  |
  | -> GAT (x3)     |                  |                  |
  | -> Attn pooling  |                  |                  |
  +--------+--------+                  +--------+---------+
           |                                    |
           +----------------+-------------------+
                            v
                    Shared Classifier
                     (MLP -> P(binds))
```

### GNN Branch (structure)

1. **Binding site extraction** – residues within 6 Å of cofactor, including explicit hydrogens for H-bond detection
2. **Node features** – protein: ESM-2 (1280) + BLOSUM62 (20) + physicochemical (7) + position (3) = 1310D; ligand: element (6) + functional group (14) + aromaticity (1) + bonds (1) + cofactor ID (15) = 36D (Note: element one-hot includes H for hydrogen atoms)
3. **Edges** – P-P (Cα contact map < 8 Å), P-L (atom contacts < 4.5 Å with interaction type), L-L (covalent bonds)
4. **GNN** – 3× GAT layers with residual connections, then attention pooling over **protein nodes only**

### Sequence Branch (no structure needed)

ESM-2 per-residue embeddings → 3× 1D-CNN (kernel=5) → Self-Attention (4 heads) → learned pooling → embedding

### Training

Interleaved batches from both branches. Optional consistency loss aligns GNN and Sequence embeddings for the same protein.

**Metrics:** AUC, F1 score, Average Precision (per-branch and combined).

**Data split:** Cluster-based (MMseqs2, default 40% identity) to prevent data leakage between train/val/test.

---

## Data Layout

```
data/<COFACTOR>/
├── PDB/
│   ├── positive/           # Native cofactor-bound structures (protonated)
│   └── negative/           # Boltz-docked structures (protonated)
└── sequences/
    ├── positive/           # CSV: UniProt sequences with cofactor annotation
    └── negative/           # CSV: sequences without cofactor annotation
```

PDB files must be **protonated** (contain H atoms) with the cofactor as HETATM records.

CSV format: `uniprot_id,sequence,label,cofactor`

---

## Installation

```bash
conda env create -f environment.yml      # CPU
conda env create -f environment_gpu.yml  # GPU (CUDA 12.1)
conda activate sqbcp
```

---

## Usage

### Quick test
```bash
python run_pipeline.py --test
```

### Full training
```bash
python run_pipeline.py --ligand NAD --epochs 100
```

### GNN-only (no sequence branch)
```bash
python run_pipeline.py --no-seq
```

### Prepare datasets without training
```bash
python run_pipeline.py --save-splits ./splits --prepare-only
```

### Train from pre-saved splits
```bash
python run_pipeline.py --load-splits ./splits
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--test` | — | Quick test with random features |
| `--test-data` | — | Use small test dataset from `data/<ligand>/test/` |
| `--ligand` | `NAD` | Cofactor type |
| `--epochs` | `100` | Training epochs |
| `--batch-size` | `32` | Graph batch size |
| `--lr` | `0.001` | Learning rate |
| `--no-seq` | — | GNN-only mode |
| `--esm-model` | `facebook/esm2_t33_650M_UR50D` | ESM-2 model |
| `--cluster-identity` | `0.4` | Sequence identity threshold for split (0.3–0.9) |
| `--save-splits` | — | Save train/val/test splits to directory |
| `--prepare-only` | — | Only prepare datasets, skip training (requires `--save-splits`) |
| `--load-splits` | — | Load pre-saved splits instead of re-clustering |
| `--pdb-dir` | auto | Override positive PDB directory |
| `--pdb-neg-dir` | auto | Override negative PDB directory |

---

## Project Structure

```
run_pipeline.py             Main orchestration (extraction → features → split → train)
Binding_site_ex.py          PDB parsing, binding site + ligand extraction
esm2_feature_ex.py          ESM-2 embedding extraction
additional_features.py      BLOSUM62, physicochemical, positional + ligand features
binding_site_graph.py       PyG graph construction (heterogeneous P-L graph)
sequence_dataset.py         Sequence-only dataset + ESM cache
sequence_clustering.py      MMseqs2 clustering + cluster-based train/val/test split
download_data.py            RCSB PDB + UniProt download utilities
dual_predictor.py           Dual-branch model (GNNBranch + SequenceBranch)
dual_train.py               Dual-branch trainer
binding_site_predictor.py   GNN-only model
train.py                    GNN-only trainer
visualize_graph.py          Graph visualization
```

---

## Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `hidden_dim` | 256 | Reduce to 128 for small datasets |
| `num_gnn_layers` | 3 | |
| `num_attention_heads` | 4 | |
| `dropout` | 0.5 | Increase to 0.7 for <100 structures |
| `distance_threshold` | 6.0 Å | Binding site radius |
| `consistency_weight` | 0.3 | 0 = disable consistency loss |
| `cluster_identity` | 0.4 | 0.3 = strict, 0.7 = relaxed |

---

## Supported Cofactors

NAD, NADP, FAD, FMN, ATP, ADP, AMP, GTP, GDP, COA, SAM, THF, PLP, TPP, HEM

Add new cofactors by extending `COFACTOR_FUNCTIONAL_GROUPS` in `Binding_site_ex.py` and `KNOWN_COFACTORS` in `additional_features.py`.

---

## References

- **ESM-2**: Lin et al. (2023) Science.
- **PyTorch Geometric**: Fey & Lenssen (2019) ICLR Workshop.
- **Boltz**: Wohlwend et al. (2024) arXiv:2412.16861.
