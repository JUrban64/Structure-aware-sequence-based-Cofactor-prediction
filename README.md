# SQBCP – Structure-aware Sequence-based Cofactor Binding Predictor

Binary classifier predicting whether a protein **natively** binds a given cofactor (e.g. NAD), using a **dual-branch** architecture that learns from both 3D structures (GNN on heterogeneous protein–ligand graphs) and sequences without structure (1D-CNN + Self-Attention over ESM-2 embeddings).

---

## Motivation & Problem Statement

Cofactors (NAD, FAD, ATP, COA, …) are essential for enzymatic catalysis. Predicting whether a protein binds a specific cofactor — and more importantly, whether that binding is **native (correct)** or **artificial (incorrect)** — is crucial for functional annotation and drug design.

**Key idea:** The model distinguishes *quality* of protein–ligand interactions, not merely presence/absence.

| Class | Data source | Meaning |
|-------|-------------|---------|
| **Positive (label=1)** | Experimental PDB structures | Native, evolutionarily optimized cofactor binding |
| **Negative (label=0)** | Boltz-docked PDB structures | Artificially placed cofactor with poor interactions |

Both classes contain the same ligand (e.g. NAD) in the structure. The model learns to recognize the subtle geometric and physicochemical signatures that distinguish correct binding from incorrect docking.

---

## Architecture Overview

```
                    +----------------------------------------------+
                    |              INPUT DATA                       |
                    |                                               |
                    |  PDB structures           Sequences (UniProt) |
                    |  (positive: native NAD)   (~10k with GO/EC    |
                    |  (negative: docked NAD)    annotations)       |
                    +-------+-----------------------+--------------+
                            |                       |
               +------------v------------+  +-------v---------------+
               |     GNN Branch          |  |   Sequence Branch     |
               |  (heterogeneous graph)  |  |  (1D-CNN + Attention) |
               |                         |  |                       |
               |  Protein nodes (1310D)  |  |  ESM-2 per-residue    |
               |  + Ligand nodes (36D)   |  |  embeddings [L,1280]  |
               |  3 edge types:          |  |  -> 1D-CNN (k=5, 3x)  |
               |    P-P, P-L, L-L        |  |  -> Self-Attention    |
               |  -> GAT/GCN (3 layers)  |  |  -> Learned pooling   |
               |  -> Protein-only        |  |                       |
               |    attention pooling    |  |                       |
               |  -> [B, 2H] -> proj [B,H]| |  -> [B, H]           |
               +------------+------------+  +-------+---------------+
                            |                       |
                            +-----------+-----------+
                                        |
                            +-----------v-----------+
                            |   Shared Classifier    |
                            |   (3-layer MLP)        |
                            |   H -> 256 -> 128 -> 2 |
                            +-----------+------------+
                                        |
                                        v
                          P(natively binds cofactor)
```

### Training Modes

| Mode | Input | Active branches | Use case |
|------|-------|-----------------|----------|
| `structure` | PyG graph | GNN + classifier | PDB structures (hundreds) |
| `sequence` | ESM embeddings | Seq + classifier | Sequences without structure (thousands) |
| `both` | Both | GNN + Seq + fusion + classifier | PDB data through both branches (consistency loss) |

The **shared classifier** is trained on data from both branches, giving the sequence branch access to structural supervision signal even without 3D coordinates.

---

## GNN Branch — Heterogeneous Protein–Ligand Graph

The core structural component builds a heterogeneous graph for each protein–cofactor complex:

```
PDB file
    |
    v
+----------------------------------------------------------------+
|  1. Binding Site Extraction  (Binding_site_ex.py)              |
|                                                                |
|  - Parse PDB structure (BioPython)                             |
|  - Locate cofactor (HETATM records with matching residue name) |
|  - Select protein residues within 6 A of any ligand atom      |
|  - Extract ligand atoms with element type + functional group   |
|  - Compute Ca-Ca contact map (P-P edges, < 8 A)               |
|  - Compute protein-ligand contacts (P-L edges, < 4.5 A)       |
|    with interaction classification                             |
|  - Estimate ligand covalent bonds (L-L edges, 0.5-1.9 A)      |
+------------+---------------------------------------------------+
             |
             v
+----------------------------------------------------------------+
|  2. Node Features                                              |
|                                                                |
|  PROTEIN NODES (1310D):           LIGAND NODES (36D):          |
|  +---------------------+          +----------------------+     |
|  | ESM-2 embedding  [1280] |      | Element one-hot   [5]|     |
|  | BLOSUM62 row     [20]   |      | Functional group [14]|     |
|  | Physicochemical  [7]    |      | Is aromatic       [1]|     |
|  | Relative position [3]   |      | Bond count (norm) [1]|     |
|  +---------------------+          | Cofactor ID      [15]|     |
|                                   +----------------------+     |
|  (esm2_feature_ex.py              (additional_features.py      |
|   + additional_features.py)        -> LigandFeatures)          |
+------------+---------------------------------------------------+
             |
             v
+----------------------------------------------------------------+
|  3. Heterogeneous Graph  (binding_site_graph.py)               |
|                                                                |
|  Nodes:  [0 .. n_prot-1] = protein residues                   |
|          [n_prot .. n_prot+n_lig-1] = ligand atoms             |
|                                                                |
|  Edges:                                                        |
|    P-P ---- Ca-Ca contact map ------ edge attr: distance [1D]  |
|    P-L ---- atom-residue contacts -- edge attr: dist + type [5D]|
|    L-L ---- covalent bonds --------- edge attr: distance [1D]  |
|                                                                |
|  P-L interaction types: hbond_candidate | hydrophobic          |
|                          ionic | other                         |
+------------+---------------------------------------------------+
             |
             v
+----------------------------------------------------------------+
|  4. GNN Model  (dual_predictor.py -> GNNBranch)                |
|                                                                |
|  Separate input projections:                                   |
|    protein_projection: Linear(1310 -> 256) + LN + ReLU        |
|    ligand_projection:  Linear(36 -> 256) + LN + ReLU          |
|  + Node type embedding: Embedding(2, 256)                      |
|                                                                |
|  3x GAT layers (4 heads each, concat=False):                   |
|    h^(l+1) = LN(GAT(h^(l), edge_index) + h^(l))              |
|    (residual connections from layer 2 onward)                  |
|                                                                |
|  Protein-only attention pooling:                               |
|    - Multi-head attention (4 heads) computed over all nodes    |
|    - Softmax restricted to protein nodes only                  |
|    - Ligand nodes get weight=0 in final aggregation            |
|    - + Global mean pool (protein nodes only)                   |
|  -> graph embedding [2 x hidden_dim] -> proj -> [hidden_dim]  |
+----------------------------------------------------------------+
```

**Why protein-only pooling?** Ligand nodes enrich protein representations through message passing (GNN layers propagate information across P-L edges), but the final graph embedding is computed from protein nodes alone. This ensures the classifier learns *how the protein responds to the ligand*, not the ligand itself — critical since both positive and negative examples contain the same cofactor.

---

## Sequence Branch

For sequences without known 3D structure:

```
Sequence (string)
    |
    v
+------------------------------+
|  ESM-2 (facebook/esm2_t33_  |  Per-residue embeddings
|  650M_UR50D)                 |  -> [L, 1280]
+------------+-----------------+
             |
             v
+------------------------------+
|  Input projection            |  Linear(1280 -> 256) + LN + ReLU
|                              |
|  3x 1D-CNN (kernel=5, pad=2) |  Capture local motifs
|  + residual + LayerNorm      |  (binding fingerprints)
|                              |
|  Self-Attention (4 heads)    |  Contextualize residues
|  + residual + LayerNorm      |
|                              |
|  Learned pooling query       |  Aggregate [L,H] -> [1,H]
|  (cross-attention)           |
|                              |
|  -> seq embedding [256]      |
+------------------------------+
```

---

## Training Strategy

The `DualTrainer` uses **interleaved batch training**:

1. **Sequence batch** → forward through Seq branch → update Seq branch + shared classifier
2. **Graph batch** → forward through GNN branch → update GNN branch + shared classifier
3. Repeat, alternating, until both loaders exhausted

Optional **consistency loss** (on PDB data passed through both branches):

$$\mathcal{L}_{\text{consistency}} = 1 - \frac{1}{B}\sum_{i=1}^{B} \cos(\mathbf{e}_{\text{GNN}}^{(i)},\; \mathbf{e}_{\text{Seq}}^{(i)})$$

This encourages the sequence branch to approximate structural information learned by the GNN branch, improving sequence-only inference at test time.

**Total loss:**
$$\mathcal{L} = \lambda_s \cdot \mathcal{L}_{\text{struct}} + \lambda_q \cdot \mathcal{L}_{\text{seq}} + \lambda_c \cdot \mathcal{L}_{\text{consistency}}$$

Default weights: $\lambda_s = 1.0$, $\lambda_q = 1.0$, $\lambda_c = 0.3$.

Optimizer: Adam ($\text{lr}=10^{-3}$, weight decay $10^{-5}$). Scheduler: ReduceLROnPlateau (patience=10, factor=0.5). Gradient clipping at norm 1.0.

---

## Supported Cofactors

| Cofactor | Functional groups mapped | Atom count (typical) |
|----------|--------------------------|----------------------|
| **NAD/NADP** | adenine, ribose_A, phosphate, ribose_N, nicotinamide | ~44 |
| **FAD/FMN** | isoalloxazine, ribitol, phosphate, ribose, adenine | ~53 |
| **ATP/ADP/AMP** | adenine, ribose, α/β/γ-phosphate | ~31–47 |
| **GTP/GDP** | guanine, ribose, phosphate | ~31–47 |
| **COA** | adenine, ribose, phosphate, pantothenate, cysteamine | ~51 |
| **SAM, THF, PLP, TPP, HEM** | Registered in `KNOWN_COFACTORS` (cofactor ID one-hot) | varies |

New cofactors can be added by extending `COFACTOR_FUNCTIONAL_GROUPS` in `Binding_site_ex.py` and `KNOWN_COFACTORS` in `additional_features.py`.

---

## Data Layout

```
data/
+-- NAD/                            # One directory per cofactor
    +-- PDB/
    |   +-- positive/
    |   |   +-- vycisteno_protonated/   # Native NAD-bound PDB structures
    |   |       +-- 1A4Z_H.pdb
    |   |       +-- ...
    |   +-- negative/
    |       +-- boltz_negatives_protonated/  # Boltz-docked NAD (artificial)
    |           +-- A0A067XR63_model_0_H.pdb
    |           +-- ...
    +-- sequences/
        +-- positive/
        |   +-- NAD_only_dataset.csv    # UniProt sequences with NAD annotation
        +-- negative/
            +-- NO_cofa_15000_id0.csv   # Sequences without cofactor annotation
```

**PDB files** must be protonated (contain H atoms) and include the cofactor as HETATM records.

**CSV format** for sequences:
```
uniprot_id,sequence,label,cofactor
P12345,MVLSPADKTN...,1,NAD
Q67890,MGKYVLTSIG...,0,
```

---

## Installation

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate sqbcp
```

For GPU support, edit `environment.yml`:
```yaml
# Remove:
- cpuonly
# Add:
- pytorch-cuda=12.1    # or your CUDA version
```

---

## Quick Start

```bash
# 1. Quick test (no ESM, random features, one PDB)
python run_pipeline.py --test

# 2. Full training pipeline
python run_pipeline.py --epochs 100 --ligand NAD

# 3. GNN-only (no sequence branch)
python run_pipeline.py --no-seq --epochs 100
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--test` | — | Quick test on one PDB file (random features) |
| `--ligand` | `NAD` | Cofactor type |
| `--epochs` | `100` | Number of training epochs |
| `--batch-size` | `32` | Graph batch size |
| `--lr` | `0.001` | Learning rate |
| `--no-seq` | — | Train GNN-only (skip sequence branch) |
| `--pdb-dir` | auto | Override positive PDB directory |
| `--pdb-neg-dir` | auto | Override negative PDB directory |
| `--esm-model` | `facebook/esm2_t33_650M_UR50D` | ESM-2 model name |

---

## Step-by-Step Usage (Python API)

### 1. Extract binding sites

```python
from Binding_site_ex import BindingSiteExtractor

extractor = BindingSiteExtractor(distance_threshold=6.0)
bs_info = extractor.extract_binding_site('structure.pdb', ligand_name='NAD')

print(f"Binding site: {bs_info['n_binding_site']} residues")
print(f"Ligand atoms: {len(bs_info['ligand_atoms'])}")
print(f"P-L contacts: {len(bs_info['protein_ligand_contacts'])}")
print(f"L-L bonds:    {len(bs_info['ligand_bonds'])}")
```

**Output dict keys:**

| Key | Type | Description |
|-----|------|-------------|
| `full_sequence` | `str` | Full protein sequence |
| `binding_site_sequence` | `str` | Residues within distance threshold |
| `binding_site_indices` | `list[int]` | Indices in full sequence |
| `contact_map` | `np.ndarray` | Ca–Ca contact map (P-P edges) |
| `ligand_atoms` | `list[dict]` | `{name, element, coord, functional_group}` |
| `ligand_bonds` | `list[tuple]` | `(atom_i, atom_j, distance)` |
| `protein_ligand_contacts` | `list[dict]` | `{protein_idx, ligand_idx, distance, interaction_type}` |

### 2. Compute ESM-2 embeddings

```python
from esm2_feature_ex import ESMFeatureExtractor

esm = ESMFeatureExtractor(model_name="facebook/esm2_t33_650M_UR50D")

for bs in binding_sites:
    bs['esm_embeddings'] = esm.extract_binding_site_embeddings(
        bs['full_sequence'], bs['binding_site_indices']
    )  # -> [n_residues, 1280]
```

| ESM-2 model | Parameters | Embedding dim | Memory |
|-------------|------------|---------------|--------|
| `esm2_t30_150M_UR50D` | 150M | 640 | ~1 GB |
| `esm2_t33_650M_UR50D` | **650M** | **1280** | ~3 GB |
| `esm2_t36_3B_UR50D` | 3B | 2560 | ~12 GB |

### 3. Build graph dataset

```python
from binding_site_graph import BindingSiteGraphDataset

dataset = BindingSiteGraphDataset(
    binding_sites,
    include_ligand=True,
    feature_config={
        'use_esm': True,
        'use_blosum': True,
        'use_physchem': True,
        'use_position': True
    }
)

g = dataset[0]
print(f"Protein nodes: {g.n_protein_nodes}, Ligand nodes: {g.n_ligand_nodes}")
print(f"Edge types: {g.edge_type.unique()}")  # tensor([0, 1, 2])
print(f"Cofactor: {g.cofactor_id}")
```

### 4. Train

```python
from dual_predictor import DualBranchPredictor
from dual_train import DualTrainer

model = DualBranchPredictor(
    esm_dim=1280, node_dim=1310, ligand_dim=36,
    hidden_dim=256, num_gnn_layers=3, use_gat=True
)

trainer = DualTrainer(
    model, graph_train_loader, graph_val_loader,
    seq_train_loader, seq_val_loader,
    device='cuda', consistency_weight=0.3
)
trainer.train(num_epochs=100)
```

### 5. Inference

**From PDB structure (GNN branch):**
```python
model.load_state_dict(torch.load('best_dual_model.pth'))
model.eval()
logits, _ = model(mode='structure', graph_data=graph)
prob = F.softmax(logits, dim=1)[0, 1].item()
```

**From sequence only (Seq branch):**
```python
emb = esm.extract_embeddings(sequence)
logits, _ = model(mode='sequence',
                  esm_embeddings=torch.FloatTensor(emb).unsqueeze(0))
prob = F.softmax(logits, dim=1)[0, 1].item()
```

---

## Project Structure

```
+-- run_pipeline.py           # Main orchestration script
+-- environment.yml           # Conda environment specification
|
+-- Binding_site_ex.py        # PDB parsing, binding site + ligand extraction
+-- esm2_feature_ex.py        # ESM-2 protein language model embeddings
+-- additional_features.py    # BLOSUM62, physicochemical, positional + LigandFeatures
+-- binding_site_graph.py     # Heterogeneous PyG graph construction
+-- sequence_dataset.py       # Sequence-only dataset + ESM embedding cache
+-- download_data.py          # RCSB PDB + UniProt data download utilities
|
+-- binding_site_predictor.py # GNN-only model (BindingSiteNADPredictor)
+-- dual_predictor.py         # Dual-branch model (GNNBranch + SequenceBranch)
+-- seq_only_predictor.py     # Legacy sequence-only inference wrapper
|
+-- train.py                  # GNN-only training loop
+-- dual_train.py             # Dual-branch training loop
|
+-- data/                     # Training data (per cofactor)
|   +-- NAD/
|       +-- PDB/positive/     # Native structures
|       +-- PDB/negative/     # Boltz-docked structures
|       +-- sequences/        # UniProt CSV files
+-- cache/                    # ESM embedding cache
+-- best_model.pth            # Saved GNN-only model
+-- best_dual_model.pth       # Saved dual-branch model
```

---

## Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 256 | Shared hidden dimension |
| `num_gnn_layers` | 3 | GAT/GCN layers in GNN branch |
| `num_attention_heads` | 4 | Heads in GAT and attention pooling |
| `dropout` | 0.5 | Dropout rate |
| `distance_threshold` | 6.0 Å | Binding site radius |
| `use_gat` | True | GAT (True) vs GCN (False) |
| `consistency_weight` | 0.3 | Weight of consistency loss |
| ESM dimension | 1280 | ESM-2 t33 650M |
| Protein node dim | 1310 | 1280 + 20 + 7 + 3 |
| Ligand node dim | 36 | 5 + 14 + 1 + 1 + 15 |

**Tips for small datasets (<100 structures):**
- Increase `dropout` to 0.7
- Reduce `hidden_dim` to 128
- Use fewer GNN layers (2 instead of 3)
- Reduce `batch_size` to 8–16

---

## Design Decisions

1. **ESM-2 embeddings** — protein language model trained on millions of sequences captures evolutionary and structural information without MSA computation.

2. **Heterogeneous graph** — binding site as a graph with both protein and ligand nodes; GNN learns how the protein interacts with a specific cofactor placement.

3. **Separate feature spaces** — proteins (1310D) and ligands (36D) have independent projection layers into a shared hidden space; the model learns protein–ligand interactions through P-L message passing.

4. **GAT (Graph Attention Network)** — learns which contacts are most important for prediction; well-suited for small graphs (15–50 nodes).

5. **Protein-only pooling** — final graph embedding is computed from protein nodes only; ligand nodes serve to enrich protein representations via message passing. This prevents the model from relying on ligand features alone (since both classes have the same cofactor).

6. **Dual-branch architecture** — leverages thousands of sequences from UniProt alongside hundreds of PDB structures; the shared classifier transfers structural knowledge to the sequence branch.

7. **Consistency loss** — penalizes divergence between GNN and sequence embeddings for the same protein, teaching the sequence branch to approximate structural information.

8. **Negative data design** — rather than using proteins without any cofactor (trivial to distinguish), negatives are proteins with the *same cofactor artificially docked* (Boltz). This forces the model to learn subtle interaction quality patterns.

---

## References

- **ESM-2**: Lin et al. (2023) *Evolutionary-scale prediction of atomic-level protein structure with a language model.* Science.
- **BLOSUM62**: Henikoff & Henikoff (1992) *Amino acid substitution matrices from protein blocks.* PNAS 89:10915–10919.
- **Kyte-Doolittle**: Kyte & Doolittle (1982) *A simple method for displaying the hydropathic character of a protein.* J Mol Biol 157:105–132.
- **Chou-Fasman**: Chou & Fasman (1978) *Prediction of the secondary structure of proteins from their amino acid sequence.* Adv Enzymol 47:45–148.
- **PyTorch Geometric**: Fey & Lenssen (2019) *Fast Graph Representation Learning with PyTorch Geometric.* ICLR Workshop.
- **Boltz**: Wohlwend et al. (2024) *Boltz-1: Democratizing Biomolecular Interaction Modeling.* arXiv:2412.16861.
