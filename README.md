# SQBCP – Sequence & Structure Based Cofactor Binding Predictor

**Dual-branch** prediktor vazby kofaktorů, využívající ESM-2 embeddingy, grafovou neuronovou síť (GAT/GCN) s **heterogenním protein-ligand grafem** pro strukturní data a 1D-CNN+Attention větev pro sekvence bez struktury.

> **Klíčové vlastnosti:**
> - Model se učí jak z PDB struktur (stovky), tak z anotovaných sekvencí bez struktury (tisíce z UniProt)
> - **Heterogenní graf** – proteinové i ligandové uzly s protein-ligand interakčními hranami
> - Podpora **15 typů kofaktorů**: NAD, NADP, FAD, FMN, ATP, ADP, AMP, GTP, GDP, COA, SAM, THF, PLP, TPP, HEM

---

## Architektura & logika

### Dual-branch architektura

```
                ┌─────────────────────────────────────────┐
                │           VSTUPNÍ DATA                  │
                │                                         │
                │  PDB struktury        Sekvence (UniProt)│
                │  (~500 s kofaktorem)  (~10 000 s anotací)│
                └────────┬─────────────────────┬──────────┘
                         │                     │
            ┌────────────▼──────────┐  ┌───────▼──────────────┐
            │  GNN Branch           │  │ Sequence Branch      │
            │  (heterogenní graf)   │  │ (sequence_dataset)   │
            │                       │  │                      │
            │  PDB → Binding Site   │  │ Sekvence → ESM-2     │
            │  → protein + ligand   │  │ embeddingy           │
            │    uzly               │  │ → 1D-CNN (local      │
            │  → P-P, P-L, L-L     │  │   motifs)            │
            │    hrany              │  │ → Self-Attention     │
            │  → GAT/GCN vrstvy    │  │ → Learned pooling    │
            │  → Protein-only      │  │                      │
            │    Attn pooling       │  │                      │
            │  [B, hidden_dim]      │  │ [B, hidden_dim]      │
            └────────────┬──────────┘  └───────┬──────────────┘
                         │                     │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  Shared Classifier   │
                         │  (sdílený MLP)       │
                         │  → 2 třídy           │
                         └──────────┬──────────┘
                                    │
                                    ▼
                          P(binds cofactor)
```

### Detail GNN Branch (strukturní data) – Heterogenní graf

```
PDB soubor
    │
    ▼
┌──────────────────────────┐
│  1. Binding Site Extractor│  (Binding_site_ex.py)
│  - parsuje PDB strukturu │
│  - najde ligand (kofaktor)│
│  - identifikuje residues │
│    do 6 Å od ligandu     │
│  - extrahuje ligandové   │
│    atomy + funkční skupiny│
│  - vytvoří kontaktní mapu│
│    (Cα-Cα < 8 Å)        │
│  - spočítá P-L kontakty  │
│    (< 4.5 Å) s typem     │
│    interakce              │
│  - odhadne L-L kovalentní │
│    vazby                  │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│  2. Node Features                                │
│                                                  │
│  PROTEINOVÉ UZLY (1310D):     LIGANDOVÉ UZLY (36D):  │
│  ESM-2 [1280]                 Element one-hot [5]│
│  + BLOSUM62 [20]              Func. skupina [14] │
│  + Physicochemical [7]        Aromaticita [1]    │
│  + Position [3]               N. vazeb [1]       │
│  (esm2_feature_ex.py          Cofactor ID [15]   │
│   + additional_features.py)   (LigandFeatures)   │
└──────────┬───────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│  3. Heterogenní Graf  (binding_site_graph.py)    │
│                                                  │
│  [Protein]──P-P──[Protein]   kontaktní mapa      │
│  [Protein]──P-L──[Ligand]    interakční hrany    │
│  [Ligand] ──L-L──[Ligand]    kovalentní vazby    │
│                                                  │
│  P-L edge attrs: distance + typ interakce (5D)   │
│  (hbond_candidate, hydrophobic, ionic, other)    │
└──────────┬───────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│  4. GNN (GAT/GCN)  (dual_predictor.py → GNNBranch)│
│  - Oddělené projekce:                            │
│    protein_projection (1310→256)                 │
│    ligand_projection (36→256)                    │
│  - Node type embedding (protein/ligand)          │
│  - 3× GAT/GCN vrstvy (message passing)          │
│  - Protein-only Attention pooling                │
│    (ligand uzly vyloučeny z poolingu)            │
│  → graph embedding [256]                         │
└──────────────────────────────────────────────────┘
```

### Detail Sequence Branch (sekvence bez struktury)

```
Sekvence (string)
    │
    ▼
┌──────────────────────────┐
│  1. ESM-2 Embeddings     │  (esm2_feature_ex.py)
│  → per-residue [L, 1280] │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  2. Sequence Branch      │  (dual_predictor.py → SequenceBranch)
│  - Input projection      │
│  - 3× 1D-CNN (lokální    │
│    motivy, kernel=5)     │
│  - Self-Attention        │
│  - Learned pooling       │
│  → seq embedding [256]   │
└──────────────────────────┘
```

### Trénovací režimy

| Režim | Vstup | Větev | Kdy použít |
|-------|-------|-------|------------|
| `sequence` | Sekvence (ESM emb.) | Seq branch + classifier | Sekvence bez struktury |
| `structure` | PyG graf (heterogenní) | GNN branch + classifier | PDB data se strukturou |
| `both` | Oboje | Obě + fusion + consistency | PDB data (obě větve) |

### Typy hran v heterogenním grafu

| Typ hrany | Zkratka | Zdroj | Edge atributy |
|-----------|---------|-------|---------------|
| Protein–Protein | P-P | Kontaktní mapa (Cα-Cα < 8Å) | vzdálenost (1D) |
| Protein–Ligand | P-L | Prostorový kontakt (< 4.5Å) | vzdálenost + typ interakce (5D) |
| Ligand–Ligand | L-L | Kovalentní vazby (0.5–1.9Å) | vzdálenost (1D) |

**Typy interakcí P-L hran:** `hbond_candidate`, `hydrophobic`, `ionic`, `other`

### Podporované kofaktory

| Kofaktor | Funkční skupiny (mapování atomů) |
|----------|----------------------------------|
| NAD/NADP | adenin, ribóza, fosfát, nikotinamid |
| FAD/FMN | isoalloxazin, ribitol, fosfát, adenin |
| ATP/ADP/AMP | adenin, ribóza, fosfát |
| GTP/GDP | guanin, ribóza, fosfát |
| COA | adenin, ribóza, fosfát, pantothenát, cysteamin |
| SAM, THF, PLP, TPP, HEM | (předdefinované v KNOWN_COFACTORS) |

### Proč tento přístup?

1. **ESM-2 embeddingy** – protein language model trénovaný na milionech sekvencí zachycuje evoluční a strukturní informaci bez potřeby MSA
2. **Heterogenní graf** – binding site jako graf s proteinovými i ligandovými uzly; GNN se učí, jak protein interaguje s konkrétním kofaktorem
3. **Oddělené feature prostory** – proteiny (1310D) a ligandy (36D) mají vlastní projekce do sdíleného prostoru, model se učí protein-ligand interakce skrze P-L hrany
4. **GAT (Graph Attention)** – učí se, které kontakty jsou důležitější pro predikci, vhodné pro malé grafy (15–50 uzlů)
5. **Protein-only pooling** – finální graf embedding se počítá pouze z proteinových uzlů; ligandové uzly slouží k obohacení proteinové reprezentace skrze message passing
6. **Dual-branch** – využívá mnohem více sekvenčních dat (tisíce z UniProt) vedle stovek PDB struktur
7. **Multi-cofactor ready** – architektura nativně podporuje více typů kofaktorů, rozšiřitelná přidáním nových záznamů do `COFACTOR_FUNCTIONAL_GROUPS` a `KNOWN_COFACTORS`
8. **Consistency loss** – na PDB datech penalizuje rozdíl mezi GNN a Seq embeddingy → Seq branch se učí aproximovat strukturní informaci

---

## Instalace

```bash
# 1. Vytvořit conda environment
conda env create -f environment.yml

# 2. Aktivovat
conda activate sqbcp
```

### GPU verze

V `environment.yml` změnit:
```yaml
# smazat:
- cpuonly
# přidat:
- pytorch-cuda=12.1    # nebo vaše verze CUDA
```

---

## Příprava dat

### Vstupní data

- PDB soubory s navázaným kofaktorem (NAD, ATP, FAD, COA, ...)
- Umístit do složky, např. `./pdb_files/`

### Struktura PDB souboru

- Musí obsahovat protein chain s ≥10 residues
- Musí obsahovat kofaktor jako HETATM záznam s daným jménem (např. `NAD`)

### Automatické stažení dat

```bash
python download_data.py
# Stáhne PDB soubory z RCSB + sekvence z UniProt
```

---

## Použití – krok po kroku

### Krok 1: Extrakce binding sites z PDB

```python
from Binding_site_ex import BindingSiteExtractor
import glob

extractor = BindingSiteExtractor(distance_threshold=6.0)

pdb_files = glob.glob('./pdb_files/*.pdb')

binding_sites = []
for pdb_file in pdb_files:
    try:
        bs_info = extractor.extract_binding_site(pdb_file, ligand_name='NAD')
        binding_sites.append(bs_info)
        print(f"{pdb_file}: {bs_info['n_binding_site']} residues, "
              f"{len(bs_info.get('ligand_atoms', []))} ligand atoms, "
              f"{len(bs_info.get('protein_ligand_contacts', []))} P-L contacts")
    except Exception as e:
        print(f"Error {pdb_file}: {e}")

print(f"Celkem: {len(binding_sites)} struktur")
```

**Výstup `extract_binding_site()` nyní obsahuje:**

| Klíč | Typ | Popis |
|------|-----|-------|
| `full_sequence` | str | Celá sekvence proteinu |
| `binding_site_sequence` | str | Sekvence binding site residues |
| `binding_site_indices` | list[int] | Indexy residues v sekvenci |
| `contact_map` | np.ndarray | Cα-Cα kontaktní mapa (P-P hrany) |
| `ligand_atoms` | list[dict] | Ligandové atomy: `{atom_name, element, coord, functional_group}` |
| `ligand_bonds` | list[tuple] | L-L kovalentní vazby: `(atom_i, atom_j, distance)` |
| `protein_ligand_contacts` | list[dict] | P-L kontakty: `{residue_idx, atom_idx, distance, interaction_type}` |

**Parametry:**
- `distance_threshold` – maximální vzdálenost atomu residue od ligandu (v Å), default 6.0
- `ligand_name` – třípísmenný kód ligandu v PDB (NAD, ATP, FAD, COA, ...)

### Krok 2: Extrakce ESM-2 embeddingů

```python
from esm2_feature_ex import ESMFeatureExtractor

esm_extractor = ESMFeatureExtractor(
    model_name="facebook/esm2_t33_650M_UR50D"
)

for bs_info in binding_sites:
    bs_embeddings = esm_extractor.extract_binding_site_embeddings(
        bs_info['full_sequence'],
        bs_info['binding_site_indices']
    )
    bs_info['esm_embeddings'] = bs_embeddings
    print(f"Embeddings shape: {bs_embeddings.shape}")
```

**Dostupné modely ESM-2:**

| Model | Parametry | Embedding dim | Paměť |
|-------|-----------|---------------|-------|
| `esm2_t30_150M_UR50D` | 150M | 640 | ~1 GB |
| `esm2_t33_650M_UR50D` | 650M | 1280 | ~3 GB |
| `esm2_t36_3B_UR50D` | 3B | 2560 | ~12 GB |

> **Pozor:** Při změně modelu se změní `node_dim` v prediktoru!  
> 640 + 30 = 670 (pro 150M), 1280 + 30 = 1310 (pro 650M), 2560 + 30 = 2590 (pro 3B)

### Krok 3: Sestavení grafového datasetu (heterogenní)

```python
from binding_site_graph import BindingSiteGraphDataset

# Heterogenní graf s protein + ligand uzly
dataset = BindingSiteGraphDataset(
    binding_sites,
    include_ligand=True,  # Zapnout ligandové uzly a P-L hrany
    feature_config={
        'use_esm': True,       # ESM-2 embeddingy (1280D)
        'use_blosum': True,    # BLOSUM62 encoding (20D)
        'use_physchem': True,  # Physicochemical (7D)
        'use_position': True   # Relativní pozice (3D)
    }
)

graph = dataset[0]
print(f"Grafů: {len(dataset)}")
print(f"Protein uzlů: {graph.n_protein_nodes}")
print(f"Ligand uzlů: {graph.n_ligand_nodes}")
print(f"Node types: {graph.node_type}")        # 0=protein, 1=ligand
print(f"Edge types: {graph.edge_type.unique()}") # 0=P-P, 1=P-L, 2=L-L
print(f"Cofactor: {graph.cofactor_id}")
```

### Krok 4a: Trénink (jen PDB struktury – GNN-only)

```python
import torch
from binding_site_predictor import BindingSiteNADPredictor
from train import Trainer
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

model = BindingSiteNADPredictor(
    node_dim=1310, ligand_dim=36, use_gat=True
)
train_graphs, val_graphs = train_test_split(dataset.graphs, test_size=0.2)
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=32)

trainer = Trainer(model, train_loader, val_loader, device='cpu')
trainer.train(num_epochs=100)
```

### Krok 4b: Dual trénink (PDB + sekvence – doporučeno) 🆕

Využívá mnohem více dat – sekvence z UniProt bez nutnosti 3D struktury:

```python
import torch
from dual_predictor import DualBranchPredictor
from dual_train import DualTrainer
from sequence_dataset import SequenceDataset, collate_sequences, load_sequences_from_csv
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.model_selection import train_test_split

# 1. Strukturní data (PDB) – stávající pipeline
train_graphs, val_graphs = train_test_split(dataset.graphs, test_size=0.2)
graph_train_loader = PyGDataLoader(train_graphs, batch_size=32, shuffle=True)
graph_val_loader = PyGDataLoader(val_graphs, batch_size=32)

# 2. Sekvenční data (UniProt) – NOVÝ zdroj dat
sequences, labels = load_sequences_from_csv('data/nad_sequences.csv')
seq_dataset = SequenceDataset(sequences, labels, esm_extractor=esm)
seq_train, seq_val = train_test_split(list(range(len(seq_dataset))), test_size=0.2)
seq_train_loader = DataLoader(
    torch.utils.data.Subset(seq_dataset, seq_train),
    batch_size=16, shuffle=True, collate_fn=collate_sequences
)
seq_val_loader = DataLoader(
    torch.utils.data.Subset(seq_dataset, seq_val),
    batch_size=16, collate_fn=collate_sequences
)

# 3. Dual-branch model (s heterogenním grafem)
model = DualBranchPredictor(
    esm_dim=1280, node_dim=1310, ligand_dim=36,
    hidden_dim=256, num_gnn_layers=3, use_gat=True
)

# 4. Dual trainer
trainer = DualTrainer(
    model=model,
    graph_train_loader=graph_train_loader,
    graph_val_loader=graph_val_loader,
    seq_train_loader=seq_train_loader,
    seq_val_loader=seq_val_loader,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    consistency_weight=0.3,
)
trainer.train(num_epochs=100)
```

Nejlepší model se automaticky uloží jako `best_dual_model.pth`.

**Výstup tréninku:**
```
Epoch 1/100
  Train - Loss: 0.6932, Acc: 0.5200
  Val   - Loss: 0.6815, Acc: 0.5800, AUC: 0.6120
  → New best AUC: 0.6120
...
```

---

## Predikce

### A) Ze známé struktury (PDB soubor)

```python
from Binding_site_ex import BindingSiteExtractor
from esm2_feature_ex import ESMFeatureExtractor
from additional_features import create_node_features
from binding_site_predictor import BindingSiteNADPredictor
from binding_site_graph import BindingSiteGraphDataset
import torch
import torch.nn.functional as F

# 1. Načíst model (s podporou ligandových uzlů)
model = BindingSiteNADPredictor(node_dim=1310, ligand_dim=36, use_gat=True)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 2. Extrahovat binding site (včetně ligandových atomů)
extractor = BindingSiteExtractor(distance_threshold=6.0)
bs_info = extractor.extract_binding_site('query.pdb', ligand_name='NAD')

# 3. ESM embeddingy
esm = ESMFeatureExtractor()
bs_info['esm_embeddings'] = esm.extract_binding_site_embeddings(
    bs_info['full_sequence'], bs_info['binding_site_indices']
)

# 4. Sestavit heterogenní graf
dataset = BindingSiteGraphDataset([bs_info], include_ligand=True)
graph = dataset[0]

# 5. Predikce
with torch.no_grad():
    logits = model(graph)
    prob = F.softmax(logits, dim=1)[0, 1].item()

print(f"P(binds cofactor) = {prob:.4f}")
```

### B) Jen ze sekvence – Dual model (doporučeno) 🆕

```python
from dual_predictor import DualBranchPredictor
from esm2_feature_ex import ESMFeatureExtractor
import torch
import torch.nn.functional as F

# Načíst dual model
model = DualBranchPredictor(
    esm_dim=1280, node_dim=1310, ligand_dim=36, use_gat=True
)
model.load_state_dict(torch.load('best_dual_model.pth'))
model.eval()

# ESM embeddings
esm = ESMFeatureExtractor()
sequence = "MKVLITGAGSGIGKAIA..."
emb = esm.extract_embeddings(sequence)  # [L, 1280]

# Predikce (sequence-only mode)
with torch.no_grad():
    esm_tensor = torch.FloatTensor(emb).unsqueeze(0)  # [1, L, 1280]
    logits, _ = model(mode='sequence', esm_embeddings=esm_tensor)
    prob = F.softmax(logits, dim=1)[0, 1].item()

print(f"P(binds cofactor) = {prob:.4f}")
```

> **Výhoda dual modelu:** Nevyžaduje kontaktní mapu ani 3D strukturu. Seq branch se učil na tisících sekvencí → přesnější než starý `seq_only_predictor.py`.

### C) Jen ze sekvence – starý přístup (vyžaduje contact predictor)

```python
from seq_only_predictor import SequenceOnlyPredictor

predictor = SequenceOnlyPredictor(model, esm_extractor, contact_predictor)

sequence = "MKVLITGAGSGIGKAIA..."
prob = predictor.predict(sequence)
print(f"P(binds cofactor) = {prob:.4f}")
```

> **Poznámka:** Starší přístup – vyžaduje contact predictor pro odhad kontaktní mapy.

---

## Struktura projektu

```
├── environment.yml           # Conda environment
├── README.md                 # Tento soubor
│
│── # Data pipeline
├── Binding_site_ex.py        # Extrakce binding site + ligandových atomů z PDB
├── esm2_feature_ex.py        # ESM-2 protein embeddingy
├── additional_features.py    # BLOSUM, physicochemical, pozice + 🆕 LigandFeatures
├── binding_site_graph.py     # 🆕 Heterogenní PyG grafy (protein + ligand uzly)
├── sequence_dataset.py       # Dataset pro sekvence bez struktury
├── download_data.py          # Stažení PDB + UniProt dat
│
│── # Modely
├── binding_site_predictor.py # GNN-only model (GAT/GCN) – 🆕 heterogenní graf
├── dual_predictor.py         # Dual-branch model (GNN + Seq) – 🆕 heterogenní graf
├── seq_only_predictor.py     # Starší seq-only inference wrapper
│
│── # Trénink
├── train.py                  # Tréninková smyčka (GNN-only)
├── dual_train.py             # Dual training (PDB + sekvence)
├── run_pipeline.py           # 🆕 Hlavní orchestrační skript
│
├── *.pdb                     # PDB vstupní soubory
├── best_model.pth            # Uložený GNN-only model
└── best_dual_model.pth       # Uložený dual model
```

## Důležité poznámky

1. **Negativní vzorky** – v aktuální verzi jsou všechny PDB struktury pozitivní (obsahují kofaktor). Pro trénink je nutné přidat negativní vzorky (proteiny, které kofaktor nevážou), jinak model nebude schopen rozlišovat.

2. **Heterogenní graf** – `include_ligand=True` (default) vytváří graf s proteinovými i ligandovými uzly. Pokud chcete jen proteinový graf (zpětně kompatibilní režim), nastavte `include_ligand=False`.

3. **Multi-cofactor rozšíření** – pro přidání nového kofaktoru:
   - Přidejte mapování atomů do `COFACTOR_FUNCTIONAL_GROUPS` v [Binding_site_ex.py](Binding_site_ex.py)
   - Přidejte kofaktor do `KNOWN_COFACTORS` v [additional_features.py](additional_features.py)
   - Funkční skupiny přidejte do `FUNCTIONAL_GROUPS` v [additional_features.py](additional_features.py)

4. **Protein-only pooling** – ligandové uzly obohacují proteinovou reprezentaci skrze GNN message passing, ale NEjsou zahrnuty do finálního graf embeddingu. To zajišťuje, že predikce je založena na proteinové odpovědi na ligand, nikoli na ligandu samotném.

5. **Batch size** – pro malé datasety (<100 grafů) snižte `batch_size` na 8–16.

6. **Přetrénování** – model má ~500k parametrů. Při malém datasetu zvažte:
   - Zvýšit `dropout` (0.5 → 0.7)
   - Snížit `hidden_dim` (256 → 128)
   - Použít méně GNN vrstev (3 → 2)

---

## Rychlý start

```bash
# 1. Instalace
conda env create -f environment.yml
conda activate sqbcp

# 2. Test pipeline na jednom PDB souboru
python run_pipeline.py --test

# 3. Stažení dat
python download_data.py

# 4. Plný trénink
python run_pipeline.py --epochs 100 --ligand NAD
```

---

## Citace & reference

- **ESM-2**: Lin et al. (2023) "Evolutionary-scale prediction of atomic-level protein structure with a language model." *Science*
- **Kyte-Doolittle**: Kyte & Doolittle (1982) "A simple method for displaying the hydropathic character of a protein." *J Mol Biol* 157:105-132
- **BLOSUM62**: Henikoff & Henikoff (1992) "Amino acid substitution matrices from protein blocks." *PNAS* 89:10915-10919
- **Chou-Fasman**: Chou & Fasman (1978) "Prediction of the secondary structure of proteins from their amino acid sequence." *Adv Enzymol* 47:45-148
- **PyTorch Geometric**: Fey & Lenssen (2019) "Fast Graph Representation Learning with PyTorch Geometric." *ICLR Workshop*
