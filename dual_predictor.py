"""
Dual-branch model: Structure (GNN) + Sequence (1D-CNN/Attention)

Umožňuje trénovat na:
  1. PDB datech se strukturou (GNN větev)
  2. Sekvencích bez struktury (Sequence větev) – mnohem více dat
  3. PDB datech oběma větvemi současně (consistency loss)

Architektura (s protein-ligand interakčním grafem):

  ┌────────────────────────────────────────────┐
  │ GNN Branch (heterogenní graf)              │
  │                                            │
  │  Protein nodes ──P-P──▶ GAT ◀──P-L──┐     │
  │  (1310D → proj)                      │     │
  │                         GAT ◀──L-L──┐│     │
  │  Ligand nodes ─────────────────────▶ ││    │
  │  (36D → proj)                        ▼▼    │
  │                      attention pooling     │
  │                      (protein nodes only)  │
  │                      → graph emb [H]       │
  └──────────────────────┬─────────────────────┘
                         │
  ┌──────────────────────┴──────────────────────┐
  │              Shared Classifier              │
  └─────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────┐
  │ Sequence Branch (beze změny)                │
  │ ESM → 1D-CNN → Self-Attention → pooling     │
  └──────────────────────┬──────────────────────┘
                         │
                         ▼
                  P(binds cofactor)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool

from additional_features import LigandFeatures


# ============================================================
# SEQUENCE BRANCH – pro sekvence BEZ struktury
# ============================================================
class SequenceBranch(nn.Module):
    """
    Zpracovává sekvenci (ESM embeddings) bez grafové struktury.
    Používá 1D konvoluce + self-attention k extrakci globální
    reprezentace sekvence.
    
    Vstup: ESM per-residue embeddings [batch, L, esm_dim]
    Výstup: sequence embedding [batch, hidden_dim]
    """
    
    def __init__(self, esm_dim=1280, hidden_dim=256, 
                 num_cnn_layers=3, num_attention_heads=4, dropout=0.3):
        super().__init__()
        
        self.esm_dim = esm_dim
        self.hidden_dim = hidden_dim
        
        # ---- Input projection ----
        self.input_proj = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ---- 1D CNN layers (capture local motifs) ----
        self.cnn_layers = nn.ModuleList()
        self.cnn_norms = nn.ModuleList()
        for i in range(num_cnn_layers):
            self.cnn_layers.append(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
            )
            self.cnn_norms.append(nn.LayerNorm(hidden_dim))
        
        # ---- Self-attention pooling ----
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # ---- Pooling attention (aggregate seq → single vector) ----
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pool_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, esm_embeddings, mask=None):
        """
        Args:
            esm_embeddings: [batch, L, esm_dim]  per-residue ESM embeddings
            mask: [batch, L] bool mask (True = padding, ignore)
        
        Returns:
            seq_embedding: [batch, hidden_dim]
        """
        batch_size = esm_embeddings.size(0)
        
        # Input projection: [B, L, esm_dim] → [B, L, hidden_dim]
        x = self.input_proj(esm_embeddings)
        
        # 1D CNN: [B, L, H] → [B, H, L] → CNN → [B, L, H]
        for cnn, norm in zip(self.cnn_layers, self.cnn_norms):
            residual = x
            x_conv = x.transpose(1, 2)           # [B, H, L]
            x_conv = cnn(x_conv)                  # [B, H, L]
            x_conv = x_conv.transpose(1, 2)       # [B, L, H]
            x = norm(x_conv + residual)           # residual
            x = F.relu(x)
            x = self.dropout(x)
        
        # Self-attention: contextualize residues
        key_padding_mask = mask if mask is not None else None
        attn_out, _ = self.attention(
            x, x, x, key_padding_mask=key_padding_mask
        )
        x = self.attn_norm(attn_out + x)  # residual
        
        # Learned pooling: use learnable query to aggregate
        query = self.pool_query.expand(batch_size, -1, -1)  # [B, 1, H]
        pooled, _ = self.pool_attention(
            query, x, x, key_padding_mask=key_padding_mask
        )
        seq_embedding = pooled.squeeze(1)  # [B, H]
        
        return seq_embedding


# ============================================================
# GNN BRANCH – heterogenní protein-ligand graf
# ============================================================
class GNNBranch(nn.Module):
    """
    Zpracovává protein-ligand interakční graf.
    
    Klíčové úpravy oproti homogennímu grafu:
      - Separate input projection pro protein (1310D) a ligand (36D) uzly
      - Node type embedding přidaný k uzlovým features
      - Edge type embedding (P-P=0, P-L=1, L-L=2) 
      - Attention pooling POUZE přes protein uzly (graf embedding
        reprezentuje protein, ne ligand)
    
    Vstup: PyG Batch s atributy:
        x, edge_index, edge_attr, batch,
        node_type [N], edge_type [E],
        n_protein_nodes, protein_dim, ligand_dim
    
    Výstup: graph embedding [batch_size, 2*hidden_dim]
    """
    
    # Konstanty
    NUM_NODE_TYPES = 2   # 0=protein, 1=ligand
    NUM_EDGE_TYPES = 3   # 0=PP, 1=PL, 2=LL
    
    def __init__(self, node_dim=1310, hidden_dim=256, 
                 num_gnn_layers=3, num_attention_heads=4,
                 dropout=0.5, use_gat=True,
                 ligand_dim=None):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_gat = use_gat
        self.node_dim = node_dim  # protein feature dim
        
        if ligand_dim is None:
            ligand_dim = LigandFeatures.LIGAND_FEAT_DIM  # 36
        self.ligand_dim = ligand_dim
        
        # ---- Separate input projections ----
        # Protein: 1310D → hidden_dim
        self.protein_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Ligand: 36D → hidden_dim
        self.ligand_projection = nn.Sequential(
            nn.Linear(ligand_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ---- Node type embedding ----
        self.node_type_embedding = nn.Embedding(self.NUM_NODE_TYPES, hidden_dim)
        
        # ---- Edge type embedding ----
        self.edge_type_embedding = nn.Embedding(self.NUM_EDGE_TYPES, hidden_dim)
        
        # ---- GNN layers ----
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        if use_gat:
            for i in range(num_gnn_layers):
                self.gnn_layers.append(
                    GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
                )
                self.norms.append(nn.LayerNorm(hidden_dim))
        else:
            for i in range(num_gnn_layers):
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
                self.norms.append(nn.LayerNorm(hidden_dim))
        
        # ---- Multi-head attention pooling (přes protein nodes) ----
        self.attention_dim = 128
        self.num_heads = num_attention_heads
        self.W1 = nn.Linear(hidden_dim, self.attention_dim)
        self.W2 = nn.Linear(self.attention_dim, num_attention_heads)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        """
        Args:
            data: PyG Batch object s heterogenními atributy
        
        Returns:
            graph_embedding: [batch_size, 2*hidden_dim]
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        
        # Získej node_type (pokud neexistuje → všechno protein)
        node_type = getattr(data, 'node_type', None)
        edge_type = getattr(data, 'edge_type', None)
        
        if node_type is None:
            node_type = torch.zeros(x.size(0), dtype=torch.long, 
                                    device=x.device)
        
        # ---- Separate input projection ----
        protein_mask = (node_type == 0)
        ligand_mask = (node_type == 1)
        
        # Projektuj protein a ligand uzly zvlášť
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        
        if protein_mask.any():
            # Protein uzly: vezmi jen prvních protein_dim features
            prot_x = x[protein_mask, :self.node_dim]
            h[protein_mask] = self.protein_projection(prot_x)
        
        if ligand_mask.any():
            # Ligand uzly: vezmi jen prvních ligand_dim features
            lig_x = x[ligand_mask, :self.ligand_dim]
            h[ligand_mask] = self.ligand_projection(lig_x)
        
        # ---- Přidej node type embedding ----
        h = h + self.node_type_embedding(node_type)
        
        # ---- GNN propagation ----
        for i, (gnn, norm) in enumerate(zip(self.gnn_layers, self.norms)):
            if self.use_gat:
                x_new = gnn(h, edge_index)
            else:
                x_new = gnn(h, edge_index)
            
            if i > 0:
                x_new = x_new + h  # residual
            
            h = norm(x_new)
            h = F.relu(h)
            h = self.dropout(h)
        
        # M: final node embeddings [total_nodes, hidden_dim]
        M = h
        
        # ---- Attention pooling POUZE přes protein uzly ----
        # Ligand uzly ovlivnily protein embeddingy přes message passing,
        # ale finální embedding grafu je agregace proteinových uzlů.
        
        # Maskuj ligand uzly pro pooling: nastavíme jim -inf attention
        attention_input = self.W1(M)
        attention_scores = self.W2(torch.tanh(attention_input))
        
        # Protein-only softmax: ligand uzly dostanou score -inf
        attention_weights = self._batch_softmax_masked(
            attention_scores, batch, protein_mask
        )
        
        pooled_per_head = []
        for head in range(self.num_heads):
            weights = attention_weights[:, head].unsqueeze(1)
            weighted_features = M * weights
            pooled = global_add_pool(weighted_features, batch)
            pooled_per_head.append(pooled)
        
        attention_pooled = torch.stack(pooled_per_head).mean(dim=0)
        
        # Global pooling (jen protein)
        M_protein = M.clone()
        M_protein[ligand_mask] = 0.0
        global_pooled = global_mean_pool(M_protein, batch)
        
        combined = torch.cat([attention_pooled, global_pooled], dim=1)
        
        return combined  # [batch_size, 2 * hidden_dim]
    
    def _batch_softmax_masked(self, scores, batch, protein_mask):
        """
        Softmax per graph, ale POUZE přes protein uzly.
        Ligand uzly dostanou weight = 0.
        """
        batch_size = batch.max().item() + 1
        num_heads = scores.size(1)
        weights = torch.zeros_like(scores)
        
        for i in range(batch_size):
            graph_mask = (batch == i)
            prot_in_graph = graph_mask & protein_mask
            
            if not prot_in_graph.any():
                # Fallback: všechny uzly (nemá protein?)
                for head in range(num_heads):
                    graph_scores = scores[graph_mask, head]
                    weights[graph_mask, head] = F.softmax(graph_scores, dim=0)
                continue
            
            for head in range(num_heads):
                prot_scores = scores[prot_in_graph, head]
                prot_weights = F.softmax(prot_scores, dim=0)
                weights[prot_in_graph, head] = prot_weights
                # Ligand uzly zůstanou na 0
        
        return weights


# ============================================================
# DUAL-BRANCH MODEL
# ============================================================
class DualBranchPredictor(nn.Module):
    """
    Kombinuje GNN (struktura) a Sequence (1D-CNN+Attn) větve.
    
    Trénovací režimy:
      - 'structure': pouze GNN větev (PDB data)
      - 'sequence':  pouze Seq větev (sekvence bez struktury)
      - 'both':      obě větve + consistency loss (PDB data)
    
    Inference:
      - Pokud máte strukturu → 'both' nebo 'structure'
      - Pokud máte jen sekvenci → 'sequence'
    """
    
    def __init__(self, 
                 # Sequence branch params
                 esm_dim=1280,
                 # GNN branch params
                 node_dim=1310,
                 ligand_dim=None,  # None → LigandFeatures.LIGAND_FEAT_DIM (36)
                 # Shared params
                 hidden_dim=256,
                 num_gnn_layers=3,
                 num_attention_heads=4,
                 dropout=0.5,
                 use_gat=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # ---- GNN Branch ----
        self.gnn_branch = GNNBranch(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            use_gat=use_gat,
            ligand_dim=ligand_dim
        )
        # GNN branch outputs 2*hidden_dim, project to hidden_dim
        self.gnn_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ---- Sequence Branch ----
        self.seq_branch = SequenceBranch(
            esm_dim=esm_dim,
            hidden_dim=hidden_dim,
            num_cnn_layers=3,
            num_attention_heads=num_attention_heads,
            dropout=dropout
        )
        
        # ---- Shared Classifier ----
        # Sdílený MLP – obě větve produkují [batch, hidden_dim],
        # classifier je stejný pro obě → učí se společný feature space
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # binds / doesn't bind
        )
        
        # ---- Fusion layer (pro 'both' režim) ----
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, mode='sequence', graph_data=None, 
                esm_embeddings=None, seq_mask=None):
        """
        Args:
            mode: 'structure' | 'sequence' | 'both'
            graph_data: PyG Batch (pro 'structure' a 'both')
            esm_embeddings: [B, L, esm_dim] (pro 'sequence' a 'both')
            seq_mask: [B, L] padding mask
        
        Returns:
            logits: [batch_size, 2]
            embeddings: dict s embedding vektory pro consistency loss
        """
        embeddings = {}
        
        if mode == 'structure':
            gnn_out = self.gnn_branch(graph_data)       # [B, 2H]
            gnn_emb = self.gnn_proj(gnn_out)             # [B, H]
            embeddings['structure'] = gnn_emb
            logits = self.classifier(gnn_emb)
            
        elif mode == 'sequence':
            seq_emb = self.seq_branch(esm_embeddings, seq_mask)  # [B, H]
            embeddings['sequence'] = seq_emb
            logits = self.classifier(seq_emb)
            
        elif mode == 'both':
            # Obě větve
            gnn_out = self.gnn_branch(graph_data)
            gnn_emb = self.gnn_proj(gnn_out)             # [B, H]
            seq_emb = self.seq_branch(esm_embeddings, seq_mask)  # [B, H]
            
            embeddings['structure'] = gnn_emb
            embeddings['sequence'] = seq_emb
            
            # Fúze obou větví
            fused = self.fusion(torch.cat([gnn_emb, seq_emb], dim=1))
            logits = self.classifier(fused)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return logits, embeddings
    
    def get_consistency_loss(self, embeddings, temperature=0.5):
        """
        Consistency loss: penalizuje pokud GNN a Seq větev
        dávají různé embeddingy pro STEJNÝ protein.
        
        Pomáhá Seq větvi naučit se aproximovat strukturní informaci.
        """
        if 'structure' not in embeddings or 'sequence' not in embeddings:
            return torch.tensor(0.0)
        
        struct_emb = F.normalize(embeddings['structure'], dim=1)
        seq_emb = F.normalize(embeddings['sequence'], dim=1)
        
        # Cosine similarity loss
        cos_sim = (struct_emb * seq_emb).sum(dim=1)  # [B]
        consistency_loss = (1 - cos_sim).mean()
        
        return consistency_loss


# ============================================================
# Inicializace
# ============================================================
if __name__ == '__main__':
    model = DualBranchPredictor(
        esm_dim=1280,
        node_dim=1310,
        ligand_dim=36,  # LigandFeatures.LIGAND_FEAT_DIM
        hidden_dim=256,
        num_gnn_layers=3,
        num_attention_heads=4,
        dropout=0.5,
        use_gat=True
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    gnn_params = sum(p.numel() for p in model.gnn_branch.parameters())
    seq_params = sum(p.numel() for p in model.seq_branch.parameters())
    shared_params = sum(p.numel() for p in model.classifier.parameters())
    
    print(f"Total parameters:     {total_params:,}")
    print(f"  GNN branch:         {gnn_params:,}")
    print(f"  Sequence branch:    {seq_params:,}")
    print(f"  Shared classifier:  {shared_params:,}")
