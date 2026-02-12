"""
Dual-branch model: Structure (GNN) + Sequence (1D-CNN/Attention)

Umožňuje trénovat na:
  1. PDB datech se strukturou (GNN větev)
  2. Sekvencích bez struktury (Sequence větev) – mnohem více dat
  3. PDB datech oběma větvemi současně (consistency loss)

Architektura:
  ┌──────────────────┐     ┌──────────────────────┐
  │ GNN Branch       │     │ Sequence Branch      │
  │ (binding site    │     │ (full/partial seq    │
  │  graph + GAT)    │     │  ESM + 1D-CNN + Attn)│
  └────────┬─────────┘     └──────────┬───────────┘
           │ [hidden_dim]             │ [hidden_dim]
           └──────────┬───────────────┘
                      ▼
              ┌───────────────┐
              │ Shared MLP    │
              │ Classifier    │
              └───────┬───────┘
                      ▼
                P(binds NAD)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool


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
# GNN BRANCH – pro strukturní data z PDB (stávající logika)
# ============================================================
class GNNBranch(nn.Module):
    """
    Zpracovává binding site graf (uzly = residues, hrany = kontakty).
    Stávající GNN logika z BindingSiteNADPredictor.
    
    Vstup: PyG Batch (x, edge_index, edge_attr, batch)
    Výstup: graph embedding [batch_size, hidden_dim]
    """
    
    def __init__(self, node_dim=1310, hidden_dim=256, 
                 num_gnn_layers=3, num_attention_heads=4,
                 dropout=0.5, use_gat=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_gat = use_gat
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GNN layers
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
        
        # Multi-head attention pooling
        self.attention_dim = 128
        self.num_heads = num_attention_heads
        self.W1 = nn.Linear(hidden_dim, self.attention_dim)
        self.W2 = nn.Linear(self.attention_dim, num_attention_heads)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        """
        Args:
            data: PyG Batch object
        
        Returns:
            graph_embedding: [batch_size, hidden_dim]
        """
        x, edge_index, edge_attr, batch = \
            data.x, data.edge_index, data.edge_attr, data.batch
        
        # Input projection
        x = self.input_projection(x)
        
        # GNN propagation
        for i, (gnn, norm) in enumerate(zip(self.gnn_layers, self.norms)):
            if self.use_gat:
                x_new = gnn(x, edge_index)
            else:
                x_new = gnn(x, edge_index, edge_attr)
            
            if i > 0:
                x_new = x_new + x  # residual
            
            x = norm(x_new)
            x = F.relu(x)
            x = self.dropout(x)
        
        M = x  # final node embeddings
        
        # Self-attention pooling
        attention_input = self.W1(M)
        attention_scores = self.W2(torch.tanh(attention_input))
        attention_weights = self._batch_softmax(attention_scores, batch)
        
        pooled_per_head = []
        for head in range(self.num_heads):
            weights = attention_weights[:, head].unsqueeze(1)
            weighted_features = M * weights
            pooled = global_add_pool(weighted_features, batch)
            pooled_per_head.append(pooled)
        
        attention_pooled = torch.stack(pooled_per_head).mean(dim=0)
        
        # Global pooling
        global_pooled = global_mean_pool(M, batch)
        
        # Concatenate → project back to hidden_dim
        combined = torch.cat([attention_pooled, global_pooled], dim=1)
        
        return combined  # [batch_size, 2 * hidden_dim]
    
    def _batch_softmax(self, scores, batch):
        batch_size = batch.max().item() + 1
        num_heads = scores.size(1)
        weights = torch.zeros_like(scores)
        
        for i in range(batch_size):
            mask = (batch == i)
            for head in range(num_heads):
                graph_scores = scores[mask, head]
                graph_weights = F.softmax(graph_scores, dim=0)
                weights[mask, head] = graph_weights
        
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
            use_gat=use_gat
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
