import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool

from additional_features import LigandFeatures

class BindingSiteNADPredictor(nn.Module):
    """
    GNN model for cofactor binding prediction from protein-ligand 
    interaction graphs.
    
    Features:
    - Heterogenní graf: protein residues + ligand atomy
    - Separate input projection pro protein (1310D) a ligand (36D)
    - Node type embedding (protein=0, ligand=1)
    - Attention pooling POUZE přes protein uzly
    - Multiple aggregation heads
    """
    
    NUM_NODE_TYPES = 2
    
    def __init__(self, 
                 node_dim=1310,  # 1280 (ESM) + 20 (BLOSUM) + 7 (physchem) + 3 (pos)
                 hidden_dim=256,
                 num_gnn_layers=3,
                 num_attention_heads=4,
                 dropout=0.5,
                 use_gat=False,
                 ligand_dim=None):
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.use_gat = use_gat
        
        if ligand_dim is None:
            ligand_dim = LigandFeatures.LIGAND_FEAT_DIM  # 36
        self.ligand_dim = ligand_dim
        
        # ============================================
        # SEPARATE INPUT PROJECTIONS
        # ============================================
        self.protein_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.ligand_projection = nn.Sequential(
            nn.Linear(ligand_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ============================================
        # NODE TYPE EMBEDDING
        # ============================================
        self.node_type_embedding = nn.Embedding(self.NUM_NODE_TYPES, hidden_dim)
        
        # ============================================
        # GNN LAYERS
        # ============================================
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        if use_gat:
            # Graph Attention Network
            # GAT learns which connections are important
            for i in range(num_gnn_layers):
                if i == 0:
                    self.gnn_layers.append(
                        GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
                    )
                else:
                    self.gnn_layers.append(
                        GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
                    )
                self.norms.append(nn.LayerNorm(hidden_dim))
        else:
            # Standard GCN
            for i in range(num_gnn_layers):
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
                self.norms.append(nn.LayerNorm(hidden_dim))
        
        # ============================================
        # MULTI-HEAD ATTENTION POOLING
        # ============================================
        self.attention_dim = 128
        self.num_heads = num_attention_heads
        
        self.W1 = nn.Linear(hidden_dim, self.attention_dim)
        self.W2 = nn.Linear(self.attention_dim, num_attention_heads)
        
        # ============================================
        # GLOBAL POOLING (dodatečné)
        # ============================================
        # Combine attention pooling with standard pooling
        self.use_global_pool = True
        
        # ============================================
        # CLASSIFICATION HEAD
        # ============================================
        # Input: attention pooled (hidden_dim) + global pooled (hidden_dim)
        classifier_input = hidden_dim * 2 if self.use_global_pool else hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # Binary: binds NAD or not
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        """
        Args:
            data: PyG Batch object (heterogenní graf)
                - data.x: [total_nodes, max_dim] padded features
                - data.edge_index: [2, total_edges]
                - data.batch: [total_nodes] batch assignment
                - data.node_type: [total_nodes] (0=protein, 1=ligand)
        
        Returns:
            logits: [batch_size, 2]
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        
        # Node type (default: all protein for backward compat)
        node_type = getattr(data, 'node_type', None)
        if node_type is None:
            node_type = torch.zeros(x.size(0), dtype=torch.long, 
                                    device=x.device)
        
        protein_mask = (node_type == 0)
        ligand_mask = (node_type == 1)
        
        # ============================================
        # SEPARATE INPUT PROJECTION
        # ============================================
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        
        if protein_mask.any():
            prot_x = x[protein_mask, :self.node_dim]
            h[protein_mask] = self.protein_projection(prot_x)
        
        if ligand_mask.any():
            lig_x = x[ligand_mask, :self.ligand_dim]
            h[ligand_mask] = self.ligand_projection(lig_x)
        
        # Add node type embedding
        h = h + self.node_type_embedding(node_type)
        
        # ============================================
        # GNN PROPAGATION
        # ============================================
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
        
        # ============================================
        # ATTENTION POOLING (protein nodes only)
        # ============================================
        attention_input = self.W1(M)
        attention_scores = self.W2(torch.tanh(attention_input))
        
        # Protein-only softmax
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
        
        # ============================================
        # GLOBAL POOLING (protein only)
        # ============================================
        if self.use_global_pool:
            M_protein = M.clone()
            M_protein[ligand_mask] = 0.0
            global_pooled = global_mean_pool(M_protein, batch)
            
            graph_embedding = torch.cat([
                attention_pooled, global_pooled
            ], dim=1)
        else:
            graph_embedding = attention_pooled
        
        # ============================================
        # CLASSIFICATION
        # ============================================
        logits = self.classifier(graph_embedding)
        
        return logits
    
    def _batch_softmax_masked(self, scores, batch, protein_mask):
        """
        Softmax per graph, only over protein nodes.
        Ligand nodes get weight = 0.
        """
        batch_size = batch.max().item() + 1
        num_heads = scores.size(1)
        weights = torch.zeros_like(scores)
        
        for i in range(batch_size):
            graph_mask = (batch == i)
            prot_in_graph = graph_mask & protein_mask
            
            if not prot_in_graph.any():
                for head in range(num_heads):
                    gs = scores[graph_mask, head]
                    weights[graph_mask, head] = F.softmax(gs, dim=0)
                continue
            
            for head in range(num_heads):
                prot_scores = scores[prot_in_graph, head]
                prot_weights = F.softmax(prot_scores, dim=0)
                weights[prot_in_graph, head] = prot_weights
        
        return weights


# Initialize model
if __name__ == '__main__':
    model = BindingSiteNADPredictor(
        node_dim=1310,  # ESM(1280) + BLOSUM(20) + Physchem(7) + Pos(3)
        hidden_dim=256,
        num_gnn_layers=3,
        num_attention_heads=4,
        dropout=0.5,
        use_gat=True,
        ligand_dim=36  # LigandFeatures.LIGAND_FEAT_DIM
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Protein projection: {sum(p.numel() for p in model.protein_projection.parameters()):,}")
    print(f"  Ligand projection:  {sum(p.numel() for p in model.ligand_projection.parameters()):,}")
    print(f"  Node type emb:      {sum(p.numel() for p in model.node_type_embedding.parameters()):,}")