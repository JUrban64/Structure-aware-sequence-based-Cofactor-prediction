import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool

class BindingSiteNADPredictor(nn.Module):
    """
    GNN model for NAD binding prediction from binding site graphs
    
    Features:
    - Works on small graphs (15-30 nodes)
    - Uses ESM embeddings + additional features
    - Multiple aggregation heads
    """
    
    def __init__(self, 
                 node_dim=1310,  # 1280 (ESM) + 20 (BLOSUM) + 7 (physchem) + 3 (pos)
                 hidden_dim=256,
                 num_gnn_layers=3,
                 num_attention_heads=4,
                 dropout=0.5,
                 use_gat=False):  # Use GAT instead of GCN
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.use_gat = use_gat
        
        # ============================================
        # INPUT PROJECTION
        # ============================================
        # Project high-dim ESM features to hidden_dim
        self.input_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
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
            data: PyG Batch object
                - data.x: [total_nodes_in_batch, node_dim]
                - data.edge_index: [2, total_edges]
                - data.edge_attr: [total_edges, 1]
                - data.batch: [total_nodes] batch assignment
        
        Returns:
            logits: [batch_size, 2]
        """
        x, edge_index, edge_attr, batch = \
            data.x, data.edge_index, data.edge_attr, data.batch
        
        # ============================================
        # INPUT PROJECTION
        # ============================================
        x = self.input_projection(x)  # [total_nodes, hidden_dim]
        
        # ============================================
        # GNN PROPAGATION
        # ============================================
        for i, (gnn, norm) in enumerate(zip(self.gnn_layers, self.norms)):
            # GNN layer
            if self.use_gat:
                x_new = gnn(x, edge_index)
            else:
                x_new = gnn(x, edge_index, edge_attr)
            
            # Residual connection
            if i > 0:
                x_new = x_new + x
            
            # Normalize and activate
            x = norm(x_new)
            x = F.relu(x)
            x = self.dropout(x)
        
        # M: final node embeddings [total_nodes, hidden_dim]
        M = x
        
        # ============================================
        # SELF-ATTENTION POOLING
        # ============================================
        # Compute attention scores
        attention_input = self.W1(M)  # [total_nodes, attention_dim]
        attention_scores = self.W2(torch.tanh(attention_input))  # [total_nodes, num_heads]
        
        # Softmax per graph in batch
        attention_weights = self._batch_softmax(attention_scores, batch)
        
        # Weighted pooling for each head
        pooled_per_head = []
        for head in range(self.num_heads):
            weights = attention_weights[:, head].unsqueeze(1)  # [total_nodes, 1]
            weighted_features = M * weights  # [total_nodes, hidden_dim]
            
            # Sum per graph
            pooled = global_add_pool(weighted_features, batch)  # [batch_size, hidden_dim]
            pooled_per_head.append(pooled)
        
        # Average across heads
        attention_pooled = torch.stack(pooled_per_head).mean(dim=0)  # [batch_size, hidden_dim]
        
        # ============================================
        # GLOBAL POOLING (optional but helpful)
        # ============================================
        if self.use_global_pool:
            global_pooled = global_mean_pool(M, batch)  # [batch_size, hidden_dim]
            
            # Concatenate
            graph_embedding = torch.cat([
                attention_pooled, 
                global_pooled
            ], dim=1)  # [batch_size, 2*hidden_dim]
        else:
            graph_embedding = attention_pooled
        
        # ============================================
        # CLASSIFICATION
        # ============================================
        logits = self.classifier(graph_embedding)  # [batch_size, 2]
        
        return logits
    
    def _batch_softmax(self, scores, batch):
        """
        Apply softmax per graph in batch
        
        Args:
            scores: [total_nodes, num_heads]
            batch: [total_nodes] batch assignment
        
        Returns:
            weights: [total_nodes, num_heads] normalized per graph
        """
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


# Initialize model
model = BindingSiteNADPredictor(
    node_dim=1310,  # ESM(1280) + BLOSUM(20) + Physchem(7) + Pos(3)
    hidden_dim=256,
    num_gnn_layers=3,
    num_attention_heads=4,
    dropout=0.5,
    use_gat=True  # GAT je často lepší pro malé grafy
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")