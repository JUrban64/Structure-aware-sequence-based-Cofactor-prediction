from torch_geometric.data import Data
import torch




class BindingSiteGraphDataset:
    """
    Dataset of binding site graphs
    """
    
    def __init__(self, binding_sites_data, feature_config=None):
        """
        Args:
            binding_sites_data: list of binding site info dicts
            feature_config: dict specifying which features to use
        """
        self.data = binding_sites_data
        
        if feature_config is None:
            feature_config = {
                'use_esm': True,
                'use_blosum': True,
                'use_physchem': True,
                'use_position': True
            }
        
        self.feature_config = feature_config
        self.graphs = self._build_graphs()
    
    def _build_graphs(self):
        """Convert all binding sites to PyG graphs"""
        graphs = []
        
        for bs_info in self.data:
            graph = self._build_single_graph(bs_info)
            graphs.append(graph)
        
        return graphs
    
    def _build_single_graph(self, bs_info):
        """
        Build PyG graph for single binding site
        
        Returns:
            PyG Data object
        """
        # Node features
        node_features = create_node_features(
            bs_info, 
            **self.feature_config
        )
        x = torch.FloatTensor(node_features)
        
        # Edge construction from contact map
        contact_map = bs_info['contact_map']
        edge_index, edge_attr = self._contact_map_to_edges(contact_map)
        
        # Label (for training)
        # 1 = binds NAD (or specific cofactor)
        # 0 = doesn't bind
        y = torch.LongTensor([1])  # All PDB structures bind NAD
        
        # Additional metadata
        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            sequence=bs_info['binding_site_sequence'],
            pdb_id=bs_info['pdb_file'],
            n_residues=bs_info['n_binding_site']
        )
        
        return graph
    
    def _contact_map_to_edges(self, contact_map, threshold=0.5):
        """
        Convert contact map to edge list
        
        Args:
            contact_map: [n, n] numpy array
            threshold: minimum value to create edge
        
        Returns:
            edge_index: [2, num_edges]
            edge_attr: [num_edges, 1] (contact probability)
        """
        n = contact_map.shape[0]
        edge_list = []
        edge_weights = []
        
        # Vytvoř edges pro všechny kontakty
        for i in range(n):
            for j in range(n):
                if contact_map[i, j] > threshold:
                    edge_list.append([i, j])
                    edge_weights.append(contact_map[i, j])
        
        if len(edge_list) == 0:
            # Fallback: fully connected
            edge_list = [
                [i, j] 
                for i in range(n) 
                for j in range(n)
            ]
            edge_weights = [1.0] * len(edge_list)
        
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        edge_attr = torch.FloatTensor(edge_weights).unsqueeze(1)
        
        return edge_index, edge_attr
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]


# Create dataset
dataset = BindingSiteGraphDataset(
    binding_sites,
    feature_config={
        'use_esm': True,
        'use_blosum': True,
        'use_physchem': True,
        'use_position': True
    }
)

print(f"Created {len(dataset)} graphs")
print(f"Example graph:")
print(f"  Nodes: {dataset[0].x.shape}")
print(f"  Edges: {dataset[0].edge_index.shape}")
print(f"  Node features dim: {dataset[0].x.shape[1]}")