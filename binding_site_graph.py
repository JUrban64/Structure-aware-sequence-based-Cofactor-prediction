from torch_geometric.data import Data
import torch
import numpy as np
from additional_features import (
    create_node_features, create_ligand_node_features, LigandFeatures
)


# Typy hran
EDGE_TYPE_PP = 0   # Protein–Protein  (kontaktní mapa)
EDGE_TYPE_PL = 1   # Protein–Ligand   (interakční hrany)
EDGE_TYPE_LL = 2   # Ligand–Ligand    (kovalentní vazby)

# Typy uzlů
NODE_TYPE_PROTEIN = 0
NODE_TYPE_LIGAND = 1

# Typy interakcí (pro edge feature encoding)
INTERACTION_TYPES = ['hbond_candidate', 'hydrophobic', 'ionic', 'other']
ITYPE_TO_IDX = {t: i for i, t in enumerate(INTERACTION_TYPES)}


class BindingSiteGraphDataset:
    """
    Dataset of protein-ligand interaction graphs.
    
    Každý graf obsahuje:
      - Protein uzly (residues v binding site) s proteinovými features
      - Ligand uzly (atomy kofaktoru) s ligandovými features
      - Tři typy hran:
          P-P: kontaktní mapa proteinových residues
          P-L: protein-ligand interakce (distance-based)
          L-L: kovalentní vazby uvnitř ligandu
      - node_type: [N] tensor (0=protein, 1=ligand)
      - edge_type: [E] tensor (0=PP, 1=PL, 2=LL)
      - cofactor_id: str ('NAD', 'FAD', ...)
    """
    
    def __init__(self, binding_sites_data, feature_config=None,
                 include_ligand=True):
        """
        Args:
            binding_sites_data: list of binding site info dicts
            feature_config: dict specifying which protein features to use
            include_ligand: bool, zda přidat ligandové uzly a P-L/L-L hrany
        """
        self.data = binding_sites_data
        self.include_ligand = include_ligand
        
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
        Build PyG graph for single binding site.
        
        Protein-ligand interaction graph:
            Nodes:  [0 .. n_prot-1] = protein residues
                    [n_prot .. n_prot+n_lig-1] = ligand atoms
            Edges:  P-P (contact map), P-L (interactions), L-L (bonds)
        
        Returns:
            PyG Data object with extra attributes:
                - node_type: [N] int tensor
                - edge_type: [E] int tensor
                - edge_interaction: [E, 4] float tensor (interaction type one-hot, only for P-L edges)
                - n_protein_nodes: int
                - n_ligand_nodes: int
                - cofactor_id: str
        """
        # ---- Protein node features ----
        protein_features = create_node_features(
            bs_info, **self.feature_config
        )
        n_prot = protein_features.shape[0]
        protein_dim = protein_features.shape[1]
        
        # ---- Ligand node features ----
        ligand_atoms = bs_info.get('ligand_atoms', [])
        has_ligand = self.include_ligand and len(ligand_atoms) > 0
        
        if has_ligand:
            ligand_features = create_ligand_node_features(bs_info)
            n_lig = ligand_features.shape[0]
            ligand_dim = ligand_features.shape[1]  # 36
        else:
            n_lig = 0
            ligand_dim = LigandFeatures.LIGAND_FEAT_DIM
        
        n_total = n_prot + n_lig
        
        # ---- Sloučení uzlů: pad na společnou dimenzi ----
        # Protein a ligand mají RŮZNÉ dimenze features →
        # uložíme je zvlášť a model je projektuje na hidden_dim
        # přes separate input projections.
        # 
        # V PyG Data uložíme:
        #   x_protein: [n_prot, protein_dim]  (1310D default)
        #   x_ligand:  [n_lig, ligand_dim]    (36D)
        # A pro GNN potřebujeme společný x → padujeme na max_dim.
        
        max_dim = max(protein_dim, ligand_dim)
        
        # Pad protein features (pokud by ligand byl větší – nepravděpodobné)
        if protein_dim < max_dim:
            prot_pad = np.zeros((n_prot, max_dim - protein_dim))
            protein_padded = np.concatenate([protein_features, prot_pad], axis=1)
        else:
            protein_padded = protein_features
        
        if has_ligand:
            # Pad ligand features (ligand_dim=36 << protein_dim=1310)
            lig_pad = np.zeros((n_lig, max_dim - ligand_dim))
            ligand_padded = np.concatenate([ligand_features, lig_pad], axis=1)
            
            # Sloučit do jednoho feature tensoru
            all_features = np.concatenate([protein_padded, ligand_padded], axis=0)
        else:
            all_features = protein_padded
        
        x = torch.FloatTensor(all_features)
        
        # ---- Node type ----
        node_type = torch.zeros(n_total, dtype=torch.long)
        if has_ligand:
            node_type[n_prot:] = NODE_TYPE_LIGAND
        
        # ---- EDGES ----
        all_edges = []      # list of [src, dst]
        all_edge_types = [] # list of int
        all_edge_attr = []  # list of [weight] or [interaction features]
        
        # 1) P-P edges: z kontaktní mapy
        contact_map = bs_info['contact_map']
        pp_edges, pp_attr = self._contact_map_to_edges(contact_map)
        if pp_edges.numel() > 0:
            n_pp = pp_edges.shape[1]
            all_edges.append(pp_edges)
            all_edge_types.extend([EDGE_TYPE_PP] * n_pp)
            all_edge_attr.append(pp_attr)
        
        # 2) P-L edges: protein-ligand interakce
        if has_ligand:
            pl_contacts = bs_info.get('protein_ligand_contacts', [])
            if pl_contacts:
                pl_src, pl_dst, pl_feat = [], [], []
                for contact in pl_contacts:
                    prot_idx = contact['protein_idx']
                    lig_idx = contact['ligand_idx'] + n_prot  # offset!
                    
                    # Bidirectional
                    pl_src.extend([prot_idx, lig_idx])
                    pl_dst.extend([lig_idx, prot_idx])
                    
                    # Edge feature: distance (normalized) + interaction type
                    dist_norm = contact['distance'] / 4.5  # normalized
                    itype_oh = [0.0] * len(INTERACTION_TYPES)
                    itype_idx = ITYPE_TO_IDX.get(
                        contact['interaction_type'], 
                        ITYPE_TO_IDX['other']
                    )
                    itype_oh[itype_idx] = 1.0
                    
                    edge_feat = [dist_norm] + itype_oh  # [5]
                    pl_feat.extend([edge_feat, edge_feat])  # bidi
                
                pl_edges = torch.LongTensor([pl_src, pl_dst])
                all_edges.append(pl_edges)
                all_edge_types.extend([EDGE_TYPE_PL] * len(pl_src))
                
                # Pad P-L edge attr na stejný dim jako P-P (1D → 5D)
                pl_attr = torch.FloatTensor(pl_feat)
                all_edge_attr.append(pl_attr)
        
        # 3) L-L edges: kovalentní vazby uvnitř ligandu
        if has_ligand:
            lig_bonds = bs_info.get('ligand_bonds', [])
            if lig_bonds:
                ll_src, ll_dst, ll_feat = [], [], []
                for i, j, dist in lig_bonds:
                    src = i + n_prot  # offset
                    dst = j + n_prot
                    ll_src.extend([src, dst])
                    ll_dst.extend([dst, src])
                    
                    bond_feat = [dist / 1.9]  # normalized bond length
                    ll_feat.extend([bond_feat, bond_feat])
                
                ll_edges = torch.LongTensor([ll_src, ll_dst])
                all_edges.append(ll_edges)
                all_edge_types.extend([EDGE_TYPE_LL] * len(ll_src))
                ll_attr = torch.FloatTensor(ll_feat)
                all_edge_attr.append(ll_attr)
        
        # ---- Sloučení hran ----
        if all_edges:
            edge_index = torch.cat(all_edges, dim=1)
            edge_type = torch.LongTensor(all_edge_types)
            
            # Sjednoť edge_attr na společnou dimenzi (padování)
            max_edge_dim = max(ea.shape[1] for ea in all_edge_attr)
            padded_attrs = []
            for ea in all_edge_attr:
                if ea.shape[1] < max_edge_dim:
                    pad = torch.zeros(ea.shape[0], max_edge_dim - ea.shape[1])
                    ea = torch.cat([ea, pad], dim=1)
                padded_attrs.append(ea)
            edge_attr = torch.cat(padded_attrs, dim=0)
        else:
            # Fallback: fully connected protein-only
            edge_list = [[i, j] for i in range(n_prot) for j in range(n_prot)]
            edge_index = torch.LongTensor(edge_list).t().contiguous()
            edge_type = torch.zeros(len(edge_list), dtype=torch.long)
            edge_attr = torch.ones(len(edge_list), 1)
        
        # ---- Label ----
        y = torch.LongTensor([1])  # Default; run_pipeline přepíše
        
        # ---- Sestavení PyG Data ----
        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_type=edge_type,
            node_type=node_type,
            y=y,
            # Metadata
            sequence=bs_info['binding_site_sequence'],
            pdb_id=bs_info['pdb_file'],
            n_residues=bs_info['n_binding_site'],
            n_protein_nodes=n_prot,
            n_ligand_nodes=n_lig,
            protein_dim=protein_dim,
            ligand_dim=ligand_dim,
            cofactor_id=bs_info.get('ligand_name', 'UNK'),
        )
        
        return graph
    
    def _contact_map_to_edges(self, contact_map, threshold=0.5):
        """
        Convert contact map to edge list (P-P edges only).
        
        Returns:
            edge_index: [2, num_edges]
            edge_attr: [num_edges, 1] (contact probability)
        """
        n = contact_map.shape[0]
        edge_list = []
        edge_weights = []
        
        for i in range(n):
            for j in range(n):
                if contact_map[i, j] > threshold:
                    edge_list.append([i, j])
                    edge_weights.append(contact_map[i, j])
        
        if len(edge_list) == 0:
            # Fallback: fully connected
            edge_list = [
                [i, j] for i in range(n) for j in range(n)
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
if __name__ == '__main__':
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