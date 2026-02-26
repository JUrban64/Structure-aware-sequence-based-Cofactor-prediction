from torch_geometric.data import Data
import torch
import numpy as np
from additional_features import (
    create_node_features, LigandFeatures
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

# Pevná dimenze edge features: [distance_norm, hbond, hydrophobic, ionic, other]
EDGE_ATTR_DIM = 5


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
        # Labely a sekvence se drží zvlášť pro split bez stavby grafů
        self.labels = [bs.get('label', 1) for bs in self.data]
        self.sequences = [bs.get('binding_site_sequence', '') for bs in self.data]
        self.pdb_ids = [bs.get('pdb_file', '') for bs in self.data]
    
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
            lig_feat_extractor = LigandFeatures()
            ligand_features = lig_feat_extractor.get_atom_features(
                ligand_atoms,
                bs_info.get('ligand_bonds', []),
                bs_info.get('ligand_name', 'UNK')
            )
            n_lig = ligand_features.shape[0]
            ligand_dim = ligand_features.shape[1]
        else:
            n_lig = 0
            ligand_dim = LigandFeatures.LIGAND_FEAT_DIM
        
        n_total = n_prot + n_lig
        
        max_dim = max(protein_dim, ligand_dim)
        
        if protein_dim < max_dim:
            prot_pad = np.zeros((n_prot, max_dim - protein_dim))
            protein_padded = np.concatenate([protein_features, prot_pad], axis=1)
        else:
            protein_padded = protein_features
        
        if has_ligand:
            lig_pad = np.zeros((n_lig, max_dim - ligand_dim))
            ligand_padded = np.concatenate([ligand_features, lig_pad], axis=1)
            all_features = np.concatenate([protein_padded, ligand_padded], axis=0)
        else:
            all_features = protein_padded
        
        x = torch.FloatTensor(all_features)
        
        # ---- Node type ----
        node_type = torch.zeros(n_total, dtype=torch.long)
        if has_ligand:
            node_type[n_prot:] = NODE_TYPE_LIGAND
        
        # ---- EDGES ----
        # Všechny edge_attr mají PEVNOU dimenzi EDGE_ATTR_DIM = 5:
        #   [distance_norm, hbond, hydrophobic, ionic, other]
        all_edges = []      # list of [src, dst]
        all_edge_types = [] # list of int
        all_edge_attr = []  # list of [EDGE_ATTR_DIM] float vectors
        
        # 1) P-P edges: z kontaktní mapy (vektorizovaně přes np.where)
        contact_map = bs_info['contact_map']
        pp_rows, pp_cols = np.where(contact_map > 0.5)
        if len(pp_rows) > 0:
            pp_weights = contact_map[pp_rows, pp_cols]
            pp_edges = np.stack([pp_rows, pp_cols], axis=1)          # [E_pp, 2]
            pp_attr = np.zeros((len(pp_rows), EDGE_ATTR_DIM))       # [E_pp, 5]
            pp_attr[:, 0] = pp_weights
            all_edges.extend(pp_edges.tolist())
            all_edge_types.extend([EDGE_TYPE_PP] * len(pp_rows))
            all_edge_attr.extend(pp_attr.tolist())
        
        # 2) P-L edges: protein-ligand interakce
        if has_ligand:
            pl_contacts = bs_info.get('protein_ligand_contacts', [])
            if pl_contacts:
                for contact in pl_contacts:
                    prot_idx = contact['protein_idx']
                    lig_idx = contact['ligand_idx'] + n_prot  # offset!
                    
                    # Edge feature: distance (normalized) + interaction type
                    dist_norm = contact['distance'] / 4.5
                    itype_oh = [0.0] * len(INTERACTION_TYPES)
                    itype_idx = ITYPE_TO_IDX.get(
                        contact['interaction_type'], 
                        ITYPE_TO_IDX['other']
                    )
                    itype_oh[itype_idx] = 1.0
                    
                    edge_feat = [dist_norm] + itype_oh  # [5]
                    
                    # Bidirectional
                    all_edges.append([prot_idx, lig_idx])
                    all_edge_types.append(EDGE_TYPE_PL)
                    all_edge_attr.append(edge_feat)
                    
                    all_edges.append([lig_idx, prot_idx])
                    all_edge_types.append(EDGE_TYPE_PL)
                    all_edge_attr.append(edge_feat)
        
        # 3) L-L edges: kovalentní vazby uvnitř ligandu
        if has_ligand:
            lig_bonds = bs_info.get('ligand_bonds', [])
            if lig_bonds:
                for i, j, dist in lig_bonds:
                    src = i + n_prot
                    dst = j + n_prot
                    # L-L edge attr: [bond_length_norm, 0, 0, 0, 0]
                    bond_feat = [dist / 1.9, 0.0, 0.0, 0.0, 0.0]
                    
                    all_edges.append([src, dst])
                    all_edge_types.append(EDGE_TYPE_LL)
                    all_edge_attr.append(bond_feat)
                    
                    all_edges.append([dst, src])
                    all_edge_types.append(EDGE_TYPE_LL)
                    all_edge_attr.append(bond_feat)
        
        # ---- Sloučení hran ----
        if all_edges:
            edge_index = torch.LongTensor(all_edges).t().contiguous()
            edge_type = torch.LongTensor(all_edge_types)
            edge_attr = torch.FloatTensor(all_edge_attr)  # [E, 5] – vždy stejná dim
        else:
            # Fallback: fully connected protein-only
            edge_list = [[i, j] for i in range(n_prot) for j in range(n_prot)]
            edge_index = torch.LongTensor(edge_list).t().contiguous()
            edge_type = torch.zeros(len(edge_list), dtype=torch.long)
            edge_attr = torch.zeros(len(edge_list), EDGE_ATTR_DIM)
        
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
            full_sequence=bs_info.get('full_sequence', bs_info['binding_site_sequence']),
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
        Vektorizovaná verze pomocí np.where.
        
        Returns:
            edge_index: [2, num_edges]
            edge_attr: [num_edges, 1] (contact probability)
        """
        rows, cols = np.where(contact_map > threshold)
        
        if len(rows) == 0:
            # Fallback: fully connected
            n = contact_map.shape[0]
            rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
            rows, cols = rows.ravel(), cols.ravel()
            edge_weights = np.ones(len(rows), dtype=np.float32)
        else:
            edge_weights = contact_map[rows, cols].astype(np.float32)
        
        edge_index = torch.LongTensor(np.stack([rows, cols])).contiguous()
        edge_attr = torch.FloatTensor(edge_weights).unsqueeze(1)
        
        return edge_index, edge_attr
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Lazy: sestaví graf až když ho DataLoader potřebuje."""
        graph = self._build_single_graph(self.data[idx])
        graph.y = torch.LongTensor([self.labels[idx]])
        return graph


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