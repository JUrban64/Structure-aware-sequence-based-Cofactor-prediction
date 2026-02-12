import numpy as np

# ============================================================
# Konfigurace pro ligandové features
# ============================================================

# Prvky běžné v kofaktorech (pro one-hot encoding)
LIGAND_ELEMENTS = ['C', 'N', 'O', 'P', 'S']  # 5D one-hot

# Funkční skupiny kofaktorů (pro one-hot encoding)
FUNCTIONAL_GROUPS = [
    'adenine', 'nicotinamide', 'isoalloxazine',
    'ribose', 'ribose_A', 'ribose_N', 'ribitol',
    'phosphate', 'alpha_P', 'beta_P', 'gamma_P',
    'pantothenate', 'cysteamine',
    'unknown'
]  # 14D one-hot

# Známé kofaktory (pro cofactor_id one-hot)
KNOWN_COFACTORS = ['NAD', 'NADP', 'FAD', 'FMN', 'ATP', 'ADP',
                   'AMP', 'GTP', 'GDP', 'COA', 'SAM', 'THF',
                   'PLP', 'TPP', 'HEM']  # 15 kofaktorů

COFACTOR_TO_IDX = {c: i for i, c in enumerate(KNOWN_COFACTORS)}


class LigandFeatures:
    """
    Feature extraction pro ligandové atomy/uzly v grafu.
    
    Každý ligandový atom dostane feature vektor:
        - element one-hot  [5]  (C, N, O, P, S)
        - functional_group one-hot [14] (adenine, ribose, phosphate, ...)
        - is_aromatic [1]  (odhad z funkční skupiny)
        - n_bonds [1]  (počet vazeb, normalizováno)
        - cofactor_id one-hot [15]  (NAD, FAD, ATP, ...)
        ─────────────────────────────────────────
        Total: 36D na ligandový atom
    
    Dim je záměrně nízký — ligandové uzly se poté projektují
    na hidden_dim stejně jako proteinové.
    """
    
    LIGAND_FEAT_DIM = 36  # 5 + 14 + 1 + 1 + 15
    
    # Aromatické funkční skupiny
    AROMATIC_FGS = {'adenine', 'nicotinamide', 'isoalloxazine'}
    
    def __init__(self):
        self.element_to_idx = {e: i for i, e in enumerate(LIGAND_ELEMENTS)}
        self.fg_to_idx = {fg: i for i, fg in enumerate(FUNCTIONAL_GROUPS)}
    
    def get_atom_features(self, ligand_atoms, ligand_bonds, cofactor_name):
        """
        Vytvoří feature matici pro všechny atomy ligandu.
        
        Args:
            ligand_atoms: list of dicts from BindingSiteExtractor._extract_ligand_atoms()
            ligand_bonds: list of (i, j, dist) from _compute_ligand_bonds()
            cofactor_name: str, e.g. 'NAD', 'FAD'
        
        Returns:
            features: np.array [n_lig_atoms, 36]
        """
        if len(ligand_atoms) == 0:
            return np.zeros((0, self.LIGAND_FEAT_DIM))
        
        # Spočítej počet vazeb pro každý atom
        bond_counts = np.zeros(len(ligand_atoms))
        for i, j, _ in ligand_bonds:
            bond_counts[i] += 1
            bond_counts[j] += 1
        max_bonds = max(bond_counts.max(), 1)
        
        # Cofactor one-hot (sdílený pro všechny atomy)
        cof_onehot = np.zeros(len(KNOWN_COFACTORS))
        if cofactor_name in COFACTOR_TO_IDX:
            cof_onehot[COFACTOR_TO_IDX[cofactor_name]] = 1.0
        
        features = []
        for idx, atom in enumerate(ligand_atoms):
            feat = []
            
            # 1) Element one-hot [5]
            elem_oh = np.zeros(len(LIGAND_ELEMENTS))
            eidx = self.element_to_idx.get(atom['element'], -1)
            if eidx >= 0:
                elem_oh[eidx] = 1.0
            feat.append(elem_oh)
            
            # 2) Functional group one-hot [14]
            fg_oh = np.zeros(len(FUNCTIONAL_GROUPS))
            fg_name = atom.get('functional_group', 'unknown')
            fidx = self.fg_to_idx.get(fg_name, self.fg_to_idx['unknown'])
            fg_oh[fidx] = 1.0
            feat.append(fg_oh)
            
            # 3) Is aromatic [1]
            is_arom = np.array([1.0 if fg_name in self.AROMATIC_FGS else 0.0])
            feat.append(is_arom)
            
            # 4) Normalized bond count [1]
            n_bonds = np.array([bond_counts[idx] / max_bonds])
            feat.append(n_bonds)
            
            # 5) Cofactor ID one-hot [15]
            feat.append(cof_onehot)
            
            features.append(np.concatenate(feat))
        
        return np.array(features, dtype=np.float32)
    
    def get_cofactor_global_embedding(self, cofactor_name):
        """
        Vrátí globální vektor pro daný kofaktor (pro conditioning).
        Použitelné v budoucnu pro multi-cofactor predikci.
        
        Returns:
            embedding: np.array [15]
        """
        emb = np.zeros(len(KNOWN_COFACTORS))
        if cofactor_name in COFACTOR_TO_IDX:
            emb[COFACTOR_TO_IDX[cofactor_name]] = 1.0
        return emb

class AdditionalFeatures:
    """
    Extract BLOSUM, physicochemical, and positional features
    """
    
    def __init__(self):
        # BLOSUM62 matrix
        self.blosum62 = self._load_blosum62()
        
        # Physicochemical properties (normalized to 0-1)
        # 7 descriptors per amino acid:
        #   [0] Hydrophobicity    (Kyte-Doolittle scale, normalized)
        #   [1] Volume            (normalized van der Waals volume)
        #   [2] Polarity          (Grantham polarity, normalized)
        #   [3] Polarizability    (Charton, normalized)
        #   [4] Isoelectric point (pI, normalized)
        #   [5] Helix propensity  (Chou-Fasman, normalized)
        #   [6] Sheet propensity  (Chou-Fasman, normalized)
        # References:
        #   Kyte & Doolittle (1982) J Mol Biol 157:105-132
        #   Grantham (1974) Science 185:862-864
        #   Chou & Fasman (1978) Adv Enzymol 47:45-148
        #   Meiler et al. (2001) J Mol Model 7:360-369
        self.physicochemical = {
            'A': [0.700, 0.167, 0.395, 0.366, 0.462, 0.697, 0.413],
            'R': [0.000, 0.596, 0.691, 0.592, 0.840, 0.528, 0.572],
            'N': [0.111, 0.309, 0.827, 0.463, 0.437, 0.472, 0.307],
            'D': [0.111, 0.284, 0.901, 0.366, 0.227, 0.528, 0.244],
            'C': [0.783, 0.228, 0.457, 0.341, 0.424, 0.389, 0.572],
            'Q': [0.111, 0.382, 0.691, 0.512, 0.445, 0.556, 0.492],
            'E': [0.111, 0.358, 0.827, 0.463, 0.255, 0.611, 0.307],
            'G': [0.457, 0.000, 0.457, 0.000, 0.462, 0.299, 0.382],
            'H': [0.144, 0.400, 0.691, 0.512, 0.575, 0.528, 0.413],
            'I': [1.000, 0.384, 0.271, 0.585, 0.462, 0.528, 0.730],
            'L': [0.922, 0.384, 0.271, 0.585, 0.462, 0.639, 0.492],
            'K': [0.067, 0.457, 0.691, 0.561, 0.726, 0.556, 0.382],
            'M': [0.711, 0.417, 0.346, 0.585, 0.447, 0.611, 0.492],
            'F': [0.811, 0.558, 0.271, 0.707, 0.440, 0.556, 0.651],
            'P': [0.322, 0.239, 0.457, 0.366, 0.492, 0.299, 0.190],
            'S': [0.411, 0.136, 0.531, 0.268, 0.445, 0.389, 0.382],
            'T': [0.422, 0.253, 0.457, 0.341, 0.440, 0.417, 0.572],
            'W': [0.400, 0.733, 0.271, 0.878, 0.456, 0.528, 0.651],
            'Y': [0.356, 0.596, 0.383, 0.707, 0.440, 0.361, 0.651],
            'V': [0.967, 0.309, 0.271, 0.488, 0.462, 0.472, 0.730],
        }
    
    def get_blosum_features(self, sequence):
        """
        Args:
            sequence: AA sequence
        Returns:
            blosum: [L, 20]
        """
        features = []
        for aa in sequence:
            if aa in self.blosum62:
                features.append(self.blosum62[aa])
            else:
                features.append([0] * 20)  # Unknown AA
        return np.array(features)
    
    def get_physicochemical_features(self, sequence):
        """
        Returns:
            physchem: [L, 7]
        """
        features = []
        for aa in sequence:
            if aa in self.physicochemical:
                features.append(self.physicochemical[aa])
            else:
                features.append([0.5] * 7)
        return np.array(features)
    
    def get_relative_position_features(self, bs_indices, seq_length):
        """
        Relative position in sequence
        
        Returns:
            pos_features: [n_bs, 3]
                - normalized position in sequence
                - distance from N-terminus
                - distance from C-terminus
        """
        features = []
        for idx in bs_indices:
            norm_pos = idx / seq_length
            dist_n = idx / seq_length
            dist_c = (seq_length - idx) / seq_length
            features.append([norm_pos, dist_n, dist_c])
        return np.array(features)
    
    def _load_blosum62(self):
        """
        Load BLOSUM62 matrix as per-AA row vectors.
        Returns dict: AA → [20] scores against all 20 standard AAs.
        Order: ARNDCQEGHILKMFPSTWYV
        """
        aa_list = 'ARNDCQEGHILKMFPSTWYV'
        # Full symmetric BLOSUM62 matrix (rows/cols in aa_list order)
        # Reference: Henikoff & Henikoff (1992) PNAS 89:10915-10919
        matrix = [
            # A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
            [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],  # A
            [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
            [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],  # N
            [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],  # D
            [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
            [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],  # Q
            [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],  # E
            [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],  # G
            [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],  # H
            [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],  # I
            [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],  # L
            [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],  # K
            [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],  # M
            [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],  # F
            [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],  # P
            [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],  # S
            [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],  # T
            [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],  # W
            [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],  # Y
            [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4],  # V
        ]
        
        blosum62 = {}
        for i, aa in enumerate(aa_list):
            blosum62[aa] = matrix[i]
        
        return blosum62


# Combine all features
def create_node_features(bs_info, use_esm=True, use_blosum=True, 
                         use_physchem=True, use_position=True):
    """
    Combine multiple feature types FOR PROTEIN NODES ONLY.
    
    Returns:
        node_features: [n_bs, total_dim]  (protein_dim = 1310 default)
    """
    features = []
    
    # ESM embeddings (1280D)
    if use_esm:
        features.append(bs_info['esm_embeddings'])
    
    # BLOSUM (20D)
    if use_blosum:
        additional = AdditionalFeatures()
        blosum = additional.get_blosum_features(
            bs_info['binding_site_sequence']
        )
        features.append(blosum)
    
    # Physicochemical (7D)
    if use_physchem:
        physchem = additional.get_physicochemical_features(
            bs_info['binding_site_sequence']
        )
        features.append(physchem)
    
    # Positional (3D)
    if use_position:
        pos = additional.get_relative_position_features(
            bs_info['binding_site_indices'],
            len(bs_info['full_sequence'])
        )
        features.append(pos)
    
    # Concatenate
    node_features = np.concatenate(features, axis=1)
    
    return node_features


def create_ligand_node_features(bs_info):
    """
    Feature extraction pro ligandové uzly.
    
    Args:
        bs_info: dict z BindingSiteExtractor (musí obsahovat
                 'ligand_atoms', 'ligand_bonds', 'ligand_name')
    
    Returns:
        ligand_features: np.array [n_lig_atoms, 36]
            (nebo [0, 36] pokud nejsou žádné ligandové atomy)
    """
    lig_feat = LigandFeatures()
    
    ligand_atoms = bs_info.get('ligand_atoms', [])
    ligand_bonds = bs_info.get('ligand_bonds', [])
    cofactor_name = bs_info.get('ligand_name', 'UNK')
    
    return lig_feat.get_atom_features(ligand_atoms, ligand_bonds, cofactor_name)