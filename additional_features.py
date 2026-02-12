import numpy as np

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
        """Load BLOSUM62 matrix"""
        # Simplified version
        aa_list = 'ARNDCQEGHILKMFPSTWYV'
        blosum = {}
        # ... load actual BLOSUM62 values
        return blosum


# Combine all features
def create_node_features(bs_info, use_esm=True, use_blosum=True, 
                         use_physchem=True, use_position=True):
    """
    Combine multiple feature types
    
    Returns:
        node_features: [n_bs, total_dim]
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