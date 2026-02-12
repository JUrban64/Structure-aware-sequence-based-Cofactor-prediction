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
        blosum62 = {
    ('W', 'F'): 1, ('L', 'R'): -2, ('S', 'P'): -1, ('V', 'T'): 0,
    ('Q', 'Q'): 5, ('N', 'A'): -2, ('Z', 'Y'): -2, ('W', 'R'): -3,
    ('Q', 'A'): -1, ('S', 'D'): 0, ('H', 'H'): 8, ('S', 'H'): -1,
    ('H', 'D'): -1, ('L', 'N'): -3, ('W', 'A'): -3, ('Y', 'M'): -1,
    ('G', 'R'): -2, ('Y', 'I'): -1, ('Y', 'E'): -2, ('B', 'Y'): -3,
    ('Y', 'A'): -2, ('V', 'D'): -3, ('B', 'S'): 0, ('Y', 'Y'): 7,
    ('G', 'N'): 0, ('E', 'C'): -4, ('Y', 'Q'): -1, ('Z', 'Z'): 4,
    ('V', 'A'): 0, ('C', 'C'): 9, ('M', 'R'): -1, ('V', 'E'): -2,
    ('T', 'N'): 0, ('P', 'P'): 7, ('V', 'I'): 3, ('V', 'S'): -2,
    ('Z', 'P'): -1, ('V', 'M'): 1, ('T', 'F'): -2, ('V', 'Q'): -2,
    ('K', 'K'): 5, ('P', 'D'): -1, ('I', 'H'): -3, ('I', 'D'): -3,
    ('T', 'R'): -1, ('P', 'L'): -3, ('K', 'G'): -2, ('M', 'N'): -2,
    ('P', 'H'): -2, ('F', 'Q'): -3, ('Z', 'G'): -2, ('X', 'L'): -1,
    ('T', 'M'): -1, ('Z', 'C'): -3, ('X', 'H'): -1, ('D', 'R'): -2,
    ('B', 'W'): -4, ('X', 'D'): -1, ('Z', 'K'): 1, ('F', 'A'): -2,
    ('Z', 'W'): -3, ('F', 'E'): -3, ('D', 'N'): 1, ('B', 'K'): 0,
    ('X', 'X'): -1, ('F', 'I'): 0, ('B', 'G'): -1, ('X', 'T'): 0,
    ('F', 'M'): 0, ('B', 'C'): -3, ('Z', 'I'): -3, ('Z', 'V'): -2,
    ('S', 'S'): 4, ('L', 'Q'): -2, ('W', 'E'): -3, ('Q', 'R'): 1,
    ('N', 'N'): 6, ('W', 'M'): -1, ('Q', 'C'): -3, ('W', 'I'): -3,
    ('S', 'C'): -1, ('L', 'A'): -1, ('S', 'G'): 0, ('L', 'E'): -3,
    ('W', 'Q'): -2, ('H', 'G'): -2, ('S', 'K'): 0, ('Q', 'N'): 0,
    ('N', 'R'): 0, ('H', 'C'): -3, ('Y', 'N'): -2, ('G', 'Q'): -2,
    ('Y', 'F'): 3, ('C', 'A'): 0, ('V', 'L'): 1, ('G', 'E'): -2,
    ('G', 'A'): 0, ('K', 'R'): 2, ('E', 'D'): 2, ('Y', 'R'): -2,
    ('M', 'Q'): 0, ('T', 'I'): -1, ('C', 'D'): -3, ('V', 'F'): -1,
    ('T', 'A'): 0, ('T', 'P'): -1, ('B', 'P'): -2, ('T', 'E'): -1,
    ('V', 'N'): -3, ('P', 'G'): -2, ('M', 'A'): -1, ('K', 'H'): -1,
    ('V', 'R'): -3, ('P', 'C'): -3, ('M', 'E'): -2, ('K', 'L'): -2,
    ('V', 'V'): 4, ('M', 'I'): 1, ('T', 'Q'): -1, ('I', 'G'): -4,
    ('P', 'K'): -1, ('M', 'M'): 5, ('K', 'D'): -1, ('I', 'C'): -1,
    ('Z', 'D'): 1, ('F', 'R'): -3, ('X', 'K'): -1, ('Q', 'D'): 0,
    ('X', 'G'): -1, ('Z', 'L'): -3, ('X', 'C'): -2, ('Z', 'H'): 0,
    ('B', 'L'): -4, ('B', 'H'): 0, ('F', 'F'): 6, ('X', 'W'): -2,
    ('B', 'D'): 4, ('D', 'A'): -2, ('S', 'L'): -2, ('X', 'S'): 0,
    ('F', 'N'): -3, ('S', 'R'): -1, ('W', 'D'): -4, ('V', 'Y'): -1,
    ('W', 'L'): -2, ('H', 'R'): 0, ('W', 'H'): -2, ('H', 'N'): 1,
    ('W', 'T'): -2, ('T', 'T'): 5, ('S', 'F'): -2, ('W', 'P'): -4,
    ('L', 'D'): -4, ('B', 'I'): -3, ('L', 'H'): -3, ('S', 'N'): 1,
    ('B', 'T'): -1, ('L', 'L'): 4, ('Y', 'K'): -2, ('E', 'Q'): 2,
    ('Y', 'G'): -3, ('Z', 'S'): 0, ('Y', 'C'): -2, ('G', 'D'): -1,
    ('B', 'V'): -3, ('E', 'A'): -1, ('Y', 'W'): 2, ('E', 'E'): 5,
    ('Y', 'S'): -2, ('C', 'N'): -3, ('V', 'C'): -1, ('T', 'H'): -2,
    ('P', 'R'): -2, ('V', 'G'): -3, ('T', 'L'): -1, ('V', 'K'): -2,
    ('K', 'Q'): 1, ('R', 'A'): -1, ('I', 'R'): -3, ('T', 'D'): -1,
    ('P', 'F'): -4, ('I', 'N'): -3, ('K', 'I'): -3, ('M', 'D'): -3,
    ('V', 'W'): -3, ('W', 'W'): 11, ('M', 'H'): -2, ('P', 'N'): -2,
    ('K', 'A'): -1, ('M', 'L'): 2, ('K', 'E'): 1, ('Z', 'E'): 4,
    ('X', 'N'): -1, ('Z', 'A'): -1, ('Z', 'M'): -1, ('X', 'F'): -1,
    ('K', 'C'): -3, ('B', 'Q'): 0, ('X', 'B'): -1, ('B', 'M'): -3,
    ('F', 'C'): -2, ('Z', 'Q'): 3, ('X', 'Z'): -1, ('F', 'G'): -3,
    ('B', 'E'): 1, ('X', 'V'): -1, ('F', 'K'): -3, ('B', 'A'): -2,
    ('X', 'R'): -1, ('D', 'D'): 6, ('W', 'G'): -2, ('Z', 'F'): -3,
    ('S', 'Q'): 0, ('W', 'C'): -2, ('W', 'K'): -3, ('H', 'Q'): 0,
    ('L', 'C'): -1, ('W', 'N'): -4, ('S', 'A'): 1, ('L', 'G'): -4,
    ('W', 'S'): -3, ('S', 'E'): 0, ('H', 'E'): 0, ('S', 'I'): -2,
    ('H', 'A'): -2, ('S', 'M'): -1, ('Y', 'L'): -1, ('Y', 'H'): 2,
    ('Y', 'D'): -3, ('E', 'R'): 0, ('X', 'P'): -2, ('G', 'G'): 6,
    ('G', 'C'): -3, ('E', 'N'): 0, ('Y', 'T'): -2, ('Y', 'P'): -3,
    ('T', 'K'): -1, ('A', 'A'): 4, ('P', 'Q'): -1, ('T', 'C'): -1,
    ('V', 'H'): -3, ('T', 'G'): -2, ('I', 'Q'): -3, ('Z', 'T'): -1,
    ('C', 'R'): -3, ('V', 'P'): -2, ('P', 'E'): -1, ('M', 'C'): -1,
    ('K', 'N'): 0, ('I', 'I'): 4, ('P', 'A'): -1, ('M', 'G'): -3,
    ('T', 'S'): 1, ('I', 'E'): -3, ('P', 'M'): -2, ('M', 'K'): -1,
    ('I', 'A'): -1, ('P', 'I'): -3, ('R', 'R'): 5, ('X', 'M'): -1,
    ('L', 'I'): 2, ('X', 'I'): -1, ('Z', 'B'): 1, ('X', 'E'): -1,
    ('Z', 'N'): 0, ('X', 'A'): 0, ('B', 'R'): -1, ('B', 'N'): 3,
    ('F', 'D'): -3, ('X', 'Y'): -1, ('Z', 'R'): 0, ('F', 'H'): -1,
    ('B', 'F'): -3, ('F', 'L'): 0, ('X', 'Q'): -1, ('B', 'B'): 4
}
        # ... load actual BLOSUM62 values
        return blosum62


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