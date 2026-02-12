class SequenceOnlyPredictor:
    """
    Pro sekvence bez známé struktury:
    1. Predict binding site (sequence-based)
    2. Extract ESM features
    3. Predict local contact map
    4. Build graph → inference
    """
    
    def __init__(self, model, esm_extractor, contact_predictor):
        self.model = model
        self.esm_extractor = esm_extractor
        self.contact_predictor = contact_predictor
    
    def predict_binding_site(self, sequence):
        """
        Možnost 1: Assume celá sekvence (pokud je krátká)
        Možnost 2: Use predictor (např. P2Rank, DeepSite)
        Možnost 3: Sliding window approach
        """
        # Pro jednoduchost: použít celou sekvenci pokud L < 100
        if len(sequence) < 100:
            return list(range(len(sequence)))
        
        # Jinak: predict binding site residues
        # ... (můžete použít ML model nebo heuristiku)
        
        return list(range(len(sequence)))  # placeholder
    
    def predict(self, sequence):
        # 1. Identify binding site
        bs_indices = self.predict_binding_site(sequence)
        bs_sequence = ''.join([sequence[i] for i in bs_indices])
        
        # 2. Extract features
        esm_emb = self.esm_extractor.extract_embeddings(sequence)
        bs_esm = esm_emb[bs_indices]
        
        # 3. Additional features
        node_features = create_node_features({
            'esm_embeddings': bs_esm,
            'binding_site_sequence': bs_sequence,
            'binding_site_indices': bs_indices,
            'full_sequence': sequence
        })
        
        # 4. Predict contact map
        contact_map = self.contact_predictor.predict(bs_sequence)
        
        # 5. Build graph
        graph = self._build_graph(node_features, contact_map)
        
        # 6. Inference
        with torch.no_grad():
            logits = self.model(graph)
            prob = F.softmax(logits, dim=1)[0, 1]
        
        return prob.item()
