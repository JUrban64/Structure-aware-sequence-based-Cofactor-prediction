from transformers import AutoTokenizer, EsmModel
import torch



class ESMFeatureExtractor:
    """
    Extract ESM-2 embeddings for protein sequences
    """
    
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D"):
        """
        Options:
        - esm2_t33_650M_UR50D (650M params, 1280D embeddings)
        - esm2_t36_3B_UR50D (3B params, 2560D embeddings) - nejlepší
        - esm2_t30_150M_UR50D (150M params, 640D embeddings) - rychlejší
        """
        print(f"Loading ESM-2 model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name)
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        print(f"Model loaded on {self.device}")
    
    def extract_embeddings(self, sequence):
        """
        Extract per-residue embeddings
        
        Args:
            sequence: amino acid sequence (string)
        
        Returns:
            embeddings: [L, 1280] numpy array
        """
        # Tokenize
        inputs = self.tokenizer(
            sequence, 
            return_tensors="pt",
            add_special_tokens=True  # Adds <cls> and <eos>
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract embeddings (remove <cls> and <eos> tokens)
        embeddings = outputs.last_hidden_state[0, 1:-1, :]  # [L, 1280]
        
        return embeddings.cpu().numpy()
    
    def extract_binding_site_embeddings(self, full_sequence, bs_indices):
        """
        Extract embeddings only for binding site residues
        
        Args:
            full_sequence: full protein sequence
            bs_indices: list of binding site residue indices
        
        Returns:
            bs_embeddings: [n_bs, 1280]
        """
        # Get full embeddings
        full_embeddings = self.extract_embeddings(full_sequence)
        
        # Select binding site residues
        bs_embeddings = full_embeddings[bs_indices, :]
        
        return bs_embeddings
    
    def batch_extract(self, sequences, batch_size=8):
        """
        Extract embeddings for multiple sequences (more efficient)
        """
        all_embeddings = []
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract (handle padding)
            for j, seq in enumerate(batch):
                seq_len = len(seq)
                emb = outputs.last_hidden_state[j, 1:seq_len+1, :]
                all_embeddings.append(emb.cpu().numpy())
        
        return all_embeddings


# Použití
esm_extractor = ESMFeatureExtractor()



# Pre-compute embeddings pro všechny binding sites
for bs_info in binding_sites:
    # Extract ESM embeddings
    bs_embeddings = esm_extractor.extract_binding_site_embeddings(
        bs_info['full_sequence'],
        bs_info['binding_site_indices']
    )
    
    bs_info['esm_embeddings'] = bs_embeddings
    print(f"Extracted embeddings: {bs_embeddings.shape}")