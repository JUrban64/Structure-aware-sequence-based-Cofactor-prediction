"""
Dataset pro sekvence BEZ struktury (sequence-only training).

Zdroje dat:
  - UniProt anotace (cofactor binding annotation)
  - Swiss-Prot reviewed entries s GO terms pro NAD binding
  - Jakýkoli CSV/FASTA se sekvencemi a labely

Tento dataset extrahuje ESM embeddings z celých sekvencí
a vytváří tensory pro SequenceBranch modelu.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json
import csv


class SequenceDataset(Dataset):
    """
    Dataset pro sekvence bez PDB struktury.
    
    Vstupní formát (CSV):
        uniprot_id,sequence,label,cofactor
        P12345,MVLSPADKTN...,1,NAD
        Q67890,MGKYVLTSIG...,0,
    
    Nebo z FASTA + JSON labels.
    """
    
    def __init__(self, sequences, labels, esm_extractor=None,
                 precomputed_embeddings=None, max_length=1024):
        """
        Args:
            sequences: list of AA sequences (strings)
            labels: list of int labels (1=binds NAD, 0=doesn't)
            esm_extractor: ESMFeatureExtractor instance (pro on-the-fly extraction)
            precomputed_embeddings: dict {seq_id: np.array [L, 1280]}
            max_length: maximální délka sekvence (delší se oříznou)
        """
        assert len(sequences) == len(labels)
        
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        self.esm_extractor = esm_extractor
        self.precomputed = precomputed_embeddings or {}
        
        # Pre-compute embeddings pokud máme extraktor a nejsou precomputed
        if esm_extractor is not None and len(self.precomputed) == 0:
            print("Pre-computing ESM embeddings for sequence dataset...")
            self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        """Pre-compute a uložit ESM embeddings pro všechny sekvence."""
        for i, seq in enumerate(self.sequences):
            if i not in self.precomputed:
                truncated = seq[:self.max_length]
                emb = self.esm_extractor.extract_embeddings(truncated)
                self.precomputed[i] = emb
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(self.sequences)} sequences")
        
        print(f"  Done. {len(self.precomputed)} embeddings computed.")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with:
                'embeddings': [L, 1280] tensor
                'label': int
                'length': int (actual sequence length)
                'sequence': str
        """
        seq = self.sequences[idx][:self.max_length]
        label = self.labels[idx]
        
        if idx in self.precomputed:
            emb = self.precomputed[idx]
        elif self.esm_extractor is not None:
            emb = self.esm_extractor.extract_embeddings(seq)
            self.precomputed[idx] = emb
        else:
            raise RuntimeError(
                f"No embeddings for index {idx} and no ESM extractor provided"
            )
        
        return {
            'embeddings': torch.FloatTensor(emb),     # [L, 1280]
            'label': torch.LongTensor([label])[0],     # scalar
            'length': len(seq),
            'sequence': seq
        }


def collate_sequences(batch):
    """
    Custom collate function pro variable-length sekvence.
    Padduje na max délku v batchi.
    
    Returns:
        embeddings: [B, max_L, 1280] padded tensor
        mask: [B, max_L] bool mask (True = padding)
        labels: [B] tensor
    """
    embeddings = [item['embeddings'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    lengths = [item['length'] for item in batch]
    
    max_len = max(lengths)
    emb_dim = embeddings[0].size(1)
    
    # Pad
    padded = torch.zeros(len(batch), max_len, emb_dim)
    mask = torch.ones(len(batch), max_len, dtype=torch.bool)  # True = padding
    
    for i, (emb, length) in enumerate(zip(embeddings, lengths)):
        padded[i, :length, :] = emb
        mask[i, :length] = False  # not padding
    
    return {
        'embeddings': padded,
        'mask': mask,
        'labels': labels,
    }


# ============================================================
# Pomocné funkce pro načtení dat
# ============================================================

def load_sequences_from_csv(csv_path, cofactor_filter='NAD'):
    """
    Načte sekvence z CSV souboru.
    
    Očekávaný formát:
        uniprot_id,sequence,label,cofactor
        P12345,MVLSPADKTN...,1,NAD
        Q67890,MGKYVLTSIG...,0,
    
    Args:
        csv_path: cesta k CSV
        cofactor_filter: filtrovat jen tento kofaktor (None = všechny)
    
    Returns:
        sequences, labels: lists
    """
    sequences = []
    labels = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if cofactor_filter and row.get('cofactor', '') != cofactor_filter:
                if int(row['label']) == 1:
                    continue  # skip positive examples for other cofactors
            
            sequences.append(row['sequence'])
            labels.append(int(row['label']))
    
    print(f"Loaded {len(sequences)} sequences from {csv_path}")
    print(f"  Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")
    
    return sequences, labels


def load_sequences_from_fasta(fasta_path, labels_path):
    """
    Načte sekvence z FASTA a labely ze JSON souboru.
    
    labels.json formát:
        {"P12345": 1, "Q67890": 0, ...}
    """
    from Bio import SeqIO
    
    with open(labels_path, 'r') as f:
        label_dict = json.load(f)
    
    sequences = []
    labels = []
    
    for record in SeqIO.parse(fasta_path, 'fasta'):
        seq_id = record.id.split('|')[1] if '|' in record.id else record.id
        if seq_id in label_dict:
            sequences.append(str(record.seq))
            labels.append(label_dict[seq_id])
    
    print(f"Loaded {len(sequences)} sequences from {fasta_path}")
    print(f"  Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")
    
    return sequences, labels


def load_from_uniprot_annotations(keywords=None):
    """
    Placeholder pro stahování anotovaných sekvencí z UniProt.
    
    Příklad query pro NAD-binding proteiny:
        https://rest.uniprot.org/uniprotkb/search?
            query=(cc_cofactor:"NAD")%20AND%20(reviewed:true)
            &format=fasta
    
    Pro negativní příklady:
        - Random reviewed sekvence BEZ cofactor anotace
        - Nebo proteiny s jiným kofaktorem
    """
    print("Pro stažení dat z UniProt použijte:")
    print("  Positive (NAD-binding):")
    print("    https://rest.uniprot.org/uniprotkb/search?"
          "query=(cc_cofactor:NAD)+AND+(reviewed:true)&format=fasta")
    print("  Negative (non-NAD):")
    print("    https://rest.uniprot.org/uniprotkb/search?"
          "query=(reviewed:true)+NOT+(cc_cofactor:NAD)&format=fasta")
    print()
    print("Nebo z příkazové řádky:")
    print("  curl -o nad_positive.fasta 'URL_POSITIVE'")
    print("  curl -o nad_negative.fasta 'URL_NEGATIVE'")
    
    return [], []


def save_embeddings(embeddings_dict, output_path):
    """Uloží precomputed embeddings pro pozdější použití."""
    np.savez_compressed(output_path, **{
        str(k): v for k, v in embeddings_dict.items()
    })
    print(f"Saved {len(embeddings_dict)} embeddings to {output_path}")


def load_embeddings(input_path):
    """Načte precomputed embeddings."""
    data = np.load(input_path, allow_pickle=True)
    embeddings = {int(k): data[k] for k in data.files}
    print(f"Loaded {len(embeddings)} embeddings from {input_path}")
    return embeddings


# ============================================================
# Příklad použití
# ============================================================
if __name__ == '__main__':
    # Příklad: vytvoření datasetu z CSV
    # sequences, labels = load_sequences_from_csv('data/nad_sequences.csv')
    
    # Příklad: vytvoření datasetu přímo
    example_sequences = [
        "MGKVLITGASSGIGKAT",  # krátká NAD-binding sekvence (příklad)
        "MSKGEELFTGVVPILVEL",  # ne-NAD protein (příklad)
    ]
    example_labels = [1, 0]
    
    dataset = SequenceDataset(
        sequences=example_sequences,
        labels=example_labels,
        esm_extractor=None,  # potřebuje ESMFeatureExtractor
        max_length=512
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Positive: {sum(example_labels)}, Negative: {len(example_labels) - sum(example_labels)}")
