"""
Dataset pro sekvence BEZ struktury (sequence-only training).

Zdroje dat:
  - UniProt anotace (cofactor binding annotation)
  - Swiss-Prot reviewed entries s GO terms pro NAD binding
  - Jakýkoli CSV/FASTA se sekvencemi a labely

Tento dataset extrahuje ESM embeddings z celých sekvencí
a vytváří tensory pro SequenceBranch modelu.
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json
import csv


class SequenceDataset(Dataset):
    """
    Dataset pro sekvence bez PDB struktury.
    
    Podporuje dva režimy:
      1) Lazy-loading z disku (emb_dir) – doporučeno pro úsporu RAM
      2) In-memory embeddingy (precomputed_embeddings / esm_extractor) – zpětná kompatibilita
    
    Vstupní formát (CSV):
        uniprot_id,sequence,label,cofactor
        P12345,MVLSPADKTN...,1,NAD
        Q67890,MGKYVLTSIG...,0,
    
    Nebo z FASTA + JSON labels.
    """
    
    def __init__(self, sequences, labels, esm_extractor=None,
                 precomputed_embeddings=None, max_length=1024,
                 emb_dir=None, seq_ids=None):
        """
        Args:
            sequences: list of AA sequences (strings)
            labels: list of int labels (1=binds NAD, 0=doesn't)
            esm_extractor: ESMFeatureExtractor instance (pro on-the-fly extraction)
            precomputed_embeddings: dict {seq_id: np.array [L, 1280]}
            max_length: maximální délka sekvence (delší se oříznou)
            emb_dir: složka s .npy soubory (lazy-loading z disku)
            seq_ids: list of identifikátorů sekvencí (pro pojmenování .npy souborů)
        """
        assert len(sequences) == len(labels)
        
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        self.esm_extractor = esm_extractor
        self.precomputed = precomputed_embeddings or {}
        self.emb_dir = emb_dir
        self.seq_ids = seq_ids or list(range(len(sequences)))
        
        # Lazy-loading režim: embeddingy se čtou z disku v __getitem__
        if self.emb_dir is not None:
            print(f"  SequenceDataset: lazy-loading z {self.emb_dir} "
                  f"({len(self.sequences)} sekvencí)")
        elif esm_extractor is not None and len(self.precomputed) == 0:
            # Zpětná kompatibilita: in-memory precompute
            print("Pre-computing ESM embeddings for sequence dataset...")
            self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        """Pre-compute a uložit ESM embeddings pro všechny sekvence (in-memory)."""
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
        
        # Režim 1: lazy-loading z disku
        if self.emb_dir is not None:
            sid = self.seq_ids[idx]
            npy_path = os.path.join(self.emb_dir, f"{sid}.npy")
            if os.path.exists(npy_path):
                emb = np.load(npy_path).astype(np.float32)
            elif self.esm_extractor is not None:
                emb = self.esm_extractor.extract_embeddings(seq)
                np.save(npy_path, emb.astype(np.float16))
            else:
                raise RuntimeError(
                    f"Embedding soubor {npy_path} neexistuje a "
                    f"ESM extractor není dostupný"
                )
        # Režim 2: in-memory
        elif idx in self.precomputed:
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
    
    Podporované formáty sloupců:
      A) Sequence,Cofactor_id,Cofactor_Name  (nový formát)
      B) Entry,Protein names,Sequence,Cofactor_id,Cofactor_Name  (nový formát negative)
      C) uniprot_id,sequence,label,cofactor  (starý formát)
    
    Label se odvodí z Cofactor_id: > 0 → positive, == 0 → negative.
    
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
            # Nový formát: Sequence + Cofactor_id
            if 'Sequence' in row:
                seq = row['Sequence'].strip()
                cofactor_id = int(row.get('Cofactor_id', 0))
                label = 1 if cofactor_id > 0 else 0
                # Filtrování podle jména kofaktoru
                cofactor_name = row.get('Cofactor_Name', '').strip()
                if cofactor_filter and label == 1:
                    if cofactor_name and cofactor_name != cofactor_filter:
                        continue
            # Starý formát: sequence + label
            elif 'sequence' in row:
                seq = row['sequence'].strip()
                label = int(row.get('label', 0))
                if cofactor_filter and row.get('cofactor', '') != cofactor_filter:
                    if label == 1:
                        continue
            else:
                continue
            
            if len(seq) > 10:  # Přeskoč příliš krátké sekvence
                sequences.append(seq)
                labels.append(label)
    
    print(f"Loaded {len(sequences)} sequences from {csv_path}")
    print(f"  Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")
    
    return sequences, labels


def load_sequences_from_separate_csvs(positive_csv, negative_csv, 
                                       cofactor_filter='NAD',
                                       max_negative=None):
    """
    Načte sekvence ze dvou oddělených CSV souborů (positive + negative).
    
    Args:
        positive_csv: cesta k CSV s pozitivními sekvencemi
        negative_csv: cesta k CSV s negativními sekvencemi
        cofactor_filter: filtrovat kofaktor (pro positive)
        max_negative: maximální počet negativních (pro balancování)
    
    Returns:
        sequences, labels: lists
    """
    sequences = []
    labels = []
    
    # Načti pozitivní
    if os.path.exists(positive_csv):
        pos_seq, pos_lab = load_sequences_from_csv(positive_csv, cofactor_filter)
        sequences.extend(pos_seq)
        labels.extend(pos_lab)
    else:
        print(f"  ⚠ Pozitivní CSV nenalezen: {positive_csv}")
    
    # Načti negativní
    if os.path.exists(negative_csv):
        neg_seq, neg_lab = load_sequences_from_csv(negative_csv, cofactor_filter=None)
        if max_negative and len(neg_seq) > max_negative:
            import random
            random.seed(42)
            idx = random.sample(range(len(neg_seq)), max_negative)
            neg_seq = [neg_seq[i] for i in idx]
            neg_lab = [neg_lab[i] for i in idx]
            print(f"  Omezeno na {max_negative} negativních sekvencí")
        sequences.extend(neg_seq)
        labels.extend(neg_lab)
    else:
        print(f"  ⚠ Negativní CSV nenalezen: {negative_csv}")
    
    print(f"Celkem: {len(sequences)} sekvencí "
          f"(pozitivní: {sum(labels)}, negativní: {len(labels) - sum(labels)})")
    
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
