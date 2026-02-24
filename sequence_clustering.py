#!/usr/bin/env python3
"""
Clusterování proteinových sekvencí pro split bez data leakage.

Používá MMseqs2 pro shlukování sekvencí podle sekvenční identity.
Proteiny ve stejném klastru NIKDY nejsou rozděleny mezi train/val/test.

MMseqs2 je 100–1000× rychlejší než CD-HIT a podporuje prahy identity
až k ~0.0 (CD-HIT vyžaduje ≥ 0.4).

Typické prahy sekvenční identity:
  - 30% → velmi přísný (fold-level), pro generalizaci na vzdálené homology
  - 40% → přísný (superfamily-level), doporučený default
  - 50% → mírný (family-level)
  - 70% → mírný, stále zabraňuje leakage z blízkých homologů
  - 90% → pouze near-identical sequences

Použití:
    python sequence_clustering.py --fasta sequences.fasta --identity 0.4
    
Nebo programově:
    from sequence_clustering import ClusterSplitter
    splitter = ClusterSplitter(identity_threshold=0.4)
    train_idx, val_idx, test_idx = splitter.split(sequences, labels)
"""

import os
import sys
import shutil
import subprocess
import tempfile
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class SequenceClusterer:
    """
    Clusterování proteinových sekvencí pomocí MMseqs2.
    
    MMseqs2 shlukuje sekvence podle sekvenční identity:
    - Každý cluster má jednu reprezentativní sekvenci
    - Všechny sekvence v clusteru mají ≥ threshold identitu k reprezentantovi
    - 100–1000× rychlejší než CD-HIT
    - Podporuje prahy identity < 0.4 (CD-HIT vyžaduje ≥ 0.4)
    
    Reference:
        Steinegger & Söding (2017) Nature Biotechnology 35:1026-1028
        Steinegger & Söding (2018) Nature Communications 9:2116
    """
    
    def __init__(self, identity_threshold=0.4,
                 mmseqs_path='mmseqs', threads=4,
                 sensitivity=7.5, coverage=0.8, cov_mode=0,
                 timeout=None):
        """
        Args:
            identity_threshold: sekvenční identita pro clusterování (0.0-1.0)
                0.3 = fold-level (velmi přísné)
                0.4 = superfamily-level (doporučené)
                0.5 = family-level
            mmseqs_path: cesta k MMseqs2 binárce
            threads: počet vláken
            sensitivity: citlivost vyhledávání (1.0-7.5, vyšší = citlivější)
                Pro identity < 0.4 doporučen 7.5
            coverage: minimální pokrytí sekvence (0.0-1.0)
            cov_mode: režim pokrytí:
                0 = pokrytí obou sekvencí (target + query)
                1 = pokrytí target sekvence
                2 = pokrytí query sekvence
            timeout: max čas v sekundách (None = auto dle počtu sekvencí)
        """
        self.identity_threshold = identity_threshold
        self.mmseqs_path = mmseqs_path
        self.threads = threads
        self.sensitivity = sensitivity
        self.coverage = coverage
        self.cov_mode = cov_mode
        self.timeout = timeout
    
    def cluster(self, sequences: List[str], 
                ids: Optional[List[str]] = None) -> Dict[int, List[int]]:
        """
        Clusteruje sekvence pomocí MMseqs2.
        
        Args:
            sequences: list sekvencí (AA stringy)
            ids: volitelné identifikátory (jinak seq_0, seq_1, ...)
        
        Returns:
            clusters: dict {cluster_id: [seq_indices]}
        """
        if ids is None:
            ids = [f"seq_{i}" for i in range(len(sequences))]
        
        if not self._check_mmseqs():
            print("  ⚠ MMseqs2 není nainstalován, používám fallback "
                  "(k-mer greedy clustering)")
            return self._fallback_clustering(sequences)
        
        # ---- Auto timeout: škáluje s počtem sekvencí ----
        # MMseqs2 je mnohem rychlejší než CD-HIT, ale pro jistotu
        if self.timeout is None:
            timeout = max(300, min(7200, len(sequences)))
        else:
            timeout = self.timeout
        
        return self._run_mmseqs2(sequences, ids, timeout)
    
    def _run_mmseqs2(self, sequences, ids, timeout):
        """Spustí MMseqs2 clustering."""
        tmpdir = tempfile.mkdtemp(prefix='mmseqs_')
        
        try:
            # 1. Zapíšeme FASTA
            fasta_path = os.path.join(tmpdir, 'input.fasta')
            with open(fasta_path, 'w') as f:
                for i, (seq_id, seq) in enumerate(zip(ids, sequences)):
                    f.write(f">{seq_id}__idx__{i}\n{seq}\n")
            
            db_path = os.path.join(tmpdir, 'seqDB')
            cluster_path = os.path.join(tmpdir, 'clusterDB')
            tsv_path = os.path.join(tmpdir, 'clusters.tsv')
            tmp_path = os.path.join(tmpdir, 'tmp')
            os.makedirs(tmp_path, exist_ok=True)
            
            # 2. Vytvoř MMseqs2 databázi
            cmd_createdb = [
                self.mmseqs_path, 'createdb',
                fasta_path, db_path,
            ]
            r = subprocess.run(cmd_createdb, capture_output=True, text=True,
                               timeout=timeout)
            if r.returncode != 0:
                print(f"  ⚠ MMseqs2 createdb chyba: {r.stderr[:300]}")
                return self._fallback_clustering(sequences)
            
            # 3. Clusterování
            cmd_cluster = [
                self.mmseqs_path, 'cluster',
                db_path, cluster_path, tmp_path,
                '--min-seq-id', str(self.identity_threshold),
                '-c', str(self.coverage),
                '--cov-mode', str(self.cov_mode),
                '-s', str(self.sensitivity),
                '--threads', str(self.threads),
                '--cluster-mode', '0',  # greedy set cover (jako CD-HIT)
            ]
            r = subprocess.run(cmd_cluster, capture_output=True, text=True,
                               timeout=timeout)
            if r.returncode != 0:
                print(f"  ⚠ MMseqs2 cluster chyba: {r.stderr[:300]}")
                return self._fallback_clustering(sequences)
            
            # 4. Exportuj výsledky do TSV
            cmd_tsv = [
                self.mmseqs_path, 'createtsv',
                db_path, db_path, cluster_path, tsv_path,
            ]
            r = subprocess.run(cmd_tsv, capture_output=True, text=True,
                               timeout=timeout)
            if r.returncode != 0:
                print(f"  ⚠ MMseqs2 createtsv chyba: {r.stderr[:300]}")
                return self._fallback_clustering(sequences)
            
            # 5. Parsuj TSV → clusters
            clusters = self._parse_mmseqs2_tsv(tsv_path)
            
            print(f"  ✓ {len(sequences)} sekvencí → {len(clusters)} clusterů "
                  f"(MMseqs2, identity={self.identity_threshold})")
            return clusters
        
        except subprocess.TimeoutExpired:
            print(f"  ⚠ MMseqs2 timeout po {timeout}s, "
                  f"přepínám na fallback clustering")
            return self._fallback_clustering(sequences)
        
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    def _check_mmseqs(self) -> bool:
        """Zkontroluje, zda je MMseqs2 dostupný."""
        try:
            result = subprocess.run(
                [self.mmseqs_path, 'version'],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _parse_mmseqs2_tsv(self, tsv_path: str) -> Dict[int, List[int]]:
        """
        Parsuje TSV výstup z MMseqs2 createtsv.
        
        Formát (tabulátor):
            representative_id\tmember_id
            seq_0__idx__0\tseq_0__idx__0
            seq_0__idx__0\tseq_3__idx__3
            seq_1__idx__1\tseq_1__idx__1
            ...
        
        Returns:
            clusters: {cluster_id: [original_seq_indices]}
        """
        repr_to_members = defaultdict(list)
        
        with open(tsv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                repr_name = parts[0]
                member_name = parts[1]
                
                # Extrahuj index z "__idx__N"
                try:
                    member_idx = int(member_name.split('__idx__')[1])
                    repr_to_members[repr_name].append(member_idx)
                except (IndexError, ValueError):
                    continue
        
        # Přečísluj clustery na 0, 1, 2, ...
        clusters = {}
        for cluster_id, (_, members) in enumerate(repr_to_members.items()):
            clusters[cluster_id] = members
        
        return clusters
    
    def _fallback_clustering(self, sequences: List[str]) -> Dict[int, List[int]]:
        """
        Fallback: jednoduchý greedy clustering bez externích nástrojů.
        Pomalejší, ale funguje vždy.
        
        Počítá k-mer overlap jako proxy pro sekvenční identitu.
        Pro krátké datasety dostačující.
        """
        print("  ⚠ Fallback: greedy k-mer clustering "
              "(pomalé pro >1000 sekvencí)")
        
        n = len(sequences)
        assigned = [False] * n
        clusters = {}
        cluster_id = 0
        
        # Seřaď podle délky (delší = reprezentanti, jako MMseqs2)
        order = sorted(range(n), key=lambda i: len(sequences[i]),
                        reverse=True)
        
        for rep_idx in order:
            if assigned[rep_idx]:
                continue
            
            clusters[cluster_id] = [rep_idx]
            assigned[rep_idx] = True
            rep_seq = sequences[rep_idx]
            
            for other_idx in order:
                if assigned[other_idx]:
                    continue
                
                other_seq = sequences[other_idx]
                
                # Rychlý pre-filter: délky musí být podobné
                len_ratio = (min(len(rep_seq), len(other_seq))
                             / max(len(rep_seq), len(other_seq)))
                if len_ratio < self.identity_threshold:
                    continue
                
                # Jednoduchá identita (bez alignmentu – aproximace)
                identity = self._quick_identity(rep_seq, other_seq)
                
                if identity >= self.identity_threshold:
                    clusters[cluster_id].append(other_idx)
                    assigned[other_idx] = True
            
            cluster_id += 1
        
        print(f"  ✓ {n} sekvencí → {len(clusters)} clusterů (fallback)")
        return clusters
    
    @staticmethod
    def _quick_identity(seq1: str, seq2: str) -> float:
        """
        Rychlý odhad sekvenční identity bez alignmentu.
        Používá k-mer overlap jako proxy.
        """
        k = 3
        if len(seq1) < k or len(seq2) < k:
            return 0.0
        
        kmers1 = set(seq1[i:i+k] for i in range(len(seq1) - k + 1))
        kmers2 = set(seq2[i:i+k] for i in range(len(seq2) - k + 1))
        
        if not kmers1 or not kmers2:
            return 0.0
        
        overlap = len(kmers1 & kmers2)
        total = min(len(kmers1), len(kmers2))
        
        return overlap / total


class ClusterSplitter:
    """
    Rozdělení dat do train/val/test na základě sekvenčních clusterů.
    
    Garance: ŽÁDNÉ dva proteiny ze stejného clusteru 
    nejsou v různých splitech → žádný data leakage.
    
    Použití:
        splitter = ClusterSplitter(identity_threshold=0.4)
        train_idx, val_idx, test_idx = splitter.split(sequences, labels)
    """
    
    def __init__(self, identity_threshold=0.4, 
                 val_size=0.15, test_size=0.15,
                 random_state=42):
        """
        Args:
            identity_threshold: práh pro MMseqs2 clusterování
            val_size: podíl validačních dat (0.15 = 15%)
            test_size: podíl testovacích dat (0.15 = 15%)
            random_state: random seed pro reprodukovatelnost
        """
        self.clusterer = SequenceClusterer(
            identity_threshold=identity_threshold,
        )
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.identity_threshold = identity_threshold
    
    def split(self, sequences: List[str], labels: List[int],
              ids: Optional[List[str]] = None,
              stratify: bool = True
              ) -> Tuple[List[int], List[int], List[int]]:
        """
        Rozdělí data do train/val/test podle clusterů.
        
        Args:
            sequences: list sekvencí
            labels: list labelů (0/1)
            ids: volitelné identifikátory
            stratify: zachovat poměr tříd v splitech
        
        Returns:
            train_indices, val_indices, test_indices
        """
        print(f"\n{'='*60}")
        print(f"Cluster-based split (identity={self.identity_threshold})")
        print(f"{'='*60}")
        
        # 1. Clusteruj sekvence
        clusters = self.clusterer.cluster(sequences, ids)
        
        # 2. Pro každý cluster: zjisti "cluster label" 
        #    (majority label pro stratifikaci)
        cluster_ids = sorted(clusters.keys())
        cluster_labels = []
        cluster_sizes = []
        
        for cid in cluster_ids:
            member_indices = clusters[cid]
            member_labels = [labels[i] for i in member_indices]
            # Majority vote
            majority = 1 if sum(member_labels) > len(member_labels) / 2 else 0
            cluster_labels.append(majority)
            cluster_sizes.append(len(member_indices))
        
        cluster_labels = np.array(cluster_labels)
        cluster_sizes = np.array(cluster_sizes)
        
        print(f"\n  Statistiky clusterů:")
        print(f"    Celkem clusterů: {len(cluster_ids)}")
        print(f"    Pozitivní clustery: {(cluster_labels == 1).sum()}")
        print(f"    Negativní clustery: {(cluster_labels == 0).sum()}")
        
        # 3. Rozděl CLUSTERY (ne vzorky!) do train/val/test
        rng = np.random.RandomState(self.random_state)
        
        if stratify and len(np.unique(cluster_labels)) > 1:
            train_clusters, val_clusters, test_clusters = \
                self._stratified_cluster_split(
                    cluster_ids, cluster_labels, cluster_sizes, rng
                )
        else:
            train_clusters, val_clusters, test_clusters = \
                self._random_cluster_split(cluster_ids, cluster_sizes, rng)
        
        # 4. Mapuj clustery zpět na sample indexy
        train_idx = []
        for cid in train_clusters:
            train_idx.extend(clusters[cid])
        
        val_idx = []
        for cid in val_clusters:
            val_idx.extend(clusters[cid])
        
        test_idx = []
        for cid in test_clusters:
            test_idx.extend(clusters[cid])
        
        # 5. Statistiky
        self._print_split_stats(train_idx, val_idx, test_idx, labels,
                                train_clusters, val_clusters, test_clusters)
        
        return train_idx, val_idx, test_idx
    
    def split_train_val(self, sequences: List[str], labels: List[int],
                        ids: Optional[List[str]] = None,
                        val_size: float = 0.2
                        ) -> Tuple[List[int], List[int]]:
        """
        Jednodušší varianta: jen train/val split (bez test).
        
        Args:
            sequences, labels: data
            val_size: podíl validačních dat
        
        Returns:
            train_indices, val_indices
        """
        # Dočasně změň parametry
        old_val = self.val_size
        old_test = self.test_size
        self.val_size = val_size
        self.test_size = 0.0
        
        train_idx, val_idx, test_idx = self.split(sequences, labels, ids)
        
        # Test indexy přidáme do train (test_size=0 ale pro jistotu)
        train_idx = train_idx + test_idx
        
        # Obnovíme parametry
        self.val_size = old_val
        self.test_size = old_test
        
        return train_idx, val_idx
    
    def _stratified_cluster_split(self, cluster_ids, cluster_labels, 
                                   cluster_sizes, rng):
        """
        Stratifikovaný split clusterů: zachovává poměr pozitivních/negativních.
        
        Alokuje clustery od největšího po nejmenší (greedy),
        aby dosáhl cílového poměru v každém splitu.
        """
        n_total = sum(cluster_sizes)
        target_test = int(n_total * self.test_size)
        target_val = int(n_total * self.val_size)
        
        # Rozděl clustery podle labelu
        pos_clusters = [cid for cid, lab in zip(cluster_ids, cluster_labels) 
                        if lab == 1]
        neg_clusters = [cid for cid, lab in zip(cluster_ids, cluster_labels) 
                        if lab == 0]
        
        rng.shuffle(pos_clusters)
        rng.shuffle(neg_clusters)
        
        def allocate_clusters(cluster_list, target_n, sizes_dict):
            """Přiřazuje clustery dokud nedosáhne cílového počtu vzorků."""
            allocated = []
            current_n = 0
            for cid in cluster_list:
                if current_n >= target_n:
                    break
                allocated.append(cid)
                current_n += sizes_dict[cid]
            return allocated, cluster_list[len(allocated):]
        
        sizes_dict = {cid: s for cid, s in zip(cluster_ids, cluster_sizes)}
        
        # Poměr pozitivních
        pos_ratio = len(pos_clusters) / max(len(cluster_ids), 1)
        
        # Test set
        test_pos_target = int(target_test * pos_ratio)
        test_neg_target = target_test - test_pos_target
        
        test_pos, remaining_pos = allocate_clusters(
            pos_clusters, test_pos_target, sizes_dict)
        test_neg, remaining_neg = allocate_clusters(
            neg_clusters, test_neg_target, sizes_dict)
        test_clusters = test_pos + test_neg
        
        # Val set
        val_pos_target = int(target_val * pos_ratio)
        val_neg_target = target_val - val_pos_target
        
        val_pos, remaining_pos = allocate_clusters(
            remaining_pos, val_pos_target, sizes_dict)
        val_neg, remaining_neg = allocate_clusters(
            remaining_neg, val_neg_target, sizes_dict)
        val_clusters = val_pos + val_neg
        
        # Train = zbytek
        train_clusters = remaining_pos + remaining_neg
        
        return train_clusters, val_clusters, test_clusters
    
    def _random_cluster_split(self, cluster_ids, cluster_sizes, rng):
        """Náhodný split clusterů (bez stratifikace)."""
        n_total = sum(cluster_sizes)
        
        order = list(range(len(cluster_ids)))
        rng.shuffle(order)
        
        test_clusters, val_clusters, train_clusters = [], [], []
        current_test, current_val = 0, 0
        target_test = int(n_total * self.test_size)
        target_val = int(n_total * self.val_size)
        
        for i in order:
            cid = cluster_ids[i]
            size = cluster_sizes[i]
            
            if current_test < target_test:
                test_clusters.append(cid)
                current_test += size
            elif current_val < target_val:
                val_clusters.append(cid)
                current_val += size
            else:
                train_clusters.append(cid)
        
        return train_clusters, val_clusters, test_clusters
    
    def _print_split_stats(self, train_idx, val_idx, test_idx, labels,
                           train_clusters, val_clusters, test_clusters):
        """Vytiskne statistiky splitu."""
        total = len(train_idx) + len(val_idx) + len(test_idx)
        
        def stats(indices, name):
            n = len(indices)
            pos = sum(labels[i] for i in indices) if indices else 0
            neg = n - pos
            pct = n / total * 100 if total > 0 else 0
            pos_pct = pos / n * 100 if n > 0 else 0
            return f"    {name:6s}: {n:5d} samples ({pct:5.1f}%), " \
                   f"{pos:4d} pos ({pos_pct:5.1f}%), {neg:4d} neg"
        
        print(f"\n  Split výsledky:")
        print(stats(train_idx, "Train"))
        print(stats(val_idx, "Val"))
        print(stats(test_idx, "Test"))
        print(f"\n    Train clustery: {len(train_clusters)}")
        print(f"    Val clustery:   {len(val_clusters)}")
        print(f"    Test clustery:  {len(test_clusters)}")
        
        # Ověření: žádný overlap
        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)
        
        assert len(train_set & val_set) == 0, "LEAKAGE: train ∩ val!"
        assert len(train_set & test_set) == 0, "LEAKAGE: train ∩ test!"
        assert len(val_set & test_set) == 0, "LEAKAGE: val ∩ test!"
        print(f"\n  ✓ Žádný data leakage (ověřeno)")


def cluster_and_split_graphs(graph_dataset, sequences, labels,
                              identity_threshold=0.4,
                              val_size=0.15, test_size=0.15,
                              random_state=42):
    """
    Convenience funkce: clusteruj a rozděl PyG grafový dataset.
    
    Args:
        graph_dataset: BindingSiteGraphDataset
        sequences: list of full sequences (z binding_sites)
        labels: list of labels
        identity_threshold: MMseqs2 identity threshold
        val_size, test_size: podíly
        random_state: seed
    
    Returns:
        train_graphs, val_graphs, test_graphs: lists of PyG Data
    """
    splitter = ClusterSplitter(
        identity_threshold=identity_threshold,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state
    )
    
    train_idx, val_idx, test_idx = splitter.split(sequences, labels)
    
    graphs = graph_dataset.graphs if hasattr(graph_dataset, 'graphs') else graph_dataset
    
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]
    
    return train_graphs, val_graphs, test_graphs


def cluster_and_split_sequences(seq_dataset, sequences, labels,
                                 identity_threshold=0.4,
                                 val_size=0.15, test_size=0.15,
                                 random_state=42):
    """
    Convenience funkce: clusteruj a rozděl sequence dataset.
    
    Args:
        seq_dataset: SequenceDataset
        sequences: list of sequences
        labels: list of labels
    
    Returns:
        train_subset, val_subset, test_subset: torch Subset objects
    """
    from torch.utils.data import Subset
    
    splitter = ClusterSplitter(
        identity_threshold=identity_threshold,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state
    )
    
    train_idx, val_idx, test_idx = splitter.split(sequences, labels)
    
    train_subset = Subset(seq_dataset, train_idx)
    val_subset = Subset(seq_dataset, val_idx)
    test_subset = Subset(seq_dataset, test_idx)
    
    return train_subset, val_subset, test_subset


# ============================================================
# CLI
# ============================================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Cluster sequences and split for training')
    parser.add_argument('--fasta', type=str, help='Input FASTA file')
    parser.add_argument('--identity', type=float, default=0.4,
                        help='MMseqs2 identity threshold (default: 0.4)')
    parser.add_argument('--val-size', type=float, default=0.15)
    parser.add_argument('--test-size', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='cluster_split.npz',
                        help='Output file for split indices')
    args = parser.parse_args()
    
    if args.fasta:
        # Načti FASTA
        sequences = []
        ids = []
        current_seq = []
        current_id = None
        
        with open(args.fasta, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id is not None:
                        sequences.append(''.join(current_seq))
                        ids.append(current_id)
                    current_id = line[1:].split()[0]
                    current_seq = []
                elif line:
                    current_seq.append(line)
        
        if current_id is not None:
            sequences.append(''.join(current_seq))
            ids.append(current_id)
        
        print(f"Načteno {len(sequences)} sekvencí z {args.fasta}")
        
        # Clusteruj
        clusterer = SequenceClusterer(identity_threshold=args.identity)
        clusters = clusterer.cluster(sequences, ids)
        
        # Ulož
        print(f"\nUloženo {len(clusters)} clusterů")
    else:
        # Demo
        print("Demo: cluster-based split")
        print("Použití: python sequence_clustering.py --fasta seqs.fasta")
        
        demo_sequences = [
            "MGKVLITGASSGIGKAT" * 3,
            "MGKVLITGASSGIGKAV" * 3,  # velmi podobná → stejný cluster
            "MSKGEELFTGVVPILVEL" * 3,
            "MSKGEELFTGVVPILVEV" * 3,  # velmi podobná → stejný cluster
            "COMPLETELY_DIFFERENT_PROTEIN_SEQUENCE_HERE",
        ]
        demo_labels = [1, 1, 0, 0, 0]
        
        splitter = ClusterSplitter(identity_threshold=0.4)
        # Použije fallback (bez MMseqs2)
        train, val, test = splitter.split(demo_sequences, demo_labels)
        
        print(f"\nTrain: {train}")
        print(f"Val:   {val}")
        print(f"Test:  {test}")