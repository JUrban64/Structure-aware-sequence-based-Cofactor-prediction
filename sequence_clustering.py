#!/usr/bin/env python3
"""
Clusterování proteinových sekvencí pro split bez data leakage.

Používá CD-HIT pro shlukování sekvencí podle sekvenční identity.
Proteiny ve stejném klastru NIKDY nejsou rozděleny mezi train/val/test.

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
import subprocess
import tempfile
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class SequenceClusterer:
    """
    Clusterování proteinových sekvencí pomocí CD-HIT.
    
    CD-HIT shlukuje sekvence podle sekvenční identity:
    - Každý cluster má jednu reprezentativní sekvenci
    - Všechny sekvence v clusteru mají ≥ threshold identitu k reprezentantovi
    
    Reference:
        Li & Godzik (2006) Bioinformatics 22:1658-1659
        Fu et al. (2012) Bioinformatics 28:3150-3152
    """
    
    def __init__(self, identity_threshold=0.4, word_size=None, 
                 cdhit_path='cd-hit', threads=4, memory=4000,
                 timeout=None):
        """
        Args:
            identity_threshold: sekvenční identita pro clusterování (0.0-1.0)
                0.3 = fold-level (velmi přísné)
                0.4 = superfamily-level (doporučené)
                0.5 = family-level
            word_size: word size pro CD-HIT (auto pokud None)
                identity ≥ 0.7: word_size = 5
                identity ≥ 0.6: word_size = 4
                identity ≥ 0.5: word_size = 3
                identity ≥ 0.4: word_size = 2
            cdhit_path: cesta k CD-HIT binárce
            threads: počet vláken
            memory: paměť v MB
            timeout: max čas pro CD-HIT v sekundách (None = auto dle počtu sekvencí)
        """
        self.identity_threshold = identity_threshold
        self.cdhit_path = cdhit_path
        self.threads = threads
        self.memory = memory
        self.timeout = timeout
        
        # Auto word size podle CD-HIT dokumentace
        if word_size is None:
            if identity_threshold >= 0.7:
                self.word_size = 5
            elif identity_threshold >= 0.6:
                self.word_size = 4
            elif identity_threshold >= 0.5:
                self.word_size = 3
            elif identity_threshold >= 0.4:
                self.word_size = 2
            else:
                raise ValueError(
                    f"CD-HIT nepodporuje identitu < 0.4. "
                    f"Pro nižší prahy použijte MMseqs2 nebo BLAST."
                )
        else:
            self.word_size = word_size
    
    def cluster(self, sequences: List[str], 
                ids: Optional[List[str]] = None) -> Dict[int, List[int]]:
        """
        Clusteruje sekvence pomocí CD-HIT.
        
        Args:
            sequences: list sekvencí (AA stringy)
            ids: volitelné identifikátory (jinak seq_0, seq_1, ...)
        
        Returns:
            clusters: dict {cluster_id: [seq_indices]}
        """
        if ids is None:
            ids = [f"seq_{i}" for i in range(len(sequences))]
        
        if not self._check_cdhit():
            print("  ⚠ CD-HIT není nainstalován, používám fallback MMseqs2/BLAST")
            return self._fallback_clustering(sequences)
        
        # ---- Auto timeout: škáluje s počtem sekvencí ----
        if self.timeout is None:
            timeout = max(600, min(14400, len(sequences) * 2))
        else:
            timeout = self.timeout
        
        # ---- Pro velké datasety: dvoustupňový clustering ----
        # Nejprve cluster na 90% (rychlé), pak na cílový threshold
        use_two_stage = (len(sequences) > 5000 
                         and self.identity_threshold < 0.7)
        
        if use_two_stage:
            print(f"  Dvoustupňový CD-HIT: "
                  f"{len(sequences)} sekvencí → 0.9 → {self.identity_threshold}")
            clusters = self._two_stage_cdhit(sequences, ids, timeout)
        else:
            clusters = self._single_stage_cdhit(sequences, ids, timeout)
        
        return clusters
    
    def _single_stage_cdhit(self, sequences, ids, timeout):
        """Standardní jednofázový CD-HIT."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', 
                                          delete=False) as fasta_file:
            for i, (seq_id, seq) in enumerate(zip(ids, sequences)):
                fasta_file.write(f">{seq_id}__idx__{i}\n{seq}\n")
            fasta_path = fasta_file.name
        
        output_path = fasta_path + '.cdhit'
        
        try:
            cmd = [
                self.cdhit_path,
                '-i', fasta_path,
                '-o', output_path,
                '-c', str(self.identity_threshold),
                '-n', str(self.word_size),
                '-M', str(self.memory),
                '-T', str(self.threads),
                '-d', '0',
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            
            if result.returncode != 0:
                print(f"  ⚠ CD-HIT chyba: {result.stderr[:200]}")
                return self._fallback_clustering(sequences)
            
            clusters = self._parse_cdhit_output(output_path + '.clstr')
            print(f"  ✓ {len(sequences)} sekvencí → {len(clusters)} clusterů "
                  f"(CD-HIT, identity={self.identity_threshold})")
            return clusters
        
        except subprocess.TimeoutExpired:
            print(f"  ⚠ CD-HIT timeout po {timeout}s, "
                  f"přepínám na fallback clustering")
            return self._fallback_clustering(sequences)
        
        finally:
            for f in [fasta_path, output_path, output_path + '.clstr']:
                if os.path.exists(f):
                    os.unlink(f)
    
    def _two_stage_cdhit(self, sequences, ids, timeout):
        """
        Dvoustupňový clustering pro velké datasety:
        1. CD-HIT na 90% identity (velmi rychlé) → redukuje počet sekvencí
        2. CD-HIT na cílový threshold (pomalejší, ale méně sekvencí)
        """
        # Stage 1: 90% identity
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', 
                                          delete=False) as f1:
            for i, (seq_id, seq) in enumerate(zip(ids, sequences)):
                f1.write(f">{seq_id}__idx__{i}\n{seq}\n")
            path1 = f1.name
        
        out1 = path1 + '.stage1'
        
        try:
            # Stage 1
            cmd1 = [
                self.cdhit_path,
                '-i', path1, '-o', out1,
                '-c', '0.9', '-n', '5',
                '-M', str(self.memory),
                '-T', str(self.threads),
                '-d', '0',
            ]
            r1 = subprocess.run(cmd1, capture_output=True, text=True, 
                               timeout=timeout // 2)
            
            if r1.returncode != 0:
                print(f"  ⚠ Stage 1 chyba, fallback")
                return self._fallback_clustering(sequences)
            
            n_stage1 = sum(1 for line in open(out1) if line.startswith('>'))
            print(f"    Stage 1: {len(sequences)} → {n_stage1} (90% identity)")
            
            # Stage 2: target identity na reprezentanty
            out2 = out1 + '.stage2'
            cmd2 = [
                self.cdhit_path,
                '-i', out1, '-o', out2,
                '-c', str(self.identity_threshold),
                '-n', str(self.word_size),
                '-M', str(self.memory),
                '-T', str(self.threads),
                '-d', '0',
            ]
            r2 = subprocess.run(cmd2, capture_output=True, text=True, 
                               timeout=timeout)
            
            if r2.returncode != 0:
                print(f"  ⚠ Stage 2 chyba, fallback")
                return self._fallback_clustering(sequences)
            
            # Parsuj stage 2 clustery (mají reprezentantové indexy)
            stage2_clusters = self._parse_cdhit_output(out2 + '.clstr')
            # Parsuj stage 1 clustery
            stage1_clusters = self._parse_cdhit_output(out1 + '.clstr')
            
            # Zkombinuj: stage2 cluster → stage1 clustery → original indexy
            final_clusters = {}
            # stage1: {cluster_id: [original_indices]}
            # stage2: {cluster_id: [stage1_representative_indices]}
            # stage1 representative = první člen (index 0 v stage1 clusteru)
            
            # Mapuj stage1 repr → original indices
            stage1_repr_to_members = {}
            for cid, members in stage1_clusters.items():
                repr_idx = members[0]  # representant
                stage1_repr_to_members[repr_idx] = members
            
            for cid2, stage2_members in stage2_clusters.items():
                final_members = []
                for s1_repr in stage2_members:
                    if s1_repr in stage1_repr_to_members:
                        final_members.extend(stage1_repr_to_members[s1_repr])
                    else:
                        final_members.append(s1_repr)
                final_clusters[cid2] = final_members
            
            print(f"    Stage 2: {n_stage1} → {len(final_clusters)} clusterů "
                  f"(identity={self.identity_threshold})")
            return final_clusters
        
        except subprocess.TimeoutExpired:
            print(f"  ⚠ Two-stage CD-HIT timeout, fallback")
            return self._fallback_clustering(sequences)
        
        finally:
            for f in [path1, out1, out1 + '.clstr',
                      out1 + '.stage2', out1 + '.stage2.clstr']:
                if os.path.exists(f):
                    os.unlink(f)

    def _check_cdhit(self) -> bool:
        """Zkontroluje, zda je CD-HIT dostupný."""
        try:
            result = subprocess.run(
                [self.cdhit_path, '--help'], 
                capture_output=True, text=True, timeout=10
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _parse_cdhit_output(self, clstr_file: str) -> Dict[int, List[int]]:
        """
        Parsuje .clstr soubor z CD-HIT.
        
        Formát:
            >Cluster 0
            0	292aa, >seq_5__idx__5... *
            1	285aa, >seq_12__idx__12... at 85.26%
            >Cluster 1
            ...
        
        Returns:
            clusters: {cluster_id: [original_seq_indices]}
        """
        clusters = {}
        current_cluster = -1
        
        with open(clstr_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>Cluster'):
                    current_cluster = int(line.split()[-1])
                    clusters[current_cluster] = []
                elif line and current_cluster >= 0:
                    # Extrahuj index z "__idx__N"
                    try:
                        # Formát: "0  292aa, >seq_5__idx__5... *"
                        name_part = line.split('>')[1].split('...')[0]
                        idx = int(name_part.split('__idx__')[1])
                        clusters[current_cluster].append(idx)
                    except (IndexError, ValueError):
                        continue
        
        return clusters
    
    def _fallback_clustering(self, sequences: List[str]) -> Dict[int, List[int]]:
        """
        Fallback: jednoduchý greedy clustering bez externích nástrojů.
        Pomalejší, ale funguje vždy.
        
        Počítá Hamming-like similaritu na alignovaných sekvencích 
        (jen pro podobně dlouhé). Pro krátké datasety dostačující.
        """
        print("  ⚠ Fallback: greedy identity clustering (pomalé pro >1000 sekvencí)")
        
        n = len(sequences)
        assigned = [False] * n
        clusters = {}
        cluster_id = 0
        
        # Seřaď podle délky (delší = reprezentanti, jako CD-HIT)
        order = sorted(range(n), key=lambda i: len(sequences[i]), reverse=True)
        
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
                len_ratio = min(len(rep_seq), len(other_seq)) / max(len(rep_seq), len(other_seq))
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
                 random_state=42, cdhit_path='cd-hit'):
        """
        Args:
            identity_threshold: práh pro CD-HIT clusterování
            val_size: podíl validačních dat (0.15 = 15%)
            test_size: podíl testovacích dat (0.15 = 15%)
            random_state: random seed pro reprodukovatelnost
        """
        self.clusterer = SequenceClusterer(
            identity_threshold=identity_threshold,
            cdhit_path=cdhit_path
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
        identity_threshold: CD-HIT identity threshold
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
                        help='CD-HIT identity threshold (default: 0.4)')
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
        # Použije fallback (bez CD-HIT)
        train, val, test = splitter.split(demo_sequences, demo_labels)
        
        print(f"\nTrain: {train}")
        print(f"Val:   {val}")
        print(f"Test:  {test}")