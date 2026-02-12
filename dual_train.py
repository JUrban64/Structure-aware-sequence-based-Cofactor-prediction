"""
Dual-branch Trainer: trénuje na PDB strukturách I sekvencích.

Trénovací strategie:
  1. Sekvence-only batche  – trénují Seq branch + sdílený classifier
  2. Struktura-only batche – trénují GNN branch + sdílený classifier  
  3. Oba současně batche   – trénují obě větve + consistency loss

Typický scénář:
  - 500 PDB struktur s NAD (strukturní data)
  - 10 000 sekvencí z UniProt s NAD anotací (sekvenční data)
  → Model se učí na obou, sdílený classifier se učí z mnohem více dat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np
from sklearn.metrics import roc_auc_score

from sequence_dataset import collate_sequences


class DualTrainer:
    """
    Trainer pro DualBranchPredictor.
    
    Střídá batche z:
      - graph_loader (PDB strukturní data → GNN branch)
      - seq_loader (sekvence bez struktury → Sequence branch)
      - (volitelně) both_loader (PDB data procházejí oběma větvemi)
    """
    
    def __init__(self, model, 
                 graph_train_loader, graph_val_loader,
                 seq_train_loader, seq_val_loader,
                 device='cuda',
                 lr=0.001,
                 weight_decay=1e-5,
                 consistency_weight=0.3,
                 seq_weight=1.0,
                 struct_weight=1.0):
        """
        Args:
            model: DualBranchPredictor
            graph_train_loader: PyG DataLoader (PDB grafy)
            graph_val_loader: PyG DataLoader
            seq_train_loader: torch DataLoader (sekvence)
            seq_val_loader: torch DataLoader
            consistency_weight: váha consistency loss (0 = vypnuto)
            seq_weight: váha sequence loss
            struct_weight: váha structure loss
        """
        self.model = model.to(device)
        self.device = device
        
        self.graph_train_loader = graph_train_loader
        self.graph_val_loader = graph_val_loader
        self.seq_train_loader = seq_train_loader
        self.seq_val_loader = seq_val_loader
        
        self.consistency_weight = consistency_weight
        self.seq_weight = seq_weight
        self.struct_weight = struct_weight
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self):
        """
        Jeden epoch: střídá batche ze struct a seq loaderů.
        
        Strategie: interleaved training
          1. Batch ze seq loaderu → train seq branch
          2. Batch z graph loaderu → train gnn branch (+ optional consistency)
          3. Opakovat
        """
        self.model.train()
        
        total_loss = 0.0
        total_struct_loss = 0.0
        total_seq_loss = 0.0
        total_consistency_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0
        
        # Iterátory
        seq_iter = iter(self.seq_train_loader)
        graph_iter = iter(self.graph_train_loader)
        
        # Střídáme: sequence batch, graph batch, sequence batch, ...
        seq_done = False
        graph_done = False
        
        while not (seq_done and graph_done):
            # ---- Sequence batch ----
            if not seq_done:
                try:
                    seq_batch = next(seq_iter)
                except StopIteration:
                    seq_done = True
                    seq_batch = None
            
            if seq_batch is not None:
                self.optimizer.zero_grad()
                
                esm_emb = seq_batch['embeddings'].to(self.device)
                seq_mask = seq_batch['mask'].to(self.device)
                seq_labels = seq_batch['labels'].to(self.device)
                
                logits, _ = self.model(
                    mode='sequence',
                    esm_embeddings=esm_emb,
                    seq_mask=seq_mask
                )
                
                seq_loss = self.criterion(logits, seq_labels) * self.seq_weight
                seq_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_seq_loss += seq_loss.item()
                total_loss += seq_loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == seq_labels).sum().item()
                total += seq_labels.size(0)
                n_batches += 1
            
            # ---- Graph batch ----
            if not graph_done:
                try:
                    graph_batch = next(graph_iter)
                except StopIteration:
                    graph_done = True
                    graph_batch = None
            
            if graph_batch is not None:
                self.optimizer.zero_grad()
                
                graph_batch = graph_batch.to(self.device)
                
                # Struktura pouze
                logits, embeddings = self.model(
                    mode='structure',
                    graph_data=graph_batch
                )
                
                struct_loss = self.criterion(
                    logits, graph_batch.y
                ) * self.struct_weight
                
                loss = struct_loss
                total_struct_loss += struct_loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == graph_batch.y).sum().item()
                total += graph_batch.num_graphs
                n_batches += 1
        
        if total == 0:
            return 0.0, 0.0, {}
        
        metrics = {
            'total_loss': total_loss / n_batches,
            'struct_loss': total_struct_loss / max(1, n_batches),
            'seq_loss': total_seq_loss / max(1, n_batches),
            'consistency_loss': total_consistency_loss / max(1, n_batches),
            'accuracy': correct / total
        }
        
        return metrics['total_loss'], metrics['accuracy'], metrics
    
    def train_epoch_with_consistency(self, both_loader):
        """
        Trénuje s consistency loss na PDB datech,
        kde můžeme pustit OBOJE (graf i sekvenci).
        
        both_loader: DataLoader, kde každý batch obsahuje
        graf data I ESM embeddings celé sekvence.
        
        Volat po standardním train_epoch() pro extra consistency.
        """
        self.model.train()
        total_consistency = 0.0
        n = 0
        
        for batch in both_loader:
            self.optimizer.zero_grad()
            
            graph_data = batch['graph'].to(self.device)
            esm_emb = batch['esm_embeddings'].to(self.device)
            seq_mask = batch['mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            logits, embeddings = self.model(
                mode='both',
                graph_data=graph_data,
                esm_embeddings=esm_emb,
                seq_mask=seq_mask
            )
            
            # Classification loss
            cls_loss = self.criterion(logits, labels)
            
            # Consistency loss
            consistency_loss = self.model.get_consistency_loss(embeddings)
            
            loss = cls_loss + self.consistency_weight * consistency_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_consistency += consistency_loss.item()
            n += 1
        
        return total_consistency / max(1, n)
    
    def validate(self):
        """Validace na obou typech dat."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            # ---- Validate on graphs ----
            if self.graph_val_loader is not None:
                for batch in self.graph_val_loader:
                    batch = batch.to(self.device)
                    logits, _ = self.model(mode='structure', graph_data=batch)
                    loss = self.criterion(logits, batch.y)
                    
                    total_loss += loss.item() * batch.num_graphs
                    pred = logits.argmax(dim=1)
                    correct += (pred == batch.y).sum().item()
                    total += batch.num_graphs
                    
                    probs = F.softmax(logits, dim=1)[:, 1]
                    all_preds.extend(probs.cpu().numpy())
                    all_labels.extend(batch.y.cpu().numpy())
            
            # ---- Validate on sequences ----
            if self.seq_val_loader is not None:
                for seq_batch in self.seq_val_loader:
                    esm_emb = seq_batch['embeddings'].to(self.device)
                    seq_mask = seq_batch['mask'].to(self.device)
                    labels = seq_batch['labels'].to(self.device)
                    
                    logits, _ = self.model(
                        mode='sequence',
                        esm_embeddings=esm_emb,
                        seq_mask=seq_mask
                    )
                    loss = self.criterion(logits, labels)
                    
                    total_loss += loss.item() * labels.size(0)
                    pred = logits.argmax(dim=1)
                    correct += (pred == labels).sum().item()
                    total += labels.size(0)
                    
                    probs = F.softmax(logits, dim=1)[:, 1]
                    all_preds.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        
        if total == 0:
            return 0.0, 0.0, 0.0
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        # AUC (potřebuje obě třídy)
        if len(set(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_preds)
        else:
            auc = 0.0
        
        return avg_loss, accuracy, auc
    
    def validate_per_branch(self):
        """
        Validace zvlášť pro každou větev – užitečné pro porovnání
        jak dobře si vede GNN vs Sequence branch.
        """
        self.model.eval()
        results = {}
        
        with torch.no_grad():
            # GNN branch
            if self.graph_val_loader is not None:
                preds, labels = [], []
                for batch in self.graph_val_loader:
                    batch = batch.to(self.device)
                    logits, _ = self.model(mode='structure', graph_data=batch)
                    probs = F.softmax(logits, dim=1)[:, 1]
                    preds.extend(probs.cpu().numpy())
                    labels.extend(batch.y.cpu().numpy())
                
                if len(set(labels)) > 1:
                    results['gnn_auc'] = roc_auc_score(labels, preds)
                else:
                    results['gnn_auc'] = 0.0
                results['gnn_n'] = len(labels)
            
            # Sequence branch
            if self.seq_val_loader is not None:
                preds, labels = [], []
                for seq_batch in self.seq_val_loader:
                    esm_emb = seq_batch['embeddings'].to(self.device)
                    seq_mask = seq_batch['mask'].to(self.device)
                    lab = seq_batch['labels'].to(self.device)
                    
                    logits, _ = self.model(
                        mode='sequence',
                        esm_embeddings=esm_emb,
                        seq_mask=seq_mask
                    )
                    probs = F.softmax(logits, dim=1)[:, 1]
                    preds.extend(probs.cpu().numpy())
                    labels.extend(lab.cpu().numpy())
                
                if len(set(labels)) > 1:
                    results['seq_auc'] = roc_auc_score(labels, preds)
                else:
                    results['seq_auc'] = 0.0
                results['seq_n'] = len(labels)
        
        return results
    
    def train(self, num_epochs=100, both_loader=None):
        """
        Hlavní trénovací smyčka.
        
        Args:
            num_epochs: počet epoch
            both_loader: volitelný loader pro consistency training
        """
        best_auc = 0
        
        for epoch in range(num_epochs):
            # ---- Train ----
            train_loss, train_acc, train_metrics = self.train_epoch()
            
            # Consistency (volitelné, pokud máme PDB data s ESM emb.)
            consistency = 0.0
            if both_loader is not None and self.consistency_weight > 0:
                consistency = self.train_epoch_with_consistency(both_loader)
            
            # ---- Validate ----
            val_loss, val_acc, val_auc = self.validate()
            
            # Per-branch metrics
            branch_metrics = self.validate_per_branch()
            
            # LR scheduling
            self.scheduler.step(val_auc)
            
            # Logging
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"    Struct loss: {train_metrics.get('struct_loss', 0):.4f}, "
                  f"Seq loss: {train_metrics.get('seq_loss', 0):.4f}")
            if consistency > 0:
                print(f"    Consistency loss: {consistency:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                  f"AUC: {val_auc:.4f}")
            if 'gnn_auc' in branch_metrics:
                print(f"    GNN branch AUC:  {branch_metrics['gnn_auc']:.4f} "
                      f"(n={branch_metrics['gnn_n']})")
            if 'seq_auc' in branch_metrics:
                print(f"    Seq branch AUC:  {branch_metrics['seq_auc']:.4f} "
                      f"(n={branch_metrics['seq_n']})")
            
            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(self.model.state_dict(), 'best_dual_model.pth')
                print(f"  → New best AUC: {best_auc:.4f}")
        
        print(f"\nTraining complete. Best AUC: {best_auc:.4f}")


# ============================================================
# Příklad kompletního pipeline
# ============================================================
if __name__ == '__main__':
    """
    Ukázkový pipeline:
    
    1. Načti PDB data (stávající pipeline)
    2. Načti sequence-only data (nový)
    3. Vytvoř DualBranchPredictor
    4. Trénuj na obou typech dat
    """
    from dual_predictor import DualBranchPredictor
    from sequence_dataset import SequenceDataset, collate_sequences
    from sklearn.model_selection import train_test_split
    
    print("=" * 60)
    print("DUAL TRAINING PIPELINE")
    print("=" * 60)
    
    # ---- 1. Strukturní data (PDB) ----
    # Předpokládáme, že binding_sites a dataset jsou již vytvořeny
    # z existujícího pipeline (Binding_site_ex → ESM → graph)
    #
    # from binding_site_graph import BindingSiteGraphDataset
    # dataset = BindingSiteGraphDataset(binding_sites, ...)
    # train_graphs, val_graphs = train_test_split(dataset.graphs, ...)
    # graph_train_loader = PyGDataLoader(train_graphs, batch_size=32)
    # graph_val_loader = PyGDataLoader(val_graphs, batch_size=32)
    
    # ---- 2. Sekvenční data (UniProt) ----
    # sequences, labels = load_sequences_from_csv('data/nad_sequences.csv')
    # seq_dataset = SequenceDataset(sequences, labels, esm_extractor)
    # seq_train, seq_val = train_test_split(...)
    # seq_train_loader = DataLoader(seq_train, batch_size=16, 
    #                               collate_fn=collate_sequences)
    # seq_val_loader = DataLoader(seq_val, batch_size=16,
    #                             collate_fn=collate_sequences)
    
    # ---- 3. Model ----
    model = DualBranchPredictor(
        esm_dim=1280,
        node_dim=1310,
        hidden_dim=256,
        num_gnn_layers=3,
        num_attention_heads=4,
        dropout=0.5,
        use_gat=True
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ---- 4. Trainer ----
    # trainer = DualTrainer(
    #     model=model,
    #     graph_train_loader=graph_train_loader,
    #     graph_val_loader=graph_val_loader,
    #     seq_train_loader=seq_train_loader,
    #     seq_val_loader=seq_val_loader,
    #     device='cuda' if torch.cuda.is_available() else 'cpu',
    #     consistency_weight=0.3,  # váha consistency loss
    #     seq_weight=1.0,
    #     struct_weight=1.0
    # )
    # trainer.train(num_epochs=100)
    
    print("\nPro spuštění tréninku:")
    print("  1. Připravte PDB data (stávající pipeline)")
    print("  2. Připravte CSV/FASTA se sekvencemi + labely")
    print("  3. Odkomentujte kód výše a spusťte")
