import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=0.001,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Loss function (weighted for imbalanced data)
        # Assuming you'll add negative samples
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in self.train_loader:
            batch = batch.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            logits = self.model(batch)
            
            # Loss
            loss = self.criterion(logits, batch.y)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * batch.num_graphs
            pred = logits.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.num_graphs
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                
                logits = self.model(batch)
                loss = self.criterion(logits, batch.y)
                
                total_loss += loss.item() * batch.num_graphs
                pred = logits.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.num_graphs
                
                # For AUC calculation
                probs = F.softmax(logits, dim=1)[:, 1]
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        pred_classes = (all_preds >= 0.5).astype(int)
        
        # AUC
        if len(set(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_preds)
            ap = average_precision_score(all_labels, all_preds)
        else:
            auc = 0.0
            ap = 0.0
        
        # F1 score
        f1 = f1_score(all_labels, pred_classes, zero_division=0)
        
        return avg_loss, accuracy, auc, f1, ap
    
    def train(self, num_epochs=100):
        best_auc = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_auc, val_f1, val_ap = self.validate()
            
            # LR scheduling
            self.scheduler.step(val_auc)
            
            # Logging
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                  f"AUC: {val_auc:.4f}, F1: {val_f1:.4f}, AP: {val_ap:.4f}")
            
            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"  → New best AUC: {best_auc:.4f}")


if __name__ == '__main__':
    from torch_geometric.data import Batch

    # Split dataset
    train_graphs, val_graphs = train_test_split(
        list(dataset), 
        test_size=0.2, 
        random_state=42
    )

    # Create dataloaders
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32)

    # Train
    trainer = Trainer(model, train_loader, val_loader)
    trainer.train(num_epochs=100)