"""
COCOM v11: Embedding Transfer Experiment

Load model trained on CIFAR-10 with 2x2 patches (256 seq_len)
Replace embedding to work with 1024 raw pixels
Freeze all except embedding, retrain only embedding
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from tqdm import tqdm

from cocom_11 import COCOMv11
from cocom_11.layers.normalization import RMSNorm


class EmbeddingTransferTrainer:
    """Transfer trained experts to new embedding (1024 raw pixels)."""
    
    def __init__(
        self,
        checkpoint_path: str = "./checkpoints/cocom_v11_cifar10_full.pt",
        data_dir: str = "./lra_data/image",
        batch_size: int = 256,
        device: str = 'cuda',
        use_fp16: bool = True,
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and 'cuda' in device
        
        # Load data
        data_path = Path(data_dir)
        train_sequences = np.load(data_path / "train_sequences.npy")
        train_labels = np.load(data_path / "train_labels.npy")
        test_sequences = np.load(data_path / "test_sequences.npy")
        test_labels = np.load(data_path / "test_labels.npy")
        
        print(f"Loaded train: {len(train_sequences)} samples, seq_len={train_sequences.shape[1]}")
        print(f"Loaded test: {len(test_sequences)} samples")
        
        num_classes = int(train_labels.max()) + 1
        new_seq_len = train_sequences.shape[1]  # 1024
        print(f"Target seq_len: {new_seq_len}")
        
        train_dataset = TensorDataset(
            torch.tensor(train_sequences, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.long)
        )
        test_dataset = TensorDataset(
            torch.tensor(test_sequences, dtype=torch.float32),
            torch.tensor(test_labels, dtype=torch.long)
        )
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Create model with ORIGINAL config (2x2 patch, 256 seq_len)
        self.model = COCOMv11(
            d_model=256,
            n_heads=8,
            num_classes=num_classes,
            img_size=32,
            patch_size=2,  # Original: 2x2 = 256 patches
            input_type='image',
            use_rope=True,
            rope_dims=2,
        ).to(self.device)
        
        # Load trained weights
        checkpoint = Path(checkpoint_path)
        if checkpoint.exists():
            state_dict = torch.load(checkpoint, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✅ Loaded checkpoint: {checkpoint}")
        else:
            print(f"❌ Checkpoint not found: {checkpoint}")
            return
        
        # Now replace embedding for 1024 seq_len
        print("\nReplacing embedding for 1024 raw pixels...")
        
        # New embedding: Linear for continuous pixel values
        self.model.input_embed = nn.Linear(1, 256).to(self.device)
        self.model.input_type = 'continuous'
        self.model.seq_len = new_seq_len
        self.model.grid_size = int(new_seq_len ** 0.5)  # 32 for 1024
        
        # New positional embedding for 1024 positions
        self.model.pos_embed = nn.Parameter(
            torch.randn(1, new_seq_len, 256, device=self.device) * 0.02
        )
        
        # Freeze ALL except embedding
        for name, param in self.model.named_parameters():
            if 'input_embed' in name or 'pos_embed' in name or 'norm' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        
        self.scaler = GradScaler(enabled=self.use_fp16)
        self.checkpoint_dir = Path("./checkpoints")
    
    def train_epoch(self, optimizer, scheduler=None) -> dict:
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        correct_s = correct_t = correct_e = correct_arb = 0
        
        pbar = tqdm(self.train_loader, desc="  embed", leave=False)
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=self.use_fp16):
                final_logits, analysis = self.model(x)
                loss = F.cross_entropy(analysis['avg_logits'], y)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
            
            if scheduler is not None:
                scheduler.step()
            
            total_correct += (final_logits.argmax(-1) == y).sum().item()
            total_samples += y.size(0)
            total_loss += loss.item() * y.size(0)
            
            correct_s += (analysis['pred_s'] == y).sum().item()
            correct_t += (analysis['pred_t'] == y).sum().item()
            correct_e += (analysis['pred_e'] == y).sum().item()
            correct_arb += (analysis['arb_logits'].argmax(-1) == y).sum().item()
            
            pbar.set_postfix(
                loss=loss.item(),
                acc=total_correct/total_samples,
                S=correct_s/total_samples,
                T=correct_t/total_samples,
                E=correct_e/total_samples,
            )
        
        return {
            'loss': total_loss / total_samples,
            'acc': total_correct / total_samples,
            'acc_s': correct_s / total_samples,
            'acc_t': correct_t / total_samples,
            'acc_e': correct_e / total_samples,
            'acc_arb': correct_arb / total_samples,
        }
    
    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        total_correct = 0
        total_samples = 0
        correct_s = correct_t = correct_e = correct_arb = 0
        
        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            with autocast(enabled=self.use_fp16):
                final_logits, analysis = self.model(x)
            
            total_correct += (final_logits.argmax(-1) == y).sum().item()
            total_samples += y.size(0)
            correct_s += (analysis['pred_s'] == y).sum().item()
            correct_t += (analysis['pred_t'] == y).sum().item()
            correct_e += (analysis['pred_e'] == y).sum().item()
            correct_arb += (analysis['arb_logits'].argmax(-1) == y).sum().item()
        
        return {
            'acc': total_correct / total_samples,
            'acc_s': correct_s / total_samples,
            'acc_t': correct_t / total_samples,
            'acc_e': correct_e / total_samples,
            'acc_arb': correct_arb / total_samples,
        }
    
    def run(self, epochs: int = 10, lr: float = 0.001):
        print("\n" + "="*70)
        print("EMBEDDING TRANSFER: 256 patches → 1024 raw pixels")
        print("="*70)
        print(f"Training only embedding for {epochs} epochs, lr={lr}")
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=0.01
        )
        
        total_steps = len(self.train_loader) * epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=total_steps,
            pct_start=0.1, anneal_strategy='cos'
        )
        
        best_acc = 0
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(optimizer, scheduler)
            test_metrics = self.evaluate()
            
            saved = ""
            if test_metrics['acc'] > best_acc:
                best_acc = test_metrics['acc']
                torch.save(self.model.state_dict(), 
                          self.checkpoint_dir / "cocom_v11_cifar10_transfer.pt")
                saved = " 💾"
            
            print(f"  Epoch {epoch+1}/{epochs} | Test: {test_metrics['acc']:.4f} | "
                  f"S={test_metrics['acc_s']:.3f} T={test_metrics['acc_t']:.3f} "
                  f"E={test_metrics['acc_e']:.3f} Arb={test_metrics['acc_arb']:.3f}{saved}")
        
        print("\n" + "="*70)
        print("TRANSFER COMPLETE")
        print("="*70)
        print(f"Best Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
        print(f"Original (2x2 patches): 68.77%")
        print(f"Transfer (1024 pixels): {best_acc*100:.2f}%")
        
        return best_acc


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--checkpoint', type=str, 
                        default='./checkpoints/cocom_v11_cifar10_full.pt')
    args = parser.parse_args()
    
    trainer = EmbeddingTransferTrainer(
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    trainer.run(epochs=args.epochs, lr=args.lr)


