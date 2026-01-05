"""
COCOM v11: LRA Text (IMDb Sentiment) Training

LRA Text: Char-level IMDb reviews, 4096 tokens, binary classification
Using tokens_conv1d for grouped token embedding
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import json

from cocom_11 import COCOMv11


class LRATextDataset(Dataset):
    """LRA Text dataset from pickle files."""
    
    def __init__(self, pickle_path: str):
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        self.sequences = [item['input_ids_0'] for item in data]
        self.labels = [item['label'] for item in data]
        
        # Get vocab size
        max_token = max(max(seq) for seq in self.sequences)
        self.vocab_size = max_token + 1
        self.seq_len = len(self.sequences[0])
        
        print(f"Loaded {len(self.sequences)} samples, seq_len={self.seq_len}, vocab_size={self.vocab_size}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


class LRATextTrainerV11:
    """Trainer for COCOM v11 on LRA Text."""
    
    STAGES = [
        {'name': 'tactical', 'epochs': 8, 'trainable': ['embed', 'tactical'], 'lr': 0.003, 'warmup': True},
        {'name': 'strategic', 'epochs': 5, 'trainable': ['strategic'], 'lr': 0.001, 'warmup': False},
        {'name': 'exploratory', 'epochs': 5, 'trainable': ['exploratory'], 'lr': 0.001, 'warmup': False},
        {'name': 'full', 'epochs': 20, 'trainable': ['all'], 'lr': 3e-4, 'warmup': False},
    ]
    
    def __init__(
        self,
        data_dir: str = ".",
        batch_size: int = 32,
        device: str = 'cuda',
        use_fp16: bool = True,
        group_size: int = 16,  # Group tokens for shorter sequence
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and 'cuda' in device
        self.group_size = group_size
        
        # Load data
        self.train_data = LRATextDataset(f"{data_dir}/lra-text.train.pickle")
        self.test_data = LRATextDataset(f"{data_dir}/lra-text.test.pickle")
        
        vocab_size = max(self.train_data.vocab_size, self.test_data.vocab_size)
        seq_len = self.train_data.seq_len
        new_seq_len = seq_len // group_size
        
        print(f"\nOriginal: vocab_size={vocab_size}, seq_len={seq_len}")
        print(f"Grouped: kernel_size={group_size}, new_seq_len={new_seq_len}")
        
        self.train_loader = DataLoader(
            self.train_data, batch_size=batch_size, shuffle=True, num_workers=0
        )
        self.test_loader = DataLoader(
            self.test_data, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        # Create model with grouped tokenization
        self.model = COCOMv11(
            d_model=256,
            n_heads=8,
            num_classes=2,  # Binary classification
            seq_len=seq_len,
            input_type='tokens_conv1d',
            vocab_size=vocab_size,
            embed_kernel_size=group_size,
            use_rope=True,
            rope_dims=1,
        ).to(self.device)
        
        params = sum(p.numel() for p in self.model.parameters())
        print(f"\nCOCOMv11 Parameters: {params:,}")
        print(f"Sequence: {seq_len} → {self.model.seq_len}")
        
        self.scaler = GradScaler(enabled=self.use_fp16)
        self.checkpoint_dir = Path("./checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.history = []
    
    def get_stage_logits(self, stage_name: str, analysis: dict) -> torch.Tensor:
        if stage_name == 'tactical':
            return analysis['logits_t']
        elif stage_name == 'strategic':
            return analysis['logits_s']
        elif stage_name == 'exploratory':
            return analysis['logits_e']
        else:
            return analysis['avg_logits']
    
    def get_loss(self, stage_name: str, analysis: dict, y: torch.Tensor) -> torch.Tensor:
        if stage_name == 'tactical':
            return F.cross_entropy(analysis['logits_t'], y)
        elif stage_name == 'strategic':
            return F.cross_entropy(analysis['logits_s'], y)
        elif stage_name == 'exploratory':
            return F.cross_entropy(analysis['logits_e'], y)
        else:
            avg_loss = F.cross_entropy(analysis['avg_logits'], y)
            arb_loss = F.cross_entropy(analysis['arb_logits'], y)
            return avg_loss + 0.5 * arb_loss
    
    def train_epoch(self, stage_name: str, optimizer, scheduler=None) -> dict:
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        correct_s = correct_t = correct_e = correct_arb = 0
        
        pbar = tqdm(self.train_loader, desc=f"  {stage_name}", leave=False)
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=self.use_fp16):
                final_logits, analysis = self.model(x)
                loss = self.get_loss(stage_name, analysis, y)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
            
            if scheduler is not None:
                scheduler.step()
            
            stage_logits = self.get_stage_logits(stage_name, analysis)
            total_correct += (stage_logits.argmax(-1) == y).sum().item()
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
            'stage_acc': total_correct / total_samples,
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
        total_agree = 0
        n_batches = 0
        
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
            total_agree += analysis['agreement_rate']
            n_batches += 1
        
        return {
            'acc': total_correct / total_samples,
            'acc_s': correct_s / total_samples,
            'acc_t': correct_t / total_samples,
            'acc_e': correct_e / total_samples,
            'acc_arb': correct_arb / total_samples,
            'agree': total_agree / n_batches
        }
    
    def run(self):
        print("\n" + "="*70)
        print(f"COCOM v11: LRA TEXT (group={self.group_size})")
        print("="*70)
        
        best_acc = 0
        
        for stage in self.STAGES:
            stage_name = stage['name']
            epochs = stage['epochs']
            trainable = stage['trainable']
            lr = stage['lr']
            
            print(f"\n{'='*70}")
            print(f"STAGE: {stage_name.upper()} ({epochs} epochs)")
            print("="*70)
            
            self.model.freeze_all_except(trainable)
            trainable_params = self.model.get_trainable_params()
            print(f"Trainable: {trainable_params:,}")
            
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr,
                betas=(0.9, 0.99) if stage_name == 'tactical' else (0.9, 0.999),
                weight_decay=0.01
            )
            
            total_steps = len(self.train_loader) * epochs
            
            if stage.get('warmup', False):
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=lr, total_steps=total_steps,
                    pct_start=0.15, anneal_strategy='cos',
                    div_factor=25, final_div_factor=100
                )
            else:
                scheduler = None
            
            stage_best = 0
            no_improve = 0
            
            for epoch in range(epochs):
                train_metrics = self.train_epoch(stage_name, optimizer, scheduler)
                test_metrics = self.evaluate()
                
                saved = ""
                if test_metrics['acc'] > stage_best:
                    stage_best = test_metrics['acc']
                    no_improve = 0
                    torch.save(self.model.state_dict(), 
                              self.checkpoint_dir / f"cocom_v11_text_{stage_name}.pt")
                    saved = " 💾"
                    
                    if test_metrics['acc'] > best_acc:
                        best_acc = test_metrics['acc']
                        torch.save(self.model.state_dict(), 
                                  self.checkpoint_dir / "cocom_v11_text_best.pt")
                else:
                    no_improve += 1
                
                print(f"  Epoch {epoch+1}/{epochs} | Test: {test_metrics['acc']:.4f} | "
                      f"S={test_metrics['acc_s']:.3f} T={test_metrics['acc_t']:.3f} "
                      f"E={test_metrics['acc_e']:.3f} Arb={test_metrics['acc_arb']:.3f}{saved}")
                
                self.history.append({
                    'stage': stage_name, 
                    'epoch': epoch + 1,
                    **test_metrics
                })
                
                if stage_name == 'full' and no_improve >= 5:
                    print(f"  Early stopping")
                    break
            
            print(f"Stage {stage_name} best: {stage_best:.4f}")
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Best Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
        
        torch.save(self.model.state_dict(), 
                  self.checkpoint_dir / "cocom_v11_text_full.pt")
        
        with open(self.checkpoint_dir / "cocom_v11_text_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return best_acc


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--group_size', type=int, default=16, help='Token grouping (8, 16, 32)')
    parser.add_argument('--fp32', action='store_true')
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Group size: {args.group_size}")
    
    trainer = LRATextTrainerV11(
        batch_size=args.batch_size,
        device=device,
        use_fp16=not args.fp32,
        group_size=args.group_size,
    )
    
    trainer.run()


