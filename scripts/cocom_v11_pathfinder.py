"""
COCOM v11: Pathfinder-32 Training with RWKV Tactical Expert

Changes from v10.2:
1. RWKV replaces EMA in Tactical Expert
2. Modular architecture from cocom_11/
3. RoPE enabled by default
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from cocom_11 import COCOMv11


class PathfinderTrainerV11:
    """Trainer for COCOM v11 with RWKV Tactical Expert."""
    
    # Training stages - optimized for RWKV Tactical
    STAGES = [
        {'name': 'tactical', 'epochs': 8, 'trainable': ['embed', 'tactical'], 'lr': 0.003, 'betas': (0.9, 0.98), 'warmup': True},
        {'name': 'strategic', 'epochs': 8, 'trainable': ['strategic'], 'lr': 0.001, 'betas': (0.9, 0.999), 'warmup': False},
        {'name': 'exploratory', 'epochs': 8, 'trainable': ['exploratory'], 'lr': 0.001, 'betas': (0.9, 0.999), 'warmup': False},
        {'name': 'full', 'epochs': 20, 'trainable': ['all'], 'lr': 3e-4, 'betas': (0.9, 0.999), 'warmup': False},
    ]
    
    def __init__(
        self,
        data_dir: str = "./lra_data/pathfinder_kaggle32",
        batch_size: int = 32,
        device: str = 'cuda',
        use_fp16: bool = True,
        input_type: str = 'image',
        patch_size: int = 4,
        embed_kernel_size: int = 4,
        use_rope: bool = True,
        rope_dims: int = 2,
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
        
        print(f"Loaded train: {len(train_sequences)} samples")
        print(f"Loaded test: {len(test_sequences)} samples")
        
        seq_len = train_sequences.shape[1]
        num_classes = int(train_labels.max()) + 1
        print(f"Seq length: {seq_len}, Classes: {num_classes}")
        
        # Create data loaders
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
        
        # Calculate img_size from seq_len (assuming square image)
        img_size = int(seq_len ** 0.5)
        
        # Create model (RoPE, RMSNorm, SwiGLU enabled by default)
        self.model = COCOMv11(
            d_model=256,
            n_heads=8,
            num_classes=num_classes,
            img_size=img_size,
            patch_size=patch_size,
            input_type=input_type,
            seq_len=seq_len,
            embed_kernel_size=embed_kernel_size,
            use_rope=use_rope,
            rope_dims=rope_dims,
        ).to(self.device)
        
        params = sum(p.numel() for p in self.model.parameters())
        print(f"COCOMv11 Parameters: {params:,}")
        print(f"Input: {input_type}, img_size={img_size}, patch_size={patch_size}")
        print(f"RoPE: {use_rope} (dims={rope_dims}), RWKV Tactical, RMSNorm, SwiGLU")
        
        self.scaler = GradScaler(enabled=self.use_fp16)
        self.checkpoint_dir = Path("./checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.history = []
    
    def get_stage_logits(self, stage_name: str, analysis: dict) -> torch.Tensor:
        """Get the logits that should be used for loss in each stage."""
        if stage_name == 'tactical':
            return analysis['logits_t']
        elif stage_name == 'strategic':
            return analysis['logits_s']
        elif stage_name == 'exploratory':
            return analysis['logits_e']
        elif stage_name == 'arbitrator':
            return analysis['arb_logits']
        else:  # full - v10 original
            return analysis['avg_logits']
    
    def get_loss(self, stage_name: str, analysis: dict, y: torch.Tensor) -> torch.Tensor:
        """Compute loss for the current training stage."""
        if stage_name == 'tactical':
            return F.cross_entropy(analysis['logits_t'], y)
        elif stage_name == 'strategic':
            return F.cross_entropy(analysis['logits_s'], y)
        elif stage_name == 'exploratory':
            return F.cross_entropy(analysis['logits_e'], y)
        elif stage_name == 'arbitrator':
            return F.cross_entropy(analysis['arb_logits'], y)
        else:  # full - как в v10: avg_loss + 0.5 * arb_loss
            avg_loss = F.cross_entropy(analysis['avg_logits'], y)
            arb_loss = F.cross_entropy(analysis['arb_logits'], y)
            return avg_loss + 0.5 * arb_loss
    
    def _compute_final_logits_v2(self, analysis: dict) -> torch.Tensor:
        """
        Improved final logits computation:
        - If 2+ experts agree, use MAJORITY logits (not avg of all 3)
        - If no consensus, use arbitrator
        """
        pred_s = analysis['pred_s']
        pred_t = analysis['pred_t']
        pred_e = analysis['pred_e']
        
        logits_s = analysis['logits_s']
        logits_t = analysis['logits_t']
        logits_e = analysis['logits_e']
        arb_logits = analysis['arb_logits']
        
        B = pred_s.size(0)
        num_classes = logits_s.size(1)
        
        # Check pairwise agreement
        agree_st = (pred_s == pred_t)
        agree_te = (pred_t == pred_e)
        agree_se = (pred_s == pred_e)
        
        # All agree - use avg (all are similar anyway)
        all_agree = agree_st & agree_te
        
        # Only 2 agree - use majority (average of the 2)
        only_st = agree_st & ~agree_te
        only_te = agree_te & ~agree_st
        only_se = agree_se & ~agree_st
        
        # No consensus - use arbitrator
        no_consensus = ~(agree_st | agree_te | agree_se)
        
        final_logits = torch.zeros_like(logits_s)
        
        # All 3 agree
        mask_all = all_agree.unsqueeze(-1).expand(-1, num_classes)
        final_logits = torch.where(mask_all, (logits_s + logits_t + logits_e) / 3, final_logits)
        
        # S and T agree (not E)
        mask_st = only_st.unsqueeze(-1).expand(-1, num_classes)
        final_logits = torch.where(mask_st, (logits_s + logits_t) / 2, final_logits)
        
        # T and E agree (not S)
        mask_te = only_te.unsqueeze(-1).expand(-1, num_classes)
        final_logits = torch.where(mask_te, (logits_t + logits_e) / 2, final_logits)
        
        # S and E agree (not T)
        mask_se = only_se.unsqueeze(-1).expand(-1, num_classes)
        final_logits = torch.where(mask_se, (logits_s + logits_e) / 2, final_logits)
        
        # No consensus - arbitrator decides
        mask_arb = no_consensus.unsqueeze(-1).expand(-1, num_classes)
        final_logits = torch.where(mask_arb, arb_logits, final_logits)
        
        return final_logits
    
    
    def train_epoch(self, stage_name: str, optimizer, scheduler=None) -> dict:
        """Train one epoch with optional LR scheduler."""
        self.model.train()
        total_loss = 0
        total_correct = 0  # For the trained component
        total_samples = 0
        correct_s = correct_t = correct_e = correct_arb = correct_final = 0
        
        
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
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Stage-specific accuracy (what we're actually training)
            stage_logits = self.get_stage_logits(stage_name, analysis)
            total_correct += (stage_logits.argmax(-1) == y).sum().item()
            total_samples += y.size(0)
            total_loss += loss.item() * y.size(0)
            
            # Track all expert accuracies
            correct_s += (analysis['pred_s'] == y).sum().item()
            correct_t += (analysis['pred_t'] == y).sum().item()
            correct_e += (analysis['pred_e'] == y).sum().item()
            correct_arb += (analysis['arb_logits'].argmax(-1) == y).sum().item()
            correct_final += (final_logits.argmax(-1) == y).sum().item()
            
            pbar.set_postfix(
                loss=loss.item(),
                acc=total_correct/total_samples,
                S=correct_s/total_samples,
                T=correct_t/total_samples,
                E=correct_e/total_samples,
                lr=f"{current_lr:.6f}"
            )
        
        return {
            'loss': total_loss / total_samples,
            'stage_acc': total_correct / total_samples,
            'acc_s': correct_s / total_samples,
            'acc_t': correct_t / total_samples,
            'acc_e': correct_e / total_samples,
            'acc_arb': correct_arb / total_samples,
            'acc_final': correct_final / total_samples
        }
    
    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        total_correct = 0
        total_samples = 0
        correct_s = correct_t = correct_e = correct_arb = 0
        correct_majority = 0
        total_agree = 0
        n_batches = 0
        
        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            with autocast(enabled=self.use_fp16):
                _, analysis = self.model(x)
            
            # Compute v2 final logits
            final_logits = self._compute_final_logits_v2(analysis)
            
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
        print("COCOM v11: PATHFINDER-32 TRAINING (RWKV TACTICAL)")
        print("="*70)
        
        print("\nKey features in v11:")
        print("  1. RWKV replaces EMA in Tactical Expert")
        print("  2. RoPE enabled by default")
        print("  3. Modular architecture")
        print("  4. Majority voting for consensus")
        
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
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Training: {trainable}")
            
            betas = stage.get('betas', (0.9, 0.999))
            use_scheduler = stage.get('warmup', False) or stage_name == 'tactical'
            
            # RWKV-style AdamW with special eps
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr,
                betas=betas,
                eps=1e-8,
                weight_decay=0.001  # RWKV recommends 0.001
            )
            
            total_steps = len(self.train_loader) * epochs
            
            # OneCycleLR with warmup + cosine decay
            if use_scheduler:
                # Use CLI lr as max_lr (can be overriden via --tactical_lr etc)
                max_lr = lr
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    total_steps=total_steps,
                    pct_start=0.1,  # 10% warmup
                    anneal_strategy='cos',
                    div_factor=25,  # initial_lr = max_lr / 25
                    final_div_factor=100  # final_lr = initial_lr / 100
                )
                print(f"Using OneCycleLR: max_lr={max_lr:.6f}, warmup=10%, cosine decay")
            else:
                scheduler = None
                print(f"Using constant LR: {lr}")
            
            stage_best = 0
            epochs_without_improvement = 0
            
            for epoch in range(epochs):
                train_metrics = self.train_epoch(stage_name, optimizer, scheduler)
                test_metrics = self.evaluate()
                
                saved = ""
                if test_metrics['acc'] > stage_best:
                    stage_best = test_metrics['acc']
                    epochs_without_improvement = 0
                    torch.save(self.model.state_dict(), 
                              self.checkpoint_dir / f"cocom_v11_pathfinder_{stage_name}.pt")
                    saved = " 💾"
                    
                    if test_metrics['acc'] > best_acc:
                        best_acc = test_metrics['acc']
                        torch.save(self.model.state_dict(), 
                                  self.checkpoint_dir / "cocom_v11_pathfinder_best.pt")
                else:
                    epochs_without_improvement += 1
                
                print(f"  Epoch {epoch+1}/{epochs} | Test: {test_metrics['acc']:.4f} | "
                      f"S={test_metrics['acc_s']:.3f} T={test_metrics['acc_t']:.3f} "
                      f"E={test_metrics['acc_e']:.3f} Arb={test_metrics['acc_arb']:.3f}{saved}")
                
                self.history.append({
                    'stage': stage_name, 
                    'epoch': epoch + 1,
                    'train_stage_acc': train_metrics['stage_acc'],
                    **test_metrics
                })
                
                # Early stopping for full stage
                if stage_name == 'full' and epochs_without_improvement >= 5:
                    print(f"  Early stopping after {epoch+1} epochs")
                    break
            
            print(f"Stage {stage_name} best: {stage_best:.4f}")
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Best Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
        
        with open(self.checkpoint_dir / "cocom_v11_pathfinder_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return best_acc


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--fp32', action='store_true')
    # Epochs
    parser.add_argument('--tactical_epochs', type=int, default=8)
    parser.add_argument('--strategic_epochs', type=int, default=8)
    parser.add_argument('--exploratory_epochs', type=int, default=8)
    parser.add_argument('--full_epochs', type=int, default=20)
    # LR settings
    parser.add_argument('--tactical_lr', type=float, default=0.005)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--full_lr', type=float, default=5e-4)
    # Warmup
    parser.add_argument('--warmup_epochs', type=int, default=2, help='Warmup epochs for tactical stage')
    # Input configuration
    parser.add_argument('--input_type', type=str, default='image', 
                        choices=['image', 'image_conv1d', 'continuous'],
                        help='Input embedding type: image (2D patches), image_conv1d (1D conv), continuous (per-pixel linear)')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size for image mode (2, 4, 8, 16)')
    parser.add_argument('--embed_kernel_size', type=int, default=4, help='Kernel size for image_conv1d mode')
    # RoPE
    parser.add_argument('--use_rope', action='store_true', help='Enable Rotary Position Embedding')
    parser.add_argument('--rope_dims', type=int, default=2, choices=[1, 2], 
                        help='RoPE dimensions: 1 for 1D (ListOps), 2 for 2D (images)')
    # Note: RMSNorm, SwiGLU, and RWKV are enabled by default in COCOMv11
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Override stages (4 stages: tactical, strategic, exploratory, full)
    PathfinderTrainerV11.STAGES[0]['epochs'] = args.tactical_epochs
    PathfinderTrainerV11.STAGES[1]['epochs'] = args.strategic_epochs
    PathfinderTrainerV11.STAGES[2]['epochs'] = args.exploratory_epochs
    PathfinderTrainerV11.STAGES[3]['epochs'] = args.full_epochs
    
    PathfinderTrainerV11.STAGES[0]['lr'] = args.tactical_lr
    PathfinderTrainerV11.STAGES[1]['lr'] = args.base_lr
    PathfinderTrainerV11.STAGES[2]['lr'] = args.base_lr
    PathfinderTrainerV11.STAGES[3]['lr'] = args.full_lr
    
    # Store warmup epochs
    PathfinderTrainerV11.WARMUP_EPOCHS = args.warmup_epochs
    
    print(f"LR: tactical={args.tactical_lr}, base={args.base_lr}, full={args.full_lr}")
    print(f"Warmup: {args.warmup_epochs} epochs")
    
    trainer = PathfinderTrainerV11(
        batch_size=args.batch_size,
        device=device,
        use_fp16=not args.fp32,
        input_type=args.input_type,
        patch_size=args.patch_size,
        embed_kernel_size=args.embed_kernel_size,
        use_rope=args.use_rope,
        rope_dims=args.rope_dims,
    )
    
    trainer.run()



