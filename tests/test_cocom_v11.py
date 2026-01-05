"""
COCOM v11: Comprehensive Evaluation

Полная проверка модели с метриками:
1. Общая точность системы
2. Точность каждой головы (S, T, E, Arb)
3. Похожесть голов друг на друга (agreement, cosine similarity)
4. Анализ Arbitrator (mix_weight, confidence)
5. Confusion matrix и per-class accuracy
6. Сравнение: когда эксперты согласны vs не согласны
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

from cocom_11 import COCOMv11


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def compute_agreement(pred_a: torch.Tensor, pred_b: torch.Tensor) -> float:
    """Compute agreement rate between two prediction tensors."""
    return (pred_a == pred_b).float().mean().item()


class V11Evaluator:
    def __init__(
        self,
        checkpoint_path: str,
        data_dir: str = "./lra_data",
        device: str = 'cuda',
        batch_size: int = 4
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        
        # Load data
        data_path = Path(data_dir) / "listops_kaggle"
        self.test_sequences = np.load(data_path / "test_sequences.npy", mmap_mode='r')
        self.test_labels = np.load(data_path / "test_labels.npy", mmap_mode='r')
        print(f"Test samples: {len(self.test_sequences)}")
        
        # Load checkpoint
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            print(f"❌ Checkpoint not found: {checkpoint}")
            return
        
        state_dict = torch.load(checkpoint, map_location=self.device)
        
        # Extract model dimensions from checkpoint
        seq_len = state_dict['pos_embed'].shape[1]
        vocab_size = state_dict['input_embed.weight'].shape[0]
        d_model = state_dict['pos_embed'].shape[2]
        
        print(f"Model params from checkpoint:")
        print(f"  seq_len={seq_len}, vocab_size={vocab_size}, d_model={d_model}")
        
        # Verify data compatibility
        data_seq_len = self.test_sequences.shape[1]
        if data_seq_len > seq_len:
            print(f"⚠️ Data seq_len ({data_seq_len}) > model seq_len ({seq_len})")
            print(f"   Truncating sequences to {seq_len}")
            self.test_sequences = self.test_sequences[:, :seq_len]
        
        self.vocab_size = vocab_size
        self.model = COCOMv11(
            d_model=d_model,
            n_heads=8,
            num_classes=10,
            seq_len=seq_len,
            input_type='tokens',
            vocab_size=vocab_size,
            use_rope=True,
            rope_dims=1,
        ).to(self.device)
        
        self.model.load_state_dict(state_dict)
        print(f"✅ Loaded model from {checkpoint}")
        
        self.model.eval()
        
        params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {params:,}")
    
    def run_evaluation(self) -> dict:
        """Run complete evaluation and return metrics."""
        print("\n" + "="*70)
        print("COCOM v11: COMPREHENSIVE EVALUATION")
        print("="*70)
        
        # Collect predictions and metrics
        all_preds = {'final': [], 's': [], 't': [], 'e': [], 'arb': []}
        all_labels = []
        all_logits = {'s': [], 't': [], 'e': [], 'arb': []}
        
        arb_analysis_sums = defaultdict(float)
        agreement_rates = []
        n_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(self.test_sequences), self.batch_size):
                batch_x = torch.tensor(
                    self.test_sequences[i:i+self.batch_size].copy(), 
                    dtype=torch.long
                ).to(self.device)
                batch_x = batch_x.clamp(0, self.vocab_size - 1)
                
                batch_y = torch.tensor(
                    self.test_labels[i:i+self.batch_size].copy(), 
                    dtype=torch.long
                ).to(self.device)
                
                with autocast():
                    logits, analysis = self.model(batch_x)
                
                # Store predictions
                all_preds['final'].append(logits.argmax(-1).cpu())
                all_preds['s'].append(analysis['pred_s'].cpu())
                all_preds['t'].append(analysis['pred_t'].cpu())
                all_preds['e'].append(analysis['pred_e'].cpu())
                all_preds['arb'].append(analysis['arb_logits'].argmax(-1).cpu())
                all_labels.append(batch_y.cpu())
                
                # Store logits
                all_logits['s'].append(analysis['logits_s'].cpu())
                all_logits['t'].append(analysis['logits_t'].cpu())
                all_logits['e'].append(analysis['logits_e'].cpu())
                all_logits['arb'].append(analysis['arb_logits'].cpu())
                
                # Arbitrator analysis
                agreement_rates.append(analysis['agreement_rate'])
                arb_analysis_sums['confidence_s'] += analysis['confidence_s']
                arb_analysis_sums['confidence_t'] += analysis['confidence_t']
                arb_analysis_sums['confidence_e'] += analysis['confidence_e']
                arb_analysis_sums['mix_weight'] += analysis['mix_weight']
                n_batches += 1
                
                if n_batches % 100 == 0:
                    print(f"  Processed {n_batches * self.batch_size} samples...")
        
        # Concatenate all
        for key in all_preds:
            all_preds[key] = torch.cat(all_preds[key])
        for key in all_logits:
            all_logits[key] = torch.cat(all_logits[key])
        all_labels = torch.cat(all_labels)
        
        # Compute metrics
        metrics = {}
        
        # 1. Accuracy per head
        print("\n" + "-"*70)
        print("1. ACCURACY PER HEAD")
        print("-"*70)
        for key in ['final', 's', 't', 'e', 'arb']:
            acc = (all_preds[key] == all_labels).float().mean().item()
            metrics[f'acc_{key}'] = acc
            name = {'final': 'Final (consensus)', 's': 'Strategic', 't': 'Tactical', 
                    'e': 'Exploratory', 'arb': 'Arbitrator'}[key]
            print(f"  {name:20s}: {acc*100:5.2f}%")
        
        # 2. Head agreement
        print("\n" + "-"*70)
        print("2. HEAD AGREEMENT (pairwise)")
        print("-"*70)
        pairs = [('s', 't'), ('s', 'e'), ('t', 'e'), ('s', 'arb'), ('t', 'arb'), ('e', 'arb')]
        for a, b in pairs:
            agree = compute_agreement(all_preds[a], all_preds[b])
            metrics[f'agree_{a}_{b}'] = agree
            name_a = {'s': 'Strategic', 't': 'Tactical', 'e': 'Exploratory', 'arb': 'Arbitrator'}[a]
            name_b = {'s': 'Strategic', 't': 'Tactical', 'e': 'Exploratory', 'arb': 'Arbitrator'}[b]
            print(f"  {name_a:12s} vs {name_b:12s}: {agree*100:5.2f}%")
        
        # 3. Logit similarity
        print("\n" + "-"*70)
        print("3. LOGIT SIMILARITY (cosine)")
        print("-"*70)
        for a, b in [('s', 't'), ('s', 'e'), ('t', 'e')]:
            sim = cosine_similarity(all_logits[a], all_logits[b])
            metrics[f'sim_{a}_{b}'] = sim
            name_a = {'s': 'Strategic', 't': 'Tactical', 'e': 'Exploratory'}[a]
            name_b = {'s': 'Strategic', 't': 'Tactical', 'e': 'Exploratory'}[b]
            print(f"  {name_a:12s} vs {name_b:12s}: {sim:6.3f}")
        
        # 4. Arbitrator analysis
        print("\n" + "-"*70)
        print("4. ARBITRATOR ANALYSIS")
        print("-"*70)
        metrics['avg_agreement'] = np.mean(agreement_rates)
        metrics['avg_confidence_s'] = arb_analysis_sums['confidence_s'] / n_batches
        metrics['avg_confidence_t'] = arb_analysis_sums['confidence_t'] / n_batches
        metrics['avg_confidence_e'] = arb_analysis_sums['confidence_e'] / n_batches
        metrics['avg_mix_weight'] = arb_analysis_sums['mix_weight'] / n_batches
        
        print(f"  Expert agreement rate:   {metrics['avg_agreement']*100:5.2f}%")
        print(f"  Confidence - Strategic:  {metrics['avg_confidence_s']:6.3f}")
        print(f"  Confidence - Tactical:   {metrics['avg_confidence_t']:6.3f}")
        print(f"  Confidence - Exploratory: {metrics['avg_confidence_e']:6.3f}")
        print(f"  Mix weight (0=own, 1=experts): {metrics['avg_mix_weight']:6.3f}")
        
        # 5. Consensus analysis
        print("\n" + "-"*70)
        print("5. CONSENSUS ANALYSIS")
        print("-"*70)
        
        all_agree = (all_preds['s'] == all_preds['t']) & (all_preds['t'] == all_preds['e'])
        any_agree = ((all_preds['s'] == all_preds['t']) | 
                     (all_preds['t'] == all_preds['e']) | 
                     (all_preds['s'] == all_preds['e']))
        no_agree = ~any_agree
        
        n_all_agree = all_agree.sum().item()
        n_any_agree = any_agree.sum().item()
        n_no_agree = no_agree.sum().item()
        
        print(f"  All 3 experts agree:    {n_all_agree:5d} ({n_all_agree/len(all_labels)*100:5.2f}%)")
        print(f"  At least 2 agree:       {n_any_agree:5d} ({n_any_agree/len(all_labels)*100:5.2f}%)")
        print(f"  No agreement:           {n_no_agree:5d} ({n_no_agree/len(all_labels)*100:5.2f}%)")
        
        if n_all_agree > 0:
            acc_all_agree = (all_preds['final'][all_agree] == all_labels[all_agree]).float().mean().item()
            metrics['acc_when_all_agree'] = acc_all_agree
            print(f"\n  Accuracy when all agree:  {acc_all_agree*100:5.2f}%")
        
        if n_no_agree > 0:
            acc_no_agree = (all_preds['final'][no_agree] == all_labels[no_agree]).float().mean().item()
            metrics['acc_when_no_agree'] = acc_no_agree
            print(f"  Accuracy when no agree:   {acc_no_agree*100:5.2f}%")
        
        # 6. Per-class accuracy
        print("\n" + "-"*70)
        print("6. PER-CLASS ACCURACY")
        print("-"*70)
        for c in range(10):
            mask = all_labels == c
            if mask.sum() > 0:
                class_acc = (all_preds['final'][mask] == all_labels[mask]).float().mean().item()
                metrics[f'acc_class_{c}'] = class_acc
                print(f"  Class {c}: {class_acc*100:5.2f}% ({mask.sum().item()} samples)")
        
        # 7. Expert-only test
        print("\n" + "-"*70)
        print("7. EXPERT-ONLY TEST (avg logits, no arbitrator)")
        print("-"*70)
        avg_logits = (all_logits['s'] + all_logits['t'] + all_logits['e']) / 3
        avg_preds = avg_logits.argmax(-1)
        acc_avg = (avg_preds == all_labels).float().mean().item()
        metrics['acc_avg_experts'] = acc_avg
        print(f"  Simple average of 3 experts: {acc_avg*100:5.2f}%")
        print(f"  vs Full system (with Arb):   {metrics['acc_final']*100:5.2f}%")
        print(f"  Arbitrator adds: {(metrics['acc_final'] - acc_avg)*100:+5.2f}%")
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        best_head = max(['s', 't', 'e'], key=lambda k: metrics[f'acc_{k}'])
        best_head_name = {'s': 'Strategic', 't': 'Tactical', 'e': 'Exploratory'}[best_head]
        print(f"Best head:      {best_head_name} ({metrics[f'acc_{best_head}']*100:.2f}%)")
        print(f"Final accuracy: {metrics['acc_final']*100:.2f}%")
        print(f"Arbitrator:     {metrics['acc_arb']*100:.2f}%")
        
        return metrics
    
    def save_report(self, metrics: dict, output_path: str = "cocom_v11_report.json"):
        """Save metrics to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                        default='./checkpoints/cocom_v11_listops_full.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--output', type=str, default='./cocom_v11_report.json')
    args = parser.parse_args()
    
    evaluator = V11Evaluator(
        checkpoint_path=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size
    )
    
    metrics = evaluator.run_evaluation()
    evaluator.save_report(metrics, args.output)

