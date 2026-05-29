#!/usr/bin/env python
"""Compare three post-hoc temperature-scaling strategies on saved MoE checkpoints:

  - none:        raw mixture probabilities
  - mixture:     one scalar temperature fit on the aggregate mixture logits
  - per-expert:  fit one temperature per expert, scale before mixing

Reports mean ± std accuracy / ECE / hard-ECE across seeds, per method.

Usage:
    python scripts/eval_temperature_scaling.py                   # CIFAR-10H
    python scripts/eval_temperature_scaling.py --dataset pacs --target_domain sketch
    python scripts/eval_temperature_scaling.py --dataset civilcomments
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from calibrated_moe.calibration import find_optimal_temperature
from calibrated_moe.datasets import AgreementDataset, get_dataset
from calibrated_moe.evaluation import (
    collect_logits,
    evaluate_by_difficulty,
    find_per_expert_temperatures,
)
from calibrated_moe.models import MoE, get_backbone


METHODS = ['vanilla', 'robust', 'robust_filtered']


def load_model(ckpt_path, device):
    """Reconstruct an MoE from a saved checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt['config']
    backbone = get_backbone(c['backbone'], small_input=c['small_input'],
                            num_blocks=c['num_blocks'], pretrained=False)
    model = MoE(num_experts=c['num_experts'], num_classes=c['num_classes'],
                hidden_dim=c.get('router_hidden_dim', 128), backbone=backbone).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, c


def eval_method(method, results_dir, val_loader, test_loader, difficulty,
                hard_threshold, easy_threshold, device):
    pattern = f"{results_dir}/{method}_s*/{method}_s*.pt"
    checkpoints = sorted(glob.glob(pattern))
    if not checkpoints:
        print(f"  No checkpoints found for {method}")
        return []

    results = []
    common = dict(hard_threshold=hard_threshold, easy_threshold=easy_threshold, is_moe=True)
    for ckpt_path in checkpoints:
        seed = int(ckpt_path.split(f"{method}_s")[-1].replace('.pt', ''))
        model, _ = load_model(ckpt_path, device)

        base = evaluate_by_difficulty(model, test_loader, device, difficulty, **common)

        logits, labels = collect_logits(model, val_loader, device, is_moe=True)
        mix_temp = find_optimal_temperature(logits, labels)
        mix_ts = evaluate_by_difficulty(model, test_loader, device, difficulty,
                                        temperature=mix_temp, **common)

        expert_temps = find_per_expert_temperatures(model, val_loader, device)
        per_ts = evaluate_by_difficulty(model, test_loader, device, difficulty,
                                        per_expert_temperatures=expert_temps, **common)

        print(f"  s{seed}: base ECE={base['ece']:.4f}/{base['hard_ece']:.4f}  "
              f"mix_ts={mix_ts['ece']:.4f}/{mix_ts['hard_ece']:.4f}  "
              f"per_ts={per_ts['ece']:.4f}/{per_ts['hard_ece']:.4f}  "
              f"T=[{', '.join(f'{t:.2f}' for t in expert_temps)}]")

        results.append({
            'seed': seed,
            'base': base,
            'mixture_ts': {**mix_ts, 'temperature': mix_temp},
            'per_expert_ts': {**per_ts, 'expert_temperatures': expert_temps},
        })
    return results


def print_summary(method, results):
    for key, label in [('base', 'No TS'), ('mixture_ts', 'Mix TS'), ('per_expert_ts', 'PerExp TS')]:
        accs = [r[key]['accuracy'] for r in results]
        eces = [r[key]['ece'] for r in results]
        hard_eces = [r[key]['hard_ece'] for r in results]
        print(f"  {method:18s} {label:10s}: "
              f"Acc={np.mean(accs):.4f}±{np.std(accs):.4f}  "
              f"ECE={np.mean(eces):.5f}±{np.std(eces):.5f}  "
              f"Hard ECE={np.mean(hard_eces):.5f}±{np.std(hard_eces):.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10h',
                        choices=['cifar10h', 'pacs', 'civilcomments'])
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--target_domain', default='',
                        help='For PACS: held-out domain. Defaults to sketch.')
    parser.add_argument('--results_dir', default='',
                        help='Defaults to experiment_results/<dataset>/results.')
    parser.add_argument('--output_path', default='',
                        help='Defaults to experiment_results/<dataset>/per_expert_ts_all_methods.json.')
    parser.add_argument('--hard_threshold', type=float, default=0.7,
                        help='Sample is "hard" if difficulty < this value.')
    parser.add_argument('--easy_threshold', type=float, default=0.9,
                        help='Sample is "easy" if difficulty > this value.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'pacs' and not args.target_domain:
        args.target_domain = 'sketch'

    trainset, testset, _, difficulty = get_dataset(
        args.dataset, args.data_dir, target_domain=args.target_domain)

    # Held-out validation split (10% of train, seed=42).
    n_train = len(trainset)
    n_val = int(0.1 * n_train)
    gen = torch.Generator().manual_seed(42)
    val_indices = torch.randperm(n_train, generator=gen).tolist()[:n_val]

    val_ds = AgreementDataset(trainset, np.ones(n_train), indices=val_indices)
    test_ds = AgreementDataset(testset, difficulty)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    print(f"Dataset: {args.dataset}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    print("=" * 80)

    results_dir = args.results_dir or f"experiment_results/{args.dataset}/results"

    all_results = {}
    for method in METHODS:
        print(f"\n--- {method} ---")
        results = eval_method(method, results_dir, val_loader, test_loader, difficulty,
                              args.hard_threshold, args.easy_threshold, device)
        if results:
            all_results[method] = results

    print("\n" + "=" * 80)
    print(f"SUMMARY — {args.dataset} (mean ± std over seeds)")
    print("=" * 80)
    for method, results in all_results.items():
        print_summary(method, results)
        print()

    out_path = args.output_path or f"experiment_results/{args.dataset}/per_expert_ts_all_methods.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
