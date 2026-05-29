#!/usr/bin/env python
"""Derive MoCaE results from existing Vanilla MoE checkpoints.

MoCaE is `vanilla` training followed by post-hoc per-expert temperature scaling
applied at evaluation time. Because the underlying model is identical to a
vanilla-trained MoE with the same seed, this script avoids re-training: it
loads existing `vanilla_s{seed}.pt` checkpoints and writes the corresponding
`mocae_s{seed}/` result JSONs.

For each seed found, this script:
  1. Loads `vanilla_s{seed}.pt`
  2. Reconstructs the val split (last 10% of train, same convention as train.py)
  3. Fits per-expert temperatures on val
  4. Evaluates on test with per-expert TS  ->  `base`
  5. Fits a scalar aggregate temperature on per-expert-TS'd val logits
  6. Evaluates on test with both TSs       ->  `temp_scaled`
  7. Saves <results_dir>/mocae_s{seed}/mocae_s{seed}_eta{eta}_w{warmup}.json

Equivalent to `train.py --method mocae` but reuses the vanilla checkpoint
instead of retraining.

Usage:
    python scripts/derive_mocae.py --dataset cifar10h
    python scripts/derive_mocae.py --dataset civilcomments \
        --data_dir <path>/civilcomments
    python scripts/derive_mocae.py --dataset pacs --target_domain sketch \
        --data_dir <path>/domainbed_datasets
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from calibrated_moe.calibration import find_optimal_temperature
from calibrated_moe.datasets import AgreementDataset, get_dataset
from calibrated_moe.evaluation import (
    collect_logits,
    evaluate_by_difficulty,
    find_per_expert_temperatures,
)
from calibrated_moe.models import MoE, get_backbone


# Defaults for legacy CIFAR-10H checkpoints whose config field is sparse.
_CIFAR10H_DEFAULTS = dict(
    dataset='cifar10h', target_domain='', backbone='resnet18',
    small_input=True, num_blocks=3, pretrained=False, num_classes=10,
)


def _build_model(ckpt_config, device):
    c = ckpt_config
    backbone_name = c.get('backbone') or _CIFAR10H_DEFAULTS['backbone']
    small_input = c.get('small_input')
    if small_input is None:
        small_input = _CIFAR10H_DEFAULTS['small_input']
    num_blocks = c.get('num_blocks') or _CIFAR10H_DEFAULTS['num_blocks']
    num_classes = c.get('num_classes') or _CIFAR10H_DEFAULTS['num_classes']
    backbone = get_backbone(backbone_name, small_input=small_input,
                            num_blocks=num_blocks, pretrained=False)
    return MoE(num_experts=c.get('num_experts', 4),
               num_classes=num_classes,
               hidden_dim=c.get('router_hidden_dim', 128),
               backbone=backbone).to(device)


def _build_loaders(args):
    dataset_kwargs = {}
    if args.target_domain:
        dataset_kwargs['target_domain'] = args.target_domain
    trainset, testset, _, difficulty = get_dataset(
        args.dataset, args.data_dir, **dataset_kwargs)

    train_difficulty = np.ones(len(trainset))
    train_dataset = AgreementDataset(trainset, train_difficulty)
    n_train = len(train_dataset)
    n_val = int(0.1 * n_train)
    val_indices = list(range(n_train - n_val, n_train))
    val_subset = Subset(train_dataset, val_indices)
    test_dataset = AgreementDataset(testset, difficulty)

    common = dict(batch_size=args.batch_size, num_workers=2)
    val_loader = DataLoader(val_subset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return val_loader, test_loader, difficulty


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True,
                        choices=['cifar10h', 'pacs', 'office_home', 'civilcomments'])
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--target_domain', default='',
                        help='For PACS / Office-Home: held-out target domain.')
    parser.add_argument('--results_dir', default='',
                        help='Default: experiment_results/{dataset[_target_domain]}/results')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hard_threshold', type=float, default=0.7)
    parser.add_argument('--easy_threshold', type=float, default=0.9)
    args = parser.parse_args()

    if args.dataset in ('pacs', 'office_home') and not args.target_domain:
        raise SystemExit('--target_domain is required for pacs / office_home')

    if not args.results_dir:
        # PACS uses short domain names in its result-dir suffixes (`pacs_art`,
        # not `pacs_art_painting`).
        suffix = args.target_domain.split('_')[0] if args.target_domain else ''
        suffix = f'_{suffix}' if suffix else ''
        args.results_dir = f'experiment_results/{args.dataset}{suffix}/results'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Building val/test loaders for {args.dataset}{' (' + args.target_domain + ')' if args.target_domain else ''}...", flush=True)
    val_loader, test_loader, difficulty = _build_loaders(args)

    vanilla_pts = sorted(glob.glob(f'{args.results_dir}/vanilla_s*/vanilla_s*.pt'))
    if not vanilla_pts:
        raise SystemExit(f'No vanilla checkpoints found at {args.results_dir}/vanilla_s*/vanilla_s*.pt')

    print(f"Found {len(vanilla_pts)} vanilla checkpoints.", flush=True)

    for ckpt_path in vanilla_pts:
        seed_str = os.path.basename(ckpt_path).split('_s')[1].replace('.pt', '')
        seed = int(seed_str)
        print(f"\n=== mocae_s{seed} ===", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = _build_model(ckpt['config'], device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        per_expert_temps = find_per_expert_temperatures(model, val_loader, device)
        print(f"  Per-expert temperatures: {[round(t, 3) for t in per_expert_temps]}", flush=True)

        base = evaluate_by_difficulty(
            model, test_loader, device, difficulty,
            hard_threshold=args.hard_threshold,
            easy_threshold=args.easy_threshold,
            is_moe=True, per_expert_temperatures=per_expert_temps)

        logits, labels = collect_logits(
            model, val_loader, device, is_moe=True,
            per_expert_temperatures=per_expert_temps)
        temperature = find_optimal_temperature(logits, labels)

        temp_scaled = evaluate_by_difficulty(
            model, test_loader, device, difficulty,
            hard_threshold=args.hard_threshold,
            easy_threshold=args.easy_threshold,
            is_moe=True, temperature=temperature,
            per_expert_temperatures=per_expert_temps)

        new_config = {**ckpt['config'], 'method': 'mocae', 'experiment_name': f'mocae_s{seed}'}
        results = {
            'config': new_config,
            'method': 'mocae',
            'base': base,
            'temp_scaled': temp_scaled,
            'temperature': temperature,
            'per_expert_temperatures': per_expert_temps,
        }
        eta = ckpt['config'].get('eta', 2.0)
        warmup = ckpt['config'].get('warmup_epochs', 20)
        out_dir = f'{args.results_dir}/mocae_s{seed}'
        os.makedirs(out_dir, exist_ok=True)
        out_path = f'{out_dir}/mocae_s{seed}_eta{eta}_w{warmup}.json'
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {out_path}")
        print(f"  Base   Acc={base['accuracy']:.4f}  Hard Acc={base['hard_acc']:.4f}  "
              f"ECE={base['ece']:.4f}  Hard ECE={base['hard_ece']:.4f}")
        print(f"  TS     ECE={temp_scaled['ece']:.4f}  Hard ECE={temp_scaled['hard_ece']:.4f}  "
              f"(T={temperature:.3f})")


if __name__ == '__main__':
    main()
