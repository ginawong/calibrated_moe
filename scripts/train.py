#!/usr/bin/env python
"""Train one method on one dataset / seed, save checkpoint and metrics.

Example:
    python scripts/train.py \\
        --method robust_filtered --dataset cifar10h --seed 42 \\
        --output_dir experiment_results/cifar10h/results \\
        --experiment_name robust_filtered_s42 --save_checkpoint

Output: <output_dir>/<experiment_name>/{<method>_s<seed>.pt,  <experiment_name>_eta<eta>_w<warmup>.json}
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.multiprocessing
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Add repo's src/ to path so `calibrated_moe.*` imports work without install.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from calibrated_moe.calibration import find_optimal_temperature
from calibrated_moe.datasets import (
    AgreementDataset,
    CIFAR10_MEAN,
    CIFAR10_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_dataset,
)
from calibrated_moe.evaluation import (
    collect_logits,
    evaluate_by_difficulty,
    find_per_expert_temperatures,
)
from calibrated_moe.models import MoE, SingleExpert, get_backbone, set_seed
from calibrated_moe.training import (
    train_fgr,
    train_fgr_robust,
    train_robust,
    train_robust_filtered,
    train_single,
    train_vanilla,
)


torch.multiprocessing.set_sharing_strategy('file_system')

METHODS = ['single', 'vanilla', 'mocae', 'fgr', 'robust', 'robust_filtered', 'fgr_robust']


@dataclass
class Config:
    method: str = 'vanilla'
    seed: int = 42

    # Dataset
    dataset: str = 'cifar10h'
    data_dir: str = './data'
    num_classes: int = 10
    target_domain: str = ''

    # Difficulty thresholds for the {easy, hard} split used in evaluation.
    high_difficulty_threshold: float = 0.9  # samples with difficulty > this are "easy"
    low_difficulty_threshold: float = 0.7   # samples with difficulty < this are "hard"

    # Model
    backbone: str = 'resnet18'
    small_input: bool = True
    num_blocks: int = 3
    pretrained: bool = False
    num_experts: int = 4
    router_hidden_dim: int = 128

    # Training
    epochs: int = 50
    warmup_epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Method hyperparameters
    eta: float = 2.0
    fgr_rho: float = 0.05
    disagree_threshold: float = 0.01
    regret_threshold: float = 1e-6

    # I/O
    output_dir: str = 'experiment_results'
    experiment_name: str = ''
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_checkpoint: bool = False


def _build_loaders(config):
    """Build (train, val, test) DataLoaders + the per-test difficulty array."""
    trainset, testset, num_classes, difficulty = get_dataset(
        config.dataset, config.data_dir, target_domain=config.target_domain)
    config.num_classes = num_classes

    train_difficulty = np.ones(len(trainset))
    train_dataset = AgreementDataset(trainset, train_difficulty)

    n_train = len(train_dataset)
    n_val = int(0.1 * n_train)
    train_indices = list(range(n_train - n_val))
    val_indices = list(range(n_train - n_val, n_train))
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    test_dataset = AgreementDataset(testset, difficulty)

    # num_workers=2 matches the paper's training run (legacy cifar10h_pzx_shift.py).
    # PyTorch worker processes get their own RNG state, so changing this value
    # changes the augmentation order even with the same torch.manual_seed.
    common = dict(batch_size=config.batch_size, num_workers=2)
    train_loader = DataLoader(train_subset, shuffle=True, **common)
    val_loader = DataLoader(val_subset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, val_loader, test_loader, difficulty


def _build_model(config):
    backbone = get_backbone(config.backbone, small_input=config.small_input,
                            num_blocks=config.num_blocks, pretrained=config.pretrained)
    if config.method == 'single':
        return SingleExpert(num_classes=config.num_classes, backbone=backbone).to(config.device)
    return MoE(num_experts=config.num_experts, num_classes=config.num_classes,
               hidden_dim=config.router_hidden_dim, backbone=backbone).to(config.device)


def _dataset_norm(config):
    if config.dataset == 'cifar10h':
        return CIFAR10_MEAN, CIFAR10_STD
    return IMAGENET_MEAN, IMAGENET_STD


def _run_epoch(config, model, train_loader, optimizer, epoch):
    """Run one training epoch and return mean loss."""
    method = config.method
    if method == 'single':
        return train_single(model, train_loader, optimizer, config.device)
    # `mocae` trains identically to `vanilla`; per-expert TS is applied post-hoc.
    if method in ('vanilla', 'mocae'):
        return train_vanilla(model, train_loader, optimizer, config.device)

    # Methods below use ERM during warmup, then switch to their objective.
    if epoch < config.warmup_epochs:
        return train_vanilla(model, train_loader, optimizer, config.device)

    if method == 'robust':
        return train_robust(model, train_loader, optimizer, config.device, config.eta)
    if method == 'robust_filtered':
        return train_robust_filtered(model, train_loader, optimizer, config.device,
                                     config.eta,
                                     disagree_threshold=config.disagree_threshold,
                                     regret_threshold=config.regret_threshold)
    if method == 'fgr':
        mean, std = _dataset_norm(config)
        return train_fgr(model, train_loader, optimizer, config.device,
                         rho=config.fgr_rho, norm_mean=mean, norm_std=std)
    if method == 'fgr_robust':
        mean, std = _dataset_norm(config)
        return train_fgr_robust(model, train_loader, optimizer, config.device,
                                config.eta, rho=config.fgr_rho,
                                norm_mean=mean, norm_std=std)
    raise ValueError(f"Unknown method: {method}")


def train_one(config):
    """Train one (method, seed, dataset) combination, save outputs."""
    set_seed(config.seed)
    device = config.device

    print(f"{'=' * 70}\nTraining {config.method} on {config.dataset} (seed {config.seed})\n{'=' * 70}", flush=True)

    # Match the legacy paper-training order: build the model BEFORE the loaders
    # so that the RNG state when weights initialise is identical to the legacy
    # script's. (Building loaders consumes a small amount of RNG via
    # torchvision dataset construction.)
    model = _build_model(config)
    train_loader, val_loader, test_loader, difficulty = _build_loaders(config)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)

    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    is_moe = config.method != 'single'

    for epoch in range(config.epochs):
        t0 = time.perf_counter()
        loss = _run_epoch(config, model, train_loader, optimizer, epoch)
        scheduler.step()
        elapsed = time.perf_counter() - t0

        if (epoch + 1) % 10 == 0:
            m = evaluate_by_difficulty(
                model, test_loader, device, difficulty,
                hard_threshold=config.low_difficulty_threshold,
                easy_threshold=config.high_difficulty_threshold,
                is_moe=is_moe)
            print(f"  epoch {epoch+1:3d}: loss={loss:.4f}  acc={m['accuracy']:.3f}  "
                  f"ece={m['ece']:.3f}  hard_ece={m['hard_ece']:.3f}  [{elapsed:.1f}s]", flush=True)
        else:
            print(f"  epoch {epoch+1:3d}: loss={loss:.4f}  [{elapsed:.1f}s]", flush=True)

    # MoCaE: post-hoc per-expert temperature scaling on the held-out val split.
    # This IS the MoCaE method — applied to a vanilla-trained MoE at eval time.
    per_expert_temps = None
    if config.method == 'mocae':
        print("Fitting per-expert temperatures on the held-out validation split...", flush=True)
        per_expert_temps = find_per_expert_temperatures(model, val_loader, device)

    # Final eval: base + temperature-scaled. For MoCaE, both rows include per-expert TS;
    # `temp_scaled` additionally applies a scalar aggregate temperature on top.
    base = evaluate_by_difficulty(
        model, test_loader, device, difficulty,
        hard_threshold=config.low_difficulty_threshold,
        easy_threshold=config.high_difficulty_threshold,
        is_moe=is_moe, per_expert_temperatures=per_expert_temps)

    print("Fitting aggregate temperature on the held-out validation split...", flush=True)
    logits, labels = collect_logits(
        model, val_loader, device, is_moe=is_moe,
        per_expert_temperatures=per_expert_temps)
    temperature = find_optimal_temperature(logits, labels)

    temp_scaled = evaluate_by_difficulty(
        model, test_loader, device, difficulty,
        hard_threshold=config.low_difficulty_threshold,
        easy_threshold=config.high_difficulty_threshold,
        is_moe=is_moe, temperature=temperature,
        per_expert_temperatures=per_expert_temps)

    results = {
        'config': asdict(config),
        'method': config.method,
        'base': base,
        'temp_scaled': temp_scaled,
        'temperature': temperature,
    }
    if per_expert_temps is not None:
        results['per_expert_temperatures'] = per_expert_temps

    exp_dir = (os.path.join(config.output_dir, config.experiment_name)
               if config.experiment_name else config.output_dir)
    os.makedirs(exp_dir, exist_ok=True)

    json_path = os.path.join(exp_dir,
        f"{config.experiment_name or config.method}_eta{config.eta}_w{config.warmup_epochs}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics: {json_path}")

    if config.save_checkpoint:
        ckpt_path = os.path.join(exp_dir, f"{config.method}_s{config.seed}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': asdict(config),
            'results': results,
            'method': config.method,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    print(f"\nFinal: ECE={base['ece']:.4f}  ECE+TS={temp_scaled['ece']:.4f}  "
          f"Hard ECE={base['hard_ece']:.4f}  Hard ECE+TS={temp_scaled['hard_ece']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--method', choices=METHODS, default='vanilla',
                        help=f'Training method ({", ".join(METHODS)}).')
    parser.add_argument('--seed', type=int, default=42)

    # Dataset
    parser.add_argument('--dataset', default='cifar10h',
                        choices=['cifar10h', 'pacs', 'office_home', 'civilcomments'])
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--target_domain', default='',
                        help='For domain-shift datasets: held-out domain name.')

    # Difficulty thresholds
    parser.add_argument('--high_difficulty_threshold', type=float, default=0.9)
    parser.add_argument('--low_difficulty_threshold', type=float, default=0.7)

    # Model
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--small_input', type=lambda x: x.lower() != 'false', default=True,
                        help='True for 32x32 (CIFAR), False for 224x224 (PACS, etc.).')
    parser.add_argument('--num_blocks', type=int, default=3,
                        help='Number of ResNet residual stages (1-4).')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use ImageNet pretrained backbone (needed for PACS).')
    parser.add_argument('--num_experts', type=int, default=4)

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--warmup', dest='warmup_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Method hyperparameters
    parser.add_argument('--eta', type=float, default=2.0,
                        help='Maximum-entropy adversarial reweighting temperature.')
    parser.add_argument('--fgr_rho', type=float, default=0.05)
    parser.add_argument('--disagree_threshold', type=float, default=0.01,
                        help='Robust Filtered: expert-disagreement threshold for routing relevance.')
    parser.add_argument('--regret_threshold', type=float, default=1e-6,
                        help='Robust Filtered: mixture-regret threshold for routing relevance.')

    # I/O
    parser.add_argument('--output_dir', default='experiment_results')
    parser.add_argument('--experiment_name', default='')
    parser.add_argument('--save_checkpoint', action='store_true')
    args = parser.parse_args()

    cfg = Config(method=args.method, seed=args.seed,
                 dataset=args.dataset, data_dir=args.data_dir, target_domain=args.target_domain,
                 high_difficulty_threshold=args.high_difficulty_threshold,
                 low_difficulty_threshold=args.low_difficulty_threshold,
                 backbone=args.backbone, small_input=args.small_input,
                 num_blocks=args.num_blocks, pretrained=args.pretrained,
                 num_experts=args.num_experts,
                 epochs=args.epochs, warmup_epochs=args.warmup_epochs,
                 batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
                 eta=args.eta, fgr_rho=args.fgr_rho,
                 disagree_threshold=args.disagree_threshold,
                 regret_threshold=args.regret_threshold,
                 output_dir=args.output_dir,
                 experiment_name=args.experiment_name,
                 save_checkpoint=args.save_checkpoint)
    train_one(cfg)


if __name__ == '__main__':
    main()
