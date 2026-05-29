#!/usr/bin/env python
"""Reliability diagrams + confidence-vs-agreement + per-group breakdown.

Produces a 4+3 grid (Single Expert, Vanilla MoE, MoCaE, FGR on top; Robust MoE,
Robust Filtered, FGR+Robust on bottom) for each dataset, with and without
post-hoc temperature scaling.

Outputs land under <output_dir>/{cifar10h,civilcomments,pacs}/.

Usage:
    python scripts/figures_reliability.py                    # writes to ./figures/
    python scripts/figures_reliability.py --output_dir PAPER_RESULTS/figures
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from calibrated_moe.calibration import apply_temperature_scaling
from calibrated_moe.datasets import get_dataset
from calibrated_moe.evaluation import forward_with_per_expert_ts, load_mocae_temperatures
from calibrated_moe.models import MoE, SingleExpert, get_backbone

DEFAULT_OUTPUT_DIR = 'figures'
CC_CACHE_PATH = '.cache/civilcomments_predictions.pt'


def figure_path(output_name, base_dir):
    """Route a figure filename to the dataset subdirectory of base_dir."""
    name = os.path.basename(output_name)
    if name.startswith('cifar10h_') or name.startswith('baselines_cifar10h_'):
        sub = 'cifar10h'
    elif name.startswith('civilcomments_') or name.startswith('baselines_civilcomments_'):
        sub = 'civilcomments'
    elif name.startswith('pacs_') or name.startswith('baselines_pacs_'):
        sub = 'pacs'
    else:
        sub = ''
    out_dir = os.path.join(base_dir, sub) if sub else base_dir
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, name)


METHODS = [
    # Row 1: baselines (4)
    'single', 'vanilla', 'mocae', 'fgr',
    # Row 2: robust methods (3)
    'robust', 'robust_filtered', 'fgr_robust',
]

COLORS = {
    'single':          '#95a5a6',
    'vanilla':         '#e74c3c',
    'mocae':           '#f39c12',
    'robust':          '#3498db',
    'robust_filtered': '#9b59b6',
    'fgr':             '#d35400',
    'fgr_robust':      '#1abc9c',
}

LABELS = {
    'single':          'Single Expert',
    'vanilla':         'Vanilla MoE',
    'mocae':           'MoCaE',
    'robust':          'Robust MoE',
    'robust_filtered': 'Robust Filtered',
    'fgr':             'FGR',
    'fgr_robust':      'FGR + Robust',
}

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 100,
})

SEEDS = [42, 43, 44, 45, 46]


# ==========================================
# Model loading & inference
# ==========================================
def load_model(checkpoint_path, device, is_moe=True):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get('config', {})

    backbone = get_backbone(
        cfg.get('backbone', 'resnet18'),
        small_input=cfg.get('small_input', True),
        num_blocks=cfg.get('num_blocks', 3),
        pretrained=False,
    )
    num_classes = cfg.get('num_classes', 10)

    if not is_moe:
        model = SingleExpert(num_classes=num_classes, backbone=backbone)
    else:
        model = MoE(
            num_experts=cfg.get('num_experts', 4),
            num_classes=num_classes,
            hidden_dim=cfg.get('router_hidden_dim', 128),
            backbone=backbone,
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    return model


def get_predictions(model, loader, device, per_expert_temps=None):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch_x, labels in loader:
            if isinstance(batch_x, dict):
                batch_x = {k: v.to(device) for k, v in batch_x.items()}
            else:
                batch_x = batch_x.to(device)
            if per_expert_temps is not None:
                probs, _ = forward_with_per_expert_ts(model, batch_x, per_expert_temps)
            else:
                probs, _ = model(batch_x)
            all_probs.append(probs.cpu())
            all_labels.append(labels)
    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    return {
        'probs': probs,
        'labels': labels,
        'preds': probs.argmax(dim=1),
        'conf': probs.max(dim=1).values,
    }


# ==========================================
# Reliability diagram plotting
# ==========================================
def _compute_bins(probs, labels, mask, n_bins, bin_edges):
    conf = probs.max(dim=1).values.numpy()[mask]
    preds = probs.argmax(dim=1).numpy()[mask]
    correct = (preds == labels.numpy()[mask]).astype(float)

    bin_accs, bin_confs, bin_counts = [], [], []
    for i in range(n_bins):
        in_bin = (conf > bin_edges[i]) & (conf <= bin_edges[i + 1])
        if in_bin.sum() > 0:
            bin_accs.append(correct[in_bin].mean())
            bin_confs.append(conf[in_bin].mean())
            bin_counts.append(in_bin.sum())
        else:
            bin_accs.append(0)
            bin_confs.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_counts.append(0)
    return np.array(bin_accs), np.array(bin_confs), np.array(bin_counts)


def plot_reliability_grid(all_predictions, title_prefix, output_name, output_dir,
                          mask=None, mask_label=None, temperatures=None,
                          temp_scaled=False):
    """Plot reliability diagrams for 7 methods in a 4+3 grid."""
    available = [m for m in METHODS if m in all_predictions]
    n = len(available)
    ncols = 4
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)

    if mask is None:
        n_samples = len(next(iter(all_predictions.values()))[0]['labels'])
        mask = np.ones(n_samples, dtype=bool)

    for idx, method in enumerate(available):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        seed_preds = all_predictions[method]
        all_bin_accs = []
        all_eces = []

        for pred in seed_preds:
            probs = pred['probs']
            if temp_scaled and temperatures and method in temperatures:
                probs = apply_temperature_scaling(probs, temperatures[method])

            bin_accs, bin_confs, bin_counts = _compute_bins(
                probs, pred['labels'], mask, n_bins, bin_edges)
            all_bin_accs.append(bin_accs)

            total = bin_counts.sum()
            ece = sum(
                (c / total) * abs(a - co)
                for c, a, co in zip(bin_counts, bin_accs, bin_confs)
                if c > 0
            )
            all_eces.append(ece)

        bin_accs_mean = np.mean(all_bin_accs, axis=0)
        bin_accs_std = np.std(all_bin_accs, axis=0)
        ece_mean = np.mean(all_eces)
        ece_std = np.std(all_eces)

        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = 1.0 / n_bins * 0.85

        ece_label = 'ECE+TS' if temp_scaled else 'ECE'

        ax.bar(centers, bin_accs_mean, width=width, yerr=bin_accs_std,
               color=COLORS[method], alpha=0.75,
               edgecolor='none',
               capsize=3, error_kw={'linewidth': 1})
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{LABELS[method]}  ({ece_label}={ece_mean:.3f}\u00b1{ece_std:.3f})',
                     fontsize=13, color=COLORS[method], fontweight='bold')
        ax.grid(alpha=0.2)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    n_seeds = len(next(iter(all_predictions.values())))
    ts_suffix = ' (Temp Scaled)' if temp_scaled else ''
    suffix = f' \u2014 {mask_label}' if mask_label else ''
    fig.suptitle(f'{title_prefix} Reliability{ts_suffix}{suffix}, {n_seeds}-trial mean \u00b1 std',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = figure_path(f'{output_name}.pdf', output_dir)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ==========================================
# CIFAR-10H: confidence vs agreement
# ==========================================
def plot_confidence_vs_agreement(predictions, agreement, output_dir):
    available = [m for m in METHODS if m in predictions]

    fig, ax = plt.subplots(figsize=(10, 5))

    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    bin_labels = ['<0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '>0.9']
    n_bins = len(bin_labels)

    width = 0.8 / len(available)
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, len(available))

    for idx, method in enumerate(available):
        pred = predictions[method]
        conf = pred['conf'].numpy()

        means = []
        stds = []
        for b in range(n_bins):
            mask = (agreement >= bins[b]) & (agreement < bins[b + 1])
            if mask.sum() > 0:
                means.append(conf[mask].mean())
                stds.append(conf[mask].std() / np.sqrt(mask.sum()))
            else:
                means.append(0)
                stds.append(0)

        ax.bar(np.arange(n_bins) + offsets[idx], means, width,
               color=COLORS[method], alpha=0.8, edgecolor='none',
               label=LABELS[method])
        ax.errorbar(np.arange(n_bins) + offsets[idx], means, yerr=stds, fmt='none',
                    color='black', capsize=3, linewidth=1)

    # Accuracy reference (from vanilla)
    if 'vanilla' in predictions:
        pred = predictions['vanilla']
        preds = pred['preds'].numpy()
        labels = pred['labels'].numpy()
        acc_means = []
        for b in range(n_bins):
            mask = (agreement >= bins[b]) & (agreement < bins[b + 1])
            if mask.sum() > 0:
                acc_means.append((preds[mask] == labels[mask]).mean())
            else:
                acc_means.append(0)
        ax.plot(np.arange(n_bins), acc_means, 'k--', linewidth=2.5,
                marker='o', markersize=8, label='Accuracy (Vanilla)', zorder=5)

    # Hard region shading
    ax.axvspan(-0.5, 2.5, alpha=0.08, color='red')
    ax.text(1.0, 0.03, 'Hard region', fontsize=18, color='red',
            ha='center', fontweight='bold', alpha=0.7)

    ax.set_xlabel('Human Agreement Level', fontsize=20)
    ax.set_ylabel('Mean Confidence', fontsize=20)
    ax.set_title('CIFAR-10H: Confidence vs Human Agreement', fontsize=22)
    ax.set_xticks(np.arange(n_bins))
    ax.set_xticklabels(bin_labels, fontsize=17)
    ax.tick_params(axis='y', labelsize=17)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=15, loc='lower right', ncol=2)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    out_path = figure_path('cifar10h_confidence_vs_agreement.pdf', output_dir)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ==========================================
# CivilComments: group breakdown
# ==========================================
def compute_ece_np(conf, correct, n_bins=15):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(conf)
    if total == 0:
        return 0.0
    for i in range(n_bins):
        in_bin = (conf > bin_edges[i]) & (conf <= bin_edges[i + 1])
        n_in = in_bin.sum()
        if n_in > 0:
            avg_conf = conf[in_bin].mean()
            avg_acc = correct[in_bin].mean()
            ece += (n_in / total) * abs(avg_acc - avg_conf)
    return ece


def plot_group_breakdown(all_predictions, group_masks, output_dir):
    available = [m for m in METHODS if m in all_predictions]
    groups = list(group_masks.keys())
    group_labels = {
        'male': 'Male', 'female': 'Female', 'LGBTQ': 'LGBTQ',
        'christian': 'Christian', 'muslim': 'Muslim',
        'other_religions': 'Other Rel.', 'black': 'Black', 'white': 'White',
    }

    n_methods = len(available)
    n_groups = len(groups)

    ece_all = np.zeros((n_methods, n_groups, len(SEEDS)))
    acc_all = np.zeros((n_methods, n_groups, len(SEEDS)))
    count_row = []

    for j, group in enumerate(groups):
        mask = group_masks[group]
        count_row.append(int(mask.sum()))
        for i, method in enumerate(available):
            for s, pred in enumerate(all_predictions[method]):
                conf = pred['conf'].numpy()[mask]
                correct = (pred['preds'].numpy()[mask] == pred['labels'].numpy()[mask]).astype(float)
                ece_all[i, j, s] = compute_ece_np(conf, correct)
                acc_all[i, j, s] = correct.mean()

    ece_mean = ece_all.mean(axis=2)
    ece_std = ece_all.std(axis=2)
    acc_mean = acc_all.mean(axis=2)
    acc_std = acc_all.std(axis=2)
    n_seeds = ece_all.shape[2]

    x = np.arange(n_groups)
    width = 0.8 / n_methods
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, n_methods)

    for mean_mat, std_mat, metric_name, ylabel, filename in [
        (ece_mean, ece_std, 'ECE', 'ECE', 'civilcomments_group_ece.pdf'),
        (acc_mean, acc_std, 'Accuracy', 'Accuracy', 'civilcomments_group_accuracy.pdf'),
    ]:
        fig, ax = plt.subplots(figsize=(14, 5))

        for i, method in enumerate(available):
            ax.bar(x + offsets[i], mean_mat[i], width, yerr=std_mat[i],
                   color=COLORS[method], alpha=0.85, edgecolor='none',
                   capsize=3, error_kw={'linewidth': 1},
                   label=LABELS[method])

        ax.set_xlabel('Identity Group', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{group_labels[g]}\n({count_row[j]:,})'
                           for j, g in enumerate(groups)], fontsize=11)
        ax.legend(fontsize=10, loc='best')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        ax.set_title(f'CivilComments: {metric_name} by Identity Group ({n_seeds}-trial mean \u00b1 std)',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        out_path = figure_path(filename, output_dir)
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")


# ==========================================
# Dataset-specific loading
# ==========================================
def load_temperatures(results_dir):
    """Load temperature scaling values from seed-42 JSONs."""
    temperatures = {}
    for method in METHODS:
        json_files = glob.glob(f'{results_dir}/{method}_s42/*.json')
        if json_files:
            with open(json_files[0]) as f:
                result_data = json.load(f)
            temperatures[method] = result_data.get('temperature', 1.0)
        else:
            temperatures[method] = 1.0
    return temperatures


def load_cifar10h_predictions(device, results_dir='experiment_results/cifar10h/results'):
    _, testset, _, agreement = get_dataset('cifar10h', './data')
    loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)

    all_predictions = {}
    for method in METHODS:
        is_moe = method != 'single'
        method_preds = []
        for seed in SEEDS:
            # MoCaE shares vanilla's checkpoint; per-expert TS is applied at inference.
            if method == 'mocae':
                ckpt = f'{results_dir}/vanilla_s{seed}/vanilla_s{seed}.pt'
                per_expert_temps = load_mocae_temperatures(results_dir, seed)
            else:
                ckpt = f'{results_dir}/{method}_s{seed}/{method}_s{seed}.pt'
                per_expert_temps = None
            if not os.path.exists(ckpt):
                print(f"  WARNING: {ckpt} not found")
                continue
            print(f"  Loading {method} s{seed}...", end=' ', flush=True)
            model = load_model(ckpt, device, is_moe=is_moe)
            method_preds.append(get_predictions(model, loader, device,
                                                per_expert_temps=per_expert_temps))
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()
            print("done")
        if method_preds:
            all_predictions[method] = method_preds

    temperatures = load_temperatures(results_dir)
    return all_predictions, agreement, temperatures


def load_pacs_predictions(results_dir, target_domain, device, data_dir):
    _, testset, _, _ = get_dataset('pacs', data_dir, target_domain=target_domain)
    loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)
    n_test = len(testset)

    all_predictions = {}
    for method in METHODS:
        is_moe = method != 'single'
        method_preds = []
        for seed in SEEDS:
            # MoCaE shares vanilla's checkpoint; per-expert TS is applied at inference.
            if method == 'mocae':
                ckpt = f'{results_dir}/vanilla_s{seed}/vanilla_s{seed}.pt'
                per_expert_temps = load_mocae_temperatures(results_dir, seed)
            else:
                ckpt = f'{results_dir}/{method}_s{seed}/{method}_s{seed}.pt'
                per_expert_temps = None
            if not os.path.exists(ckpt):
                print(f"  WARNING: {ckpt} not found")
                continue
            print(f"  Loading {method} s{seed}...", end=' ', flush=True)
            model = load_model(ckpt, device, is_moe=is_moe)
            method_preds.append(get_predictions(model, loader, device,
                                                per_expert_temps=per_expert_temps))
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()
            print("done")
        if method_preds:
            all_predictions[method] = method_preds

    temperatures = load_temperatures(results_dir)
    return all_predictions, n_test, temperatures


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR,
                        help=f'Base output directory; figures are written to '
                             f'<output_dir>/{{cifar10h,civilcomments,pacs}}/ '
                             f'(default: {DEFAULT_OUTPUT_DIR}/)')
    parser.add_argument('--cifar10h_results_dir',
                        default='experiment_results/cifar10h/results',
                        help='Override the CIFAR-10H checkpoint directory '
                             '(default: experiment_results/cifar10h/results/).')
    parser.add_argument('--pacs_data_dir', default='./data',
                        help='Root directory of the (preprocessed) PACS data '
                             '(default: ./data).')
    parser.add_argument('--skip-cifar10h', action='store_true')
    parser.add_argument('--skip-civilcomments', action='store_true')
    parser.add_argument('--skip-pacs', action='store_true')
    args = parser.parse_args()

    out = args.output_dir

    # === CIFAR-10H ===
    if not args.skip_cifar10h:
        print("\n" + "=" * 60)
        print("CIFAR-10H")
        print("=" * 60)
        preds, agreement, temps = load_cifar10h_predictions(args.device, args.cifar10h_results_dir)
        hard_mask = agreement < 0.7
        all_mask = np.ones(len(agreement), dtype=bool)

        # Baselines (4+3 grid)
        for subset, mask, label in [
            ('all', all_mask, f'All Images (n={int(all_mask.sum())})'),
            ('hard', hard_mask, f'Hard Images (agreement < 0.7, n={int(hard_mask.sum())})'),
        ]:
            plot_reliability_grid(preds, 'CIFAR-10H',
                                  f'baselines_cifar10h_reliability_{subset}', out,
                                  mask=mask, mask_label=label)

        # Non-baselines reliability (same grid, with and without temp scaling)
        for subset, mask, label in [
            ('all', all_mask, f'All Images (n={int(all_mask.sum())})'),
            ('hard', hard_mask, f'Hard Images (agreement < 0.7, n={int(hard_mask.sum())})'),
        ]:
            plot_reliability_grid(preds, 'CIFAR-10H',
                                  f'cifar10h_reliability_{subset}', out,
                                  mask=mask, mask_label=label)
            plot_reliability_grid(preds, 'CIFAR-10H',
                                  f'cifar10h_reliability_{subset}_ts', out,
                                  mask=mask, mask_label=label,
                                  temperatures=temps, temp_scaled=True)

        # Confidence vs agreement (uses seed 42)
        predictions_s42 = {m: ps[0] for m, ps in preds.items()}
        plot_confidence_vs_agreement(predictions_s42, agreement, out)

    # === CivilComments (from cache) ===
    if not args.skip_civilcomments:
        print("\n" + "=" * 60)
        print("CivilComments (from cache)")
        print("=" * 60)

        cache = torch.load(CC_CACHE_PATH, map_location='cpu', weights_only=False)
        all_predictions = {m: cache['all_predictions'][m] for m in METHODS
                          if m in cache['all_predictions']}
        difficulty = cache['difficulty']
        temperatures = cache.get('temperatures', {})
        group_masks = cache.get('group_masks', {})

        if isinstance(difficulty, torch.Tensor):
            difficulty = difficulty.numpy()

        hard_mask = (difficulty == 0)  # identity-mentioning comments
        all_mask = np.ones(len(difficulty), dtype=bool)

        # Baselines (4+3 grid)
        for subset, mask, label in [
            ('all', all_mask, f'All (n={int(all_mask.sum())})'),
            ('hard', hard_mask, f'Hard (identity-mentioning, n={int(hard_mask.sum())})'),
        ]:
            plot_reliability_grid(all_predictions, 'CivilComments',
                                  f'baselines_civilcomments_reliability_{subset}', out,
                                  mask=mask, mask_label=label)

        # Non-baselines reliability
        for subset, mask, label in [
            ('all', all_mask, f'All Comments (n={int(all_mask.sum()):,})'),
            ('hard', hard_mask, f'Hard Comments (identity-mentioning, n={int(hard_mask.sum()):,})'),
        ]:
            plot_reliability_grid(all_predictions, 'CivilComments',
                                  f'civilcomments_reliability_{subset}', out,
                                  mask=mask, mask_label=label)
            plot_reliability_grid(all_predictions, 'CivilComments',
                                  f'civilcomments_reliability_{subset}_ts', out,
                                  mask=mask, mask_label=label,
                                  temperatures=temperatures, temp_scaled=True)

        # Group breakdown
        if group_masks:
            plot_group_breakdown(all_predictions, group_masks, out)

    # === PACS (all domains) ===
    if not args.skip_pacs:
        PACS_VARIANTS = [
            ('experiment_results/pacs_sketch/results',  'sketch',       'sketch'),
            ('experiment_results/pacs_art/results',     'art_painting', 'art'),
            ('experiment_results/pacs_cartoon/results',  'cartoon',      'cartoon'),
            ('experiment_results/pacs_photo/results',    'photo',        'photo'),
        ]

        for results_dir, target_domain, label in PACS_VARIANTS:
            print(f"\n{'=' * 60}")
            print(f"PACS target={target_domain}")
            print(f"{'=' * 60}")
            preds, n_test, temps = load_pacs_predictions(results_dir, target_domain, args.device,
                                                          args.pacs_data_dir)

            domain_title = label.replace('_', ' ').title()

            # Baselines (only sketch)
            if label == 'sketch':
                plot_reliability_grid(preds, f'PACS ({domain_title})',
                                      f'baselines_pacs_sketch_reliability', out,
                                      mask_label=f'Target: {domain_title} (n={n_test})')

            # Non-baselines reliability
            plot_reliability_grid(preds, f'PACS ({domain_title})',
                                  f'pacs_{label}_reliability', out,
                                  mask_label=f'Target: {domain_title} (n={n_test})')
            plot_reliability_grid(preds, f'PACS ({domain_title})',
                                  f'pacs_{label}_reliability_ts', out,
                                  mask_label=f'Target: {domain_title} (n={n_test})',
                                  temperatures=temps, temp_scaled=True)

    print(f"\nAll PDFs saved under {out}/")


if __name__ == '__main__':
    main()
