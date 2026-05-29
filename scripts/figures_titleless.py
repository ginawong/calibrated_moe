#!/usr/bin/env python
"""Titleless versions of cifar10h_confidence_vs_agreement,
civilcomments_group_accuracy / group_ece, and the stacked group_acc_ece figure.

Outputs are suffixed with `_titleless` so they coexist with the titled versions
from scripts/figures_reliability.py in the same output directory.

Usage:
    python scripts/figures_titleless.py
    python scripts/figures_titleless.py --output_dir PAPER_RESULTS/figures
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from calibrated_moe.datasets import get_dataset
from calibrated_moe.evaluation import forward_with_per_expert_ts, load_mocae_temperatures
from calibrated_moe.models import MoE, SingleExpert, get_backbone

DEFAULT_OUTPUT_DIR = 'figures'
CC_CACHE_PATH = '.cache/civilcomments_predictions.pt'


def figure_path(output_name, base_dir):
    """Route a figure filename to the dataset subdirectory of base_dir."""
    name = os.path.basename(output_name)
    if name.startswith('cifar10h_'):
        sub = 'cifar10h'
    elif name.startswith('civilcomments_'):
        sub = 'civilcomments'
    elif name.startswith('pacs_'):
        sub = 'pacs'
    else:
        sub = ''
    out_dir = os.path.join(base_dir, sub) if sub else base_dir
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, name)

METHODS = ['single', 'vanilla', 'mocae', 'fgr', 'robust', 'robust_filtered', 'fgr_robust']

COLORS = {
    'single': '#95a5a6', 'vanilla': '#e74c3c', 'mocae': '#f39c12',
    'robust': '#3498db', 'robust_filtered': '#9b59b6',
    'fgr': '#d35400', 'fgr_robust': '#1abc9c',
}

LABELS = {
    'single': 'Single Expert', 'vanilla': 'Vanilla MoE', 'mocae': 'MoCaE',
    'robust': 'Robust MoE', 'robust_filtered': 'Robust Filtered',
    'fgr': 'FGR', 'fgr_robust': 'FGR + Robust',
}

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
        'probs': probs, 'labels': labels,
        'preds': probs.argmax(dim=1), 'conf': probs.max(dim=1).values,
    }


# ==========================================
# Confidence vs Agreement (titleless, larger font)
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
        means, stds = [], []
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

    if 'vanilla' in predictions:
        pred = predictions['vanilla']
        preds_np = pred['preds'].numpy()
        labels_np = pred['labels'].numpy()
        acc_means = []
        for b in range(n_bins):
            mask = (agreement >= bins[b]) & (agreement < bins[b + 1])
            if mask.sum() > 0:
                acc_means.append((preds_np[mask] == labels_np[mask]).mean())
            else:
                acc_means.append(0)
        ax.plot(np.arange(n_bins), acc_means, 'k--', linewidth=2.5,
                marker='o', markersize=8, label='Accuracy (Vanilla)', zorder=5)

    ax.axvspan(-0.5, 2.5, alpha=0.08, color='red')
    ax.text(1.0, 0.03, 'Hard region', fontsize=15, color='red',
            ha='center', fontweight='bold', alpha=0.7)

    ax.set_xlabel('Human Agreement Level', fontsize=16)
    ax.set_ylabel('Mean Confidence', fontsize=16)
    ax.set_xticks(np.arange(n_bins))
    ax.set_xticklabels(bin_labels, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=12, loc='lower right', ncol=2)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    out_path = figure_path('cifar10h_confidence_vs_agreement_titleless.pdf', output_dir)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ==========================================
# Group breakdown (titleless, larger font)
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
            ece += (n_in / total) * abs(conf[in_bin].mean() - correct[in_bin].mean())
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

    x = np.arange(n_groups)
    width = 0.8 / n_methods
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, n_methods)

    for mean_mat, std_mat, ylabel, filename in [
        (ece_mean, ece_std, 'ECE', 'civilcomments_group_ece.pdf'),
        (acc_mean, acc_std, 'Accuracy', 'civilcomments_group_accuracy.pdf'),
    ]:
        fig, ax = plt.subplots(figsize=(14, 5))

        for i, method in enumerate(available):
            ax.bar(x + offsets[i], mean_mat[i], width, yerr=std_mat[i],
                   color=COLORS[method], alpha=0.85, edgecolor='none',
                   capsize=3, error_kw={'linewidth': 1},
                   label=LABELS[method])

        ax.set_xlabel('Identity Group', fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{group_labels[g]}\n({count_row[j]:,})'
                           for j, g in enumerate(groups)], fontsize=13)
        ax.tick_params(axis='y', labelsize=14)
        ax.legend(fontsize=12, loc='best')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        if ylabel == 'Accuracy':
            ax.set_ylim(bottom=0.6)
        plt.tight_layout()

        stem, ext = os.path.splitext(filename)
        out_path = figure_path(f'{stem}_titleless{ext}', output_dir)
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")

    # Stacked: accuracy on top, ECE on bottom (shared x-axis)
    fig, (ax_acc, ax_ece) = plt.subplots(
        2, 1, figsize=(18, 7), sharex=True,
        gridspec_kw={'hspace': 0.06},
    )

    for i, method in enumerate(available):
        ax_acc.bar(x + offsets[i], acc_mean[i], width, yerr=acc_std[i],
                   color=COLORS[method], alpha=0.85, edgecolor='none',
                   capsize=3, error_kw={'linewidth': 1},
                   label=LABELS[method])
        ax_ece.bar(x + offsets[i], ece_mean[i], width, yerr=ece_std[i],
                   color=COLORS[method], alpha=0.85, edgecolor='none',
                   capsize=3, error_kw={'linewidth': 1},
                   label=LABELS[method])

    ax_acc.set_ylabel('Accuracy', fontsize=16)
    ax_acc.tick_params(axis='y', labelsize=14)
    ax_acc.tick_params(axis='x', which='both', length=0)
    ax_acc.spines['top'].set_visible(False)
    ax_acc.spines['right'].set_visible(False)
    ax_acc.grid(axis='y', alpha=0.3)
    ax_acc.legend(fontsize=12, loc='upper right', ncol=2)

    ax_ece.set_xlabel('Identity Group', fontsize=16)
    ax_ece.set_ylabel('ECE', fontsize=16)
    ax_ece.set_xticks(x)
    ax_ece.set_xticklabels([f'{group_labels[g]}\n({count_row[j]:,})'
                           for j, g in enumerate(groups)], fontsize=13)
    ax_ece.tick_params(axis='y', labelsize=14)
    ax_ece.spines['top'].set_visible(False)
    ax_ece.spines['right'].set_visible(False)
    ax_ece.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = figure_path('civilcomments_group_acc_ece_titleless.pdf', output_dir)
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")

    ax_acc.set_ylim(0.6, ax_acc.get_ylim()[1])
    out_path = figure_path('civilcomments_group_acc_ece_zoom_titleless.pdf', output_dir)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ==========================================
# Main
# ==========================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR,
                        help=f'Base output directory; figures land in '
                             f'<output_dir>/{{cifar10h,civilcomments}}/ '
                             f'with a `_titleless` filename suffix '
                             f'(default: {DEFAULT_OUTPUT_DIR}/)')
    args = parser.parse_args()

    out = args.output_dir

    # === CIFAR-10H: confidence vs agreement ===
    print("\n" + "=" * 60)
    print("CIFAR-10H: confidence vs agreement (titleless)")
    print("=" * 60)
    results_dir = 'experiment_results/cifar10h/results'
    _, testset, _, agreement = get_dataset('cifar10h', './data')
    loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)

    predictions_s42 = {}
    for method in METHODS:
        is_moe = method != 'single'
        # MoCaE shares vanilla's checkpoint; per-expert TS is applied at inference.
        if method == 'mocae':
            ckpt = f'{results_dir}/vanilla_s42/vanilla_s42.pt'
            per_expert_temps = load_mocae_temperatures(results_dir, 42)
        else:
            ckpt = f'{results_dir}/{method}_s42/{method}_s42.pt'
            per_expert_temps = None
        if not os.path.exists(ckpt):
            print(f"  WARNING: {ckpt} not found")
            continue
        print(f"  Loading {method} s42...", end=' ', flush=True)
        model = load_model(ckpt, args.device, is_moe=is_moe)
        predictions_s42[method] = get_predictions(model, loader, args.device,
                                                  per_expert_temps=per_expert_temps)
        del model
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        print("done")

    plot_confidence_vs_agreement(predictions_s42, agreement, out)

    # === CivilComments: group breakdown ===
    print("\n" + "=" * 60)
    print("CivilComments: group breakdown (titleless)")
    print("=" * 60)
    cache = torch.load(CC_CACHE_PATH, map_location='cpu', weights_only=False)
    all_predictions = {m: cache['all_predictions'][m] for m in METHODS
                      if m in cache['all_predictions']}
    group_masks = cache.get('group_masks', {})

    if group_masks:
        plot_group_breakdown(all_predictions, group_masks, out)
    else:
        print("  WARNING: no group_masks in cache")

    print(f"\nAll titleless PDFs saved under {out}/")


if __name__ == '__main__':
    main()
