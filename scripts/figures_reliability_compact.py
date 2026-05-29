#!/usr/bin/env python
"""Single-row reliability diagrams (all 7 methods in one row, tight spacing,
large fonts) intended for compact paper inclusion. Outputs are suffixed with
`_compact` so they coexist with the grid-layout versions from
scripts/figures_reliability.py in the same output directory.

Usage:
    python scripts/figures_reliability_compact.py
    python scripts/figures_reliability_compact.py --output_dir PAPER_RESULTS/figures
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FixedFormatter, FixedLocator
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

METHODS = [
    'single', 'vanilla', 'mocae', 'fgr',
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
    'robust_filtered': 'Robust Filt.',
    'fgr':             'FGR',
    'fgr_robust':      'FGR + Robust',
}

SEEDS = [42, 43, 44, 45, 46]


# ==========================================
# Model loading & inference (same as main script)
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
# Binning
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


# ==========================================
# Compact reliability grid
# ==========================================
def plot_compact_reliability(all_predictions, output_name, output_dir,
                              mask=None, temperatures=None, temp_scaled=False):
    """Single-row 1x7 reliability: no suptitle, labels only on leftmost, big fonts."""
    available = [m for m in METHODS if m in all_predictions]
    n = len(available)

    # Large fonts for small-figure readability
    TITLE_SIZE = 22
    LABEL_SIZE = 22
    TICK_SIZE = 18
    ECE_SIZE = 16

    # Insert a narrow divider column between baselines (first 4) and robust methods
    divider_pos = 4  # after FGR, before Robust MoE
    if n > divider_pos:
        width_ratios = [1] * divider_pos + [0.08] + [1] * (n - divider_pos)
        n_cols = n + 1
    else:
        width_ratios = [1] * n
        n_cols = n
        divider_pos = None

    fig, all_axes = plt.subplots(1, n_cols, figsize=(3.6 * n + (0.3 if divider_pos else 0), 3.8),
                                  gridspec_kw={'wspace': 0.12, 'width_ratios': width_ratios})

    # Separate plot axes from divider axis
    if divider_pos is not None:
        axes = list(all_axes[:divider_pos]) + list(all_axes[divider_pos + 1:])
        div_ax = all_axes[divider_pos]
        div_ax.set_xlim(0, 1)
        div_ax.set_ylim(0, 1)
        div_ax.axvline(0.5, color='#999999', linewidth=1.2, alpha=0.5)
        div_ax.axis('off')
    else:
        axes = list(all_axes)

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)

    if mask is None:
        n_samples = len(next(iter(all_predictions.values()))[0]['labels'])
        mask = np.ones(n_samples, dtype=bool)

    for idx, method in enumerate(available):
        ax = axes[idx]

        seed_preds = all_predictions[method]
        all_bin_accs, all_eces = [], []

        for pred in seed_preds:
            probs = pred['probs']
            if temp_scaled and temperatures and method in temperatures:
                probs = apply_temperature_scaling(probs, temperatures[method])

            bin_accs, bin_confs, bin_counts = _compute_bins(
                probs, pred['labels'], mask, n_bins, bin_edges)
            all_bin_accs.append(bin_accs)
            total = bin_counts.sum()
            ece = sum((c / total) * abs(a - co)
                      for c, a, co in zip(bin_counts, bin_accs, bin_confs) if c > 0)
            all_eces.append(ece)

        bin_accs_mean = np.mean(all_bin_accs, axis=0)
        bin_accs_std = np.std(all_bin_accs, axis=0)
        ece_mean = np.mean(all_eces)
        ece_std = np.std(all_eces)

        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = 1.0 / n_bins * 0.85

        ax.bar(centers, bin_accs_mean, width=width, yerr=bin_accs_std,
               color=COLORS[method], alpha=0.75, edgecolor='none',
               capsize=2, error_kw={'linewidth': 0.8})
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.6)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

        # Title: method name + ECE, no figure-level suptitle
        ece_label = 'ECE+TS' if temp_scaled else 'ECE'
        ax.set_title(f'{LABELS[method]}\n{ece_label}={ece_mean:.3f}',
                     fontsize=TITLE_SIZE, color=COLORS[method], fontweight='bold',
                     pad=4, linespacing=1.1)

        # Ticks: only 0, 0.5, 1 — short labels to avoid edge-tick collisions
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xticklabels(['0', '0.5', '1'])
        ax.set_yticklabels(['0', '0.5', '1'])

        # "Confidence" xlabel + x tick labels on every subplot; ylabel + y tick labels only on leftmost
        ax.set_xlabel('Confidence', fontsize=LABEL_SIZE, labelpad=2)
        ax.tick_params(axis='x', labelsize=TICK_SIZE)
        if idx == 0:
            ax.set_ylabel('Accuracy', fontsize=LABEL_SIZE, labelpad=2)
            ax.tick_params(axis='y', labelsize=TICK_SIZE)
        else:
            ax.tick_params(axis='y', labelleft=False)

        ax.grid(alpha=0.15)

    out_path = figure_path(f'{output_name}_compact.pdf', output_dir)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"  Saved: {out_path}")


# ==========================================
# Data loading
# ==========================================
def load_temperatures(results_dir):
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


def load_cifar10h_predictions(device):
    results_dir = 'experiment_results/cifar10h/results'
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
                        help=f'Base output directory; figures land in '
                             f'<output_dir>/{{cifar10h,civilcomments,pacs}}/ '
                             f'with a `_compact` filename suffix '
                             f'(default: {DEFAULT_OUTPUT_DIR}/)')
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
        preds, agreement, temps = load_cifar10h_predictions(args.device)
        hard_mask = agreement < 0.7
        all_mask = np.ones(len(agreement), dtype=bool)

        for name, mask, ts in [
            ('cifar10h_reliability_all', all_mask, False),
            ('cifar10h_reliability_hard', hard_mask, False),
            ('cifar10h_reliability_all_ts', all_mask, True),
            ('cifar10h_reliability_hard_ts', hard_mask, True),
        ]:
            plot_compact_reliability(preds, name, out, mask=mask,
                                     temperatures=temps, temp_scaled=ts)

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
        if isinstance(difficulty, torch.Tensor):
            difficulty = difficulty.numpy()

        hard_mask = (difficulty == 0)
        all_mask = np.ones(len(difficulty), dtype=bool)

        for name, mask, ts in [
            ('civilcomments_reliability_all', all_mask, False),
            ('civilcomments_reliability_hard', hard_mask, False),
            ('civilcomments_reliability_all_ts', all_mask, True),
            ('civilcomments_reliability_hard_ts', hard_mask, True),
        ]:
            plot_compact_reliability(all_predictions, name, out, mask=mask,
                                     temperatures=temperatures, temp_scaled=ts)

    # === PACS ===
    if not args.skip_pacs:
        PACS_VARIANTS = [
            ('experiment_results/pacs_sketch/results', 'sketch', 'sketch'),
            ('experiment_results/pacs_art/results', 'art_painting', 'art'),
            ('experiment_results/pacs_cartoon/results', 'cartoon', 'cartoon'),
            ('experiment_results/pacs_photo/results', 'photo', 'photo'),
        ]
        for results_dir, target_domain, label in PACS_VARIANTS:
            print(f"\n{'=' * 60}")
            print(f"PACS target={target_domain}")
            print(f"{'=' * 60}")
            preds, n_test, temps = load_pacs_predictions(results_dir, target_domain, args.device,
                                                          args.pacs_data_dir)
            for ts_suffix, ts in [('', False), ('_ts', True)]:
                plot_compact_reliability(preds, f'pacs_{label}_reliability{ts_suffix}',
                                          out, temperatures=temps, temp_scaled=ts)

    print(f"\nAll compact PDFs saved under {out}/")


if __name__ == '__main__':
    main()
