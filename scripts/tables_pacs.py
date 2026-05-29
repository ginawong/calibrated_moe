#!/usr/bin/env python
"""PACS leave-one-domain-out results table per target domain.

For each held-out target domain (sketch, art_painting, cartoon, photo):
  - aggregate per-(method, seed) JSON metrics into a results table
  - generate reliability diagrams on the target-domain test set

Usage:
    python scripts/tables_pacs.py --output_dir output/pacs
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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from calibrated_moe.calibration import apply_temperature_scaling
from calibrated_moe.datasets import get_dataset
from calibrated_moe.evaluation import forward_with_per_expert_ts, load_mocae_temperatures
from calibrated_moe.models import MoE, SingleExpert, get_backbone


DEFAULT_OUTPUT_DIR = 'output/pacs'

# (results_dir, target_domain, label for filenames)
PACS_VARIANTS = [
    ('experiment_results/pacs_sketch/results',  'sketch',       'sketch'),
    ('experiment_results/pacs_art/results',     'art_painting', 'art'),
    ('experiment_results/pacs_cartoon/results', 'cartoon',      'cartoon'),
    ('experiment_results/pacs_photo/results',   'photo',        'photo'),
]

ALL_METHODS = [
    'single', 'vanilla', 'mocae', 'fgr',
    'robust', 'robust_filtered', 'fgr_robust',
]

COLORS = {
    'single':          '#95a5a6',
    'vanilla':         '#e74c3c',
    'mocae':           '#f39c12',
    'robust':          '#3498db',
    'robust_filtered': '#9b59b6',
    'fgr':             '#1abc9c',
    'fgr_robust':      '#d35400',
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


# ==========================================
# Data loading
# ==========================================
def _is_canonical_run_dir(dirname, method):
    prefix = f'{method}_s'
    if not dirname.startswith(prefix):
        return False
    return dirname[len(prefix):].isdigit()


def load_all_results(results_dir):
    """Load JSON metrics from canonical `{method}_s{seed}/` directories only."""
    results = []
    for method in ALL_METHODS:
        for path in glob.glob(f'{results_dir}/{method}_s*/*.json'):
            parent = os.path.basename(os.path.dirname(path))
            if not _is_canonical_run_dir(parent, method):
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, KeyError):
                continue
            if 'base' not in data or data.get('method') != method:
                continue
            results.append(data)
    return results


def aggregate(all_results):
    groups = defaultdict(list)
    for r in all_results:
        method = r.get('method')
        if method in ALL_METHODS:
            groups[method].append(r)

    agg = {}
    for method, runs in groups.items():
        metrics = {}
        for section, key in [
            ('base', 'accuracy'), ('base', 'ece'),
            ('temp_scaled', 'ece'),
        ]:
            vals = [r[section][key] for r in runs if section in r and key in r[section]]
            if vals:
                metrics[f'{section}_{key}_mean'] = np.mean(vals)
                metrics[f'{section}_{key}_std'] = np.std(vals)
                metrics[f'{section}_{key}_n'] = len(vals)
        agg[method] = metrics
    return agg


# ==========================================
# Model loading & inference
# ==========================================
def load_model(checkpoint_path, device, is_moe=True):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get('config', {})
    method = checkpoint.get('method', '')

    backbone = get_backbone(
        cfg.get('backbone', 'resnet18'),
        small_input=cfg.get('small_input', False),
        num_blocks=cfg.get('num_blocks', 4),
        pretrained=False,
    )
    num_classes = cfg.get('num_classes', 7)

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
        for images, labels in loader:
            images = images.to(device)
            if per_expert_temps is not None:
                probs, _ = forward_with_per_expert_ts(model, images, per_expert_temps)
            else:
                probs, _ = model(images)
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


SEEDS = [42, 43, 44, 45, 46]


def load_all_predictions(results_dir, target_domain, device):
    """Load all seed checkpoints for each method, run inference."""
    sample_json = glob.glob(f'{results_dir}/**/*.json', recursive=True)[0]
    with open(sample_json) as f:
        data_dir = json.load(f)['config']['data_dir']

    print(f"  Loading PACS data (target={target_domain})...")
    _, testset, _, _ = get_dataset('pacs', data_dir, target_domain=target_domain)
    loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)
    n_test = len(testset)

    all_predictions = {}  # method -> list of pred dicts (one per seed)
    temperatures = {}
    for method in ALL_METHODS:
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
        if not method_preds:
            print(f"  Skipping {method} — no checkpoints found")
            continue
        all_predictions[method] = method_preds

        # Load temperature from seed-42 JSON
        json_pattern = f'{results_dir}/{method}_s42/*.json'
        json_files = glob.glob(json_pattern)
        if json_files:
            with open(json_files[0]) as f:
                result_data = json.load(f)
            temperatures[method] = result_data.get('temperature', 1.0)
        else:
            temperatures[method] = 1.0

    return all_predictions, temperatures, n_test


# ==========================================
# Figure: Results table
# ==========================================
def plot_table(agg, target_label, output_dir):
    columns = ['Method', 'Acc', 'ECE', 'ECE+TS']
    col_keys = [('base_accuracy',), ('base_ece',), ('temp_scaled_ece',)]

    rows = []
    for method in ALL_METHODS:
        if method not in agg:
            continue
        m = agg[method]
        row = [LABELS[method]]
        for (key,) in col_keys:
            mean = m.get(f'{key}_mean')
            std = m.get(f'{key}_std')
            n = m.get(f'{key}_n', 0)
            if mean is None:
                row.append('--')
            elif n > 1 and std is not None:
                row.append(f'{mean:.3f}\u00b1{std:.3f}')
            else:
                row.append(f'{mean:.3f}')
        rows.append(row)

    fig, ax = plt.subplots(figsize=(10, 0.5 * len(rows) + 1.8))
    ax.axis('off')

    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.6)

    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    for i, method in enumerate([m for m in ALL_METHODS if m in agg]):
        table[i + 1, 0].set_text_props(color=COLORS[method], fontweight='bold')

    # Bold best: higher better for Acc, lower better for ECE columns
    higher_better = [True, False, False]
    for col_idx in range(3):
        vals = []
        for row in rows:
            try:
                vals.append(float(row[col_idx + 1].split('\u00b1')[0]))
            except (ValueError, IndexError):
                vals.append(None)
        valid = [v for v in vals if v is not None]
        if not valid:
            continue
        best = max(valid) if higher_better[col_idx] else min(valid)
        for row_idx, v in enumerate(vals):
            if v is not None and abs(v - best) < 1e-4:
                table[row_idx + 1, col_idx + 1].set_text_props(fontweight='bold')

    domain_title = target_label.replace('_', ' ').title()
    ax.set_title(f'PACS (target={domain_title}): All Methods (5-trial mean \u00b1 std)',
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    _savefig(os.path.join(output_dir, f'pacs_{target_label}_table.png'))

    # Plain text version
    header = f"{'Method':<20} {'Acc':>12} {'ECE':>12} {'ECE+TS':>12}"
    lines = [header, '-' * len(header)]
    for row in rows:
        lines.append(f"{row[0]:<20} {row[1]:>12} {row[2]:>12} {row[3]:>12}")
    txt_path = os.path.join(output_dir, f'pacs_{target_label}_table.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {txt_path}")

    # HTML version (Jekyll / MathJax compatible)
    domain_title = target_label.replace('_', ' ').title()
    html_lines = [
        f'<table>',
        f'<caption>PACS (target={domain_title}): All Methods (5-trial mean &pm; std)</caption>',
        '<thead>',
        '<tr>' + ''.join(f'<th>{c}</th>' for c in columns) + '</tr>',
        '</thead>',
        '<tbody>',
    ]
    for row in rows:
        cells = ''.join(f'<td>{cell}</td>' for cell in row)
        html_lines.append(f'<tr>{cells}</tr>')
    html_lines += ['</tbody>', '</table>']
    html_path = os.path.join(output_dir, f'pacs_{target_label}_table.html')
    with open(html_path, 'w') as f:
        f.write('\n'.join(html_lines) + '\n')
    print(f"  Saved: {html_path}")

    # LaTeX version
    tex_lines = [
        r'\begin{tabular}{lccc}',
        r'\toprule',
        ' & '.join(columns) + r' \\',
        r'\midrule',
    ]
    for row in rows:
        tex_row = row[0]
        for cell in row[1:]:
            tex_row += ' & ' + cell.replace('\u00b1', r'$\pm$')
        tex_lines.append(tex_row + r' \\')
    tex_lines += [r'\bottomrule', r'\end{tabular}']
    tex_path = os.path.join(output_dir, f'pacs_{target_label}_table.tex')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines) + '\n')
    print(f"  Saved: {tex_path}")


# ==========================================
# Figure: Reliability diagrams
# ==========================================
def _compute_bins(probs, labels, n_bins, bin_edges):
    """Compute per-bin accuracy, confidence, and counts for one seed."""
    conf = probs.max(dim=1).values.numpy()
    preds = probs.argmax(dim=1).numpy()
    correct = (preds == labels.numpy()).astype(float)

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


def plot_reliability_diagrams(all_predictions, n_test, target_label, output_dir,
                              temperatures=None, temp_scaled=False):
    """Plot reliability diagrams with multi-seed mean +/- std per bin."""
    available = [m for m in ALL_METHODS if m in all_predictions]
    n = len(available)
    ncols = min(n, 5)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)

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
                probs, pred['labels'], n_bins, bin_edges)
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

        ax.bar(centers, bin_accs_mean, width=width, yerr=bin_accs_std,
               color=COLORS[method], alpha=0.75,
               edgecolor='black', linewidth=0.5,
               capsize=3, error_kw={'linewidth': 1})

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ece_label = 'ECE+TS' if temp_scaled else 'ECE'
        ax.set_title(f'{LABELS[method]}  ({ece_label}={ece_mean:.3f}\u00b1{ece_std:.3f})',
                     fontsize=13, color=COLORS[method], fontweight='bold')
        ax.grid(alpha=0.2)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    n_seeds = len(next(iter(all_predictions.values())))
    domain_title = target_label.replace('_', ' ').title()
    ts_suffix = ' (Temp Scaled)' if temp_scaled else ''
    fig.suptitle(f'PACS Reliability{ts_suffix} \u2014 Target: {domain_title} (n={n_test}, {n_seeds}-trial mean \u00b1 std)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    suffix = '_ts' if temp_scaled else ''
    _savefig(os.path.join(output_dir, f'pacs_{target_label}_reliability{suffix}.png'))


# ==========================================
# Helpers
# ==========================================
def _savefig(path):
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Generate PACS results tables and figures.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR}/)')
    parser.add_argument('--table_only', action='store_true',
                        help='Generate the .tex/.txt/.html/.png tables only.')
    args = parser.parse_args()

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    for results_dir, target_domain, label in PACS_VARIANTS:
        print(f"\n{'='*60}")
        print(f"PACS target={target_domain} ({results_dir})")
        print(f"{'='*60}")

        print("\nLoading JSON results...")
        all_results = load_all_results(results_dir)
        print(f"  Found {len(all_results)} canonical run files")
        agg = aggregate(all_results)
        print(f"  Methods: {[m for m in ALL_METHODS if m in agg]}")

        print(f"\nTable: pacs_{label}_table")
        plot_table(agg, label, out)

        if args.table_only:
            continue

        print("\nLoading models (all seeds) for reliability diagrams...")
        all_predictions, temperatures, n_test = load_all_predictions(results_dir, target_domain, args.device)
        print(f"  Got predictions for: {list(all_predictions.keys())}")

        print(f"\nReliability diagrams: pacs_{label}_reliability")
        plot_reliability_diagrams(all_predictions, n_test, label, out)
        print(f"\nReliability diagrams (temp scaled): pacs_{label}_reliability_ts")
        plot_reliability_diagrams(all_predictions, n_test, label, out,
                                  temperatures=temperatures, temp_scaled=True)

    print(f"\nAll PACS outputs saved under {out}/")


if __name__ == '__main__':
    main()
