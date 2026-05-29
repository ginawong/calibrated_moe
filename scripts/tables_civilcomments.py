#!/usr/bin/env python
"""CivilComments results table (LaTeX, plain text, HTML, PNG) and per-identity
group breakdown.

Reads per-(method, seed) JSON metrics from <results_dir>/<method>_s<seed>/ and
aggregates across seeds for the 7 paper methods.

The script also caches model predictions to <cache_path> (default
.cache/civilcomments_predictions.pt) on the first run; subsequent runs reuse
the cache. Predictions are needed for reliability diagrams and per-group
breakdown.

Usage:
    python scripts/tables_civilcomments.py --output_dir output/civilcomments
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
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from calibrated_moe.calibration import apply_temperature_scaling
from calibrated_moe.datasets import get_dataset
from calibrated_moe.evaluation import forward_with_per_expert_ts, load_mocae_temperatures
from calibrated_moe.models import MoE, SingleExpert, get_backbone


DEFAULT_RESULTS_DIR = 'experiment_results/civilcomments/results'
DEFAULT_OUTPUT_DIR = 'output/civilcomments'
# Heavy prediction cache; shared with scripts/figures_reliability.py and figures_titleless.py.
CACHE_PATH = '.cache/civilcomments_predictions.pt'
SEEDS = [42, 43, 44, 45, 46]

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
            ('base', 'accuracy'), ('base', 'ece'), ('base', 'hard_ece'),
            ('base', 'hard_acc'), ('base', 'easy_acc'), ('base', 'easy_ece'),
            ('temp_scaled', 'ece'), ('temp_scaled', 'hard_ece'),
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
        cfg.get('backbone', 'distilbert'),
        small_input=cfg.get('small_input', False),
        num_blocks=cfg.get('num_blocks', 3),
        pretrained=False,  # weights come from the checkpoint
    )
    num_classes = cfg.get('num_classes', 2)

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
        for batch_x, labels in tqdm(loader, desc='    Inference', ncols=0, leave=False):
            # CivilComments returns dicts with input_ids + attention_mask
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


def load_all_predictions(results_dir, device):
    """Load checkpoints for each method across all seeds and run inference.

    Returns (predictions, all_predictions, temperatures, n_test, difficulty).
    """
    sample_json = glob.glob(f'{results_dir}/**/*.json', recursive=True)[0]
    with open(sample_json) as f:
        data_dir = json.load(f)['config']['data_dir']

    print(f"  Loading CivilComments data from {data_dir}...")
    _, testset, _, difficulty = get_dataset('civilcomments', data_dir)
    loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    n_test = len(testset)

    predictions = {}
    all_predictions = {}
    temperatures = {}
    for method in tqdm(ALL_METHODS, desc='Methods', ncols=0):
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
            model = load_model(ckpt, device, is_moe=is_moe)
            preds = get_predictions(model, loader, device, per_expert_temps=per_expert_temps)
            method_preds.append(preds)
            if seed == 42:
                predictions[method] = preds
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()

        if not method_preds:
            print(f"  Skipping {method} — no checkpoints found")
            continue
        all_predictions[method] = method_preds

        json_files = glob.glob(f'{results_dir}/{method}_s42/*.json')
        if json_files:
            with open(json_files[0]) as f:
                result_data = json.load(f)
            temperatures[method] = result_data.get('temperature', 1.0)
        else:
            temperatures[method] = 1.0

    return predictions, all_predictions, temperatures, n_test, difficulty


def load_test_group_masks(data_dir):
    """Load per-identity-group boolean masks for the test set.

    Returns dict: group_name -> bool array of length n_test.
    """
    from wilds import get_dataset as wilds_get_dataset

    ds = wilds_get_dataset(dataset='civilcomments', root_dir=data_dir, download=False)
    split_arr = ds.split_array if isinstance(ds.split_array, np.ndarray) else ds.split_array.numpy()
    meta = ds.metadata_array if isinstance(ds.metadata_array, np.ndarray) else ds.metadata_array.numpy()

    test_mask = (split_arr == 2)  # test split
    test_meta = meta[test_mask]

    identity_fields = ['male', 'female', 'LGBTQ', 'christian', 'muslim',
                       'other_religions', 'black', 'white']
    group_masks = {}
    for field in identity_fields:
        idx = ds.metadata_fields.index(field)
        group_masks[field] = (test_meta[:, idx] == 1)

    return group_masks


def compute_ece_np(conf, correct, n_bins=15):
    """Compute ECE from numpy arrays."""
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


# ==========================================
# Figure 1: Results table
# ==========================================
def plot_table(agg, output_dir):
    columns = ['Method', 'Acc', 'Hard Acc', 'ECE', 'ECE+TS', 'Hard ECE', 'Hard ECE+TS']
    col_keys = [
        'base_accuracy', 'base_hard_acc', 'base_ece', 'temp_scaled_ece',
        'base_hard_ece', 'temp_scaled_hard_ece',
    ]

    rows = []
    for method in ALL_METHODS:
        if method not in agg:
            continue
        m = agg[method]
        row = [LABELS[method]]
        for key in col_keys:
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

    fig, ax = plt.subplots(figsize=(15, 0.5 * len(rows) + 1.8))
    ax.axis('off')

    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    for i, method in enumerate([m for m in ALL_METHODS if m in agg]):
        table[i + 1, 0].set_text_props(color=COLORS[method], fontweight='bold')

    # Bold best: higher for Acc/Hard Acc, lower for ECE columns
    higher_better = [True, True, False, False, False, False]
    for col_idx in range(6):
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

    ax.set_title('CivilComments: All Methods (5-trial mean \u00b1 std)',
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    _savefig(os.path.join(output_dir, 'civilcomments_table.png'))

    # Plain text
    header = f"{'Method':<20} {'Acc':>12} {'Hard Acc':>12} {'ECE':>12} {'ECE+TS':>12} {'Hard ECE':>12} {'Hard ECE+TS':>12}"
    lines = [header, '-' * len(header)]
    for row in rows:
        lines.append(f"{row[0]:<20} {row[1]:>12} {row[2]:>12} {row[3]:>12} {row[4]:>12} {row[5]:>12} {row[6]:>12}")
    txt_path = os.path.join(output_dir, 'civilcomments_table.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {txt_path}")

    # HTML version (Jekyll / MathJax compatible)
    html_lines = [
        '<table>',
        '<caption>CivilComments: All Methods (5-trial mean &pm; std)</caption>',
        '<thead>',
        '<tr>' + ''.join(f'<th>{c}</th>' for c in columns) + '</tr>',
        '</thead>',
        '<tbody>',
    ]
    for row in rows:
        cells = ''.join(f'<td>{cell}</td>' for cell in row)
        html_lines.append(f'<tr>{cells}</tr>')
    html_lines += ['</tbody>', '</table>']
    html_path = os.path.join(output_dir, 'civilcomments_table.html')
    with open(html_path, 'w') as f:
        f.write('\n'.join(html_lines) + '\n')
    print(f"  Saved: {html_path}")

    # LaTeX version
    n_data_cols = len(columns) - 1
    tex_lines = [
        r'\begin{tabular}{l' + 'c' * n_data_cols + '}',
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
    tex_path = os.path.join(output_dir, 'civilcomments_table.tex')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines) + '\n')
    print(f"  Saved: {tex_path}")


# ==========================================
# Figure 2: Reliability diagrams
# ==========================================
def _compute_bins(probs, labels, mask, n_bins, bin_edges):
    """Compute per-bin accuracy, confidence, and counts for one seed."""
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


def plot_reliability_diagrams(all_predictions, n_test, difficulty, output_dir,
                              subset='all', temperatures=None, temp_scaled=False):
    """Plot reliability diagrams for each method in a 2-row grid.

    Uses all seeds to compute mean +/- std per bin.
    subset: 'all' for all comments, 'hard' for identity-mentioning comments.
    """
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

    if subset == 'hard':
        mask = (difficulty == 0)
        title_suffix = f'Hard Comments (identity-mentioning, n={int(mask.sum()):,})'
    else:
        mask = np.ones(len(difficulty), dtype=bool)
        title_suffix = f'All Comments (n={int(mask.sum()):,})'

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
                probs, pred['labels'], mask, n_bins, bin_edges)
            all_bin_accs.append(bin_accs)

            total = bin_counts.sum()
            ece = sum(
                (c / total) * abs(a - co)
                for c, a, co in zip(bin_counts, bin_accs, bin_confs)
                if c > 0
            )
            all_eces.append(ece)

        # mean +/- std across seeds
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
    ts_suffix = ' (Temp Scaled)' if temp_scaled else ''
    fig.suptitle(f'CivilComments Reliability{ts_suffix} \u2014 {title_suffix}, {n_seeds}-trial mean \u00b1 std',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    suffix = 'hard' if subset == 'hard' else 'all'
    ts_tag = '_ts' if temp_scaled else ''
    _savefig(os.path.join(output_dir, f'civilcomments_reliability_{suffix}{ts_tag}.png'))


# ==========================================
# Figure 3: Per-identity-group breakdown
# ==========================================
def plot_group_breakdown(all_predictions, group_masks, output_dir):
    """Grouped bar charts of ECE and accuracy per identity group per method.

    Uses all seeds to compute mean +/- std.
    """
    available = [m for m in ALL_METHODS if m in all_predictions]
    groups = list(group_masks.keys())
    group_labels = {
        'male': 'Male', 'female': 'Female', 'LGBTQ': 'LGBTQ',
        'christian': 'Christian', 'muslim': 'Muslim',
        'other_religions': 'Other Rel.', 'black': 'Black', 'white': 'White',
    }

    # Compute ECE and accuracy per (method, group, seed)
    # Shape: [n_methods, n_groups, n_seeds]
    n_methods = len(available)
    n_groups = len(groups)
    count_row = []

    ece_all = np.zeros((n_methods, n_groups, len(SEEDS)))
    acc_all = np.zeros((n_methods, n_groups, len(SEEDS)))

    for j, group in enumerate(groups):
        mask = group_masks[group]
        if j == 0:
            count_row = []
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

    # --- Text table ---
    g_headers = [f'{group_labels[g]} ({count_row[j]:,})' for j, g in enumerate(groups)]
    header = f"{'Method':<20}" + ''.join(f'{h:>20}' for h in g_headers)
    lines_ece = [f'ECE by Identity Group ({n_seeds}-trial mean\u00b1std)', header, '-' * len(header)]
    lines_acc = ['', f'Accuracy by Identity Group ({n_seeds}-trial mean\u00b1std)', header, '-' * len(header)]
    for i, method in enumerate(available):
        row_ece = f"{LABELS[method]:<20}" + ''.join(
            f'{ece_mean[i,j]:.3f}\u00b1{ece_std[i,j]:.3f}'.rjust(20) for j in range(n_groups))
        row_acc = f"{LABELS[method]:<20}" + ''.join(
            f'{acc_mean[i,j]:.3f}\u00b1{acc_std[i,j]:.3f}'.rjust(20) for j in range(n_groups))
        lines_ece.append(row_ece)
        lines_acc.append(row_acc)

    txt_path = os.path.join(output_dir, 'civilcomments_group_breakdown.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines_ece + lines_acc) + '\n')
    print(f"  Saved: {txt_path}")

    # --- Grouped bar charts ---
    x = np.arange(n_groups)
    width = 0.8 / n_methods
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, n_methods)

    for mean_mat, std_mat, metric_name, ylabel, filename in [
        (ece_mean, ece_std, 'ECE', 'ECE', 'civilcomments_group_ece.png'),
        (acc_mean, acc_std, 'Accuracy', 'Accuracy', 'civilcomments_group_accuracy.png'),
    ]:
        fig, ax = plt.subplots(figsize=(14, 5))

        for i, method in enumerate(available):
            ax.bar(x + offsets[i], mean_mat[i], width, yerr=std_mat[i],
                   color=COLORS[method], alpha=0.85, edgecolor='black', linewidth=0.5,
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
        _savefig(os.path.join(output_dir, filename))


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
    parser = argparse.ArgumentParser(description='Generate CivilComments table and figures.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--results_dir', default=DEFAULT_RESULTS_DIR,
                        help=f'Per-(method, seed) results layout (default: {DEFAULT_RESULTS_DIR}/)')
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR}/)')
    parser.add_argument('--cache_path', default=CACHE_PATH,
                        help=f'Heavy prediction cache (default: {CACHE_PATH})')
    parser.add_argument('--no_cache', action='store_true',
                        help='Force re-running inference (ignore cached predictions).')
    parser.add_argument('--table_only', action='store_true',
                        help='Generate the .tex/.txt/.html/.png table only.')
    args = parser.parse_args()

    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.dirname(args.cache_path), exist_ok=True)

    print("Loading JSON results...")
    all_results = load_all_results(args.results_dir)
    print(f"  Found {len(all_results)} canonical run files")
    agg = aggregate(all_results)
    print(f"  Methods: {[m for m in ALL_METHODS if m in agg]}")

    print("\nResults table")
    plot_table(agg, out)

    if args.table_only:
        print(f"\nDone. Table saved under {out}/")
        return

    use_cache = not args.no_cache and os.path.exists(args.cache_path)
    if use_cache:
        print(f"\nLoading cached predictions from {args.cache_path}...")
        cache = torch.load(args.cache_path, map_location='cpu', weights_only=False)
        predictions = cache['predictions']
        all_predictions = cache['all_predictions']
        temperatures = cache['temperatures']
        n_test = cache['n_test']
        difficulty = cache['difficulty']
        group_masks = cache['group_masks']
        print(f"  Loaded {len(predictions)} methods, {n_test:,} test samples")
    else:
        print("\nLoading models and running inference (all seeds)...")
        predictions, all_predictions, temperatures, n_test, difficulty = \
            load_all_predictions(args.results_dir, args.device)
        print(f"  Got predictions for: {list(predictions.keys())}")
        print(f"  Test set: {n_test:,} samples ({int((difficulty == 0).sum()):,} hard, "
              f"{int((difficulty == 1).sum()):,} easy)")

        sample_json = glob.glob(f'{args.results_dir}/**/*.json', recursive=True)[0]
        with open(sample_json) as f:
            data_dir = json.load(f)['config']['data_dir']

        print("\nLoading per-group identity metadata...")
        group_masks = load_test_group_masks(data_dir)
        for g, mask in group_masks.items():
            print(f"  {g}: {int(mask.sum()):,} comments")

        print(f"\nSaving cache to {args.cache_path}...")
        torch.save({
            'predictions': predictions,
            'all_predictions': all_predictions,
            'temperatures': temperatures,
            'n_test': n_test,
            'difficulty': difficulty,
            'group_masks': group_masks,
        }, args.cache_path)
        print("  Cache saved.")

    print("\nReliability diagrams (all)")
    plot_reliability_diagrams(all_predictions, n_test, difficulty, out, subset='all')
    print("\nReliability diagrams (all, temp scaled)")
    plot_reliability_diagrams(all_predictions, n_test, difficulty, out, subset='all',
                              temperatures=temperatures, temp_scaled=True)
    print("\nReliability diagrams (hard)")
    plot_reliability_diagrams(all_predictions, n_test, difficulty, out, subset='hard')
    print("\nReliability diagrams (hard, temp scaled)")
    plot_reliability_diagrams(all_predictions, n_test, difficulty, out, subset='hard',
                              temperatures=temperatures, temp_scaled=True)

    print("\nPer-identity-group breakdown")
    plot_group_breakdown(all_predictions, group_masks, out)

    print(f"\nAll CivilComments outputs saved under {out}/")


if __name__ == '__main__':
    main()
