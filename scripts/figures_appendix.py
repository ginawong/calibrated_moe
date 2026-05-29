#!/usr/bin/env python
"""Appendix figures that combine multiple methods.

  - confidence_histograms_full.pdf — 3×3 grid of confidence histograms,
        rows = {Vanilla, MoCaE, Robust}, cols = {Overall, Easy, Hard}.
        Uses seed-42 vanilla / mocae / robust checkpoints.

  - example_failures.pdf — image grid showing CIFAR-10H hard images where
        Vanilla is confidently wrong but Robust gives lower confidence on
        plausible classes. Uses the same seed-42 checkpoints.

Usage:
    python scripts/figures_appendix.py --output_dir figures
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from calibrated_moe.datasets import CIFAR10_MEAN, CIFAR10_STD, load_cifar10h
from calibrated_moe.evaluation import forward_with_per_expert_ts, load_mocae_temperatures
from calibrated_moe.models import MoE, get_backbone

# ----------------------------------------------------------------------------
# Display style
# ----------------------------------------------------------------------------

COLORS = {
    'vanilla': '#e74c3c',
    'mocae':   '#f39c12',
    'robust':  '#3498db',
}
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

_MEAN = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
_STD = torch.tensor(CIFAR10_STD).view(3, 1, 1)


def _denormalize(img_tensor):
    img = img_tensor.cpu() * _STD + _MEAN
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def _figure_dir(base_dir, dataset_sub):
    out = os.path.join(base_dir, dataset_sub) if dataset_sub else base_dir
    os.makedirs(out, exist_ok=True)
    return out


# ----------------------------------------------------------------------------
# CIFAR-10H data + model loading
# ----------------------------------------------------------------------------

def _load_testset_and_agreement(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)
    soft = load_cifar10h(data_dir)
    agreement = soft.max(axis=1)
    return testset, agreement


def _load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt['config']
    backbone = get_backbone(c.get('backbone', 'resnet18'),
                            small_input=c.get('small_input', True),
                            num_blocks=c.get('num_blocks', 3),
                            pretrained=False)
    model = MoE(num_experts=c.get('num_experts', 4),
                num_classes=c.get('num_classes', 10),
                hidden_dim=c.get('router_hidden_dim', 128),
                backbone=backbone).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def _get_predictions(model, loader, device, per_expert_temps=None):
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


def _collect_predictions(results_dir, methods, seed, device, data_dir='./data'):
    testset, agreement = _load_testset_and_agreement(data_dir)
    raw_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)
    predictions = {}
    for method in methods:
        # MoCaE shares vanilla's checkpoint; per-expert TS is applied at inference.
        if method == 'mocae':
            ckpt = f'{results_dir}/vanilla_s{seed}/vanilla_s{seed}.pt'
            per_expert_temps = load_mocae_temperatures(results_dir, seed)
        else:
            ckpt = f'{results_dir}/{method}_s{seed}/{method}_s{seed}.pt'
            per_expert_temps = None
        if not os.path.exists(ckpt):
            print(f"  WARNING: {ckpt} not found, skipping")
            continue
        print(f"  Loading {method} s{seed}...", flush=True)
        model = _load_model(ckpt, device)
        predictions[method] = _get_predictions(model, raw_loader, device,
                                                per_expert_temps=per_expert_temps)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    return predictions, testset, agreement


# ----------------------------------------------------------------------------
# Figure: confidence_histograms_full
# ----------------------------------------------------------------------------

def plot_confidence_histograms_full(predictions, agreement, output_path):
    """3×3 grid of confidence histograms: methods × difficulty levels."""
    method_specs = [
        ('vanilla', 'Vanilla MoE', COLORS['vanilla']),
        ('mocae',   'MoCaE',       COLORS['mocae']),
        ('robust',  'Robust MoE',  COLORS['robust']),
    ]
    available = [(m, l, c) for m, l, c in method_specs if m in predictions]
    if len(available) < 2:
        print("  Skipping confidence_histograms_full: need at least 2 methods")
        return

    levels = [
        ('Overall', np.ones(len(agreement), dtype=bool)),
        ('Easy',    agreement > 0.9),
        ('Hard',    agreement < 0.7),
    ]
    bins = np.linspace(0.1, 1.0, 30)

    fig, axes = plt.subplots(len(available), len(levels),
                             figsize=(5 * len(levels), 3 * len(available)))
    if len(available) == 1:
        axes = axes.reshape(1, -1)

    for row, (method, label, color) in enumerate(available):
        preds = predictions[method]
        conf = preds['conf'].numpy()
        correct = (preds['preds'] == preds['labels']).numpy()

        for col, (level_name, mask) in enumerate(levels):
            ax = axes[row, col]
            conf_m = conf[mask]
            correct_m = correct[mask]

            ax.hist(conf_m, bins=bins, alpha=0.7, color=color,
                    edgecolor='black', linewidth=0.5)

            mean_conf = conf_m.mean()
            acc = correct_m.mean()
            ax.axvline(mean_conf, color='black', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_conf:.2f}')
            ax.axvline(acc, color='green', linestyle=':', linewidth=2,
                       label=f'Acc: {acc:.2f}')

            gap = mean_conf - acc
            gap_color = 'red' if gap > 0.05 else 'green'
            ax.text(0.95, 0.95, f'Gap: {gap:+.2f}',
                    transform=ax.transAxes, fontsize=14, ha='right', va='top',
                    fontweight='bold', color=gap_color,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlim(0.1, 1.0)
            ax.set_xlabel('Confidence', fontsize=14)
            ax.tick_params(axis='both', labelsize=12)
            if col == 0:
                ax.set_ylabel(f'{label}\nCount', fontsize=14, color=color)
            if row == 0:
                ax.set_title(f'{level_name} (n={int(mask.sum())})', fontsize=16)
            ax.legend(fontsize=10, loc='upper left')
            ax.grid(alpha=0.3)

    fig.suptitle('Confidence Distributions by Method and Difficulty', fontsize=18)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ----------------------------------------------------------------------------
# Figure: example_failures
# ----------------------------------------------------------------------------

def plot_example_failures(testset, predictions, agreement, output_path, n_examples=8):
    """Hard images where Vanilla is confidently wrong but Robust corrects course."""
    vanilla = predictions.get('vanilla')
    robust = predictions.get('robust')
    if vanilla is None or robust is None:
        print("  Skipping example_failures: need vanilla and robust predictions")
        return

    v_conf = vanilla['conf'].numpy()
    r_conf = robust['conf'].numpy()
    v_correct = (vanilla['preds'] == vanilla['labels']).numpy()

    hard_mask = agreement < 0.7
    interesting = hard_mask & ~v_correct & (v_conf > 0.6) & (r_conf < v_conf)
    indices = np.where(interesting)[0]
    if len(indices) == 0:
        print("  No example_failures found (vanilla confidently wrong but robust less confident)")
        return

    np.random.seed(42)
    indices = np.random.choice(indices, min(n_examples, len(indices)), replace=False)

    methods = [('vanilla', vanilla, 'Vanilla', COLORS['vanilla'])]
    if 'mocae' in predictions:
        methods.append(('mocae', predictions['mocae'], 'MoCaE', COLORS['mocae']))
    methods.append(('robust', robust, 'Robust', COLORS['robust']))

    n_cols = len(indices)
    fig = plt.figure(figsize=(3.2 * n_cols, 7))
    gs = GridSpec(2, n_cols, height_ratios=[1, 1.3], hspace=0.15, top=0.82)

    first_bar_ax = None
    for col, idx in enumerate(indices):
        img, label = testset[idx]

        ax_img = fig.add_subplot(gs[0, col])
        ax_img.imshow(_denormalize(img))
        ax_img.set_title(f'True: {CLASSES[label]}\nAgree: {agreement[idx]:.2f}',
                         fontsize=14, fontweight='bold')
        ax_img.axis('off')

        ax_bar = (fig.add_subplot(gs[1, col]) if col == 0
                  else fig.add_subplot(gs[1, col], sharey=first_bar_ax))
        if col == 0:
            first_bar_ax = ax_bar

        # Pick the top-4 classes by max-importance across methods
        all_probs = np.stack([p['probs'][idx].numpy() for _, p, _, _ in methods])
        top_classes = np.argsort(all_probs.max(axis=0))[-4:][::-1]

        x = np.arange(len(top_classes))
        width = 0.25
        offsets = np.linspace(-(len(methods) - 1) * width / 2,
                              (len(methods) - 1) * width / 2,
                              len(methods))

        for i, (_, preds, method_label, color) in enumerate(methods):
            probs = preds['probs'][idx].numpy()
            pred_class = preds['preds'][idx].item()
            ax_bar.bar(x + offsets[i], probs[top_classes], width,
                       label=method_label, color=color, alpha=0.8,
                       edgecolor='black', linewidth=0.5)
            # Star above the predicted class
            for j, cls in enumerate(top_classes):
                if cls == pred_class:
                    ax_bar.annotate('★', (x[j] + offsets[i], probs[cls] + 0.02),
                                    ha='center', fontsize=14, color=color)

        # Mark true class
        for j, cls in enumerate(top_classes):
            if cls == label:
                ax_bar.axvline(x[j], color='green', linestyle='--', alpha=0.5, linewidth=2)

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([CLASSES[c][:6] for c in top_classes],
                               fontsize=12, rotation=30, ha='right')
        ax_bar.set_ylim(0, 1.1)
        ax_bar.grid(axis='y', alpha=0.3)
        if col == 0:
            ax_bar.set_ylabel('Confidence', fontsize=14)
            ax_bar.tick_params(axis='y', labelsize=12)
            ax_bar.legend(fontsize=12, loc='upper right')
        else:
            ax_bar.tick_params(axis='y', labelleft=False)

    fig.suptitle('Hard Images Where Vanilla MoE is Overconfident', fontsize=18, y=0.94)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', default='figures',
                        help='Base output directory; figures land under <output_dir>/cifar10h/.')
    parser.add_argument('--results_dir', default='experiment_results/cifar10h/results',
                        help='Canonical {method}_s{seed}/ checkpoints.')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_confidence', action='store_true')
    parser.add_argument('--skip_failures', action='store_true')
    args = parser.parse_args()

    cifar10h_out = _figure_dir(args.output_dir, 'cifar10h')

    print("Loading vanilla / mocae / robust predictions...")
    predictions, testset, agreement = _collect_predictions(
        args.results_dir, ['vanilla', 'mocae', 'robust'], args.seed, args.device,
        data_dir=args.data_dir)

    if not args.skip_confidence:
        print("\nFigure: confidence_histograms_full")
        plot_confidence_histograms_full(
            predictions, agreement,
            os.path.join(cifar10h_out, 'confidence_histograms_full.pdf'))

    if not args.skip_failures:
        print("\nFigure: example_failures")
        plot_example_failures(
            testset, predictions, agreement,
            os.path.join(cifar10h_out, 'example_failures.pdf'))

    print(f"\nDone. Outputs under {args.output_dir}/")


if __name__ == '__main__':
    main()
