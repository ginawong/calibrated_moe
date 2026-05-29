#!/usr/bin/env python
"""Example image grid from the PACS dataset.

Rows = domains (art_painting, cartoon, photo, sketch), columns = classes.

Usage:
    python scripts/figures_pacs_examples.py --data_dir /path/to/domainbed_datasets
    python scripts/figures_pacs_examples.py --titleless --output_dir PAPER_RESULTS/figures
"""

import argparse
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

DOMAINS = ['photo', 'art_painting', 'cartoon', 'sketch']
CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

DOMAIN_LABELS = {
    'art_painting': 'Art Painting',
    'cartoon': 'Cartoon',
    'photo': 'Photo',
    'sketch': 'Sketch',
}

DEFAULT_OUTPUT_DIR = 'figures'


def main():
    parser = argparse.ArgumentParser(description='Generate PACS example image grid')
    parser.add_argument('--data_dir', default='./data',
                        help='Path to DomainBed datasets root')
    parser.add_argument('--cols', type=int, default=len(CLASSES),
                        help='Number of classes to show (default: all 7)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for image selection')
    parser.add_argument('--titleless', action='store_true',
                        help='Drop suptitle, slightly larger non-bold labels; '
                             'appends `_titleless` to the output filename.')
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR,
                        help=f'Base output dir; file lands under '
                             f'<output_dir>/pacs/pacs_examples[_titleless].pdf '
                             f'(default: {DEFAULT_OUTPUT_DIR}/)')
    args = parser.parse_args()

    random.seed(args.seed)
    pacs_dir = os.path.join(args.data_dir, 'PACS')
    classes = CLASSES[:args.cols]

    nrows = len(DOMAINS)
    ncols = len(classes)

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))

    title_fontsize = 16 if args.titleless else 14
    ylabel_fontsize = 15 if args.titleless else 13
    label_weight = 'normal' if args.titleless else 'bold'

    for row, domain in enumerate(DOMAINS):
        domain_dir = os.path.join(pacs_dir, domain)
        n_samples = sum(len(os.listdir(os.path.join(domain_dir, c)))
                        for c in os.listdir(domain_dir)
                        if os.path.isdir(os.path.join(domain_dir, c)))
        for col, cls in enumerate(classes):
            ax = axes[row, col]
            cls_dir = os.path.join(pacs_dir, domain, cls)
            images = sorted(os.listdir(cls_dir))
            chosen = random.choice(images)
            img = Image.open(os.path.join(cls_dir, chosen)).convert('RGB')
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(cls.capitalize(), fontsize=title_fontsize,
                             fontweight=label_weight)
            if col == 0:
                ax.set_ylabel(f'{DOMAIN_LABELS[domain]}\n(n={n_samples})',
                              fontsize=ylabel_fontsize, fontweight=label_weight)

    if not args.titleless:
        fig.suptitle(f'PACS Dataset ({len(DOMAINS)} domains, {len(CLASSES)} classes)',
                     fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_dir = os.path.join(args.output_dir, 'pacs')
    os.makedirs(out_dir, exist_ok=True)
    suffix = '_titleless' if args.titleless else ''
    out_path = os.path.join(out_dir, f'pacs_examples{suffix}.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
