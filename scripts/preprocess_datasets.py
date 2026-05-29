#!/usr/bin/env python
"""Preprocess DomainBed datasets into .pt tensor files for fast loading.

For each domain in a dataset, resizes all images to 224x224 and saves as a
single .pt file containing:
    {'images': tensor [N,3,224,224], 'targets': tensor [N], 'classes': list}

Domains are discovered by listing subdirectories of the dataset root.
Classes are discovered by listing subdirectories within each domain.

Output files are saved alongside the original domain directories:
    domainbed_datasets/PACS/sketch_224x224.pt
    domainbed_datasets/office_home/Clipart_224x224.pt

Skips domains that already have a .pt file.

Usage:
    python scripts/preprocess_datasets.py --data_dir /path/to/domainbed_datasets/PACS
    python scripts/preprocess_datasets.py --data_dir /path/to/domainbed_datasets/office_home
"""
import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image


IMG_SIZE = 224


def discover_domains(dataset_dir):
    """List domain directories (subdirs that contain class subdirs)."""
    domains = []
    for name in sorted(os.listdir(dataset_dir)):
        path = os.path.join(dataset_dir, name)
        if os.path.isdir(path) and not name.startswith('.'):
            # Check it contains subdirectories (classes), not just files
            has_subdirs = any(os.path.isdir(os.path.join(path, f)) for f in os.listdir(path))
            if has_subdirs:
                domains.append(name)
    return domains


def discover_classes(dataset_dir, domains):
    """Discover class names from subdirectories, consistent across all domains."""
    all_classes = set()
    for domain in domains:
        domain_dir = os.path.join(dataset_dir, domain)
        classes = [d for d in sorted(os.listdir(domain_dir))
                   if os.path.isdir(os.path.join(domain_dir, d))]
        all_classes.update(classes)
    return sorted(all_classes)


def preprocess_domain(domain_dir, class_to_idx, out_path):
    """Load all images from a domain directory, resize, and save as .pt."""
    if os.path.exists(out_path):
        print(f"  Skipping {out_path} (already exists)", flush=True)
        return

    resize = transforms.Resize((IMG_SIZE, IMG_SIZE))
    imgs = []
    targets = []

    # Count total files first
    total = 0
    for cls_name in sorted(os.listdir(domain_dir)):
        if cls_name not in class_to_idx:
            continue
        cls_dir = os.path.join(domain_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        total += len([f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))])

    report_every = max(1, total // 10)
    count = 0

    print(f"  Processing {total} images -> {out_path}", flush=True)
    for cls_name in sorted(os.listdir(domain_dir)):
        if cls_name not in class_to_idx:
            continue
        cls_idx = class_to_idx[cls_name]
        cls_dir = os.path.join(domain_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        for fname in sorted(os.listdir(cls_dir)):
            fpath = os.path.join(cls_dir, fname)
            if not os.path.isfile(fpath):
                continue
            img = Image.open(fpath).convert('RGB')
            img = resize(img)
            imgs.append(transforms.functional.to_tensor(img))
            targets.append(cls_idx)
            count += 1
            if count % report_every == 0:
                print(f"    {count}/{total} ({count/total*100:.0f}%)", flush=True)

    data = {
        'images': torch.stack(imgs),
        'targets': torch.tensor(targets, dtype=torch.long),
        'classes': sorted(class_to_idx.keys()),
    }
    torch.save(data, out_path)
    mb = data['images'].element_size() * data['images'].nelement() / 1e6
    print(f"  Saved {len(imgs)} images ({mb:.0f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to a dataset directory (e.g. domainbed_datasets/PACS)')
    args = parser.parse_args()

    dataset_dir = args.data_dir
    if not os.path.isdir(dataset_dir):
        print(f"Error: {dataset_dir} not found")
        return

    domains = discover_domains(dataset_dir)
    classes = discover_classes(dataset_dir, domains)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    print(f"Dataset: {dataset_dir}")
    print(f"Domains: {domains}")
    print(f"Classes: {classes} ({len(classes)} total)", flush=True)

    for domain in domains:
        domain_dir = os.path.join(dataset_dir, domain)
        out_path = os.path.join(dataset_dir, f"{domain}_224x224.pt")
        preprocess_domain(domain_dir, class_to_idx, out_path)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
