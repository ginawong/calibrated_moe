"""Dataset loading and difficulty-aware wrappers.

Supported datasets:
  cifar10h       — CIFAR-10 with per-image human agreement (difficulty = soft-label max).
  pacs           — PACS leave-one-domain-out (difficulty = 0 on target domain).
  office_home    — Office-Home leave-one-domain-out (difficulty = 0 on target domain).
  civilcomments  — WILDS CivilComments with identity-mention difficulty.

`get_dataset(name, data_dir, **kwargs)` returns
    (trainset, testset, num_classes, difficulty_scores)
where `difficulty_scores` is a per-test-sample float in [0, 1] (lower = harder).
"""

import os
import urllib.request

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

CIFAR10H_URL = 'https://github.com/jcpeterson/cifar-10h/raw/master/data/cifar10h-probs.npy'

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PACS_DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']
PACS_CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

OFFICE_HOME_DOMAINS = ['Art', 'Clipart', 'Product', 'Real World']


# ----------------------------------------------------------------------------
# CIFAR-10H human labels
# ----------------------------------------------------------------------------

def load_cifar10h(data_dir='./data', url=CIFAR10H_URL):
    """Load CIFAR-10H human soft labels, downloading on first use."""
    os.makedirs(data_dir, exist_ok=True)
    local_path = os.path.join(data_dir, 'cifar10h-probs.npy')
    if not os.path.exists(local_path):
        print(f"Downloading CIFAR-10H labels to {local_path}...")
        urllib.request.urlretrieve(url, local_path)
    return np.load(local_path)


def compute_agreement_scores(soft_labels):
    """Per-image human agreement = max annotator probability."""
    return soft_labels.max(axis=1)


# ----------------------------------------------------------------------------
# Difficulty-aware wrapper
# ----------------------------------------------------------------------------

class AgreementDataset(torch.utils.data.Dataset):
    """Wraps a base dataset, returning (img, label, index, difficulty)."""

    def __init__(self, base_dataset, agreement, indices=None):
        self.base = base_dataset
        self.agreement = agreement
        self.indices = indices if indices is not None else np.arange(len(base_dataset))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.base[real_idx]
        agr = self.agreement[real_idx] if real_idx < len(self.agreement) else 1.0
        return img, label, real_idx, agr


# ----------------------------------------------------------------------------
# Domain-shift datasets (PACS, Office-Home)
# ----------------------------------------------------------------------------

class PreloadedDataset(torch.utils.data.Dataset):
    """Dataset backed by a preloaded image tensor (from preprocess_datasets.py)."""

    def __init__(self, images, targets, transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, int(self.targets[idx])


def _load_domain_shift_dataset(data_dir, domains, target_domain, img_size=224):
    """Leave-one-domain-out split from preprocessed `{domain}_224x224.pt` files."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(img_size, padding=img_size // 8),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    transform_test = transforms.Compose([
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_images, train_targets = [], []
    test_images, test_targets = [], []
    num_classes = None

    for domain in domains:
        pt_path = os.path.join(data_dir, f"{domain}_224x224.pt")
        if not os.path.exists(pt_path):
            raise FileNotFoundError(
                f"Preprocessed file not found: {pt_path}\n"
                f"Run: python scripts/preprocess_datasets.py --data_dir {data_dir}")

        print(f"  Loading {pt_path}...", flush=True)
        data = torch.load(pt_path, weights_only=False)
        images = data['images']
        targets = data['targets']
        num_classes = len(data['classes'])

        if domain == target_domain:
            test_images.append(images)
            test_targets.append(targets)
        else:
            train_images.append(images)
            train_targets.append(targets)

    train_images = torch.cat(train_images)
    train_targets = torch.cat(train_targets)
    test_images = torch.cat(test_images)
    test_targets = torch.cat(test_targets)

    print(f"  Train: {len(train_targets)} images from {len(domains)-1} source domains", flush=True)
    print(f"  Test:  {len(test_targets)} images from target domain '{target_domain}'", flush=True)

    # All target-domain test samples have difficulty 0 (held-out).
    difficulty = np.zeros(len(test_targets), dtype=np.float32)
    trainset = PreloadedDataset(train_images, train_targets, transform=transform_train)
    testset = PreloadedDataset(test_images, test_targets, transform=transform_test)
    return trainset, testset, num_classes, difficulty


# ----------------------------------------------------------------------------
# CivilComments
# ----------------------------------------------------------------------------

class CivilCommentsDataset(torch.utils.data.Dataset):
    """Pre-tokenized CivilComments comments."""

    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx]}
        return x, int(self.labels[idx])


def _load_civilcomments(data_dir, max_train=50000, max_len=128):
    """WILDS CivilComments with identity-mention as the hard subpopulation.

    Difficulty: 1.0 for comments with no identity mention (easy),
                0.0 for comments mentioning any identity group (hard).
    """
    from wilds import get_dataset as wilds_get_dataset
    from transformers import DistilBertTokenizer

    ds = wilds_get_dataset(dataset='civilcomments', root_dir=data_dir, download=False)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenize_split(split_name):
        split_idx = {'train': 0, 'val': 1, 'test': 2}[split_name]
        split_arr = ds.split_array if isinstance(ds.split_array, np.ndarray) else ds.split_array.numpy()
        mask = (split_arr == split_idx)
        indices = np.where(mask)[0]
        texts = [str(ds._text_array[i]) for i in indices]
        y_arr = ds.y_array if isinstance(ds.y_array, np.ndarray) else ds.y_array.numpy()
        meta_arr = ds.metadata_array if isinstance(ds.metadata_array, np.ndarray) else ds.metadata_array.numpy()
        labels = y_arr[mask]
        identity = meta_arr[mask, ds.metadata_fields.index('identity_any')]

        all_ids, all_masks = [], []
        for i in range(0, len(texts), 10000):
            chunk = texts[i:i+10000]
            tokens = tokenizer(chunk, padding='max_length', truncation=True,
                               max_length=max_len, return_tensors='pt')
            all_ids.append(tokens['input_ids'])
            all_masks.append(tokens['attention_mask'])
        input_ids = torch.cat(all_ids)
        attention_mask = torch.cat(all_masks)
        return input_ids, attention_mask, labels, identity

    print("Tokenizing CivilComments (this takes a minute)...")
    train_ids, train_mask, train_labels, _ = tokenize_split('train')

    if max_train and len(train_labels) > max_train:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(train_labels), max_train, replace=False)
        train_ids = train_ids[idx]
        train_mask = train_mask[idx]
        train_labels = train_labels[idx]

    test_ids, test_mask, test_labels, test_identity = tokenize_split('test')

    trainset = CivilCommentsDataset(train_ids, train_mask, train_labels)
    testset = CivilCommentsDataset(test_ids, test_mask, test_labels)
    difficulty = np.where(test_identity == 1, 0.0, 1.0)

    print(f"  Train: {len(trainset)}, Test: {len(testset)}")
    print(f"  Hard (identity-mentioning): {int((difficulty == 0).sum())}")
    print(f"  Easy (no identity):         {int((difficulty == 1).sum())}")

    return trainset, testset, 2, difficulty


# ----------------------------------------------------------------------------
# Top-level entry point
# ----------------------------------------------------------------------------

def get_dataset(name, data_dir='./data', **kwargs):
    """Load (trainset, testset, num_classes, difficulty_scores) for `name`.

    difficulty_scores: per-test-sample float in [0, 1] where lower = harder.

    kwargs:
        target_domain (str): held-out domain name for `pacs`/`office_home`.
        max_train (int):     train-subsample cap for `civilcomments`.
    """
    if name == 'cifar10h':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test)
        soft_labels = load_cifar10h(data_dir)
        difficulty = compute_agreement_scores(soft_labels)
        return trainset, testset, 10, difficulty

    if name == 'pacs':
        target_domain = kwargs['target_domain']
        pacs_dir = os.path.join(data_dir, 'PACS')
        return _load_domain_shift_dataset(pacs_dir, PACS_DOMAINS, target_domain, img_size=224)

    if name == 'office_home':
        target_domain = kwargs['target_domain']
        oh_dir = os.path.join(data_dir, 'office_home')
        return _load_domain_shift_dataset(oh_dir, OFFICE_HOME_DOMAINS, target_domain, img_size=224)

    if name == 'civilcomments':
        return _load_civilcomments(data_dir, max_train=kwargs.get('max_train', 50000))

    raise ValueError(f"Unknown dataset: {name}. "
                     f"Available: cifar10h, pacs, office_home, civilcomments")
