"""Backbones and MoE / SingleExpert classifiers."""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: do NOT touch torch.backends.cudnn.{deterministic,benchmark} here.
    # The paper's training was run with PyTorch defaults (both False); setting
    # benchmark=True lets cuDNN pick non-deterministic faster algorithms which
    # changes the training trajectory enough to perturb final ECE by ~10x.


# ----------------------------------------------------------------------------
# Backbones
# ----------------------------------------------------------------------------

class ResNetBackbone(nn.Module):
    """ResNet backbone adapted for small (32x32) or standard (224x224) images.

    Args:
        depth: 18, 34, or 50.
        small_input: True adapts the stem to 32x32 (3x3 stride-1 conv, no maxpool).
        num_blocks: Number of residual stages to use (1-4). Setting this to 3
            on ResNet-18 yields a 256-dim feature (used for CIFAR-10H); 4 yields 512.
        pretrained: Load ImageNet pretrained weights.
    """

    def __init__(self, depth=18, small_input=True, num_blocks=3, pretrained=False):
        super().__init__()
        builder = {18: torchvision.models.resnet18,
                   34: torchvision.models.resnet34,
                   50: torchvision.models.resnet50}[depth]
        weights = 'IMAGENET1K_V1' if pretrained else None
        resnet = builder(weights=weights)

        if small_input:
            resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            resnet.maxpool = nn.Identity()

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self._num_blocks = num_blocks
        all_layers = [resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]
        for i in range(num_blocks):
            setattr(self, f'layer{i+1}', all_layers[i])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Hardcode feature_dim per (depth, num_blocks). The previous implementation
        # ran a dummy forward in __init__ to infer it dynamically — but that forward
        # ran in train() mode (the constructor default), polluting BN running stats
        # with zeros before real training began. That perturbed final ECE noticeably.
        _feature_dims = {
            18: {1: 64, 2: 128, 3: 256, 4: 512},
            34: {1: 64, 2: 128, 3: 256, 4: 512},
            50: {1: 256, 2: 512, 3: 1024, 4: 2048},
        }
        self.feature_dim = _feature_dims[depth][num_blocks]

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        for i in range(self._num_blocks):
            x = getattr(self, f'layer{i+1}')(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class DistilBERTBackbone(nn.Module):
    """DistilBERT backbone for text. Returns [CLS] embedding.

    Input: dict with 'input_ids' and 'attention_mask' tensors,
           OR a single tensor of input_ids (attention_mask assumed all-1s).
    """

    def __init__(self, pretrained=True):
        super().__init__()
        from transformers import DistilBertModel
        if pretrained:
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        else:
            from transformers import DistilBertConfig
            self.bert = DistilBertModel(DistilBertConfig())
        self.feature_dim = 768

    def forward(self, x):
        if isinstance(x, dict):
            out = self.bert(input_ids=x['input_ids'], attention_mask=x['attention_mask'])
        else:
            out = self.bert(input_ids=x)
        return out.last_hidden_state[:, 0]


class ViTBackbone(nn.Module):
    """Vision Transformer (ViT-B/16). Resizes inputs <224px to 224 internally."""

    def __init__(self, pretrained=True, resize_small=True):
        super().__init__()
        weights = 'IMAGENET1K_V1' if pretrained else None
        self.vit = torchvision.models.vit_b_16(weights=weights)
        self.vit.heads = nn.Identity()
        self.feature_dim = 768
        self.resize_small = resize_small
        self._resize = transforms.Resize((224, 224), antialias=True)

    def forward(self, x):
        if self.resize_small and x.shape[-1] < 224:
            x = self._resize(x)
        return self.vit(x)


def get_backbone(name='resnet18', small_input=True, num_blocks=3, pretrained=False):
    """Create a backbone by name.

    Names: 'resnet18', 'resnet34', 'resnet50', 'distilbert', 'vit'.
    """
    if name == 'distilbert':
        return DistilBERTBackbone(pretrained=pretrained)
    if name == 'vit':
        return ViTBackbone(pretrained=pretrained, resize_small=small_input)
    depth = int(name.replace('resnet', ''))
    return ResNetBackbone(depth=depth, small_input=small_input, num_blocks=num_blocks,
                          pretrained=pretrained)


# ----------------------------------------------------------------------------
# Classifiers
# ----------------------------------------------------------------------------

class MoE(nn.Module):
    """Soft-routed mixture of K linear experts with an MLP router.

    Returns (combined_probabilities, routing_weights):
        combined: [B, num_classes] simplex (softmax-mixed expert probs).
        routing_weights: [B, num_experts] router softmax.
    """

    def __init__(self, num_experts=4, num_classes=10, hidden_dim=128, backbone=None):
        super().__init__()
        self.backbone = backbone if backbone is not None else ResNetBackbone()
        dim = self.backbone.feature_dim
        self.router = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )
        self.experts = nn.ModuleList([nn.Linear(dim, num_classes) for _ in range(num_experts)])

    def forward(self, x):
        feat = self.backbone(x)
        routing_weights = F.softmax(self.router(feat), dim=1)
        expert_probs = torch.stack([F.softmax(e(feat), dim=1) for e in self.experts], dim=1)
        combined = (routing_weights.unsqueeze(-1) * expert_probs).sum(dim=1)
        return combined, routing_weights


class SingleExpert(nn.Module):
    """Backbone + single linear classifier (the non-MoE baseline).

    Returns (probabilities, logits) for compatibility with training loops that
    need raw logits (cross-entropy).
    """

    def __init__(self, num_classes=10, backbone=None):
        super().__init__()
        self.backbone = backbone if backbone is not None else ResNetBackbone()
        self.classifier = nn.Linear(self.backbone.feature_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        return F.softmax(logits, dim=1), logits
