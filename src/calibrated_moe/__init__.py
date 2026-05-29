"""Distributionally robust training for Mixture of Experts."""

from calibrated_moe.models import (
    MoE,
    SingleExpert,
    ResNetBackbone,
    DistilBERTBackbone,
    ViTBackbone,
    get_backbone,
    set_seed,
)
from calibrated_moe.datasets import (
    get_dataset,
    AgreementDataset,
    load_cifar10h,
    compute_agreement_scores,
    PACS_DOMAINS,
    PACS_CLASSES,
)
from calibrated_moe.calibration import (
    compute_ece,
    find_optimal_temperature,
    apply_temperature_scaling,
)

__all__ = [
    "MoE",
    "SingleExpert",
    "ResNetBackbone",
    "DistilBERTBackbone",
    "ViTBackbone",
    "get_backbone",
    "set_seed",
    "get_dataset",
    "AgreementDataset",
    "load_cifar10h",
    "compute_agreement_scores",
    "PACS_DOMAINS",
    "PACS_CLASSES",
    "compute_ece",
    "find_optimal_temperature",
    "apply_temperature_scaling",
]
