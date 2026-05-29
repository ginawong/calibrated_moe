"""Evaluation helpers: prediction collection, per-expert TS, agreement-bucketed eval."""

import glob
import json

import torch
import torch.nn.functional as F

from calibrated_moe.calibration import (
    apply_temperature_scaling,
    compute_ece,
    find_optimal_temperature,
)


def _to_device(x, device):
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)


def collect_logits(model, val_loader, device, is_moe=True, per_expert_temperatures=None):
    """Collect (logits, labels) over a loader, for fitting a temperature.

    If `per_expert_temperatures` is provided (MoE only), per-expert temperature
    scaling is applied before taking the log, so that any subsequently fit
    aggregate temperature stacks on top of per-expert TS.
    """
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y, _, _ in val_loader:
            x = _to_device(x, device)
            if is_moe and per_expert_temperatures is not None:
                probs, _ = forward_with_per_expert_ts(model, x, per_expert_temperatures)
                logits = probs.log()
            elif is_moe:
                probs, _ = model(x)
                logits = probs.log()
            else:
                _, logits = model(x)
            all_logits.append(logits.cpu())
            all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


def find_per_expert_temperatures(model, val_loader, device, t_min=0.1, t_max=10.0):
    """Fit a temperature for each expert independently on the validation set.

    Returns a list of K floats (one per expert), each clamped to [t_min, t_max]
    so weak experts can't push the routed mixture to extreme confidences.
    """
    model.eval()
    all_expert_logits = []
    all_labels = []
    with torch.no_grad():
        for x, y, _, _ in val_loader:
            x = _to_device(x, device)
            feat = model.backbone(x)
            batch_expert_logits = [expert(feat).cpu() for expert in model.experts]
            if not all_expert_logits:
                all_expert_logits = [[] for _ in range(len(model.experts))]
            for i, el in enumerate(batch_expert_logits):
                all_expert_logits[i].append(el)
            all_labels.append(y)
    labels = torch.cat(all_labels)
    temps = []
    for i in range(len(model.experts)):
        logits_i = torch.cat(all_expert_logits[i])
        t_i = find_optimal_temperature(logits_i, labels)
        temps.append(max(t_min, min(t_i, t_max)))
    return temps


def forward_with_per_expert_ts(model, x, temperatures):
    """MoE forward pass that temperature-scales each expert's logits before mixing."""
    feat = model.backbone(x)
    routing_weights = F.softmax(model.router(feat), dim=1)
    expert_probs = torch.stack(
        [F.softmax(expert(feat) / temperatures[i], dim=1)
         for i, expert in enumerate(model.experts)], dim=1)
    combined = (routing_weights.unsqueeze(-1) * expert_probs).sum(dim=1)
    return combined, routing_weights


def load_mocae_temperatures(results_dir, seed):
    """Read per-expert temperatures from a mocae_s{seed}/ result JSON.

    Returns the list of K floats produced by `find_per_expert_temperatures`
    during MoCaE evaluation, or None if no such JSON exists.
    """
    paths = sorted(glob.glob(f"{results_dir}/mocae_s{seed}/mocae_s{seed}_eta*.json"))
    if not paths:
        return None
    with open(paths[0]) as f:
        return json.load(f).get('per_expert_temperatures')


def evaluate_by_difficulty(model, test_loader, device, difficulty,
                           hard_threshold, easy_threshold,
                           is_moe=True, temperature=None, per_expert_temperatures=None):
    """Evaluate, breaking down accuracy and ECE by per-sample difficulty.

    difficulty: per-test-sample float in [0, 1] (lower = harder).
    Samples with difficulty > easy_threshold are "easy"; < hard_threshold are "hard".

    Returns a dict with overall + easy/hard accuracy and ECE, plus routing entropy.
    """
    model.eval()
    all_probs, all_labels, all_indices, all_routing = [], [], [], []

    with torch.no_grad():
        for x, y, idx, _ in test_loader:
            x = _to_device(x, device)
            if is_moe and per_expert_temperatures is not None:
                probs, routing = forward_with_per_expert_ts(model, x, per_expert_temperatures)
                all_routing.append(routing.cpu())
            elif is_moe:
                probs, routing = model(x)
                all_routing.append(routing.cpu())
            else:
                probs, _ = model(x)
            all_probs.append(probs.cpu())
            all_labels.append(y)
            all_indices.append(idx)

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    indices = torch.cat(all_indices).numpy()

    if temperature is not None:
        probs = apply_temperature_scaling(probs, temperature)

    preds = probs.argmax(dim=1)
    sample_difficulty = difficulty[indices]

    overall_acc = (preds == labels).float().mean().item()
    overall_ece = compute_ece(probs, labels)

    easy_mask = torch.tensor(sample_difficulty > easy_threshold)
    hard_mask = torch.tensor(sample_difficulty < hard_threshold)
    easy_acc = (preds[easy_mask] == labels[easy_mask]).float().mean().item() if easy_mask.sum() > 0 else 0.0
    hard_acc = (preds[hard_mask] == labels[hard_mask]).float().mean().item() if hard_mask.sum() > 0 else 0.0
    easy_ece = compute_ece(probs[easy_mask], labels[easy_mask]) if easy_mask.sum() > 0 else 0.0
    hard_ece = compute_ece(probs[hard_mask], labels[hard_mask]) if hard_mask.sum() > 0 else 0.0

    result = {
        'accuracy': overall_acc,
        'ece': overall_ece,
        'easy_acc': easy_acc,
        'hard_acc': hard_acc,
        'easy_ece': easy_ece,
        'hard_ece': hard_ece,
        'n_easy': int(easy_mask.sum().item()),
        'n_hard': int(hard_mask.sum().item()),
    }
    if temperature is not None:
        result['temperature'] = temperature

    if is_moe and all_routing:
        routing = torch.cat(all_routing)
        routing = routing / (routing.sum(dim=1, keepdim=True) + 1e-10)
        result['routing_entropy'] = -(routing * (routing + 1e-10).log()).sum(dim=1).mean().item()
        if easy_mask.sum() > 0:
            result['easy_routing_entropy'] = -(routing[easy_mask] * (routing[easy_mask] + 1e-10).log()).sum(dim=1).mean().item()
        if hard_mask.sum() > 0:
            result['hard_routing_entropy'] = -(routing[hard_mask] * (routing[hard_mask] + 1e-10).log()).sum(dim=1).mean().item()

    return result
