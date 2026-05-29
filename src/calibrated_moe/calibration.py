"""Calibration metrics and temperature scaling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def compute_ece(probs, labels, n_bins=15):
    """Top-class Expected Calibration Error."""
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].float().mean()
            ece += prop_in_bin * torch.abs(avg_accuracy - avg_confidence)

    return ece.item()


def find_optimal_temperature(logits, labels):
    """Fit a scalar temperature on (logits, labels) via LBFGS on cross-entropy."""
    temperature = nn.Parameter(torch.ones(1, device=logits.device))
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval_loss():
        optimizer.zero_grad()
        scaled_logits = logits / temperature
        loss = F.cross_entropy(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    return temperature.item()


def apply_temperature_scaling(probs, temperature):
    """Scale probabilities by temperature (via log -> divide -> softmax)."""
    logits = probs.log()
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=1)
