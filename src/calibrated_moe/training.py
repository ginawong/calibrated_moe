"""Training loops for the paper's seven methods.

All MoE training loops take an `MoE` model (returns (combined_probs, routing_weights))
and a DataLoader yielding (x, y, idx, difficulty). `train_single` takes a
`SingleExpert` model.

  vanilla         — ERM on cross-entropy of the aggregate mixture.
  mocae           — Trains identically to `vanilla`; per-expert temperature scaling
                    is applied post-hoc at evaluation time (see `scripts/train.py`).
  fgr             — Frequency-aware Gradient Rectification (DCT-filtered + soft-ECE projection).
  robust          — Maximum-entropy adversarial reweighting of the aggregate CE.
  robust_filtered — Robust on a routing-relevant subset, plus an ERM anchor over the full batch.
  fgr_robust      — FGR with the Robust objective in place of CE as the main loss.
  single          — ERM on cross-entropy with a single classifier (no MoE).
"""

import torch
import torch.nn.functional as F

from calibrated_moe.fgr import dct_filter_batch, soft_ece_loss, rectify_gradients


def _to_device(x, device):
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)


# ----------------------------------------------------------------------------
# Vanilla / Single
# ----------------------------------------------------------------------------

def train_vanilla(model, loader, optimizer, device):
    """ERM on the aggregate MoE cross-entropy loss."""
    model.train()
    total_loss = 0.0
    for x, y, _, _ in loader:
        x, y = _to_device(x, device), y.to(device)
        optimizer.zero_grad()
        probs, _ = model(x)
        loss = F.nll_loss(probs.log(), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_single(model, loader, optimizer, device):
    """ERM cross-entropy on a non-MoE single-classifier model."""
    model.train()
    total_loss = 0.0
    for x, y, _, _ in loader:
        x, y = _to_device(x, device), y.to(device)
        optimizer.zero_grad()
        _, logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ----------------------------------------------------------------------------
# Robust (maximum-entropy adversarial reweighting)
# ----------------------------------------------------------------------------

def train_robust(model, loader, optimizer, device, eta):
    """Robust MoE: tilted-softmax adversarial reweighting of the aggregate CE.

    weights_i = softmax(eta * loss_i),  loss = sum_i weights_i * loss_i.
    """
    model.train()
    total_loss = 0.0
    for x, y, _, _ in loader:
        x, y = _to_device(x, device), y.to(device)
        optimizer.zero_grad()
        probs, _ = model(x)
        per_sample_loss = F.nll_loss(probs.log(), y, reduction='none')
        weights = F.softmax(eta * per_sample_loss, dim=0)
        loss = (weights * per_sample_loss).sum()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ----------------------------------------------------------------------------
# Robust Filtered (routing-relevant subset + ERM anchor)
# ----------------------------------------------------------------------------

def train_robust_filtered(model, loader, optimizer, device, eta,
                          disagree_threshold=0.01, regret_threshold=1e-6):
    """Robust Filtered: ERM anchor + tilted-softmax CVaR on a routing-relevant subset.

    A sample is routing-relevant if either:
      (a) mixture regret > regret_threshold:   mix_loss - min_k expert_k_loss > tau_r
      (b) routing-weighted expert disagreement > disagree_threshold:
          sum_k r_k * || expert_k_probs - mix_probs ||^2 > tau_d
    """
    model.train()
    total_loss = 0.0
    for x, y, _, _ in loader:
        x, y = _to_device(x, device), y.to(device)
        optimizer.zero_grad()

        probs, routing = model(x)
        feat = model.backbone(x)
        mix_loss = F.nll_loss(probs.log(), y, reduction='none')

        expert_probs_list = []
        expert_losses = []
        for expert in model.experts:
            ep = F.softmax(expert(feat), dim=1)
            expert_probs_list.append(ep)
            expert_losses.append(F.nll_loss(ep.log(), y, reduction='none'))
        expert_losses = torch.stack(expert_losses, dim=1)
        expert_probs_all = torch.stack(expert_probs_list, dim=1)

        best_expert_loss = expert_losses.min(dim=1).values
        regret = F.relu(mix_loss - best_expert_loss)
        diff = expert_probs_all - probs.unsqueeze(1)
        disagreement = (routing * (diff ** 2).sum(dim=2)).sum(dim=1)
        relevant = (regret > regret_threshold) | (disagreement > disagree_threshold)

        if relevant.sum() > 0:
            relevant_loss = mix_loss[relevant]
            weights = F.softmax(eta * relevant_loss, dim=0)
            cvar_loss = (weights * relevant_loss).sum()
        else:
            cvar_loss = mix_loss.mean()

        loss = mix_loss.mean() + cvar_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ----------------------------------------------------------------------------
# FGR and FGR + Robust
# ----------------------------------------------------------------------------

def _fgr_step(model, optimizer, x, y, x_mixed, main_loss_fn):
    """Shared FGR rectification step.

    main_loss_fn(probs, y) returns the main loss term (scalar); we compute the
    main gradient on x_mixed and the calibration (soft-ECE) gradient on x.
    """
    optimizer.zero_grad()
    probs_mix, _ = model(x_mixed)
    loss_main = main_loss_fn(probs_mix, y)
    loss_main.backward()
    g_main = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    optimizer.zero_grad()
    probs_orig, _ = model(x)
    loss_calib = soft_ece_loss(probs_orig, y)
    loss_calib.backward()
    g_calib = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    optimizer.zero_grad()
    rectify_gradients(model, g_main, g_calib)
    optimizer.step()
    return loss_main.item()


def _ce_main_loss(probs, y):
    return F.nll_loss(probs.log(), y)


def _robust_main_loss(eta):
    def _loss(probs, y):
        per_sample = F.nll_loss(probs.log(), y, reduction='none')
        weights = F.softmax(eta * per_sample, dim=0)
        return (weights * per_sample).sum()
    return _loss


def train_fgr(model, loader, optimizer, device, rho=0.05,
              norm_mean=None, norm_std=None):
    """FGR (Zhang et al. 2025) with CE as the main loss."""
    return _train_fgr_like(model, loader, optimizer, device, rho,
                           norm_mean, norm_std, _ce_main_loss)


def train_fgr_robust(model, loader, optimizer, device, eta, rho=0.05,
                     norm_mean=None, norm_std=None):
    """FGR with the Robust MoE objective as the main loss."""
    return _train_fgr_like(model, loader, optimizer, device, rho,
                           norm_mean, norm_std, _robust_main_loss(eta))


def _train_fgr_like(model, loader, optimizer, device, rho, norm_mean, norm_std, main_loss_fn):
    model.train()
    total_loss = 0.0
    for x, y, _, _ in loader:
        x, y = _to_device(x, device), y.to(device)

        # Filter a fraction `rho` of the batch (images only — text passes through).
        if isinstance(x, torch.Tensor) and x.dim() == 4 and norm_mean is not None:
            bs = x.size(0)
            n_filt = max(1, int(rho * bs))
            filt_idx = torch.randperm(bs, device='cpu')[:n_filt]
            x_mixed = x.clone()
            x_mixed[filt_idx] = dct_filter_batch(x[filt_idx], norm_mean, norm_std)
        else:
            x_mixed = x

        total_loss += _fgr_step(model, optimizer, x, y, x_mixed, main_loss_fn)
    return total_loss / len(loader)
