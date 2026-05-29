"""Frequency-aware Gradient Rectification (Zhang et al. 2025).

Helpers used by `train_fgr` and `train_fgr_robust`:
  - 8x8 block DCT low-pass filtering of a fraction of each batch's images
  - Differentiable soft-ECE loss (Karandikar et al. 2021) for the calibration term
  - Gradient projection that removes the component of the classification gradient
    that conflicts with the calibration gradient (the "rectification" step)
"""

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 8x8 DCT-II / IDCT and JPEG quantization matrices
# ----------------------------------------------------------------------------

_DCT8 = None


def _get_dct8():
    global _DCT8
    if _DCT8 is None:
        n = 8
        C = np.zeros((n, n), dtype=np.float64)
        for k in range(n):
            for i in range(n):
                if k == 0:
                    C[k, i] = 1.0 / np.sqrt(n)
                else:
                    C[k, i] = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
        _DCT8 = C
    return _DCT8


_Q_LUMINANCE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float64)

_Q_CHROMINANCE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float64)


def _rgb_to_ycbcr(img):
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128.0
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128.0
    return np.stack([Y, Cb, Cr], axis=2)


def _ycbcr_to_rgb(img):
    Y, Cb, Cr = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    R = Y + 1.402 * (Cr - 128.0)
    G = Y - 0.344136 * (Cb - 128.0) - 0.714136 * (Cr - 128.0)
    B = Y + 1.772 * (Cb - 128.0)
    return np.clip(np.stack([R, G, B], axis=2), 0, 255)


def _dct_filter_image(img_255, lam):
    """8x8 block DCT low-pass filter on one [H, W, 3] image in 0..255.

    lam controls aggressiveness: lower lam => coarser quantization => more low-pass.
    """
    C = _get_dct8()
    CT = C.T
    H, W = img_255.shape[:2]
    ycbcr = _rgb_to_ycbcr(img_255.astype(np.float64))
    scale = (101.0 - lam) / 50.0
    Q_l = np.maximum(1.0, _Q_LUMINANCE * scale)
    Q_c = np.maximum(1.0, _Q_CHROMINANCE * scale)
    for ch in range(3):
        Q = Q_l if ch == 0 else Q_c
        channel = ycbcr[:, :, ch]
        for r in range(0, H - 7, 8):
            for c in range(0, W - 7, 8):
                block = channel[r:r+8, c:c+8]
                F_blk = C @ block @ CT
                F_q = np.round(F_blk / Q) * Q
                channel[r:r+8, c:c+8] = CT @ F_q @ C
        ycbcr[:, :, ch] = channel
    return _ycbcr_to_rgb(ycbcr).astype(np.float32)


def dct_filter_batch(images, norm_mean, norm_std, lam_choices=(15, 18, 25)):
    """DCT-filter a batch of normalized images, then re-normalize.

    images: [B, 3, H, W] tensor, normalized with (norm_mean, norm_std).
    Returns same shape, on the same device.
    """
    device = images.device
    mean = torch.tensor(norm_mean, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(norm_std, dtype=images.dtype).view(1, 3, 1, 1)
    imgs_01 = images.cpu() * std + mean
    imgs_255 = (imgs_01.clamp(0, 1) * 255.0).numpy().transpose(0, 2, 3, 1)

    filtered = np.empty_like(imgs_255)
    for i in range(len(imgs_255)):
        lam = int(np.random.choice(lam_choices))
        filtered[i] = _dct_filter_image(imgs_255[i], lam)

    filt_01 = torch.from_numpy(filtered / 255.0).float().permute(0, 3, 1, 2)
    filt_norm = (filt_01 - mean) / std
    return filt_norm.to(device)


# ----------------------------------------------------------------------------
# Soft-ECE loss
# ----------------------------------------------------------------------------

def soft_ece_loss(probs, labels, n_bins=15, temp=0.01):
    """Differentiable soft-ECE (Karandikar et al. 2021).

    Soft bin assignment via temperature-controlled Gaussian-kernel membership.
    """
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels).float()

    bin_centers = torch.linspace(
        0.5 / n_bins, 1.0 - 0.5 / n_bins, n_bins, device=probs.device)
    diffs = confidences.unsqueeze(1) - bin_centers.unsqueeze(0)
    membership = F.softmax(-diffs ** 2 / temp, dim=1)

    bin_weight = membership.sum(dim=0)
    bin_acc = (membership * accuracies.unsqueeze(1)).sum(dim=0)
    bin_conf = (membership * confidences.unsqueeze(1)).sum(dim=0)

    nonempty = bin_weight > 1e-8
    acc = torch.where(nonempty, bin_acc / bin_weight, torch.zeros_like(bin_weight))
    conf = torch.where(nonempty, bin_conf / bin_weight, torch.zeros_like(bin_weight))
    prop = bin_weight / (bin_weight.sum() + 1e-10)
    return torch.sqrt((prop * (acc - conf) ** 2).sum() + 1e-10)


# ----------------------------------------------------------------------------
# Gradient rectification
# ----------------------------------------------------------------------------

def rectify_gradients(model, g_main, g_calib):
    """Set model.grads to a calibration-aware combination of (g_main, g_calib).

    If g_main . g_calib < 0 (improving classification hurts calibration), project
    g_main onto the hyperplane orthogonal to g_calib; otherwise use g_main as-is.
    """
    dot = sum((g_main[n] * g_calib[n]).sum() for n in g_main if n in g_calib)
    if dot < 0:
        norm_sq = sum((g_calib[n] ** 2).sum() for n in g_calib) + 1e-10
        coeff = dot / norm_sq
        for name, param in model.named_parameters():
            if name in g_main:
                if name in g_calib:
                    param.grad = g_main[name] - coeff * g_calib[name]
                else:
                    param.grad = g_main[name]
    else:
        for name, param in model.named_parameters():
            if name in g_main:
                param.grad = g_main[name]
