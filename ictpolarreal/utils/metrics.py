from __future__ import annotations

import numpy as np


def mse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    err = (pred.astype(np.float32) - target.astype(np.float32)) ** 2
    if mask is not None:
        err = err * mask.astype(np.float32)
        denom = float(mask.sum() * pred.shape[-1])
        return float(err.sum() / max(denom, 1.0))
    return float(err.mean())


def psnr(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    value = mse(pred, target, mask)
    if value <= 1e-12:
        return 99.0
    return float(-10.0 * np.log10(value))


def mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    err = np.abs(pred.astype(np.float32) - target.astype(np.float32))
    if mask is not None:
        err = err * mask.astype(np.float32)
        denom = float(mask.sum() * pred.shape[-1])
        return float(err.sum() / max(denom, 1.0))
    return float(err.mean())


def ssim_global(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Small dependency-free SSIM approximation for release smoke evaluation."""
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    if mask is not None:
        if mask.ndim == 2:
            mask = mask[..., None]
        valid = mask.astype(bool)
        if valid.sum() == 0:
            return 0.0
        pred = pred[valid.repeat(pred.shape[-1], axis=-1)].reshape(-1, pred.shape[-1])
        target = target[valid.repeat(target.shape[-1], axis=-1)].reshape(-1, target.shape[-1])
    pred = pred.reshape(-1, pred.shape[-1])
    target = target.reshape(-1, target.shape[-1])
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = pred.mean(axis=0)
    mu_y = target.mean(axis=0)
    var_x = pred.var(axis=0)
    var_y = target.var(axis=0)
    cov_xy = ((pred - mu_x) * (target - mu_y)).mean(axis=0)
    score = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return float(np.nan_to_num(score).mean())
