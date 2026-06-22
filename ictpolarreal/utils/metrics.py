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

