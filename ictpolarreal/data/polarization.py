from __future__ import annotations

import numpy as np


def separate_cross_parallel(cross: np.ndarray, parallel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return diffuse and specular components from polarized captures.

    The ICTPolarReal capture uses cross and parallel polarizers. Following the
    project convention, diffuse is approximated by ``2 * cross`` and specular by
    ``2 * max(parallel - cross, 0)``.
    """
    cross = cross.astype(np.float32)
    parallel = parallel.astype(np.float32)
    diffuse = 2.0 * cross
    specular = 2.0 * np.maximum(parallel - cross, 0.0)
    return diffuse, specular


def hdr_to_ldr(hdr: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    hdr = np.nan_to_num(hdr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.percentile(hdr, percentile))
    if scale <= 1e-8:
        return np.zeros_like(hdr, dtype=np.float32)
    return np.clip(hdr / scale, 0.0, 1.0)


def apply_mask(image: np.ndarray, mask: np.ndarray | None, background: float = 1.0) -> np.ndarray:
    if mask is None:
        return image
    if mask.ndim == 2:
        mask = mask[..., None]
    mask = (mask > 0.5).astype(np.float32)
    return image * mask + float(background) * (1.0 - mask)


def synthesize_olat(images: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted OLAT synthesis.

    Args:
        images: Array shaped ``(N,H,W,C)``.
        weights: Array shaped ``(N,)`` or ``(N,C)``.
    """
    images = images.astype(np.float32)
    weights = weights.astype(np.float32)
    if weights.ndim == 1:
        weights = weights[:, None]
    if weights.ndim != 2:
        raise ValueError("weights must have shape (N,) or (N,C)")
    if images.shape[0] != weights.shape[0]:
        raise ValueError("image and weight counts must match")
    return np.einsum("nhwc,nc->hwc", images, weights)

