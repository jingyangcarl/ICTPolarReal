from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


IMAGE_EXTS = (".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def read_image(path: str | Path, *, channels: int | None = 3) -> np.ndarray:
    import imageio.v3 as iio

    arr = iio.imread(path)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[..., None]
    if channels is not None:
        if arr.shape[-1] >= channels:
            arr = arr[..., :channels]
        elif arr.shape[-1] == 1 and channels == 3:
            arr = np.repeat(arr, 3, axis=-1)
    arr = arr.astype(np.float32)
    if arr.max(initial=0) > 2.0:
        arr = arr / 255.0
    return arr


def write_image(path: str | Path, image: np.ndarray) -> None:
    import imageio.v3 as iio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0 + 0.5).astype(np.uint8)
    iio.imwrite(path, arr)


def find_first_existing(root: Path, stem: str, exts: Iterable[str] = IMAGE_EXTS) -> Path | None:
    for ext in exts:
        path = root / f"{stem}{ext}"
        if path.exists():
            return path
    return None
