from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ictpolarreal.utils.io import find_first_existing, read_image

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # keep non-training CLIs usable without PyTorch
    torch = None
    Dataset = object


@dataclass(frozen=True)
class CameraSample:
    object_name: str
    camera: str
    camera_dir: Path

    def image_path(self, stem: str) -> Path | None:
        return find_first_existing(self.camera_dir, stem)

    def light_path(self, kind: str, light_id: int) -> Path | None:
        light_dir = self.camera_dir / kind
        name = f"{light_id:06d}"
        return find_first_existing(light_dir, name)


def iter_camera_samples(data_root: str | Path) -> Iterator[CameraSample]:
    root = Path(data_root)
    for object_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if object_dir.name.startswith(".") or object_dir.name in {"calibration", "metadata"}:
            continue
        for camera_dir in sorted(object_dir.glob("cam[0-9][0-9]")):
            yield CameraSample(object_dir.name, camera_dir.name, camera_dir)


class ICTPolarRealDataset(Dataset):
    """Minimal PyTorch dataset for decomposition and relighting experiments."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        input_name: str = "static",
        target_name: str = "albedo",
        cameras: list[str] | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.samples = [
            sample
            for sample in iter_camera_samples(data_root)
            if cameras is None or sample.camera in cameras
        ]
        self.input_name = input_name
        self.target_name = target_name
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        if torch is None:
            raise ModuleNotFoundError("ICTPolarRealDataset requires PyTorch. Install with `pip install -e .`.")
        sample = self.samples[idx]
        input_path = sample.image_path(self.input_name)
        target_path = sample.image_path(self.target_name)
        if input_path is None:
            raise FileNotFoundError(f"Missing {self.input_name} for {sample.camera_dir}")
        if target_path is None:
            raise FileNotFoundError(f"Missing {self.target_name} for {sample.camera_dir}")

        image = read_image(input_path)
        target = read_image(target_path)
        mask_path = sample.image_path("mask")
        mask = read_image(mask_path, channels=1) if mask_path else np.ones(target.shape[:2] + (1,), dtype=np.float32)
        mask = np.clip(mask, 0.0, 1.0)

        return {
            "image": torch.from_numpy(image.transpose(2, 0, 1)),
            "target": torch.from_numpy(target.transpose(2, 0, 1)),
            "mask": torch.from_numpy(mask.transpose(2, 0, 1)),
            "object": sample.object_name,
            "camera": sample.camera,
        }
