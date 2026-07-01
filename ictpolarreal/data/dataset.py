from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ictpolarreal.data.polarization import separate_cross_parallel
from ictpolarreal.data.olat import paired_light_frames
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
    """PyTorch dataset for inverse decomposition and forward relighting baselines."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        input_name: str = "static",
        target_name: str = "albedo",
        input_mode: str = "image",
        target_mode: str = "image",
        material_root: str | Path | None = None,
        light_id: int | None = None,
        cameras: list[str] | None = None,
        max_samples: int | None = None,
    ) -> None:
        if input_mode not in {"image", "polarization", "gbuffer"}:
            raise ValueError("input_mode must be image, polarization, or gbuffer")
        if target_mode not in {"image", "polarization", "gbuffer"}:
            raise ValueError("target_mode must be image, polarization, or gbuffer")
        self.data_root = Path(data_root)
        self.material_root = Path(material_root) if material_root else None
        self.samples = [
            sample
            for sample in iter_camera_samples(data_root)
            if cameras is None or sample.camera in cameras
        ]
        self.input_name = input_name
        self.target_name = target_name
        self.input_mode = input_mode
        self.target_mode = target_mode
        self.light_id = light_id
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        if torch is None:
            raise ModuleNotFoundError("ICTPolarRealDataset requires PyTorch. Install with `pip install -e .`.")
        sample = self.samples[idx]
        image = self._read_mode(sample, self.input_mode, self.input_name)
        target = self._read_mode(sample, self.target_mode, self.target_name)
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

    def target_slices(self) -> list[tuple[str, slice]]:
        sample = self.samples[0]
        names = self._mode_names(self.target_mode, self.target_name)
        start = 0
        out = []
        for name in names:
            channels = self._read_named(sample, name, for_gbuffer=self.target_mode == "gbuffer").shape[-1]
            out.append((name, slice(start, start + channels)))
            start += channels
        return out

    def _read_mode(self, sample: CameraSample, mode: str, name: str) -> np.ndarray:
        if mode == "polarization":
            return self._read_polarization(sample)
        if mode == "gbuffer" and name == "gbuffer":
            return np.concatenate([self._read_named(sample, item, for_gbuffer=True) for item in self._mode_names(mode, name)], axis=-1)
        return self._read_named(sample, name, for_gbuffer=mode == "gbuffer")

    def _mode_names(self, mode: str, name: str) -> list[str]:
        if mode == "gbuffer" and name == "gbuffer":
            return ["albedo", "normal", "roughness", "specular"]
        return [name]

    def _read_polarization(self, sample: CameraSample) -> np.ndarray:
        light_id = self.light_id if self.light_id is not None else self._first_paired_light(sample)
        static = self._read_named(sample, "static")
        cross_path = sample.light_path("cross", light_id)
        parallel_path = sample.light_path("parallel", light_id)
        if cross_path is None or parallel_path is None:
            raise FileNotFoundError(f"Missing paired cross/parallel light {light_id:06d} for {sample.camera_dir}")
        cross = _prepare_image(read_image(cross_path, channels=3))
        parallel = _prepare_image(read_image(parallel_path, channels=3))
        diffuse, specular = separate_cross_parallel(cross, parallel)
        return np.concatenate([static, cross, parallel, _prepare_image(diffuse), _prepare_image(specular)], axis=-1)

    def _first_paired_light(self, sample: CameraSample) -> int:
        _, pairs = paired_light_frames(sample.camera_dir)
        if not pairs:
            raise FileNotFoundError(f"Missing paired cross/parallel OLAT images for {sample.camera_dir}")
        return pairs[0][0].frame_id

    def _read_named(self, sample: CameraSample, name: str, *, for_gbuffer: bool = False) -> np.ndarray:
        path = self._resolve_named_path(sample, name)
        if path is None:
            raise FileNotFoundError(f"Missing {name} for {sample.camera_dir}")
        channels = _gbuffer_channels(name) if for_gbuffer else 3
        image = read_image(path, channels=channels)
        return _prepare_image(image, encode_normal=name in {"normal", "diffuse_normal", "specular_normal", "tangent", "bitangent"})

    def _resolve_named_path(self, sample: CameraSample, name: str) -> Path | None:
        names = _ALIASES.get(name, [name])
        roots = [sample.camera_dir]
        if self.material_root is not None:
            roots.extend(
                [
                    self.material_root / sample.object_name / sample.camera / "brdf",
                    self.material_root / sample.object_name / sample.camera / "material_properties",
                    self.material_root / sample.object_name / sample.camera,
                ]
            )
        for root in roots:
            for stem in names:
                path = find_first_existing(root, stem)
                if path is not None:
                    return path
        return None


_ALIASES = {
    "albedo": ["albedo", "diffuse_albedo"],
    "normal": ["normal", "diffuse_normal"],
    "specular": ["specular", "specular_albedo"],
    "roughness": ["roughness"],
}


def _gbuffer_channels(name: str) -> int:
    if name in {"roughness", "specular", "specular_albedo", "anisotropy", "occlusion"}:
        return 1
    return 3


def _prepare_image(image: np.ndarray, *, encode_normal: bool = False) -> np.ndarray:
    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if encode_normal and image.min(initial=0.0) < 0.0:
        image = image * 0.5 + 0.5
    return np.clip(image, 0.0, 1.0)
