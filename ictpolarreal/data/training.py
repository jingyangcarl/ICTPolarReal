from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ictpolarreal.data.dataset import CameraSample, iter_camera_samples
from ictpolarreal.data.olat import CALIBRATED_LIGHT_COUNT, LightFrame, paired_light_frames, select_light_pairs
from ictpolarreal.processing.material_decomposition import load_light_directions
from ictpolarreal.utils.io import find_first_existing, read_image

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # keep data inspection usable without training dependencies
    torch = None
    Dataset = object


@dataclass(frozen=True)
class TrainingRecord:
    camera: CameraSample
    cross_frame: LightFrame | None
    parallel_frame: LightFrame | None

    @property
    def light_index(self) -> int | None:
        return None if self.cross_frame is None else self.cross_frame.light_index

    @property
    def frame_id(self) -> int | None:
        return None if self.parallel_frame is None else self.parallel_frame.frame_id


class ICTPolarRealTrainingDataset(Dataset):
    """RGB2X training samples prepared from ICTPolarReal OLAT captures.

    Each camera contributes one all-white/static sample when all three static
    images are available, plus one sample per calibrated cross/parallel OLAT
    pair. Returned image tensors follow the diffusion convention ``[-1, 1]``;
    masks remain in ``[0, 1]``.
    """

    def __init__(
        self,
        data_root: str | Path,
        *,
        material_root: str | Path,
        resolution: int = 512,
        max_lights: int | None = None,
        light_start: int = 0,
        frame_layout: str = "auto",
        light_root: str | Path | None = None,
        include_static: bool = True,
        require_polarization_reference: bool = False,
        max_samples: int | None = None,
    ) -> None:
        if torch is None:
            raise ModuleNotFoundError("Training requires PyTorch. Run `bash run.sh setup`.")
        if resolution < 8:
            raise ValueError("resolution must be at least 8")

        self.data_root = Path(data_root)
        self.material_root = Path(material_root)
        self.resolution = resolution
        self.light_root = Path(light_root) if light_root else None
        self.records: list[TrainingRecord] = []

        for camera in iter_camera_samples(self.data_root):
            static_paths = self._static_paths(camera)
            if include_static and all(path is not None for path in static_paths):
                self.records.append(TrainingRecord(camera, None, None))
            if require_polarization_reference and any(path is None for path in static_paths[1:]):
                raise FileNotFoundError(
                    f"Forward polarization training requires static_cross and static_parallel for {camera.camera_dir}"
                )

            _, pairs = paired_light_frames(camera.camera_dir, frame_layout=frame_layout)
            for cross_frame, parallel_frame in select_light_pairs(
                pairs,
                light_start=light_start,
                max_lights=max_lights,
            ):
                self.records.append(TrainingRecord(camera, cross_frame, parallel_frame))

        if max_samples is not None:
            self.records = self.records[:max_samples]
        if not self.records:
            raise FileNotFoundError(f"No static or paired OLAT training samples found under {self.data_root}")

        light_indices = sorted({record.light_index for record in self.records if record.light_index is not None})
        directions = load_light_directions(self.data_root, light_indices, light_root=self.light_root)
        self.light_directions = dict(zip(light_indices, directions, strict=True))

        for record in self.records:
            self._require_material_maps(record.camera)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        record = self.records[index]
        camera = record.camera
        static_path, static_cross_path, static_parallel_path = self._static_paths(camera)

        if record.light_index is None:
            if static_path is None or static_cross_path is None or static_parallel_path is None:
                raise FileNotFoundError(f"Incomplete static polarization capture for {camera.camera_dir}")
            rgb_path = static_path
            cross_path = static_cross_path
            parallel_path = static_parallel_path
        else:
            assert record.cross_frame is not None and record.parallel_frame is not None
            cross_path = camera.light_path("cross", record.cross_frame.frame_id)
            parallel_path = camera.light_path("parallel", record.parallel_frame.frame_id)
            if cross_path is None or parallel_path is None:
                raise FileNotFoundError(f"Missing OLAT pair for {camera.camera_dir}, light {record.light_index}")
            rgb_path = parallel_path

        rgb = _read_color(rgb_path)
        cross = _read_color(cross_path)
        parallel = _read_color(parallel_path)
        output_hw = _scaled_hw(rgb.shape[:2], self.resolution)

        mask_path = camera.image_path("mask")
        if mask_path is None:
            mask = np.ones(output_hw + (1,), dtype=np.float32)
        else:
            mask = _resize(read_image(mask_path, channels=1), output_hw, nearest=True)
            mask = (mask > 0.5).astype(np.float32)

        rgb = _resize(rgb, output_hw)
        cross = _resize(cross, output_hw)
        parallel = _resize(parallel, output_hw)
        albedo = _resize(self._read_material(camera, "albedo"), output_hw)
        specular = _resize(self._read_material(camera, "specular"), output_hw)
        normal_world = _resize(self._read_normal(camera), output_hw)
        normal_world = _normalize_vectors(normal_world) * mask

        rotation = _camera_rotation(self.data_root, camera)
        normal_camera = np.einsum("ij,hwi->hwj", rotation, normal_world).astype(np.float32)
        normal_camera = _normalize_vectors(normal_camera)
        normal_forward = normal_camera.copy()
        normal_forward[..., 0] *= -1.0

        if record.light_index is None:
            irradiance = np.ones(output_hw + (3,), dtype=np.float32)
            light_index = -1
            frame_id = -1
        else:
            light_direction = self.light_directions[record.light_index]
            cosine = np.maximum(np.sum(normal_world * light_direction, axis=-1, keepdims=True), 0.0)
            irradiance = np.repeat(cosine / CALIBRATED_LIGHT_COUNT, 3, axis=-1)
            irradiance = _to_diffusion_range(irradiance)
            light_index = record.light_index
            assert record.frame_id is not None
            frame_id = record.frame_id

        reference_cross = _read_optional_color(static_cross_path, cross)
        reference_parallel = _read_optional_color(static_parallel_path, parallel)

        return {
            "rgb": _to_tensor(_to_diffusion_range(rgb)),
            "albedo": _to_tensor(_to_diffusion_range(albedo)),
            "normal_inverse": _to_tensor(normal_camera),
            "normal_forward": _to_tensor(normal_forward),
            "specular": _to_tensor(_to_diffusion_range(specular)),
            "cross": _to_tensor(_to_diffusion_range(cross)),
            "parallel": _to_tensor(_to_diffusion_range(parallel)),
            "reference_cross": _to_tensor(_to_diffusion_range(_resize(reference_cross, output_hw))),
            "reference_parallel": _to_tensor(_to_diffusion_range(_resize(reference_parallel, output_hw))),
            "irradiance": _to_tensor(irradiance),
            "mask": _to_tensor(mask),
            "object": camera.object_name,
            "camera": camera.camera,
            "light_index": light_index,
            "frame_id": frame_id,
        }

    def summary(self) -> dict[str, object]:
        cameras = {(record.camera.object_name, record.camera.camera) for record in self.records}
        static_count = sum(record.light_index is None for record in self.records)
        return {
            "samples": len(self.records),
            "cameras": len(cameras),
            "static_samples": static_count,
            "olat_samples": len(self.records) - static_count,
            "resolution": self.resolution,
        }

    def _static_paths(self, camera: CameraSample) -> tuple[Path | None, Path | None, Path | None]:
        return (
            camera.image_path("static"),
            camera.image_path("static_cross"),
            camera.image_path("static_parallel"),
        )

    def _material_path(self, camera: CameraSample, name: str) -> Path | None:
        aliases = {
            "albedo": ("albedo", "diffuse_albedo"),
            "normal": ("normal", "diffuse_normal", "normal_w2c"),
            "specular": ("specular", "specular_albedo"),
        }[name]
        roots = (
            self.material_root / camera.object_name / camera.camera / "brdf",
            self.material_root / camera.object_name / camera.camera,
            camera.camera_dir,
        )
        for root in roots:
            for stem in aliases:
                path = find_first_existing(root, stem)
                if path is not None:
                    return path
        return None

    def _require_material_maps(self, camera: CameraSample) -> None:
        missing = [name for name in ("albedo", "normal", "specular") if self._material_path(camera, name) is None]
        if missing:
            raise FileNotFoundError(
                f"Missing material map(s) {', '.join(missing)} for {camera.camera_dir}. "
                "Run `bash run.sh process` first."
            )

    def _read_material(self, camera: CameraSample, name: str) -> np.ndarray:
        path = self._material_path(camera, name)
        assert path is not None
        return _read_color(path)

    def _read_normal(self, camera: CameraSample) -> np.ndarray:
        path = self._material_path(camera, "normal")
        assert path is not None
        normal = read_image(path, channels=3)
        if path.suffix.lower() != ".exr":
            normal = normal * 2.0 - 1.0
        return _normalize_vectors(normal)


def _read_optional_color(path: Path | None, fallback: np.ndarray) -> np.ndarray:
    return fallback if path is None else _read_color(path)


def _read_color(path: Path) -> np.ndarray:
    image = read_image(path, channels=3)
    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if path.suffix.lower() == ".exr" or image.max(initial=0.0) > 1.0:
        return _tone_map_reinhard(image)
    return np.clip(image, 0.0, 1.0)


def _tone_map_reinhard(image: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    finite = image[np.isfinite(image)]
    scale = float(np.percentile(finite, percentile)) if finite.size else 1.0
    normalized = np.maximum(image / max(scale, 1e-8), 0.0)
    white_squared = 11.2**2
    mapped = normalized * (1.0 + normalized / white_squared) / (1.0 + normalized + 1e-8)
    return np.clip(mapped, 0.0, 1.0).astype(np.float32)


def _scaled_hw(hw: tuple[int, int], max_side: int) -> tuple[int, int]:
    height, width = hw
    scale = max_side / max(height, width)
    scaled_height = max(8, int(round(height * scale / 8.0)) * 8)
    scaled_width = max(8, int(round(width * scale / 8.0)) * 8)
    return scaled_height, scaled_width


def _resize(image: np.ndarray, hw: tuple[int, int], *, nearest: bool = False) -> np.ndarray:
    if image.shape[:2] == hw:
        return image.astype(np.float32)
    import cv2

    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    resized = cv2.resize(image, (hw[1], hw[0]), interpolation=interpolation)
    if resized.ndim == 2:
        resized = resized[..., None]
    return resized.astype(np.float32)


def _camera_rotation(data_root: Path, camera: CameraSample) -> np.ndarray:
    camera_id = camera.camera.removeprefix("cam")
    candidates = (
        data_root / "cameras" / f"camera{camera_id}.txt",
        camera.camera_dir / "camera.txt",
        Path(__file__).resolve().parents[2] / "metadata" / "cameras" / f"camera{camera_id}.txt",
    )
    camera_path = next((path for path in candidates if path.exists()), None)
    if camera_path is None:
        return np.eye(3, dtype=np.float32)
    lines = camera_path.read_text().splitlines()
    if len(lines) < 15:
        raise ValueError(f"Invalid camera calibration file: {camera_path}")
    return np.asarray([line.split()[:3] for line in lines[12:15]], dtype=np.float32)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, lengths, out=np.zeros_like(vectors, dtype=np.float32), where=lengths > 1e-8)


def _to_diffusion_range(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0).astype(np.float32) * 2.0 - 1.0


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
