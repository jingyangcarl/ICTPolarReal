from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ictpolarreal.data.training import (
    ICTPolarRealTrainingDataset,
    _normalize_vectors,
    _resize,
    _to_diffusion_range,
    _to_tensor,
    _tone_map_reinhard,
)
from ictpolarreal.utils.io import write_image


LIGHT_PROBE_MAPPING = "Light_Probe_Mapping_Main_WO_Daughter.txt"
LIGHT_ORDER = "LSX3_light_z_spiral.txt"
LIGHTING_CONVENTION = "diffusion-renderer-lightstage-v2"


@dataclass(frozen=True)
class ForwardEvaluationRecord:
    camera_key: tuple[str, str]
    lighting_type: str
    lighting_name: str
    base_index: int | None = None
    hdri_path: Path | None = None


class ICTPolarRealForwardEvaluationDataset:
    """Fixed OLAT and HDRI conditions used by the forward benchmark."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        material_root: str | Path,
        hdri_root: str | Path,
        resolution: int = 512,
        max_lights: int | None = None,
        light_start: int = 0,
        frame_layout: str = "auto",
        light_root: str | Path | None = None,
        require_polarization_reference: bool = False,
        olat_samples: int = 20,
        hdri_samples: int = 20,
        hdri_olat_lights: int = 20,
    ) -> None:
        self.base = ICTPolarRealTrainingDataset(
            data_root,
            material_root=material_root,
            resolution=resolution,
            max_lights=max_lights,
            light_start=light_start,
            frame_layout=frame_layout,
            light_root=light_root,
            require_polarization_reference=require_polarization_reference,
        )
        self.hdri_root = Path(hdri_root)
        self.hdri_olat_lights = hdri_olat_lights
        self.light_mapping, self.light_order = _load_lightstage_mapping(
            data_root,
            light_root,
        )
        self.records: list[ForwardEvaluationRecord] = []
        self.camera_keys: list[tuple[str, str]] = []
        self.static_indices: dict[tuple[str, str], int] = {}
        self.olat_indices: dict[tuple[str, str], list[int]] = {}
        self._synthesis_cache: dict[
            tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = {}

        for index, record in enumerate(self.base.records):
            key = (record.camera.object_name, record.camera.camera)
            if key not in self.camera_keys:
                self.camera_keys.append(key)
            if record.light_index is None:
                self.static_indices[key] = index
            else:
                self.olat_indices.setdefault(key, []).append(index)

        hdri_paths = _select_evenly(_find_hdri_paths(self.hdri_root), hdri_samples)
        if hdri_samples > 0 and not hdri_paths:
            raise FileNotFoundError(
                f"No HDRI files found under {self.hdri_root}. Set --hdri-root to the evaluation HDRIs."
            )

        for key in self.camera_keys:
            if key not in self.static_indices or not self.olat_indices.get(key):
                continue
            for base_index in _select_evenly(self.olat_indices[key], olat_samples):
                frame_id = self.base.records[base_index].frame_id
                assert frame_id is not None
                self.records.append(
                    ForwardEvaluationRecord(
                        key,
                        "olat",
                        f"olat_{frame_id:06d}",
                        base_index=base_index,
                    )
                )
            for hdri_path in hdri_paths:
                self.records.append(
                    ForwardEvaluationRecord(
                        key,
                        "hdri",
                        f"hdri_{_safe_name(hdri_path.stem)}",
                        hdri_path=hdri_path,
                    )
                )
        if not self.records:
            raise FileNotFoundError("No complete camera has both static references and paired OLAT captures")
        self.camera_keys = list(dict.fromkeys(record.camera_key for record in self.records))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        if record.lighting_type == "olat":
            assert record.base_index is not None
            sample = dict(self.base[record.base_index])
        else:
            assert record.hdri_path is not None
            sample = dict(self.base[self.static_indices[record.camera_key]])
            target, irradiance = self._synthesize_hdri(record.camera_key, record.hdri_path)
            sample["rgb"] = _to_tensor(_to_diffusion_range(target))
            sample["irradiance"] = _to_tensor(_to_diffusion_range(irradiance))
            sample["light_index"] = -1
            sample["frame_id"] = -1
        sample["lighting_type"] = record.lighting_type
        sample["lighting_name"] = record.lighting_name
        sample["environment_path"] = str(record.hdri_path) if record.hdri_path else ""
        return sample

    def summary(self) -> dict[str, object]:
        return {
            "samples": len(self.records),
            "cameras": len({record.camera_key for record in self.records}),
            "olat_samples": sum(record.lighting_type == "olat" for record in self.records),
            "hdri_samples": sum(record.lighting_type == "hdri" for record in self.records),
            "resolution": self.base.resolution,
        }

    def evaluation_indices(self, camera_count: int) -> list[int]:
        selected_cameras = set(self.camera_keys[:camera_count])
        return [index for index, record in enumerate(self.records) if record.camera_key in selected_cameras]

    def export_environment(self, index: int, output_root: str | Path) -> Path:
        record = self.records[index]
        if record.hdri_path is not None:
            return record.hdri_path
        assert record.base_index is not None
        base_record = self.base.records[record.base_index]
        assert base_record.light_index is not None
        output_path = (
            Path(output_root)
            / "environments"
            / LIGHTING_CONVENTION
            / f"{record.lighting_name}.exr"
        )
        if output_path.exists():
            return output_path
        environment = np.zeros((256, 512, 3), dtype=np.float32)
        label = _mapping_label(self.light_order, base_record.light_index)
        environment[self.light_mapping == label] = 1.0
        write_image(output_path, environment)
        return output_path

    def _synthesize_hdri(
        self,
        camera_key: tuple[str, str],
        hdri_path: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        if camera_key not in self._synthesis_cache:
            selected = _select_evenly(self.olat_indices[camera_key], self.hdri_olat_lights)
            static_sample = self.base[self.static_indices[camera_key]]
            output_hw = tuple(int(value) for value in static_sample["rgb"].shape[-2:])
            images = []
            directions = []
            light_indices = []
            for base_index in selected:
                record = self.base.records[base_index]
                assert record.parallel_frame is not None and record.light_index is not None
                path = record.camera.light_path("parallel", record.parallel_frame.frame_id)
                if path is None:
                    continue
                images.append(_resize(_read_hdr(path), output_hw))
                directions.append(self.base.light_directions[record.light_index])
                light_indices.append(record.light_index)
            if not images:
                raise FileNotFoundError(f"No usable OLAT images found for {camera_key}")
            camera = self.base.records[self.static_indices[camera_key]].camera
            normal = _resize(self.base._read_normal(camera), output_hw)
            self._synthesis_cache[camera_key] = (
                np.stack(images),
                np.stack(directions),
                _normalize_vectors(normal),
                np.asarray(light_indices, dtype=np.int32),
            )

        images, directions, normal, light_indices = self._synthesis_cache[camera_key]
        radiance = _sample_lightstage_environment(
            _read_hdr(hdri_path),
            light_indices,
            self.light_mapping,
            self.light_order,
        )
        luminance = radiance @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
        exposure = float(np.percentile(luminance[luminance > 0], 75)) if np.any(luminance > 0) else 1.0
        weights = radiance / max(exposure, 1e-6) / len(radiance)
        target_hdr = np.einsum("nhwc,nc->hwc", images, weights)
        cosine = np.maximum(np.einsum("nc,hwc->nhw", directions, normal), 0.0)
        irradiance = np.einsum("nhw,nc->hwc", cosine, weights)
        irradiance_scale = float(np.percentile(irradiance, 99.5))
        irradiance = np.clip(irradiance / max(irradiance_scale, 1e-8), 0.0, 1.0)
        return _tone_map_reinhard(target_hdr), irradiance.astype(np.float32)


def _find_hdri_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    extensions = {".exr", ".hdr", ".png", ".jpg", ".jpeg"}
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions
    )


def _select_evenly(values: list, count: int) -> list:
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return list(values)
    indices = np.linspace(0, len(values) - 1, count, dtype=np.int32)
    return [values[index] for index in indices]


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def _read_hdr(path: str | Path) -> np.ndarray:
    import imageio.v3 as iio

    image = np.asarray(iio.imread(path))
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / np.iinfo(image.dtype).max
    return np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _sample_lightstage_environment(
    environment: np.ndarray,
    light_indices: np.ndarray,
    mapping: np.ndarray,
    order: np.ndarray,
) -> np.ndarray:
    if environment.shape[:2] != mapping.shape:
        import cv2

        mapping = cv2.resize(
            mapping,
            environment.shape[1::-1],
            interpolation=cv2.INTER_NEAREST,
        )
    radiance = []
    for light_index in light_indices:
        pixels = environment[mapping == _mapping_label(order, int(light_index))]
        radiance.append(pixels.mean(axis=0) if len(pixels) else np.zeros(3, dtype=np.float32))
    return np.asarray(radiance, dtype=np.float32)


def _mapping_label(order: np.ndarray, light_index: int) -> int:
    if light_index < 0 or light_index >= len(order):
        raise IndexError(f"Light index {light_index} is outside the LSX z-spiral table")
    return int(order[light_index])


def _load_lightstage_mapping(
    data_root: str | Path,
    light_root: str | Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    root = Path(data_root)
    candidates = []
    if light_root:
        candidates.append(Path(light_root))
    candidates.extend(
        (
            root / "calibration",
            root / "metadata",
            root / "LSX",
            Path(__file__).resolve().parents[2] / "metadata",
            Path(__file__).resolve().parents[4] / "data" / "LSX",
        )
    )
    for candidate in candidates:
        mapping_path = candidate / LIGHT_PROBE_MAPPING
        order_path = candidate / LIGHT_ORDER
        if not mapping_path.is_file() or not order_path.is_file():
            continue
        mapping = np.loadtxt(mapping_path, dtype=np.int32)
        order = np.loadtxt(order_path, dtype=np.int32)
        if mapping.size != 256 * 512:
            raise ValueError(f"Expected 256x512 LSX mapping values in {mapping_path}")
        if order.ndim != 1 or not len(order):
            raise ValueError(f"Invalid LSX z-spiral table in {order_path}")
        return mapping.reshape(256, 512), order
    raise FileNotFoundError(
        f"Missing {LIGHT_PROBE_MAPPING} and {LIGHT_ORDER} under --light-root, "
        "the data root, or metadata/."
    )
