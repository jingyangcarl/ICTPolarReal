from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ictpolarreal.utils.io import read_image


LIGHTING_PROFILES = ("olat", "hdri", "mix")
HDRI_EXTENSIONS = {".hdr", ".exr", ".tif", ".tiff", ".png"}
DEFAULT_PROJECTION_HEIGHT = 384


@dataclass
class EnvironmentCondition:
    condition_id: str
    source_path: Path | None
    source_name: str
    source_sha256: str
    rotation_degrees: int
    split: str
    variance_score: float
    weights: np.ndarray
    preview: np.ndarray
    source_kind: str = "environment_map"


@dataclass
class EnvironmentConditions:
    train: list[EnvironmentCondition]
    evaluation: list[EnvironmentCondition]
    projection_height: int
    projection_width: int
    rotations: int
    hdri_root: Path
    fit_support_indices: np.ndarray
    evaluation_support_indices: np.ndarray

    @property
    def all(self) -> list[EnvironmentCondition]:
        return [*self.train, *self.evaluation]


def parse_lighting_profiles(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        requested = [part.strip().lower() for part in value.split(",") if part.strip()]
    else:
        requested = [str(part).strip().lower() for part in value if str(part).strip()]
    if not requested:
        raise ValueError("at least one end2end lighting profile is required")
    if "all" in requested:
        if len(requested) != 1:
            raise ValueError("end2end lighting profile 'all' cannot be combined with other values")
        return LIGHTING_PROFILES
    unknown = sorted(set(requested).difference(LIGHTING_PROFILES))
    if unknown:
        raise ValueError(
            f"unknown end2end lighting profile(s): {', '.join(unknown)}; "
            f"expected {', '.join(LIGHTING_PROFILES)} or all"
        )
    requested_set = set(requested)
    return tuple(profile for profile in LIGHTING_PROFILES if profile in requested_set)


def mix_condition_kind(step: int, rotations: int) -> str:
    if step < 0:
        raise ValueError("mix schedule step must be non-negative")
    if rotations <= 0:
        raise ValueError("HDRI rotations must be positive")
    return "hdri" if step % (2 * rotations) < rotations else "olat"


def discover_environment_maps(root: str | Path) -> list[Path]:
    directory = Path(root).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"HDRI root does not exist: {directory}")
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in HDRI_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No HDR/EXR environment maps found under {directory}")
    return files


def rank_environment_maps(
    paths: Sequence[Path], light_dirs: np.ndarray, count: int
) -> list[tuple[Path, float]]:
    if count <= 0:
        raise ValueError("requested HDRI count must be positive")
    if len(paths) < count:
        raise ValueError(f"HDRI root contains {len(paths)} maps, but {count} are required")
    directions = _normalize_vectors(light_dirs)
    ranked = []
    for path in paths:
        image = _read_environment(path)
        samples = sample_latlong_environment(image, directions)
        score = float(np.var(samples, axis=0).mean())
        ranked.append((path, score))
    ranked.sort(key=lambda item: (-item[1], item[0].name))
    return ranked[:count]


def split_ranked_environments(
    ranked: Sequence[tuple[Path, float]], train_count: int, eval_count: int
) -> tuple[list[tuple[Path, float]], list[tuple[Path, float]]]:
    if train_count <= 0:
        raise ValueError("end2end HDRI training count must be positive")
    if eval_count < 0:
        raise ValueError("end2end HDRI evaluation count must be non-negative")
    total = train_count + eval_count
    if len(ranked) < total:
        raise ValueError(f"need {total} ranked HDRIs, received {len(ranked)}")
    selected = list(ranked[:total])
    return selected[:train_count], selected[train_count:]


def prepare_environment_conditions(
    hdri_root: str | Path,
    light_dirs: np.ndarray,
    *,
    train_count: int,
    eval_count: int,
    rotations: int,
    out_dir: str | Path,
    projection_height: int = DEFAULT_PROJECTION_HEIGHT,
    fit_support_indices: Sequence[int] | np.ndarray | None = None,
    evaluation_support_indices: Sequence[int] | np.ndarray | None = None,
) -> EnvironmentConditions:
    if rotations <= 0:
        raise ValueError("end2end HDRI rotations must be positive")
    if projection_height <= 0:
        raise ValueError("HDRI projection height must be positive")
    root = Path(hdri_root).expanduser().resolve()
    n_lights = len(light_dirs)
    fit_support = _validate_support_indices(
        fit_support_indices,
        n_lights,
        "fit",
    )
    evaluation_support = _validate_support_indices(
        evaluation_support_indices,
        n_lights,
        "evaluation",
    )
    paths = discover_environment_maps(root)
    ranked = rank_environment_maps(
        paths,
        np.asarray(light_dirs)[fit_support],
        train_count + eval_count,
    )
    train_files, eval_files = split_ranked_environments(ranked, train_count, eval_count)
    projection_width = projection_height * 2
    fit_assignment, fit_solid_angles = build_voronoi_projection(
        np.asarray(light_dirs)[fit_support], projection_height, projection_width
    )
    if np.array_equal(fit_support, evaluation_support):
        evaluation_assignment = fit_assignment
        evaluation_solid_angles = fit_solid_angles
    else:
        evaluation_assignment, evaluation_solid_angles = build_voronoi_projection(
            np.asarray(light_dirs)[evaluation_support],
            projection_height,
            projection_width,
        )
    output = Path(out_dir)
    # Evaluation cases already embed the lighting thumbnail they use.  Hundreds
    # of duplicate fit-condition PNGs made the material output hard to inspect.
    stale_preview_dir = output / "previews"
    if stale_preview_dir.exists():
        shutil.rmtree(stale_preview_dir)

    def make_conditions(
        entries: Sequence[tuple[Path, float]], split: str
    ) -> list[EnvironmentCondition]:
        if split == "fit":
            support = fit_support
            assignment = fit_assignment
            solid_angles = fit_solid_angles
        else:
            support = evaluation_support
            assignment = evaluation_assignment
            solid_angles = evaluation_solid_angles
        conditions = []
        for source_path, score in entries:
            source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
            environment = _resize_environment(
                _read_environment(source_path), projection_height, projection_width
            )
            scale = float(np.quantile(environment.reshape(-1, 3), 0.995))
            normalized = environment / max(scale, 1e-8)
            for rotation_index in range(rotations):
                rotation_degrees = int(round(rotation_index * 360.0 / rotations))
                shift = int(round(rotation_index * projection_width / rotations))
                rotated = np.roll(normalized, shift=shift, axis=1)
                local_weights = project_environment_to_lights(
                    rotated,
                    assignment,
                    solid_angles,
                    len(support),
                )
                weights = np.zeros((n_lights, 3), dtype=np.float32)
                weights[support] = local_weights
                preview = np.clip(rotated, 0.0, 1.0).astype(np.float32)
                source_stem = _safe_stem(source_path.stem)
                condition_id = (
                    f"{source_stem}_{source_hash[:8]}_rot{rotation_degrees:03d}"
                )
                conditions.append(
                    EnvironmentCondition(
                        condition_id=condition_id,
                        source_path=source_path,
                        source_name=source_path.name,
                        source_sha256=source_hash,
                        rotation_degrees=rotation_degrees,
                        split=split,
                        variance_score=float(score),
                        weights=weights,
                        preview=preview,
                    )
                )
        return conditions

    def make_calibration_conditions() -> list[EnvironmentCondition]:
        conditions = []
        colors = {
            "w": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            "r": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            "g": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            "b": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        }
        for name, color in colors.items():
            environment = np.broadcast_to(
                color,
                (projection_height, projection_width, 3),
            ).copy()
            descriptor = f"generated-calibration:{name}:linear-rgb"
            source_hash = hashlib.sha256(descriptor.encode("utf-8")).hexdigest()
            for rotation_index in range(rotations):
                rotation_degrees = int(round(rotation_index * 360.0 / rotations))
                condition_id = f"calibration_{name}_rot{rotation_degrees:03d}"
                conditions.append(
                    EnvironmentCondition(
                        condition_id=condition_id,
                        source_path=None,
                        source_name=f"calibration_{name}",
                        source_sha256=source_hash,
                        rotation_degrees=rotation_degrees,
                        split="fit",
                        variance_score=0.0,
                        weights=_embed_support_weights(
                            project_environment_to_lights(
                                environment,
                                fit_assignment,
                                fit_solid_angles,
                                len(fit_support),
                            ),
                            fit_support,
                            n_lights,
                        ),
                        preview=environment,
                        source_kind="generated_calibration",
                    )
                )
        return conditions

    training = [*make_calibration_conditions(), *make_conditions(train_files, "fit")]
    evaluation = make_conditions(eval_files, "heldout")
    conditions = EnvironmentConditions(
        train=training,
        evaluation=evaluation,
        projection_height=projection_height,
        projection_width=projection_width,
        rotations=rotations,
        hdri_root=root,
        fit_support_indices=fit_support,
        evaluation_support_indices=evaluation_support,
    )
    write_environment_manifest(conditions, output)
    return conditions


def write_environment_manifest(
    conditions: EnvironmentConditions, out_dir: str | Path
) -> None:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_conditions = conditions.all
    condition_ids = [condition.condition_id for condition in all_conditions]
    if len(set(condition_ids)) != len(condition_ids):
        raise ValueError("HDRI condition ids must be unique")
    weights = np.stack([condition.weights for condition in all_conditions], axis=0)
    np.savez_compressed(
        output / "weights.npz",
        condition_ids=np.asarray(condition_ids),
        splits=np.asarray([condition.split for condition in all_conditions]),
        weights=weights,
        fit_support_indices=conditions.fit_support_indices,
        evaluation_support_indices=conditions.evaluation_support_indices,
    )
    payload = {
        "schema": "ictpolarreal.hdri-conditions.v1",
        "target_origin": "synthesized_from_measured_olat",
        "projection": {
            "type": "nearest-light spherical Voronoi with solid-angle integration",
            "height": conditions.projection_height,
            "width": conditions.projection_width,
            "support_policy": "Voronoi cells recomputed independently on each split basis",
            "fit_support_indices": [
                int(index) for index in conditions.fit_support_indices
            ],
            "evaluation_support_indices": [
                int(index) for index in conditions.evaluation_support_indices
            ],
        },
        "hdri_root": str(conditions.hdri_root),
        "rotations": conditions.rotations,
        "calibration_environments": ["w", "r", "g", "b"],
        "natural_fit_conditions": sum(
            condition.split == "fit" and condition.source_kind == "environment_map"
            for condition in all_conditions
        ),
        "calibration_fit_conditions": sum(
            condition.source_kind == "generated_calibration"
            for condition in all_conditions
        ),
        "evaluation_conditions": sum(
            condition.split == "heldout" for condition in all_conditions
        ),
        "weights": "weights.npz",
        "conditions": [
            {
                "condition_id": condition.condition_id,
                "split": condition.split,
                "source": condition.source_name,
                "source_kind": condition.source_kind,
                "source_sha256": condition.source_sha256,
                "rotation_degrees": condition.rotation_degrees,
                "variance_score": condition.variance_score,
            }
            for condition in all_conditions
        ],
    }
    (output / "conditions.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_voronoi_projection(
    light_dirs: np.ndarray, height: int, width: int, *, row_chunk: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    directions = _normalize_vectors(light_dirs)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("light directions must have shape (N,3)")
    if height <= 0 or width <= 0:
        raise ValueError("projection dimensions must be positive")
    columns = (np.arange(width, dtype=np.float32) + 0.5) / width
    theta = columns * (2.0 * math.pi) - math.pi
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    assignment = np.empty((height, width), dtype=np.int32)
    solid_angles = np.empty((height, width), dtype=np.float32)
    dphi = math.pi / height
    dtheta = 2.0 * math.pi / width
    for start in range(0, height, row_chunk):
        stop = min(start + row_chunk, height)
        rows = (np.arange(start, stop, dtype=np.float32) + 0.5) / height
        phi = rows * math.pi
        sin_phi = np.sin(phi)[:, None]
        y = np.cos(phi)[:, None]
        x = sin_phi * sin_theta[None, :]
        z = sin_phi * cos_theta[None, :]
        y_full = np.broadcast_to(y, x.shape)
        spherical = np.stack([x, y_full, z], axis=-1).reshape(-1, 3)
        scores = spherical @ directions.T
        assignment[start:stop] = np.argmax(scores, axis=1).reshape(stop - start, width)
        solid_angles[start:stop] = np.broadcast_to(
            sin_phi * (dphi * dtheta), (stop - start, width)
        )
    return assignment, solid_angles


def project_environment_to_lights(
    environment: np.ndarray,
    assignment: np.ndarray,
    solid_angles: np.ndarray,
    n_lights: int,
) -> np.ndarray:
    image = np.asarray(environment, dtype=np.float32)
    if image.shape != assignment.shape + (3,):
        raise ValueError(
            f"environment must have shape {assignment.shape + (3,)}, got {image.shape}"
        )
    if solid_angles.shape != assignment.shape:
        raise ValueError("solid-angle map must match the Voronoi assignment")
    indices = assignment.reshape(-1)
    weights = np.empty((n_lights, 3), dtype=np.float32)
    for channel in range(3):
        values = image[..., channel].reshape(-1) * solid_angles.reshape(-1)
        weights[:, channel] = np.bincount(
            indices, weights=values, minlength=n_lights
        )[:n_lights]
    return weights


def sample_latlong_environment(
    environment: np.ndarray, light_dirs: np.ndarray
) -> np.ndarray:
    image = np.asarray(environment, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("environment image must have shape (H,W,3)")
    directions = _normalize_vectors(light_dirs)
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    phi = np.arccos(np.clip(y, -1.0, 1.0))
    theta = np.arctan2(x, z)
    rows = np.clip(np.rint(phi / math.pi * (image.shape[0] - 1)), 0, image.shape[0] - 1)
    columns = np.rint((theta + math.pi) / (2.0 * math.pi) * image.shape[1]).astype(np.int64)
    columns %= image.shape[1]
    return image[rows.astype(np.int64), columns]


def _read_environment(path: Path) -> np.ndarray:
    image = read_image(path, channels=3)
    return np.maximum(
        np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    ).astype(np.float32)


def _resize_environment(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image.astype(np.float32, copy=False)
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "HDRI profile preparation requires OpenCV for float environment resizing"
        ) from exc
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError("lighting profile received a zero-length direction")
    return values / norms


def _validate_support_indices(
    indices: Sequence[int] | np.ndarray | None,
    n_lights: int,
    label: str,
) -> np.ndarray:
    if indices is None:
        return np.arange(n_lights, dtype=np.int64)
    support = np.asarray(indices, dtype=np.int64)
    if support.ndim != 1 or not len(support):
        raise ValueError(f"{label} HDRI projection support must be a non-empty vector")
    if len(np.unique(support)) != len(support):
        raise ValueError(f"{label} HDRI projection support contains duplicate indices")
    if support.min() < 0 or support.max() >= n_lights:
        raise ValueError(
            f"{label} HDRI projection support must be within 0..{n_lights - 1}"
        )
    return support


def _embed_support_weights(
    local_weights: np.ndarray,
    support_indices: np.ndarray,
    n_lights: int,
) -> np.ndarray:
    weights = np.zeros((n_lights, 3), dtype=np.float32)
    weights[support_indices] = local_weights
    return weights


def _safe_stem(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )
    return cleaned[:96] or "environment"
