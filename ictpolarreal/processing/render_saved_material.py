from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ictpolarreal.data.dataset import CameraSample
from ictpolarreal.processing.end2end_acquisition import (
    FIT_MASK_RULE,
    MIN_ORIENTED_N_DOT_V,
    NORMAL_ORIENTATION_RULE,
    PERCENTILE,
    _array_sha256,
    _file_sha256,
    _foreground_mask,
    _load_imaginaire_disney,
    _normalize_vectors,
    _orient_normals_to_view,
    _safe_torch_quantile,
)
from ictpolarreal.processing.material_decomposition import (
    load_end2end_view_directions,
    load_light_directions,
)
from ictpolarreal.processing.lighting_profiles import (
    DEFAULT_PROJECTION_HEIGHT,
    _embed_support_weights,
    _read_environment,
    _resize_environment,
    _safe_stem,
    build_voronoi_projection,
    discover_environment_maps,
    project_environment_to_lights,
)
from ictpolarreal.utils.io import read_image, write_image


RENDER_MANIFEST_SCHEMA = "ictpolarreal.saved-disney-render.v4"
EXPECTED_CONDITIONS_SCHEMA = "ictpolarreal.hdri-conditions.v1"
EXPECTED_MODEL_SCHEMA = "ictpolarreal.disney-state-artifact.v1"
PRESENTATION_MASK_SCHEMA = "ictpolarreal.saved-render-presentation-mask.v2"
PRESENTATION_MASK_RULE = "soft_clean_dataset_mask_applied_once_without_n_dot_v_culling"
PRESENTATION_NORMAL_RULE = NORMAL_ORIENTATION_RULE
DEFAULT_VALIDATION_TOLERANCE = 0.0
SUPPORTED_ORIENTED_REPLAY_ADAPTERS = {
    (
        "ictpolarreal.profile-acquisition-adapter.v7",
        "ictpolarreal-frequency-consensus-v2",
    ): (
        "ictpolarreal.end2end-disney.v13",
        "ictpolarreal.end2end-checkpoint.v13",
    ),
    (
        "ictpolarreal.profile-acquisition-adapter.v7",
        "ictpolarreal-frequency-consensus-adaptive-v2",
    ): (
        "ictpolarreal.end2end-disney.v13",
        "ictpolarreal.end2end-checkpoint.v13",
    ),
    (
        "ictpolarreal.profile-acquisition-adapter.v8",
        "ictpolarreal-frequency-consensus-regularizer-v2",
    ): (
        "ictpolarreal.end2end-disney.v14",
        "ictpolarreal.end2end-checkpoint.v14",
    ),
}
SD_RENDERINGS_PRESET = "sd-renderings"
SD_RENDERINGS_SCHEMA = "ictpolarreal.sd-renderings-preset.v1"
SD_RENDERINGS_FIT_SUPPORT_PRESET = "sd-renderings-fit-support"
SD_RENDERINGS_FIT_SUPPORT_SCHEMA = (
    "ictpolarreal.sd-renderings-fit-support-preset.v1"
)
SD_RENDERINGS_PRESETS = (
    SD_RENDERINGS_PRESET,
    SD_RENDERINGS_FIT_SUPPORT_PRESET,
)
AVAILABLE_FIT_SUPPORT_SCHEMA = "ictpolarreal.available-fit-support.v1"
AVAILABLE_FIT_SUPPORT_POLICY = "recorded_available_fit_support"
SD_RENDERINGS_ROTATION_LABEL = 90
SD_RENDERINGS_CUMULATIVE_ROLLS = 2
SD_C04_LIGHTS_SHA256 = (
    "bf04ca41814c3ff04aeb71f0a6f7957fb22eb506c3e7e5aa39011f5e7b75afdd"
)
SD_C04_TRAINER_XYZ_SHA256 = (
    "19de6db8bcffe4e6a036972627e4ada0990aae066c991ca90ee4ab4b6634d5f7"
)
SD_OLAT_EXPECTED_TOP32 = (
    "hdriheaven_flipped_teufelsberg_ground_2_2k_1k.hdr",
    "hdriheaven_flipped_hansaplatz_2k_1k.hdr",
    "hdriheaven_original_hansaplatz_2k_1k.hdr",
    "hdriheaven_flipped_satara_night_2k_1k.hdr",
    "hdriheaven_flipped_night_bridge_2k_1k.hdr",
    "hdriheaven_original_studio_small_06_2k_1k.hdr",
    "hdriheaven_flipped_studio_small_02_2k_1k.hdr",
    "hdriheaven_flipped_studio_small_01_2k_1k.hdr",
    "hdriheaven_original_studio_small_01_2k_1k.hdr",
    "hdrmaps_flipped_069_hdrmaps_com_free_1k.hdr",
    "hdriheaven_original_georgentor_2k_1k.hdr",
    "hdrmaps_flipped_004_hdrmaps_com_free_1k.hdr",
    "hdriheaven_flipped_between_bridges_2k_1k.hdr",
    "hdriheaven_flipped_machine_shop_02_2k_1k.hdr",
    "hdriheaven_original_vignaioli_night_2k_1k.hdr",
    "hdrmaps_original_101_hdrmaps_com_free_1k.hdr",
    "hdriheaven_original_night_bridge_2k_1k.hdr",
    "hdriheaven_original_dresden_station_night_2k_1k.hdr",
    "hdrmaps_original_083_hdrmaps_com_free_1k.hdr",
    "hdriheaven_flipped_concrete_tunnel_02_2k_1k.hdr",
    "hdriheaven_original_aircraft_workshop_01_2k_1k.hdr",
    "hdriheaven_original_old_bus_depot_2k_1k.hdr",
    "hdriheaven_flipped_abandoned_workshop_2k_1k.hdr",
    "hdriheaven_original_studio_small_07_2k_1k.hdr",
    "hdriheaven_flipped_birbeck_street_underpass_2k_1k.hdr",
    "hdriheaven_flipped_reading_room_2k_1k.hdr",
    "hdriheaven_flipped_old_bus_depot_2k_1k.hdr",
    "hdriheaven_flipped_wooden_lounge_2k_1k.hdr",
    "hdriheaven_original_concrete_tunnel_02_2k_1k.hdr",
    "hdriheaven_flipped_storeroom_2k_1k.hdr",
    "hdriheaven_flipped_yaris_interior_garage_2k_1k.hdr",
    "hdriheaven_original_cape_hill_2k_1k.hdr",
)
SD_SHEET_REFERENCE_SHA256 = (
    "720a4e975be60f50629f72b106281b759a847c3d9b3b7b45326ce71e874a791d"
)
SD_SHEET_HISTORICAL_COMMIT = "877b6f0732cdadd55ed621771d327a574a077137"
SD_SHEET_MAP_ORDER = (
    "normal",
    "baseColor",
    "metallic",
    "roughness",
    "specular",
    "specularTint",
    "subsurface",
    "anisotropic",
    "sheen",
    "sheenTint",
    "clearcoat",
    "clearcoatGloss",
)
SD_SHEET_PANEL_ORDER = ("gt", "pred", "greyball", "chromeball")


@dataclass(frozen=True)
class RecordedLighting:
    manifest: dict[str, Any]
    conditions: tuple[dict[str, Any], ...]
    condition_ids: np.ndarray
    splits: np.ndarray
    weights: np.ndarray
    fit_support_indices: np.ndarray
    evaluation_support_indices: np.ndarray


@dataclass(frozen=True)
class ReplayInputs:
    acquisition: dict[str, Any]
    acquisition_path: Path
    model_path: Path
    height: int
    width: int
    light_ids: np.ndarray
    frame_ids: np.ndarray
    parallel_targets: np.ndarray
    light_directions: np.ndarray
    view_directions: np.ndarray
    fit_foreground: np.ndarray
    presentation_alpha: np.ndarray
    presentation_mask_path: Path
    hash_checks: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class RenderCondition:
    condition_id: str
    weights: np.ndarray
    support_indices: np.ndarray
    record: dict[str, Any]


@dataclass(frozen=True)
class PreparedPreset:
    conditions: tuple[RenderCondition, ...]
    provenance: dict[str, Any]


def parse_condition_indices(expression: str, *, count: int | None = None) -> list[int]:
    """Parse comma-separated absolute indices and ``start:stop[:step]`` ranges."""
    if not expression.strip():
        raise ValueError("condition index expression must not be empty")
    indices: list[int] = []
    seen: set[int] = set()
    for raw_token in expression.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError(f"invalid empty token in condition indices {expression!r}")
        if ":" not in token:
            values = [int(token)]
        else:
            parts = token.split(":")
            if len(parts) not in {2, 3} or not parts[0] or not parts[1]:
                raise ValueError(
                    f"condition range {token!r} must be start:stop or start:stop:step"
                )
            start, stop = int(parts[0]), int(parts[1])
            step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
            if step <= 0:
                raise ValueError(f"condition range {token!r} needs a positive step")
            values = range(start, stop, step)
        for index in values:
            if index < 0:
                raise ValueError(f"condition indices must be non-negative; got {index}")
            if count is not None and index >= count:
                raise IndexError(
                    f"condition index {index} is outside the recorded range "
                    f"0..{count - 1}"
                )
            if index not in seen:
                seen.add(index)
                indices.append(index)
    if not indices:
        raise ValueError(f"condition expression {expression!r} selected no rows")
    return indices


def load_recorded_lighting(camera_dir: str | Path) -> RecordedLighting:
    camera_dir = Path(camera_dir).expanduser().resolve()
    assets_dir = camera_dir / "evaluation" / "assets"
    conditions_path = assets_dir / "conditions.json"
    weights_path = assets_dir / "weights.npz"
    if not conditions_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            "saved-material replay requires evaluation/assets/conditions.json and "
            f"weights.npz under {camera_dir}"
        )
    manifest = json.loads(conditions_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != EXPECTED_CONDITIONS_SCHEMA:
        raise ValueError(
            f"unsupported lighting manifest schema {manifest.get('schema')!r}"
        )
    conditions_raw = manifest.get("conditions")
    if not isinstance(conditions_raw, list) or not conditions_raw:
        raise ValueError(f"{conditions_path} does not contain recorded conditions")
    conditions = tuple(dict(record) for record in conditions_raw)

    with np.load(weights_path, allow_pickle=False) as archive:
        required = {
            "condition_ids",
            "splits",
            "weights",
            "fit_support_indices",
            "evaluation_support_indices",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"{weights_path} is missing arrays: {missing}")
        condition_ids = np.asarray(archive["condition_ids"]).copy()
        splits = np.asarray(archive["splits"]).copy()
        weights = np.asarray(archive["weights"], dtype=np.float32).copy()
        fit_support = np.asarray(
            archive["fit_support_indices"], dtype=np.int64
        ).copy()
        evaluation_support = np.asarray(
            archive["evaluation_support_indices"], dtype=np.int64
        ).copy()

    count = len(conditions)
    if condition_ids.shape != (count,) or splits.shape != (count,):
        raise ValueError("lighting archive row metadata does not match conditions.json")
    if weights.ndim != 3 or weights.shape[0] != count or weights.shape[2] != 3:
        raise ValueError(
            f"lighting weights must have shape ({count},N,3); got {weights.shape}"
        )
    json_ids = np.asarray([record.get("condition_id") for record in conditions])
    json_splits = np.asarray([record.get("split") for record in conditions])
    if not np.array_equal(condition_ids, json_ids):
        raise ValueError("condition_ids in weights.npz do not match conditions.json")
    if not np.array_equal(splits, json_splits):
        raise ValueError("splits in weights.npz do not match conditions.json")
    if len(np.unique(condition_ids)) != count:
        raise ValueError("recorded condition ids must be unique")

    projection = manifest.get("projection", {})
    if not np.array_equal(
        fit_support,
        np.asarray(projection.get("fit_support_indices", []), dtype=np.int64),
    ):
        raise ValueError("fit support differs between conditions.json and weights.npz")
    if not np.array_equal(
        evaluation_support,
        np.asarray(
            projection.get("evaluation_support_indices", []), dtype=np.int64
        ),
    ):
        raise ValueError(
            "evaluation support differs between conditions.json and weights.npz"
        )
    _validate_support(fit_support, weights.shape[1], "fit")
    _validate_support(evaluation_support, weights.shape[1], "evaluation")
    return RecordedLighting(
        manifest=manifest,
        conditions=conditions,
        condition_ids=condition_ids,
        splits=splits,
        weights=weights,
        fit_support_indices=fit_support,
        evaluation_support_indices=evaluation_support,
    )


def support_for_condition(lighting: RecordedLighting, index: int) -> np.ndarray:
    if index < 0 or index >= len(lighting.conditions):
        raise IndexError(f"condition index {index} is out of range")
    split = str(lighting.splits[index])
    if split == "fit":
        return lighting.fit_support_indices
    if split in {"heldout", "evaluation"}:
        return lighting.evaluation_support_indices
    raise ValueError(f"unsupported recorded condition split {split!r} at row {index}")


def acquisition_fit_support(
    acquisition: Mapping[str, Any], n_lights: int
) -> tuple[np.ndarray, str]:
    """Return the acquisition's exact fit rows and recorded selection mode."""
    light_split = acquisition.get("light_split")
    if not isinstance(light_split, Mapping):
        raise ValueError("acquisition.json is missing light_split provenance")
    raw_indices = light_split.get("fit_stack_indices")
    if (
        not isinstance(raw_indices, list)
        or not raw_indices
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_indices
        )
    ):
        raise ValueError(
            "acquisition fit_stack_indices must be a non-empty integer list"
        )
    indices = np.asarray(raw_indices, dtype=np.int64)
    _validate_support(indices, n_lights, "acquisition fit")

    selection = light_split.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("acquisition light split is missing selection provenance")
    fit_count = selection.get("fit_count")
    if (
        isinstance(fit_count, bool)
        or not isinstance(fit_count, int)
        or fit_count <= 0
        or fit_count != len(indices)
    ):
        raise ValueError(
            "acquisition selection fit_count must match fit_stack_indices"
        )
    selection_mode = selection.get("mode")
    if not isinstance(selection_mode, str) or not selection_mode:
        raise ValueError("acquisition selection mode must be a non-empty string")

    fit_conditions = acquisition.get("fit_conditions")
    if (
        not isinstance(fit_conditions, Mapping)
        or fit_conditions.get("olat") != fit_count
    ):
        raise ValueError(
            "acquisition fit_conditions.olat must match the recorded fit support"
        )
    return indices, selection_mode


def reconstruct_light_ids(
    acquisition: Mapping[str, Any], n_lights: int
) -> np.ndarray:
    light_split = acquisition.get("light_split")
    if not isinstance(light_split, Mapping):
        raise ValueError("acquisition.json is missing light_split provenance")
    light_ids = np.full(n_lights, -1, dtype=np.int64)
    filled = np.zeros(n_lights, dtype=bool)
    for group in ("fit", "heldout", "excluded"):
        stack = np.asarray(
            light_split.get(f"{group}_stack_indices", []), dtype=np.int64
        )
        ids = np.asarray(
            light_split.get(f"{group}_light_indices", []), dtype=np.int64
        )
        if stack.shape != ids.shape:
            raise ValueError(f"{group} stack/light provenance lengths differ")
        if np.any(stack < 0) or np.any(stack >= n_lights):
            raise ValueError(f"{group} stack indices fall outside 0..{n_lights - 1}")
        if np.any(filled[stack]):
            raise ValueError(f"{group} stack provenance overlaps an earlier split")
        light_ids[stack] = ids
        filled[stack] = True
    if not bool(filled.all()):
        missing = np.flatnonzero(~filled).tolist()
        raise ValueError(f"light provenance does not cover stack rows: {missing}")
    if len(np.unique(light_ids)) != n_lights:
        raise ValueError("reconstructed light ids are not unique")
    return light_ids


def reconstruct_frame_ids(
    acquisition: Mapping[str, Any], n_lights: int
) -> np.ndarray:
    """Reconstruct capture frame IDs in the acquisition stack's native order."""
    light_split = acquisition.get("light_split")
    if not isinstance(light_split, Mapping):
        raise ValueError("acquisition.json is missing light_split provenance")
    frame_ids = np.full(n_lights, -1, dtype=np.int64)
    filled = np.zeros(n_lights, dtype=bool)
    for group in ("fit", "heldout", "excluded"):
        stack = np.asarray(
            light_split.get(f"{group}_stack_indices", []), dtype=np.int64
        )
        ids = np.asarray(
            light_split.get(f"{group}_frame_ids", []), dtype=np.int64
        )
        if stack.shape != ids.shape:
            raise ValueError(f"{group} stack/frame provenance lengths differ")
        if np.any(stack < 0) or np.any(stack >= n_lights):
            raise ValueError(f"{group} stack indices fall outside 0..{n_lights - 1}")
        if np.any(filled[stack]):
            raise ValueError(f"{group} stack provenance overlaps an earlier split")
        frame_ids[stack] = ids
        filled[stack] = True
    if not bool(filled.all()):
        missing = np.flatnonzero(~filled).tolist()
        raise ValueError(f"frame provenance does not cover stack rows: {missing}")
    if len(np.unique(frame_ids)) != n_lights:
        raise ValueError("reconstructed frame ids are not unique")
    return frame_ids


def load_parallel_targets(
    sample: CameraSample,
    frame_ids: Sequence[int] | np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Load measured parallel OLATs in the recorded acquisition stack order."""
    images: list[np.ndarray] = []
    for frame_id in np.asarray(frame_ids, dtype=np.int64):
        path = sample.light_path("parallel", int(frame_id))
        if path is None or not path.is_file():
            raise FileNotFoundError(
                f"missing parallel OLAT frame {int(frame_id)}: {path}"
            )
        image = read_image(path, channels=3)
        if image.shape != (height, width, 3):
            raise ValueError(
                f"parallel OLAT frame {int(frame_id)} has shape {image.shape}, "
                f"expected {(height, width, 3)}"
            )
        images.append(np.asarray(image, dtype=np.float32))
    stack = np.stack(images, axis=0).astype(np.float32, copy=False)
    stack = np.nan_to_num(
        stack, copy=False, nan=0.0, posinf=0.0, neginf=0.0
    )
    return np.maximum(stack, 0.0).astype(np.float32, copy=False)


def _presentation_alpha(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Preserve the clean dataset mask's soft edge for report rendering."""
    image = np.asarray(mask, dtype=np.float32)
    if image.ndim == 2:
        image = image[..., None]
    if image.shape != (height, width, 1):
        raise ValueError(
            f"presentation mask has shape {image.shape}, expected "
            f"{(height, width, 1)}"
        )
    if not np.isfinite(image).all():
        raise ValueError("presentation mask contains non-finite values")
    alpha = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)
    if not np.any(alpha > 0.0):
        raise ValueError("presentation mask has no foreground alpha")
    return np.ascontiguousarray(alpha)


def prepare_replay_inputs(
    *,
    camera_dir: str | Path,
    data_root: str | Path,
    object_name: str,
    camera: str,
    imaginaire_root: str | Path,
    lighting: RecordedLighting,
    light_root: str | Path | None = None,
) -> ReplayInputs:
    camera_dir = Path(camera_dir).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    material_dir = camera_dir / "material" / "olat"
    acquisition_path = material_dir / "acquisition.json"
    if not acquisition_path.is_file():
        raise FileNotFoundError(
            f"missing OLAT acquisition provenance: {acquisition_path}"
        )
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    if acquisition.get("lighting_profile") != "olat":
        raise ValueError(f"{acquisition_path} is not the OLAT material profile")
    signature = acquisition.get("checkpoint_signature")
    if not isinstance(signature, Mapping):
        raise ValueError(f"{acquisition_path} has no checkpoint signature")
    _validate_oriented_replay_contract(acquisition, signature)
    height = _positive_dimension(signature.get("height"), "height")
    width = _positive_dimension(signature.get("width"), "width")

    model_artifact = acquisition.get("model_artifact")
    if not isinstance(model_artifact, Mapping):
        raise ValueError(f"{acquisition_path} has no model artifact")
    if model_artifact.get("schema") != EXPECTED_MODEL_SCHEMA:
        raise ValueError(
            f"unsupported model artifact schema {model_artifact.get('schema')!r}"
        )
    model_path = material_dir / str(model_artifact.get("path", ""))
    if not model_path.is_file():
        raise FileNotFoundError(f"missing saved Disney state: {model_path}")

    checks: dict[str, dict[str, Any]] = {}
    _record_hash_check(
        checks,
        "material_state",
        _file_sha256(model_path),
        model_artifact.get("sha256"),
    )
    expected_bytes = model_artifact.get("bytes")
    if (
        not isinstance(expected_bytes, int)
        or model_path.stat().st_size != expected_bytes
    ):
        raise ValueError(
            f"saved Disney state byte count differs: got {model_path.stat().st_size}, "
            f"expected {expected_bytes}"
        )

    imaginaire_root = Path(imaginaire_root).expanduser().resolve()
    disney_source = imaginaire_root / "CookTorrance_IBL" / "disney_brdf.py"
    if not disney_source.is_file():
        raise FileNotFoundError(f"missing Imaginaire Disney source: {disney_source}")
    _record_hash_check(
        checks,
        "disney_source",
        _file_sha256(disney_source),
        acquisition.get("imaginaire", {}).get("disney_brdf_sha256"),
    )
    _record_hash_check(
        checks,
        "condition_weights",
        _array_sha256(lighting.weights),
        acquisition.get("hdri_condition_weights_sha256"),
    )

    n_lights = int(lighting.weights.shape[1])
    recorded_fit_support, _selection_mode = acquisition_fit_support(
        acquisition, n_lights
    )
    _record_hash_check(
        checks,
        "fit_support_indices",
        _array_sha256(lighting.fit_support_indices),
        _array_sha256(recorded_fit_support),
    )
    light_ids = reconstruct_light_ids(acquisition, n_lights)
    frame_ids = reconstruct_frame_ids(acquisition, n_lights)
    _record_hash_check(
        checks,
        "frame_ids",
        _array_sha256(frame_ids),
        signature.get("frame_ids_sha256"),
    )
    _record_hash_check(
        checks,
        "light_ids",
        _array_sha256(light_ids),
        signature.get("light_ids_sha256"),
    )
    light_directions = load_light_directions(
        data_root, light_ids, light_root=light_root
    )
    _record_hash_check(
        checks,
        "light_directions",
        _array_sha256(light_directions),
        signature.get("light_directions_sha256"),
    )

    sample = CameraSample(object_name, camera, data_root / object_name / camera)
    if not sample.camera_dir.is_dir():
        raise FileNotFoundError(f"camera sample does not exist: {sample.camera_dir}")
    parallel_targets_hwc = load_parallel_targets(
        sample, frame_ids, height=height, width=width
    )
    input_hashes = acquisition.get("input_hashes", {})
    _record_hash_check(
        checks,
        "raw_parallel_targets",
        _array_sha256(parallel_targets_hwc),
        input_hashes.get("raw_parallel_targets_sha256"),
    )
    parallel_targets = np.ascontiguousarray(
        parallel_targets_hwc.transpose(0, 3, 1, 2)
    )
    mask_path = sample.image_path("mask")
    normal_path = sample.image_path("normal")
    if mask_path is None or normal_path is None:
        raise FileNotFoundError(
            "exact saved-material replay requires the acquisition mask and normal"
        )
    mask_image = read_image(mask_path, channels=1)
    presentation_alpha = _presentation_alpha(mask_image, height, width)
    capture_foreground = _foreground_mask(mask_image, height, width)
    source_normal = _normalize_vectors(read_image(normal_path))
    view_directions = _normalize_vectors(
        load_end2end_view_directions(data_root, sample, (height, width))
    )
    if source_normal.shape != (height, width, 3):
        raise ValueError(
            f"acquisition normal has shape {source_normal.shape}, expected "
            f"{(height, width, 3)}"
        )
    normal, _normal_orientation = _orient_normals_to_view(
        source_normal,
        view_directions,
        foreground=capture_foreground,
    )
    foreground = capture_foreground.copy()
    for name, array in (
        ("capture_foreground", capture_foreground),
        ("source_normal", source_normal),
        ("normal", normal),
        ("view_directions", view_directions),
        ("foreground", foreground),
    ):
        _record_hash_check(
            checks,
            name,
            _array_sha256(array),
            input_hashes.get(f"{name}_sha256"),
        )
    return ReplayInputs(
        acquisition=acquisition,
        acquisition_path=acquisition_path.resolve(),
        model_path=model_path,
        height=height,
        width=width,
        light_ids=light_ids,
        frame_ids=frame_ids,
        parallel_targets=parallel_targets,
        light_directions=light_directions,
        view_directions=view_directions,
        fit_foreground=foreground,
        presentation_alpha=presentation_alpha,
        presentation_mask_path=mask_path.resolve(),
        hash_checks=checks,
    )


def _validate_oriented_replay_contract(
    acquisition: Mapping[str, Any],
    signature: Mapping[str, Any],
) -> None:
    """Reject stale or ambiguous records before reconstructing v2 fit inputs."""
    adapter = acquisition.get("adapter")
    signature_adapter = signature.get("adapter")
    if not isinstance(adapter, Mapping) or not isinstance(
        signature_adapter, Mapping
    ):
        raise ValueError("saved-material replay requires recorded adapter provenance")
    if dict(adapter) != dict(signature_adapter):
        raise ValueError(
            "acquisition and checkpoint adapter provenance differ"
        )
    identity = (adapter.get("schema"), adapter.get("algorithm_version"))
    expected_schemas = SUPPORTED_ORIENTED_REPLAY_ADAPTERS.get(identity)
    if expected_schemas is None:
        raise ValueError(
            "saved-material replay requires a supported view-oriented v2 adapter; "
            f"found {identity}"
        )
    if (acquisition.get("schema"), signature.get("schema")) != expected_schemas:
        raise ValueError(
            "saved-material replay acquisition/checkpoint schemas do not match "
            f"adapter {identity}"
        )
    if adapter.get("fit_mask_rule") != FIT_MASK_RULE:
        raise ValueError("saved-material replay adapter has the wrong fit-mask rule")
    if adapter.get("normal_orientation_rule") != NORMAL_ORIENTATION_RULE:
        raise ValueError(
            "saved-material replay adapter has the wrong normal-orientation rule"
        )
    try:
        adapter_threshold = float(adapter["minimum_oriented_n_dot_v"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "saved-material replay adapter has no valid orientation threshold"
        ) from exc
    if not math.isclose(
        adapter_threshold,
        MIN_ORIENTED_N_DOT_V,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            "saved-material replay adapter orientation threshold differs"
        )

    surface = acquisition.get("surface_validity")
    signature_surface = signature.get("surface_validity")
    if not isinstance(surface, Mapping) or not isinstance(
        signature_surface, Mapping
    ):
        raise ValueError(
            "saved-material replay requires recorded surface-validity provenance"
        )
    if dict(surface) != dict(signature_surface):
        raise ValueError(
            "acquisition and checkpoint surface-validity provenance differ"
        )
    if surface.get("fit_mask_rule") != FIT_MASK_RULE:
        raise ValueError("saved-material replay surface has the wrong fit-mask rule")
    if surface.get("normal_orientation_rule") != NORMAL_ORIENTATION_RULE:
        raise ValueError(
            "saved-material replay surface has the wrong normal-orientation rule"
        )
    try:
        surface_threshold = float(surface["minimum_n_dot_v"])
        capture_pixels = int(surface["capture_foreground_pixels"])
        fit_pixels = int(surface["front_facing_pixels"])
        excluded_pixels = int(surface["excluded_back_facing_pixels"])
        excluded_fraction = float(surface["excluded_fraction"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "saved-material replay surface-validity fields are invalid"
        ) from exc
    if not math.isclose(
        surface_threshold,
        MIN_ORIENTED_N_DOT_V,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            "saved-material replay surface orientation threshold differs"
        )
    if (
        capture_pixels <= 0
        or fit_pixels != capture_pixels
        or excluded_pixels != 0
        or excluded_fraction != 0.0
    ):
        raise ValueError(
            "saved-material replay requires every clean foreground pixel to be fitted"
        )

    input_hashes = acquisition.get("input_hashes")
    if not isinstance(input_hashes, Mapping):
        raise ValueError("saved-material replay requires recorded input hashes")
    capture_hash = input_hashes.get("capture_foreground_sha256")
    fit_hash = input_hashes.get("foreground_sha256")
    source_normal_hash = input_hashes.get("source_normal_sha256")
    if (
        not isinstance(capture_hash, str)
        or len(capture_hash) != 64
        or fit_hash != capture_hash
    ):
        raise ValueError(
            "saved-material replay fit foreground must equal the clean capture mask"
        )
    if not isinstance(source_normal_hash, str) or len(source_normal_hash) != 64:
        raise ValueError(
            "saved-material replay requires the source-normal hash"
        )


def prepare_sd_renderings_preset(
    *,
    torch,
    replay: ReplayInputs,
    lighting: RecordedLighting,
    hdri_root: str | Path,
    c04_lights_path: str | Path,
    device,
    projection_height: int = DEFAULT_PROJECTION_HEIGHT,
    lighting_preset: str = SD_RENDERINGS_PRESET,
) -> PreparedPreset:
    """Build the 36-condition SD layout under one explicit support contract.

    The source ranking deliberately follows the trainer's C04 nearest-pixel
    sampler. ``sd-renderings`` retains the exact 164-light fit basis.  The
    separately provenanced ``sd-renderings-fit-support`` mode projects over all
    fit rows actually recorded by a reduced acquisition.
    """
    root = Path(hdri_root).expanduser().resolve()
    c04_path = Path(c04_lights_path).expanduser().resolve()
    if lighting_preset not in SD_RENDERINGS_PRESETS:
        raise ValueError(f"unsupported SD lighting preset {lighting_preset!r}")
    if projection_height <= 0:
        raise ValueError("preset projection height must be positive")
    support = np.asarray(lighting.fit_support_indices, dtype=np.int64)
    n_lights = len(replay.light_directions)
    _validate_support(support, n_lights, "preset fit")
    acquisition_support, selection_mode = acquisition_fit_support(
        replay.acquisition, n_lights
    )
    if not np.array_equal(support, acquisition_support):
        raise ValueError(
            "recorded lighting fit support differs from acquisition "
            "light_split.fit_stack_indices"
        )
    support_sha256 = _array_sha256(support)
    fit_support_check = replay.hash_checks.get("fit_support_indices")
    if (
        not isinstance(fit_support_check, Mapping)
        or fit_support_check.get("passed") is not True
        or fit_support_check.get("actual_sha256") != support_sha256
        or fit_support_check.get("expected_sha256") != support_sha256
    ):
        raise ValueError(
            "saved-material preset requires a passing fit-support hash check"
        )
    if lighting_preset == SD_RENDERINGS_PRESET and len(support) != 164:
        raise ValueError(
            "sd-renderings requires the historical 164-light fit support; "
            f"this acquisition has {len(support)} rows. Use "
            f"{SD_RENDERINGS_FIT_SUPPORT_PRESET!r} for an explicitly "
            "reduced-support rendering."
        )
    paths = discover_environment_maps(root)
    c04_hash = _file_sha256(c04_path)
    if c04_hash != SD_C04_LIGHTS_SHA256:
        raise ValueError(
            f"unexpected C04 light file hash {c04_hash}; expected "
            f"{SD_C04_LIGHTS_SHA256}"
        )
    c04_directions = load_sd_c04_directions(torch, c04_path, device=device)
    c04_directions_numpy = (
        c04_directions.detach().float().cpu().numpy().astype(np.float32)
    )
    nonzero_c04_directions = int(
        np.count_nonzero(np.linalg.norm(c04_directions_numpy, axis=1) > 0.0)
    )
    if nonzero_c04_directions != 155:
        raise ValueError(
            "C04 parser must reconstruct 155 recorded directions in the "
            f"trainer's 164-row table; got {nonzero_c04_directions}"
        )
    ranked = rank_sd_environment_maps(
        torch,
        paths,
        c04_directions,
        count=len(SD_OLAT_EXPECTED_TOP32),
        device=device,
    )
    actual_names = tuple(path.name for path, _score in ranked)
    if actual_names != SD_OLAT_EXPECTED_TOP32:
        mismatch = next(
            (
                index
                for index, (actual, expected) in enumerate(
                    zip(actual_names, SD_OLAT_EXPECTED_TOP32), start=1
                )
                if actual != expected
            ),
            None,
        )
        raise RuntimeError(
            "SD C04 HDRI ranking does not match the recovered SuperDimension "
            f"cache order (first mismatch rank {mismatch}): "
            f"actual={actual_names[:8]}, expected={SD_OLAT_EXPECTED_TOP32[:8]}"
        )

    projection_width = projection_height * 2
    assignment, solid_angles = build_voronoi_projection(
        replay.light_directions[support], projection_height, projection_width
    )
    conditions: list[RenderCondition] = []
    orientation = {
        "label_rotation_degrees": SD_RENDERINGS_ROTATION_LABEL,
        "trainer_rotation_index": 1,
        "cumulative_quarter_rolls": SD_RENDERINGS_CUMULATIVE_ROLLS,
        "horizontal_shift_fraction": 0.5,
        "effective_raw_shift_degrees": 180,
        "reason": (
            "the trainer rolls in-place before every labeled rotation; its "
            "rot90 entry is the second cumulative quarter-roll"
        ),
    }
    projection: dict[str, Any] = {
        "type": "nearest-light spherical Voronoi with solid-angle integration",
        "height": int(projection_height),
        "width": int(projection_width),
        "support": (
            "ICTPolarReal OLAT fit support"
            if lighting_preset == SD_RENDERINGS_PRESET
            else "ICTPolarReal available OLAT fit support"
        ),
        "support_count": int(len(support)),
        "support_indices_sha256": support_sha256,
        "normalization": "scalar whole-environment p99.5 before projection",
    }
    condition_support_provenance: dict[str, Any] = {}
    if lighting_preset == SD_RENDERINGS_FIT_SUPPORT_PRESET:
        projection.update(
            support_policy=AVAILABLE_FIT_SUPPORT_POLICY,
            historical_lighting_exact=False,
        )
        condition_support_provenance = {
            "support_policy": AVAILABLE_FIT_SUPPORT_POLICY,
            "support_indices_sha256": support_sha256,
            "historical_lighting_exact": False,
        }

    calibration_colors = (
        ("w", np.asarray([1.0, 1.0, 1.0], dtype=np.float32)),
        ("r", np.asarray([1.0, 0.0, 0.0], dtype=np.float32)),
        ("g", np.asarray([0.0, 1.0, 0.0], dtype=np.float32)),
        ("b", np.asarray([0.0, 0.0, 1.0], dtype=np.float32)),
    )
    for preset_index, (name, color) in enumerate(calibration_colors):
        environment = np.broadcast_to(
            color, (projection_height, projection_width, 3)
        ).copy()
        local_weights = project_environment_to_lights(
            environment, assignment, solid_angles, len(support)
        )
        weights = _embed_support_weights(local_weights, support, n_lights)
        descriptor = f"generated-calibration:{name}:linear-rgb"
        source_hash = hashlib.sha256(descriptor.encode("utf-8")).hexdigest()
        condition_id = (
            f"calibration_{name}_sd_c04_rot{SD_RENDERINGS_ROTATION_LABEL:03d}"
        )
        conditions.append(
            RenderCondition(
                condition_id=condition_id,
                weights=weights,
                support_indices=support,
                record={
                    "preset_index": preset_index,
                    "absolute_index": None,
                    "condition_id": condition_id,
                    "source": f"calibration_{name}",
                    "source_sha256": source_hash,
                    "source_kind": "generated_calibration",
                    "source_rank": None,
                    "rotation_degrees": SD_RENDERINGS_ROTATION_LABEL,
                    "split": "fit",
                    "variance_score": 0.0,
                    "support_count": int(len(support)),
                    "weight_sha256": _array_sha256(weights),
                    "orientation": orientation,
                    **condition_support_provenance,
                },
            )
        )

    for source_rank, (source_path, score) in enumerate(ranked, start=1):
        source_hash = _file_sha256(source_path)
        raw = _read_environment(source_path)
        environment = _resize_environment(
            raw, projection_height, projection_width
        )
        shift = (
            SD_RENDERINGS_CUMULATIVE_ROLLS
            * projection_width
            // 4
        )
        rotated = np.roll(environment, shift=shift, axis=1)
        scale = float(np.quantile(rotated.reshape(-1, 3), 0.995))
        normalized = rotated / max(scale, 1e-8)
        local_weights = project_environment_to_lights(
            normalized, assignment, solid_angles, len(support)
        )
        weights = _embed_support_weights(local_weights, support, n_lights)
        condition_id = (
            f"{_safe_stem(source_path.stem)}_{source_hash[:8]}_"
            f"sd_c04_rot{SD_RENDERINGS_ROTATION_LABEL:03d}"
        )
        conditions.append(
            RenderCondition(
                condition_id=condition_id,
                weights=weights,
                support_indices=support,
                record={
                    "preset_index": len(conditions),
                    "absolute_index": None,
                    "condition_id": condition_id,
                    "source": source_path.name,
                    "source_sha256": source_hash,
                    "source_kind": "environment_map",
                    "source_rank": source_rank,
                    "rotation_degrees": SD_RENDERINGS_ROTATION_LABEL,
                    "split": "fit",
                    "variance_score": float(score),
                    "normalization_scale": scale,
                    "support_count": int(len(support)),
                    "weight_sha256": _array_sha256(weights),
                    "orientation": orientation,
                    **condition_support_provenance,
                },
            )
        )
    if len(conditions) != 36:
        raise RuntimeError(
            f"SD renderings preset produced {len(conditions)} conditions"
        )

    imaginaire_root = c04_path.parents[2]
    trainer_path = (
        imaginaire_root
        / "imaginaire"
        / "trainers"
        / "portrait_relighting"
        / "relighting_switchlight_pretrain.py"
    )
    sampler_path = imaginaire_root / "CookTorrance_IBL" / "CookTorrance.py"
    provenance: dict[str, Any] = {
        "schema": (
            SD_RENDERINGS_SCHEMA
            if lighting_preset == SD_RENDERINGS_PRESET
            else SD_RENDERINGS_FIT_SUPPORT_SCHEMA
        ),
        "hdri_root": str(root),
        "candidate_count": len(paths),
        "calibration_count": 4,
        "environment_count": len(ranked),
        "ranking": {
            "camera": "C04",
            "lights_path": str(c04_path),
            "lights_sha256": c04_hash,
            "light_table": "164x5 zero-filled tensor indexed by recorded IDs",
            "xyz_table_runtime_sha256": _array_sha256(c04_directions_numpy),
            "xyz_table_trainer_cpu_audit_sha256": SD_C04_TRAINER_XYZ_SHA256,
            "recorded_direction_count": nonzero_c04_directions,
            "sample": (
                "CookTorrance nearest lat-long pixel from atan2(y,x), acos(z), "
                "then trunc(u*W-0.5,v*H-0.5)"
            ),
            "score": "torch unbiased variance across 164 samples, then mean RGB",
            "candidate_order": "sorted filename; stable descending score",
            "expected_top32_guard": list(SD_OLAT_EXPECTED_TOP32),
        },
        "orientation": orientation,
        "projection": projection,
        "trainer_source": {
            "path": str(trainer_path),
            "sha256": _file_sha256(trainer_path),
        },
        "sampler_source": {
            "path": str(sampler_path),
            "sha256": _file_sha256(sampler_path),
        },
    }
    if lighting_preset == SD_RENDERINGS_FIT_SUPPORT_PRESET:
        provenance.update(
            historical_lighting_exact=False,
            available_fit_support={
                "schema": AVAILABLE_FIT_SUPPORT_SCHEMA,
                "source": "acquisition.light_split.fit_stack_indices",
                "support_policy": AVAILABLE_FIT_SUPPORT_POLICY,
                "fit_count": int(len(support)),
                "fit_stack_indices_sha256": support_sha256,
                "selection_mode": selection_mode,
                "acquisition_path": str(replay.acquisition_path.resolve()),
                "acquisition_sha256": _file_sha256(replay.acquisition_path),
            },
        )
    return PreparedPreset(tuple(conditions), provenance)


def load_sd_c04_directions(torch, path: str | Path, *, device):
    """Match Trainer.load_olat: construct trigonometry on CPU, then transfer."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing SuperDimension C04 light file: {path}")
    ids: list[int] = []
    theta: list[float] = []
    phi: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        identifier, angles = line.split(None, 1)
        first, second = angles.replace(",", " ").split()[:2]
        ids.append(int(identifier))
        theta.append(float(first))
        phi.append(float(second))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"C04 light IDs must be non-empty and unique: {path}")
    if min(ids) < 0 or max(ids) >= 164:
        raise ValueError("C04 light IDs must fit the trainer's 164-row table")
    ids_tensor = torch.tensor(ids, dtype=torch.long, device="cpu")
    theta_tensor = torch.tensor(theta, dtype=torch.float32, device="cpu")
    phi_tensor = torch.tensor(phi, dtype=torch.float32, device="cpu")
    th = torch.deg2rad(theta_tensor)
    ph = torch.deg2rad(phi_tensor)
    x = torch.cos(ph) * torch.sin(th)
    y = torch.sin(ph)
    z = torch.cos(ph) * torch.cos(th)
    table = torch.zeros((164, 5), dtype=torch.float32, device="cpu")
    table[ids_tensor] = torch.stack([theta_tensor, phi_tensor, x, y, z], dim=1)
    return table[:, 2:5].to(device=device)


def rank_sd_environment_maps(
    torch,
    paths: Sequence[Path],
    directions,
    *,
    count: int,
    device,
) -> list[tuple[Path, float]]:
    if count <= 0 or len(paths) < count:
        raise ValueError(f"cannot select {count} environments from {len(paths)} maps")
    try:
        imageio = importlib.import_module("imageio")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("SD HDRI ranking requires imageio") from exc
    ranked: list[tuple[Path, float]] = []
    with torch.inference_mode():
        for position, path in enumerate(paths, start=1):
            image = np.asarray(imageio.imread(path), dtype=np.float32)
            if image.ndim == 2:
                image = np.repeat(image[..., None], 3, axis=-1)
            if image.ndim != 3 or image.shape[-1] < 3:
                raise ValueError(f"invalid HDRI shape {image.shape} for {path}")
            environment = torch.from_numpy(
                np.ascontiguousarray(image[..., :3])
            ).to(device=device)
            samples = sample_sd_environment(torch, environment, directions)
            score = float(samples.var(dim=0).mean().item())
            ranked.append((path, score))
            if position == 1 or position % 100 == 0 or position == len(paths):
                print(
                    f"[saved-render] SD C04 ranking {position}/{len(paths)}",
                    flush=True,
                )
            del environment, samples
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:count]


def sample_sd_environment(torch, environment, directions):
    """Exact nearest-pixel sampler used by CookTorrance.sample_env_map."""
    values = directions.reshape(-1, 3)
    x_direction = values[:, 0]
    y_direction = values[:, 1]
    ratio = y_direction / x_direction
    azimuth = torch.atan(ratio)
    pi = torch.acos(torch.zeros(1, device=values.device)).item() * 2
    pi_tensor = torch.tensor([pi], device=values.device, dtype=azimuth.dtype)
    zero = torch.tensor([0.0], device=values.device, dtype=azimuth.dtype)
    azimuth = torch.where(
        (x_direction < 0) & (y_direction >= 0), azimuth + pi_tensor, azimuth
    )
    azimuth = torch.where(
        (x_direction < 0) & (y_direction < 0), azimuth - pi_tensor, azimuth
    )
    azimuth = torch.where(
        (x_direction == 0) & (y_direction > 0), pi_tensor / 2, azimuth
    )
    azimuth = torch.where(
        (x_direction == 0) & (y_direction < 0), -pi_tensor / 2, azimuth
    )
    azimuth = torch.where(
        (x_direction == 0) & (y_direction == 0), zero, azimuth
    )
    polar = torch.acos(torch.clip(values[:, 2], -1.0, 1.0))
    u = (azimuth + torch.pi) / (2 * torch.pi)
    v = polar / torch.pi
    height, width = environment.shape[:2]
    columns = (u * width - 0.5).to(torch.int32)
    rows = (v * height - 0.5).to(torch.int32)
    columns = torch.clamp(columns, 0, width - 1).long()
    rows = torch.clamp(rows, 0, height - 1).long()
    return environment[rows, columns].reshape(directions.shape[:-1] + (3,))


def recorded_render_conditions(
    lighting: RecordedLighting, indices: Sequence[int]
) -> tuple[RenderCondition, ...]:
    selected = _validate_requested_indices(indices, len(lighting.conditions))
    conditions: list[RenderCondition] = []
    for index in selected:
        source = lighting.conditions[index]
        weights = np.asarray(lighting.weights[index], dtype=np.float32)
        support = support_for_condition(lighting, index)
        conditions.append(
            RenderCondition(
                condition_id=str(source["condition_id"]),
                weights=weights,
                support_indices=support,
                record={
                    "absolute_index": int(index),
                    "condition_id": source["condition_id"],
                    "source": source.get("source"),
                    "source_sha256": source.get("source_sha256"),
                    "source_kind": source.get("source_kind"),
                    "rotation_degrees": source.get("rotation_degrees"),
                    "split": source.get("split"),
                    "variance_score": source.get("variance_score"),
                    "support_count": int(len(support)),
                    "weight_sha256": _array_sha256(weights),
                },
            )
        )
    return tuple(conditions)


def build_sheet_contract(
    raw_parallel_targets_check: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe the immutable SuperDimension sheet contract for the compositor."""
    return {
        "reference_sha256": SD_SHEET_REFERENCE_SHA256,
        "historical_commit": SD_SHEET_HISTORICAL_COMMIT,
        "columns": 12,
        "rows": 13,
        "tile_size": 768,
        "map_order": list(SD_SHEET_MAP_ORDER),
        "panel_order": list(SD_SHEET_PANEL_ORDER),
        "labels": {
            "font": "cv2.FONT_HERSHEY_SIMPLEX",
            "font_size_ratio": 0.08,
            "color_rgb": [255, 255, 0],
            "origin_x": 4,
            "baseline_bottom_px": 4,
            "line_type": "cv2.LINE_AA",
            "templates": {
                "gt": "gt #{label_number}",
                "pred": "pred #{label_number}",
                "greyball": "greyball #{label_number}",
                "chromeball": "chromeball #{label_number}",
            },
        },
        "raw_parallel_targets": dict(raw_parallel_targets_check),
    }


def _display_map_to_rgb_float(value: Any, *, height: int, width: int) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] not in {1, 3}:
        if image.shape[0] in {1, 3}:
            image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or image.shape[-1] not in {1, 3}:
        raise ValueError(f"display map must be HWC RGB-compatible; got {image.shape}")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape != (height, width, 3):
        raise ValueError(
            f"display map has shape {image.shape}, expected {(height, width, 3)}"
        )
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / float(np.iinfo(image.dtype).max)
    else:
        image = image.astype(np.float32)
    return np.clip(image, 0.0, 1.0)


def _write_material_maps(
    *,
    model,
    staging_dir: Path,
    output_dir: Path,
    camera_dir: Path,
    height: int,
    width: int,
) -> list[dict[str, Any]]:
    display_maps = model.to_display_maps(gamma=1.0)
    missing = [name for name in SD_SHEET_MAP_ORDER if name not in display_maps]
    if missing:
        raise ValueError(f"saved Disney model is missing display maps: {missing}")
    maps_dir = staging_dir / "material_maps"
    maps_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for name in SD_SHEET_MAP_ORDER:
        path = maps_dir / f"{name}.png"
        write_image(
            path,
            _display_map_to_rgb_float(
                display_maps[name], height=height, width=width
            ),
        )
        records.append(
            {
                "name": name,
                "render_path": _relative_path(
                    output_dir / "material_maps" / path.name, camera_dir
                ),
                "render_sha256": _file_sha256(path),
            }
        )
    return records


def _face_forward_presentation_parameters(
    torch, model, views, presentation_alpha
) -> tuple[dict[str, Any], int]:
    """Apply the acquisition's continuous view-orientation guard idempotently."""
    parameters = dict(model._param_maps())
    normal = parameters["normal"]
    view_norm = views.norm(dim=-1, keepdim=True)
    if not bool(torch.isfinite(views).all()) or bool((view_norm <= 1e-8).any()):
        raise ValueError("presentation view directions must be finite and non-zero")
    unit_view = views / view_norm.clamp_min(1e-8)

    normal_norm = normal.norm(dim=-1, keepdim=True)
    normal_valid = torch.isfinite(normal).all(dim=-1, keepdim=True) & (
        normal_norm > 1e-8
    )
    unit_normal = torch.nan_to_num(normal) / torch.nan_to_num(
        normal_norm, nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(1e-8)
    unit_normal = torch.where(normal_valid, unit_normal, unit_view)

    source_dot = (unit_normal * unit_view).sum(dim=-1, keepdim=True)
    reflected = normal_valid & (source_dot < 0.0)
    oriented = unit_normal - 2.0 * source_dot.clamp_max(0.0) * unit_view
    oriented = oriented / oriented.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    oriented_dot = (oriented * unit_view).sum(dim=-1, keepdim=True)
    near_tangent = oriented_dot < MIN_ORIENTED_N_DOT_V
    tangent = oriented - oriented_dot * unit_view
    tangent_direction = tangent / tangent.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    lifted = (
        tangent_direction * math.sqrt(1.0 - MIN_ORIENTED_N_DOT_V**2)
        + MIN_ORIENTED_N_DOT_V * unit_view
    )
    oriented = torch.where(near_tangent, lifted, oriented)
    oriented = oriented / oriented.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    parameters["normal"] = oriented

    changed = (~normal_valid) | reflected | near_tangent
    visible_changed = changed & (presentation_alpha > 0.0)
    return parameters, int(visible_changed.count_nonzero().item())


def _historical_probe_parameters(
    torch,
    model,
    kind: str,
    *,
    base_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Use the exact material constraints from the historical SD renderer."""
    if kind not in {"greyball", "chromeball"}:
        raise ValueError(f"unknown historical probe kind {kind!r}")
    parameters = dict(
        model._param_maps() if base_parameters is None else base_parameters
    )
    base = parameters["baseColor"]
    scalar = parameters["metallic"]
    parameters["baseColor"] = (
        torch.ones_like(base) if kind == "greyball" else torch.zeros_like(base)
    )
    values = {
        "metallic": 0.0 if kind == "greyball" else 0.8,
        "subsurface": 0.0,
        "specular": 0.0 if kind == "greyball" else 1.0,
        "roughness": 1.0 if kind == "greyball" else 0.2,
        "specularTint": 0.0,
        "anisotropic": 0.0,
        "sheen": 0.0,
        "sheenTint": 0.0,
        "clearcoat": 0.0,
        "clearcoatGloss": 0.0,
    }
    for name, value in values.items():
        parameters[name] = torch.full_like(scalar, value)
    return parameters


def _render_measured_gt(
    *,
    torch,
    parallel_targets,
    presentation_alpha,
    weights: np.ndarray,
    support_indices: np.ndarray,
) -> np.ndarray:
    """Synthesize the historical GT panel from measured parallel OLATs."""
    n_lights = int(parallel_targets.shape[0])
    values = np.asarray(weights, dtype=np.float32)
    if values.shape != (n_lights, 3):
        raise ValueError(
            f"condition weights must have shape {(n_lights, 3)}, got {values.shape}"
        )
    support = np.asarray(support_indices, dtype=np.int64)
    _validate_support(support, n_lights, "condition")
    support_tensor = torch.as_tensor(
        support, dtype=torch.long, device=parallel_targets.device
    )
    weights_support = torch.as_tensor(
        np.ascontiguousarray(values), dtype=parallel_targets.dtype,
        device=parallel_targets.device,
    ).index_select(0, support_tensor)
    raw_support = parallel_targets.index_select(0, support_tensor)
    alpha_chw = presentation_alpha.permute(2, 0, 1).contiguous()
    with torch.inference_mode():
        synthesized = torch.einsum("nc,nchw->chw", weights_support, raw_support)
        rendered = _normalize_presentation_render(synthesized, alpha_chw)
    return rendered.detach().float().cpu().permute(1, 2, 0).numpy()


def _normalize_presentation_render(render, alpha_chw):
    """Apply the clean soft alpha once, then use historical p99.5 scaling."""
    finite = (
        render.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        * alpha_chw
    )
    if not bool((alpha_chw > 0.0).any()):
        raise ValueError("cannot normalize a presentation render without alpha")
    scale = _safe_torch_quantile(
        finite.reshape(-1), PERCENTILE / 100.0
    ).clamp_min(1e-8)
    return (finite / scale).clamp_max(1.0)


def _presentation_mask_provenance(
    replay: ReplayInputs, *, faceforwarded_foreground_pixels: int
) -> dict[str, Any]:
    alpha = replay.presentation_alpha
    capture_pixels = int(np.count_nonzero(alpha[..., 0] > 0.5))
    fit_pixels = int(np.count_nonzero(replay.fit_foreground[..., 0] > 0.5))
    capture_sha256 = _array_sha256((alpha > 0.5).astype(np.float32))
    expected_capture_sha256 = replay.hash_checks["capture_foreground"][
        "actual_sha256"
    ]
    if capture_sha256 != expected_capture_sha256:
        raise RuntimeError(
            "clean presentation mask no longer reconstructs the recorded capture "
            "foreground"
        )
    fit_sha256 = replay.hash_checks["foreground"]["actual_sha256"]
    if fit_pixels != capture_pixels or fit_sha256 != capture_sha256:
        raise RuntimeError(
            "view-oriented replay requires fit foreground to equal the clean "
            "capture mask"
        )
    return {
        "schema": PRESENTATION_MASK_SCHEMA,
        "fit_rule": FIT_MASK_RULE,
        "fit_foreground_sha256": fit_sha256,
        "presentation_rule": PRESENTATION_MASK_RULE,
        "presentation_normal_rule": PRESENTATION_NORMAL_RULE,
        "presentation_mask_path": str(replay.presentation_mask_path),
        "presentation_mask_file_sha256": _file_sha256(
            replay.presentation_mask_path
        ),
        "presentation_alpha_sha256": _array_sha256(alpha),
        "capture_foreground_sha256": capture_sha256,
        "capture_foreground_pixels": capture_pixels,
        "fit_foreground_pixels": fit_pixels,
        "restored_foreground_pixels": 0,
        "fractional_alpha_pixels": int(
            np.count_nonzero((alpha > 0.0) & (alpha < 1.0))
        ),
        "faceforwarded_foreground_pixels": faceforwarded_foreground_pixels,
    }


def _install_staged_directory(
    staging_dir: Path,
    output_dir: Path,
    *,
    replace_output: bool,
) -> None:
    """Install a complete render stage while preserving any prior stage."""
    if not staging_dir.is_dir():
        raise FileNotFoundError(f"render staging directory is missing: {staging_dir}")
    if output_dir.exists() and not replace_output:
        raise FileExistsError(
            f"render output already exists: {output_dir}; pass --replace-output"
        )

    backup_dir = output_dir.with_name(
        f"{output_dir.name}.previous-{os.getpid()}"
    )
    if backup_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite an earlier render backup: {backup_dir}"
        )

    had_previous = output_dir.exists()
    if had_previous:
        os.replace(output_dir, backup_dir)
    try:
        os.replace(staging_dir, output_dir)
    except Exception as install_error:
        if had_previous:
            try:
                os.replace(backup_dir, output_dir)
            except Exception as restore_error:
                raise RuntimeError(
                    "failed to install the new render stage and could not restore "
                    f"the prior stage; it remains at {backup_dir}"
                ) from restore_error
        raise install_error
    if had_previous:
        shutil.rmtree(backup_dir)


def render_saved_material(
    *,
    camera_dir: str | Path,
    data_root: str | Path,
    object_name: str,
    camera: str,
    imaginaire_root: str | Path,
    output_dir: str | Path,
    condition_indices: Sequence[int] | None = None,
    lighting_preset: str | None = None,
    hdri_root: str | Path | None = None,
    sd_c04_lights_path: str | Path | None = None,
    validation_condition_index: int | None = None,
    light_root: str | Path | None = None,
    light_chunk: int | None = None,
    validation_tolerance: float = DEFAULT_VALIDATION_TOLERANCE,
    replace_output: bool = False,
) -> dict[str, Any]:
    import torch

    camera_dir = Path(camera_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    expected_output_dir = camera_dir / ".rendering_stage"
    if output_dir != expected_output_dir:
        raise ValueError(
            "saved SD sheet renders must use the private staging directory "
            f"{expected_output_dir}; got {output_dir}"
        )
    if output_dir.exists() and not replace_output:
        raise FileExistsError(
            f"render output already exists: {output_dir}; pass --replace-output"
        )
    if light_chunk is not None and light_chunk <= 0:
        raise ValueError("light_chunk must be positive when explicitly set")
    if validation_tolerance < 0 or not np.isfinite(validation_tolerance):
        raise ValueError("validation tolerance must be finite and non-negative")
    if (condition_indices is None) == (lighting_preset is None):
        raise ValueError(
            "choose exactly one of condition_indices or lighting_preset"
        )
    if lighting_preset is not None and lighting_preset not in SD_RENDERINGS_PRESETS:
        raise ValueError(f"unsupported lighting preset {lighting_preset!r}")
    if lighting_preset in SD_RENDERINGS_PRESETS:
        if hdri_root is None or sd_c04_lights_path is None:
            raise ValueError(
                f"{lighting_preset} requires hdri_root and sd_c04_lights_path"
            )
        if validation_condition_index is None:
            raise ValueError(
                f"{lighting_preset} requires a recorded validation condition index"
            )
    gpu = require_slurm_cuda(torch)
    lighting = load_recorded_lighting(camera_dir)
    if validation_condition_index is not None:
        _validate_requested_indices(
            [validation_condition_index], len(lighting.conditions)
        )
    replay = prepare_replay_inputs(
        camera_dir=camera_dir,
        data_root=data_root,
        object_name=object_name,
        camera=camera,
        imaginaire_root=imaginaire_root,
        lighting=lighting,
        light_root=light_root,
    )
    device = torch.device("cuda")
    preset: PreparedPreset | None = None
    if lighting_preset in SD_RENDERINGS_PRESETS:
        preset = prepare_sd_renderings_preset(
            torch=torch,
            replay=replay,
            lighting=lighting,
            hdri_root=hdri_root,
            c04_lights_path=sd_c04_lights_path,
            device=device,
            lighting_preset=lighting_preset,
        )
        render_conditions = preset.conditions
        selected_absolute_indices = None
    else:
        assert condition_indices is not None
        render_conditions = recorded_render_conditions(lighting, condition_indices)
        selected_absolute_indices = [
            int(condition.record["absolute_index"])
            for condition in render_conditions
        ]
    _root, disney_module, _source = _load_imaginaire_disney(imaginaire_root)
    config = disney_module.DisneyParamConfig(per_pixel=True, height_mode="none")
    model = disney_module.DisneyBRDFSimplifiedMultiLayer(
        replay.height, replay.width, device=device, cfg=config
    ).to(device)
    state = _load_state_dict(torch, replay.model_path)
    model.load_state_dict(state, strict=True)
    model.eval()
    del state

    lights = torch.as_tensor(
        np.ascontiguousarray(_normalize_vectors(replay.light_directions)),
        device=device,
    )
    views = torch.as_tensor(
        np.ascontiguousarray(replay.view_directions), device=device
    )
    fit_foreground = torch.as_tensor(
        np.ascontiguousarray(replay.fit_foreground), device=device
    )
    presentation_alpha = torch.as_tensor(
        np.ascontiguousarray(replay.presentation_alpha), device=device
    )
    parallel_targets = torch.as_tensor(
        np.ascontiguousarray(replay.parallel_targets), device=device
    )
    presentation_parameters, faceforwarded_foreground_pixels = (
        _face_forward_presentation_parameters(
            torch, model, views, presentation_alpha
        )
    )
    greyball_parameters = _historical_probe_parameters(
        torch,
        model,
        "greyball",
        base_parameters=presentation_parameters,
    )
    chromeball_parameters = _historical_probe_parameters(
        torch,
        model,
        "chromeball",
        base_parameters=presentation_parameters,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.with_name(f"{output_dir.name}.tmp-{os.getpid()}")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    conditions_dir = staging_dir / "conditions"
    validation_dir = staging_dir / "validation"

    ordered_records: list[dict[str, Any]] = []
    safe_condition_ids: set[str] = set()
    validation: dict[str, Any] | None = None
    try:
        conditions_dir.mkdir(parents=True, exist_ok=False)
        material_maps = _write_material_maps(
            model=model,
            staging_dir=staging_dir,
            output_dir=output_dir,
            camera_dir=camera_dir,
            height=replay.height,
            width=replay.width,
        )
        if validation_condition_index is not None:
            validation_dir.mkdir(parents=True, exist_ok=False)
            rendered = _render_condition(
                torch=torch,
                model=model,
                views=views,
                lights=lights,
                foreground=fit_foreground,
                weights=lighting.weights[validation_condition_index],
                support_indices=support_for_condition(
                    lighting, validation_condition_index
                ),
                light_chunk=light_chunk,
            )
            condition = lighting.conditions[validation_condition_index]
            validation_path = validation_dir / f"{condition['condition_id']}.png"
            write_image(validation_path, rendered)
            canonical_path = (
                camera_dir
                / "evaluation"
                / "hdri"
                / "cases"
                / str(condition["condition_id"])
                / "predictions"
                / "olat.png"
            )
            validation = _compare_rendered_png(
                validation_path,
                canonical_path,
                tolerance=validation_tolerance,
            )
            validation.update(
                {
                    "absolute_index": int(validation_condition_index),
                    "condition_id": condition["condition_id"],
                    "render_path": _relative_path(
                        output_dir / "validation" / validation_path.name,
                        camera_dir,
                    ),
                    "canonical_path": _relative_path(canonical_path, camera_dir),
                }
            )
            if not validation["passed"]:
                raise RuntimeError(
                    "saved-material replay validation differs from the acquisition "
                    f"render: max_abs={validation['max_abs']:.8f}, "
                    f"tolerance={validation_tolerance:.8f}"
                )

        for position, condition in enumerate(render_conditions, start=1):
            print(
                f"[saved-render] {position}/{len(render_conditions)} "
                f"condition={condition.condition_id}",
                flush=True,
            )
            safe_condition_id = _safe_stem(condition.condition_id)
            if safe_condition_id in safe_condition_ids:
                raise ValueError(
                    "condition ids collide after path sanitization: "
                    f"{condition.condition_id!r} -> {safe_condition_id!r}"
                )
            safe_condition_ids.add(safe_condition_id)
            condition_dir = conditions_dir / safe_condition_id
            condition_dir.mkdir(parents=False, exist_ok=False)
            gt = _render_measured_gt(
                torch=torch,
                parallel_targets=parallel_targets,
                presentation_alpha=presentation_alpha,
                weights=condition.weights,
                support_indices=condition.support_indices,
            )
            pred = _render_condition(
                torch=torch,
                model=model,
                views=views,
                lights=lights,
                foreground=presentation_alpha,
                weights=condition.weights,
                support_indices=condition.support_indices,
                light_chunk=light_chunk,
                parameters=presentation_parameters,
            )
            greyball = _render_condition(
                torch=torch,
                model=model,
                views=views,
                lights=lights,
                foreground=presentation_alpha,
                weights=condition.weights,
                support_indices=condition.support_indices,
                light_chunk=light_chunk,
                parameters=greyball_parameters,
            )
            chromeball = _render_condition(
                torch=torch,
                model=model,
                views=views,
                lights=lights,
                foreground=presentation_alpha,
                weights=condition.weights,
                support_indices=condition.support_indices,
                light_chunk=light_chunk,
                parameters=chromeball_parameters,
            )
            rendered_panels = {
                "gt": gt,
                "pred": pred,
                "greyball": greyball,
                "chromeball": chromeball,
            }
            panels: dict[str, dict[str, Any]] = {}
            for panel_name in SD_SHEET_PANEL_ORDER:
                panel_path = condition_dir / f"{panel_name}.png"
                write_image(panel_path, rendered_panels[panel_name])
                panels[panel_name] = {
                    "render_path": _relative_path(
                        output_dir
                        / "conditions"
                        / safe_condition_id
                        / panel_path.name,
                        camera_dir,
                    ),
                    "render_sha256": _file_sha256(panel_path),
                }
            record = dict(condition.record)
            record.update(
                label_number=1 + 4 * int(
                    condition.record.get("preset_index", position - 1)
                ),
                safe_condition_id=safe_condition_id,
                panels=panels,
            )
            ordered_records.append(record)

        manifest = {
            "schema": RENDER_MANIFEST_SCHEMA,
            "profile": "olat",
            "lighting_preset": lighting_preset,
            "preset_provenance": preset.provenance if preset is not None else None,
            "material": {
                "path": _relative_path(replay.model_path, camera_dir),
                "sha256": replay.hash_checks["material_state"]["actual_sha256"],
            },
            "material_maps": material_maps,
            "selected_absolute_indices": selected_absolute_indices,
            "conditions": ordered_records,
            "sheet_contract": build_sheet_contract(
                replay.hash_checks["raw_parallel_targets"]
            ),
            "validation": validation,
            "input_hash_checks": replay.hash_checks,
            "presentation_mask": _presentation_mask_provenance(
                replay,
                faceforwarded_foreground_pixels=(
                    faceforwarded_foreground_pixels
                ),
            ),
            "renderer": {
                "model": "DisneyBRDFSimplifiedMultiLayer",
                "integration": (
                    "solid-angle-integrated Voronoi RGB weights on the recorded "
                    "ICT OLAT basis with explicit unit light_weights"
                ),
                "tone_map": (
                    "pred/probes: Imaginaire linear whole-image p99.5; GT: "
                    "nonnegative measured-parallel weighted sum; sheet panels "
                    "use the clean dataset alpha exactly once before whole-image "
                    "p99.5; replay validation retains the recorded fit mask; all "
                    "clamp to [0,1]"
                ),
                "light_chunk": light_chunk,
                "exact_original_summation_order": light_chunk is None,
                "device": "cuda",
                "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "gpu": gpu,
            },
        }
        manifest_path = staging_dir / "render_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        _install_staged_directory(
            staging_dir,
            output_dir,
            replace_output=replace_output,
        )
        return manifest
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def require_slurm_cuda(torch) -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError(
            "saved Disney rendering is Slurm-only; submit this CLI with one GPU"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Slurm job {job_id} has no visible CUDA device; refusing CPU rendering"
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "saved Disney rendering expects exactly one visible CUDA device; got "
            f"{torch.cuda.device_count()}"
        )
    properties = torch.cuda.get_device_properties(0)
    total_bytes = int(properties.total_memory)
    gpu = {
        "name": str(properties.name),
        "total_memory_bytes": total_bytes,
        "total_memory_gib": round(total_bytes / float(1 << 30), 3),
    }
    print(
        f"[saved-render] Slurm job {job_id}: {gpu['name']}, "
        f"{gpu['total_memory_gib']:.1f} GiB",
        flush=True,
    )
    return gpu


def _render_condition(
    *,
    torch,
    model,
    views,
    lights,
    foreground,
    weights: np.ndarray,
    support_indices: np.ndarray,
    light_chunk: int | None,
    parameters: Mapping[str, Any] | None = None,
) -> np.ndarray:
    support = np.asarray(support_indices, dtype=np.int64)
    if np.asarray(weights).shape != (len(lights), 3):
        raise ValueError(
            f"condition weights must have shape {(len(lights), 3)}, got "
            f"{np.asarray(weights).shape}"
        )
    support_tensor = torch.as_tensor(support, dtype=torch.long, device=lights.device)
    rgb = torch.as_tensor(
        np.ascontiguousarray(weights), device=lights.device
    ).index_select(0, support_tensor)
    unit_weights = torch.ones(len(support), device=lights.device)
    render_arguments = {
        "V": views,
        "L_dir": lights.index_select(0, support_tensor),
        "L_rgb": rgb,
        "light_weights": unit_weights,
        "light_chunk": light_chunk,
        "mask": foreground,
    }
    if parameters is not None:
        render_arguments["P_detach"] = parameters
    with torch.inference_mode():
        prediction, _, _ = model(**render_arguments)
    rendered = prediction.detach().float().cpu().permute(1, 2, 0).numpy()
    # Disney applies ``foreground`` before its whole-image p99.5 tone map.
    # Multiplying again would square soft antialias values at the clean boundary.
    return np.clip(rendered, 0.0, 1.0)


def _load_state_dict(torch, path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _compare_rendered_png(
    rendered_path: Path,
    canonical_path: Path,
    *,
    tolerance: float,
) -> dict[str, Any]:
    if not canonical_path.is_file():
        raise FileNotFoundError(
            f"canonical acquisition render for validation is missing: {canonical_path}"
        )
    rendered = read_image(rendered_path)
    canonical = read_image(canonical_path)
    if rendered.shape != canonical.shape:
        raise ValueError(
            f"validation image shapes differ: {rendered.shape} vs {canonical.shape}"
        )
    difference = np.abs(rendered.astype(np.float32) - canonical.astype(np.float32))
    max_abs = float(difference.max(initial=0.0))
    return {
        "passed": bool(max_abs <= tolerance),
        "tolerance": float(tolerance),
        "max_abs": max_abs,
        "mean_abs": float(difference.mean()),
        "different_channel_values": int(np.count_nonzero(difference)),
        "render_sha256": _file_sha256(rendered_path),
        "canonical_sha256": _file_sha256(canonical_path),
        "byte_identical": bool(
            _file_sha256(rendered_path) == _file_sha256(canonical_path)
        ),
    }


def _validate_support(indices: np.ndarray, n_lights: int, label: str) -> None:
    if indices.ndim != 1 or not len(indices):
        raise ValueError(f"{label} support must be a non-empty one-dimensional array")
    if np.any(indices < 0) or np.any(indices >= n_lights):
        raise ValueError(f"{label} support falls outside the recorded light basis")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{label} support contains duplicate light rows")


def _record_hash_check(
    checks: dict[str, dict[str, Any]],
    name: str,
    actual: str,
    expected: Any,
) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"missing recorded SHA-256 for {name}")
    passed = actual == expected
    checks[name] = {
        "expected_sha256": expected,
        "actual_sha256": actual,
        "passed": passed,
    }
    if not passed:
        raise ValueError(
            f"reconstructed {name} hash differs: got {actual}, expected {expected}"
        )


def _positive_dimension(value: Any, label: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"checkpoint {label} must be a positive integer")
    return value


def _validate_requested_indices(indices: Sequence[int], count: int) -> list[int]:
    if not indices:
        raise ValueError("at least one render condition index is required")
    output: list[int] = []
    seen: set[int] = set()
    for raw in indices:
        if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
            raise TypeError(f"condition index must be an integer; got {raw!r}")
        index = int(raw)
        if index < 0 or index >= count:
            raise IndexError(f"condition index {index} is outside 0..{count - 1}")
        if index not in seen:
            seen.add(index)
            output.append(index)
    return output


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay saved ICTPolarReal Disney material under recorded HDRI rows. "
            "This CUDA worker intentionally runs only inside Slurm."
        )
    )
    parser.add_argument("--camera-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--object", dest="object_name", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--imaginaire-root", type=Path, required=True)
    lighting = parser.add_mutually_exclusive_group(required=True)
    lighting.add_argument(
        "--condition-indices",
        help="Absolute rows, e.g. 1:142:4 or 1,5,9",
    )
    lighting.add_argument(
        "--lighting-preset",
        choices=SD_RENDERINGS_PRESETS,
    )
    parser.add_argument("--hdri-root", type=Path)
    parser.add_argument("--sd-c04-lights", type=Path)
    parser.add_argument("--validation-condition-index", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--light-root", type=Path)
    parser.add_argument(
        "--light-chunk",
        type=int,
        help=(
            "Optional non-exact memory override. Omit it to reproduce the original "
            "single-chunk summation order."
        ),
    )
    parser.add_argument(
        "--validation-tolerance",
        type=float,
        default=DEFAULT_VALIDATION_TOLERANCE,
    )
    parser.add_argument("--replace-output", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    args = build_parser().parse_args(argv)
    indices = None
    if args.condition_indices is not None:
        lighting = load_recorded_lighting(args.camera_dir)
        indices = parse_condition_indices(
            args.condition_indices, count=len(lighting.conditions)
        )
    c04_lights = args.sd_c04_lights
    if args.lighting_preset in SD_RENDERINGS_PRESETS and c04_lights is None:
        c04_lights = (
            args.imaginaire_root
            / "OLATPipeClean"
            / "LightsLocationsRelativetoCamera"
            / "C04.txt"
        )
    manifest = render_saved_material(
        camera_dir=args.camera_dir,
        data_root=args.data_root,
        object_name=args.object_name,
        camera=args.camera,
        imaginaire_root=args.imaginaire_root,
        output_dir=args.output_dir,
        condition_indices=indices,
        lighting_preset=args.lighting_preset,
        hdri_root=args.hdri_root,
        sd_c04_lights_path=c04_lights,
        validation_condition_index=args.validation_condition_index,
        light_root=args.light_root,
        light_chunk=args.light_chunk,
        validation_tolerance=args.validation_tolerance,
        replace_output=args.replace_output,
    )
    print(
        f"[saved-render] wrote {len(manifest['conditions'])} conditions to "
        f"{args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
