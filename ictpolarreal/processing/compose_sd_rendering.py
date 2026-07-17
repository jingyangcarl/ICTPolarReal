"""Compose the full historical SuperDimension material-rendering sheet.

The CUDA renderer stages the twelve Disney material maps and four panels for
each of the 36 source-exact SD lighting conditions.  This CPU-only compositor
validates that private renderer-v2 transaction and writes the single public
camera-level pickup, ``rendering.png``.  Its 12-column by 13-row layout matches
the identified SuperDimension ``renderings.jpg`` contract exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence


RENDER_MANIFEST_SCHEMA = "ictpolarreal.saved-disney-render.v2"
EXPECTED_CONDITION_COUNT = 36
EXPECTED_SD_SOURCE_RANKS: tuple[int, ...] = tuple(range(1, 33))
CONDITION_LABEL_NUMBERS: tuple[int, ...] = tuple(range(1, 142, 4))
MATERIAL_MAP_ORDER: tuple[str, ...] = (
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
PANEL_ORDER: tuple[str, ...] = ("gt", "pred", "greyball", "chromeball")
SHEET_COLUMNS = 12
SHEET_ROWS = 13
SHEET_TILE_SIZE = 768
EXPECTED_SHEET_TILE_COUNT = SHEET_COLUMNS * SHEET_ROWS
REFERENCE_RENDERINGS_SHA256 = (
    "720a4e975be60f50629f72b106281b759a847c3d9b3b7b45326ce71e874a791d"
)
REFERENCE_HISTORICAL_COMMIT = "877b6f0732cdadd55ed621771d327a574a077137"
LABEL_FONT_SIZE_RATIO = 0.08
LABEL_COLOR_RGB = (255, 255, 0)
LABEL_ORIGIN_X = 4
LABEL_BASELINE_BOTTOM_PX = 4

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{label} must be a hexadecimal SHA-256 digest")
    return value.lower()


def _require_passing_hash_checks(value: Any, label: str) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a non-empty hash-check object")
    has_actual = "actual_sha256" in value
    has_expected = "expected_sha256" in value
    if has_actual or has_expected:
        actual = _require_sha256(value.get("actual_sha256"), f"{label}.actual_sha256")
        expected = _require_sha256(
            value.get("expected_sha256"), f"{label}.expected_sha256"
        )
        if value.get("passed") is not True or actual != expected:
            raise ValueError(f"renderer input hash check failed: {label}")
    for key, item in value.items():
        child_label = f"{label}.{key}"
        if isinstance(item, dict):
            if "passed" in item and item["passed"] is not True:
                raise ValueError(f"renderer input hash check failed: {child_label}")
            _require_passing_hash_checks(item, child_label)
        elif key == "passed" and item is not True:
            raise ValueError(f"renderer input hash check failed: {label}")


def _validate_raw_parallel_hash_check(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("input_hash_checks.raw_parallel_targets is required")
    expected = _require_sha256(
        value.get("expected_sha256"),
        "input_hash_checks.raw_parallel_targets.expected_sha256",
    )
    actual = _require_sha256(
        value.get("actual_sha256"),
        "input_hash_checks.raw_parallel_targets.actual_sha256",
    )
    if value.get("passed") is not True or actual != expected:
        raise ValueError("renderer raw parallel target hash check did not pass")


def _camera_relative_path(camera_dir: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = camera_dir / path
    path = path.resolve()
    try:
        path.relative_to(camera_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must stay under the camera result: {path}") from exc
    return path


def _safe_component(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{label} must be one safe path component")
    return value


def _expected_sheet_contract(
    raw_parallel_targets: Any | None = None,
) -> dict[str, Any]:
    contract = {
        "reference_sha256": REFERENCE_RENDERINGS_SHA256,
        "historical_commit": REFERENCE_HISTORICAL_COMMIT,
        "columns": SHEET_COLUMNS,
        "rows": SHEET_ROWS,
        "tile_size": SHEET_TILE_SIZE,
        "map_order": list(MATERIAL_MAP_ORDER),
        "panel_order": list(PANEL_ORDER),
        "labels": {
            "font": "cv2.FONT_HERSHEY_SIMPLEX",
            "font_size_ratio": LABEL_FONT_SIZE_RATIO,
            "color_rgb": list(LABEL_COLOR_RGB),
            "origin_x": LABEL_ORIGIN_X,
            "baseline_bottom_px": LABEL_BASELINE_BOTTOM_PX,
            "line_type": "cv2.LINE_AA",
            "templates": {
                "gt": "gt #{label_number}",
                "pred": "pred #{label_number}",
                "greyball": "greyball #{label_number}",
                "chromeball": "chromeball #{label_number}",
            },
        },
    }
    if raw_parallel_targets is not None:
        if not isinstance(raw_parallel_targets, dict):
            raise ValueError("raw_parallel_targets mirror must be an object")
        contract["raw_parallel_targets"] = dict(raw_parallel_targets)
    return contract


def _validate_sheet_contract(value: Any, raw_parallel_targets: Any) -> None:
    expected = _expected_sheet_contract(raw_parallel_targets)
    if not isinstance(value, dict):
        raise ValueError("renderer manifest is missing sheet_contract")
    if set(value) != set(expected):
        raise ValueError(
            "sheet_contract keys differ from the historical reference contract"
        )
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(
                f"sheet_contract.{key} differs from the historical reference: "
                f"got {value.get(key)!r}, expected {expected_value!r}"
            )


def _validate_sd_renderings_preset(
    provenance: Any,
    conditions: list[dict[str, Any]],
) -> None:
    if not isinstance(provenance, dict):
        raise ValueError("sd-renderings is missing preset_provenance")
    if provenance.get("schema") != "ictpolarreal.sd-renderings-preset.v1":
        raise ValueError("sd-renderings has an unsupported provenance schema")
    if provenance.get("calibration_count") != 4:
        raise ValueError("sd-renderings provenance must record four calibrations")
    if provenance.get("environment_count") != 32:
        raise ValueError("sd-renderings provenance must record 32 environments")
    candidate_count = provenance.get("candidate_count")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count < 32
    ):
        raise ValueError("sd-renderings provenance has an invalid candidate count")

    if [record.get("preset_index") for record in conditions] != list(
        range(EXPECTED_CONDITION_COUNT)
    ):
        raise ValueError("sd-renderings preset_index values must preserve order 0..35")
    if [record.get("source") for record in conditions[:4]] != [
        "calibration_w",
        "calibration_r",
        "calibration_g",
        "calibration_b",
    ]:
        raise ValueError("sd-renderings calibration order must be white, red, green, blue")
    if any(record.get("absolute_index") is not None for record in conditions):
        raise ValueError("generated sd-renderings conditions cannot claim ICT row indices")
    if any(record.get("rotation_degrees") != 90 for record in conditions):
        raise ValueError("sd-renderings conditions must use the recovered rot090 view")
    if any(
        record.get("source_kind") != "generated_calibration"
        for record in conditions[:4]
    ):
        raise ValueError("sd-renderings must begin with four generated calibrations")
    if any(
        record.get("source_kind") != "environment_map" for record in conditions[4:]
    ):
        raise ValueError("sd-renderings ranks must be environment-map conditions")
    ranks = [record.get("source_rank") for record in conditions]
    if ranks[:4] != [None] * 4 or ranks[4:] != list(EXPECTED_SD_SOURCE_RANKS):
        raise ValueError(
            "sd-renderings must order four calibrations followed by source ranks 1..32"
        )

    orientation = provenance.get("orientation")
    if not isinstance(orientation, dict):
        raise ValueError("sd-renderings provenance is missing orientation")
    if any(record.get("orientation") != orientation for record in conditions):
        raise ValueError("sd-renderings condition orientation differs from provenance")
    projection = provenance.get("projection")
    if not isinstance(projection, dict):
        raise ValueError("sd-renderings provenance is missing projection")
    support_count = projection.get("support_count")
    if support_count != 164 or any(
        record.get("support_count") != support_count for record in conditions
    ):
        raise ValueError("sd-renderings must use the recorded 164-light fit support")

    ranking = provenance.get("ranking")
    if not isinstance(ranking, dict) or ranking.get("camera") != "C04":
        raise ValueError("sd-renderings provenance must record the C04 ranking camera")
    expected_sources = ranking.get("expected_top32_guard")
    if not isinstance(expected_sources, list) or len(expected_sources) != 32:
        raise ValueError("sd-renderings provenance must contain its guarded top-32 list")
    if [record.get("source") for record in conditions[4:]] != expected_sources:
        raise ValueError("sd-renderings condition order differs from its top-32 guard")
    _require_sha256(ranking.get("lights_sha256"), "preset ranking lights_sha256")
    _require_sha256(
        ranking.get("xyz_table_runtime_sha256"),
        "preset ranking xyz_table_runtime_sha256",
    )
    _require_sha256(
        ranking.get("xyz_table_trainer_cpu_audit_sha256"),
        "preset ranking xyz_table_trainer_cpu_audit_sha256",
    )
    if ranking.get("recorded_direction_count") != 155:
        raise ValueError("sd-renderings C04 provenance must contain 155 directions")
    for source_key in ("trainer_source", "sampler_source"):
        source = provenance.get(source_key)
        if not isinstance(source, dict):
            raise ValueError(f"sd-renderings provenance is missing {source_key}")
        _require_sha256(source.get("sha256"), f"preset {source_key}.sha256")


def _validate_replay_provenance(
    payload: dict[str, Any],
    *,
    camera_dir: Path,
    render_dir: Path,
) -> None:
    material = payload.get("material")
    if not isinstance(material, dict):
        raise ValueError("renderer manifest is missing material provenance")
    material_sha256 = _require_sha256(
        material.get("sha256"), "renderer material.sha256"
    )
    material_path = _camera_relative_path(
        camera_dir, material.get("path"), "renderer material.path"
    )
    expected_material_path = (camera_dir / "material" / "olat" / "disney_brdf.pt").resolve()
    if material_path != expected_material_path:
        raise ValueError(
            "renderer material.path must identify material/olat/disney_brdf.pt"
        )
    if not material_path.is_file() or _sha256(material_path) != material_sha256:
        raise ValueError("renderer material provenance does not match the saved state")

    renderer = payload.get("renderer")
    if not isinstance(renderer, dict) or not renderer:
        raise ValueError("renderer manifest is missing renderer provenance")
    if renderer.get("model") != "DisneyBRDFSimplifiedMultiLayer":
        raise ValueError("renderer manifest has the wrong Disney model")
    if renderer.get("device") != "cuda" or not renderer.get("slurm_job_id"):
        raise ValueError("renderer provenance must identify its CUDA Slurm replay")
    if renderer.get("exact_original_summation_order") is not True:
        raise ValueError("renderer replay did not preserve the original summation order")

    input_hash_checks = payload.get("input_hash_checks")
    _require_passing_hash_checks(input_hash_checks, "input_hash_checks")
    assert isinstance(input_hash_checks, dict)
    _validate_raw_parallel_hash_check(input_hash_checks.get("raw_parallel_targets"))

    validation = payload.get("validation")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise ValueError("renderer manifest does not contain a passing validation record")
    if (
        validation.get("byte_identical") is not True
        or validation.get("different_channel_values") != 0
        or validation.get("max_abs") != 0.0
        or validation.get("mean_abs") != 0.0
        or validation.get("tolerance") != 0.0
    ):
        raise ValueError("renderer replay validation is not decoded-pixel exact")
    render_sha256 = _require_sha256(
        validation.get("render_sha256"), "renderer validation.render_sha256"
    )
    canonical_sha256 = _require_sha256(
        validation.get("canonical_sha256"), "renderer validation.canonical_sha256"
    )
    if render_sha256 != canonical_sha256:
        raise ValueError("renderer validation hashes do not match")

    canonical_path = _camera_relative_path(
        camera_dir,
        validation.get("canonical_path"),
        "renderer validation.canonical_path",
    )
    if not canonical_path.is_file() or _sha256(canonical_path) != canonical_sha256:
        raise ValueError("renderer canonical validation image does not match provenance")
    validation_path_value = validation.get("render_path")
    if not isinstance(validation_path_value, str):
        raise ValueError("renderer validation is missing render_path")
    validation_name = Path(validation_path_value).name
    if Path(validation_path_value).parent.name != "validation":
        raise ValueError("renderer validation render_path has the wrong directory")
    validation_path = render_dir / "validation" / validation_name
    recorded_validation_path = _camera_relative_path(
        camera_dir,
        validation_path_value,
        "renderer validation.render_path",
    )
    if recorded_validation_path != validation_path.resolve():
        raise ValueError("renderer validation render_path differs from its staged path")
    if not validation_path.is_file() or _sha256(validation_path) != render_sha256:
        raise ValueError("staged replay validation image does not match provenance")


def _require_staged_png(
    *,
    camera_dir: Path,
    render_dir: Path,
    recorded_path: Any,
    expected_path: Path,
    expected_sha256: Any,
    label: str,
) -> Path:
    path = _camera_relative_path(camera_dir, recorded_path, f"{label}.render_path")
    if path != expected_path.resolve():
        raise ValueError(
            f"{label}.render_path differs from its private staging contract: {path}"
        )
    digest = _require_sha256(expected_sha256, f"{label}.render_sha256")
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}; the sheet is all-or-nothing: {path}")
    if _sha256(path) != digest:
        raise ValueError(f"renderer hash does not match staged PNG for {label}")
    try:
        path.relative_to(render_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must stay under the private renderer stage") from exc
    return path


def _read_material_maps(
    payload: dict[str, Any], *, camera_dir: Path, render_dir: Path
) -> list[tuple[str, Path]]:
    records = payload.get("material_maps")
    if not isinstance(records, list) or len(records) != len(MATERIAL_MAP_ORDER):
        raise ValueError("renderer manifest must contain exactly 12 ordered material_maps")
    if [record.get("name") if isinstance(record, dict) else None for record in records] != list(
        MATERIAL_MAP_ORDER
    ):
        raise ValueError("renderer material_maps do not preserve the historical map order")
    selected: list[tuple[str, Path]] = []
    for record, name in zip(records, MATERIAL_MAP_ORDER):
        assert isinstance(record, dict)
        if set(record) != {"name", "render_path", "render_sha256"}:
            raise ValueError(f"material map {name} has unexpected manifest fields")
        expected_path = render_dir / "material_maps" / f"{name}.png"
        path = _require_staged_png(
            camera_dir=camera_dir,
            render_dir=render_dir,
            recorded_path=record.get("render_path"),
            expected_path=expected_path,
            expected_sha256=record.get("render_sha256"),
            label=f"material map {name}",
        )
        selected.append((name, path))
    return selected


def _read_condition_panels(
    conditions: list[dict[str, Any]], *, camera_dir: Path, render_dir: Path
) -> list[tuple[str, Path]]:
    selected: list[tuple[str, Path]] = []
    condition_ids: set[str] = set()
    safe_ids: set[str] = set()
    for position, record in enumerate(conditions):
        condition_id = _safe_component(
            record.get("condition_id"), f"condition {position + 1}.condition_id"
        )
        safe_id = _safe_component(
            record.get("safe_condition_id"),
            f"condition {position + 1}.safe_condition_id",
        )
        if condition_id in condition_ids or safe_id in safe_ids:
            raise ValueError("renderer condition identifiers must be unique")
        condition_ids.add(condition_id)
        safe_ids.add(safe_id)

        label_number = record.get("label_number")
        expected_label_number = CONDITION_LABEL_NUMBERS[position]
        if label_number != expected_label_number:
            raise ValueError(
                f"condition {condition_id} label_number must be {expected_label_number}"
            )
        if not isinstance(record.get("source"), str) or not record["source"]:
            raise ValueError(f"renderer condition {condition_id} has no lighting source")
        _require_sha256(
            record.get("source_sha256"), f"condition {condition_id} source_sha256"
        )
        _require_sha256(
            record.get("weight_sha256"), f"condition {condition_id} weight_sha256"
        )

        panels = record.get("panels")
        if not isinstance(panels, dict) or list(panels) != list(PANEL_ORDER):
            raise ValueError(
                f"condition {condition_id} panels must preserve order "
                f"{list(PANEL_ORDER)}"
            )
        for panel_name in PANEL_ORDER:
            panel = panels[panel_name]
            if not isinstance(panel, dict):
                raise ValueError(
                    f"condition {condition_id} panel {panel_name} must be an object"
                )
            if set(panel) != {"render_path", "render_sha256"}:
                raise ValueError(
                    f"condition {condition_id} panel {panel_name} has unexpected fields"
                )
            expected_path = (
                render_dir / "conditions" / safe_id / f"{panel_name}.png"
            )
            path = _require_staged_png(
                camera_dir=camera_dir,
                render_dir=render_dir,
                recorded_path=panel.get("render_path"),
                expected_path=expected_path,
                expected_sha256=panel.get("render_sha256"),
                label=f"condition {condition_id} panel {panel_name}",
            )
            selected.append((f"{panel_name} #{label_number}", path))
    return selected


def _read_renderer_manifest(
    path: Path,
    *,
    render_dir: Path,
    camera_dir: Path,
) -> tuple[list[tuple[str, Path]], str]:
    manifest_sha256 = _sha256(path) if path.is_file() else None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"renderer manifest does not exist: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"renderer manifest must contain a JSON object: {path}")
    if payload.get("schema") != RENDER_MANIFEST_SCHEMA:
        raise ValueError(
            f"renderer manifest schema must be {RENDER_MANIFEST_SCHEMA!r}: {path}"
        )
    if payload.get("profile") != "olat":
        raise ValueError(f"renderer manifest must use the acquired OLAT profile: {path}")
    if payload.get("lighting_preset") != "sd-renderings":
        raise ValueError("renderer manifest must use the source-exact SD lighting preset")
    if payload.get("selected_absolute_indices") is not None:
        raise ValueError("the SD sheet cannot use recorded absolute condition indices")
    input_hash_checks = payload.get("input_hash_checks")
    raw_parallel_targets = (
        input_hash_checks.get("raw_parallel_targets")
        if isinstance(input_hash_checks, dict)
        else None
    )
    if raw_parallel_targets is None:
        raise ValueError("input_hash_checks.raw_parallel_targets is required")
    _validate_sheet_contract(
        payload.get("sheet_contract"), raw_parallel_targets
    )

    conditions = payload.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != EXPECTED_CONDITION_COUNT:
        raise ValueError(
            f"renderer manifest must describe exactly {EXPECTED_CONDITION_COUNT} "
            "ordered conditions"
        )
    if not all(isinstance(record, dict) for record in conditions):
        raise ValueError("every renderer condition must be an object")
    typed_conditions = conditions
    _validate_sd_renderings_preset(payload.get("preset_provenance"), typed_conditions)
    _validate_replay_provenance(
        payload,
        camera_dir=camera_dir,
        render_dir=render_dir,
    )

    selected = _read_material_maps(
        payload, camera_dir=camera_dir, render_dir=render_dir
    )
    selected.extend(
        _read_condition_panels(
            typed_conditions, camera_dir=camera_dir, render_dir=render_dir
        )
    )
    if len(selected) != EXPECTED_SHEET_TILE_COUNT:
        raise RuntimeError(
            f"full SD sheet needs {EXPECTED_SHEET_TILE_COUNT} cells, got {len(selected)}"
        )
    assert manifest_sha256 is not None
    return selected, manifest_sha256


def _read_native_rgb_png(path: Path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        if image.format != "PNG":
            raise ValueError(f"sheet source must be a PNG: {path}")
        if image.mode != "RGB":
            raise ValueError(f"sheet source must already be RGB, got {image.mode}: {path}")
        array = np.asarray(image, dtype=np.uint8).copy()
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"sheet source has invalid RGB shape {array.shape}: {path}")
    return array


def _fit_square_rgb(tile, *, tile_size: int):
    """Contain-fit one RGB tile in a centered black square without cropping."""

    import numpy as np
    from PIL import Image

    if tile.dtype != np.uint8 or tile.ndim != 3 or tile.shape[2] != 3:
        raise ValueError("square-fit input must be an HxWx3 uint8 RGB image")
    if tile_size <= 0:
        raise ValueError("sheet tile_size must be positive")
    height, width = tile.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("sheet source cannot have an empty dimension")
    scale = min(tile_size / width, tile_size / height)
    fitted_width = max(1, min(tile_size, int(round(width * scale))))
    fitted_height = max(1, min(tile_size, int(round(height * scale))))
    if (fitted_width, fitted_height) == (width, height):
        resized = tile
    else:
        resized = np.asarray(
            Image.fromarray(tile).resize(
                (fitted_width, fitted_height), Image.Resampling.LANCZOS
            ),
            dtype=np.uint8,
        )
    canvas = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    offset_x = (tile_size - fitted_width) // 2
    offset_y = (tile_size - fitted_height) // 2
    canvas[
        offset_y : offset_y + fitted_height,
        offset_x : offset_x + fitted_width,
    ] = resized
    return canvas


def _overlay_historical_label(tile, text: str):
    import cv2
    import numpy as np

    if tile.dtype != np.uint8 or tile.ndim != 3 or tile.shape[2] != 3:
        raise ValueError("label input must be an HxWx3 uint8 RGB image")
    height, width = tile.shape[:2]
    if height != width:
        raise ValueError("historical labels require a square fitted cell")
    font_size = int(height * LABEL_FONT_SIZE_RATIO)
    font_scale = font_size / 30.0
    thickness = max(1, font_size // 15)
    (text_width, _), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    if text_width > width - 8:
        raise ValueError(f"historical label {text!r} does not fit tile width {width}")
    labeled = np.ascontiguousarray(tile.copy())
    cv2.putText(
        labeled,
        text,
        (LABEL_ORIGIN_X, height - LABEL_BASELINE_BOTTOM_PX),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        LABEL_COLOR_RGB,
        thickness,
        cv2.LINE_AA,
    )
    return labeled


def _compose_full_sheet(
    selected: list[tuple[str, Path]], *, tile_size: int | None = None
):
    import numpy as np

    tile_size = SHEET_TILE_SIZE if tile_size is None else tile_size
    if len(selected) != EXPECTED_SHEET_TILE_COUNT:
        raise ValueError(
            f"full sheet requires exactly {EXPECTED_SHEET_TILE_COUNT} ordered cells"
        )
    sheet = np.empty(
        (SHEET_ROWS * tile_size, SHEET_COLUMNS * tile_size, 3), dtype=np.uint8
    )
    cell_sha256: list[str] = []
    for position, (label, source_path) in enumerate(selected):
        native = _read_native_rgb_png(source_path)
        fitted = _fit_square_rgb(native, tile_size=tile_size)
        cell = _overlay_historical_label(fitted, label)
        row, column = divmod(position, SHEET_COLUMNS)
        sheet[
            row * tile_size : (row + 1) * tile_size,
            column * tile_size : (column + 1) * tile_size,
        ] = cell
        cell_sha256.append(_array_sha256(cell))
    return sheet, cell_sha256


def _validate_rendering(
    path: Path,
    *,
    expected_cell_sha256: Sequence[str],
    tile_size: int,
    expected_sha256: str | None = None,
) -> str:
    import numpy as np
    from PIL import Image

    if len(expected_cell_sha256) != EXPECTED_SHEET_TILE_COUNT:
        raise ValueError("rendering validation requires one hash per historical cell")
    expected_size = (SHEET_COLUMNS * tile_size, SHEET_ROWS * tile_size)
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or image.size != expected_size:
            raise RuntimeError(
                "invalid rendering.png contract: "
                f"format={image.format}, mode={image.mode}, size={image.size}, "
                f"expected PNG/RGB/{expected_size}"
            )
        rendering = np.asarray(image, dtype=np.uint8)
    for position, expected_digest in enumerate(expected_cell_sha256):
        row, column = divmod(position, SHEET_COLUMNS)
        actual = rendering[
            row * tile_size : (row + 1) * tile_size,
            column * tile_size : (column + 1) * tile_size,
        ]
        if _array_sha256(actual) != expected_digest:
            raise RuntimeError(
                f"rendering.png cell order/content diverges at position {position + 1}"
            )
    digest = _sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError("atomic rendering.png hash changed during replacement")
    return digest


def _remove_legacy_rendering_artifacts(camera_dir: Path) -> None:
    material_dir = camera_dir / "material"
    olat_dir = material_dir / "olat"
    for stale_artifact in (
        camera_dir / "rendering.json",
        camera_dir / ".rendering.json.pending",
        camera_dir / ".rendering.json.tmp",
        material_dir / "rendering.png",
        material_dir / "rendering.json",
        material_dir / ".rendering.png.pending",
        material_dir / ".rendering.json.pending",
        material_dir / ".rendering.json.tmp",
        olat_dir / "rendering.png",
        olat_dir / "rendering.json",
        olat_dir / ".rendering.png.pending",
        olat_dir / ".rendering.json.pending",
        olat_dir / ".rendering.json.tmp",
    ):
        stale_artifact.unlink(missing_ok=True)


def compose_sd_rendering(
    camera_dir: str | Path,
    *,
    render_dir: str | Path | None = None,
    keep_stage: bool = False,
) -> Path:
    """Write the full source-exact camera-level ``rendering.png`` sheet."""

    from PIL import Image

    camera_dir = Path(camera_dir).expanduser().resolve()
    default_render_dir = camera_dir / ".rendering_stage"
    render_dir = (
        Path(render_dir).expanduser().resolve()
        if render_dir is not None
        else default_render_dir
    )
    try:
        render_dir.relative_to(camera_dir)
    except ValueError as exc:
        raise ValueError("private renderer stage must stay under the camera result") from exc

    renderer_manifest_path = render_dir / "render_manifest.json"
    selected, manifest_sha256 = _read_renderer_manifest(
        renderer_manifest_path,
        render_dir=render_dir,
        camera_dir=camera_dir,
    )
    sheet, cell_sha256 = _compose_full_sheet(selected)

    camera_dir.mkdir(parents=True, exist_ok=True)
    rendering_path = camera_dir / "rendering.png"
    pending_path = camera_dir / ".rendering.png.pending"
    backup_path = camera_dir / ".rendering.png.previous"
    installed = False
    had_previous = rendering_path.is_file()
    succeeded = False
    try:
        Image.fromarray(sheet).save(pending_path, format="PNG", compress_level=6)
        del sheet
        pending_sha256 = _validate_rendering(
            pending_path,
            expected_cell_sha256=cell_sha256,
            tile_size=SHEET_TILE_SIZE,
        )
        if _sha256(renderer_manifest_path) != manifest_sha256:
            raise RuntimeError("renderer manifest changed during sheet composition")

        backup_path.unlink(missing_ok=True)
        if had_previous:
            os.link(rendering_path, backup_path)
        pending_path.replace(rendering_path)
        installed = True
        _validate_rendering(
            rendering_path,
            expected_cell_sha256=cell_sha256,
            tile_size=SHEET_TILE_SIZE,
            expected_sha256=pending_sha256,
        )
        succeeded = True
    except Exception:
        if installed:
            if backup_path.is_file():
                backup_path.replace(rendering_path)
            elif not had_previous:
                rendering_path.unlink(missing_ok=True)
        raise
    finally:
        pending_path.unlink(missing_ok=True)
        if succeeded:
            backup_path.unlink(missing_ok=True)

    _remove_legacy_rendering_artifacts(camera_dir)
    if not keep_stage and render_dir == default_render_dir:
        shutil.rmtree(render_dir)
    return rendering_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compose the full historical SD material-rendering sheet: twelve "
            "material maps followed by 36 [gt, pred, greyball, chromeball] "
            "conditions in a gapless 12x13 camera-level rendering.png."
        )
    )
    parser.add_argument(
        "--camera-dir",
        type=Path,
        required=True,
        help="camera result directory containing the fixed-post-fit OLAT material",
    )
    parser.add_argument(
        "--render-dir",
        type=Path,
        help=(
            "private renderer staging directory (default: "
            "<camera-dir>/.rendering_stage)"
        ),
    )
    parser.add_argument(
        "--keep-stage",
        action="store_true",
        help="retain the private renderer staging directory after composition",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rendering_path = compose_sd_rendering(
        args.camera_dir,
        render_dir=args.render_dir,
        keep_stage=args.keep_stage,
    )
    print(rendering_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
