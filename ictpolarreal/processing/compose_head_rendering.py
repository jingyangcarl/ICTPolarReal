"""Compose the native-resolution SD-OLAT predicted-head grid.

The saved-material renderer writes 36 condition PNGs and a replay manifest to
the private ``material/olat/.rendering_stage`` directory.  This CPU-only step
validates that source contract, overlays the canonical small ``pred #N`` label,
and writes one public artifact: ``material/olat/rendering.png``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Sequence


EXPECTED_HEAD_COUNT = 36
EXPECTED_SD_SOURCE_RANKS: tuple[int, ...] = tuple(range(1, 33))
PREDICTION_LABEL_NUMBERS: tuple[int, ...] = tuple(range(1, 142, 4))
SHEET_COLUMNS = 12
SHEET_ROWS = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    for key, item in value.items():
        child_label = f"{label}.{key}"
        if isinstance(item, dict):
            if "passed" in item and item["passed"] is not True:
                raise ValueError(f"renderer input hash check failed: {child_label}")
            _require_passing_hash_checks(item, child_label)
        elif key == "passed" and item is not True:
            raise ValueError(f"renderer input hash check failed: {label}")


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


def _validate_sd_olat_preset(
    provenance: Any,
    conditions: list[dict[str, Any]],
) -> None:
    if not isinstance(provenance, dict):
        raise ValueError("sd-olat-heads is missing preset_provenance")
    if provenance.get("schema") != "ictpolarreal.sd-olat-heads-preset.v1":
        raise ValueError("sd-olat-heads has an unsupported provenance schema")
    if provenance.get("calibration_count") != 4:
        raise ValueError("sd-olat-heads provenance must record four calibrations")
    if provenance.get("environment_count") != 32:
        raise ValueError("sd-olat-heads provenance must record 32 environments")
    candidate_count = provenance.get("candidate_count")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count < 32
    ):
        raise ValueError("sd-olat-heads provenance has an invalid candidate count")

    if [record.get("preset_index") for record in conditions] != list(
        range(EXPECTED_HEAD_COUNT)
    ):
        raise ValueError("sd-olat-heads preset_index values must preserve order 0..35")
    if [record.get("source") for record in conditions[:4]] != [
        "calibration_w",
        "calibration_r",
        "calibration_g",
        "calibration_b",
    ]:
        raise ValueError("sd-olat-heads calibration order must be white, red, green, blue")
    if any(record.get("absolute_index") is not None for record in conditions):
        raise ValueError("generated sd-olat-heads conditions cannot claim ICT row indices")
    if any(record.get("rotation_degrees") != 90 for record in conditions):
        raise ValueError("sd-olat-heads conditions must use the recovered rot090 view")
    if any(
        record.get("source_kind") != "generated_calibration"
        for record in conditions[:4]
    ):
        raise ValueError("sd-olat-heads must begin with four generated calibrations")
    if any(
        record.get("source_kind") != "environment_map" for record in conditions[4:]
    ):
        raise ValueError("sd-olat-heads ranks must be environment-map conditions")
    ranks = [record.get("source_rank") for record in conditions]
    if ranks[:4] != [None] * 4 or ranks[4:] != list(EXPECTED_SD_SOURCE_RANKS):
        raise ValueError(
            "sd-olat-heads must order four calibrations followed by source ranks 1..32"
        )

    orientation = provenance.get("orientation")
    if not isinstance(orientation, dict):
        raise ValueError("sd-olat-heads provenance is missing orientation")
    if any(record.get("orientation") != orientation for record in conditions):
        raise ValueError("sd-olat-heads condition orientation differs from provenance")
    projection = provenance.get("projection")
    if not isinstance(projection, dict):
        raise ValueError("sd-olat-heads provenance is missing projection")
    support_count = projection.get("support_count")
    if support_count != 164 or any(
        record.get("support_count") != support_count for record in conditions
    ):
        raise ValueError("sd-olat-heads must use the recorded 164-light fit support")

    ranking = provenance.get("ranking")
    if not isinstance(ranking, dict) or ranking.get("camera") != "C04":
        raise ValueError("sd-olat-heads provenance must record the C04 ranking camera")
    expected_sources = ranking.get("expected_top32_guard")
    if not isinstance(expected_sources, list) or len(expected_sources) != 32:
        raise ValueError("sd-olat-heads provenance must contain its guarded top-32 list")
    if [record.get("source") for record in conditions[4:]] != expected_sources:
        raise ValueError("sd-olat-heads condition order differs from its top-32 guard")
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
        raise ValueError("sd-olat-heads C04 provenance must contain 155 directions")
    for source_key in ("trainer_source", "sampler_source"):
        source = provenance.get(source_key)
        if not isinstance(source, dict):
            raise ValueError(f"sd-olat-heads provenance is missing {source_key}")
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

    _require_passing_hash_checks(
        payload.get("input_hash_checks"), "input_hash_checks"
    )
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
    if not validation_path.is_file() or _sha256(validation_path) != render_sha256:
        raise ValueError("staged replay validation image does not match provenance")


def _read_renderer_manifest(
    path: Path,
    *,
    render_dir: Path,
    camera_dir: Path,
) -> tuple[list[tuple[dict[str, Any], Path]], str]:
    manifest_sha256 = _sha256(path) if path.is_file() else None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"renderer manifest does not exist: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"renderer manifest must contain a JSON object: {path}")
    if payload.get("schema") != "ictpolarreal.saved-disney-render.v1":
        raise ValueError(
            "renderer manifest schema must be "
            f"'ictpolarreal.saved-disney-render.v1': {path}"
        )
    if payload.get("profile") != "olat":
        raise ValueError(f"renderer manifest must use the acquired OLAT profile: {path}")
    if payload.get("lighting_preset") != "sd-olat-heads":
        raise ValueError("renderer manifest must use the sd-olat-heads preset")
    if payload.get("selected_absolute_indices") is not None:
        raise ValueError("sd-olat-heads cannot use recorded absolute condition indices")

    conditions = payload.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != EXPECTED_HEAD_COUNT:
        raise ValueError(
            f"renderer manifest must describe exactly {EXPECTED_HEAD_COUNT} "
            "ordered conditions"
        )
    _validate_sd_olat_preset(payload.get("preset_provenance"), conditions)
    _validate_replay_provenance(
        payload,
        camera_dir=camera_dir,
        render_dir=render_dir,
    )

    selected: list[tuple[dict[str, Any], Path]] = []
    identifiers: set[str] = set()
    missing: list[str] = []
    for position, record in enumerate(conditions):
        if not isinstance(record, dict):
            raise ValueError(
                f"renderer condition at ordered position {position} is not an object"
            )
        condition_id = record.get("condition_id")
        if not isinstance(condition_id, str) or not condition_id:
            raise ValueError(
                f"renderer condition at ordered position {position} has no condition_id"
            )
        if Path(condition_id).name != condition_id:
            raise ValueError(
                f"condition_id must be a safe path component: {condition_id!r}"
            )
        if condition_id in identifiers:
            raise ValueError(f"renderer condition_id is duplicated: {condition_id}")
        identifiers.add(condition_id)
        if not isinstance(record.get("source"), str) or not record["source"]:
            raise ValueError(f"renderer condition {condition_id} has no lighting source")
        _require_sha256(
            record.get("source_sha256"), f"condition {condition_id} source_sha256"
        )
        _require_sha256(
            record.get("weight_sha256"), f"condition {condition_id} weight_sha256"
        )
        expected_render_sha256 = _require_sha256(
            record.get("render_sha256"), f"condition {condition_id} render_sha256"
        )
        recorded_render_path = record.get("render_path")
        if not isinstance(recorded_render_path, str) or (
            Path(recorded_render_path).name != f"{condition_id}.png"
            or Path(recorded_render_path).parent.name != "conditions"
        ):
            raise ValueError(
                f"renderer condition {condition_id} has an inconsistent render_path"
            )
        render_path = render_dir / "conditions" / f"{condition_id}.png"
        if not render_path.is_file():
            missing.append(f"position {position + 1}: {render_path}")
        elif _sha256(render_path) != expected_render_sha256:
            raise ValueError(
                f"renderer hash does not match condition PNG for {condition_id}"
            )
        selected.append((record, render_path))
    if missing:
        detail = "\n  ".join(missing)
        raise FileNotFoundError(
            "missing predicted condition render(s); the grid is all-or-nothing:\n  "
            + detail
        )
    assert manifest_sha256 is not None
    return selected, manifest_sha256


def _read_native_rgb_png(path: Path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        if image.format != "PNG":
            raise ValueError(f"condition render must be a PNG: {path}")
        if image.mode != "RGB":
            raise ValueError(
                f"condition render must already be RGB, got {image.mode}: {path}"
            )
        array = np.asarray(image, dtype=np.uint8).copy()
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"condition render has invalid RGB shape {array.shape}: {path}")
    return array


def _overlay_prediction_label(tile, number: int):
    import cv2
    import numpy as np

    if tile.dtype != np.uint8 or tile.ndim != 3 or tile.shape[2] != 3:
        raise ValueError("prediction label input must be an HxWx3 uint8 RGB image")
    height, width = tile.shape[:2]
    font_size = max(10, int(height * 0.07))
    font_scale = font_size / 30.0
    thickness = max(1, font_size // 15)
    text = f"pred #{number}"
    (text_width, _), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    if text_width > width - 8:
        raise ValueError(
            f"canonical prediction label {text!r} does not fit native tile width {width}"
        )
    labeled = np.ascontiguousarray(tile.copy())
    cv2.putText(
        labeled,
        text,
        (4, height - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 0),
        thickness,
        cv2.LINE_AA,
    )
    return labeled


def _compose_native_grid(selected: list[tuple[dict[str, Any], Path]]):
    import numpy as np

    if len(selected) != EXPECTED_HEAD_COUNT:
        raise ValueError(f"native grid requires exactly {EXPECTED_HEAD_COUNT} tiles")
    if len(PREDICTION_LABEL_NUMBERS) != EXPECTED_HEAD_COUNT:
        raise RuntimeError("canonical prediction label sequence has the wrong length")

    expected_shape: tuple[int, int, int] | None = None
    labeled_tiles = []
    for position, ((_, render_path), label_number) in enumerate(
        zip(selected, PREDICTION_LABEL_NUMBERS)
    ):
        tile = _read_native_rgb_png(render_path)
        if expected_shape is None:
            expected_shape = tile.shape
        elif tile.shape != expected_shape:
            raise ValueError(
                "all condition renders must have identical native RGB dimensions; "
                f"position {position + 1} has {tile.shape}, expected {expected_shape}"
            )
        labeled_tiles.append(_overlay_prediction_label(tile, label_number))

    assert expected_shape is not None
    height, width, _ = expected_shape
    sheet = np.empty(
        (SHEET_ROWS * height, SHEET_COLUMNS * width, 3), dtype=np.uint8
    )
    for position, tile in enumerate(labeled_tiles):
        row, column = divmod(position, SHEET_COLUMNS)
        sheet[
            row * height : (row + 1) * height,
            column * width : (column + 1) * width,
        ] = tile
    return sheet, labeled_tiles, (width, height)


def _validate_rendering(
    path: Path,
    *,
    expected_tiles: list[Any],
    tile_size: tuple[int, int],
    expected_sha256: str | None = None,
) -> str:
    import numpy as np
    from PIL import Image

    tile_width, tile_height = tile_size
    expected_size = (SHEET_COLUMNS * tile_width, SHEET_ROWS * tile_height)
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or image.size != expected_size:
            raise RuntimeError(
                "invalid rendering.png contract: "
                f"format={image.format}, mode={image.mode}, size={image.size}, "
                f"expected PNG/RGB/{expected_size}"
            )
        rendering = np.asarray(image, dtype=np.uint8)
    for position, expected in enumerate(expected_tiles):
        row, column = divmod(position, SHEET_COLUMNS)
        actual = rendering[
            row * tile_height : (row + 1) * tile_height,
            column * tile_width : (column + 1) * tile_width,
        ]
        if not np.array_equal(actual, expected):
            raise RuntimeError(
                f"rendering.png tile order/content diverges at position {position + 1}"
            )
    digest = _sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError("atomic rendering.png hash changed during replacement")
    return digest


def compose_head_rendering(
    camera_dir: str | Path,
    *,
    render_dir: str | Path | None = None,
    keep_heads: bool = False,
) -> Path:
    """Write the singular native-resolution ``material/olat/rendering.png``."""

    from PIL import Image

    camera_dir = Path(camera_dir)
    material_dir = camera_dir / "material" / "olat"
    default_render_dir = material_dir / ".rendering_stage"
    render_dir = Path(render_dir) if render_dir is not None else default_render_dir
    renderer_manifest_path = render_dir / "render_manifest.json"
    selected, manifest_sha256 = _read_renderer_manifest(
        renderer_manifest_path,
        render_dir=render_dir,
        camera_dir=camera_dir,
    )
    sheet, labeled_tiles, tile_size = _compose_native_grid(selected)

    material_dir.mkdir(parents=True, exist_ok=True)
    rendering_path = material_dir / "rendering.png"
    pending_path = material_dir / ".rendering.png.pending"
    try:
        Image.fromarray(sheet, mode="RGB").save(
            pending_path, format="PNG", compress_level=6
        )
        pending_sha256 = _validate_rendering(
            pending_path,
            expected_tiles=labeled_tiles,
            tile_size=tile_size,
        )
        if _sha256(renderer_manifest_path) != manifest_sha256:
            raise RuntimeError("renderer manifest changed during grid composition")
        pending_path.replace(rendering_path)
        _validate_rendering(
            rendering_path,
            expected_tiles=labeled_tiles,
            tile_size=tile_size,
            expected_sha256=pending_sha256,
        )
    finally:
        pending_path.unlink(missing_ok=True)

    # The compact pickup deliberately has no sidecar report.  Remove the old
    # compositor's sidecar only after the replacement PNG has validated.
    for stale_sidecar in (
        material_dir / "rendering.json",
        material_dir / ".rendering.json.pending",
        material_dir / ".rendering.json.tmp",
    ):
        stale_sidecar.unlink(missing_ok=True)

    if not keep_heads and render_dir.resolve() == default_render_dir.resolve():
        shutil.rmtree(render_dir)
    return rendering_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compose the source-exact 36-condition SD-OLAT heads as one gapless "
            "12x3 native-resolution rendering.png."
        )
    )
    parser.add_argument(
        "--camera-dir",
        type=Path,
        required=True,
        help="camera result directory containing material/olat",
    )
    parser.add_argument(
        "--render-dir",
        type=Path,
        help=(
            "private renderer staging directory (default: "
            "<camera-dir>/material/olat/.rendering_stage)"
        ),
    )
    parser.add_argument(
        "--keep-heads",
        action="store_true",
        help="retain the private renderer staging directory after composition",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rendering_path = compose_head_rendering(
        args.camera_dir,
        render_dir=args.render_dir,
        keep_heads=args.keep_heads,
    )
    print(rendering_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
