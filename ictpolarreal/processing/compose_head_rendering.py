"""Compose the SD-OLAT-style, predicted-render-only contact sheet.

The renderer writes one PNG per lighting condition into a private staging
directory under ``material/olat``.  Its ordered manifest is the sole authority
for the 36 displayed lights, including source-exact SuperDimension presets
whose HDRIs do not appear in the ICT fit-condition manifest.  This module keeps
the presentation step CPU-only and writes one deterministic public
``material/olat/rendering.png`` plus ``material/olat/rendering.json``.  Staged
condition tiles are removed after successful composition by default.

Example::

    python -m ictpolarreal.processing.compose_head_rendering \
        --camera-dir outputs/.../dragondruit/cam07
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence


EXPECTED_HEAD_COUNT = 36
EXPECTED_SD_SOURCE_RANKS: tuple[int, ...] = tuple(range(1, 33))

SHEET_COLUMNS = 6
TILE_WIDTH = 400
IMAGE_HEIGHT = 726
LABEL_HEIGHT = 104
GAP = 12
PADDING = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_font(size: int, *, bold: bool):
    from PIL import ImageFont

    candidates = (
        (
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "LiberationSans-Bold.ttf",
            "DejaVuSans-Bold.ttf",
        )
        if bold
        else (
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "LiberationSans-Regular.ttf",
            "DejaVuSans.ttf",
        )
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size), Path(candidate).name
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size), "Pillow-default-scalable"
    except TypeError as exc:
        raise RuntimeError(
            "a scalable TrueType font is required for readable rendering labels; "
            "install Liberation Sans or DejaVu Sans"
        ) from exc


def _display_source(condition: dict[str, Any]) -> str:
    source = str(condition.get("source") or condition["condition_id"])
    name = Path(source).name
    lowered = name.lower()
    if lowered in {"calibration_w", "calibration_white"}:
        return "Calibration white"
    if lowered in {"calibration_r", "calibration_red"}:
        return "Calibration red"
    if lowered in {"calibration_g", "calibration_green"}:
        return "Calibration green"
    if lowered in {"calibration_b", "calibration_blue"}:
        return "Calibration blue"

    for suffix in (".exr", ".hdr", ".png", ".jpg", ".jpeg"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    hdrmaps_source = False
    for prefix in (
        "hdriheaven_original_",
        "hdriheaven_flipped_",
        "hdrihaven_original_",
        "hdrihaven_flipped_",
        "hdrmaps_original_",
        "hdrmaps_flipped_",
    ):
        if name.lower().startswith(prefix):
            name = name[len(prefix) :]
            hdrmaps_source = prefix.startswith("hdrmaps_")
            break
    for suffix in ("_hdrmaps_com_free_1k", "_2k_1k", "_4k_2k", "_1k"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    display = " ".join(name.replace("-", " ").replace("_", " ").split()).title()
    return f"HDRMaps {display}" if hdrmaps_source else display


def _display_detail(condition: dict[str, Any]) -> str:
    source = str(condition.get("source") or "").lower()
    rotation = int(condition.get("rotation_degrees", 0))
    source_rank = condition.get("source_rank")
    absolute_index = condition.get("absolute_index")
    if source.startswith(("hdriheaven_flipped_", "hdrihaven_flipped_", "hdrmaps_flipped_")):
        variant = "flipped"
    elif source.startswith(("hdriheaven_original_", "hdrihaven_original_", "hdrmaps_original_")):
        variant = "original"
    else:
        variant = None

    if source_rank is not None:
        parts = [f"SD rank {source_rank:02d}"]
        if variant is not None:
            parts.append(variant)
        parts.append(f"rot{rotation:02d}")
        return "  ·  ".join(parts)
    if absolute_index is not None:
        return f"recorded row {absolute_index:03d}  ·  rot{rotation:02d}"
    return f"SD calibration  ·  rot{rotation:02d}"


def _ellipsize(draw, text: str, font, max_width: int) -> str:
    if draw.textlength(text, font=font) <= max_width:
        return text
    ellipsis = "…"
    if draw.textlength(ellipsis, font=font) > max_width:
        return ""
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = text[:middle].rstrip() + ellipsis
        if draw.textlength(candidate, font=font) <= max_width:
            low = middle
        else:
            high = middle - 1
    return text[:low].rstrip() + ellipsis


def _fit_render(path: Path):
    from PIL import Image, ImageOps

    with Image.open(path) as source_file:
        source = ImageOps.exif_transpose(source_file).convert("RGB")
        scale = min(TILE_WIDTH / source.width, IMAGE_HEIGHT / source.height)
        size = (
            max(1, round(source.width * scale)),
            max(1, round(source.height * scale)),
        )
        resized = source.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (TILE_WIDTH, IMAGE_HEIGHT), (8, 8, 10))
    canvas.paste(
        resized,
        ((TILE_WIDTH - resized.width) // 2, (IMAGE_HEIGHT - resized.height) // 2),
    )
    return canvas


def _relative_path(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path, start=root)).as_posix()


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

    ranking = provenance.get("ranking")
    if not isinstance(ranking, dict) or ranking.get("camera") != "C04":
        raise ValueError("sd-olat-heads provenance must record the C04 ranking camera")
    expected_sources = ranking.get("expected_top32_guard")
    if not isinstance(expected_sources, list) or len(expected_sources) != 32:
        raise ValueError("sd-olat-heads provenance must contain its guarded top-32 list")
    if [record.get("source") for record in conditions[4:]] != expected_sources:
        raise ValueError("sd-olat-heads condition order differs from its top-32 guard")
    _require_sha256(ranking.get("lights_sha256"), "preset ranking lights_sha256")
    for source_key in ("trainer_source", "sampler_source"):
        source = provenance.get(source_key)
        if not isinstance(source, dict):
            raise ValueError(f"sd-olat-heads provenance is missing {source_key}")
        _require_sha256(source.get("sha256"), f"preset {source_key}.sha256")
    if not isinstance(provenance.get("orientation"), dict):
        raise ValueError("sd-olat-heads provenance is missing orientation")
    if not isinstance(provenance.get("projection"), dict):
        raise ValueError("sd-olat-heads provenance is missing projection")


def _read_renderer_manifest(
    path: Path,
    render_dir: Path,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], Path]]]:
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

    rendered_conditions = payload.get("conditions")
    if not isinstance(rendered_conditions, list) or len(rendered_conditions) != (
        EXPECTED_HEAD_COUNT
    ):
        raise ValueError(
            f"renderer manifest must describe exactly {EXPECTED_HEAD_COUNT} "
            "ordered conditions"
        )

    lighting_preset = payload.get("lighting_preset", payload.get("preset"))
    if (
        "lighting_preset" in payload
        and "preset" in payload
        and payload["lighting_preset"] != payload["preset"]
    ):
        raise ValueError("renderer manifest has conflicting lighting preset names")
    if lighting_preset is not None and (
        not isinstance(lighting_preset, str) or not lighting_preset
    ):
        raise ValueError("renderer lighting preset must be a non-empty string")
    preset_provenance = payload.get("preset_provenance")
    if lighting_preset is not None and (
        not isinstance(preset_provenance, dict) or not preset_provenance
    ):
        raise ValueError("renderer lighting preset is missing preset_provenance")

    selected_absolute_indices = payload.get("selected_absolute_indices")
    if selected_absolute_indices is not None and (
        not isinstance(selected_absolute_indices, list)
        or len(selected_absolute_indices) != EXPECTED_HEAD_COUNT
    ):
        raise ValueError(
            "selected_absolute_indices must be null or align with all 36 conditions"
        )

    selected: list[tuple[dict[str, Any], Path]] = []
    identifiers: set[str] = set()
    missing: list[str] = []
    for position, rendered in enumerate(rendered_conditions):
        if not isinstance(rendered, dict):
            raise ValueError(
                f"renderer condition at ordered position {position} is not an object"
            )
        condition_id = rendered.get("condition_id")
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
        if not isinstance(rendered.get("source"), str) or not rendered["source"]:
            raise ValueError(f"renderer condition {condition_id} has no lighting source")
        _require_sha256(
            rendered.get("source_sha256"), f"condition {condition_id} source_sha256"
        )
        if not isinstance(rendered.get("source_kind"), str):
            raise ValueError(f"renderer condition {condition_id} has no source_kind")
        rotation = rendered.get("rotation_degrees")
        if isinstance(rotation, bool) or not isinstance(rotation, (int, float)):
            raise ValueError(f"renderer condition {condition_id} has invalid rotation")
        support_count = rendered.get("support_count")
        if (
            isinstance(support_count, bool)
            or not isinstance(support_count, int)
            or support_count <= 0
        ):
            raise ValueError(
                f"renderer condition {condition_id} has invalid support_count"
            )
        source_rank = rendered.get("source_rank")
        if source_rank is not None and (
            isinstance(source_rank, bool)
            or not isinstance(source_rank, int)
            or source_rank < 0
        ):
            raise ValueError(f"renderer condition {condition_id} has invalid source_rank")
        if lighting_preset is not None:
            _require_sha256(
                rendered.get("weight_sha256"),
                f"condition {condition_id} weight_sha256",
            )
        expected_render_sha256 = _require_sha256(
            rendered.get("render_sha256"),
            f"condition {condition_id} render_sha256",
        )
        recorded_render_path = rendered.get("render_path")
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
        if selected_absolute_indices is not None:
            absolute_index = selected_absolute_indices[position]
            if isinstance(absolute_index, bool) or not isinstance(absolute_index, int):
                raise ValueError("selected_absolute_indices must contain integers")
            if rendered.get("absolute_index") != absolute_index:
                raise ValueError(
                    f"absolute index diverges at ordered position {position}"
                )
        selected.append((rendered, render_path))
    if missing:
        detail = "\n  ".join(missing)
        raise FileNotFoundError(
            "missing predicted condition render(s); the contact sheet is all-or-nothing:\n  "
            + detail
        )

    if lighting_preset == "sd-olat-heads":
        ranks = [record.get("source_rank") for record in rendered_conditions]
        if ranks[:4] != [None] * 4 or ranks[4:] != list(EXPECTED_SD_SOURCE_RANKS):
            raise ValueError(
                "sd-olat-heads must order four calibrations followed by source ranks 1..32"
            )
        if any(
            record.get("source_kind") != "generated_calibration"
            for record in rendered_conditions[:4]
        ):
            raise ValueError("sd-olat-heads must begin with four generated calibrations")
        _validate_sd_olat_preset(preset_provenance, rendered_conditions)

    validation = payload.get("validation")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise ValueError("renderer manifest does not contain a passing validation record")
    for required in ("max_abs", "mean_abs", "render_sha256", "canonical_sha256"):
        if required not in validation:
            raise ValueError(f"renderer validation is missing {required!r}")

    material = payload.get("material")
    if not isinstance(material, dict):
        raise ValueError("renderer manifest is missing material provenance")
    _require_sha256(material.get("sha256"), "renderer material.sha256")
    renderer = payload.get("renderer")
    if not isinstance(renderer, dict) or not renderer:
        raise ValueError("renderer manifest is missing renderer provenance")
    input_hash_checks = payload.get("input_hash_checks")
    _require_passing_hash_checks(input_hash_checks, "input_hash_checks")

    # Copy durable provenance and validation statistics, not paths into the
    # private stage that will be deleted after final-output validation.
    durable_validation = {
        key: value for key, value in validation.items() if key != "render_path"
    }
    provenance = {
        "input_hash_checks": input_hash_checks,
        "lighting_preset": lighting_preset,
        "manifest_sha256": _sha256(path),
        "material": material,
        "preset_provenance": preset_provenance,
        "profile": payload.get("profile"),
        "renderer": renderer,
        "selected_absolute_indices": selected_absolute_indices,
        "source_schema": payload.get("schema"),
        "validation": durable_validation,
    }
    return provenance, selected


def _validate_final_outputs(
    rendering_path: Path,
    manifest_path: Path,
    *,
    expected_size: tuple[int, int],
    expected_condition_ids: list[str],
) -> None:
    from PIL import Image

    with Image.open(rendering_path) as image:
        image.load()
        if image.mode != "RGB" or image.size != expected_size:
            raise RuntimeError(
                f"invalid final rendering image: mode={image.mode}, size={image.size}"
            )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "ictpolarreal.sd-olat-head-rendering.v1":
        raise RuntimeError("invalid final rendering manifest schema")
    if payload.get("rendering_sha256") != _sha256(rendering_path):
        raise RuntimeError("final rendering hash does not match rendering.json")
    conditions = payload.get("conditions", [])
    if [record.get("condition_id") for record in conditions] != expected_condition_ids:
        raise RuntimeError("final rendering manifest changed the authoritative order")
    selection = payload.get("selection", {})
    if selection.get("count") != EXPECTED_HEAD_COUNT:
        raise RuntimeError("final rendering manifest has the wrong condition count")
    if ".rendering_stage" in manifest_path.read_text(encoding="utf-8"):
        raise RuntimeError("final rendering manifest contains a private staging path")


def compose_head_rendering(
    camera_dir: str | Path,
    *,
    render_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    keep_heads: bool = False,
) -> tuple[Path, Path]:
    """Write the singular ``rendering.png`` and its ``rendering.json`` record.

    ``render_dir/render_manifest.json`` defines the authoritative condition
    order.  Only its predicted object renders are shown; no material maps,
    references, error maps, or light-probe tiles are added to the sheet.
    """

    from PIL import Image, ImageDraw

    camera_dir = Path(camera_dir)
    material_dir = camera_dir / "material" / "olat"
    render_dir = (
        Path(render_dir)
        if render_dir is not None
        else material_dir / ".rendering_stage"
    )
    output_dir = Path(output_dir) if output_dir is not None else material_dir
    conditions_path = camera_dir / "evaluation" / "assets" / "conditions.json"
    if not keep_heads:
        try:
            output_dir.resolve().relative_to(render_dir.resolve())
        except ValueError:
            pass
        else:
            raise ValueError(
                "output_dir cannot be inside the private render_dir when staged "
                "condition tiles will be deleted"
            )
    renderer_manifest_path = render_dir / "render_manifest.json"
    render_provenance, selected = _read_renderer_manifest(
        renderer_manifest_path, render_dir
    )

    regular_font, regular_font_name = _load_font(22, bold=False)
    label_font, label_font_name = _load_font(25, bold=True)
    ordinal_font, ordinal_font_name = _load_font(30, bold=True)

    rows = (len(selected) + SHEET_COLUMNS - 1) // SHEET_COLUMNS
    tile_height = LABEL_HEIGHT + IMAGE_HEIGHT
    sheet_width = 2 * PADDING + SHEET_COLUMNS * TILE_WIDTH + (SHEET_COLUMNS - 1) * GAP
    sheet_height = 2 * PADDING + rows * tile_height + (rows - 1) * GAP
    sheet = Image.new("RGB", (sheet_width, sheet_height), (5, 5, 7))

    manifest_conditions: list[dict[str, Any]] = []
    for ordinal, (condition, render_path) in enumerate(selected, start=1):
        tile = Image.new("RGB", (TILE_WIDTH, tile_height), (17, 18, 22))
        draw = ImageDraw.Draw(tile)
        draw.text((14, 11), f"{ordinal:02d}", font=ordinal_font, fill=(255, 255, 255))
        source_label = _ellipsize(
            draw,
            _display_source(condition),
            label_font,
            TILE_WIDTH - 80,
        )
        draw.text((72, 14), source_label, font=label_font, fill=(244, 244, 246))
        rotation = int(condition.get("rotation_degrees", 0))
        detail = _ellipsize(
            draw,
            _display_detail(condition),
            regular_font,
            TILE_WIDTH - 28,
        )
        draw.text((14, 61), detail, font=regular_font, fill=(185, 188, 196))
        draw.line((0, LABEL_HEIGHT - 1, TILE_WIDTH, LABEL_HEIGHT - 1), fill=(55, 57, 64))
        tile.paste(_fit_render(render_path), (0, LABEL_HEIGHT))

        column = (ordinal - 1) % SHEET_COLUMNS
        row = (ordinal - 1) // SHEET_COLUMNS
        x = PADDING + column * (TILE_WIDTH + GAP)
        y = PADDING + row * (tile_height + GAP)
        sheet.paste(tile, (x, y))

        output_condition = {
            "condition_id": condition["condition_id"],
            "display_label": source_label,
            "ordinal": ordinal,
            "render_sha256": condition["render_sha256"],
            "rotation_degrees": rotation,
            "source": condition.get("source"),
            "source_kind": condition.get("source_kind"),
            "source_sha256": condition.get("source_sha256"),
            "split": condition.get("split"),
            "support_count": condition.get("support_count"),
        }
        for optional_key in (
            "absolute_index",
            "normalization_scale",
            "orientation",
            "preset_index",
            "projection",
            "source_rank",
            "variance_score",
            "weight_sha256",
        ):
            if optional_key in condition:
                output_condition[optional_key] = condition[optional_key]
        manifest_conditions.append(output_condition)

    output_dir.mkdir(parents=True, exist_ok=True)
    rendering_path = output_dir / "rendering.png"
    temporary_rendering = output_dir / ".rendering.png.pending"
    sheet.save(temporary_rendering, format="PNG", compress_level=6)

    manifest_path = output_dir / "rendering.json"
    temporary_manifest = output_dir / ".rendering.json.pending"
    input_record: dict[str, Any] = {
        "selection_authority": "ordered conditions in saved renderer manifest",
        "selected_condition_tiles_retained": bool(keep_heads),
    }
    if conditions_path.is_file():
        input_record.update(
            {
                "recorded_conditions_manifest": _relative_path(
                    conditions_path, output_dir
                ),
                "recorded_conditions_manifest_sha256": _sha256(conditions_path),
            }
        )
    payload = {
        "conditions": manifest_conditions,
        "input": input_record,
        "layout": {
            "columns": SHEET_COLUMNS,
            "gap_px": GAP,
            "image_height_px": IMAGE_HEIGHT,
            "label_height_px": LABEL_HEIGHT,
            "padding_px": PADDING,
            "rows": rows,
            "sheet_height_px": sheet_height,
            "sheet_width_px": sheet_width,
            "tile_width_px": TILE_WIDTH,
        },
        "presentation": "predicted object renderings only",
        "render_provenance": render_provenance,
        "rendering": "rendering.png",
        "rendering_sha256": _sha256(temporary_rendering),
        "schema": "ictpolarreal.sd-olat-head-rendering.v1",
        "selection": {
            "authority": "render_manifest.conditions order",
            "count": len(selected),
            "lighting_preset": render_provenance.get("lighting_preset"),
        },
        "typography": {
            "label_font": label_font_name,
            "ordinal_font": ordinal_font_name,
            "regular_font": regular_font_name,
        },
    }
    expected_condition_ids = [condition["condition_id"] for condition, _ in selected]
    try:
        _write_json_atomic(temporary_manifest, payload)
        _validate_final_outputs(
            temporary_rendering,
            temporary_manifest,
            expected_size=(sheet_width, sheet_height),
            expected_condition_ids=expected_condition_ids,
        )
        temporary_rendering.replace(rendering_path)
        temporary_manifest.replace(manifest_path)
        _validate_final_outputs(
            rendering_path,
            manifest_path,
            expected_size=(sheet_width, sheet_height),
            expected_condition_ids=expected_condition_ids,
        )
    finally:
        temporary_rendering.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
    if not keep_heads:
        default_render_dir = material_dir / ".rendering_stage"
        if render_dir.resolve() == default_render_dir.resolve():
            # This exact hidden directory is owned by the render/composition
            # handoff, including its validation images and renderer manifest.
            shutil.rmtree(render_dir)
        else:
            for _, render_path in selected:
                render_path.unlink()
            # Never recursively delete a caller-supplied directory, since it may
            # contain unrelated renderer logs or outputs.
            try:
                (render_dir / "conditions").rmdir()
            except OSError:
                pass
    return rendering_path, manifest_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compose the ordered 36-condition, predicted-render-only SD-OLAT-style "
            "material acquisition view."
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
        "--output-dir",
        type=Path,
        help="pickup output directory (default: <camera-dir>/material/olat)",
    )
    parser.add_argument(
        "--keep-heads",
        action="store_true",
        help="retain the 36 private per-condition PNGs after composition",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rendering_path, manifest_path = compose_head_rendering(
        args.camera_dir,
        render_dir=args.render_dir,
        output_dir=args.output_dir,
        keep_heads=args.keep_heads,
    )
    print(rendering_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
