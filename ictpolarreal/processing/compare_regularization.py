from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ictpolarreal.data.dataset import CameraSample
from ictpolarreal.processing.material_decomposition import (
    load_end2end_view_directions,
)
from ictpolarreal.utils.io import read_image


SCALAR_MAPS = (
    "roughness",
    "specular",
    "metallic",
    "subsurface",
    "specularTint",
    "anisotropic",
    "clearcoat",
    "clearcoatGloss",
)
EVALUATION_LIGHTING = ("olat", "hdri")
MATERIAL_LABEL_GUTTER = 480


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compose a controlled baseline-versus-regularized ICTPolarReal "
            "material acquisition report."
        )
    )
    parser.add_argument("--baseline", required=True, help="Baseline camera result root.")
    parser.add_argument(
        "--regularized", required=True, help="Regularized camera result root."
    )
    parser.add_argument("--output", required=True, help="Clean comparison report root.")
    parser.add_argument(
        "--mask",
        default=None,
        help="Optional exact fit mask used for interior map-noise measurements.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "Dataset root used to reconstruct the acquisition's capture × n-dot-v "
            "fit mask. Preferred over --mask."
        ),
    )
    args = parser.parse_args()

    compose_regularization_comparison(
        args.baseline,
        args.regularized,
        args.output,
        mask_path=args.mask,
        data_root=args.data_root,
    )


def compose_regularization_comparison(
    baseline_camera: str | Path,
    regularized_camera: str | Path,
    output_dir: str | Path,
    *,
    mask_path: str | Path | None = None,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    baseline_camera = Path(baseline_camera).expanduser().resolve()
    regularized_camera = Path(regularized_camera).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    for camera_dir in (baseline_camera, regularized_camera):
        if (
            output_dir == camera_dir
            or output_dir in camera_dir.parents
            or camera_dir in output_dir.parents
        ):
            raise ValueError(
                "comparison output must not overlap either acquisition root: "
                f"output={output_dir}, acquisition={camera_dir}"
            )
    baseline_manifest = _load_complete_manifest(baseline_camera)
    regularized_manifest = _load_complete_manifest(regularized_camera)
    profiles = tuple(baseline_manifest["profiles"])
    if tuple(regularized_manifest.get("profiles", ())) != profiles:
        raise ValueError("baseline and regularized runs do not contain the same profiles")

    acquisitions = {
        "baseline": _load_acquisitions(baseline_camera, profiles),
        "regularized": _load_acquisitions(regularized_camera, profiles),
    }
    contract = _validate_comparison_contract(acquisitions, profiles)
    baseline_weight = float(
        acquisitions["baseline"][profiles[0]]["regularization"]["weight"]
    )
    regularized_weight = float(
        acquisitions["regularized"][profiles[0]]["regularization"]["weight"]
    )
    labels = {
        "baseline": f"Baseline · TV weight λ={baseline_weight:g}",
        "regularized": f"Regularized · TV weight λ={regularized_weight:g}",
    }

    sample_map = baseline_camera / "material" / profiles[0] / "maps" / "roughness.png"
    with Image.open(sample_map) as image:
        sample_size = image.size
    if mask_path is not None and data_root is not None:
        raise ValueError("pass either --data-root or --mask, not both")
    if data_root is not None:
        data_root_path = Path(data_root).expanduser().resolve()
        mask = _fit_mask_from_data_root(
            data_root_path,
            baseline_camera.parent.name,
            baseline_camera.name,
            sample_size,
        )
        mask_source = f"reconstructed acquisition fit mask from {data_root_path}"
    elif mask_path is None:
        mask = _infer_foreground_mask(
            baseline_camera / "material" / profiles[0] / "maps" / "baseColor.png"
        )
        mask_source = "fallback inferred from nonzero baseline baseColor"
    else:
        mask_file = Path(mask_path).expanduser().resolve()
        mask = _read_mask(mask_file, sample_size)
        mask_source = str(mask_file)
    _validate_fit_mask_hash(mask, acquisitions, profiles)
    interior_mask = _erode_mask(mask, radius=2)
    if not np.any(interior_mask):
        raise ValueError("comparison mask has no five-pixel interior")

    map_metrics = _measure_material_maps(
        baseline_camera,
        regularized_camera,
        profiles,
        interior_mask,
    )
    evaluation_metrics = _collect_evaluation_metrics(acquisitions, profiles)
    summary = _build_summary(
        baseline_camera,
        regularized_camera,
        profiles,
        labels,
        contract,
        map_metrics,
        evaluation_metrics,
        mask_source,
    )

    stage = output_dir.with_name(f".{output_dir.name}.tmp")
    backup = output_dir.with_name(f".{output_dir.name}.previous")
    for stale in (stage, backup):
        if stale.exists():
            shutil.rmtree(stale)
    stage.mkdir(parents=True)
    try:
        material_paths = []
        for profile in profiles:
            path = stage / "material" / f"{profile}.png"
            _write_material_comparison(
                baseline_camera,
                regularized_camera,
                profile,
                labels,
                map_metrics[profile],
                path,
            )
            material_paths.append(path)

        evaluation_paths = []
        for lighting in EVALUATION_LIGHTING:
            path = stage / "evaluation" / f"{lighting}.png"
            case_id = _shared_representative_case(
                baseline_camera,
                regularized_camera,
                lighting,
            )
            summary["representative_cases"][lighting] = case_id
            _write_evaluation_comparison(
                baseline_camera,
                regularized_camera,
                profiles,
                lighting,
                case_id,
                evaluation_metrics,
                path,
            )
            evaluation_paths.append(path)

        _write_metrics_csv(stage / "metrics.csv", map_metrics, evaluation_metrics)
        _write_json(stage / "summary.json", summary)
        _write_overview(
            baseline_camera,
            regularized_camera,
            profiles,
            map_metrics,
            evaluation_metrics,
            summary,
            stage / "overview.png",
        )
        _validate_report(stage, profiles)

        if output_dir.exists():
            output_dir.replace(backup)
        try:
            stage.replace(output_dir)
        except Exception:
            if backup.exists() and not output_dir.exists():
                backup.replace(output_dir)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return summary


def _load_complete_manifest(camera_dir: Path) -> dict[str, Any]:
    manifest_path = camera_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing acquisition manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("evaluation", {}).get("status") != "complete":
        raise ValueError(f"acquisition is not complete: {manifest_path}")
    profiles = manifest.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError(f"manifest has no profiles: {manifest_path}")
    return manifest


def _load_acquisitions(
    camera_dir: Path, profiles: Sequence[str]
) -> dict[str, dict[str, Any]]:
    return {
        profile: json.loads(
            (camera_dir / "material" / profile / "acquisition.json").read_text(
                encoding="utf-8"
            )
        )
        for profile in profiles
    }


def _validate_comparison_contract(
    acquisitions: dict[str, dict[str, dict[str, Any]]],
    profiles: Sequence[str],
) -> dict[str, Any]:
    configurations = {}
    for variant in ("baseline", "regularized"):
        per_profile = {
            profile: _regularization_configuration(acquisitions[variant][profile])
            for profile in profiles
        }
        unique = set(per_profile.values())
        if len(unique) != 1:
            raise ValueError(
                f"{variant} regularization configuration differs across profiles: "
                f"{per_profile}"
            )
        configurations[variant] = next(iter(unique))

    baseline_kind, baseline_parameters, baseline_weight = configurations["baseline"]
    regularized_kind, regularized_parameters, regularized_weight = configurations[
        "regularized"
    ]
    if (baseline_kind, baseline_parameters) != (
        regularized_kind,
        regularized_parameters,
    ):
        raise ValueError(
            "baseline and regularized runs use different regularizer kinds or maps"
        )
    if baseline_weight == regularized_weight:
        raise ValueError("baseline and regularized runs use the same TV weight")

    for profile in profiles:
        baseline = acquisitions["baseline"][profile]
        regularized = acquisitions["regularized"][profile]
        baseline_signature = _normalized_comparison_signature(baseline)
        regularized_signature = _normalized_comparison_signature(regularized)
        if baseline_signature != regularized_signature:
            raise ValueError(
                f"{profile} comparison is not controlled; checkpoint signatures "
                "differ beyond regularization.weight"
            )
    sources = {
        acquisitions[variant][profile].get("base_color_source")
        for variant in ("baseline", "regularized")
        for profile in profiles
    }
    if sources != {"dataset_albedo"}:
        raise ValueError(
            "regularization comparison must use dataset_albedo in both runs; "
            f"found {sorted(str(source) for source in sources)}"
        )
    return {
        "controlled": True,
        "matched_fields": [
            "checkpoint_signature except regularization.weight",
            "regularization.kind",
            "regularization.parameters",
            "base_color_source",
        ],
        "only_intended_difference": "masked scalar-map TV weight λ",
        "base_color_source": "dataset_albedo",
        "regularization_kind": baseline_kind,
        "regularized_parameters": list(baseline_parameters),
        "baseline_tv_weight": baseline_weight,
        "regularized_tv_weight": regularized_weight,
    }


def _regularization_configuration(
    acquisition: dict[str, Any],
) -> tuple[str, tuple[str, ...], float]:
    regularization = acquisition.get("regularization")
    if not isinstance(regularization, dict):
        raise ValueError("acquisition is missing regularization provenance")
    kind = regularization.get("kind")
    parameters = regularization.get("parameters")
    try:
        weight = float(regularization["weight"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("acquisition has an invalid TV weight") from exc
    if not isinstance(kind, str) or not kind:
        raise ValueError("acquisition has an invalid regularization kind")
    if not isinstance(parameters, list) or not parameters or not all(
        isinstance(name, str) and name for name in parameters
    ):
        raise ValueError("acquisition has invalid regularized parameters")
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("acquisition has a non-finite or negative TV weight")

    signature_regularization = acquisition.get("checkpoint_signature", {}).get(
        "regularization"
    )
    expected = {
        "kind": kind,
        "parameters": parameters,
        "weight": weight,
    }
    if signature_regularization != expected:
        raise ValueError(
            "acquisition and checkpoint signature regularization provenance differ"
        )
    return kind, tuple(parameters), weight


def _normalized_comparison_signature(acquisition: dict[str, Any]) -> dict[str, Any]:
    signature = acquisition.get("checkpoint_signature")
    if not isinstance(signature, dict):
        raise ValueError("acquisition is missing its checkpoint signature")
    normalized = json.loads(json.dumps(signature))
    regularization = normalized.get("regularization")
    if not isinstance(regularization, dict) or "weight" not in regularization:
        raise ValueError("checkpoint signature is missing its TV weight")
    regularization["weight"] = "<comparison-variable>"
    return normalized


def _read_mask(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask = image.convert("L")
        if mask.size != expected_size:
            mask = mask.resize(expected_size, Image.Resampling.NEAREST)
        return np.asarray(mask, dtype=np.uint8) > 127


def _fit_mask_from_data_root(
    data_root: Path,
    object_name: str,
    camera_name: str,
    expected_size: tuple[int, int],
) -> np.ndarray:
    camera_dir = data_root / object_name / camera_name
    sample = CameraSample(object_name, camera_name, camera_dir)
    normal_path = sample.image_path("normal")
    mask_path = sample.image_path("mask")
    if normal_path is None or mask_path is None:
        raise FileNotFoundError(
            f"fit-mask reconstruction needs normal and mask under {camera_dir}"
        )
    normal = read_image(normal_path)
    capture_mask = read_image(mask_path, channels=1)
    expected_shape = (expected_size[1], expected_size[0])
    if normal.shape[:2] != expected_shape or capture_mask.shape[:2] != expected_shape:
        raise ValueError(
            "fit-mask inputs do not match material map geometry: "
            f"normal={normal.shape[:2]}, mask={capture_mask.shape[:2]}, "
            f"expected={expected_shape}"
        )
    view = load_end2end_view_directions(data_root, sample, expected_shape)
    normal = _normalize_vectors(normal)
    view = _normalize_vectors(view)
    front_facing = np.sum(normal * view, axis=-1) > 1e-4
    return (capture_mask[..., 0] > 0.5) & front_facing


def _normalize_vectors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def _validate_fit_mask_hash(
    mask: np.ndarray,
    acquisitions: dict[str, dict[str, dict[str, Any]]],
    profiles: Sequence[str],
) -> None:
    foreground = np.ascontiguousarray(mask.astype(np.float32)[..., None])
    actual = hashlib.sha256(memoryview(foreground).cast("B")).hexdigest()
    expected = {
        acquisitions[variant][profile].get("input_hashes", {}).get(
            "foreground_sha256"
        )
        for variant in ("baseline", "regularized")
        for profile in profiles
    }
    expected.discard(None)
    if expected and expected != {actual}:
        raise ValueError(
            "comparison fit mask does not match acquisition provenance: "
            f"actual={actual}, expected={sorted(expected)}"
        )


def _infer_foreground_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.any(values > 0, axis=-1)


def _erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool, copy=True)
    size = 2 * radius + 1
    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    return windows.all(axis=(-2, -1))


def _read_scalar_map(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def _map_noise_metrics(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    horizontal_mask = mask[:, 1:] & mask[:, :-1]
    vertical_mask = mask[1:, :] & mask[:-1, :]
    horizontal = np.abs(values[:, 1:] - values[:, :-1])[horizontal_mask]
    vertical = np.abs(values[1:, :] - values[:-1, :])[vertical_mask]
    tv = 0.5 * (
        float(horizontal.mean()) if horizontal.size else 0.0
    ) + 0.5 * (float(vertical.mean()) if vertical.size else 0.0)

    padded = np.pad(values, 2, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (5, 5))
    median = np.median(windows, axis=(-2, -1))
    residual = np.abs(values - median)[mask]
    return {
        "neighbor_tv": tv,
        "median5_residual_mae": float(residual.mean()),
        "speckle_fraction_gt_0p05": float(np.mean(residual > 0.05)),
    }


def _measure_material_maps(
    baseline_camera: Path,
    regularized_camera: Path,
    profiles: Sequence[str],
    mask: np.ndarray,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    output = {}
    for profile in profiles:
        output[profile] = {}
        for map_name in SCALAR_MAPS:
            variants = {}
            for variant, camera in (
                ("baseline", baseline_camera),
                ("regularized", regularized_camera),
            ):
                path = camera / "material" / profile / "maps" / f"{map_name}.png"
                variants[variant] = _map_noise_metrics(_read_scalar_map(path), mask)
            output[profile][map_name] = variants
    return output


def _collect_evaluation_metrics(
    acquisitions: dict[str, dict[str, dict[str, Any]]],
    profiles: Sequence[str],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    output = {}
    for profile in profiles:
        output[profile] = {}
        for lighting in EVALUATION_LIGHTING:
            output[profile][lighting] = {
                variant: {
                    name: float(value)
                    for name, value in acquisitions[variant][profile]["evaluation"][
                        "evaluations"
                    ][lighting]["metrics"].items()
                    if isinstance(value, (int, float))
                }
                for variant in ("baseline", "regularized")
            }
    return output


def _build_summary(
    baseline_camera: Path,
    regularized_camera: Path,
    profiles: Sequence[str],
    labels: dict[str, str],
    contract: dict[str, Any],
    map_metrics: dict[str, Any],
    evaluation_metrics: dict[str, Any],
    mask_source: str,
) -> dict[str, Any]:
    baseline_tv = np.mean(
        [
            map_metrics[profile][name]["baseline"]["neighbor_tv"]
            for profile in profiles
            for name in SCALAR_MAPS
        ]
    )
    regularized_tv = np.mean(
        [
            map_metrics[profile][name]["regularized"]["neighbor_tv"]
            for profile in profiles
            for name in SCALAR_MAPS
        ]
    )
    baseline_hf = np.mean(
        [
            map_metrics[profile][name]["baseline"]["median5_residual_mae"]
            for profile in profiles
            for name in SCALAR_MAPS
        ]
    )
    regularized_hf = np.mean(
        [
            map_metrics[profile][name]["regularized"]["median5_residual_mae"]
            for profile in profiles
            for name in SCALAR_MAPS
        ]
    )
    evaluation_summary = {}
    for lighting in EVALUATION_LIGHTING:
        baseline_psnr = np.mean(
            [evaluation_metrics[p][lighting]["baseline"]["psnr"] for p in profiles]
        )
        regularized_psnr = np.mean(
            [evaluation_metrics[p][lighting]["regularized"]["psnr"] for p in profiles]
        )
        baseline_ssim = np.mean(
            [
                evaluation_metrics[p][lighting]["baseline"]["ssim_global"]
                for p in profiles
            ]
        )
        regularized_ssim = np.mean(
            [
                evaluation_metrics[p][lighting]["regularized"]["ssim_global"]
                for p in profiles
            ]
        )
        evaluation_summary[lighting] = {
            "mean_psnr": {
                "baseline": float(baseline_psnr),
                "regularized": float(regularized_psnr),
                "delta": float(regularized_psnr - baseline_psnr),
            },
            "mean_ssim_global": {
                "baseline": float(baseline_ssim),
                "regularized": float(regularized_ssim),
                "delta": float(regularized_ssim - baseline_ssim),
            },
        }
    return {
        "schema": "ictpolarreal.regularization-comparison.v1",
        "title": "Camera 07 · Scalar-map regularization comparison",
        "baseline": str(baseline_camera),
        "regularized": str(regularized_camera),
        "labels": labels,
        "profiles": list(profiles),
        "maps": list(SCALAR_MAPS),
        "mask_source": mask_source,
        "comparison_contract": contract,
        "aggregate_map_noise": {
            "neighbor_tv": {
                "baseline": float(baseline_tv),
                "regularized": float(regularized_tv),
                "reduction_fraction": _reduction_fraction(
                    baseline_tv, regularized_tv
                ),
            },
            "median5_residual_mae": {
                "baseline": float(baseline_hf),
                "regularized": float(regularized_hf),
                "reduction_fraction": _reduction_fraction(
                    baseline_hf, regularized_hf
                ),
            },
        },
        "aggregate_evaluation": evaluation_summary,
        "representative_cases": {},
        "artifacts": {
            "overview": "overview.png",
            "metrics": "metrics.csv",
            "material": {profile: f"material/{profile}.png" for profile in profiles},
            "evaluation": {
                lighting: f"evaluation/{lighting}.png"
                for lighting in EVALUATION_LIGHTING
            },
        },
    }


def _reduction_fraction(baseline: float, regularized: float) -> float:
    return float((baseline - regularized) / baseline) if baseline > 0 else 0.0


def _shared_representative_case(
    baseline_camera: Path, regularized_camera: Path, lighting: str
) -> str:
    summary = json.loads(
        (baseline_camera / "evaluation" / "summary.json").read_text(encoding="utf-8")
    )
    case_id = summary["suites"][lighting]["representative_case"]
    for camera in (baseline_camera, regularized_camera):
        if not (camera / "evaluation" / lighting / "cases" / case_id).is_dir():
            raise FileNotFoundError(
                f"shared {lighting} representative case is missing: {case_id}"
            )
    return str(case_id)


def _write_material_comparison(
    baseline_camera: Path,
    regularized_camera: Path,
    profile: str,
    labels: dict[str, str],
    metrics: dict[str, Any],
    output: Path,
) -> None:
    with Image.open(
        baseline_camera / "material" / profile / "maps" / f"{SCALAR_MAPS[0]}.png"
    ) as first:
        tile_width, tile_height = first.size
    # Keep the full variant/weight label and the row-level metric outside the
    # map grid.  A 300 px gutter clipped ``Regularized · TV weight λ=0.01`` at
    # the native report font size, which made the detailed sheet ambiguous.
    left = MATERIAL_LABEL_GUTTER
    top = 100
    group_header = 58
    caption_height = 52
    row_height = tile_height + caption_height
    map_groups = (SCALAR_MAPS[:4], SCALAR_MAPS[4:])
    group_height = group_header + 2 * row_height
    width = left + tile_width * 4
    height = top + group_height * len(map_groups)
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (24, 20),
        f"{profile.upper()} fit · scalar material maps",
        font=_font(48, bold=True),
        fill="white",
    )
    for group_index, map_group in enumerate(map_groups):
        group_y = top + group_index * group_height
        for column, map_name in enumerate(map_group):
            x = left + column * tile_width
            _draw_centered_text(
                draw,
                (x, group_y, tile_width, group_header),
                _display_name(map_name),
                _font(27, bold=True),
                "white",
            )
        for row, (variant, camera) in enumerate(
            (("baseline", baseline_camera), ("regularized", regularized_camera))
        ):
            y = group_y + group_header + row * row_height
            mean_tv = np.mean(
                [metrics[name][variant]["neighbor_tv"] for name in map_group]
            )
            draw.multiline_text(
                (24, y + 24),
                f"{labels[variant]}\nneighbor variation {mean_tv:.4f}",
                font=_font(29, bold=True),
                fill="white",
                spacing=12,
            )
            for column, map_name in enumerate(map_group):
                x = left + column * tile_width
                with Image.open(
                    camera / "material" / profile / "maps" / f"{map_name}.png"
                ) as image:
                    canvas.paste(image.convert("RGB"), (x, y))
                values = metrics[map_name][variant]
                _draw_centered_text(
                    draw,
                    (x, y + tile_height, tile_width, caption_height),
                    (
                        f"variation {values['neighbor_tv']:.3f} · "
                        f"HF {values['median5_residual_mae']:.3f}"
                    ),
                    _font(22),
                    "white",
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _write_evaluation_comparison(
    baseline_camera: Path,
    regularized_camera: Path,
    profiles: Sequence[str],
    lighting: str,
    case_id: str,
    metrics: dict[str, Any],
    output: Path,
) -> None:
    case = baseline_camera / "evaluation" / lighting / "cases" / case_id
    with Image.open(case / "reference.png") as reference:
        tile_width, tile_height = reference.size
    columns = ["Reference", "Baseline", "Regularized", "Baseline error", "Regularized error"]
    include_lighting = lighting == "hdri"
    if include_lighting:
        columns.insert(0, "Lighting")
    left = 300
    top = 150
    width = left + tile_width * len(columns)
    height = top + tile_height * len(profiles)
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (24, 18),
        f"{lighting.upper()} held-out relighting · shared case {case_id}",
        font=_font(45, bold=True),
        fill="white",
    )
    draw.text(
        (24, 78),
        "Errors use the acquisition report's fixed visualization scale.",
        font=_font(24),
        fill=(210, 210, 210),
    )
    for column, name in enumerate(columns):
        _draw_centered_text(
            draw,
            (left + column * tile_width, 112, tile_width, 36),
            name,
            _font(22, bold=True),
            "white",
        )
    for row, profile in enumerate(profiles):
        y = top + row * tile_height
        baseline_metric = metrics[profile][lighting]["baseline"]
        regularized_metric = metrics[profile][lighting]["regularized"]
        draw.multiline_text(
            (24, y + 24),
            (
                f"{profile.upper()} fit\n"
                f"PSNR {baseline_metric['psnr']:.2f} → {regularized_metric['psnr']:.2f}\n"
                f"SSIM {baseline_metric['ssim_global']:.3f} → "
                f"{regularized_metric['ssim_global']:.3f}"
            ),
            font=_font(25, bold=True),
            fill="white",
            spacing=10,
        )
        panel_paths = []
        if include_lighting:
            panel_paths.append(case / "lighting.png")
        panel_paths.extend(
            [
                case / "reference.png",
                case / "predictions" / f"{profile}.png",
                regularized_camera
                / "evaluation"
                / lighting
                / "cases"
                / case_id
                / "predictions"
                / f"{profile}.png",
                case / "errors" / f"{profile}.png",
                regularized_camera
                / "evaluation"
                / lighting
                / "cases"
                / case_id
                / "errors"
                / f"{profile}.png",
            ]
        )
        for column, path in enumerate(panel_paths):
            with Image.open(path) as image:
                panel = image.convert("RGB")
                if panel.size != (tile_width, tile_height):
                    panel = panel.resize(
                        (tile_width, tile_height), Image.Resampling.LANCZOS
                    )
                canvas.paste(panel, (left + column * tile_width, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _write_metrics_csv(
    path: Path,
    map_metrics: dict[str, Any],
    evaluation_metrics: dict[str, Any],
) -> None:
    rows = []
    for profile, maps in map_metrics.items():
        for map_name, variants in maps.items():
            for metric in (
                "neighbor_tv",
                "median5_residual_mae",
                "speckle_fraction_gt_0p05",
            ):
                baseline = variants["baseline"][metric]
                regularized = variants["regularized"][metric]
                rows.append(
                    {
                        "category": "material_noise",
                        "profile": profile,
                        "target": map_name,
                        "metric": metric,
                        "baseline": baseline,
                        "regularized": regularized,
                        "delta": regularized - baseline,
                        "reduction_fraction": _reduction_fraction(
                            baseline, regularized
                        ),
                    }
                )
    for profile, suites in evaluation_metrics.items():
        for lighting, variants in suites.items():
            shared = sorted(set(variants["baseline"]) & set(variants["regularized"]))
            for metric in shared:
                baseline = variants["baseline"][metric]
                regularized = variants["regularized"][metric]
                rows.append(
                    {
                        "category": "relighting",
                        "profile": profile,
                        "target": lighting,
                        "metric": metric,
                        "baseline": baseline,
                        "regularized": regularized,
                        "delta": regularized - baseline,
                        "reduction_fraction": "",
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_overview(
    baseline_camera: Path,
    regularized_camera: Path,
    profiles: Sequence[str],
    map_metrics: dict[str, Any],
    evaluation_metrics: dict[str, Any],
    summary: dict[str, Any],
    output: Path,
) -> None:
    width = 1800
    header_height = 390
    hero_maps = ("roughness", "specular", "subsurface", "anisotropic")
    material_left = 260
    material_tile_width = 170
    with Image.open(
        baseline_camera / "material" / profiles[0] / "maps" / "roughness.png"
    ) as image:
        material_tile_height = round(
            material_tile_width * image.height / image.width
        )
    material_heading_height = 128
    material_row_height = material_tile_height + 26
    material_top = header_height
    material_end = (
        material_top
        + material_heading_height
        + material_row_height * len(profiles)
    )

    overview_profile = "mix" if "mix" in profiles else profiles[-1]
    olat_case = summary["representative_cases"]["olat"]
    with Image.open(
        baseline_camera / "evaluation" / "olat" / "cases" / olat_case / "reference.png"
    ) as image:
        evaluation_tile_width = 210
        evaluation_tile_height = round(
            evaluation_tile_width * image.height / image.width
        )
    evaluation_top = material_end + 36
    evaluation_heading_height = 90
    evaluation_row_height = 58 + evaluation_tile_height + 30
    height = (
        evaluation_top
        + evaluation_heading_height
        + evaluation_row_height * len(EVALUATION_LIGHTING)
        + 30
    )
    canvas = Image.new("RGB", (width, height), (12, 12, 12))
    draw = ImageDraw.Draw(canvas)
    draw.text((36, 24), summary["title"], font=_font(58, bold=True), fill="white")
    contract = summary["comparison_contract"]
    draw.text(
        (42, 92),
        (
            "Controlled A/B · TV weight λ "
            f"{contract['baseline_tv_weight']:g} → "
            f"{contract['regularized_tv_weight']:g}"
        ),
        font=_font(30, bold=True),
        fill=(151, 205, 255),
    )
    noise = summary["aggregate_map_noise"]
    olat = summary["aggregate_evaluation"]["olat"]
    hdri = summary["aggregate_evaluation"]["hdri"]
    lines = [
        (
            f"Map neighbor variation: {noise['neighbor_tv']['baseline']:.4f} → "
            f"{noise['neighbor_tv']['regularized']:.4f} "
            f"({100.0 * noise['neighbor_tv']['reduction_fraction']:.1f}% lower)"
        ),
        (
            f"Median-filter residual: {noise['median5_residual_mae']['baseline']:.4f} → "
            f"{noise['median5_residual_mae']['regularized']:.4f} "
            f"({100.0 * noise['median5_residual_mae']['reduction_fraction']:.1f}% lower)"
        ),
        (
            f"Mean held-out OLAT: ΔPSNR {olat['mean_psnr']['delta']:+.3f} dB · "
            f"ΔSSIM {olat['mean_ssim_global']['delta']:+.4f}"
        ),
        (
            f"Mean held-out HDRI: ΔPSNR {hdri['mean_psnr']['delta']:+.3f} dB · "
            f"ΔSSIM {hdri['mean_ssim_global']['delta']:+.4f}"
        ),
    ]
    for index, line in enumerate(lines):
        draw.text(
            (42, 140 + index * 55),
            line,
            font=_font(32, bold=True),
            fill="white",
        )

    draw.text(
        (36, material_top + 10),
        "Material noise · baseline and regularized side by side",
        font=_font(43, bold=True),
        fill="white",
    )
    for map_index, map_name in enumerate(hero_maps):
        group_x = material_left + map_index * 2 * material_tile_width
        _draw_centered_text(
            draw,
            (group_x, material_top + 62, 2 * material_tile_width, 34),
            _display_name(map_name),
            _font(27, bold=True),
            "white",
        )
        _draw_centered_text(
            draw,
            (group_x, material_top + 96, 2 * material_tile_width, 28),
            "Baseline          Regularized",
            _font(21, bold=True),
            (210, 210, 210),
        )
    for profile_index, profile in enumerate(profiles):
        y = material_top + material_heading_height + profile_index * material_row_height
        baseline_mean = np.mean(
            [map_metrics[profile][name]["baseline"]["neighbor_tv"] for name in hero_maps]
        )
        regularized_mean = np.mean(
            [
                map_metrics[profile][name]["regularized"]["neighbor_tv"]
                for name in hero_maps
            ]
        )
        draw.multiline_text(
            (36, y + 26),
            (
                f"{profile.upper()} fit\n"
                f"Variation {baseline_mean:.3f} →\n{regularized_mean:.3f}"
            ),
            font=_font(28, bold=True),
            fill="white",
            spacing=10,
        )
        for map_index, map_name in enumerate(hero_maps):
            for variant_index, camera in enumerate(
                (baseline_camera, regularized_camera)
            ):
                x = (
                    material_left
                    + map_index * 2 * material_tile_width
                    + variant_index * material_tile_width
                )
                with Image.open(
                    camera / "material" / profile / "maps" / f"{map_name}.png"
                ) as image:
                    panel = image.convert("RGB").resize(
                        (material_tile_width, material_tile_height),
                        Image.Resampling.LANCZOS,
                    )
                canvas.paste(panel, (x, y))

    draw.text(
        (36, evaluation_top + 10),
        f"Relighting spot-check · {overview_profile.upper()} fit",
        font=_font(43, bold=True),
        fill="white",
    )
    for lighting_index, lighting in enumerate(EVALUATION_LIGHTING):
        block_y = (
            evaluation_top
            + evaluation_heading_height
            + lighting_index * evaluation_row_height
        )
        case_id = summary["representative_cases"][lighting]
        case = baseline_camera / "evaluation" / lighting / "cases" / case_id
        include_lighting = lighting == "hdri"
        columns = [
            "Reference",
            "Baseline",
            "Regularized",
            "Baseline error",
            "Regularized error",
        ]
        panel_paths = [
            case / "reference.png",
            case / "predictions" / f"{overview_profile}.png",
            regularized_camera
            / "evaluation"
            / lighting
            / "cases"
            / case_id
            / "predictions"
            / f"{overview_profile}.png",
            case / "errors" / f"{overview_profile}.png",
            regularized_camera
            / "evaluation"
            / lighting
            / "cases"
            / case_id
            / "errors"
            / f"{overview_profile}.png",
        ]
        if include_lighting:
            columns.insert(0, "Lighting")
            panel_paths.insert(0, case / "lighting.png")
        evaluation_left = width - len(columns) * evaluation_tile_width - 30
        metric_baseline = evaluation_metrics[overview_profile][lighting]["baseline"]
        metric_regularized = evaluation_metrics[overview_profile][lighting][
            "regularized"
        ]
        draw.multiline_text(
            (36, block_y + 68),
            (
                f"{lighting.upper()}\n"
                f"PSNR {metric_baseline['psnr']:.2f} → "
                f"{metric_regularized['psnr']:.2f}\n"
                f"SSIM {metric_baseline['ssim_global']:.3f} → "
                f"{metric_regularized['ssim_global']:.3f}"
            ),
            font=_font(27, bold=True),
            fill="white",
            spacing=9,
        )
        for column, (name, path) in enumerate(zip(columns, panel_paths)):
            x = evaluation_left + column * evaluation_tile_width
            _draw_centered_text(
                draw,
                (x, block_y, evaluation_tile_width, 48),
                name,
                _font(23, bold=True),
                "white",
            )
            with Image.open(path) as image:
                panel = image.convert("RGB").resize(
                    (evaluation_tile_width, evaluation_tile_height),
                    Image.Resampling.LANCZOS,
                )
            canvas.paste(panel, (x, block_y + 58))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _font(size: int, *, bold: bool = False):
    names = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    raise RuntimeError("a scalable TrueType font is required for comparison reports")


def _draw_centered_text(draw, box, text, font, fill) -> None:
    x, y, width, height = box
    bounds = draw.textbbox((0, 0), text, font=font)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    draw.text(
        (x + (width - text_width) / 2, y + (height - text_height) / 2 - bounds[1]),
        text,
        font=font,
        fill=fill,
    )


def _display_name(name: str) -> str:
    return {
        "specularTint": "specular tint",
        "clearcoatGloss": "clearcoat gloss",
    }.get(name, name)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _validate_report(stage: Path, profiles: Sequence[str]) -> None:
    required = [stage / "overview.png", stage / "summary.json", stage / "metrics.csv"]
    required.extend(stage / "material" / f"{profile}.png" for profile in profiles)
    required.extend(
        stage / "evaluation" / f"{lighting}.png"
        for lighting in EVALUATION_LIGHTING
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"regularization report is incomplete: {missing}")


if __name__ == "__main__":
    main()
