from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
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
from ictpolarreal.utils.metrics import (
    psnr as image_psnr,
    ssim_global as image_ssim_global,
)


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
QUALIFICATION_PROFILES = ("olat", "hdri", "mix")
MATERIAL_LABEL_GUTTER = 600
OVERVIEW_MATERIAL_LEFT = 360
DEFAULT_GUIDE_ALBEDO_SIGMA = 0.05
DEFAULT_GUIDE_NORMAL_SIGMA = 0.02
QUIET_GUIDE_PERCENTILE = 50.0
EDGE_GUIDE_PERCENTILE = 80.0
MEANINGFUL_EDGE_GRADIENT_MIN = 1.0 / 255.0
EDGE_NUMERICAL_ZERO = 1e-12
IMPULSE_FROZEN_ARTIFACT_SCHEMA = "ictpolarreal.impulse-frozen-artifact.v1"
FREQUENCY_FROZEN_ARTIFACT_SCHEMA = (
    "ictpolarreal.frequency-frozen-artifact.v1"
)
FREQUENCY_FROZEN_ARTIFACT_NAME = "frequency_consensus_frozen.npz"
FREQUENCY_BUNDLE_SCHEMA = "ictpolarreal.frequency-consensus-bundle.v1"
FREQUENCY_ADAPTIVE_BUNDLE_SCHEMA = (
    "ictpolarreal.frequency-consensus-adaptive-bundle.v1"
)
FREQUENCY_ADAPTIVE_CONSENSUS_ENTRY_SEMANTICS = (
    "strong_policy_eligible_and_final_target_differs_from_source"
)
FREQUENCY_DIAGNOSTIC_SCHEMA = (
    "ictpolarreal.frequency-consensus-diagnostic.v1"
)
FREQUENCY_DETAIL_CROP_BOX = (96, 160, 192, 256)
FREQUENCY_DETAIL_MAPS = (
    "subsurface",
    "specular",
    "anisotropic",
    "roughness",
)
FREQUENCY_DETAIL_TILE_LEFT = 220
FREQUENCY_DETAIL_TILE_TOP = 130
FREQUENCY_DETAIL_ROW_GAP = 42
FREQUENCY_ANNOTATION_LEFT = 18
FREQUENCY_ANNOTATION_GAP = 24
OVERVIEW_DETAIL_VARIANT_GAP = 24
OVERVIEW_DETAIL_GROUP_GAP = 64
FREQUENCY_BASE_BLEND = 0.40
FREQUENCY_EVIDENCE_THRESHOLD = 0.05
FREQUENCY_MIN_EVIDENCE_MAPS = 2
FREQUENCY_OWN_DEVIATION = 0.025
FREQUENCY_MEDIAN3_TARGET_WEIGHT = 0.90
FREQUENCY_MEDIAN7_TARGET_WEIGHT = 0.10
FREQUENCY_SPATIAL_SIGMA3 = 1.0
FREQUENCY_SPATIAL_SIGMA7 = 2.5
FREQUENCY_ALBEDO_SIGMA = 0.05
FREQUENCY_NORMAL_SIGMA = 0.02
FREQUENCY_EDGE_PERCENTILE = 80.0
FREQUENCY_GUIDE_BAND_THRESHOLD = 0.35
FREQUENCY_ADAPTIVE_STRONG_MEDIAN3_WEIGHT = 0.50
FREQUENCY_ADAPTIVE_STRONG_MEDIAN7_WEIGHT = 0.50
FREQUENCY_ADAPTIVE_HALO_V1_BLEND = 0.84
FREQUENCY_ADAPTIVE_FOCUSED_MAPS = ("anisotropic", "subsurface")
FREQUENCY_ADAPTIVE_EVIDENCE_CROP_SIZE = 96
FREQUENCY_ADAPTIVE_EVIDENCE_SCALE = 3
FREQUENCY_ADAPTIVE_EVIDENCE_SCHEMA = (
    "ictpolarreal.adaptive-cleanup-evidence.v1"
)
FREQUENCY_ADAPTIVE_GUIDE_SCALE_PERCENTILE = 90.0
FREQUENCY_EVALUATION_GUARD_SCHEMA = (
    "ictpolarreal.frequency-consensus-evaluation-guard.v1"
)
FREQUENCY_EVALUATION_MEAN_MSE_TOLERANCE = 1e-8
ACQUISITION_ERROR_HEATMAP_MAX = 0.25
ACQUISITION_ERROR_HEATMAP_POSITIONS = np.asarray(
    [0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32
)
ACQUISITION_ERROR_HEATMAP_COLORS = np.asarray(
    [
        [0, 0, 0],
        [15, 32, 110],
        [0, 180, 220],
        [255, 220, 35],
        [220, 25, 25],
    ],
    dtype=np.float32,
) / 255.0
EXPORT_QUANTIZATION_TOLERANCE = 0.5 / 255.0
_LEGACY_CHECKPOINT_SCHEMA = "ictpolarreal.end2end-checkpoint.v8"
_EDGE_CHECKPOINT_SCHEMA = "ictpolarreal.end2end-checkpoint.v9"
_STAGED_IMPULSE_CHECKPOINT_SCHEMA = "ictpolarreal.end2end-checkpoint.v10"
_PROXIMAL_V11_CHECKPOINT_SCHEMA = "ictpolarreal.end2end-checkpoint.v11"
_PROXIMAL_V12_CHECKPOINT_SCHEMA = "ictpolarreal.end2end-checkpoint.v12"
_FREQUENCY_CHECKPOINT_SCHEMA = "ictpolarreal.end2end-checkpoint.v13"
_CLEAN_CAPTURE_FIT_MASK_RULE = (
    "binary_clean_capture_mask_after_view_facing_normal_orientation"
)
_LEGACY_FRONT_FACING_FIT_MASK_RULE = (
    "binary_capture_mask_times_front_facing_n_dot_v"
)
_LEGACY_ADAPTER = (
    "ictpolarreal.profile-acquisition-adapter.v2",
    "ictpolarreal-masked-tv-v1",
)
_EDGE_ADAPTER = (
    "ictpolarreal.profile-acquisition-adapter.v3",
    "ictpolarreal-edge-aware-regularization-v2",
)
_STAGED_IMPULSE_ADAPTER = (
    "ictpolarreal.profile-acquisition-adapter.v4",
    "ictpolarreal-impulse-median-v3",
)
_PROXIMAL_V11_ADAPTER = (
    "ictpolarreal.profile-acquisition-adapter.v5",
    "ictpolarreal-impulse-proximal-v4",
)
_PROXIMAL_V12_ADAPTER = (
    "ictpolarreal.profile-acquisition-adapter.v6",
    "ictpolarreal-impulse-proximal-v5",
)
_FREQUENCY_ADAPTER = (
    "ictpolarreal.profile-acquisition-adapter.v7",
    "ictpolarreal-frequency-consensus-v1",
)
_FREQUENCY_ADAPTIVE_ADAPTER = (
    "ictpolarreal.profile-acquisition-adapter.v7",
    "ictpolarreal-frequency-consensus-adaptive-v1",
)
_FREQUENCY_V2_ADAPTER = (
    "ictpolarreal.profile-acquisition-adapter.v7",
    "ictpolarreal-frequency-consensus-v2",
)
_FREQUENCY_ADAPTIVE_V2_ADAPTER = (
    "ictpolarreal.profile-acquisition-adapter.v7",
    "ictpolarreal-frequency-consensus-adaptive-v2",
)
_LEGACY_IDENTITY = (_LEGACY_CHECKPOINT_SCHEMA, _LEGACY_ADAPTER)
_EDGE_IDENTITY = (_EDGE_CHECKPOINT_SCHEMA, _EDGE_ADAPTER)
_STAGED_IMPULSE_IDENTITY = (
    _STAGED_IMPULSE_CHECKPOINT_SCHEMA,
    _STAGED_IMPULSE_ADAPTER,
)
_PROXIMAL_V11_IDENTITY = (
    _PROXIMAL_V11_CHECKPOINT_SCHEMA,
    _PROXIMAL_V11_ADAPTER,
)
_PROXIMAL_V12_IDENTITY = (
    _PROXIMAL_V12_CHECKPOINT_SCHEMA,
    _PROXIMAL_V12_ADAPTER,
)
_FREQUENCY_IDENTITY = (
    _FREQUENCY_CHECKPOINT_SCHEMA,
    _FREQUENCY_ADAPTER,
)
_FREQUENCY_ADAPTIVE_IDENTITY = (
    _FREQUENCY_CHECKPOINT_SCHEMA,
    _FREQUENCY_ADAPTIVE_ADAPTER,
)
_FREQUENCY_V2_IDENTITY = (
    _FREQUENCY_CHECKPOINT_SCHEMA,
    _FREQUENCY_V2_ADAPTER,
)
_FREQUENCY_ADAPTIVE_V2_IDENTITY = (
    _FREQUENCY_CHECKPOINT_SCHEMA,
    _FREQUENCY_ADAPTIVE_V2_ADAPTER,
)
_SUPPORTED_REGULARIZERS_BY_IDENTITY = {
    _LEGACY_IDENTITY: frozenset({"l1"}),
    _EDGE_IDENTITY: frozenset({"l1", "edge-charbonnier"}),
    _STAGED_IMPULSE_IDENTITY: frozenset(
        {"l1", "edge-charbonnier", "impulse-median"}
    ),
    _PROXIMAL_V11_IDENTITY: frozenset(
        {"l1", "edge-charbonnier", "impulse-median"}
    ),
    _PROXIMAL_V12_IDENTITY: frozenset(
        {"l1", "edge-charbonnier", "impulse-median"}
    ),
    _FREQUENCY_IDENTITY: frozenset(
        {"l1", "edge-charbonnier", "impulse-median", "frequency-consensus"}
    ),
    _FREQUENCY_ADAPTIVE_IDENTITY: frozenset(
        {"frequency-consensus-adaptive"}
    ),
    _FREQUENCY_V2_IDENTITY: frozenset(
        {"l1", "edge-charbonnier", "impulse-median", "frequency-consensus"}
    ),
    _FREQUENCY_ADAPTIVE_V2_IDENTITY: frozenset(
        {"frequency-consensus-adaptive"}
    ),
}
_ACTIVE_REGULARIZER_TRANSITIONS = {
    (_FREQUENCY_IDENTITY, _FREQUENCY_ADAPTIVE_IDENTITY): (
        "active-frequency-v1-to-adaptive-v1"
    ),
    (_FREQUENCY_V2_IDENTITY, _FREQUENCY_ADAPTIVE_V2_IDENTITY): (
        "active-frequency-v2-to-adaptive-v2"
    ),
}
_ZERO_WEIGHT_TRANSITIONS = {
    (_LEGACY_IDENTITY, _EDGE_IDENTITY): "zero-weight-v8-to-v9",
    (_LEGACY_IDENTITY, _STAGED_IMPULSE_IDENTITY): "zero-weight-v8-to-v10",
    (_EDGE_IDENTITY, _STAGED_IMPULSE_IDENTITY): "zero-weight-v9-to-v10",
    (_LEGACY_IDENTITY, _PROXIMAL_V11_IDENTITY): "zero-weight-v8-to-v11",
    (_EDGE_IDENTITY, _PROXIMAL_V11_IDENTITY): "zero-weight-v9-to-v11",
    (
        _STAGED_IMPULSE_IDENTITY,
        _PROXIMAL_V11_IDENTITY,
    ): "zero-weight-v10-to-v11",
    (_LEGACY_IDENTITY, _PROXIMAL_V12_IDENTITY): "zero-weight-v8-to-v12",
    (_EDGE_IDENTITY, _PROXIMAL_V12_IDENTITY): "zero-weight-v9-to-v12",
    (
        _STAGED_IMPULSE_IDENTITY,
        _PROXIMAL_V12_IDENTITY,
    ): "zero-weight-v10-to-v12",
    (_PROXIMAL_V11_IDENTITY, _PROXIMAL_V12_IDENTITY): "zero-weight-v11-to-v12",
    (_LEGACY_IDENTITY, _FREQUENCY_IDENTITY): "zero-weight-v8-to-v13",
    (_EDGE_IDENTITY, _FREQUENCY_IDENTITY): "zero-weight-v9-to-v13",
    (
        _STAGED_IMPULSE_IDENTITY,
        _FREQUENCY_IDENTITY,
    ): "zero-weight-v10-to-v13",
    (_PROXIMAL_V11_IDENTITY, _FREQUENCY_IDENTITY): "zero-weight-v11-to-v13",
    (_PROXIMAL_V12_IDENTITY, _FREQUENCY_IDENTITY): "zero-weight-v12-to-v13",
}


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
        help="Optional exact fit mask used for interior map spatial statistics.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "Dataset root used to reconstruct the fit mask rule recorded by the "
            "acquisition. Preferred over --mask."
        ),
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=None,
        help=(
            "Optional shared fit profiles to compare, for example --profiles olat. "
            "Without this option both manifests must contain the same profile list."
        ),
    )
    args = parser.parse_args()

    compose_regularization_comparison(
        args.baseline,
        args.regularized,
        args.output,
        mask_path=args.mask,
        data_root=args.data_root,
        profiles=args.profiles,
    )


def compose_regularization_comparison(
    baseline_camera: str | Path,
    regularized_camera: str | Path,
    output_dir: str | Path,
    *,
    mask_path: str | Path | None = None,
    data_root: str | Path | None = None,
    profiles: Sequence[str] | None = None,
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
    profiles = _comparison_profiles(
        baseline_manifest,
        regularized_manifest,
        requested=profiles,
    )

    acquisitions = {
        "baseline": _load_acquisitions(baseline_camera, profiles),
        "regularized": _load_acquisitions(regularized_camera, profiles),
    }
    contract = _validate_comparison_contract(acquisitions, profiles)
    guide_diagnostic = _guide_diagnostic_configuration(contract)
    baseline_weight = float(
        acquisitions["baseline"][profiles[0]]["regularization"]["weight"]
    )
    regularized_weight = float(
        acquisitions["regularized"][profiles[0]]["regularization"]["weight"]
    )
    labels = {
        "baseline": (
            "Baseline · regularizer inactive (λ=0)"
            if baseline_weight == 0.0
            else (
                "Baseline · "
                f"{_display_regularizer(contract['baseline_regularization_kind'])} "
                f"λ={baseline_weight:g}"
            )
        ),
        "regularized": (
            f"Regularized · {_display_regularizer(contract['regularization_kind'])} "
            f"λ={regularized_weight:g}"
        ),
    }

    sample_map = baseline_camera / "material" / profiles[0] / "maps" / "roughness.png"
    with Image.open(sample_map) as image:
        sample_size = image.size
    if mask_path is not None and data_root is not None:
        raise ValueError("pass either --data-root or --mask, not both")
    if data_root is not None:
        data_root_path = Path(data_root).expanduser().resolve()
        fit_mask_rules = {
            rule
            for variant in ("baseline", "regularized")
            for profile in profiles
            for rule in (
                acquisitions[variant][profile]
                .get("surface_validity", {})
                .get("fit_mask_rule"),
                acquisitions[variant][profile]
                .get("adapter", {})
                .get("fit_mask_rule"),
            )
            if isinstance(rule, str) and rule
        }
        if len(fit_mask_rules) > 1:
            raise ValueError(
                "comparison acquisitions use different fitting-mask rules: "
                f"{sorted(fit_mask_rules)}"
            )
        fit_mask_rule = next(iter(fit_mask_rules), None)
        mask = _fit_mask_from_data_root(
            data_root_path,
            baseline_camera.parent.name,
            baseline_camera.name,
            sample_size,
            fit_mask_rule=fit_mask_rule,
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

    map_metrics, guide_regions = _measure_material_maps(
        baseline_camera,
        regularized_camera,
        profiles,
        interior_mask,
        guide_diagnostic,
    )
    impulse_cleanup = _measure_impulse_cleanup(
        baseline_camera,
        regularized_camera,
        profiles,
        mask,
        acquisitions,
        contract,
    )
    frequency_cleanup = _measure_frequency_cleanup(
        baseline_camera,
        regularized_camera,
        profiles,
        mask,
        acquisitions,
        contract,
    )
    if contract.get("active_frequency_upgrade"):
        if frequency_cleanup is None or frequency_cleanup.get("available") is not True:
            raise ValueError("active frequency upgrade is missing cleanup diagnostics")
        frequency_cleanup["adaptive_cleanup_evidence"] = (
            _measure_adaptive_cleanup_evidence(
                baseline_camera,
                regularized_camera,
                profiles,
                interior_mask,
                guide_diagnostic,
                map_metrics,
                contract,
            )
        )
    evaluation_metrics = _collect_evaluation_metrics(acquisitions, profiles)
    case_png_metrics = _collect_case_png_metrics(
        baseline_camera,
        regularized_camera,
        acquisitions,
        profiles,
        mask,
    )
    summary = _build_summary(
        baseline_camera,
        regularized_camera,
        profiles,
        labels,
        contract,
        map_metrics,
        evaluation_metrics,
        mask_source,
        guide_regions,
        guide_diagnostic,
        impulse_cleanup,
        frequency_cleanup,
        case_png_metrics,
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
            _write_evaluation_comparison(
                baseline_camera,
                regularized_camera,
                profiles,
                lighting,
                evaluation_metrics,
                case_png_metrics,
                path,
            )
            evaluation_paths.append(path)

        _write_metrics_csv(
            stage / "metrics.csv",
            map_metrics,
            evaluation_metrics,
            impulse_cleanup,
            frequency_cleanup,
            case_png_metrics,
        )
        if frequency_cleanup is not None and frequency_cleanup.get("available") is True:
            _write_frequency_detail(
                baseline_camera,
                regularized_camera,
                acquisitions,
                frequency_cleanup,
                stage / "material" / "frequency_hotspot_1to1.png",
            )
            _write_frequency_full_maps(
                baseline_camera,
                regularized_camera,
                acquisitions,
                frequency_cleanup,
                stage / "material" / "frequency_fullmaps_1to1.png",
            )
            adaptive_evidence = frequency_cleanup.get(
                "adaptive_cleanup_evidence"
            )
            if contract.get("active_frequency_upgrade"):
                if not isinstance(adaptive_evidence, dict):
                    raise ValueError("adaptive cleanup evidence metadata is missing")
                _write_adaptive_cleanup_evidence(
                    baseline_camera,
                    regularized_camera,
                    profiles,
                    interior_mask,
                    guide_diagnostic,
                    adaptive_evidence,
                    summary,
                    stage / "material" / "adaptive_cleanup_evidence.png",
                )
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
        _validate_report(
            stage,
            profiles,
            require_frequency_detail=(
                frequency_cleanup is not None
                and frequency_cleanup.get("available") is True
            ),
            require_adaptive_cleanup_evidence=bool(
                contract.get("active_frequency_upgrade")
            ),
        )

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


def _comparison_profiles(
    baseline_manifest: dict[str, Any],
    regularized_manifest: dict[str, Any],
    *,
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    baseline = tuple(str(profile) for profile in baseline_manifest["profiles"])
    regularized = tuple(str(profile) for profile in regularized_manifest["profiles"])
    if requested is None:
        if regularized != baseline:
            raise ValueError(
                "baseline and regularized runs do not contain the same profiles; "
                "pass --profiles with the shared profiles to compare"
            )
        return baseline

    selected = tuple(str(profile).strip() for profile in requested)
    if not selected or any(not profile for profile in selected):
        raise ValueError("comparison profiles must be non-empty names")
    if len(set(selected)) != len(selected):
        raise ValueError(f"comparison profiles contain duplicates: {selected}")
    missing_baseline = [profile for profile in selected if profile not in baseline]
    missing_regularized = [
        profile for profile in selected if profile not in regularized
    ]
    if missing_baseline or missing_regularized:
        raise ValueError(
            "requested comparison profiles are not present in both manifests: "
            f"baseline_missing={missing_baseline}, "
            f"regularized_missing={missing_regularized}"
        )
    return selected


def _load_acquisitions(
    camera_dir: Path, profiles: Sequence[str]
) -> dict[str, dict[str, Any]]:
    acquisitions = {}
    for profile in profiles:
        path = camera_dir / "material" / profile / "acquisition.json"
        acquisition = json.loads(path.read_text(encoding="utf-8"))
        _validate_acquisition_profile_binding(
            acquisition,
            requested_profile=profile,
            source=path,
        )
        acquisitions[profile] = acquisition
    return acquisitions


def _validate_acquisition_profile_binding(
    acquisition: dict[str, Any],
    *,
    requested_profile: str,
    source: Path | str,
) -> None:
    signature = acquisition.get("checkpoint_signature")
    evaluation = acquisition.get("evaluation")
    bindings = {
        "lighting_profile": acquisition.get("lighting_profile"),
        "checkpoint_signature.profile": (
            signature.get("profile") if isinstance(signature, dict) else None
        ),
        "evaluation.profile": (
            evaluation.get("profile") if isinstance(evaluation, dict) else None
        ),
    }
    mismatched = {
        name: value
        for name, value in bindings.items()
        if value != requested_profile
    }
    if mismatched:
        raise ValueError(
            "acquisition profile binding differs from its requested material "
            f"directory: source={source}, requested={requested_profile!r}, "
            f"bindings={mismatched}"
        )


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

    (
        baseline_kind,
        baseline_parameters,
        baseline_weight,
        baseline_settings,
        baseline_stage_plan,
    ) = configurations["baseline"]
    (
        regularized_kind,
        regularized_parameters,
        regularized_weight,
        regularized_settings,
        regularized_stage_plan,
    ) = configurations["regularized"]
    if baseline_parameters != regularized_parameters:
        raise ValueError("baseline and regularized runs regularize different maps")
    active_frequency_upgrade = (
        baseline_kind == "frequency-consensus"
        and regularized_kind == "frequency-consensus-adaptive"
        and baseline_weight > 0.0
        and regularized_weight > 0.0
    )
    if active_frequency_upgrade:
        if baseline_weight != regularized_weight:
            raise ValueError(
                "active frequency-consensus comparison requires equal weights"
            )
        if baseline_stage_plan != regularized_stage_plan:
            raise ValueError(
                "active frequency-consensus comparison requires an equal stage plan"
            )
    elif baseline_weight == regularized_weight:
        raise ValueError(
            "baseline and regularized runs use the same regularization weight"
        )
    baseline_regularizer_inactive = baseline_weight == 0.0
    if not baseline_regularizer_inactive and not active_frequency_upgrade and (
        baseline_kind,
        baseline_settings,
        baseline_stage_plan,
    ) != (regularized_kind, regularized_settings, regularized_stage_plan):
        raise ValueError(
            "nonzero baseline and regularized runs use different regularizers"
        )
    regularizer_configuration_variable = (
        baseline_kind,
        baseline_settings,
        baseline_stage_plan,
    ) != (regularized_kind, regularized_settings, regularized_stage_plan)

    transition_modes = set()
    for profile in profiles:
        baseline = acquisitions["baseline"][profile]
        regularized = acquisitions["regularized"][profile]
        transition = _comparison_signature_transition(
            baseline,
            regularized,
            allow_legacy_transition=baseline_regularizer_inactive,
            allow_active_frequency_transition=active_frequency_upgrade,
        )
        transition_modes.add(transition)
        if active_frequency_upgrade and transition not in {
            "active-frequency-v1-to-adaptive-v1",
            "active-frequency-v2-to-adaptive-v2",
        }:
            raise ValueError(
                "active frequency-consensus comparison requires the known "
                "matched fixed-to-adaptive adapter transition"
            )
        if baseline_regularizer_inactive and regularizer_configuration_variable:
            _validate_variable_regularizer_transition(
                baseline,
                regularized,
                transition=transition,
                baseline_kind=baseline_kind,
                regularized_kind=regularized_kind,
            )
        baseline_signature = _normalized_comparison_signature(
            baseline,
            variable_regularizer=baseline_regularizer_inactive,
            active_frequency_upgrade=active_frequency_upgrade,
            transition=transition,
        )
        regularized_signature = _normalized_comparison_signature(
            regularized,
            variable_regularizer=baseline_regularizer_inactive,
            active_frequency_upgrade=active_frequency_upgrade,
            transition=transition,
        )
        if baseline_signature != regularized_signature:
            raise ValueError(
                f"{profile} comparison is not controlled; checkpoint signatures "
                "differ beyond the active regularizer configuration"
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
    if len(transition_modes) != 1:
        raise ValueError("comparison adapter transition differs across profiles")
    transition_mode = next(iter(transition_modes))
    active_comparison_modes = {
        "active-frequency-v1-to-adaptive-v1": (
            "frequency-consensus-v1-to-adaptive-v1"
        ),
        "active-frequency-v2-to-adaptive-v2": (
            "frequency-consensus-v2-to-adaptive-v2"
        ),
    }
    return {
        "controlled": True,
        "matched_fields": [
            "all substantive checkpoint-signature fields",
            "regularization.parameters",
            "base_color_source",
        ],
        "only_intended_difference": (
            "active scalar-map regularizer kind/settings/stage plan/weight; the "
            "baseline regularizer has zero weight"
            if baseline_regularizer_inactive
            else (
                "post-fit target-selection policy at equal regularization weight; "
                "the full data fit and stage plan are matched"
                if active_frequency_upgrade
                else "scalar-map regularization weight λ"
            )
        ),
        "base_color_source": "dataset_albedo",
        "baseline_regularization_kind": baseline_kind,
        "regularization_kind": regularized_kind,
        "regularized_parameters": list(baseline_parameters),
        "baseline_tv_weight": baseline_weight,
        "regularized_tv_weight": regularized_weight,
        "baseline_regularizer_inactive": baseline_regularizer_inactive,
        "active_frequency_upgrade": active_frequency_upgrade,
        "comparison_mode": (
            active_comparison_modes[transition_mode]
            if active_frequency_upgrade
            else "zero-weight-to-active"
            if baseline_regularizer_inactive
            else "regularization-weight"
        ),
        "regularizer_configuration_variable": regularizer_configuration_variable,
        "regularized_settings": json.loads(regularized_settings),
        "regularized_stage_plan": json.loads(regularized_stage_plan),
        "signature_compatibility": sorted(transition_modes),
    }


def _regularization_configuration(
    acquisition: dict[str, Any],
) -> tuple[str, tuple[str, ...], float, str, str]:
    regularization = acquisition.get("regularization")
    if not isinstance(regularization, dict):
        raise ValueError("acquisition is missing regularization provenance")
    kind = regularization.get("kind")
    parameters = regularization.get("parameters")
    settings = regularization.get("settings", {})
    try:
        weight = float(regularization["weight"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("acquisition has an invalid regularization weight") from exc
    if not isinstance(kind, str) or not kind:
        raise ValueError("acquisition has an invalid regularization kind")
    if not isinstance(parameters, list) or not parameters or not all(
        isinstance(name, str) and name for name in parameters
    ):
        raise ValueError("acquisition has invalid regularized parameters")
    if not isinstance(settings, dict):
        raise ValueError("acquisition has invalid regularization settings")
    stage_plan = regularization.get("stage_plan")
    if stage_plan is not None and not isinstance(stage_plan, dict):
        raise ValueError("acquisition has an invalid regularization stage plan")
    if not np.isfinite(weight) or weight < 0:
        raise ValueError(
            "acquisition has a non-finite or negative regularization weight"
        )

    signature_regularization = acquisition.get("checkpoint_signature", {}).get(
        "regularization"
    )
    expected = {
        "kind": kind,
        "parameters": parameters,
        "weight": weight,
    }
    if "settings" in regularization:
        expected["settings"] = settings
    if "stage_plan" in regularization:
        expected["stage_plan"] = regularization["stage_plan"]
    if signature_regularization != expected:
        raise ValueError(
            "acquisition and checkpoint signature regularization provenance differ"
        )
    return (
        _canonical_regularization_kind(kind),
        tuple(parameters),
        weight,
        json.dumps(settings, sort_keys=True, separators=(",", ":")),
        json.dumps(
            stage_plan,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _canonical_regularization_kind(kind: str) -> str:
    return {
        "masked_l1_total_variation": "l1",
        "l1": "l1",
        "edge-charbonnier": "edge-charbonnier",
        "impulse-median": "impulse-median",
        "frequency-consensus": "frequency-consensus",
        "frequency-consensus-adaptive": "frequency-consensus-adaptive",
    }.get(kind, kind)


def _comparison_signature_transition(
    baseline: dict[str, Any],
    regularized: dict[str, Any],
    *,
    allow_legacy_transition: bool,
    allow_active_frequency_transition: bool = False,
) -> str:
    baseline_signature = baseline.get("checkpoint_signature")
    regularized_signature = regularized.get("checkpoint_signature")
    if not isinstance(baseline_signature, dict) or not isinstance(
        regularized_signature, dict
    ):
        raise ValueError("acquisition is missing its checkpoint signature")

    baseline_identity = (
        baseline_signature.get("schema"),
        _adapter_identity(baseline_signature),
    )
    regularized_identity = (
        regularized_signature.get("schema"),
        _adapter_identity(regularized_signature),
    )
    if baseline_identity == regularized_identity:
        return "exact-schema"
    transition = _ZERO_WEIGHT_TRANSITIONS.get(
        (baseline_identity, regularized_identity)
    )
    if allow_legacy_transition and transition is not None:
        return transition
    transition = _ACTIVE_REGULARIZER_TRANSITIONS.get(
        (baseline_identity, regularized_identity)
    )
    if allow_active_frequency_transition and transition is not None:
        return transition
    raise ValueError(
        "comparison is not controlled; checkpoint signatures use an unsupported "
        f"schema/algorithm transition: {baseline_identity} -> {regularized_identity}"
    )


def _adapter_identity(signature: dict[str, Any]) -> tuple[Any, Any]:
    adapter = signature.get("adapter")
    if not isinstance(adapter, dict):
        return (None, None)
    return (adapter.get("schema"), adapter.get("algorithm_version"))


def _validate_variable_regularizer_transition(
    baseline: dict[str, Any],
    regularized: dict[str, Any],
    *,
    transition: str,
    baseline_kind: str,
    regularized_kind: str,
) -> None:
    """Allow kind/settings changes only under adapters that support them.

    Equal but absent schema metadata is sufficient for a weight-only comparison,
    but it is not evidence that two different regularizer implementations are a
    controlled change.  The latter needs one of the known numerical adapters.
    """
    baseline_identity = (
        baseline["checkpoint_signature"].get("schema"),
        _adapter_identity(baseline["checkpoint_signature"]),
    )
    regularized_identity = (
        regularized["checkpoint_signature"].get("schema"),
        _adapter_identity(regularized["checkpoint_signature"]),
    )
    if transition == "exact-schema":
        supported = _SUPPORTED_REGULARIZERS_BY_IDENTITY.get(baseline_identity)
        if (
            supported is not None
            and regularized_identity == baseline_identity
            and baseline_kind in supported
            and regularized_kind in supported
        ):
            return
    else:
        expected_transition = _ZERO_WEIGHT_TRANSITIONS.get(
            (baseline_identity, regularized_identity)
        )
        baseline_supported = _SUPPORTED_REGULARIZERS_BY_IDENTITY.get(
            baseline_identity, frozenset()
        )
        regularized_supported = _SUPPORTED_REGULARIZERS_BY_IDENTITY.get(
            regularized_identity, frozenset()
        )
        target_kind_supported = regularized_kind in regularized_supported
        if regularized_identity in {
            _STAGED_IMPULSE_IDENTITY,
            _PROXIMAL_V11_IDENTITY,
            _PROXIMAL_V12_IDENTITY,
        }:
            target_kind_supported = regularized_kind == "impulse-median"
        elif regularized_identity in {
            _FREQUENCY_IDENTITY,
            _FREQUENCY_V2_IDENTITY,
        }:
            target_kind_supported = regularized_kind == "frequency-consensus"
        if (
            transition == expected_transition
            and baseline_kind in baseline_supported
            and target_kind_supported
        ):
            return
    raise ValueError(
        "regularizer kind/settings/stage plan may vary from a zero-weight "
        "baseline only under a known supported adapter transition"
    )


def _normalized_comparison_signature(
    acquisition: dict[str, Any],
    *,
    variable_regularizer: bool = False,
    active_frequency_upgrade: bool = False,
    transition: str = "exact-schema",
) -> dict[str, Any]:
    signature = acquisition.get("checkpoint_signature")
    if not isinstance(signature, dict):
        raise ValueError("acquisition is missing its checkpoint signature")
    normalized = json.loads(json.dumps(signature))
    regularization = normalized.get("regularization")
    if not isinstance(regularization, dict) or "weight" not in regularization:
        raise ValueError("checkpoint signature is missing its regularization weight")
    regularization["weight"] = "<comparison-variable>"
    if "kind" in regularization:
        regularization["kind"] = _canonical_regularization_kind(
            regularization["kind"]
        )
    if variable_regularizer:
        regularization["kind"] = "<active-regularizer-variable>"
        regularization.pop("settings", None)
        regularization.pop("stage_plan", None)
    elif active_frequency_upgrade:
        regularization["kind"] = "<frequency-consensus-policy-variable>"
        regularization.pop("settings", None)
    if transition != "exact-schema":
        normalized["schema"] = "<compatible-regularizer-schema>"
        adapter = normalized.get("adapter")
        if not isinstance(adapter, dict):
            raise ValueError("checkpoint signature is missing adapter provenance")
        adapter["schema"] = "<compatible-regularizer-adapter>"
        adapter["algorithm_version"] = "<compatible-regularizer-algorithm>"
    return normalized


def _guide_diagnostic_configuration(contract: dict[str, Any]) -> dict[str, Any]:
    """Describe the guide used only to stratify report measurements.

    For the edge-aware regularizer, use and validate the recorded fitting
    constants.  Other regularizers still receive a fixed guide diagnostic, but
    the report explicitly records that it is independent of their objective.
    """
    kind = str(contract["regularization_kind"])
    settings = contract.get("regularized_settings", {})
    if not isinstance(settings, dict):
        raise ValueError("comparison contract has invalid regularizer settings")
    configuration = {
        "guide": "baseline exported baseColor + normal",
        "score": "mean_abs_albedo/albedo_sigma + (1-cosine_normal)/normal_sigma",
        "albedo_sigma": DEFAULT_GUIDE_ALBEDO_SIGMA,
        "normal_sigma": DEFAULT_GUIDE_NORMAL_SIGMA,
        "source": "fixed_report_diagnostic",
        "input_precision": "8-bit exported material maps",
        "matches_active_regularizer_configuration": False,
    }
    if kind != "edge-charbonnier":
        return configuration

    expected_semantics = {
        "guide": "normalized_albedo_and_normal",
        "albedo_difference": "mean_absolute_rgb",
        "normal_difference": "one_minus_cosine",
    }
    for name, expected in expected_semantics.items():
        value = settings.get(name)
        if value is not None and value != expected:
            raise ValueError(
                f"edge-aware guide diagnostic does not support {name}={value!r}; "
                f"expected {expected!r}"
            )
    has_albedo_sigma = "albedo_sigma" in settings
    has_normal_sigma = "normal_sigma" in settings
    if has_albedo_sigma != has_normal_sigma:
        raise ValueError(
            "edge-aware regularizer provenance must record both guide sigmas or neither"
        )
    if not has_albedo_sigma:
        configuration["source"] = "fallback_defaults_missing_candidate_sigmas"
        return configuration

    try:
        albedo_sigma = float(settings["albedo_sigma"])
        normal_sigma = float(settings["normal_sigma"])
    except (TypeError, ValueError) as exc:
        raise ValueError("edge-aware guide sigmas must be numeric") from exc
    if not math.isfinite(albedo_sigma) or albedo_sigma <= 0.0:
        raise ValueError("edge-aware albedo guide sigma must be finite and positive")
    if not math.isfinite(normal_sigma) or normal_sigma <= 0.0:
        raise ValueError("edge-aware normal guide sigma must be finite and positive")
    configuration.update(
        {
            "albedo_sigma": albedo_sigma,
            "normal_sigma": normal_sigma,
            "source": "active_regularizer_settings",
            "matches_active_regularizer_configuration": True,
        }
    )
    return configuration


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
    *,
    fit_mask_rule: str | None = None,
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
    capture = capture_mask[..., 0] > 0.5
    if fit_mask_rule == _CLEAN_CAPTURE_FIT_MASK_RULE:
        return capture
    if fit_mask_rule not in {None, _LEGACY_FRONT_FACING_FIT_MASK_RULE}:
        raise ValueError(f"unsupported fitting-mask rule: {fit_mask_rule!r}")
    view = load_end2end_view_directions(data_root, sample, expected_shape)
    normal = _normalize_vectors(normal)
    view = _normalize_vectors(view)
    front_facing = np.sum(normal * view, axis=-1) > 1e-4
    return capture & front_facing


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


def _read_rgb_map(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _guide_region_masks(
    base_color: np.ndarray,
    normal_encoded: np.ndarray,
    mask: np.ndarray,
    *,
    albedo_sigma: float = DEFAULT_GUIDE_ALBEDO_SIGMA,
    normal_sigma: float = DEFAULT_GUIDE_NORMAL_SIGMA,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if base_color.shape != (*mask.shape, 3) or normal_encoded.shape != (
        *mask.shape,
        3,
    ):
        raise ValueError(
            "guide maps do not match the comparison mask: "
            f"baseColor={base_color.shape}, normal={normal_encoded.shape}, "
            f"mask={mask.shape}"
        )
    if not math.isfinite(albedo_sigma) or albedo_sigma <= 0.0:
        raise ValueError("guide diagnostic albedo sigma must be finite and positive")
    if not math.isfinite(normal_sigma) or normal_sigma <= 0.0:
        raise ValueError("guide diagnostic normal sigma must be finite and positive")
    normal = _normalize_vectors(normal_encoded * 2.0 - 1.0)
    horizontal_valid = mask[:, 1:] & mask[:, :-1]
    vertical_valid = mask[1:, :] & mask[:-1, :]
    horizontal_albedo = np.mean(
        np.abs(base_color[:, 1:] - base_color[:, :-1]), axis=-1
    )
    vertical_albedo = np.mean(
        np.abs(base_color[1:] - base_color[:-1]), axis=-1
    )
    horizontal_normal = np.clip(
        1.0 - np.sum(normal[:, 1:] * normal[:, :-1], axis=-1), 0.0, 2.0
    )
    vertical_normal = np.clip(
        1.0 - np.sum(normal[1:] * normal[:-1], axis=-1), 0.0, 2.0
    )
    horizontal_score = (
        horizontal_albedo / albedo_sigma
        + horizontal_normal / normal_sigma
    )
    vertical_score = (
        vertical_albedo / albedo_sigma
        + vertical_normal / normal_sigma
    )
    valid_scores = np.concatenate(
        [horizontal_score[horizontal_valid], vertical_score[vertical_valid]]
    )
    if valid_scores.size == 0:
        raise ValueError("comparison mask has no valid guide-neighbor pairs")
    quiet_threshold = float(np.percentile(valid_scores, QUIET_GUIDE_PERCENTILE))
    edge_threshold = float(np.percentile(valid_scores, EDGE_GUIDE_PERCENTILE))
    minimum_edge_strength = max(edge_threshold, 1e-6)
    horizontal_quiet = horizontal_valid & (horizontal_score <= quiet_threshold)
    vertical_quiet = vertical_valid & (vertical_score <= quiet_threshold)
    horizontal_edge = horizontal_valid & (horizontal_score >= minimum_edge_strength)
    vertical_edge = vertical_valid & (vertical_score >= minimum_edge_strength)

    local_maximum = np.zeros(mask.shape, dtype=np.float32)
    has_pair = np.zeros(mask.shape, dtype=bool)
    horizontal_values = np.where(horizontal_valid, horizontal_score, 0.0)
    vertical_values = np.where(vertical_valid, vertical_score, 0.0)
    local_maximum[:, :-1] = np.maximum(local_maximum[:, :-1], horizontal_values)
    local_maximum[:, 1:] = np.maximum(local_maximum[:, 1:], horizontal_values)
    local_maximum[:-1, :] = np.maximum(local_maximum[:-1, :], vertical_values)
    local_maximum[1:, :] = np.maximum(local_maximum[1:, :], vertical_values)
    has_pair[:, :-1] |= horizontal_valid
    has_pair[:, 1:] |= horizontal_valid
    has_pair[:-1, :] |= vertical_valid
    has_pair[1:, :] |= vertical_valid
    quiet_pixels = mask & has_pair & (local_maximum <= quiet_threshold)
    if not np.any(quiet_pixels):
        raise ValueError("guide-aware comparison found no quiet-region pixels")

    masks = {
        "all_horizontal": horizontal_valid,
        "all_vertical": vertical_valid,
        "quiet_horizontal": horizontal_quiet,
        "quiet_vertical": vertical_quiet,
        "edge_horizontal": horizontal_edge,
        "edge_vertical": vertical_edge,
        "quiet_pixels": quiet_pixels,
    }
    metadata = {
        "guide": "baseline exported baseColor + normal",
        "score": "mean_abs_albedo/albedo_sigma + (1-cosine_normal)/normal_sigma",
        "albedo_sigma": float(albedo_sigma),
        "normal_sigma": float(normal_sigma),
        "quiet_percentile": QUIET_GUIDE_PERCENTILE,
        "edge_percentile": EDGE_GUIDE_PERCENTILE,
        "quiet_threshold": quiet_threshold,
        "edge_threshold": edge_threshold,
        "valid_pair_count": int(valid_scores.size),
        "quiet_pair_count": int(
            np.count_nonzero(horizontal_quiet) + np.count_nonzero(vertical_quiet)
        ),
        "edge_pair_count": int(
            np.count_nonzero(horizontal_edge) + np.count_nonzero(vertical_edge)
        ),
        "quiet_pixel_count": int(np.count_nonzero(quiet_pixels)),
    }
    return masks, metadata


def _signed_pair_gradients(
    values: np.ndarray,
    horizontal_mask: np.ndarray,
    vertical_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    horizontal = (values[:, 1:] - values[:, :-1])[horizontal_mask]
    vertical = (values[1:, :] - values[:-1, :])[vertical_mask]
    return horizontal.astype(np.float64), vertical.astype(np.float64)


def _pair_gradient_magnitude_mean(
    values: np.ndarray,
    horizontal_mask: np.ndarray,
    vertical_mask: np.ndarray,
) -> float:
    horizontal, vertical = _signed_pair_gradients(
        values, horizontal_mask, vertical_mask
    )
    axis_means = [
        float(np.mean(np.abs(axis)))
        for axis in (horizontal, vertical)
        if axis.size
    ]
    return float(np.mean(axis_means)) if axis_means else 0.0


def _signed_gradient_cosine(
    baseline: tuple[np.ndarray, np.ndarray],
    regularized: tuple[np.ndarray, np.ndarray],
) -> float | None:
    baseline_vector = np.concatenate(baseline)
    regularized_vector = np.concatenate(regularized)
    if baseline_vector.shape != regularized_vector.shape:
        raise ValueError("guide-edge gradient vectors do not have matching shapes")
    denominator = float(
        np.linalg.norm(baseline_vector) * np.linalg.norm(regularized_vector)
    )
    if denominator <= 1e-12:
        return None
    return float(
        np.clip(
            np.dot(baseline_vector, regularized_vector) / denominator,
            -1.0,
            1.0,
        )
    )


def _gaussian_filter_nearest(values: np.ndarray, sigma: float) -> np.ndarray:
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("Gaussian diagnostic sigma must be finite and positive")
    radius = int(4.0 * sigma + 0.5)
    coordinate = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (coordinate / sigma) ** 2)
    kernel /= kernel.sum()
    filtered = np.asarray(values, dtype=np.float64)
    for axis in (0, 1):
        padding = [(0, 0)] * filtered.ndim
        padding[axis] = (radius, radius)
        padded = np.pad(filtered, padding, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(
            padded,
            kernel.size,
            axis=axis,
        )
        filtered = np.tensordot(windows, kernel, axes=([-1], [0]))
    return filtered


def _band3_8(values: np.ndarray) -> np.ndarray:
    return _gaussian_filter_nearest(values, 0.8) - _gaussian_filter_nearest(
        values, 2.4
    )


def _guide_textured_band_mask(
    albedo: np.ndarray,
    normal: np.ndarray,
    interior: np.ndarray,
) -> np.ndarray:
    albedo_band = _band3_8(albedo)
    normal_band = _band3_8(normal)
    albedo_strength = np.sqrt(np.sum(albedo_band**2, axis=-1))
    normal_strength = np.sqrt(np.sum(normal_band**2, axis=-1))
    albedo_scale = max(float(np.percentile(albedo_strength[interior], 90.0)), 1e-6)
    normal_scale = max(float(np.percentile(normal_strength[interior], 90.0)), 1e-6)
    score = np.maximum(
        albedo_strength / albedo_scale,
        normal_strength / normal_scale,
    )
    selected = interior & (score >= FREQUENCY_GUIDE_BAND_THRESHOLD)
    if not np.any(selected):
        raise ValueError("frequency band diagnostic has no guide-textured pixels")
    return selected


def _map_spatial_metrics(
    values: np.ndarray,
    mask: np.ndarray,
    guide_masks: dict[str, np.ndarray],
) -> dict[str, float]:
    neighbor_variation = _pair_gradient_magnitude_mean(
        values,
        guide_masks["all_horizontal"],
        guide_masks["all_vertical"],
    )
    quiet_variation = _pair_gradient_magnitude_mean(
        values,
        guide_masks["quiet_horizontal"],
        guide_masks["quiet_vertical"],
    )
    edge_gradient_magnitude = _pair_gradient_magnitude_mean(
        values,
        guide_masks["edge_horizontal"],
        guide_masks["edge_vertical"],
    )

    residual_map = _own_median5_residual(values)
    residual = residual_map[mask]
    quiet_residual = residual_map[guide_masks["quiet_pixels"]]
    result = {
        "neighbor_variation": neighbor_variation,
        "median5_residual_mae": float(residual.mean()),
        "median5_residual_outlier_fraction_gt_0p05": float(
            np.mean(residual > 0.05)
        ),
        "quiet_region_neighbor_variation": quiet_variation,
        "quiet_region_median5_residual_mae": float(quiet_residual.mean()),
        "quiet_region_median5_residual_outlier_fraction_gt_0p05": float(
            np.mean(quiet_residual > 0.05)
        ),
        "quiet_region_pixels": int(quiet_residual.size),
        "quiet_region_median5_residual_outlier_count_gt_0p05": int(
            np.count_nonzero(quiet_residual > 0.05)
        ),
        "guide_edge_gradient_magnitude": edge_gradient_magnitude,
    }
    left, top, right, bottom = FREQUENCY_DETAIL_CROP_BOX
    if values.shape[1] >= right and values.shape[0] >= bottom:
        hotspot_quiet = guide_masks["quiet_pixels"][top:bottom, left:right]
        hotspot_residual = residual_map[top:bottom, left:right]
        hotspot_quiet_count = int(np.count_nonzero(hotspot_quiet))
        result["fixed_hotspot_quiet_pixels"] = hotspot_quiet_count
        result[
            "fixed_hotspot_median5_residual_outlier_fraction_gt_0p05"
        ] = (
            float(np.mean(hotspot_residual[hotspot_quiet] > 0.05))
            if hotspot_quiet_count
            else None
        )
    else:
        result["fixed_hotspot_quiet_pixels"] = 0
        result[
            "fixed_hotspot_median5_residual_outlier_fraction_gt_0p05"
        ] = None
    return result


def _own_median5_residual(values: np.ndarray) -> np.ndarray:
    """Return the absolute residual to each exported map's own 5x5 median."""
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("median5 residual input must be two-dimensional")
    padded = np.pad(values, 2, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (5, 5))
    median = np.median(windows, axis=(-2, -1))
    return np.abs(values - median)


def _measure_material_maps(
    baseline_camera: Path,
    regularized_camera: Path,
    profiles: Sequence[str],
    mask: np.ndarray,
    guide_diagnostic: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    output = {}
    guide_metadata = {}
    for profile in profiles:
        maps_dir = baseline_camera / "material" / profile / "maps"
        guide_albedo = _read_rgb_map(maps_dir / "baseColor.png")
        guide_normal = _read_rgb_map(maps_dir / "normal.png") * 2.0 - 1.0
        guide_masks, guide_metadata[profile] = _guide_region_masks(
            guide_albedo,
            (guide_normal + 1.0) * 0.5,
            mask,
            albedo_sigma=float(guide_diagnostic["albedo_sigma"]),
            normal_sigma=float(guide_diagnostic["normal_sigma"]),
        )
        guide_metadata[profile]["diagnostic_source"] = guide_diagnostic["source"]
        guide_metadata[profile]["input_precision"] = guide_diagnostic[
            "input_precision"
        ]
        guide_metadata[profile][
            "matches_active_regularizer_configuration"
        ] = guide_diagnostic["matches_active_regularizer_configuration"]
        band_mask = _guide_textured_band_mask(
            guide_albedo,
            guide_normal,
            mask,
        )
        guide_metadata[profile]["guide_textured_band3_8_pixels"] = int(
            np.count_nonzero(band_mask)
        )
        output[profile] = {}
        for map_name in SCALAR_MAPS:
            variants = {}
            variant_values = {}
            for variant, camera in (
                ("baseline", baseline_camera),
                ("regularized", regularized_camera),
            ):
                path = camera / "material" / profile / "maps" / f"{map_name}.png"
                variant_values[variant] = _read_scalar_map(path)
                variants[variant] = _map_spatial_metrics(
                    variant_values[variant], mask, guide_masks
                )
            baseline_edge = variants["baseline"]["guide_edge_gradient_magnitude"]
            regularized_edge = variants["regularized"][
                "guide_edge_gradient_magnitude"
            ]
            signed_cosine = _signed_gradient_cosine(
                _signed_pair_gradients(
                    variant_values["baseline"],
                    guide_masks["edge_horizontal"],
                    guide_masks["edge_vertical"],
                ),
                _signed_pair_gradients(
                    variant_values["regularized"],
                    guide_masks["edge_horizontal"],
                    guide_masks["edge_vertical"],
                ),
            )
            baseline_band = _band3_8(variant_values["baseline"])
            regularized_band = _band3_8(variant_values["regularized"])
            baseline_band_energy = float(
                np.sum(baseline_band[band_mask] ** 2, dtype=np.float64)
            )
            regularized_band_energy = float(
                np.sum(regularized_band[band_mask] ** 2, dtype=np.float64)
            )
            variants["baseline"]["guide_textured_band3_8_energy"] = (
                baseline_band_energy
            )
            variants["regularized"]["guide_textured_band3_8_energy"] = (
                regularized_band_energy
            )
            low_frequency_mae = float(
                np.mean(
                    np.abs(
                        _gaussian_filter_nearest(
                            variant_values["regularized"], 3.0
                        )
                        - _gaussian_filter_nearest(
                            variant_values["baseline"], 3.0
                        )
                    )[mask]
                )
            )
            variants["comparison"] = {
                "quiet_region_variation_relative_change_fraction": _relative_change_fraction(
                    variants["baseline"]["quiet_region_neighbor_variation"],
                    variants["regularized"]["quiet_region_neighbor_variation"],
                ),
                "quiet_region_median_residual_outlier_relative_change_fraction": _relative_change_fraction(
                    variants["baseline"][
                        "quiet_region_median5_residual_outlier_fraction_gt_0p05"
                    ],
                    variants["regularized"][
                        "quiet_region_median5_residual_outlier_fraction_gt_0p05"
                    ],
                ),
                "guide_edge_gradient_magnitude_ratio": _ratio(
                    baseline_edge,
                    regularized_edge,
                ),
                "guide_edge_signed_gradient_cosine": signed_cosine,
                "guide_textured_band3_8_amplitude_ratio": (
                    math.sqrt(regularized_band_energy / baseline_band_energy)
                    if baseline_band_energy > 1e-18
                    else None
                ),
                "gaussian3_low_frequency_mae": low_frequency_mae,
                "meaningful_guide_edge_gradient": bool(
                    baseline_edge >= MEANINGFUL_EDGE_GRADIENT_MIN
                ),
            }
            output[profile][map_name] = variants
    return output, guide_metadata


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _load_impulse_frozen_artifact(
    material_dir: Path,
    acquisition: dict[str, Any],
    expected_shape: tuple[int, int],
) -> dict[str, Any]:
    regularization = acquisition.get("regularization")
    if not isinstance(regularization, dict):
        raise ValueError("impulse candidate is missing regularization provenance")
    stage_plan = regularization.get("stage_plan")
    if not isinstance(stage_plan, dict) or stage_plan.get("enabled") is not True:
        raise ValueError("impulse candidate does not record an enabled stage plan")
    frozen_bundle = regularization.get("frozen_bundle")
    if not isinstance(frozen_bundle, dict) or frozen_bundle.get("schema") != (
        "ictpolarreal.impulse-median-bundle.v1"
    ):
        raise ValueError("impulse candidate is missing frozen-bundle provenance")
    artifact = frozen_bundle.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("impulse candidate is missing frozen-artifact provenance")
    if artifact.get("schema") != IMPULSE_FROZEN_ARTIFACT_SCHEMA:
        raise ValueError("impulse frozen artifact has an unsupported schema")
    if artifact.get("format") != "numpy_npz_compressed":
        raise ValueError("impulse frozen artifact has an unsupported format")
    relative_path = artifact.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("impulse frozen artifact has an invalid path")
    relative = Path(relative_path)
    if relative.is_absolute() or len(relative.parts) != 1:
        raise ValueError("impulse frozen artifact path must be a material filename")
    artifact_path = (material_dir / relative).resolve()
    if artifact_path.parent != material_dir.resolve():
        raise ValueError("impulse frozen artifact path escapes its material folder")
    if not artifact_path.is_file():
        raise FileNotFoundError(f"impulse frozen artifact is missing: {artifact_path}")
    expected_bytes = artifact.get("bytes")
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes <= 0
        or artifact_path.stat().st_size != expected_bytes
    ):
        raise ValueError("impulse frozen artifact byte size does not match provenance")
    expected_file_hash = artifact.get("sha256")
    actual_file_hash = _file_sha256(artifact_path)
    if (
        not isinstance(expected_file_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_file_hash)
        or actual_file_hash != expected_file_hash
    ):
        raise ValueError("impulse frozen artifact SHA-256 does not match provenance")

    expected_keys = {"metadata"}
    for map_name in SCALAR_MAPS:
        expected_keys.update(
            {f"{map_name}__target", f"{map_name}__mask"}
        )
    try:
        with np.load(artifact_path, allow_pickle=False) as payload:
            if set(payload.files) != expected_keys:
                raise ValueError(
                    "impulse frozen artifact contains an unexpected array set"
                )
            metadata_array = payload["metadata"]
            if metadata_array.shape != ():
                raise ValueError("impulse frozen artifact metadata must be scalar")
            try:
                metadata = json.loads(str(metadata_array.item()))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    "impulse frozen artifact metadata is invalid"
                ) from exc
            maps = {
                map_name: {
                    "target": np.array(payload[f"{map_name}__target"], copy=True),
                    "mask": np.array(payload[f"{map_name}__mask"], copy=True),
                }
                for map_name in SCALAR_MAPS
            }
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("impulse frozen"):
            raise
        raise ValueError(f"could not read impulse frozen artifact: {exc}") from exc

    created_after_step = frozen_bundle.get("created_after_step")
    metadata_maps = metadata.get("maps") if isinstance(metadata, dict) else None
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema") != IMPULSE_FROZEN_ARTIFACT_SCHEMA
        or metadata.get("created_after_step") != created_after_step
        or not isinstance(metadata_maps, list)
        or not all(isinstance(name, str) for name in metadata_maps)
        or len(metadata_maps) != len(set(metadata_maps))
        or set(metadata_maps) != set(SCALAR_MAPS)
    ):
        raise ValueError(
            "impulse frozen artifact metadata does not match acquisition provenance"
        )
    if stage_plan.get("detector_after_data_step") != created_after_step:
        raise ValueError(
            "impulse frozen artifact boundary does not match the stage plan"
        )
    map_provenance = frozen_bundle.get("maps")
    if not isinstance(map_provenance, dict) or set(map_provenance) != set(
        SCALAR_MAPS
    ):
        raise ValueError("impulse frozen bundle has invalid per-map provenance")

    total_flagged = 0
    for map_name, arrays in maps.items():
        target = arrays["target"]
        flagged = arrays["mask"]
        if target.shape != expected_shape or target.dtype != np.float32:
            raise ValueError(
                f"impulse frozen target {map_name} must be float32 {expected_shape}"
            )
        if flagged.shape != expected_shape or flagged.dtype != np.bool_:
            raise ValueError(
                f"impulse frozen mask {map_name} must be bool {expected_shape}"
            )
        if not np.all(np.isfinite(target)) or np.any((target < 0.0) | (target > 1.0)):
            raise ValueError(f"impulse frozen target {map_name} is outside [0,1]")
        provenance = map_provenance[map_name]
        if not isinstance(provenance, dict):
            raise ValueError(f"impulse frozen map provenance is invalid: {map_name}")
        flagged_count = int(np.count_nonzero(flagged))
        if (
            provenance.get("flagged_centers") != flagged_count
            or provenance.get("target_sha256") != _array_sha256(target)
            or provenance.get("mask_sha256") != _array_sha256(flagged)
        ):
            raise ValueError(
                f"impulse frozen map {map_name} does not match its hashes/count"
            )
        total_flagged += flagged_count
    if frozen_bundle.get("total_flagged_centers") != total_flagged:
        raise ValueError("impulse frozen total flagged count does not match artifact")
    return {
        "path": artifact_path,
        "sha256": actual_file_hash,
        "bytes": expected_bytes,
        "created_after_step": int(created_after_step),
        "maps": maps,
        "total_flagged_centers": total_flagged,
    }


def _load_frequency_frozen_artifact(
    material_dir: Path,
    acquisition: dict[str, Any],
    expected_shape: tuple[int, int],
) -> dict[str, Any]:
    regularization = acquisition.get("regularization")
    if not isinstance(regularization, dict):
        raise ValueError("frequency candidate is missing regularization provenance")
    stage_plan = regularization.get("stage_plan")
    if not isinstance(stage_plan, dict) or stage_plan.get("enabled") is not True:
        raise ValueError("frequency candidate does not record an enabled stage plan")
    frozen_bundle = regularization.get("frozen_bundle")
    expected_bundle_schema = (
        FREQUENCY_ADAPTIVE_BUNDLE_SCHEMA
        if regularization.get("kind") == "frequency-consensus-adaptive"
        else FREQUENCY_BUNDLE_SCHEMA
    )
    if not isinstance(frozen_bundle, dict) or frozen_bundle.get("schema") != (
        expected_bundle_schema
    ):
        raise ValueError("frequency candidate is missing frozen-bundle provenance")
    if expected_bundle_schema == FREQUENCY_ADAPTIVE_BUNDLE_SCHEMA:
        return _load_frequency_adaptive_frozen_artifact(
            material_dir,
            regularization,
            frozen_bundle,
            expected_shape,
        )
    artifact = frozen_bundle.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("frequency candidate is missing frozen-artifact provenance")
    if artifact.get("schema") != FREQUENCY_FROZEN_ARTIFACT_SCHEMA:
        raise ValueError("frequency frozen artifact has an unsupported schema")
    if artifact.get("format") != "numpy_npz_compressed":
        raise ValueError("frequency frozen artifact has an unsupported format")
    relative_path = artifact.get("path")
    if relative_path != FREQUENCY_FROZEN_ARTIFACT_NAME:
        raise ValueError(
            "frequency frozen artifact path must be frequency_consensus_frozen.npz"
        )
    relative = Path(relative_path)
    if relative.is_absolute() or len(relative.parts) != 1:
        raise ValueError("frequency frozen artifact path must be a material filename")
    artifact_path = (material_dir / relative).resolve()
    if artifact_path.parent != material_dir.resolve():
        raise ValueError("frequency frozen artifact path escapes its material folder")
    if not artifact_path.is_file():
        raise FileNotFoundError(f"frequency frozen artifact is missing: {artifact_path}")
    expected_bytes = artifact.get("bytes")
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes <= 0
        or artifact_path.stat().st_size != expected_bytes
    ):
        raise ValueError("frequency frozen artifact byte size does not match provenance")
    expected_file_hash = artifact.get("sha256")
    actual_file_hash = _file_sha256(artifact_path)
    if (
        not isinstance(expected_file_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_file_hash)
        or actual_file_hash != expected_file_hash
    ):
        raise ValueError("frequency frozen artifact SHA-256 does not match provenance")

    root_names = (
        "fit_foreground",
        "full_foreground5",
        "edge_protected_full_precision",
        "edge_protected_png_quantized",
        "edge_protected",
        "update_safe",
        "evidence_count",
    )
    float_names = ("source", "median3", "median7", "target")
    mask_names = ("mask", "consensus_mask")
    expected_keys = {"metadata", *root_names}
    for map_name in SCALAR_MAPS:
        expected_keys.update(
            f"{map_name}__{name}" for name in (*float_names, *mask_names)
        )
    try:
        with np.load(artifact_path, allow_pickle=False) as payload:
            if set(payload.files) != expected_keys:
                raise ValueError(
                    "frequency frozen artifact contains an unexpected array set"
                )
            metadata_array = payload["metadata"]
            if metadata_array.shape != ():
                raise ValueError("frequency frozen artifact metadata must be scalar")
            try:
                metadata = json.loads(str(metadata_array.item()))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    "frequency frozen artifact metadata is invalid"
                ) from exc
            roots = {
                name: np.array(payload[name], copy=True) for name in root_names
            }
            maps = {
                map_name: {
                    name: np.array(payload[f"{map_name}__{name}"], copy=True)
                    for name in (*float_names, *mask_names)
                }
                for map_name in SCALAR_MAPS
            }
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("frequency frozen"):
            raise
        raise ValueError(f"could not read frequency frozen artifact: {exc}") from exc

    created_after_step = frozen_bundle.get("created_after_step")
    strength = frozen_bundle.get("strength")
    median_chunk_rows = frozen_bundle.get("median_chunk_rows")
    edge_threshold_full = frozen_bundle.get("edge_threshold_full_precision")
    edge_threshold_png = frozen_bundle.get("edge_threshold_png_quantized")
    metadata_maps = metadata.get("maps") if isinstance(metadata, dict) else None
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema") != FREQUENCY_FROZEN_ARTIFACT_SCHEMA
        or metadata.get("created_after_step") != created_after_step
        or metadata.get("strength") != strength
        or metadata.get("median_chunk_rows") != median_chunk_rows
        or metadata.get("edge_threshold_full_precision") != edge_threshold_full
        or metadata.get("edge_threshold_png_quantized") != edge_threshold_png
        or not isinstance(metadata_maps, list)
        or not all(isinstance(name, str) for name in metadata_maps)
        or len(metadata_maps) != len(set(metadata_maps))
        or set(metadata_maps) != set(SCALAR_MAPS)
    ):
        raise ValueError(
            "frequency frozen artifact metadata does not match acquisition provenance"
        )
    if (
        not isinstance(created_after_step, int)
        or isinstance(created_after_step, bool)
        or stage_plan.get("detector_after_data_step") != created_after_step
    ):
        raise ValueError(
            "frequency frozen artifact boundary does not match the stage plan"
        )
    if (
        not isinstance(strength, (int, float))
        or isinstance(strength, bool)
        or not math.isfinite(float(strength))
        or not 0.0 < float(strength) <= 1.0
        or stage_plan.get("strength") != strength
    ):
        raise ValueError("frequency frozen artifact has invalid strength provenance")
    if (
        not isinstance(median_chunk_rows, dict)
        or set(median_chunk_rows) != {"3x3", "7x7"}
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            or value > expected_shape[0]
            for value in median_chunk_rows.values()
        )
    ):
        raise ValueError("frequency frozen artifact has an invalid median chunk plan")
    for name, value in (
        ("full-precision", edge_threshold_full),
        ("PNG-quantized", edge_threshold_png),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                f"frequency frozen artifact has an invalid {name} edge threshold"
            )

    for name in root_names[:-1]:
        if roots[name].shape != expected_shape or roots[name].dtype != np.bool_:
            raise ValueError(
                f"frequency frozen {name} must be bool {expected_shape}"
            )
    evidence_count = roots["evidence_count"]
    if evidence_count.shape != expected_shape or evidence_count.dtype != np.int64:
        raise ValueError(
            f"frequency frozen evidence_count must be int64 {expected_shape}"
        )
    if np.any((evidence_count < 0) | (evidence_count > len(SCALAR_MAPS))):
        raise ValueError("frequency frozen evidence_count is outside its map range")
    if not np.array_equal(
        roots["full_foreground5"], _erode_mask(roots["fit_foreground"], radius=2)
    ):
        raise ValueError("frequency frozen full foreground has invalid semantics")
    if not np.array_equal(
        roots["edge_protected"],
        roots["edge_protected_full_precision"]
        | roots["edge_protected_png_quantized"],
    ):
        raise ValueError("frequency frozen edge-protection union is invalid")
    if not np.array_equal(
        roots["update_safe"],
        roots["full_foreground5"] & ~roots["edge_protected"],
    ):
        raise ValueError("frequency frozen update-safe mask has invalid semantics")

    tensor_hashes = {
        "root": {name: _array_sha256(roots[name]) for name in root_names},
        "maps": {
            map_name: {
                name: _array_sha256(maps[map_name][name])
                for name in (*float_names, *mask_names)
            }
            for map_name in SCALAR_MAPS
        },
    }
    if (
        metadata.get("tensor_hashes") != tensor_hashes
        or frozen_bundle.get("tensor_hashes") != tensor_hashes
    ):
        raise ValueError("frequency frozen tensor hashes do not match artifact arrays")
    root_provenance = {
        "fit_foreground": "fit_foreground_sha256",
        "edge_protected_full_precision": (
            "edge_protected_full_precision_sha256"
        ),
        "edge_protected_png_quantized": (
            "edge_protected_png_quantized_sha256"
        ),
        "edge_protected": "edge_protected_sha256",
        "update_safe": "update_safe_sha256",
        "evidence_count": "evidence_count_sha256",
    }
    for name, provenance_name in root_provenance.items():
        if frozen_bundle.get(provenance_name) != tensor_hashes["root"][name]:
            raise ValueError(
                f"frequency frozen {name} hash does not match provenance"
            )

    map_provenance = frozen_bundle.get("maps")
    if not isinstance(map_provenance, dict) or set(map_provenance) != set(
        SCALAR_MAPS
    ):
        raise ValueError("frequency frozen bundle has invalid per-map provenance")
    total_updated = 0
    total_consensus = 0
    for map_name, arrays in maps.items():
        for name in float_names:
            values = arrays[name]
            if values.shape != expected_shape or values.dtype != np.float32:
                raise ValueError(
                    f"frequency frozen {name} {map_name} must be float32 "
                    f"{expected_shape}"
                )
            if not np.all(np.isfinite(values)) or np.any(
                (values < 0.0) | (values > 1.0)
            ):
                raise ValueError(
                    f"frequency frozen {name} {map_name} is outside [0,1]"
                )
        for name in mask_names:
            values = arrays[name]
            if values.shape != expected_shape or values.dtype != np.bool_:
                raise ValueError(
                    f"frequency frozen {name} {map_name} must be bool "
                    f"{expected_shape}"
                )
        update_mask = arrays["mask"]
        consensus_mask = arrays["consensus_mask"]
        if not np.array_equal(
            update_mask,
            roots["update_safe"] & (arrays["target"] != arrays["source"]),
        ):
            raise ValueError(
                f"frequency frozen update mask has invalid semantics: {map_name}"
            )
        if np.any(consensus_mask & ~roots["update_safe"]):
            raise ValueError(
                f"frequency frozen consensus mask is not update-safe: {map_name}"
            )
        provenance = map_provenance[map_name]
        if not isinstance(provenance, dict):
            raise ValueError(f"frequency frozen map provenance is invalid: {map_name}")
        updated_count = int(np.count_nonzero(update_mask))
        consensus_count = int(np.count_nonzero(consensus_mask))
        if (
            provenance.get("updated_entries") != updated_count
            or provenance.get("consensus_entries") != consensus_count
        ):
            raise ValueError(
                f"frequency frozen map {map_name} has stale entry counts"
            )
        for name in float_names:
            if provenance.get(f"{name}_sha256") != tensor_hashes["maps"][
                map_name
            ][name]:
                raise ValueError(
                    f"frequency frozen {name} {map_name} hash does not match"
                )
        for name in mask_names:
            if provenance.get(f"{name}_sha256") != tensor_hashes["maps"][
                map_name
            ][name]:
                raise ValueError(
                    f"frequency frozen {name} {map_name} hash does not match"
                )
        total_updated += updated_count
        total_consensus += consensus_count
    expected_evidence = np.sum(
        np.stack(
            [
                np.abs(maps[name]["source"] - maps[name]["median7"])
                > FREQUENCY_EVIDENCE_THRESHOLD
                for name in SCALAR_MAPS
            ],
            axis=0,
        ),
        axis=0,
        dtype=np.int64,
    )
    if not np.array_equal(evidence_count, expected_evidence):
        raise ValueError("frequency frozen evidence count has invalid semantics")
    for map_name, arrays in maps.items():
        expected_consensus = (
            (evidence_count >= FREQUENCY_MIN_EVIDENCE_MAPS)
            & (
                np.abs(arrays["source"] - arrays["median7"])
                > FREQUENCY_OWN_DEVIATION
            )
            & roots["update_safe"]
        )
        if not np.array_equal(arrays["consensus_mask"], expected_consensus):
            raise ValueError(
                f"frequency frozen consensus mask has invalid semantics: {map_name}"
            )
        base_target = arrays["source"] + FREQUENCY_BASE_BLEND * (
            arrays["median3"] - arrays["source"]
        )
        consensus_target = (
            FREQUENCY_MEDIAN3_TARGET_WEIGHT * arrays["median3"]
            + FREQUENCY_MEDIAN7_TARGET_WEIGHT * arrays["median7"]
        )
        desired = np.where(expected_consensus, consensus_target, base_target)
        expected_target = np.where(
            roots["update_safe"],
            arrays["source"] + float(strength) * (desired - arrays["source"]),
            arrays["source"],
        )
        if not np.array_equal(arrays["target"], expected_target):
            raise ValueError(
                f"frequency frozen target has invalid semantics: {map_name}"
            )
    if (
        frozen_bundle.get("total_updated_entries") != total_updated
        or frozen_bundle.get("total_consensus_entries") != total_consensus
    ):
        raise ValueError("frequency frozen total entry counts do not match artifact")
    return {
        "bundle_schema": FREQUENCY_BUNDLE_SCHEMA,
        "path": artifact_path,
        "sha256": actual_file_hash,
        "bytes": expected_bytes,
        "created_after_step": int(created_after_step),
        "strength": float(strength),
        "median_chunk_rows": dict(median_chunk_rows),
        "edge_threshold_full_precision": float(edge_threshold_full),
        "edge_threshold_png_quantized": float(edge_threshold_png),
        **roots,
        "maps": maps,
        "total_updated_entries": total_updated,
        "total_consensus_entries": total_consensus,
    }


def _load_frequency_adaptive_frozen_artifact(
    material_dir: Path,
    regularization: dict[str, Any],
    frozen_bundle: dict[str, Any],
    expected_shape: tuple[int, int],
) -> dict[str, Any]:
    stage_plan = regularization["stage_plan"]
    artifact = frozen_bundle.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("adaptive frequency candidate is missing frozen artifact")
    if artifact.get("schema") != FREQUENCY_FROZEN_ARTIFACT_SCHEMA:
        raise ValueError("adaptive frequency frozen artifact has unsupported schema")
    if artifact.get("format") != "numpy_npz_compressed":
        raise ValueError("adaptive frequency frozen artifact has unsupported format")
    relative_path = artifact.get("path")
    if relative_path != FREQUENCY_FROZEN_ARTIFACT_NAME:
        raise ValueError("adaptive frequency artifact path is not canonical")
    relative = Path(relative_path)
    if relative.is_absolute() or len(relative.parts) != 1:
        raise ValueError("adaptive frequency artifact path must be a filename")
    artifact_path = (material_dir / relative).resolve()
    if artifact_path.parent != material_dir.resolve():
        raise ValueError("adaptive frequency artifact path escapes its material folder")
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"adaptive frequency frozen artifact is missing: {artifact_path}"
        )
    expected_bytes = artifact.get("bytes")
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes <= 0
        or artifact_path.stat().st_size != expected_bytes
    ):
        raise ValueError("adaptive frequency artifact byte size differs from provenance")
    actual_file_hash = _file_sha256(artifact_path)
    if (
        not isinstance(artifact.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])
        or artifact["sha256"] != actual_file_hash
    ):
        raise ValueError("adaptive frequency artifact SHA-256 differs from provenance")

    root_names = (
        "fit_foreground",
        "full_foreground5",
        "edge_protected_full_precision",
        "edge_protected_png_quantized",
        "edge_protected",
        "update_safe",
        "guide_texture_full_precision",
        "guide_texture_png_quantized",
        "guide_texture_core",
        "guide_texture_halo",
    )
    float_names = (
        "source",
        "median3",
        "median7",
        "fixed_target",
        "policy_target",
        "target",
    )
    mask_names = ("mask", "consensus_mask")
    expected_keys = {"metadata", *root_names}
    for map_name in SCALAR_MAPS:
        expected_keys.update(
            f"{map_name}__{name}" for name in (*float_names, *mask_names)
        )
    try:
        with np.load(artifact_path, allow_pickle=False) as payload:
            if set(payload.files) != expected_keys:
                raise ValueError(
                    "adaptive frequency frozen artifact has an unexpected array set"
                )
            metadata_array = payload["metadata"]
            if metadata_array.shape != ():
                raise ValueError(
                    "adaptive frequency frozen artifact metadata must be scalar"
                )
            try:
                metadata = json.loads(str(metadata_array.item()))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    "adaptive frequency frozen artifact metadata is invalid"
                ) from exc
            roots = {
                name: np.array(payload[name], copy=True) for name in root_names
            }
            maps = {
                map_name: {
                    name: np.array(payload[f"{map_name}__{name}"], copy=True)
                    for name in (*float_names, *mask_names)
                }
                for map_name in SCALAR_MAPS
            }
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(
            "adaptive frequency"
        ):
            raise
        raise ValueError(
            f"could not read adaptive frequency frozen artifact: {exc}"
        ) from exc

    created_after_step = frozen_bundle.get("created_after_step")
    strength = frozen_bundle.get("strength")
    median_chunk_rows = frozen_bundle.get("median_chunk_rows")
    edge_threshold_full = frozen_bundle.get("edge_threshold_full_precision")
    edge_threshold_png = frozen_bundle.get("edge_threshold_png_quantized")
    guide_scales = frozen_bundle.get("guide_texture", {}).get("scales")
    metadata_maps = metadata.get("maps") if isinstance(metadata, dict) else None
    if frozen_bundle.get("consensus_entry_semantics") != (
        FREQUENCY_ADAPTIVE_CONSENSUS_ENTRY_SEMANTICS
    ):
        raise ValueError(
            "adaptive frequency frozen-bundle consensus semantics are invalid"
        )
    if not isinstance(metadata, dict) or metadata.get(
        "consensus_entry_semantics"
    ) != FREQUENCY_ADAPTIVE_CONSENSUS_ENTRY_SEMANTICS:
        raise ValueError(
            "adaptive frequency artifact metadata consensus semantics are invalid"
        )
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema") != FREQUENCY_FROZEN_ARTIFACT_SCHEMA
        or metadata.get("bundle_schema") != FREQUENCY_ADAPTIVE_BUNDLE_SCHEMA
        or metadata.get("created_after_step") != created_after_step
        or metadata.get("strength") != strength
        or metadata.get("median_chunk_rows") != median_chunk_rows
        or metadata.get("edge_threshold_full_precision") != edge_threshold_full
        or metadata.get("edge_threshold_png_quantized") != edge_threshold_png
        or metadata.get("guide_texture_scales") != guide_scales
        or not isinstance(metadata_maps, list)
        or len(metadata_maps) != len(set(metadata_maps))
        or set(metadata_maps) != set(SCALAR_MAPS)
    ):
        raise ValueError(
            "adaptive frequency artifact metadata differs from acquisition provenance"
        )
    if (
        not isinstance(created_after_step, int)
        or isinstance(created_after_step, bool)
        or stage_plan.get("detector_after_data_step") != created_after_step
    ):
        raise ValueError("adaptive frequency artifact has a stale stage boundary")
    if (
        not isinstance(strength, (int, float))
        or isinstance(strength, bool)
        or not math.isfinite(float(strength))
        or not 0.0 < float(strength) <= 1.0
        or stage_plan.get("strength") != strength
    ):
        raise ValueError("adaptive frequency artifact has invalid strength provenance")
    if (
        not isinstance(median_chunk_rows, dict)
        or set(median_chunk_rows) != {"3x3", "7x7"}
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            or value > expected_shape[0]
            for value in median_chunk_rows.values()
        )
    ):
        raise ValueError("adaptive frequency artifact has an invalid chunk plan")
    for label, value in (
        ("full-precision", edge_threshold_full),
        ("PNG-quantized", edge_threshold_png),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                f"adaptive frequency artifact has invalid {label} edge threshold"
            )
    expected_scale_keys = {
        "full_precision_albedo_scale",
        "full_precision_normal_scale",
        "png_quantized_albedo_scale",
        "png_quantized_normal_scale",
    }
    if (
        not isinstance(guide_scales, dict)
        or set(guide_scales) != expected_scale_keys
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in guide_scales.values()
        )
    ):
        raise ValueError("adaptive frequency guide scales are invalid")

    for name in root_names:
        if roots[name].shape != expected_shape or roots[name].dtype != np.bool_:
            raise ValueError(
                f"adaptive frequency {name} must be bool {expected_shape}"
            )
    if not np.array_equal(
        roots["full_foreground5"],
        _erode_mask(roots["fit_foreground"], radius=2),
    ):
        raise ValueError("adaptive frequency full foreground is stale")
    if not np.array_equal(
        roots["edge_protected"],
        roots["edge_protected_full_precision"]
        | roots["edge_protected_png_quantized"],
    ):
        raise ValueError("adaptive frequency edge-protection union is stale")
    if not np.array_equal(
        roots["update_safe"],
        roots["full_foreground5"] & ~roots["edge_protected"],
    ):
        raise ValueError("adaptive frequency update-safe mask is stale")
    expected_core = (
        roots["guide_texture_full_precision"]
        | roots["guide_texture_png_quantized"]
    )
    if not np.array_equal(roots["guide_texture_core"], expected_core):
        raise ValueError("adaptive frequency guide-texture core is stale")
    if np.any(expected_core & ~roots["full_foreground5"]):
        raise ValueError("adaptive frequency guide texture leaves the interior")
    padded = np.pad(expected_core, 1, mode="constant", constant_values=False)
    expected_halo = np.zeros(expected_shape, dtype=bool)
    for row_offset in range(3):
        for column_offset in range(3):
            expected_halo |= padded[
                row_offset : row_offset + expected_shape[0],
                column_offset : column_offset + expected_shape[1],
            ]
    expected_halo &= roots["full_foreground5"]
    if not np.array_equal(roots["guide_texture_halo"], expected_halo):
        raise ValueError("adaptive frequency guide-texture halo is stale")

    tensor_hashes = {
        "root": {name: _array_sha256(roots[name]) for name in root_names},
        "maps": {
            map_name: {
                name: _array_sha256(maps[map_name][name])
                for name in (*float_names, *mask_names)
            }
            for map_name in SCALAR_MAPS
        },
    }
    if (
        metadata.get("tensor_hashes") != tensor_hashes
        or frozen_bundle.get("tensor_hashes") != tensor_hashes
    ):
        raise ValueError(
            "adaptive frequency frozen tensor hashes differ from artifact arrays"
        )
    root_provenance = {
        "fit_foreground": "fit_foreground_sha256",
        "edge_protected_full_precision": "edge_protected_full_precision_sha256",
        "edge_protected_png_quantized": "edge_protected_png_quantized_sha256",
        "edge_protected": "edge_protected_sha256",
        "update_safe": "update_safe_sha256",
    }
    for name, provenance_name in root_provenance.items():
        if frozen_bundle.get(provenance_name) != tensor_hashes["root"][name]:
            raise ValueError(f"adaptive frequency {name} hash is stale")
    guide_provenance = frozen_bundle.get("guide_texture")
    guide_masks = (
        guide_provenance.get("masks")
        if isinstance(guide_provenance, dict)
        else None
    )
    if (
        not isinstance(guide_masks, dict)
        or set(guide_masks) != set(root_names[-4:])
    ):
        raise ValueError("adaptive frequency guide-mask provenance is invalid")
    for name in root_names[-4:]:
        record = guide_masks[name]
        if (
            not isinstance(record, dict)
            or record.get("pixels") != int(np.count_nonzero(roots[name]))
            or record.get("sha256") != tensor_hashes["root"][name]
        ):
            raise ValueError(f"adaptive frequency {name} provenance is stale")

    map_provenance = frozen_bundle.get("maps")
    if not isinstance(map_provenance, dict) or set(map_provenance) != set(
        SCALAR_MAPS
    ):
        raise ValueError("adaptive frequency map provenance is invalid")
    total_updated = 0
    total_consensus = 0
    for map_name, arrays in maps.items():
        for name in float_names:
            values = arrays[name]
            if values.shape != expected_shape or values.dtype != np.float32:
                raise ValueError(
                    f"adaptive frequency {name} {map_name} must be float32 "
                    f"{expected_shape}"
                )
            if not np.all(np.isfinite(values)) or np.any(
                (values < 0.0) | (values > 1.0)
            ):
                raise ValueError(
                    f"adaptive frequency {name} {map_name} is outside [0,1]"
                )
        for name in mask_names:
            values = arrays[name]
            if values.shape != expected_shape or values.dtype != np.bool_:
                raise ValueError(
                    f"adaptive frequency {name} {map_name} must be bool "
                    f"{expected_shape}"
                )
        provenance = map_provenance[map_name]
        updated_count = int(np.count_nonzero(arrays["mask"]))
        consensus_count = int(np.count_nonzero(arrays["consensus_mask"]))
        if (
            not isinstance(provenance, dict)
            or provenance.get("updated_entries") != updated_count
            or provenance.get("consensus_entries") != consensus_count
        ):
            raise ValueError(f"adaptive frequency {map_name} counts are stale")
        for name in (*float_names, *mask_names):
            if provenance.get(f"{name}_sha256") != tensor_hashes["maps"][
                map_name
            ][name]:
                raise ValueError(
                    f"adaptive frequency {name} {map_name} hash is stale"
                )
        total_updated += updated_count
        total_consensus += consensus_count

    v1_evidence = np.sum(
        np.stack(
            [
                np.abs(maps[name]["source"] - maps[name]["median7"])
                > FREQUENCY_EVIDENCE_THRESHOLD
                for name in SCALAR_MAPS
            ],
            axis=0,
        ),
        axis=0,
        dtype=np.int64,
    )
    for map_name, arrays in maps.items():
        source = arrays["source"]
        median3 = arrays["median3"]
        median7 = arrays["median7"]
        fixed_consensus = (
            (v1_evidence >= FREQUENCY_MIN_EVIDENCE_MAPS)
            & (np.abs(source - median7) > FREQUENCY_OWN_DEVIATION)
            & roots["update_safe"]
        )
        fixed_base = source + FREQUENCY_BASE_BLEND * (median3 - source)
        fixed_consensus_target = (
            FREQUENCY_MEDIAN3_TARGET_WEIGHT * median3
            + FREQUENCY_MEDIAN7_TARGET_WEIGHT * median7
        )
        fixed_desired = np.where(
            fixed_consensus,
            fixed_consensus_target,
            fixed_base,
        )
        # Replay the v1 producer literally.  Even at strength 1.0, the
        # float32 subtract/multiply/add sequence is not bit-identical to the
        # algebraically simplified ``fixed_desired`` value.
        expected_fixed = np.where(
            roots["update_safe"],
            source + 1.0 * (fixed_desired - source),
            source,
        )
        if not np.array_equal(arrays["fixed_target"], expected_fixed):
            raise ValueError(f"adaptive frequency fixed v1 target is stale: {map_name}")
        strong = (
            FREQUENCY_ADAPTIVE_STRONG_MEDIAN3_WEIGHT * median3
            + FREQUENCY_ADAPTIVE_STRONG_MEDIAN7_WEIGHT * median7
        )
        if map_name in FREQUENCY_ADAPTIVE_FOCUSED_MAPS:
            expected_policy = np.where(
                roots["guide_texture_core"], expected_fixed, strong
            )
            strong_policy_eligible = (
                roots["update_safe"] & ~roots["guide_texture_core"]
            )
        else:
            halo_target = source + FREQUENCY_ADAPTIVE_HALO_V1_BLEND * (
                expected_fixed - source
            )
            expected_policy = np.where(
                roots["guide_texture_halo"], halo_target, strong
            )
            strong_policy_eligible = (
                roots["update_safe"] & ~roots["guide_texture_halo"]
            )
        expected_policy = np.where(
            roots["update_safe"], expected_policy, source
        )
        expected_target = source + float(strength) * (expected_policy - source)
        expected_consensus = strong_policy_eligible & (
            expected_target != source
        )
        expected_mask = roots["update_safe"] & (expected_target != source)
        if not np.array_equal(arrays["policy_target"], expected_policy):
            raise ValueError(f"adaptive frequency policy target is stale: {map_name}")
        if not np.array_equal(arrays["consensus_mask"], expected_consensus):
            raise ValueError(f"adaptive frequency strong mask is stale: {map_name}")
        if not np.array_equal(arrays["target"], expected_target):
            raise ValueError(f"adaptive frequency target is stale: {map_name}")
        if not np.array_equal(arrays["mask"], expected_mask):
            raise ValueError(f"adaptive frequency update mask is stale: {map_name}")
    if (
        frozen_bundle.get("total_updated_entries") != total_updated
        or frozen_bundle.get("total_consensus_entries") != total_consensus
    ):
        raise ValueError("adaptive frequency aggregate counts are stale")
    return {
        "bundle_schema": FREQUENCY_ADAPTIVE_BUNDLE_SCHEMA,
        "consensus_entry_semantics": (
            FREQUENCY_ADAPTIVE_CONSENSUS_ENTRY_SEMANTICS
        ),
        "path": artifact_path,
        "sha256": actual_file_hash,
        "bytes": expected_bytes,
        "created_after_step": int(created_after_step),
        "strength": float(strength),
        "median_chunk_rows": dict(median_chunk_rows),
        "edge_threshold_full_precision": float(edge_threshold_full),
        "edge_threshold_png_quantized": float(edge_threshold_png),
        "guide_texture_scales": dict(guide_scales),
        **roots,
        "maps": maps,
        "total_updated_entries": total_updated,
        "total_consensus_entries": total_consensus,
    }


def _optional_mean(total: float, count: int) -> float | None:
    return float(total / count) if count else None


def _measure_impulse_cleanup(
    baseline_camera: Path,
    regularized_camera: Path,
    profiles: Sequence[str],
    fit_mask: np.ndarray,
    acquisitions: dict[str, dict[str, dict[str, Any]]],
    contract: dict[str, Any],
) -> dict[str, Any] | None:
    if contract["regularization_kind"] != "impulse-median":
        return None
    candidate_identities = {
        (
            acquisitions["regularized"][profile]["checkpoint_signature"].get(
                "schema"
            ),
            _adapter_identity(
                acquisitions["regularized"][profile]["checkpoint_signature"]
            ),
        )
        for profile in profiles
    }
    if candidate_identities == {_STAGED_IMPULSE_IDENTITY}:
        return {
            "schema": "ictpolarreal.impulse-cleanup-comparison.v1",
            "available": False,
            "reason": "v10 staged cleanup did not export a frozen NPZ artifact",
        }
    supported_acquisition_schemas = {
        _PROXIMAL_V11_IDENTITY: "ictpolarreal.end2end-disney.v11",
        _PROXIMAL_V12_IDENTITY: "ictpolarreal.end2end-disney.v12",
        _FREQUENCY_V2_IDENTITY: "ictpolarreal.end2end-disney.v13",
    }
    if len(candidate_identities) != 1:
        raise ValueError(
            "impulse cleanup report requires one supported candidate identity; "
            f"found {candidate_identities}"
        )
    candidate_identity = next(iter(candidate_identities))
    expected_acquisition_schema = supported_acquisition_schemas.get(
        candidate_identity
    )
    if expected_acquisition_schema is None:
        raise ValueError(
            "impulse cleanup report requires one supported candidate identity; "
            f"found {candidate_identities}"
        )
    if fit_mask.ndim != 2 or not np.any(fit_mask):
        raise ValueError("impulse cleanup report requires a nonempty 2D fit mask")

    aggregate = {
        "fit_map_entries": 0,
        "flagged_map_entries": 0,
        "flagged_spatial_union_pixels": 0,
        "flagged_distance_baseline_sum": 0.0,
        "flagged_distance_candidate_sum": 0.0,
        "resolved_count": 0,
        "outside_union_count": 0,
        "outside_union_sum": 0.0,
        "outside_union_max": 0.0,
        "outside_per_map_count": 0,
        "outside_per_map_sum": 0.0,
        "outside_per_map_max": 0.0,
    }
    per_profile = {}
    common_dead_zone = None
    for profile in profiles:
        acquisition = acquisitions["regularized"][profile]
        if acquisition.get("schema") != expected_acquisition_schema:
            raise ValueError(
                "proximal impulse candidate acquisition schema does not match "
                "its checkpoint/adapter identity"
            )
        regularization = acquisition.get("regularization", {})
        if regularization.get("cleanup_applied") is not True:
            raise ValueError("proximal impulse candidate did not complete cleanup")
        if regularization.get("data_objective_only") is not True:
            raise ValueError("proximal impulse candidate must keep data fitting unchanged")
        settings = regularization.get("settings")
        stage_plan = regularization.get("stage_plan")
        if (
            not isinstance(settings, dict)
            or settings.get("stage")
            != "full_data_fit_then_post_fit_frozen_impulse_proximal"
            or not isinstance(stage_plan, dict)
            or stage_plan.get("cleanup_optimizer_steps") != 0
            or stage_plan.get("data_fit_steps") != acquisition.get("steps")
            or stage_plan.get("detector_after_data_step")
            != stage_plan.get("data_fit_steps")
        ):
            raise ValueError("proximal impulse candidate has an invalid post-fit plan")
        diagnostic = regularization.get("cleanup_diagnostic")
        if not isinstance(diagnostic, dict) or diagnostic.get("schema") != (
            "ictpolarreal.impulse-proximal-diagnostic.v1"
        ):
            raise ValueError("proximal impulse candidate has invalid cleanup diagnostics")
        try:
            dead_zone = float(diagnostic["dead_zone"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("proximal impulse candidate has an invalid dead zone") from exc
        if not math.isfinite(dead_zone) or dead_zone < 0.0:
            raise ValueError("proximal impulse candidate has an invalid dead zone")
        try:
            settings_dead_zone = float(settings["dead_zone"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("proximal impulse settings have an invalid dead zone") from exc
        if not math.isclose(
            settings_dead_zone, dead_zone, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("cleanup diagnostic dead zone differs from settings")
        if common_dead_zone is None:
            common_dead_zone = dead_zone
        elif not math.isclose(common_dead_zone, dead_zone, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("impulse cleanup dead zone differs across profiles")

        artifact = _load_impulse_frozen_artifact(
            regularized_camera / "material" / profile,
            acquisition,
            fit_mask.shape,
        )
        if diagnostic.get("flagged_centers") != artifact["total_flagged_centers"]:
            raise ValueError("cleanup diagnostic flagged count differs from artifact")
        diagnostic_maps = diagnostic.get("maps")
        if not isinstance(diagnostic_maps, dict) or set(diagnostic_maps) != set(
            SCALAR_MAPS
        ):
            raise ValueError("cleanup diagnostic is missing per-map counts")

        fit_pixels = int(np.count_nonzero(fit_mask))
        spatial_union = np.zeros(fit_mask.shape, dtype=bool)
        map_differences = {}
        profile_accumulator = {
            "flagged": 0,
            "baseline_distance": 0.0,
            "candidate_distance": 0.0,
            "resolved": 0,
            "outside_per_map_count": 0,
            "outside_per_map_sum": 0.0,
            "outside_per_map_max": 0.0,
        }
        per_map = {}
        resolved_threshold = dead_zone + EXPORT_QUANTIZATION_TOLERANCE
        for map_name in SCALAR_MAPS:
            baseline_values = _read_scalar_map(
                baseline_camera / "material" / profile / "maps" / f"{map_name}.png"
            )
            candidate_values = _read_scalar_map(
                regularized_camera
                / "material"
                / profile
                / "maps"
                / f"{map_name}.png"
            )
            if baseline_values.shape != fit_mask.shape or candidate_values.shape != (
                fit_mask.shape
            ):
                raise ValueError("impulse cleanup maps do not match the fit-mask shape")
            target = artifact["maps"][map_name]["target"]
            flagged = artifact["maps"][map_name]["mask"]
            if np.any(flagged & ~fit_mask):
                raise ValueError(
                    f"impulse frozen flags fall outside the fit mask: {map_name}"
                )
            flagged_count = int(np.count_nonzero(flagged))
            diagnostic_entry = diagnostic_maps.get(map_name)
            if (
                not isinstance(diagnostic_entry, dict)
                or diagnostic_entry.get("flagged_centers") != flagged_count
            ):
                raise ValueError(
                    f"cleanup diagnostic flagged count differs for {map_name}"
                )
            spatial_union |= flagged
            exported_change = np.abs(candidate_values - baseline_values)
            map_differences[map_name] = exported_change
            baseline_distance = np.abs(baseline_values[flagged] - target[flagged])
            candidate_distance = np.abs(candidate_values[flagged] - target[flagged])
            resolved_count = int(
                np.count_nonzero(candidate_distance <= resolved_threshold + 1e-12)
            )
            outside = ~flagged
            outside_values = exported_change[outside]
            outside_sum = float(outside_values.sum(dtype=np.float64))
            outside_max = float(outside_values.max(initial=0.0))
            baseline_sum = float(baseline_distance.sum(dtype=np.float64))
            candidate_sum = float(candidate_distance.sum(dtype=np.float64))
            outside_count = int(outside_values.size)
            per_map[map_name] = {
                "flagged_centers": flagged_count,
                "flagged_fraction_of_fit_foreground": (
                    float(flagged_count / fit_pixels) if fit_pixels else 0.0
                ),
                "mean_absolute_distance_to_frozen_target": {
                    "baseline": _optional_mean(baseline_sum, flagged_count),
                    "candidate": _optional_mean(candidate_sum, flagged_count),
                },
                "resolved_to_dead_zone": {
                    "count": resolved_count,
                    "fraction": (
                        float(resolved_count / flagged_count)
                        if flagged_count
                        else None
                    ),
                },
                "exported_change_outside_own_flags": {
                    "entry_count": outside_count,
                    "mean_absolute": _optional_mean(outside_sum, outside_count),
                    "maximum_absolute": outside_max,
                },
            }
            profile_accumulator["flagged"] += flagged_count
            profile_accumulator["baseline_distance"] += baseline_sum
            profile_accumulator["candidate_distance"] += candidate_sum
            profile_accumulator["resolved"] += resolved_count
            profile_accumulator["outside_per_map_count"] += outside_count
            profile_accumulator["outside_per_map_sum"] += outside_sum
            profile_accumulator["outside_per_map_max"] = max(
                profile_accumulator["outside_per_map_max"], outside_max
            )

        outside_union = ~spatial_union
        outside_union_count = int(np.count_nonzero(outside_union) * len(SCALAR_MAPS))
        outside_union_sum = float(
            sum(
                difference[outside_union].sum(dtype=np.float64)
                for difference in map_differences.values()
            )
        )
        outside_union_max = float(
            max(
                (
                    difference[outside_union].max(initial=0.0)
                    for difference in map_differences.values()
                ),
                default=0.0,
            )
        )
        flagged = profile_accumulator["flagged"]
        union_pixels = int(np.count_nonzero(spatial_union))
        per_profile[profile] = {
            "artifact": {
                "validated": True,
                "path": str(artifact["path"]),
                "sha256": artifact["sha256"],
                "bytes": artifact["bytes"],
                "created_after_step": artifact["created_after_step"],
            },
            "fit_foreground_pixels": fit_pixels,
            "dead_zone": dead_zone,
            "export_quantization_tolerance": EXPORT_QUANTIZATION_TOLERANCE,
            "resolved_threshold": resolved_threshold,
            "flagged": {
                "map_entries": flagged,
                "map_entry_fraction": float(flagged / (fit_pixels * len(SCALAR_MAPS))),
                "spatial_union_pixels": union_pixels,
                "spatial_union_fraction": float(union_pixels / fit_pixels),
            },
            "mean_absolute_distance_to_frozen_target": {
                "baseline": _optional_mean(
                    profile_accumulator["baseline_distance"], flagged
                ),
                "candidate": _optional_mean(
                    profile_accumulator["candidate_distance"], flagged
                ),
            },
            "resolved_to_dead_zone": {
                "count": profile_accumulator["resolved"],
                "fraction": (
                    float(profile_accumulator["resolved"] / flagged)
                    if flagged
                    else None
                ),
            },
            "exported_change_outside_spatial_flag_union": {
                "entry_count": outside_union_count,
                "mean_absolute": _optional_mean(outside_union_sum, outside_union_count),
                "maximum_absolute": outside_union_max,
            },
            "exported_change_outside_per_map_flags": {
                "entry_count": profile_accumulator["outside_per_map_count"],
                "mean_absolute": _optional_mean(
                    profile_accumulator["outside_per_map_sum"],
                    profile_accumulator["outside_per_map_count"],
                ),
                "maximum_absolute": profile_accumulator["outside_per_map_max"],
            },
            "maps": per_map,
        }
        aggregate["fit_map_entries"] += fit_pixels * len(SCALAR_MAPS)
        aggregate["flagged_map_entries"] += flagged
        aggregate["flagged_spatial_union_pixels"] += union_pixels
        aggregate["flagged_distance_baseline_sum"] += profile_accumulator[
            "baseline_distance"
        ]
        aggregate["flagged_distance_candidate_sum"] += profile_accumulator[
            "candidate_distance"
        ]
        aggregate["resolved_count"] += profile_accumulator["resolved"]
        aggregate["outside_union_count"] += outside_union_count
        aggregate["outside_union_sum"] += outside_union_sum
        aggregate["outside_union_max"] = max(
            aggregate["outside_union_max"], outside_union_max
        )
        aggregate["outside_per_map_count"] += profile_accumulator[
            "outside_per_map_count"
        ]
        aggregate["outside_per_map_sum"] += profile_accumulator[
            "outside_per_map_sum"
        ]
        aggregate["outside_per_map_max"] = max(
            aggregate["outside_per_map_max"],
            profile_accumulator["outside_per_map_max"],
        )

    flagged = aggregate["flagged_map_entries"]
    public_aggregate = {
        "dead_zone": common_dead_zone,
        "export_quantization_tolerance": EXPORT_QUANTIZATION_TOLERANCE,
        "flagged": {
            "map_entries": flagged,
            "map_entry_fraction": float(flagged / aggregate["fit_map_entries"]),
            "spatial_union_pixels_across_profiles": aggregate[
                "flagged_spatial_union_pixels"
            ],
        },
        "mean_absolute_distance_to_frozen_target": {
            "baseline": _optional_mean(
                aggregate["flagged_distance_baseline_sum"], flagged
            ),
            "candidate": _optional_mean(
                aggregate["flagged_distance_candidate_sum"], flagged
            ),
        },
        "resolved_to_dead_zone": {
            "count": aggregate["resolved_count"],
            "fraction": (
                float(aggregate["resolved_count"] / flagged) if flagged else None
            ),
        },
        "exported_change_outside_spatial_flag_union": {
            "entry_count": aggregate["outside_union_count"],
            "mean_absolute": _optional_mean(
                aggregate["outside_union_sum"], aggregate["outside_union_count"]
            ),
            "maximum_absolute": aggregate["outside_union_max"],
        },
        "exported_change_outside_per_map_flags": {
            "entry_count": aggregate["outside_per_map_count"],
            "mean_absolute": _optional_mean(
                aggregate["outside_per_map_sum"],
                aggregate["outside_per_map_count"],
            ),
            "maximum_absolute": aggregate["outside_per_map_max"],
        },
    }
    return {
        "schema": "ictpolarreal.impulse-cleanup-comparison.v1",
        "available": True,
        "scope": (
            "Distances and outside-flag changes are measured from 8-bit exported "
            "scalar maps against the hash-validated frozen float32 targets."
        ),
        "resolved_definition": (
            "candidate distance <= dead_zone + half an 8-bit quantization step"
        ),
        "aggregate": public_aggregate,
        "profiles": per_profile,
    }


def _validate_frequency_baseline_state(regularization: Any) -> None:
    if not isinstance(regularization, dict):
        raise ValueError("frequency comparison requires baseline provenance")
    try:
        weight = float(regularization.get("weight", -1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("frequency comparison requires a zero-weight baseline") from exc
    if not math.isfinite(weight) or weight != 0.0:
        raise ValueError("frequency comparison requires a zero-weight baseline")
    stage_plan = regularization.get("stage_plan")
    if stage_plan is not None and (
        not isinstance(stage_plan, dict) or stage_plan.get("enabled") is not False
    ):
        raise ValueError("frequency baseline stage must be disabled")
    if regularization.get("cleanup_applied") not in (None, False):
        raise ValueError("frequency baseline must not apply post-fit cleanup")


def _validate_active_frequency_baseline_state(regularization: Any) -> None:
    if not isinstance(regularization, dict):
        raise ValueError("active frequency comparison requires baseline provenance")
    if regularization.get("kind") != "frequency-consensus":
        raise ValueError(
            "active frequency comparison requires a frequency-consensus v1 baseline"
        )
    try:
        weight = float(regularization["weight"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "active frequency comparison requires a positive baseline weight"
        ) from exc
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError(
            "active frequency comparison requires a positive baseline weight"
        )
    _validate_frequency_settings(regularization.get("settings"))
    stage_plan = regularization.get("stage_plan")
    if (
        not isinstance(stage_plan, dict)
        or stage_plan.get("enabled") is not True
        or stage_plan.get("cleanup_optimizer_steps") != 0
        or stage_plan.get("post_fit_updates") != 1
    ):
        raise ValueError("active frequency baseline has an invalid post-fit plan")
    if regularization.get("cleanup_applied") is not True:
        raise ValueError("active frequency baseline did not apply its post-fit update")
    if regularization.get("data_objective_only") is not True:
        raise ValueError("active frequency baseline must keep data fitting unchanged")
    diagnostic = regularization.get("cleanup_diagnostic")
    if not isinstance(diagnostic, dict) or diagnostic.get("schema") != (
        FREQUENCY_DIAGNOSTIC_SCHEMA
    ):
        raise ValueError("active frequency baseline has invalid cleanup diagnostics")
    bundle = regularization.get("frozen_bundle")
    if not isinstance(bundle, dict) or bundle.get("schema") != (
        FREQUENCY_BUNDLE_SCHEMA
    ):
        raise ValueError("active frequency baseline has invalid frozen provenance")


def _validate_frequency_settings(settings: Any) -> None:
    expected = {
        "map_domain": "constrained_0_1_full_precision",
        "stage": "full_data_fit_then_one_frozen_frequency_consensus_update",
        "weighted_median_windows": [3, 7],
        "spatial_sigma3": FREQUENCY_SPATIAL_SIGMA3,
        "spatial_sigma7": FREQUENCY_SPATIAL_SIGMA7,
        "albedo_sigma": FREQUENCY_ALBEDO_SIGMA,
        "normal_sigma": FREQUENCY_NORMAL_SIGMA,
        "edge_percentile": FREQUENCY_EDGE_PERCENTILE,
        "base_target": f"value+{FREQUENCY_BASE_BLEND:g}*(median3-value)",
        "cross_map_evidence": (
            f"count(abs(value-median7)>{FREQUENCY_EVIDENCE_THRESHOLD:g})"
        ),
        "minimum_evidence_maps": FREQUENCY_MIN_EVIDENCE_MAPS,
        "own_deviation_threshold": FREQUENCY_OWN_DEVIATION,
        "consensus_target": (
            f"{FREQUENCY_MEDIAN3_TARGET_WEIGHT:g}*median3+"
            f"{FREQUENCY_MEDIAN7_TARGET_WEIGHT:g}*median7"
        ),
        "cleanup_optimizer_steps": 0,
    }
    if not isinstance(settings, dict):
        raise ValueError("frequency candidate has invalid algorithm settings")
    mismatched = [
        name for name, value in expected.items() if settings.get(name) != value
    ]
    if mismatched:
        raise ValueError(
            "frequency candidate algorithm settings differ from the report "
            f"contract: {mismatched}"
        )


def _validate_frequency_adaptive_settings(settings: Any) -> None:
    expected = {
        "map_domain": "constrained_0_1_full_precision",
        "stage": "full_data_fit_then_one_frozen_adaptive_consensus_update",
        "weighted_median_windows": [3, 7],
        "spatial_sigma3": FREQUENCY_SPATIAL_SIGMA3,
        "spatial_sigma7": FREQUENCY_SPATIAL_SIGMA7,
        "albedo_sigma": FREQUENCY_ALBEDO_SIGMA,
        "normal_sigma": FREQUENCY_NORMAL_SIGMA,
        "edge_percentile": FREQUENCY_EDGE_PERCENTILE,
        "guide_texture_band_sigmas": [0.8, 2.4],
        "guide_texture_scale_percentile": (
            FREQUENCY_ADAPTIVE_GUIDE_SCALE_PERCENTILE
        ),
        "guide_texture_threshold": FREQUENCY_GUIDE_BAND_THRESHOLD,
        "guide_texture_sources": [
            "full_precision_normalized_baseColor_and_normal",
            "exact_png_quantized_baseColor_and_normal",
        ],
        "guide_texture_core": "union_of_full_precision_and_png_masks",
        "guide_texture_halo": (
            "3x3_square_dilation_of_core_clipped_to_full5"
        ),
        "focused_maps": list(FREQUENCY_ADAPTIVE_FOCUSED_MAPS),
        "strong_target": (
            f"{FREQUENCY_ADAPTIVE_STRONG_MEDIAN3_WEIGHT:g}*median3+"
            f"{FREQUENCY_ADAPTIVE_STRONG_MEDIAN7_WEIGHT:g}*median7"
        ),
        "focused_policy": "v1_in_core_else_strong_within_update_safe",
        "other_policy": (
            f"source+{FREQUENCY_ADAPTIVE_HALO_V1_BLEND:g}*(v1-source)_in_"
            "halo_else_strong_within_update_safe"
        ),
        "base_target": "exact_full_strength_frequency_consensus_v1_target",
        "cleanup_optimizer_steps": 0,
    }
    if not isinstance(settings, dict):
        raise ValueError("adaptive frequency candidate has invalid algorithm settings")
    mismatched = [
        name for name, value in expected.items() if settings.get(name) != value
    ]
    if mismatched:
        raise ValueError(
            "adaptive frequency candidate algorithm settings differ from the "
            f"report contract: {mismatched}"
        )


def _validate_frequency_artifact_upgrade_pair(
    baseline: dict[str, Any],
    adaptive: dict[str, Any],
    *,
    profile: str,
) -> None:
    if baseline.get("bundle_schema") != FREQUENCY_BUNDLE_SCHEMA:
        raise ValueError(f"{profile} baseline does not contain the v1 frozen bundle")
    if adaptive.get("bundle_schema") != FREQUENCY_ADAPTIVE_BUNDLE_SCHEMA:
        raise ValueError(
            f"{profile} candidate does not contain the adaptive frozen bundle"
        )
    scalar_fields = (
        "created_after_step",
        "median_chunk_rows",
        "edge_threshold_full_precision",
        "edge_threshold_png_quantized",
    )
    mismatched = [name for name in scalar_fields if baseline.get(name) != adaptive.get(name)]
    if mismatched:
        raise ValueError(
            f"{profile} adaptive and v1 frozen data-fit state differs: {mismatched}"
        )
    if baseline.get("strength") != 1.0 or adaptive.get("strength") != 1.0:
        raise ValueError(
            "active frequency comparison requires the full reference strength"
        )
    for name in (
        "fit_foreground",
        "full_foreground5",
        "edge_protected_full_precision",
        "edge_protected_png_quantized",
        "edge_protected",
        "update_safe",
    ):
        if not np.array_equal(baseline.get(name), adaptive.get(name)):
            raise ValueError(
                f"{profile} adaptive and v1 frozen {name} differs"
            )
    for map_name in SCALAR_MAPS:
        baseline_map = baseline["maps"][map_name]
        adaptive_map = adaptive["maps"][map_name]
        for name in ("source", "median3", "median7"):
            if not np.array_equal(baseline_map[name], adaptive_map[name]):
                raise ValueError(
                    f"{profile}/{map_name} adaptive and v1 frozen {name} differs"
                )
        if not np.array_equal(
            baseline_map["target"], adaptive_map["fixed_target"]
        ):
            raise ValueError(
                f"{profile}/{map_name} adaptive frozen v1 target differs from baseline"
            )


def _validate_frequency_evaluation_guard(
    acquisition: dict[str, Any],
    *,
    profile: str,
) -> dict[str, Any]:
    """Validate the candidate's frozen same-checkpoint relighting guard."""
    regularization = acquisition.get("regularization")
    guard = (
        regularization.get("evaluation_guard")
        if isinstance(regularization, dict)
        else None
    )
    if (
        not isinstance(guard, dict)
        or set(guard) != {"schema", "same_checkpoint", "suites"}
        or guard.get("schema") != FREQUENCY_EVALUATION_GUARD_SCHEMA
        or guard.get("same_checkpoint") is not True
    ):
        raise ValueError(
            f"frequency candidate {profile} has a missing or invalid "
            "same-checkpoint evaluation guard"
        )
    suites = guard.get("suites")
    if not isinstance(suites, dict) or set(suites) != set(EVALUATION_LIGHTING):
        raise ValueError(
            f"frequency candidate {profile} evaluation guard has invalid suites"
        )
    evaluation = acquisition.get("evaluation")
    evaluations = (
        evaluation.get("evaluations") if isinstance(evaluation, dict) else None
    )
    if not isinstance(evaluations, dict):
        raise ValueError(
            f"frequency candidate {profile} has no final evaluation provenance"
        )

    expected_record_keys = {
        "pre_cleanup_mse",
        "post_cleanup_mse",
        "post_minus_pre_mse",
        "pre_cleanup_mean_mse",
        "post_cleanup_mean_mse",
        "post_minus_pre_mean_mse",
        "worsened",
    }
    public_suites = {}
    for lighting in EVALUATION_LIGHTING:
        record = suites[lighting]
        if not isinstance(record, dict) or set(record) != expected_record_keys:
            raise ValueError(
                f"frequency candidate {profile} {lighting} evaluation guard "
                "has an invalid serialized shape"
            )
        before = record.get("pre_cleanup_mse")
        after = record.get("post_cleanup_mse")
        deltas = record.get("post_minus_pre_mse")
        if (
            not isinstance(before, list)
            or not isinstance(after, list)
            or not isinstance(deltas, list)
            or not before
            or len(before) != len(after)
            or len(before) != len(deltas)
            or any(
                not isinstance(value, float) or not math.isfinite(value)
                for values in (before, after, deltas)
                for value in values
            )
            or any(value < 0.0 for values in (before, after) for value in values)
        ):
            raise ValueError(
                f"frequency candidate {profile} {lighting} evaluation guard "
                "has invalid per-case losses"
            )
        expected_deltas = [post - pre for pre, post in zip(before, after)]
        if any(
            recorded != expected
            for recorded, expected in zip(deltas, expected_deltas)
        ):
            raise ValueError(
                f"frequency candidate {profile} {lighting} evaluation guard "
                "has inconsistent per-case arithmetic"
            )
        before_mean = float(np.mean(before))
        after_mean = float(np.mean(after))
        mean_delta = after_mean - before_mean
        if (
            record.get("pre_cleanup_mean_mse") != before_mean
            or record.get("post_cleanup_mean_mse") != after_mean
            or record.get("post_minus_pre_mean_mse") != mean_delta
            or record.get("worsened") is not (after_mean > before_mean)
        ):
            raise ValueError(
                f"frequency candidate {profile} {lighting} evaluation guard "
                "has inconsistent mean or decision provenance"
            )

        final_suite = evaluations.get(lighting)
        final_metrics = (
            final_suite.get("metrics") if isinstance(final_suite, dict) else None
        )
        recorded_count = (
            final_suite.get("count") if isinstance(final_suite, dict) else None
        )
        final_mse = (
            final_metrics.get("mse") if isinstance(final_metrics, dict) else None
        )
        if (
            not isinstance(recorded_count, int)
            or isinstance(recorded_count, bool)
            or recorded_count != len(before)
            or not isinstance(final_mse, (int, float))
            or isinstance(final_mse, bool)
            or not math.isfinite(float(final_mse))
            or float(final_mse) < 0.0
            or not math.isclose(
                float(final_mse),
                after_mean,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"frequency candidate {profile} {lighting} evaluation guard "
                "does not match the final evaluation count/MSE"
            )
        if mean_delta > FREQUENCY_EVALUATION_MEAN_MSE_TOLERANCE:
            raise ValueError(
                f"frequency candidate {profile} {lighting} same-checkpoint "
                "evaluation regressed beyond tolerance: "
                f"{mean_delta:+.9g} > "
                f"{FREQUENCY_EVALUATION_MEAN_MSE_TOLERANCE:.1e}"
            )
        public_suites[lighting] = {
            "case_count": len(before),
            "pre_cleanup_mean_mse": before_mean,
            "post_cleanup_mean_mse": after_mean,
            "post_minus_pre_mean_mse": mean_delta,
            "within_no_regression_tolerance": True,
        }
    return {
        "validated": True,
        "schema": FREQUENCY_EVALUATION_GUARD_SCHEMA,
        "same_checkpoint": True,
        "mean_mse_tolerance": FREQUENCY_EVALUATION_MEAN_MSE_TOLERANCE,
        "suites": public_suites,
    }


def _measure_frequency_cleanup(
    baseline_camera: Path,
    regularized_camera: Path,
    profiles: Sequence[str],
    fit_mask: np.ndarray,
    acquisitions: dict[str, dict[str, dict[str, Any]]],
    contract: dict[str, Any],
) -> dict[str, Any] | None:
    regularization_kind = contract["regularization_kind"]
    if regularization_kind not in {
        "frequency-consensus",
        "frequency-consensus-adaptive",
    }:
        return None
    adaptive_comparison = bool(contract.get("active_frequency_upgrade"))
    if adaptive_comparison != (
        regularization_kind == "frequency-consensus-adaptive"
    ):
        raise ValueError(
            "adaptive frequency cleanup requires an active frequency-consensus "
            "baseline and the controlled adapter transition"
        )
    expected_candidate_identities = (
        {
            _FREQUENCY_ADAPTIVE_IDENTITY,
            _FREQUENCY_ADAPTIVE_V2_IDENTITY,
        }
        if adaptive_comparison
        else {
            _FREQUENCY_IDENTITY,
            _FREQUENCY_V2_IDENTITY,
        }
    )
    candidate_identities = {
        (
            acquisitions["regularized"][profile]["checkpoint_signature"].get(
                "schema"
            ),
            _adapter_identity(
                acquisitions["regularized"][profile]["checkpoint_signature"]
            ),
        )
        for profile in profiles
    }
    if (
        len(candidate_identities) != 1
        or not candidate_identities.issubset(expected_candidate_identities)
    ):
        raise ValueError(
            "frequency cleanup report requires the expected v13 adapter "
            f"identity; found {candidate_identities}"
        )
    candidate_identity = next(iter(candidate_identities))
    adapter_version = (
        "v2"
        if candidate_identity
        in {_FREQUENCY_V2_IDENTITY, _FREQUENCY_ADAPTIVE_V2_IDENTITY}
        else "v1"
    )
    if fit_mask.ndim != 2 or not np.any(fit_mask):
        raise ValueError("frequency cleanup report requires a nonempty 2D fit mask")

    aggregate = {
        "fit_map_entries": 0,
        "updated_entries": 0,
        "consensus_entries": 0,
        "updated_union_pixels": 0,
        "distance_baseline_sum": 0.0,
        "distance_candidate_sum": 0.0,
        "within_tolerance": 0,
        "source_alignment_count": 0,
        "source_alignment_sum": 0.0,
        "source_alignment_max": 0.0,
        "target_alignment_count": 0,
        "target_alignment_sum": 0.0,
        "target_alignment_max": 0.0,
        "outside_union_count": 0,
        "outside_union_sum": 0.0,
        "outside_union_max": 0.0,
        "outside_per_map_count": 0,
        "outside_per_map_sum": 0.0,
        "outside_per_map_max": 0.0,
    }
    per_profile = {}
    common_strength = None
    for profile in profiles:
        baseline_regularization = acquisitions["baseline"][profile].get(
            "regularization"
        )
        if adaptive_comparison:
            _validate_active_frequency_baseline_state(baseline_regularization)
        else:
            _validate_frequency_baseline_state(baseline_regularization)
        acquisition = acquisitions["regularized"][profile]
        if acquisition.get("schema") != "ictpolarreal.end2end-disney.v13":
            raise ValueError(
                "frequency candidate acquisition schema does not match its "
                "checkpoint/adapter identity"
            )
        regularization = acquisition.get("regularization", {})
        if regularization.get("cleanup_applied") is not True:
            raise ValueError("frequency candidate did not complete its post-fit update")
        if regularization.get("data_objective_only") is not True:
            raise ValueError("frequency candidate must keep data fitting unchanged")
        evaluation_guard = _validate_frequency_evaluation_guard(
            acquisition,
            profile=profile,
        )
        settings = regularization.get("settings")
        stage_plan = regularization.get("stage_plan")
        if adaptive_comparison:
            _validate_frequency_adaptive_settings(settings)
        else:
            _validate_frequency_settings(settings)
        if (
            not isinstance(stage_plan, dict)
            or stage_plan.get("enabled") is not True
            or stage_plan.get("cleanup_optimizer_steps") != 0
            or stage_plan.get("post_fit_updates") != 1
            or stage_plan.get("data_fit_steps") != acquisition.get("steps")
            or stage_plan.get("detector_after_data_step")
            != stage_plan.get("data_fit_steps")
        ):
            raise ValueError("frequency candidate has an invalid post-fit plan")
        try:
            weight = float(regularization["weight"])
            reference = float(stage_plan["weight_reference"])
            strength = float(stage_plan["strength"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("frequency candidate has invalid strength provenance") from exc
        expected_strength = min(weight / reference, 1.0) if reference > 0.0 else -1.0
        if (
            not math.isfinite(reference)
            or reference <= 0.0
            or not math.isfinite(strength)
            or not 0.0 < strength <= 1.0
            or not math.isclose(
                strength,
                expected_strength,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("frequency candidate has invalid strength provenance")
        if common_strength is None:
            common_strength = strength
        elif not math.isclose(common_strength, strength, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("frequency cleanup strength differs across profiles")

        diagnostic = regularization.get("cleanup_diagnostic")
        if not isinstance(diagnostic, dict) or diagnostic.get("schema") != (
            FREQUENCY_DIAGNOSTIC_SCHEMA
        ):
            raise ValueError("frequency candidate has invalid cleanup diagnostics")
        artifact = _load_frequency_frozen_artifact(
            regularized_camera / "material" / profile,
            acquisition,
            fit_mask.shape,
        )
        baseline_artifact = None
        if adaptive_comparison:
            baseline_acquisition = acquisitions["baseline"][profile]
            if baseline_acquisition.get("schema") != (
                "ictpolarreal.end2end-disney.v13"
            ):
                raise ValueError(
                    "active frequency baseline acquisition schema does not match "
                    "its checkpoint/adapter identity"
                )
            baseline_artifact = _load_frequency_frozen_artifact(
                baseline_camera / "material" / profile,
                baseline_acquisition,
                fit_mask.shape,
            )
            _validate_frequency_artifact_upgrade_pair(
                baseline_artifact,
                artifact,
                profile=profile,
            )
            _validate_frequency_evaluation_guard(
                baseline_acquisition,
                profile=profile,
            )
        if not np.array_equal(artifact["fit_foreground"], fit_mask):
            raise ValueError(
                "frequency frozen fit foreground does not match the comparison mask"
            )
        if not math.isclose(
            float(diagnostic.get("strength", -1.0)),
            artifact["strength"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("frequency diagnostic strength differs from artifact")
        if (
            diagnostic.get("updated_entries")
            != artifact["total_updated_entries"]
            or diagnostic.get("consensus_entries")
            != artifact["total_consensus_entries"]
        ):
            raise ValueError("frequency diagnostic totals differ from artifact")
        diagnostic_maps = diagnostic.get("maps")
        if not isinstance(diagnostic_maps, dict) or set(diagnostic_maps) != set(
            SCALAR_MAPS
        ):
            raise ValueError("frequency diagnostic is missing per-map counts")

        fit_pixels = int(np.count_nonzero(fit_mask))
        spatial_union = np.zeros(fit_mask.shape, dtype=bool)
        map_differences = {}
        profile_accumulator = {
            "updated": 0,
            "consensus": 0,
            "baseline_distance": 0.0,
            "candidate_distance": 0.0,
            "within_tolerance": 0,
            "source_sum": 0.0,
            "source_max": 0.0,
            "target_sum": 0.0,
            "target_max": 0.0,
            "outside_per_map_count": 0,
            "outside_per_map_sum": 0.0,
            "outside_per_map_max": 0.0,
        }
        per_map = {}
        full_precision_before = []
        diagnostic_moved_total = 0
        for map_name in SCALAR_MAPS:
            baseline_values = _read_scalar_map(
                baseline_camera / "material" / profile / "maps" / f"{map_name}.png"
            )
            candidate_values = _read_scalar_map(
                regularized_camera
                / "material"
                / profile
                / "maps"
                / f"{map_name}.png"
            )
            if baseline_values.shape != fit_mask.shape or candidate_values.shape != (
                fit_mask.shape
            ):
                raise ValueError("frequency cleanup maps do not match the fit-mask shape")
            arrays = artifact["maps"][map_name]
            artifact_update_mask = arrays["mask"]
            update_mask = (
                arrays["target"] != arrays["fixed_target"]
                if adaptive_comparison
                else artifact_update_mask
            )
            consensus_mask = arrays["consensus_mask"]
            if np.any(update_mask & ~fit_mask):
                raise ValueError(
                    f"frequency frozen updates fall outside the fit mask: {map_name}"
                )
            updated_count = int(np.count_nonzero(update_mask))
            artifact_consensus_count = int(np.count_nonzero(consensus_mask))
            consensus_count = int(
                np.count_nonzero(consensus_mask & update_mask)
                if adaptive_comparison
                else artifact_consensus_count
            )
            artifact_updated_count = int(np.count_nonzero(artifact_update_mask))
            diagnostic_entry = diagnostic_maps.get(map_name)
            if (
                not isinstance(diagnostic_entry, dict)
                or diagnostic_entry.get("updated_entries")
                != artifact_updated_count
                or diagnostic_entry.get("consensus_entries")
                != artifact_consensus_count
            ):
                raise ValueError(
                    f"frequency diagnostic counts differ for {map_name}"
                )
            moved_entries = diagnostic_entry.get("moved_entries")
            if (
                not isinstance(moved_entries, int)
                or isinstance(moved_entries, bool)
                or not 0 <= moved_entries <= artifact_updated_count
            ):
                raise ValueError(
                    f"frequency diagnostic moved count is invalid for {map_name}"
                )
            diagnostic_moved_total += moved_entries

            baseline_reference = (
                arrays["fixed_target"]
                if adaptive_comparison
                else arrays["source"]
            )
            source_alignment = np.abs(
                baseline_values[fit_mask] - baseline_reference[fit_mask]
            )
            target_alignment = np.abs(
                candidate_values[fit_mask] - arrays["target"][fit_mask]
            )
            source_max = float(source_alignment.max(initial=0.0))
            target_max = float(target_alignment.max(initial=0.0))
            tolerance = EXPORT_QUANTIZATION_TOLERANCE + 1e-7
            if source_max > tolerance:
                raise ValueError(
                    "baseline export does not match its frozen reference for "
                    f"{map_name}"
                )
            if target_max > tolerance:
                raise ValueError(
                    f"candidate export does not match frozen target for {map_name}"
                )

            baseline_distance = np.abs(
                baseline_values[update_mask] - arrays["target"][update_mask]
            )
            candidate_distance = np.abs(
                candidate_values[update_mask] - arrays["target"][update_mask]
            )
            full_before = np.abs(
                arrays["source"][artifact_update_mask]
                - arrays["target"][artifact_update_mask]
            )
            full_precision_before.append(full_before)
            expected_before = (
                float(full_before.mean()) if artifact_updated_count else 0.0
            )
            try:
                recorded_before = float(diagnostic_entry["mean_distance_before"])
                recorded_after = float(diagnostic_entry["mean_distance_after"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"frequency diagnostic distances are invalid for {map_name}"
                ) from exc
            if (
                not math.isfinite(recorded_before)
                or not math.isfinite(recorded_after)
                or recorded_after < 0.0
                or not math.isclose(
                    recorded_before,
                    expected_before,
                    rel_tol=1e-6,
                    abs_tol=1e-7,
                )
            ):
                raise ValueError(
                    f"frequency diagnostic distances differ for {map_name}"
                )
            within_tolerance = int(
                np.count_nonzero(
                    candidate_distance <= EXPORT_QUANTIZATION_TOLERANCE + 1e-12
                )
            )
            exported_change = np.abs(candidate_values - baseline_values)
            map_differences[map_name] = exported_change
            outside = fit_mask & ~update_mask
            outside_values = exported_change[outside]
            outside_sum = float(outside_values.sum(dtype=np.float64))
            outside_max = float(outside_values.max(initial=0.0))
            baseline_sum = float(baseline_distance.sum(dtype=np.float64))
            candidate_sum = float(candidate_distance.sum(dtype=np.float64))
            source_sum = float(source_alignment.sum(dtype=np.float64))
            target_sum = float(target_alignment.sum(dtype=np.float64))
            outside_count = int(outside_values.size)
            spatial_union |= update_mask
            per_map[map_name] = {
                "updated_entries": updated_count,
                "updated_fraction_of_fit_foreground": (
                    float(updated_count / fit_pixels) if fit_pixels else 0.0
                ),
                "consensus_entries": consensus_count,
                "consensus_fraction_of_updated_entries": (
                    float(consensus_count / updated_count)
                    if updated_count
                    else None
                ),
                "mean_absolute_distance_to_frozen_target": {
                    "baseline": _optional_mean(baseline_sum, updated_count),
                    "candidate": _optional_mean(candidate_sum, updated_count),
                },
                "within_export_tolerance_of_frozen_target": {
                    "count": within_tolerance,
                    "fraction": (
                        float(within_tolerance / updated_count)
                        if updated_count
                        else None
                    ),
                },
                "baseline_export_alignment_to_frozen_reference": {
                    "reference": (
                        "frequency_consensus_v1_fixed_target"
                        if adaptive_comparison
                        else "data_fit_source"
                    ),
                    "mean_absolute": _optional_mean(source_sum, fit_pixels),
                    "maximum_absolute": source_max,
                },
                "candidate_export_alignment_to_frozen_target": {
                    "mean_absolute": _optional_mean(target_sum, fit_pixels),
                    "maximum_absolute": target_max,
                },
                "exported_change_outside_own_update_mask": {
                    "entry_count": outside_count,
                    "mean_absolute": _optional_mean(outside_sum, outside_count),
                    "maximum_absolute": outside_max,
                },
            }
            profile_accumulator["updated"] += updated_count
            profile_accumulator["consensus"] += consensus_count
            profile_accumulator["baseline_distance"] += baseline_sum
            profile_accumulator["candidate_distance"] += candidate_sum
            profile_accumulator["within_tolerance"] += within_tolerance
            profile_accumulator["source_sum"] += source_sum
            profile_accumulator["source_max"] = max(
                profile_accumulator["source_max"], source_max
            )
            profile_accumulator["target_sum"] += target_sum
            profile_accumulator["target_max"] = max(
                profile_accumulator["target_max"], target_max
            )
            profile_accumulator["outside_per_map_count"] += outside_count
            profile_accumulator["outside_per_map_sum"] += outside_sum
            profile_accumulator["outside_per_map_max"] = max(
                profile_accumulator["outside_per_map_max"], outside_max
            )

        full_before_values = (
            np.concatenate([value for value in full_precision_before if value.size])
            if any(value.size for value in full_precision_before)
            else np.empty(0, dtype=np.float32)
        )
        expected_diagnostic_before = (
            float(full_before_values.mean()) if full_before_values.size else 0.0
        )
        if (
            diagnostic.get("moved_entries") != diagnostic_moved_total
            or not math.isclose(
                float(diagnostic.get("mean_distance_before", -1.0)),
                expected_diagnostic_before,
                rel_tol=1e-6,
                abs_tol=1e-7,
            )
        ):
            raise ValueError("frequency aggregate diagnostic differs from artifact")
        try:
            diagnostic_after = float(diagnostic["mean_distance_after"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("frequency aggregate diagnostic is invalid") from exc
        if not math.isfinite(diagnostic_after) or diagnostic_after < 0.0:
            raise ValueError("frequency aggregate diagnostic is invalid")

        outside_union = fit_mask & ~spatial_union
        outside_union_count = int(np.count_nonzero(outside_union) * len(SCALAR_MAPS))
        outside_union_sum = float(
            sum(
                difference[outside_union].sum(dtype=np.float64)
                for difference in map_differences.values()
            )
        )
        outside_union_max = float(
            max(
                (
                    difference[outside_union].max(initial=0.0)
                    for difference in map_differences.values()
                ),
                default=0.0,
            )
        )
        updated = profile_accumulator["updated"]
        union_pixels = int(np.count_nonzero(spatial_union))
        profile_fit_entries = fit_pixels * len(SCALAR_MAPS)
        per_profile[profile] = {
            "annotation_labels": (
                {"updated": "changed", "consensus": "strong changed"}
                if adaptive_comparison
                else {"updated": "updates", "consensus": "consensus"}
            ),
            "artifact": {
                "validated": True,
                "path": str(artifact["path"]),
                "sha256": artifact["sha256"],
                "bytes": artifact["bytes"],
                "created_after_step": artifact["created_after_step"],
                **(
                    {
                        "consensus_entry_semantics": artifact[
                            "consensus_entry_semantics"
                        ]
                    }
                    if adaptive_comparison
                    else {}
                ),
            },
            "fit_foreground_pixels": fit_pixels,
            "evaluation_guard": evaluation_guard,
            "strength": artifact["strength"],
            "edge_threshold_full_precision": artifact[
                "edge_threshold_full_precision"
            ],
            "edge_threshold_png_quantized": artifact[
                "edge_threshold_png_quantized"
            ],
            "edge_protected_pixels": int(np.count_nonzero(artifact["edge_protected"])),
            "update_safe_pixels": int(np.count_nonzero(artifact["update_safe"])),
            "updated": {
                "map_entries": updated,
                "map_entry_fraction": float(updated / profile_fit_entries),
                "spatial_union_pixels": union_pixels,
                "spatial_union_fraction": float(union_pixels / fit_pixels),
            },
            "consensus": {
                "map_entries": profile_accumulator["consensus"],
                "fraction_of_updated_entries": (
                    float(profile_accumulator["consensus"] / updated)
                    if updated
                    else None
                ),
            },
            "mean_absolute_distance_to_frozen_target": {
                "baseline": _optional_mean(
                    profile_accumulator["baseline_distance"], updated
                ),
                "candidate": _optional_mean(
                    profile_accumulator["candidate_distance"], updated
                ),
            },
            "within_export_tolerance_of_frozen_target": {
                "count": profile_accumulator["within_tolerance"],
                "fraction": (
                    float(profile_accumulator["within_tolerance"] / updated)
                    if updated
                    else None
                ),
            },
            "baseline_export_alignment_to_frozen_reference": {
                "reference": (
                    "frequency_consensus_v1_fixed_target"
                    if adaptive_comparison
                    else "data_fit_source"
                ),
                "mean_absolute": _optional_mean(
                    profile_accumulator["source_sum"], profile_fit_entries
                ),
                "maximum_absolute": profile_accumulator["source_max"],
            },
            "candidate_export_alignment_to_frozen_target": {
                "mean_absolute": _optional_mean(
                    profile_accumulator["target_sum"], profile_fit_entries
                ),
                "maximum_absolute": profile_accumulator["target_max"],
            },
            "exported_change_outside_spatial_update_union": {
                "entry_count": outside_union_count,
                "mean_absolute": _optional_mean(outside_union_sum, outside_union_count),
                "maximum_absolute": outside_union_max,
            },
            "exported_change_outside_per_map_update_masks": {
                "entry_count": profile_accumulator["outside_per_map_count"],
                "mean_absolute": _optional_mean(
                    profile_accumulator["outside_per_map_sum"],
                    profile_accumulator["outside_per_map_count"],
                ),
                "maximum_absolute": profile_accumulator["outside_per_map_max"],
            },
            "maps": per_map,
        }
        aggregate["fit_map_entries"] += profile_fit_entries
        aggregate["updated_entries"] += updated
        aggregate["consensus_entries"] += profile_accumulator["consensus"]
        aggregate["updated_union_pixels"] += union_pixels
        aggregate["distance_baseline_sum"] += profile_accumulator[
            "baseline_distance"
        ]
        aggregate["distance_candidate_sum"] += profile_accumulator[
            "candidate_distance"
        ]
        aggregate["within_tolerance"] += profile_accumulator["within_tolerance"]
        aggregate["source_alignment_count"] += profile_fit_entries
        aggregate["source_alignment_sum"] += profile_accumulator["source_sum"]
        aggregate["source_alignment_max"] = max(
            aggregate["source_alignment_max"], profile_accumulator["source_max"]
        )
        aggregate["target_alignment_count"] += profile_fit_entries
        aggregate["target_alignment_sum"] += profile_accumulator["target_sum"]
        aggregate["target_alignment_max"] = max(
            aggregate["target_alignment_max"], profile_accumulator["target_max"]
        )
        aggregate["outside_union_count"] += outside_union_count
        aggregate["outside_union_sum"] += outside_union_sum
        aggregate["outside_union_max"] = max(
            aggregate["outside_union_max"], outside_union_max
        )
        aggregate["outside_per_map_count"] += profile_accumulator[
            "outside_per_map_count"
        ]
        aggregate["outside_per_map_sum"] += profile_accumulator[
            "outside_per_map_sum"
        ]
        aggregate["outside_per_map_max"] = max(
            aggregate["outside_per_map_max"],
            profile_accumulator["outside_per_map_max"],
        )

    updated = aggregate["updated_entries"]
    public_aggregate = {
        "strength": common_strength,
        "export_quantization_tolerance": EXPORT_QUANTIZATION_TOLERANCE,
        "updated": {
            "map_entries": updated,
            "map_entry_fraction": float(updated / aggregate["fit_map_entries"]),
            "spatial_union_pixels_across_profiles": aggregate[
                "updated_union_pixels"
            ],
        },
        "consensus": {
            "map_entries": aggregate["consensus_entries"],
            "fraction_of_updated_entries": (
                float(aggregate["consensus_entries"] / updated)
                if updated
                else None
            ),
        },
        "mean_absolute_distance_to_frozen_target": {
            "baseline": _optional_mean(aggregate["distance_baseline_sum"], updated),
            "candidate": _optional_mean(
                aggregate["distance_candidate_sum"], updated
            ),
        },
        "within_export_tolerance_of_frozen_target": {
            "count": aggregate["within_tolerance"],
            "fraction": (
                float(aggregate["within_tolerance"] / updated)
                if updated
                else None
            ),
        },
        "baseline_export_alignment_to_frozen_reference": {
            "reference": (
                "frequency_consensus_v1_fixed_target"
                if adaptive_comparison
                else "data_fit_source"
            ),
            "mean_absolute": _optional_mean(
                aggregate["source_alignment_sum"],
                aggregate["source_alignment_count"],
            ),
            "maximum_absolute": aggregate["source_alignment_max"],
        },
        "candidate_export_alignment_to_frozen_target": {
            "mean_absolute": _optional_mean(
                aggregate["target_alignment_sum"],
                aggregate["target_alignment_count"],
            ),
            "maximum_absolute": aggregate["target_alignment_max"],
        },
        "exported_change_outside_spatial_update_union": {
            "entry_count": aggregate["outside_union_count"],
            "mean_absolute": _optional_mean(
                aggregate["outside_union_sum"], aggregate["outside_union_count"]
            ),
            "maximum_absolute": aggregate["outside_union_max"],
        },
        "exported_change_outside_per_map_update_masks": {
            "entry_count": aggregate["outside_per_map_count"],
            "mean_absolute": _optional_mean(
                aggregate["outside_per_map_sum"],
                aggregate["outside_per_map_count"],
            ),
            "maximum_absolute": aggregate["outside_per_map_max"],
        },
    }
    return {
        "schema": "ictpolarreal.frequency-cleanup-comparison.v1",
        "available": True,
        "comparison_mode": (
            f"frequency-consensus-{adapter_version}-to-adaptive-{adapter_version}"
            if adaptive_comparison
            else f"data-fit-source-to-frequency-consensus-{adapter_version}"
        ),
        "adapter_version": adapter_version,
        "target_policy_schema": (
            FREQUENCY_ADAPTIVE_BUNDLE_SCHEMA
            if adaptive_comparison
            else FREQUENCY_BUNDLE_SCHEMA
        ),
        "regularizer_label": _display_regularizer(regularization_kind),
        "update_count_semantics": (
            "map entries whose adaptive target differs from the frozen fixed-policy target"
            if adaptive_comparison
            else "map entries whose fixed post-fit target differs from the data-fit source"
        ),
        "consensus_count_semantics": (
            "entries whose adaptive target differs from the frozen fixed-policy target and whose "
            "strong-policy target also differs from the data-fit source"
            if adaptive_comparison
            else "cross-map consensus entries"
        ),
        "scope": (
            "Artifact integrity, frozen-reference/target alignment, update coverage, and "
            "outside-mask changes are measured from hash-validated float32 frozen "
            "state and 8-bit exported maps. Target formulas and stored mask semantics "
            "are recomputed, but the report does not independently recompute the "
            "weighted medians. These diagnostics do not "
            "establish material quality or texture preservation."
        ),
        "presentation": {
            "profile": "olat" if "olat" in profiles else profiles[0],
            "crop_box_xyxy": list(FREQUENCY_DETAIL_CROP_BOX),
            "crop_size_pixels": 96,
            "source_pixels_per_output_pixel": 1,
            "resampling": "none",
            "maps": list(FREQUENCY_DETAIL_MAPS),
        },
        "aggregate": public_aggregate,
        "profiles": per_profile,
    }


def _masked_case_png_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    if prediction.shape != reference.shape or prediction.ndim != 3:
        raise ValueError("case PNG prediction and reference shapes differ")
    if mask.shape != reference.shape[:2] or not np.any(mask):
        raise ValueError("case PNG metric mask is empty or has the wrong shape")
    selected_prediction = prediction[mask]
    selected_reference = reference[mask]
    target_mean = float(np.mean(selected_reference))
    if not math.isfinite(target_mean) or target_mean <= 1e-8:
        raise ValueError("case PNG reference has no finite masked intensity")
    prediction_mean = float(np.mean(selected_prediction))
    luma = np.asarray([0.2989, 0.5870, 0.1140], dtype=np.float32)
    prediction_luma = selected_prediction @ luma
    reference_luma = selected_reference @ luma
    if (
        float(np.std(prediction_luma)) <= 1e-8
        or float(np.std(reference_luma)) <= 1e-8
    ):
        correlation = 0.0
    else:
        correlation = float(
            np.corrcoef(prediction_luma, reference_luma)[0, 1]
        )
    metrics = {
        "psnr": image_psnr(prediction, reference, mask),
        "ssim_global": image_ssim_global(prediction, reference, mask),
        "mean_intensity_ratio": prediction_mean / target_mean,
        "luminance_correlation": correlation,
    }
    nonfinite = [name for name, value in metrics.items() if not math.isfinite(value)]
    if nonfinite:
        raise ValueError(f"case PNG metrics are non-finite: {nonfinite}")
    return metrics


def _scalar_error_heatmap_u8(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Reproduce the acquisition writer's fixed-scale RGB error PNG exactly."""
    prediction = np.clip(np.asarray(prediction, dtype=np.float32), 0.0, 1.0)
    reference = np.clip(np.asarray(reference, dtype=np.float32), 0.0, 1.0)
    if (
        prediction.shape != reference.shape
        or prediction.ndim != 3
        or prediction.shape[-1] < 3
    ):
        raise ValueError(
            "error heatmap inputs must be equal-shape RGB images: "
            f"{prediction.shape} vs {reference.shape}"
        )
    error = np.mean(
        np.abs(prediction[..., :3] - reference[..., :3]),
        axis=-1,
    )
    normalized = np.clip(error / ACQUISITION_ERROR_HEATMAP_MAX, 0.0, 1.0)
    heatmap = np.stack(
        [
            np.interp(
                normalized,
                ACQUISITION_ERROR_HEATMAP_POSITIONS,
                ACQUISITION_ERROR_HEATMAP_COLORS[:, channel],
            )
            for channel in range(3)
        ],
        axis=-1,
    ).astype(np.float32)
    return (np.clip(heatmap, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _validate_stored_error_heatmap(
    path: Path,
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, str]:
    if not path.is_file():
        raise FileNotFoundError(f"case PNG error heatmap is missing: {path}")
    expected = _scalar_error_heatmap_u8(prediction, reference)
    with Image.open(path) as image:
        stored = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if stored.shape != expected.shape or not np.array_equal(stored, expected):
        differing = (
            int(np.count_nonzero(np.any(stored != expected, axis=-1)))
            if stored.shape == expected.shape
            else None
        )
        raise ValueError(
            "stored case PNG error heatmap is stale or differs from the exact "
            f"acquisition visualization: {label}, differing_pixels={differing}"
        )
    return expected, _array_sha256(expected)


def _select_case_png_examples(
    rows: Sequence[dict[str, Any]],
    *,
    lighting: str,
    profiles: Sequence[str],
) -> dict[str, Any]:
    expected_profiles = tuple(profiles)
    per_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("lighting") == lighting:
            per_case.setdefault(str(row["case_id"]), []).append(row)
    if not per_case:
        raise ValueError(f"no validated case PNG rows for {lighting}")
    ranked = []
    for case_id, case_rows in per_case.items():
        observed_profiles = tuple(sorted(str(row["profile"]) for row in case_rows))
        if observed_profiles != tuple(sorted(expected_profiles)):
            raise ValueError(
                f"{lighting}/{case_id} does not have exactly one validated row "
                f"for every requested profile: {observed_profiles}"
            )
        mean_delta = float(
            np.mean([float(row["delta"]["psnr"]) for row in case_rows])
        )
        if not math.isfinite(mean_delta):
            raise ValueError(f"{lighting}/{case_id} has non-finite mean PSNR delta")
        ranked.append(
            {
                "case_id": case_id,
                "mean_profile_delta_psnr_db": mean_delta,
                "profile_count": len(case_rows),
                "per_profile_delta_psnr_db": {
                    str(row["profile"]): float(row["delta"]["psnr"])
                    for row in sorted(case_rows, key=lambda value: value["profile"])
                },
            }
        )
    ranked.sort(
        key=lambda entry: (
            entry["mean_profile_delta_psnr_db"],
            entry["case_id"],
        )
    )
    median_index = (len(ranked) - 1) // 2
    median = {**ranked[median_index], "rank_worst_to_best": median_index + 1}
    worst = {**ranked[0], "rank_worst_to_best": 1}
    return {
        "selection_metric": "mean_profile_delta_psnr_db",
        "median_policy": "lower_order_statistic_after_delta_then_case_id_sort",
        "case_count": len(ranked),
        "median": median,
        "worst": worst,
    }


def _recorded_suite_case_ids(
    suite: dict[str, Any],
    lighting: str,
) -> tuple[str, ...] | None:
    if lighting == "olat" and "frame_ids" in suite:
        values = suite.get("frame_ids")
        if (
            not isinstance(values, list)
            or not values
            or not all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                for value in values
            )
        ):
            raise ValueError("recorded OLAT frame IDs are invalid")
        identifiers = tuple(f"{value:06d}" for value in values)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("recorded OLAT frame IDs contain duplicates")
        return tuple(sorted(identifiers))
    for key in ("condition_ids", "case_ids"):
        if key not in suite:
            continue
        values = suite.get(key)
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) and value for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"recorded {lighting} {key} are invalid")
        return tuple(sorted(values))
    return None


def _recorded_checkpoint_case_ids(
    acquisition: dict[str, Any],
    lighting: str,
) -> tuple[str, ...] | None:
    if lighting != "hdri":
        return None
    signature = acquisition.get("checkpoint_signature")
    if not isinstance(signature, dict):
        return None
    key = "hdri_evaluation_condition_ids"
    if key not in signature:
        return None
    values = signature.get(key)
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(value, str) and value for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("recorded HDRI checkpoint condition IDs are invalid")
    return tuple(sorted(values))


def _recorded_suite_representative(
    suite: dict[str, Any],
    lighting: str,
) -> str | None:
    representative = suite.get("representative")
    if representative is None:
        return None
    if not isinstance(representative, dict):
        raise ValueError(f"recorded {lighting} representative is invalid")
    key = "frame_id" if lighting == "olat" else "condition_id"
    if key not in representative:
        return None
    value = representative.get(key)
    if lighting == "olat":
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError("recorded OLAT representative frame ID is invalid")
        return f"{value:06d}"
    if not isinstance(value, str) or not value:
        raise ValueError("recorded HDRI representative condition ID is invalid")
    return value


def _recorded_hdri_asset_case_ids(camera: Path) -> tuple[str, ...] | None:
    manifest_path = camera / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = (
        manifest.get("evaluation", {}).get("assets", {}).get("conditions")
        if isinstance(manifest, dict)
        else None
    )
    if relative is None:
        return None
    if not isinstance(relative, str) or not relative:
        raise ValueError("camera manifest has an invalid HDRI conditions path")
    conditions_path = (camera / relative).resolve()
    if camera.resolve() not in conditions_path.parents or not conditions_path.is_file():
        raise ValueError("camera HDRI conditions asset is missing or escapes its root")
    payload = json.loads(conditions_path.read_text(encoding="utf-8"))
    conditions = payload.get("conditions") if isinstance(payload, dict) else None
    if not isinstance(conditions, list):
        raise ValueError("camera HDRI conditions asset has no condition list")
    identifiers = tuple(
        sorted(
            entry["condition_id"]
            for entry in conditions
            if isinstance(entry, dict) and entry.get("split") == "heldout"
        )
    )
    if (
        not identifiers
        or len(set(identifiers)) != len(identifiers)
        or not all(isinstance(identifier, str) and identifier for identifier in identifiers)
    ):
        raise ValueError("camera HDRI conditions asset has invalid held-out IDs")
    recorded_count = payload.get("evaluation_conditions")
    if (
        not isinstance(recorded_count, int)
        or isinstance(recorded_count, bool)
        or recorded_count != len(identifiers)
    ):
        raise ValueError("camera HDRI condition count differs from held-out IDs")
    return identifiers


def _collect_case_png_metrics(
    baseline_camera: Path,
    regularized_camera: Path,
    acquisitions: dict[str, dict[str, dict[str, Any]]],
    profiles: Sequence[str],
    mask: np.ndarray,
) -> dict[str, Any]:
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError("case PNG evaluation requires a nonempty 2D mask")
    rows = []
    case_ids_by_lighting = {}
    acquisition_provenance = {}
    baseline_hdri_asset_ids = _recorded_hdri_asset_case_ids(baseline_camera)
    regularized_hdri_asset_ids = _recorded_hdri_asset_case_ids(
        regularized_camera
    )
    if baseline_hdri_asset_ids != regularized_hdri_asset_ids:
        raise ValueError("A/B HDRI condition assets record different case IDs")
    for lighting in EVALUATION_LIGHTING:
        baseline_cases_dir = baseline_camera / "evaluation" / lighting / "cases"
        regularized_cases_dir = (
            regularized_camera / "evaluation" / lighting / "cases"
        )
        baseline_case_ids = tuple(
            sorted(path.name for path in baseline_cases_dir.iterdir() if path.is_dir())
        )
        regularized_case_ids = tuple(
            sorted(
                path.name for path in regularized_cases_dir.iterdir() if path.is_dir()
            )
        )
        if not baseline_case_ids or baseline_case_ids != regularized_case_ids:
            raise ValueError(
                f"{lighting} case PNG sets are empty or differ between A/B trees"
            )
        if (
            lighting == "hdri"
            and baseline_hdri_asset_ids is not None
            and baseline_case_ids != baseline_hdri_asset_ids
        ):
            raise ValueError(
                "HDRI case PNG identifiers differ from the recorded conditions asset"
            )
        case_ids_by_lighting[lighting] = list(baseline_case_ids)
        camera_representatives = {
            "baseline": _camera_representative_case(
                baseline_camera,
                lighting,
            ),
            "regularized": _camera_representative_case(
                regularized_camera,
                lighting,
            ),
        }
        for variant, camera_representative in camera_representatives.items():
            if camera_representative not in baseline_case_ids:
                raise ValueError(
                    f"{lighting} {variant} camera-summary representative is "
                    "absent from case PNG metrics"
                )
        acquisition_provenance[lighting] = {}
        observed_count = len(baseline_case_ids)
        for variant in ("baseline", "regularized"):
            acquisition_provenance[lighting][variant] = {}
            for profile in profiles:
                acquisition = acquisitions[variant][profile]
                evaluation = acquisition.get("evaluation")
                evaluations = (
                    evaluation.get("evaluations")
                    if isinstance(evaluation, dict)
                    else None
                )
                suite = (
                    evaluations.get(lighting)
                    if isinstance(evaluations, dict)
                    else None
                )
                if not isinstance(suite, dict):
                    raise ValueError(
                        "acquisition is missing a recorded evaluation suite: "
                        f"variant={variant}, profile={profile}, lighting={lighting}"
                    )
                recorded_count = suite.get("count")
                if (
                    not isinstance(recorded_count, int)
                    or isinstance(recorded_count, bool)
                    or recorded_count <= 0
                ):
                    raise ValueError(
                        "acquisition recorded evaluation count is invalid: "
                        f"variant={variant}, profile={profile}, lighting={lighting}"
                    )
                if recorded_count != observed_count:
                    raise ValueError(
                        "case PNG count differs from acquisition recorded evaluation "
                        f"count: variant={variant}, profile={profile}, "
                        f"lighting={lighting}, observed={observed_count}, "
                        f"recorded={recorded_count}"
                    )
                suite_ids = _recorded_suite_case_ids(suite, lighting)
                checkpoint_ids = _recorded_checkpoint_case_ids(
                    acquisition,
                    lighting,
                )
                exact_ids = suite_ids or checkpoint_ids
                if (
                    suite_ids is not None
                    and checkpoint_ids is not None
                    and suite_ids != checkpoint_ids
                ):
                    raise ValueError(
                        f"recorded {lighting} suite/checkpoint identifiers disagree"
                    )
                if exact_ids is not None and exact_ids != baseline_case_ids:
                    raise ValueError(
                        f"{lighting} case PNG identifiers differ from acquisition "
                        f"provenance: variant={variant}, profile={profile}"
                    )
                recorded_representative = _recorded_suite_representative(
                    suite,
                    lighting,
                )
                if (
                    recorded_representative is not None
                    and recorded_representative not in baseline_case_ids
                ):
                    raise ValueError(
                        f"{lighting} acquisition representative is absent from case "
                        f"PNG metrics: variant={variant}, profile={profile}"
                    )
                if (
                    recorded_representative is not None
                    and recorded_representative != camera_representatives[variant]
                ):
                    raise ValueError(
                        f"{lighting} acquisition representative differs from its own "
                        f"camera summary: variant={variant}, profile={profile}"
                    )
                acquisition_provenance[lighting][variant][profile] = {
                    "recorded_count": recorded_count,
                    "exact_case_ids_available": exact_ids is not None,
                    "recorded_representative": recorded_representative,
                    "camera_summary_representative": camera_representatives[
                        variant
                    ],
                }
        for case_id in baseline_case_ids:
            baseline_case = baseline_cases_dir / case_id
            regularized_case = regularized_cases_dir / case_id
            baseline_reference_path = baseline_case / "reference.png"
            regularized_reference_path = regularized_case / "reference.png"
            if not baseline_reference_path.is_file() or not (
                regularized_reference_path.is_file()
            ):
                raise FileNotFoundError(
                    f"case PNG reference is missing: {lighting}/{case_id}"
                )
            baseline_reference = _read_rgb_map(baseline_reference_path)
            regularized_reference = _read_rgb_map(regularized_reference_path)
            if baseline_reference.shape[:2] != mask.shape:
                raise ValueError(
                    f"case PNG reference shape differs from fit mask: {lighting}/{case_id}"
                )
            if not np.array_equal(baseline_reference, regularized_reference):
                raise ValueError(
                    f"case PNG references differ between A/B trees: {lighting}/{case_id}"
                )
            reference_hash = _file_sha256(baseline_reference_path)
            for profile in profiles:
                relative_prediction = (
                    Path("evaluation")
                    / lighting
                    / "cases"
                    / case_id
                    / "predictions"
                    / f"{profile}.png"
                )
                baseline_prediction_path = baseline_camera / relative_prediction
                regularized_prediction_path = regularized_camera / relative_prediction
                if not baseline_prediction_path.is_file() or not (
                    regularized_prediction_path.is_file()
                ):
                    raise FileNotFoundError(
                        "case PNG prediction is missing: "
                        f"{lighting}/{case_id}/{profile}"
                    )
                baseline_prediction = _read_rgb_map(baseline_prediction_path)
                regularized_prediction = _read_rgb_map(
                    regularized_prediction_path
                )
                relative_error = (
                    Path("evaluation")
                    / lighting
                    / "cases"
                    / case_id
                    / "errors"
                    / f"{profile}.png"
                )
                baseline_error_path = baseline_camera / relative_error
                regularized_error_path = regularized_camera / relative_error
                _, baseline_error_pixel_hash = _validate_stored_error_heatmap(
                    baseline_error_path,
                    baseline_prediction,
                    baseline_reference,
                    label=f"baseline/{lighting}/{case_id}/{profile}",
                )
                _, regularized_error_pixel_hash = _validate_stored_error_heatmap(
                    regularized_error_path,
                    regularized_prediction,
                    baseline_reference,
                    label=f"regularized/{lighting}/{case_id}/{profile}",
                )
                baseline_metrics = _masked_case_png_metrics(
                    baseline_prediction,
                    baseline_reference,
                    mask,
                )
                regularized_metrics = _masked_case_png_metrics(
                    regularized_prediction,
                    baseline_reference,
                    mask,
                )
                rows.append(
                    {
                        "lighting": lighting,
                        "case_id": case_id,
                        "profile": profile,
                        "display_roles": [],
                        "files": {
                            "reference": str(
                                baseline_reference_path.relative_to(baseline_camera)
                            ),
                            "reference_sha256": reference_hash,
                            "baseline_prediction": str(relative_prediction),
                            "baseline_prediction_sha256": _file_sha256(
                                baseline_prediction_path
                            ),
                            "regularized_prediction": str(relative_prediction),
                            "regularized_prediction_sha256": _file_sha256(
                                regularized_prediction_path
                            ),
                            "baseline_error": str(relative_error),
                            "baseline_error_sha256": _file_sha256(
                                baseline_error_path
                            ),
                            "baseline_error_pixel_sha256": (
                                baseline_error_pixel_hash
                            ),
                            "regularized_error": str(relative_error),
                            "regularized_error_sha256": _file_sha256(
                                regularized_error_path
                            ),
                            "regularized_error_pixel_sha256": (
                                regularized_error_pixel_hash
                            ),
                        },
                        "baseline": baseline_metrics,
                        "regularized": regularized_metrics,
                        "delta": {
                            name: regularized_metrics[name] - baseline_metrics[name]
                            for name in baseline_metrics
                        },
                    }
                )
    expected_rows = sum(len(case_ids) for case_ids in case_ids_by_lighting.values()) * len(
        profiles
    )
    if len(rows) != expected_rows:
        raise ValueError("case PNG metric coverage is incomplete")
    selected_cases = {
        lighting: _select_case_png_examples(
            rows,
            lighting=lighting,
            profiles=profiles,
        )
        for lighting in EVALUATION_LIGHTING
    }
    for row in rows:
        selection = selected_cases[row["lighting"]]
        roles = []
        if row["case_id"] == selection["median"]["case_id"]:
            roles.append("median_psnr_delta")
        if row["case_id"] == selection["worst"]["case_id"]:
            roles.append("worst_psnr_delta")
        row["display_roles"] = roles
    return {
        "schema": "ictpolarreal.case-png-evaluation.v2",
        "mask_pixels": int(np.count_nonzero(mask)),
        "profiles": list(profiles),
        "case_ids": case_ids_by_lighting,
        "selected_cases": selected_cases,
        "acquisition_provenance": acquisition_provenance,
        "validated_reference_files": sum(
            len(case_ids) for case_ids in case_ids_by_lighting.values()
        ),
        "validated_prediction_files": 2 * len(rows),
        "validated_error_files": 2 * len(rows),
        "error_heatmap_validation": {
            "schema": "ictpolarreal.scalar-error-heatmap.v1",
            "reducer": "mean_absolute_rgb",
            "maximum_error": ACQUISITION_ERROR_HEATMAP_MAX,
            "colormap_positions": ACQUISITION_ERROR_HEATMAP_POSITIONS.tolist(),
            "colormap_rgb_u8": np.floor(
                ACQUISITION_ERROR_HEATMAP_COLORS * 255.0 + 0.5
            )
            .astype(np.uint8)
            .tolist(),
            "stored_png_validation": "pixel_exact",
            "report_panels": "recomputed_from_validated_reference_and_prediction",
        },
        "presentation": {
            "overview_case": "median_psnr_delta",
            "detail_cases": ["median_psnr_delta", "worst_psnr_delta"],
            "hdri_lighting_thumbnail_fit": "contain_letterbox_no_aspect_distortion",
        },
        "rows": rows,
    }


def _collect_evaluation_metrics(
    acquisitions: dict[str, dict[str, dict[str, Any]]],
    profiles: Sequence[str],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    required = (
        "psnr",
        "ssim_global",
        "mean_intensity_ratio",
        "luminance_correlation",
    )
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
            for variant, metrics in output[profile][lighting].items():
                missing = [name for name in required if name not in metrics]
                nonfinite = [
                    name
                    for name in required
                    if name in metrics and not math.isfinite(metrics[name])
                ]
                if missing or nonfinite:
                    raise ValueError(
                        "relighting evaluation metrics must be finite and complete: "
                        f"profile={profile}, lighting={lighting}, variant={variant}, "
                        f"missing={missing}, nonfinite={nonfinite}"
                    )
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
    guide_regions: dict[str, dict[str, Any]],
    guide_diagnostic: dict[str, Any],
    impulse_cleanup: dict[str, Any] | None,
    frequency_cleanup: dict[str, Any] | None,
    case_png_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    def aggregate(metric: str, variant: str) -> float:
        return float(
            np.mean(
                [
                    map_metrics[profile][name][variant][metric]
                    for profile in profiles
                    for name in SCALAR_MAPS
                ]
            )
        )

    spatial_metrics = {}
    for metric in (
        "neighbor_variation",
        "median5_residual_mae",
        "quiet_region_neighbor_variation",
        "quiet_region_median5_residual_mae",
        "quiet_region_median5_residual_outlier_fraction_gt_0p05",
    ):
        baseline_value = aggregate(metric, "baseline")
        regularized_value = aggregate(metric, "regularized")
        spatial_metrics[metric] = {
            "baseline": baseline_value,
            "regularized": regularized_value,
            "relative_change_fraction": _relative_change_fraction(
                baseline_value, regularized_value
            ),
        }
    baseline_edge = aggregate("guide_edge_gradient_magnitude", "baseline")
    regularized_edge = aggregate("guide_edge_gradient_magnitude", "regularized")
    spatial_metrics["guide_edge_gradient_magnitude"] = {
        "baseline": baseline_edge,
        "regularized": regularized_edge,
        "magnitude_ratio": _ratio(baseline_edge, regularized_edge),
    }
    meaningful_maps = []
    for profile in profiles:
        for map_name in SCALAR_MAPS:
            comparison = map_metrics[profile][map_name]["comparison"]
            if not comparison["meaningful_guide_edge_gradient"]:
                continue
            meaningful_maps.append(
                {
                    "profile": profile,
                    "map": map_name,
                    "gradient_magnitude_ratio": comparison[
                        "guide_edge_gradient_magnitude_ratio"
                    ],
                    "signed_gradient_cosine": comparison[
                        "guide_edge_signed_gradient_cosine"
                    ],
                }
            )

    def preservation_distribution(metric: str) -> dict[str, Any]:
        valid = [
            entry for entry in meaningful_maps if entry.get(metric) is not None
        ]
        if not valid:
            return {"median": None, "worst": None}
        values = [float(entry[metric]) for entry in valid]
        worst = min(valid, key=lambda entry: float(entry[metric]))
        return {
            "median": float(np.median(values)),
            "worst": {
                "value": float(worst[metric]),
                "profile": worst["profile"],
                "map": worst["map"],
            },
        }

    guide_edge_preservation = {
        "interpretation": (
            "Gradient magnitude ratio measures retained strength only; signed "
            "gradient cosine measures per-pair directional correspondence. "
            "These exported-map diagnostics do not establish that all material "
            "texture is preserved."
        ),
        "meaningful_baseline_gradient_min": MEANINGFUL_EDGE_GRADIENT_MIN,
        "meaningful_map_count": len(meaningful_maps),
        "per_map": meaningful_maps,
        "gradient_magnitude_ratio": preservation_distribution(
            "gradient_magnitude_ratio"
        ),
        "signed_gradient_cosine": preservation_distribution(
            "signed_gradient_cosine"
        ),
    }
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
    scalar_map_cleanup_qualification = (
        _build_frequency_diagnostic_gates(
            map_metrics,
            evaluation_metrics,
            case_png_metrics,
            profiles,
        )
        if frequency_cleanup is not None
        and frequency_cleanup.get("available") is True
        else None
    )
    if (
        frequency_cleanup is not None
        and scalar_map_cleanup_qualification is not None
    ):
        frequency_cleanup["scalar_map_cleanup_qualification"] = (
            scalar_map_cleanup_qualification
        )
    selected_cases = case_png_metrics["selected_cases"]
    return {
        "schema": "ictpolarreal.regularization-comparison.v4",
        "title": (
            f"{_camera_display_name(baseline_camera.name)} · "
            "Regularization comparison"
        ),
        "baseline": str(baseline_camera),
        "regularized": str(regularized_camera),
        "labels": labels,
        "profiles": list(profiles),
        "maps": list(SCALAR_MAPS),
        "mask_source": mask_source,
        "guide_diagnostic": guide_diagnostic,
        "guide_regions": guide_regions,
        "comparison_contract": contract,
        "aggregate_map_spatial_statistics": spatial_metrics,
        "guide_edge_preservation": guide_edge_preservation,
        "impulse_cleanup": impulse_cleanup,
        "frequency_cleanup": frequency_cleanup,
        "scalar_map_cleanup_qualification": scalar_map_cleanup_qualification,
        "case_png_evaluation": case_png_metrics,
        "per_map_spatial_statistics": map_metrics,
        "aggregate_evaluation": evaluation_summary,
        "displayed_evaluation_cases": {
            lighting: {
                "overview": {
                    "role": "median_psnr_delta",
                    **selected_cases[lighting]["median"],
                },
                "detail": {
                    "median_psnr_delta": selected_cases[lighting]["median"],
                    "worst_psnr_delta": selected_cases[lighting]["worst"],
                },
            }
            for lighting in EVALUATION_LIGHTING
        },
        "artifacts": {
            "overview": "overview.png",
            "metrics": "metrics.csv",
            "material": {profile: f"material/{profile}.png" for profile in profiles},
            "frequency_diagnostics": (
                {
                    "hotspot_1to1": "material/frequency_hotspot_1to1.png",
                    "fullmaps_1to1": "material/frequency_fullmaps_1to1.png",
                    **(
                        {
                            "adaptive_cleanup_evidence": (
                                "material/adaptive_cleanup_evidence.png"
                            )
                        }
                        if isinstance(
                            frequency_cleanup.get("adaptive_cleanup_evidence"),
                            dict,
                        )
                        else {}
                    ),
                }
                if frequency_cleanup is not None
                and frequency_cleanup.get("available") is True
                else None
            ),
            "evaluation": {
                lighting: {
                    "sheet": f"evaluation/{lighting}.png",
                    "median_case": selected_cases[lighting]["median"]["case_id"],
                    "worst_case": selected_cases[lighting]["worst"]["case_id"],
                    "shown_case_metric_source": "recomputed_masked_case_pngs",
                    "error_panels": (
                        "recomputed_after_pixel_exact_stored_error_validation"
                    ),
                    "lighting_thumbnail_fit": (
                        "contain_letterbox_no_aspect_distortion"
                        if lighting == "hdri"
                        else None
                    ),
                }
                for lighting in EVALUATION_LIGHTING
            },
        },
    }


def _build_frequency_diagnostic_gates(
    map_metrics: dict[str, Any],
    evaluation_metrics: dict[str, Any],
    case_png_metrics: dict[str, Any] | None,
    profiles: Sequence[str],
) -> dict[str, Any]:
    def upper_gate(observed: float | None, threshold: float) -> dict[str, Any]:
        finite = observed is not None and math.isfinite(float(observed))
        return {
            "observed": observed,
            "operator": "<=",
            "threshold": threshold,
            "meets_threshold": bool(finite and float(observed) <= threshold),
        }

    def lower_gate(observed: float | None, threshold: float) -> dict[str, Any]:
        finite = observed is not None and math.isfinite(float(observed))
        return {
            "observed": observed,
            "operator": ">=",
            "threshold": threshold,
            "meets_threshold": bool(finite and float(observed) >= threshold),
        }

    def relative_reduction_gate(
        baseline: float | None,
        regularized: float | None,
        threshold: float = 0.20,
    ) -> dict[str, Any]:
        zero_epsilon = 1e-12
        valid = (
            baseline is not None
            and regularized is not None
            and math.isfinite(float(baseline))
            and math.isfinite(float(regularized))
            and float(baseline) >= 0.0
            and float(regularized) >= 0.0
        )
        denominator_policy = "invalid_input"
        observed = None
        if valid:
            baseline_value = float(baseline)
            regularized_value = float(regularized)
            if baseline_value > zero_epsilon:
                observed = float(
                    (baseline_value - regularized_value) / baseline_value
                )
                denominator_policy = "baseline"
            elif regularized_value <= zero_epsilon:
                # A perfect active baseline leaves no measurable room for the
                # candidate to earn the positive relative-improvement gate.
                observed = 0.0
                denominator_policy = "zero_baseline_and_candidate_no_reduction"
            else:
                # Keep a zero-baseline regression finite so report composition
                # can record a failed gate instead of aborting the comparison.
                observed = float(-regularized_value / zero_epsilon)
                denominator_policy = "zero_baseline_epsilon_regression"
        gate = lower_gate(observed, threshold)
        gate.update(
            {
                "quantity": "relative_reduction_from_baseline",
                "baseline_value": baseline,
                "regularized_value": regularized,
                "zero_baseline_epsilon": zero_epsilon,
                "denominator_policy": denominator_policy,
            }
        )
        return gate

    def aggregate_delta_gate(
        baseline: float,
        regularized: float,
        threshold: float = 0.0,
    ) -> dict[str, Any]:
        gate = lower_gate(float(regularized - baseline), threshold)
        gate.update(
            {
                "quantity": "regularized_minus_baseline",
                "baseline_value": float(baseline),
                "regularized_value": float(regularized),
            }
        )
        return gate

    edge_rows = [
        (
            map_metrics[profile][name]["baseline"][
                "guide_edge_gradient_magnitude"
            ],
            map_metrics[profile][name]["regularized"][
                "guide_edge_gradient_magnitude"
            ],
            map_metrics[profile][name]["comparison"],
        )
        for profile in profiles
        for name in SCALAR_MAPS
    ]
    edge_ratios = []
    edge_cosines = []
    zero_baseline_nonzero_candidate_rows = 0
    for baseline_edge, regularized_edge, comparison in edge_rows:
        baseline_edge = float(baseline_edge)
        regularized_edge = float(regularized_edge)
        both_near_zero = (
            abs(baseline_edge) <= EDGE_NUMERICAL_ZERO
            and abs(regularized_edge) <= EDGE_NUMERICAL_ZERO
        )
        ratio = comparison["guide_edge_gradient_magnitude_ratio"]
        if ratio is None:
            if both_near_zero:
                ratio = 1.0
            elif abs(baseline_edge) <= EDGE_NUMERICAL_ZERO:
                ratio = regularized_edge / EDGE_NUMERICAL_ZERO
                zero_baseline_nonzero_candidate_rows += 1
            else:
                ratio = 0.0
        cosine = comparison["guide_edge_signed_gradient_cosine"]
        if cosine is None:
            cosine = 1.0 if both_near_zero else 0.0
        edge_ratios.append(float(ratio))
        edge_cosines.append(float(cosine))
    baseline_band_energy = sum(
        map_metrics[profile][name]["baseline"]["guide_textured_band3_8_energy"]
        for profile in profiles
        for name in SCALAR_MAPS
    )
    regularized_band_energy = sum(
        map_metrics[profile][name]["regularized"][
            "guide_textured_band3_8_energy"
        ]
        for profile in profiles
        for name in SCALAR_MAPS
    )
    aggregate_band_ratio = (
        float(math.sqrt(regularized_band_energy / baseline_band_energy))
        if baseline_band_energy > 1e-18
        else None
    )
    worst_low_frequency_mae = max(
        float(
            map_metrics[profile][name]["comparison"][
                "gaussian3_low_frequency_mae"
            ]
        )
        for profile in profiles
        for name in SCALAR_MAPS
    )
    hotspot = {}
    for map_name in ("subsurface", "specular"):
        hotspot[map_name] = {
            variant: (
                map_metrics.get("olat", {})
                .get(map_name, {})
                .get(variant, {})
                .get("fixed_hotspot_median5_residual_outlier_fraction_gt_0p05")
            )
            for variant in ("baseline", "regularized")
        }
    candidate_evaluations = [
        evaluation_metrics[profile][lighting]["regularized"]
        for profile in profiles
        for lighting in EVALUATION_LIGHTING
    ]
    aggregate_relighting_baseline_psnr = float(
        np.mean(
            [
                float(evaluation_metrics[profile][lighting]["baseline"]["psnr"])
                for profile in profiles
                for lighting in EVALUATION_LIGHTING
            ]
        )
    )
    aggregate_relighting_regularized_psnr = float(
        np.mean(
            [
                float(evaluation_metrics[profile][lighting]["regularized"]["psnr"])
                for profile in profiles
                for lighting in EVALUATION_LIGHTING
            ]
        )
    )
    aggregate_relighting_baseline_ssim = float(
        np.mean(
            [
                float(
                    evaluation_metrics[profile][lighting]["baseline"][
                        "ssim_global"
                    ]
                )
                for profile in profiles
                for lighting in EVALUATION_LIGHTING
            ]
        )
    )
    aggregate_relighting_regularized_ssim = float(
        np.mean(
            [
                float(
                    evaluation_metrics[profile][lighting]["regularized"][
                        "ssim_global"
                    ]
                )
                for profile in profiles
                for lighting in EVALUATION_LIGHTING
            ]
        )
    )
    quiet_pixels = {
        variant: sum(
            int(map_metrics[profile][name][variant]["quiet_region_pixels"])
            for profile in profiles
            for name in SCALAR_MAPS
        )
        for variant in ("baseline", "regularized")
    }
    quiet_outliers = {
        variant: sum(
            int(
                map_metrics[profile][name][variant][
                    "quiet_region_median5_residual_outlier_count_gt_0p05"
                ]
            )
            for profile in profiles
            for name in SCALAR_MAPS
        )
        for variant in ("baseline", "regularized")
    }
    aggregate_quiet_outlier_fraction = {
        variant: (
            float(quiet_outliers[variant] / quiet_pixels[variant])
            if quiet_pixels[variant]
            else None
        )
        for variant in ("baseline", "regularized")
    }
    edge_gate = {
        "observed_minimum": min(edge_ratios) if edge_ratios else None,
        "observed_maximum": max(edge_ratios) if edge_ratios else None,
        "operator": "within_inclusive_range",
        "minimum": 0.95,
        "maximum": 1.05,
        "near_zero_epsilon": EDGE_NUMERICAL_ZERO,
        "zero_baseline_nonzero_candidate_rows": (
            zero_baseline_nonzero_candidate_rows
        ),
        "meets_threshold": bool(
            edge_ratios
            and min(edge_ratios) >= 0.95
            and max(edge_ratios) <= 1.05
        ),
    }
    intensity_ratios = [
        float(metrics["mean_intensity_ratio"]) for metrics in candidate_evaluations
    ]
    intensity_gate = {
        "observed_minimum": min(intensity_ratios),
        "observed_maximum": max(intensity_ratios),
        "operator": "within_inclusive_range",
        "minimum": 0.70,
        "maximum": 1.20,
        "meets_threshold": bool(
            min(intensity_ratios) >= 0.70 and max(intensity_ratios) <= 1.20
        ),
    }
    map_improvement_gates = {
        "aggregate_quiet_outlier_relative_reduction": relative_reduction_gate(
            aggregate_quiet_outlier_fraction["baseline"],
            aggregate_quiet_outlier_fraction["regularized"],
        ),
        "olat_fixed_hotspot_subsurface_outlier_relative_reduction": (
            relative_reduction_gate(
                hotspot["subsurface"]["baseline"],
                hotspot["subsurface"]["regularized"],
            )
        ),
        "olat_fixed_hotspot_specular_outlier_relative_reduction": (
            relative_reduction_gate(
                hotspot["specular"]["baseline"],
                hotspot["specular"]["regularized"],
            )
        ),
    }
    material_safeguard_gates = {
        "candidate_aggregate_quiet_outlier_fraction": upper_gate(
            aggregate_quiet_outlier_fraction["regularized"],
            0.092,
        ),
        "candidate_olat_fixed_hotspot_subsurface_outlier_fraction": upper_gate(
            hotspot["subsurface"]["regularized"], 0.30
        ),
        "candidate_olat_fixed_hotspot_specular_outlier_fraction": upper_gate(
            hotspot["specular"]["regularized"], 0.24
        ),
        "aggregate_guide_textured_band3_8_amplitude_ratio": lower_gate(
            aggregate_band_ratio, 0.90
        ),
        "guide_edge_gradient_magnitude_ratio": edge_gate,
        "guide_edge_signed_gradient_cosine": lower_gate(
            min(edge_cosines) if edge_cosines else None,
            0.98,
        ),
        "gaussian3_low_frequency_mae": upper_gate(
            worst_low_frequency_mae,
            0.01,
        ),
    }
    material_gates = {
        **map_improvement_gates,
        **material_safeguard_gates,
    }
    relighting_change_gates = {
        "aggregate_relighting_delta_psnr_db": aggregate_delta_gate(
            aggregate_relighting_baseline_psnr,
            aggregate_relighting_regularized_psnr,
        ),
        "aggregate_relighting_delta_ssim_global": aggregate_delta_gate(
            aggregate_relighting_baseline_ssim,
            aggregate_relighting_regularized_ssim,
        ),
    }
    absolute_sanity_gates = {
        "candidate_absolute_psnr_db": lower_gate(
            min(float(metrics["psnr"]) for metrics in candidate_evaluations),
            18.0,
        ),
        "candidate_absolute_ssim_global": lower_gate(
            min(float(metrics["ssim_global"]) for metrics in candidate_evaluations),
            0.70,
        ),
        "candidate_absolute_mean_intensity_ratio": intensity_gate,
        "candidate_absolute_luminance_correlation": lower_gate(
            min(
                float(metrics["luminance_correlation"])
                for metrics in candidate_evaluations
            ),
            0.80,
        ),
    }
    if (
        not isinstance(case_png_metrics, dict)
        or case_png_metrics.get("schema") != "ictpolarreal.case-png-evaluation.v2"
        or case_png_metrics.get("profiles") != list(profiles)
        or not isinstance(case_png_metrics.get("rows"), list)
        or not case_png_metrics["rows"]
    ):
        raise ValueError("scalar-map cleanup qualification is missing case PNG metrics")
    case_rows = case_png_metrics["rows"]
    candidate_case_metrics = [row["regularized"] for row in case_rows]
    case_deltas = [row["delta"] for row in case_rows]

    def percentile(values: Sequence[float], value: float) -> float:
        result = float(np.percentile(np.asarray(values, dtype=np.float64), value))
        if not math.isfinite(result):
            raise ValueError("case PNG percentile is non-finite")
        return result

    case_psnr = [float(metrics["psnr"]) for metrics in candidate_case_metrics]
    case_ssim = [
        float(metrics["ssim_global"]) for metrics in candidate_case_metrics
    ]
    case_ratio = [
        float(metrics["mean_intensity_ratio"])
        for metrics in candidate_case_metrics
    ]
    case_correlation = [
        float(metrics["luminance_correlation"])
        for metrics in candidate_case_metrics
    ]
    case_psnr_delta = [float(delta["psnr"]) for delta in case_deltas]
    case_ssim_delta = [float(delta["ssim_global"]) for delta in case_deltas]
    case_worst_ratio_gate = {
        "observed_minimum": min(case_ratio),
        "observed_maximum": max(case_ratio),
        "operator": "within_inclusive_range",
        "minimum": 0.55,
        "maximum": 1.35,
        "meets_threshold": bool(
            min(case_ratio) >= 0.55 and max(case_ratio) <= 1.35
        ),
    }
    case_percentile_ratio_gate = {
        "observed_minimum": percentile(case_ratio, 10.0),
        "observed_maximum": percentile(case_ratio, 90.0),
        "operator": "within_inclusive_range",
        "minimum": 0.65,
        "maximum": 1.25,
        "meets_threshold": bool(
            percentile(case_ratio, 10.0) >= 0.65
            and percentile(case_ratio, 90.0) <= 1.25
        ),
    }
    case_png_gates = {
        "case_png_worst_psnr_db": lower_gate(min(case_psnr), 15.5),
        "case_png_p10_psnr_db": lower_gate(percentile(case_psnr, 10.0), 17.0),
        "case_png_worst_ssim_global": lower_gate(min(case_ssim), 0.50),
        "case_png_p10_ssim_global": lower_gate(
            percentile(case_ssim, 10.0), 0.60
        ),
        "case_png_worst_mean_intensity_ratio": case_worst_ratio_gate,
        "case_png_p10_p90_mean_intensity_ratio": case_percentile_ratio_gate,
        "case_png_worst_luminance_correlation": lower_gate(
            min(case_correlation), 0.60
        ),
        "case_png_p10_luminance_correlation": lower_gate(
            percentile(case_correlation, 10.0), 0.70
        ),
        "case_png_worst_delta_psnr_db": lower_gate(
            min(case_psnr_delta), -0.25
        ),
        "case_png_p10_delta_psnr_db": lower_gate(
            percentile(case_psnr_delta, 10.0), -0.05
        ),
        "case_png_worst_delta_ssim_global": lower_gate(
            min(case_ssim_delta), -0.005
        ),
        "case_png_p10_delta_ssim_global": lower_gate(
            percentile(case_ssim_delta, 10.0), -0.001
        ),
    }
    gates = {
        **material_gates,
        **relighting_change_gates,
        **absolute_sanity_gates,
        **case_png_gates,
    }
    for gate_name, gate in gates.items():
        observed = (
            (gate.get("observed_minimum"), gate.get("observed_maximum"))
            if gate.get("operator") == "within_inclusive_range"
            else (gate.get("observed"),)
        )
        if any(
            value is None or not math.isfinite(float(value)) for value in observed
        ):
            raise ValueError(
                "scalar-map cleanup qualification gate has a non-finite "
                f"observation: {gate_name}"
            )
    met = sum(bool(gate["meets_threshold"]) for gate in gates.values())
    numeric_thresholds_met = met == len(gates)
    coverage_complete = (
        len(profiles) == len(QUALIFICATION_PROFILES)
        and set(profiles) == set(QUALIFICATION_PROFILES)
    )
    qualification_status = (
        "INCOMPLETE"
        if not coverage_complete
        else "PASS"
        if numeric_thresholds_met
        else "FAIL"
    )
    return {
        "schema": "ictpolarreal.scalar-map-cleanup-qualification.v1",
        "qualification_name": "scalar_map_cleanup_under_safeguards",
        "crop_box_xyxy": list(FREQUENCY_DETAIL_CROP_BOX),
        "gate_count": len(gates),
        "material_gate_count": len(material_gates),
        "map_improvement_gate_count": len(map_improvement_gates),
        "material_safeguard_gate_count": len(material_safeguard_gates),
        "relighting_change_gate_count": len(relighting_change_gates),
        "absolute_sanity_gate_count": len(absolute_sanity_gates),
        "case_png_gate_count": len(case_png_gates),
        "thresholds_met": met,
        "all_numeric_thresholds_met": numeric_thresholds_met,
        "all_thresholds_met": numeric_thresholds_met and coverage_complete,
        "qualification_status": qualification_status,
        "qualification_coverage": {
            "required_profiles": list(QUALIFICATION_PROFILES),
            "evaluated_profiles": list(profiles),
            "complete": coverage_complete,
            "validated_case_rows": len(case_rows),
            "validated_prediction_files": case_png_metrics[
                "validated_prediction_files"
            ],
            "validated_error_files": case_png_metrics["validated_error_files"],
            "selected_cases": case_png_metrics["selected_cases"],
        },
        "pass_meaning": (
            "PASS means the required scalar-map outlier reductions were measured "
            "under absolute, texture, aggregate-relighting, and per-case tail "
            "safeguards. PASS is not evidence of reconstruction or relighting "
            "improvement."
        ),
        "interpretation": (
            "Three improvement gates require at least 20% relative reduction in "
            "aggregate quiet outliers and the fixed OLAT hotspot's subsurface and "
            "specular outliers. Absolute and texture safeguards remain active. "
            "Acquisition-aggregate PSNR and SSIM must be neutral or better, while "
            "twelve masked case-PNG safeguards retain worst-case and tenth-percentile "
            "tail limits. PASS additionally requires exactly OLAT, HDRI, and MIX fit "
            "profiles. This qualification establishes scalar-map cleanup under "
            "safeguards; it does not establish reconstruction improvement. It "
            "does not establish overall material quality or preservation of all "
            "texture."
        ),
        "gates": gates,
    }


def _relative_change_fraction(
    baseline: float, regularized: float
) -> float | None:
    if abs(baseline) <= 1e-12:
        return None
    return float((regularized - baseline) / baseline)


def _ratio(baseline: float, regularized: float) -> float | None:
    if abs(baseline) <= 1e-12:
        return None
    return float(regularized / baseline)


def _format_ratio(value: float | None, *, digits: int = 0) -> str:
    return "n/a" if value is None else f"{100.0 * value:.{digits}f}%"


def _format_cosine(value: float | None, *, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _format_relative_change(value: float | None, *, digits: int = 1) -> str:
    return "n/a" if value is None else f"{100.0 * value:+.{digits}f}%"


def _camera_display_name(name: str) -> str:
    match = re.fullmatch(r"cam(?:era)?[_-]?(\d+)", name.strip(), flags=re.IGNORECASE)
    if match:
        return f"Camera {int(match.group(1)):02d}"
    return name.strip() or "Camera"


def _camera_representative_case(camera: Path, lighting: str) -> str:
    summary = json.loads(
        (camera / "evaluation" / "summary.json").read_text(encoding="utf-8")
    )
    try:
        case_id = summary["suites"][lighting]["representative_case"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{lighting} camera summary has no representative case"
        ) from exc
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"{lighting} camera summary representative is invalid")
    return case_id


def _case_png_row(
    case_png_metrics: dict[str, Any],
    *,
    lighting: str,
    case_id: str,
    profile: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in case_png_metrics.get("rows", [])
        if row.get("lighting") == lighting
        and row.get("case_id") == case_id
        and row.get("profile") == profile
    ]
    if len(matches) != 1:
        raise ValueError(
            "displayed evaluation case must have exactly one validated PNG row: "
            f"{lighting}/{case_id}/{profile}, found={len(matches)}"
        )
    return matches[0]


def _contain_image(
    image: Image.Image,
    size: tuple[int, int],
    *,
    fill: tuple[int, int, int] = (18, 18, 18),
) -> Image.Image:
    """Aspect-preserving contain fit used for panoramic HDRI thumbnails."""
    target_width, target_height = size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("contained image target size must be positive")
    source = image.convert("RGB")
    if source.width <= 0 or source.height <= 0:
        raise ValueError("contained image source size must be positive")
    scale = min(target_width / source.width, target_height / source.height)
    resized_size = (
        max(1, min(target_width, int(round(source.width * scale)))),
        max(1, min(target_height, int(round(source.height * scale)))),
    )
    resized = source.resize(resized_size, Image.Resampling.LANCZOS)
    tile = Image.new("RGB", size, fill)
    tile.paste(
        resized,
        (
            (target_width - resized.width) // 2,
            (target_height - resized.height) // 2,
        ),
    )
    return tile


def _evaluation_case_panels(
    baseline_camera: Path,
    regularized_camera: Path,
    *,
    lighting: str,
    case_id: str,
    profile: str,
) -> list[tuple[str, Image.Image, bool]]:
    baseline_case = baseline_camera / "evaluation" / lighting / "cases" / case_id
    regularized_case = (
        regularized_camera / "evaluation" / lighting / "cases" / case_id
    )
    reference = _read_rgb_map(baseline_case / "reference.png")
    baseline_prediction = _read_rgb_map(
        baseline_case / "predictions" / f"{profile}.png"
    )
    regularized_prediction = _read_rgb_map(
        regularized_case / "predictions" / f"{profile}.png"
    )
    panels: list[tuple[str, Image.Image, bool]] = []
    if lighting == "hdri":
        with Image.open(baseline_case / "lighting.png") as image:
            panels.append(("Lighting", image.convert("RGB"), True))
    panels.extend(
        [
            (
                "Reference",
                Image.fromarray(
                    np.floor(np.clip(reference, 0.0, 1.0) * 255.0 + 0.5).astype(
                        np.uint8
                    )
                ),
                False,
            ),
            (
                "Baseline",
                Image.fromarray(
                    np.floor(
                        np.clip(baseline_prediction, 0.0, 1.0) * 255.0 + 0.5
                    ).astype(np.uint8)
                ),
                False,
            ),
            (
                "Regularized",
                Image.fromarray(
                    np.floor(
                        np.clip(regularized_prediction, 0.0, 1.0) * 255.0 + 0.5
                    ).astype(np.uint8)
                ),
                False,
            ),
            (
                "Baseline error",
                Image.fromarray(
                    _scalar_error_heatmap_u8(baseline_prediction, reference)
                ),
                False,
            ),
            (
                "Regularized error",
                Image.fromarray(
                    _scalar_error_heatmap_u8(regularized_prediction, reference)
                ),
                False,
            ),
        ]
    )
    return panels


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
    # map grid.  A 300 px gutter clipped the full regularizer label at
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
            mean_quiet_variation = np.mean(
                [
                    metrics[name][variant]["quiet_region_neighbor_variation"]
                    for name in map_group
                ]
            )
            variant_label = labels[variant].replace(" · ", "\n", 1)
            draw.multiline_text(
                (24, y + 24),
                (
                    f"{variant_label}\n"
                    f"quiet-region variation {mean_quiet_variation:.4f}"
                ),
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
                comparison = metrics[map_name]["comparison"]
                if variant == "baseline":
                    caption = (
                        f"quiet {values['quiet_region_neighbor_variation']:.3f} · "
                        f"|grad| {values['guide_edge_gradient_magnitude']:.3f}"
                    )
                else:
                    caption = (
                        f"|grad| {_format_ratio(comparison['guide_edge_gradient_magnitude_ratio'])} "
                        f"· cos {_format_cosine(comparison['guide_edge_signed_gradient_cosine'])}"
                    )
                _draw_centered_text(
                    draw,
                    (x, y + tile_height, tile_width, caption_height),
                    caption,
                    _font(20),
                    "white",
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _write_evaluation_comparison(
    baseline_camera: Path,
    regularized_camera: Path,
    profiles: Sequence[str],
    lighting: str,
    metrics: dict[str, Any],
    case_png_metrics: dict[str, Any],
    output: Path,
) -> None:
    selection = case_png_metrics["selected_cases"][lighting]
    selected = (
        ("Median PSNR-delta case", selection["median"]),
        ("Worst PSNR-delta case", selection["worst"]),
    )
    first_case_id = str(selection["median"]["case_id"])
    case = baseline_camera / "evaluation" / lighting / "cases" / first_case_id
    with Image.open(case / "reference.png") as reference:
        tile_width, tile_height = reference.size
    columns = [
        "Reference",
        "Baseline",
        "Regularized",
        "Baseline error",
        "Regularized error",
    ]
    include_lighting = lighting == "hdri"
    if include_lighting:
        columns.insert(0, "Lighting")
    left = 360
    top = 236
    case_header_height = 78
    section_gap = 24
    section_height = case_header_height + tile_height * len(profiles)
    width = max(1200, left + tile_width * len(columns))
    height = top + section_height * len(selected) + section_gap
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (24, 18),
        f"{lighting.upper()} held-out relighting",
        font=_font(45, bold=True),
        fill="white",
    )
    aggregate_baseline_psnr = float(
        np.mean(
            [
                metrics[profile][lighting]["baseline"]["psnr"]
                for profile in profiles
            ]
        )
    )
    aggregate_regularized_psnr = float(
        np.mean(
            [metrics[profile][lighting]["regularized"]["psnr"] for profile in profiles]
        )
    )
    aggregate_baseline_ssim = float(
        np.mean(
            [
                metrics[profile][lighting]["baseline"]["ssim_global"]
                for profile in profiles
            ]
        )
    )
    aggregate_regularized_ssim = float(
        np.mean(
            [
                metrics[profile][lighting]["regularized"]["ssim_global"]
                for profile in profiles
            ]
        )
    )
    aggregate_label = (
        f"Acquisition aggregate across {len(profiles)} fit profile(s) · "
        f"PSNR {aggregate_baseline_psnr:.2f} → {aggregate_regularized_psnr:.2f} "
        f"({aggregate_regularized_psnr - aggregate_baseline_psnr:+.3f} dB) · "
        f"SSIM {aggregate_baseline_ssim:.3f} → {aggregate_regularized_ssim:.3f} "
        f"({aggregate_regularized_ssim - aggregate_baseline_ssim:+.4f})"
    )
    draw.text(
        (24, 72),
        aggregate_label,
        font=_font_for_width(
            draw,
            aggregate_label,
            max_width=width - 48,
            preferred_size=27,
            minimum_size=18,
            bold=True,
        ),
        fill="white",
    )
    draw.text(
        (24, 112),
        (
            "Shown-case metrics are recomputed from masked PNGs. Error panels are "
            f"recomputed at fixed mean-|RGB| scale 0–{ACQUISITION_ERROR_HEATMAP_MAX:g}."
        ),
        font=_font(24),
        fill=(210, 210, 210),
    )
    for column, name in enumerate(columns):
        _draw_centered_text(
            draw,
            (left + column * tile_width, 190, tile_width, 38),
            name,
            _font(22, bold=True),
            "white",
        )
    for section_index, (role_label, selected_case) in enumerate(selected):
        section_y = top + section_index * section_height
        if section_index:
            section_y += section_gap
        case_id = str(selected_case["case_id"])
        case_label = (
            f"{role_label} · {case_id} · mean shown-case ΔPSNR across profiles "
            f"{selected_case['mean_profile_delta_psnr_db']:+.3f} dB · "
            f"rank {selected_case['rank_worst_to_best']}/{selection['case_count']}"
        )
        draw.rounded_rectangle(
            (18, section_y + 4, width - 18, section_y + case_header_height - 4),
            radius=10,
            fill=(29, 34, 42),
        )
        draw.text(
            (30, section_y + 22),
            case_label,
            font=_font_for_width(
                draw,
                case_label,
                max_width=width - 60,
                preferred_size=25,
                minimum_size=16,
                bold=True,
            ),
            fill=(230, 235, 242),
        )
        for row_index, profile in enumerate(profiles):
            y = section_y + case_header_height + row_index * tile_height
            case_metrics = _case_png_row(
                case_png_metrics,
                lighting=lighting,
                case_id=case_id,
                profile=profile,
            )
            baseline_metric = case_metrics["baseline"]
            regularized_metric = case_metrics["regularized"]
            draw.multiline_text(
                (24, y + 24),
                (
                    f"SHOWN CASE · {profile.upper()} fit\n"
                    f"PSNR {baseline_metric['psnr']:.2f} → "
                    f"{regularized_metric['psnr']:.2f} "
                    f"({case_metrics['delta']['psnr']:+.3f} dB)\n"
                    f"SSIM {baseline_metric['ssim_global']:.3f} → "
                    f"{regularized_metric['ssim_global']:.3f} "
                    f"({case_metrics['delta']['ssim_global']:+.4f})"
                ),
                font=_font(22, bold=True),
                fill="white",
                spacing=9,
            )
            panels = _evaluation_case_panels(
                baseline_camera,
                regularized_camera,
                lighting=lighting,
                case_id=case_id,
                profile=profile,
            )
            if [name for name, _, _ in panels] != columns:
                raise ValueError("evaluation panel columns differ from report header")
            for column, (_, image, contain) in enumerate(panels):
                panel = (
                    _contain_image(image, (tile_width, tile_height))
                    if contain
                    else image.convert("RGB")
                )
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
    impulse_cleanup: dict[str, Any] | None,
    frequency_cleanup: dict[str, Any] | None,
    case_png_metrics: dict[str, Any] | None,
) -> None:
    rows = []
    for profile, maps in map_metrics.items():
        for map_name, variants in maps.items():
            for metric in (
                "neighbor_variation",
                "median5_residual_mae",
                "median5_residual_outlier_fraction_gt_0p05",
                "quiet_region_neighbor_variation",
                "quiet_region_median5_residual_mae",
                "quiet_region_median5_residual_outlier_fraction_gt_0p05",
                "guide_edge_gradient_magnitude",
            ):
                baseline = variants["baseline"][metric]
                regularized = variants["regularized"][metric]
                is_edge_magnitude = metric == "guide_edge_gradient_magnitude"
                rows.append(
                    {
                        "category": "material_spatial_statistics",
                        "profile": profile,
                        "target": map_name,
                        "metric": metric,
                        "baseline": baseline,
                        "regularized": regularized,
                        "delta": regularized - baseline,
                        "relative_change_fraction": (
                            ""
                            if is_edge_magnitude
                            else _relative_change_fraction(baseline, regularized)
                        ),
                        "magnitude_ratio": (
                            variants["comparison"][
                                "guide_edge_gradient_magnitude_ratio"
                            ]
                            if is_edge_magnitude
                            else ""
                        ),
                        "correspondence": "",
                        "meaningful_baseline_gradient": variants["comparison"][
                            "meaningful_guide_edge_gradient"
                        ],
                    }
                )
            rows.append(
                {
                    "category": "material_edge_correspondence",
                    "profile": profile,
                    "target": map_name,
                    "metric": "guide_edge_signed_gradient_cosine",
                    "baseline": "",
                    "regularized": "",
                    "delta": "",
                    "relative_change_fraction": "",
                    "magnitude_ratio": "",
                    "correspondence": variants["comparison"][
                        "guide_edge_signed_gradient_cosine"
                    ],
                    "meaningful_baseline_gradient": variants["comparison"][
                        "meaningful_guide_edge_gradient"
                    ],
                }
            )
    if impulse_cleanup is not None and impulse_cleanup.get("available") is True:
        for profile, cleanup in impulse_cleanup["profiles"].items():
            cleanup_rows = (
                (
                    "all_maps",
                    "flagged_map_entries",
                    "",
                    cleanup["flagged"]["map_entries"],
                ),
                (
                    "all_maps",
                    "flagged_map_entry_fraction",
                    "",
                    cleanup["flagged"]["map_entry_fraction"],
                ),
                (
                    "all_maps",
                    "mean_absolute_distance_to_frozen_target",
                    cleanup["mean_absolute_distance_to_frozen_target"]["baseline"],
                    cleanup["mean_absolute_distance_to_frozen_target"]["candidate"],
                ),
                (
                    "all_maps",
                    "resolved_to_dead_zone_fraction",
                    "",
                    cleanup["resolved_to_dead_zone"]["fraction"],
                ),
                (
                    "all_maps",
                    "outside_spatial_flag_union_mean_absolute_exported_change",
                    "",
                    cleanup["exported_change_outside_spatial_flag_union"][
                        "mean_absolute"
                    ],
                ),
                (
                    "all_maps",
                    "outside_spatial_flag_union_maximum_absolute_exported_change",
                    "",
                    cleanup["exported_change_outside_spatial_flag_union"][
                        "maximum_absolute"
                    ],
                ),
            )
            for target, metric, baseline, candidate in cleanup_rows:
                delta = (
                    candidate - baseline
                    if isinstance(baseline, (int, float))
                    and isinstance(candidate, (int, float))
                    else ""
                )
                rows.append(
                    {
                        "category": "impulse_cleanup",
                        "profile": profile,
                        "target": target,
                        "metric": metric,
                        "baseline": baseline,
                        "regularized": candidate,
                        "delta": delta,
                        "relative_change_fraction": "",
                        "magnitude_ratio": "",
                        "correspondence": "",
                        "meaningful_baseline_gradient": "",
                    }
                )
            for map_name, map_cleanup in cleanup["maps"].items():
                map_rows = (
                    (
                        "flagged_centers",
                        "",
                        map_cleanup["flagged_centers"],
                    ),
                    (
                        "flagged_fraction_of_fit_foreground",
                        "",
                        map_cleanup["flagged_fraction_of_fit_foreground"],
                    ),
                    (
                        "mean_absolute_distance_to_frozen_target",
                        map_cleanup["mean_absolute_distance_to_frozen_target"][
                            "baseline"
                        ],
                        map_cleanup["mean_absolute_distance_to_frozen_target"][
                            "candidate"
                        ],
                    ),
                    (
                        "resolved_to_dead_zone_fraction",
                        "",
                        map_cleanup["resolved_to_dead_zone"]["fraction"],
                    ),
                    (
                        "outside_own_flags_mean_absolute_exported_change",
                        "",
                        map_cleanup["exported_change_outside_own_flags"][
                            "mean_absolute"
                        ],
                    ),
                    (
                        "outside_own_flags_maximum_absolute_exported_change",
                        "",
                        map_cleanup["exported_change_outside_own_flags"][
                            "maximum_absolute"
                        ],
                    ),
                )
                for metric, baseline, candidate in map_rows:
                    delta = (
                        candidate - baseline
                        if isinstance(baseline, (int, float))
                        and isinstance(candidate, (int, float))
                        else ""
                    )
                    rows.append(
                        {
                            "category": "impulse_cleanup",
                            "profile": profile,
                            "target": map_name,
                            "metric": metric,
                            "baseline": baseline,
                            "regularized": candidate,
                            "delta": delta,
                            "relative_change_fraction": "",
                            "magnitude_ratio": "",
                            "correspondence": "",
                            "meaningful_baseline_gradient": "",
                        }
                    )
    if frequency_cleanup is not None and frequency_cleanup.get("available") is True:
        for profile, cleanup in frequency_cleanup["profiles"].items():
            cleanup_rows = (
                (
                    "all_maps",
                    "updated_map_entries",
                    "",
                    cleanup["updated"]["map_entries"],
                ),
                (
                    "all_maps",
                    "updated_map_entry_fraction",
                    "",
                    cleanup["updated"]["map_entry_fraction"],
                ),
                (
                    "all_maps",
                    "consensus_map_entries",
                    "",
                    cleanup["consensus"]["map_entries"],
                ),
                (
                    "all_maps",
                    "consensus_fraction_of_updated_entries",
                    "",
                    cleanup["consensus"]["fraction_of_updated_entries"],
                ),
                (
                    "all_maps",
                    "mean_absolute_distance_to_frozen_target",
                    cleanup["mean_absolute_distance_to_frozen_target"]["baseline"],
                    cleanup["mean_absolute_distance_to_frozen_target"]["candidate"],
                ),
                (
                    "all_maps",
                    "within_export_tolerance_of_frozen_target_fraction",
                    "",
                    cleanup["within_export_tolerance_of_frozen_target"]["fraction"],
                ),
                (
                    "all_maps",
                    "outside_spatial_update_union_mean_absolute_exported_change",
                    "",
                    cleanup["exported_change_outside_spatial_update_union"][
                        "mean_absolute"
                    ],
                ),
                (
                    "all_maps",
                    "outside_spatial_update_union_maximum_absolute_exported_change",
                    "",
                    cleanup["exported_change_outside_spatial_update_union"][
                        "maximum_absolute"
                    ],
                ),
            )
            for target, metric, baseline, candidate in cleanup_rows:
                delta = (
                    candidate - baseline
                    if isinstance(baseline, (int, float))
                    and isinstance(candidate, (int, float))
                    else ""
                )
                rows.append(
                    {
                        "category": "frequency_cleanup",
                        "profile": profile,
                        "target": target,
                        "metric": metric,
                        "baseline": baseline,
                        "regularized": candidate,
                        "delta": delta,
                        "relative_change_fraction": "",
                        "magnitude_ratio": "",
                        "correspondence": "",
                        "meaningful_baseline_gradient": "",
                    }
                )
            for map_name, map_cleanup in cleanup["maps"].items():
                map_rows = (
                    ("updated_entries", "", map_cleanup["updated_entries"]),
                    (
                        "updated_fraction_of_fit_foreground",
                        "",
                        map_cleanup["updated_fraction_of_fit_foreground"],
                    ),
                    (
                        "consensus_entries",
                        "",
                        map_cleanup["consensus_entries"],
                    ),
                    (
                        "mean_absolute_distance_to_frozen_target",
                        map_cleanup["mean_absolute_distance_to_frozen_target"][
                            "baseline"
                        ],
                        map_cleanup["mean_absolute_distance_to_frozen_target"][
                            "candidate"
                        ],
                    ),
                    (
                        "within_export_tolerance_of_frozen_target_fraction",
                        "",
                        map_cleanup[
                            "within_export_tolerance_of_frozen_target"
                        ]["fraction"],
                    ),
                    (
                        "outside_own_update_mask_maximum_absolute_exported_change",
                        "",
                        map_cleanup["exported_change_outside_own_update_mask"][
                            "maximum_absolute"
                        ],
                    ),
                )
                for metric, baseline, candidate in map_rows:
                    delta = (
                        candidate - baseline
                        if isinstance(baseline, (int, float))
                        and isinstance(candidate, (int, float))
                        else ""
                    )
                    rows.append(
                        {
                            "category": "frequency_cleanup",
                            "profile": profile,
                            "target": map_name,
                            "metric": metric,
                            "baseline": baseline,
                            "regularized": candidate,
                            "delta": delta,
                            "relative_change_fraction": "",
                            "magnitude_ratio": "",
                            "correspondence": "",
                            "meaningful_baseline_gradient": "",
                        }
                    )
        gate_report = frequency_cleanup.get("scalar_map_cleanup_qualification")
        if isinstance(gate_report, dict):
            for gate_name, gate in gate_report.get("gates", {}).items():
                if gate.get("operator") == "within_inclusive_range":
                    gate_rows = (
                        (
                            f"{gate_name}_minimum",
                            gate.get("minimum"),
                            gate.get("observed_minimum"),
                        ),
                        (
                            f"{gate_name}_maximum",
                            gate.get("maximum"),
                            gate.get("observed_maximum"),
                        ),
                    )
                else:
                    gate_rows = (
                        (gate_name, gate.get("threshold"), gate.get("observed")),
                    )
                for metric, threshold, observed in gate_rows:
                    comparison_baseline = gate.get("baseline_value")
                    comparison_regularized = gate.get("regularized_value")
                    has_comparison_values = isinstance(
                        comparison_baseline, (int, float)
                    ) and isinstance(comparison_regularized, (int, float))
                    rows.append(
                        {
                            "category": "scalar_map_cleanup_gate",
                            "profile": "aggregate",
                            "target": (
                                f"{gate.get('operator', '')} {threshold}"
                                if has_comparison_values
                                else gate.get("operator", "")
                            ),
                            "metric": metric,
                            "baseline": (
                                comparison_baseline
                                if has_comparison_values
                                else threshold
                            ),
                            "regularized": (
                                comparison_regularized
                                if has_comparison_values
                                else observed
                            ),
                            "delta": (
                                comparison_regularized - comparison_baseline
                                if has_comparison_values
                                else observed - threshold
                                if isinstance(observed, (int, float))
                                and isinstance(threshold, (int, float))
                                else ""
                            ),
                            "relative_change_fraction": (
                                _relative_change_fraction(
                                    float(comparison_baseline),
                                    float(comparison_regularized),
                                )
                                if has_comparison_values
                                else ""
                            ),
                            "magnitude_ratio": "",
                            "correspondence": "",
                            "meaningful_baseline_gradient": "",
                            "gate_meets_threshold": gate.get(
                                "meets_threshold", False
                            ),
                        }
                    )
            coverage = gate_report.get("qualification_coverage", {})
            rows.append(
                {
                    "category": "scalar_map_cleanup_qualification",
                    "profile": "aggregate",
                    "target": "exact_profile_coverage",
                    "metric": "qualification_status",
                    "baseline": ",".join(coverage.get("required_profiles", [])),
                    "regularized": ",".join(
                        coverage.get("evaluated_profiles", [])
                    ),
                    "delta": gate_report.get("qualification_status", ""),
                    "relative_change_fraction": "",
                    "magnitude_ratio": "",
                    "correspondence": "",
                    "meaningful_baseline_gradient": "",
                    "gate_meets_threshold": coverage.get("complete", False),
                }
            )
    if case_png_metrics is not None:
        for lighting, selection in case_png_metrics["selected_cases"].items():
            for role in ("median", "worst"):
                selected = selection[role]
                rows.append(
                    {
                        "category": "evaluation_case_selection",
                        "profile": "aggregate",
                        "target": f"{lighting}/{selected['case_id']}",
                        "metric": f"{role}_mean_profile_delta_psnr_db",
                        "baseline": "",
                        "regularized": "",
                        "delta": selected["mean_profile_delta_psnr_db"],
                        "relative_change_fraction": "",
                        "magnitude_ratio": "",
                        "correspondence": "",
                        "meaningful_baseline_gradient": "",
                        "gate_meets_threshold": "",
                    }
                )
        for case in case_png_metrics["rows"]:
            for metric, baseline in case["baseline"].items():
                regularized = case["regularized"][metric]
                rows.append(
                    {
                        "category": "case_png_relighting",
                        "profile": case["profile"],
                        "target": f"{case['lighting']}/{case['case_id']}",
                        "metric": metric,
                        "baseline": baseline,
                        "regularized": regularized,
                        "delta": case["delta"][metric],
                        "relative_change_fraction": "",
                        "magnitude_ratio": "",
                        "correspondence": "",
                        "meaningful_baseline_gradient": "",
                        "gate_meets_threshold": "",
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
                        "relative_change_fraction": "",
                        "magnitude_ratio": "",
                        "correspondence": "",
                        "meaningful_baseline_gradient": "",
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "category",
                "profile",
                "target",
                "metric",
                "baseline",
                "regularized",
                "delta",
                "relative_change_fraction",
                "magnitude_ratio",
                "correspondence",
                "meaningful_baseline_gradient",
                "gate_meets_threshold",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def _rank_impulse_detail_maps(profile_cleanup: dict[str, Any]) -> tuple[str, ...]:
    cleanup_maps = profile_cleanup.get("maps")
    if not isinstance(cleanup_maps, dict):
        raise ValueError("impulse detail presentation is missing per-map cleanup data")
    display_order = {name: index for index, name in enumerate(SCALAR_MAPS)}
    return tuple(
        sorted(
            SCALAR_MAPS,
            key=lambda name: (
                -int(cleanup_maps.get(name, {}).get("flagged_centers", 0)),
                display_order[name],
            ),
        )[:2]
    )


def _densest_flagged_crop_box(
    flagged: np.ndarray,
    crop_size: int,
) -> tuple[int, int, int, int]:
    """Return a deterministic native crop centered on its densest flag window."""
    flagged = np.asarray(flagged)
    if flagged.ndim != 2:
        raise ValueError("detail flag mask must be two-dimensional")
    height, width = flagged.shape
    if crop_size <= 0 or crop_size > min(height, width):
        raise ValueError("detail crop size must fit inside the flag mask")
    flagged = flagged.astype(bool, copy=False)
    if not np.any(flagged):
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        return (left, top, left + crop_size, top + crop_size)

    def window_sum(values: np.ndarray) -> np.ndarray:
        integral = np.pad(
            np.asarray(values, dtype=np.float64),
            ((1, 0), (1, 0)),
            mode="constant",
        ).cumsum(axis=0).cumsum(axis=1)
        return (
            integral[crop_size:, crop_size:]
            - integral[:-crop_size, crop_size:]
            - integral[crop_size:, :-crop_size]
            + integral[:-crop_size, :-crop_size]
        )

    counts = window_sum(flagged)
    maximum = float(counts.max())
    candidates = np.argwhere(counts == maximum)
    rows = np.arange(height, dtype=np.float64)[:, None]
    columns = np.arange(width, dtype=np.float64)[None, :]
    row_sums = window_sum(flagged * rows)
    column_sums = window_sum(flagged * columns)
    candidate_rows = candidates[:, 0]
    candidate_columns = candidates[:, 1]
    local_centroid_rows = row_sums[candidate_rows, candidate_columns] / maximum
    local_centroid_columns = (
        column_sums[candidate_rows, candidate_columns] / maximum
    )
    crop_center_rows = candidate_rows + (crop_size - 1) / 2.0
    crop_center_columns = candidate_columns + (crop_size - 1) / 2.0
    center_distance = (crop_center_rows - local_centroid_rows) ** 2 + (
        crop_center_columns - local_centroid_columns
    ) ** 2
    # First minimize center-to-local-centroid distance, then prefer top/left.
    order = np.lexsort((candidate_columns, candidate_rows, center_distance))
    top, left = candidates[int(order[0])]
    return (
        int(left),
        int(top),
        int(left + crop_size),
        int(top + crop_size),
    )


def _window_sums(values: np.ndarray, crop_size: int) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("crop score input must be two-dimensional")
    if crop_size <= 0 or crop_size > min(values.shape):
        raise ValueError("crop size must fit inside the score input")
    integral = np.pad(
        values.astype(np.int64, copy=False),
        ((1, 0), (1, 0)),
        mode="constant",
    ).cumsum(axis=0, dtype=np.int64).cumsum(axis=1, dtype=np.int64)
    return (
        integral[crop_size:, crop_size:]
        - integral[:-crop_size, crop_size:]
        - integral[crop_size:, :-crop_size]
        + integral[:-crop_size, :-crop_size]
    )


def _select_adaptive_cleanup_crop(
    removed_masks: Sequence[np.ndarray],
    introduced_masks: Sequence[np.ndarray],
    quiet_masks: Sequence[np.ndarray],
    crop_size: int = FREQUENCY_ADAPTIVE_EVIDENCE_CROP_SIZE,
) -> dict[str, Any]:
    """Select one auditable crop shared by every lighting profile for a map."""
    if not removed_masks or not (
        len(removed_masks) == len(introduced_masks) == len(quiet_masks)
    ):
        raise ValueError("adaptive crop selection requires matched profile masks")
    shape = np.asarray(removed_masks[0]).shape
    if len(shape) != 2:
        raise ValueError("adaptive crop masks must be two-dimensional")
    normalized = []
    for label, masks in (
        ("removed", removed_masks),
        ("introduced", introduced_masks),
        ("quiet", quiet_masks),
    ):
        arrays = [np.asarray(mask, dtype=bool) for mask in masks]
        if any(array.shape != shape for array in arrays):
            raise ValueError(f"adaptive {label} crop masks do not share a shape")
        normalized.append(arrays)
    removed, introduced, quiet = normalized
    removed_scores = sum(
        (_window_sums(mask, crop_size) for mask in removed),
        start=np.zeros(
            (shape[0] - crop_size + 1, shape[1] - crop_size + 1),
            dtype=np.int64,
        ),
    )
    introduced_scores = sum(
        (_window_sums(mask, crop_size) for mask in introduced),
        start=np.zeros_like(removed_scores),
    )
    quiet_scores = sum(
        (_window_sums(mask, crop_size) for mask in quiet),
        start=np.zeros_like(removed_scores),
    )
    net_scores = removed_scores - introduced_scores
    tops, lefts = np.indices(net_scores.shape)
    # Lexicographic audit rule: maximize net removed, then removed; minimize
    # introduced; maximize quiet support; finally prefer the top/left window.
    order = np.lexsort(
        (
            lefts.ravel(),
            tops.ravel(),
            -quiet_scores.ravel(),
            introduced_scores.ravel(),
            -removed_scores.ravel(),
            -net_scores.ravel(),
        )
    )
    top = int(tops.ravel()[order[0]])
    left = int(lefts.ravel()[order[0]])
    return {
        "crop_box_xyxy": [left, top, left + crop_size, top + crop_size],
        "crop_size_pixels": crop_size,
        "aggregate_net_removed": int(net_scores[top, left]),
        "aggregate_removed": int(removed_scores[top, left]),
        "aggregate_introduced": int(introduced_scores[top, left]),
        "aggregate_quiet_pixels": int(quiet_scores[top, left]),
        "selection_rule": (
            "maximize aggregate net removed (removed minus introduced) across "
            "profiles, then maximize removed, minimize introduced, maximize "
            "quiet pixels, then prefer top and left"
        ),
        "manual_selection": False,
    }


def _adaptive_transition_counts(
    fixed_outliers: np.ndarray,
    adaptive_outliers: np.ndarray,
    quiet_pixels: np.ndarray,
) -> dict[str, Any]:
    fixed = np.asarray(fixed_outliers, dtype=bool)
    adaptive = np.asarray(adaptive_outliers, dtype=bool)
    quiet = np.asarray(quiet_pixels, dtype=bool)
    if fixed.shape != adaptive.shape or fixed.shape != quiet.shape or fixed.ndim != 2:
        raise ValueError("adaptive evidence masks must be matched 2D arrays")
    if np.any(fixed & ~quiet) or np.any(adaptive & ~quiet):
        raise ValueError("adaptive evidence outliers must be restricted to quiet pixels")
    removed = fixed & ~adaptive
    introduced = adaptive & ~fixed
    persistent = fixed & adaptive
    quiet_count = int(np.count_nonzero(quiet))
    fixed_count = int(np.count_nonzero(fixed))
    adaptive_count = int(np.count_nonzero(adaptive))
    return {
        "quiet_pixels": quiet_count,
        "fixed_v1_outliers": fixed_count,
        "adaptive_v1_outliers": adaptive_count,
        "fixed_v1_outlier_fraction": (
            float(fixed_count / quiet_count) if quiet_count else None
        ),
        "adaptive_v1_outlier_fraction": (
            float(adaptive_count / quiet_count) if quiet_count else None
        ),
        "removed": int(np.count_nonzero(removed)),
        "introduced": int(np.count_nonzero(introduced)),
        "persistent": int(np.count_nonzero(persistent)),
        "net_removed": fixed_count - adaptive_count,
        "relative_reduction_fraction": (
            float((fixed_count - adaptive_count) / fixed_count)
            if fixed_count
            else None
        ),
    }


def _aggregate_adaptive_transition_counts(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("adaptive evidence aggregate requires at least one row")
    quiet = sum(int(row["quiet_pixels"]) for row in rows)
    fixed = sum(int(row["fixed_v1_outliers"]) for row in rows)
    adaptive = sum(int(row["adaptive_v1_outliers"]) for row in rows)
    removed = sum(int(row["removed"]) for row in rows)
    introduced = sum(int(row["introduced"]) for row in rows)
    persistent = sum(int(row["persistent"]) for row in rows)
    if fixed != removed + persistent or adaptive != introduced + persistent:
        raise ValueError("adaptive evidence transition counts are inconsistent")
    return {
        "quiet_pixels": quiet,
        "fixed_v1_outliers": fixed,
        "adaptive_v1_outliers": adaptive,
        "fixed_v1_outlier_fraction": float(fixed / quiet) if quiet else None,
        "adaptive_v1_outlier_fraction": float(adaptive / quiet) if quiet else None,
        "removed": removed,
        "introduced": introduced,
        "persistent": persistent,
        "net_removed": fixed - adaptive,
        "relative_reduction_fraction": (
            float((fixed - adaptive) / fixed) if fixed else None
        ),
        "direction": (
            "improvement"
            if adaptive < fixed
            else "regression"
            if adaptive > fixed
            else "unchanged"
        ),
    }


def _adaptive_cleanup_profile_masks(
    baseline_camera: Path,
    regularized_camera: Path,
    profile: str,
    interior_mask: np.ndarray,
    guide_diagnostic: dict[str, Any],
) -> dict[str, Any]:
    maps_dir = baseline_camera / "material" / profile / "maps"
    guide_masks, _ = _guide_region_masks(
        _read_rgb_map(maps_dir / "baseColor.png"),
        _read_rgb_map(maps_dir / "normal.png"),
        interior_mask,
        albedo_sigma=float(guide_diagnostic["albedo_sigma"]),
        normal_sigma=float(guide_diagnostic["normal_sigma"]),
    )
    quiet = guide_masks["quiet_pixels"]
    maps = {}
    for map_name in SCALAR_MAPS:
        fixed_values = _read_scalar_map(maps_dir / f"{map_name}.png")
        adaptive_values = _read_scalar_map(
            regularized_camera
            / "material"
            / profile
            / "maps"
            / f"{map_name}.png"
        )
        if fixed_values.shape != interior_mask.shape or adaptive_values.shape != (
            interior_mask.shape
        ):
            raise ValueError("adaptive evidence maps do not match the fit interior")
        fixed = quiet & (
            _own_median5_residual(fixed_values) > FREQUENCY_EVIDENCE_THRESHOLD
        )
        adaptive = quiet & (
            _own_median5_residual(adaptive_values) > FREQUENCY_EVIDENCE_THRESHOLD
        )
        maps[map_name] = {
            "fixed": fixed,
            "adaptive": adaptive,
            "removed": fixed & ~adaptive,
            "introduced": adaptive & ~fixed,
            "persistent": fixed & adaptive,
        }
    return {"quiet": quiet, "maps": maps}


def _measure_adaptive_cleanup_evidence(
    baseline_camera: Path,
    regularized_camera: Path,
    profiles: Sequence[str],
    interior_mask: np.ndarray,
    guide_diagnostic: dict[str, Any],
    map_metrics: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    if not profiles:
        raise ValueError("adaptive cleanup evidence requires lighting profiles")
    comparison_mode = contract.get("comparison_mode")
    versions_by_mode = {
        "frequency-consensus-v1-to-adaptive-v1": "v1",
        "frequency-consensus-v2-to-adaptive-v2": "v2",
    }
    adapter_version = versions_by_mode.get(comparison_mode)
    if adapter_version is None:
        raise ValueError(
            "adaptive cleanup evidence requires a supported active comparison mode"
        )
    fixed_label = f"Fixed {adapter_version}"
    adaptive_label = f"Adaptive {adapter_version}"
    profile_masks = {
        profile: _adaptive_cleanup_profile_masks(
            baseline_camera,
            regularized_camera,
            profile,
            interior_mask,
            guide_diagnostic,
        )
        for profile in profiles
    }
    per_profile: dict[str, Any] = {}
    all_rows = []
    focused_rows = []
    non_focused_rows = []
    regressions = []
    for profile in profiles:
        quiet = profile_masks[profile]["quiet"]
        profile_maps = {}
        for map_name in SCALAR_MAPS:
            masks = profile_masks[profile]["maps"][map_name]
            counts = _adaptive_transition_counts(
                masks["fixed"], masks["adaptive"], quiet
            )
            expected_fixed = map_metrics[profile][map_name]["baseline"][
                "quiet_region_median5_residual_outlier_count_gt_0p05"
            ]
            expected_adaptive = map_metrics[profile][map_name]["regularized"][
                "quiet_region_median5_residual_outlier_count_gt_0p05"
            ]
            if (
                counts["fixed_v1_outliers"] != expected_fixed
                or counts["adaptive_v1_outliers"] != expected_adaptive
            ):
                raise ValueError(
                    "adaptive evidence counts differ from exported-map metrics: "
                    f"{profile}/{map_name}"
                )
            profile_maps[map_name] = counts
            all_rows.append(counts)
            if map_name in FREQUENCY_ADAPTIVE_FOCUSED_MAPS:
                focused_rows.append(counts)
            else:
                non_focused_rows.append(counts)
                if counts["adaptive_v1_outliers"] > counts["fixed_v1_outliers"]:
                    regressions.append(
                        {
                            "profile": profile,
                            "map": map_name,
                            "fixed_v1_outliers": counts["fixed_v1_outliers"],
                            "adaptive_v1_outliers": counts[
                                "adaptive_v1_outliers"
                            ],
                            "net_introduced": -counts["net_removed"],
                        }
                    )
        per_profile[profile] = {"maps": profile_maps}

    crops = {}
    for map_name in FREQUENCY_ADAPTIVE_FOCUSED_MAPS:
        selection = _select_adaptive_cleanup_crop(
            [
                profile_masks[profile]["maps"][map_name]["removed"]
                for profile in profiles
            ],
            [
                profile_masks[profile]["maps"][map_name]["introduced"]
                for profile in profiles
            ],
            [profile_masks[profile]["quiet"] for profile in profiles],
        )
        left, top, right, bottom = selection["crop_box_xyxy"]
        crop_rows = []
        per_profile_crop = {}
        for profile in profiles:
            masks = profile_masks[profile]["maps"][map_name]
            quiet = profile_masks[profile]["quiet"]
            counts = _adaptive_transition_counts(
                masks["fixed"][top:bottom, left:right],
                masks["adaptive"][top:bottom, left:right],
                quiet[top:bottom, left:right],
            )
            per_profile_crop[profile] = counts
            crop_rows.append(counts)
        aggregate_crop = _aggregate_adaptive_transition_counts(crop_rows)
        if (
            aggregate_crop["net_removed"] != selection["aggregate_net_removed"]
            or aggregate_crop["removed"] != selection["aggregate_removed"]
            or aggregate_crop["introduced"]
            != selection["aggregate_introduced"]
            or aggregate_crop["quiet_pixels"]
            != selection["aggregate_quiet_pixels"]
        ):
            raise ValueError("adaptive evidence crop scores do not match counts")
        crops[map_name] = {
            **selection,
            "aggregate": aggregate_crop,
            "profiles": per_profile_crop,
        }

    all_summary = _aggregate_adaptive_transition_counts(all_rows)
    focused_summary = _aggregate_adaptive_transition_counts(focused_rows)
    non_focused_summary = _aggregate_adaptive_transition_counts(non_focused_rows)
    regressions.sort(
        key=lambda row: (-int(row["net_introduced"]), row["profile"], row["map"])
    )
    return {
        "schema": FREQUENCY_ADAPTIVE_EVIDENCE_SCHEMA,
        "available": True,
        "comparison": (
            f"frequency-consensus-{adapter_version}-fixed-target-to-"
            f"adaptive-{adapter_version}"
        ),
        "variant_labels": {
            "baseline": fixed_label,
            "regularized": adaptive_label,
        },
        "definition": {
            "source": "exact exported scalar-map PNGs",
            "outlier": "abs(map - own median5_nearest) > 0.05",
            "threshold": FREQUENCY_EVIDENCE_THRESHOLD,
            "threshold_operator": ">",
            "region": (
                "baseline-guide quiet pixels inside the radius-2 eroded fit mask"
            ),
            "guide_configuration": {
                "albedo_sigma": float(guide_diagnostic["albedo_sigma"]),
                "normal_sigma": float(guide_diagnostic["normal_sigma"]),
                "quiet_percentile": QUIET_GUIDE_PERCENTILE,
            },
            "transitions": {
                "removed": f"{fixed_label} outlier and not {adaptive_label} outlier",
                "introduced": (
                    f"{adaptive_label} outlier and not {fixed_label} outlier"
                ),
                "persistent": "outlier in both variants",
            },
        },
        "profiles": per_profile,
        "focused_maps": list(FREQUENCY_ADAPTIVE_FOCUSED_MAPS),
        "focused_summary": focused_summary,
        "all_map_summary": all_summary,
        "non_focused_summary": non_focused_summary,
        "non_focused_regressions": regressions,
        "non_focused_regression_count": len(regressions),
        "crops": crops,
        "presentation": {
            "artifact": "material/adaptive_cleanup_evidence.png",
            "scale": FREQUENCY_ADAPTIVE_EVIDENCE_SCALE,
            "resampling": "nearest",
            "crop_shared_across_profiles_per_focused_map": True,
            "manual_selection": False,
            "disclosure": (
                "Focused-map cleanup is reported separately from non-focused "
                "regression; no claim is made that every map improves."
            ),
        },
    }


def _frequency_annotation_layout(
    profile_cleanup: dict[str, Any],
) -> dict[str, Any]:
    cleanup_maps = profile_cleanup.get("maps")
    if not isinstance(cleanup_maps, dict):
        raise ValueError("frequency annotation layout is missing map diagnostics")
    font = _font(17, bold=True)
    spacing = 5
    measurement = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(measurement)
    labels = profile_cleanup.get(
        "annotation_labels",
        {"updated": "updates", "consensus": "consensus"},
    )
    if not isinstance(labels, dict) or set(labels) != {"updated", "consensus"}:
        raise ValueError("frequency annotation layout has invalid count labels")
    rows = []
    maximum_right = FREQUENCY_ANNOTATION_LEFT
    for map_name in FREQUENCY_DETAIL_MAPS:
        map_cleanup = cleanup_maps.get(map_name)
        if not isinstance(map_cleanup, dict):
            raise ValueError(
                f"frequency annotation layout is missing {_display_name(map_name)}"
            )
        text = (
            f"{_display_name(map_name)}\n"
            f"{labels['updated']} {map_cleanup['updated_entries']}\n"
            f"{labels['consensus']} {map_cleanup['consensus_entries']}"
        )
        bounds = draw.multiline_textbbox(
            (FREQUENCY_ANNOTATION_LEFT, 0),
            text,
            font=font,
            spacing=spacing,
        )
        maximum_right = max(maximum_right, int(math.ceil(bounds[2])))
        rows.append(
            {
                "map": map_name,
                "text": text,
                "bounds_xyxy": [
                    int(math.floor(bounds[0])),
                    int(math.floor(bounds[1])),
                    int(math.ceil(bounds[2])),
                    int(math.ceil(bounds[3])),
                ],
            }
        )
    tile_left = max(
        FREQUENCY_DETAIL_TILE_LEFT,
        maximum_right + FREQUENCY_ANNOTATION_GAP,
    )
    return {
        "annotation_left": FREQUENCY_ANNOTATION_LEFT,
        "annotation_gap": FREQUENCY_ANNOTATION_GAP,
        "font_size": 17,
        "spacing": spacing,
        "tile_left": tile_left,
        "rows": rows,
    }


def _write_frequency_detail(
    baseline_camera: Path,
    regularized_camera: Path,
    acquisitions: dict[str, dict[str, dict[str, Any]]],
    frequency_cleanup: dict[str, Any],
    output: Path,
) -> None:
    presentation = frequency_cleanup.get("presentation")
    if not isinstance(presentation, dict):
        raise ValueError("frequency detail presentation metadata is missing")
    profile = presentation.get("profile")
    if not isinstance(profile, str) or profile not in acquisitions["regularized"]:
        raise ValueError("frequency detail presentation profile is invalid")
    crop_box = tuple(presentation.get("crop_box_xyxy", ()))
    if crop_box != FREQUENCY_DETAIL_CROP_BOX:
        raise ValueError("frequency detail presentation crop differs from audit")
    left, top, right, bottom = crop_box
    tile_width = right - left
    tile_height = bottom - top
    if tile_width != 96 or tile_height != 96:
        raise ValueError("frequency detail crop must be exactly 96x96 pixels")
    sample_path = (
        baseline_camera
        / "material"
        / profile
        / "maps"
        / f"{FREQUENCY_DETAIL_MAPS[0]}.png"
    )
    with Image.open(sample_path) as sample:
        source_width, source_height = sample.size
    if right > source_width or bottom > source_height:
        raise ValueError(
            "frequency detail fixed crop does not fit the exported material maps"
        )
    artifact = _load_frequency_frozen_artifact(
        regularized_camera / "material" / profile,
        acquisitions["regularized"][profile],
        (source_height, source_width),
    )

    annotation_layout = _frequency_annotation_layout(
        frequency_cleanup["profiles"][profile]
    )
    tile_left = annotation_layout["tile_left"]
    columns = ("Baseline", "Regularized", "Frozen target", "|change| x8")
    row_stride = tile_height + FREQUENCY_DETAIL_ROW_GAP
    width = tile_left + len(columns) * tile_width + 24
    height = FREQUENCY_DETAIL_TILE_TOP + len(FREQUENCY_DETAIL_MAPS) * row_stride + 20
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (18, 12),
        (
            f"{profile.upper()} {frequency_cleanup['regularizer_label']} "
            "detail · native 1:1"
        ),
        font=_font(25, bold=True),
        fill="white",
    )
    draw.text(
        (18, 52),
        "Fixed crop x=96:192, y=160:256 · no resize or interpolation",
        font=_font(17),
        fill=(215, 215, 215),
    )
    draw.text(
        (18, 78),
        "Diagnostic only; material quality and texture preservation require full review.",
        font=_font(16),
        fill=(255, 203, 112),
    )
    for column, name in enumerate(columns):
        _draw_centered_text(
            draw,
            (
                tile_left + column * tile_width,
                100,
                tile_width,
                26,
            ),
            name,
            _font(14, bold=True),
            "white",
        )

    annotation_font = _font(annotation_layout["font_size"], bold=True)
    for row, map_name in enumerate(FREQUENCY_DETAIL_MAPS):
        y = FREQUENCY_DETAIL_TILE_TOP + row * row_stride
        draw.multiline_text(
            (annotation_layout["annotation_left"], y + 8),
            annotation_layout["rows"][row]["text"],
            font=annotation_font,
            fill="white",
            spacing=annotation_layout["spacing"],
        )
        baseline_path = (
            baseline_camera / "material" / profile / "maps" / f"{map_name}.png"
        )
        regularized_path = (
            regularized_camera / "material" / profile / "maps" / f"{map_name}.png"
        )
        with Image.open(baseline_path) as image:
            baseline_crop = image.convert("L").crop(crop_box)
        with Image.open(regularized_path) as image:
            regularized_crop = image.convert("L").crop(crop_box)
        target_u8 = np.floor(
            np.clip(artifact["maps"][map_name]["target"], 0.0, 1.0) * 255.0
            + 0.5
        ).astype(np.uint8)
        target_u8[~artifact["fit_foreground"]] = 0
        target_crop = Image.fromarray(target_u8, mode="L").crop(crop_box)
        baseline_u8 = np.asarray(baseline_crop, dtype=np.uint8)
        regularized_u8 = np.asarray(regularized_crop, dtype=np.uint8)
        delta_u8 = np.clip(
            np.abs(regularized_u8.astype(np.int16) - baseline_u8.astype(np.int16))
            * 8,
            0,
            255,
        ).astype(np.uint8)
        panels = (
            baseline_crop,
            regularized_crop,
            target_crop,
            Image.fromarray(delta_u8, mode="L"),
        )
        for column, panel in enumerate(panels):
            x = tile_left + column * tile_width
            # Intentionally paste the native 96x96 crop directly.  Any resize,
            # even nearest-neighbor, would violate this diagnostic's 1:1 contract.
            canvas.paste(panel.convert("RGB"), (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _write_frequency_full_maps(
    baseline_camera: Path,
    regularized_camera: Path,
    acquisitions: dict[str, dict[str, dict[str, Any]]],
    frequency_cleanup: dict[str, Any],
    output: Path,
) -> None:
    presentation = frequency_cleanup.get("presentation")
    if not isinstance(presentation, dict):
        raise ValueError("frequency full-map presentation metadata is missing")
    profile = presentation.get("profile")
    if not isinstance(profile, str) or profile not in acquisitions["regularized"]:
        raise ValueError("frequency full-map presentation profile is invalid")
    sample_path = (
        baseline_camera
        / "material"
        / profile
        / "maps"
        / f"{FREQUENCY_DETAIL_MAPS[0]}.png"
    )
    with Image.open(sample_path) as sample:
        tile_width, tile_height = sample.size
    artifact = _load_frequency_frozen_artifact(
        regularized_camera / "material" / profile,
        acquisitions["regularized"][profile],
        (tile_height, tile_width),
    )
    annotation_layout = _frequency_annotation_layout(
        frequency_cleanup["profiles"][profile]
    )
    columns = ("Baseline", "Regularized", "Frozen target", "|change| x8")
    row_gap = 42
    tile_left = annotation_layout["tile_left"]
    tile_top = 130
    row_stride = tile_height + row_gap
    width = tile_left + len(columns) * tile_width + 24
    height = tile_top + len(FREQUENCY_DETAIL_MAPS) * row_stride + 20
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (18, 12),
        (
            f"{profile.upper()} {frequency_cleanup['regularizer_label']} "
            "full maps · native 1:1"
        ),
        font=_font(25, bold=True),
        fill="white",
    )
    draw.text(
        (18, 52),
        "Every map pixel is one output pixel · no resize or interpolation",
        font=_font(17),
        fill=(215, 215, 215),
    )
    draw.text(
        (18, 78),
        "Diagnostic only; material quality and texture preservation require relighting review.",
        font=_font(16),
        fill=(255, 203, 112),
    )
    for column, name in enumerate(columns):
        _draw_centered_text(
            draw,
            (tile_left + column * tile_width, 100, tile_width, 26),
            name,
            _font(15, bold=True),
            "white",
        )

    annotation_font = _font(annotation_layout["font_size"], bold=True)
    for row, map_name in enumerate(FREQUENCY_DETAIL_MAPS):
        y = tile_top + row * row_stride
        draw.multiline_text(
            (annotation_layout["annotation_left"], y + 12),
            annotation_layout["rows"][row]["text"],
            font=annotation_font,
            fill="white",
            spacing=annotation_layout["spacing"],
        )
        with Image.open(
            baseline_camera / "material" / profile / "maps" / f"{map_name}.png"
        ) as image:
            baseline_panel = image.convert("L")
        with Image.open(
            regularized_camera
            / "material"
            / profile
            / "maps"
            / f"{map_name}.png"
        ) as image:
            regularized_panel = image.convert("L")
        target_u8 = np.floor(
            np.clip(artifact["maps"][map_name]["target"], 0.0, 1.0) * 255.0
            + 0.5
        ).astype(np.uint8)
        target_u8[~artifact["fit_foreground"]] = 0
        target_panel = Image.fromarray(target_u8, mode="L")
        baseline_u8 = np.asarray(baseline_panel, dtype=np.uint8)
        regularized_u8 = np.asarray(regularized_panel, dtype=np.uint8)
        delta_u8 = np.clip(
            np.abs(regularized_u8.astype(np.int16) - baseline_u8.astype(np.int16))
            * 8,
            0,
            255,
        ).astype(np.uint8)
        panels = (
            baseline_panel,
            regularized_panel,
            target_panel,
            Image.fromarray(delta_u8, mode="L"),
        )
        for column, panel in enumerate(panels):
            x = tile_left + column * tile_width
            canvas.paste(panel.convert("RGB"), (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _write_adaptive_cleanup_evidence(
    baseline_camera: Path,
    regularized_camera: Path,
    profiles: Sequence[str],
    interior_mask: np.ndarray,
    guide_diagnostic: dict[str, Any],
    evidence: dict[str, Any],
    summary: dict[str, Any],
    output: Path,
) -> None:
    if evidence.get("schema") != FREQUENCY_ADAPTIVE_EVIDENCE_SCHEMA:
        raise ValueError("adaptive cleanup evidence has an invalid schema")
    if evidence.get("focused_maps") != list(FREQUENCY_ADAPTIVE_FOCUSED_MAPS):
        raise ValueError("adaptive cleanup evidence focused maps are invalid")
    variant_labels = evidence.get("variant_labels")
    if not isinstance(variant_labels, dict) or set(variant_labels) != {
        "baseline",
        "regularized",
    }:
        raise ValueError("adaptive cleanup evidence variant labels are invalid")
    fixed_label = variant_labels["baseline"]
    adaptive_label = variant_labels["regularized"]
    if not all(isinstance(label, str) and label for label in variant_labels.values()):
        raise ValueError("adaptive cleanup evidence variant labels are invalid")
    profile_masks = {
        profile: _adaptive_cleanup_profile_masks(
            baseline_camera,
            regularized_camera,
            profile,
            interior_mask,
            guide_diagnostic,
        )
        for profile in profiles
    }
    scale = FREQUENCY_ADAPTIVE_EVIDENCE_SCALE
    source_size = FREQUENCY_ADAPTIVE_EVIDENCE_CROP_SIZE
    tile_size = source_size * scale
    gutter = 350
    columns = (
        f"{fixed_label} raw",
        f"{adaptive_label} raw",
        f"{fixed_label} outliers",
        f"{adaptive_label} outliers",
        "Transition",
    )
    width = gutter + len(columns) * tile_size + 40
    header_height = 430
    section_heading_height = 130
    column_heading_height = 64
    row_gap = 26
    row_stride = tile_size + row_gap
    section_height = (
        section_heading_height
        + column_heading_height
        + len(profiles) * row_stride
        + 28
    )
    footer_height = 164
    height = (
        header_height
        + len(FREQUENCY_ADAPTIVE_FOCUSED_MAPS) * section_height
        + footer_height
    )
    canvas = Image.new("RGB", (width, height), (12, 14, 18))
    draw = ImageDraw.Draw(canvas)
    contract = summary["comparison_contract"]
    draw.text(
        (36, 22),
        f"{fixed_label} frequency consensus → {adaptive_label} · cleanup evidence",
        font=_font(43, bold=True),
        fill="white",
    )
    header_lines = (
        (
            f"Controlled A/B · {fixed_label} frequency consensus "
            f"λ={contract['baseline_tv_weight']:g} → {adaptive_label} "
            f"λ={contract['regularized_tv_weight']:g} · exact exported scalar-map PNGs",
            (151, 205, 255),
            True,
        ),
        (
            "Outlier = |map − its own 5×5 nearest-border median| > 0.05 · "
            "baseline-guide quiet pixels inside the eroded fit interior",
            (215, 222, 230),
            False,
        ),
        (
            "Shared 96×96 crops; no cherry-picking · categorical colors; "
            "|Δ|×8: frequency_fullmaps_1to1.png",
            (215, 222, 230),
            True,
        ),
    )
    for line_index, (line, color, bold) in enumerate(header_lines):
        draw.text(
            (38, 78 + 34 * line_index),
            line,
            font=_font_for_width(
                draw,
                line,
                max_width=width - 76,
                preferred_size=24,
                minimum_size=22,
                bold=bold,
            ),
            fill=color,
        )

    focused = evidence["focused_summary"]
    all_maps = evidence["all_map_summary"]
    non_focused = evidence["non_focused_summary"]
    gates = summary["scalar_map_cleanup_qualification"]["gates"]
    all_gate = gates["aggregate_quiet_outlier_relative_reduction"]
    band_gate = gates["aggregate_guide_textured_band3_8_amplitude_ratio"]
    edge_gate = gates["guide_edge_gradient_magnitude_ratio"]
    psnr_gate = gates["aggregate_relighting_delta_psnr_db"]
    ssim_gate = gates["aggregate_relighting_delta_ssim_global"]
    all_gate_threshold = float(all_gate["threshold"])

    def percentage(value: float | None, digits: int = 2) -> str:
        return "n/a" if value is None else f"{100.0 * value:.{digits}f}%"

    def relative_change_label(value: float | None) -> str:
        if value is None:
            return "n/a"
        return (
            f"{100.0 * value:.1f}% reduction"
            if value >= 0.0
            else f"{-100.0 * value:.1f}% increase"
        )

    cards = (
        (
            "FOCUSED A + S",
            f"{percentage(focused['fixed_v1_outlier_fraction'])} → "
            f"{percentage(focused['adaptive_v1_outlier_fraction'])}\n"
            f"{relative_change_label(focused['relative_reduction_fraction'])}",
            (54, 167, 255),
        ),
        (
            "ALL 8 MAPS",
            f"{percentage(all_maps['fixed_v1_outlier_fraction'])} → "
            f"{percentage(all_maps['adaptive_v1_outlier_fraction'])}\n"
            f"{relative_change_label(all_maps['relative_reduction_fraction'])}\n"
            f"{100.0 * all_gate_threshold:.0f}% gate "
            f"{'met' if all_gate['meets_threshold'] else 'MISS'}",
            (255, 177, 64) if not all_gate["meets_threshold"] else (72, 205, 132),
        ),
        (
            "TEXTURE + EDGES",
            f"3–8 px amplitude {percentage(band_gate['observed'], 2)}\n"
            f"edge ratio {100.0 * edge_gate['observed_minimum']:.1f}–"
            f"{100.0 * edge_gate['observed_maximum']:.1f}%",
            (72, 205, 132),
        ),
        (
            "RELIGHTING",
            f"ΔPSNR {float(psnr_gate['observed']):+.4f} dB · "
            f"{'met' if psnr_gate['meets_threshold'] else 'MISS'}\n"
            f"ΔSSIM {float(ssim_gate['observed']):+.5f} · "
            f"{'met' if ssim_gate['meets_threshold'] else 'MISS'}",
            (
                (72, 205, 132)
                if psnr_gate["meets_threshold"] and ssim_gate["meets_threshold"]
                else (255, 177, 64)
            ),
        ),
    )
    card_gap = 18
    card_left = 36
    card_width = (width - 2 * card_left - 3 * card_gap) // 4
    card_top = 188
    card_height = 166
    for index, (title, body, accent) in enumerate(cards):
        left = card_left + index * (card_width + card_gap)
        draw.rounded_rectangle(
            (left, card_top, left + card_width, card_top + card_height),
            radius=16,
            fill=(24, 28, 35),
            outline=accent,
            width=3,
        )
        draw.text(
            (left + 20, card_top + 16),
            title,
            font=_font(24, bold=True),
            fill=accent,
        )
        draw.multiline_text(
            (left + 20, card_top + 58),
            body,
            font=_font(25, bold=True),
            fill="white",
            spacing=9,
        )
    legend_y = 374
    legend = (
        ((0, 224, 255), "removed"),
        ((255, 145, 30), "introduced"),
        ((155, 160, 168), "persistent"),
    )
    draw.text((38, legend_y), "Transition:", font=_font(24, bold=True), fill="white")
    legend_x = 190
    for color, label in legend:
        draw.rounded_rectangle(
            (legend_x, legend_y + 1, legend_x + 27, legend_y + 28),
            radius=4,
            fill=color,
        )
        draw.text(
            (legend_x + 38, legend_y - 2),
            label,
            font=_font(24),
            fill=(225, 225, 225),
        )
        legend_x += 190

    y = header_height
    for map_name in FREQUENCY_ADAPTIVE_FOCUSED_MAPS:
        crop = evidence["crops"][map_name]
        crop_box = tuple(crop["crop_box_xyxy"])
        if len(crop_box) != 4:
            raise ValueError("adaptive evidence crop box is invalid")
        left, top, right, bottom = crop_box
        if right - left != source_size or bottom - top != source_size:
            raise ValueError("adaptive evidence crop must be 96x96")
        draw.rectangle((0, y, width, y + section_height), fill=(16, 19, 24))
        draw.text(
            (36, y + 18),
            _display_name(map_name).upper(),
            font=_font(38, bold=True),
            fill="white",
        )
        draw.text(
            (36, y + 68),
            (
                f"shared crop x={left}:{right}, y={top}:{bottom} · selected by "
                "net removed → removed → fewest introduced → quiet support → top/left"
            ),
            font=_font(24),
            fill=(200, 207, 216),
        )
        for column, label in enumerate(columns):
            _draw_centered_text(
                draw,
                (gutter + column * tile_size, y + section_heading_height, tile_size, column_heading_height),
                label,
                _font(23, bold=True),
                "white",
            )
        row_y = y + section_heading_height + column_heading_height
        for profile in profiles:
            masks = profile_masks[profile]["maps"][map_name]
            quiet = profile_masks[profile]["quiet"]
            full_counts = _adaptive_transition_counts(
                masks["fixed"], masks["adaptive"], quiet
            )
            if full_counts != evidence["profiles"][profile]["maps"][map_name]:
                raise ValueError(
                    "adaptive evidence writer counts differ from metadata: "
                    f"{profile}/{map_name}"
                )
            crop_counts = _adaptive_transition_counts(
                masks["fixed"][top:bottom, left:right],
                masks["adaptive"][top:bottom, left:right],
                quiet[top:bottom, left:right],
            )
            if crop_counts != crop["profiles"][profile]:
                raise ValueError(
                    "adaptive evidence crop counts differ from metadata: "
                    f"{profile}/{map_name}"
                )
            draw.multiline_text(
                (36, row_y + 28),
                (
                    f"{profile.upper()}\n"
                    f"full {percentage(full_counts['fixed_v1_outlier_fraction'])} → "
                    f"{percentage(full_counts['adaptive_v1_outlier_fraction'])}\n"
                    f"crop removed {crop_counts['removed']}\n"
                    f"introduced {crop_counts['introduced']}\n"
                    f"persistent {crop_counts['persistent']}"
                ),
                font=_font(24, bold=True),
                fill="white",
                spacing=10,
            )
            with Image.open(
                baseline_camera
                / "material"
                / profile
                / "maps"
                / f"{map_name}.png"
            ) as image:
                fixed_raw = image.convert("L").crop(crop_box)
            with Image.open(
                regularized_camera
                / "material"
                / profile
                / "maps"
                / f"{map_name}.png"
            ) as image:
                adaptive_raw = image.convert("L").crop(crop_box)
            quiet_crop = quiet[top:bottom, left:right]
            fixed_crop = masks["fixed"][top:bottom, left:right]
            adaptive_crop = masks["adaptive"][top:bottom, left:right]
            fixed_diagnostic = np.zeros((source_size, source_size, 3), dtype=np.uint8)
            adaptive_diagnostic = np.zeros_like(fixed_diagnostic)
            fixed_diagnostic[quiet_crop] = (24, 27, 32)
            adaptive_diagnostic[quiet_crop] = (24, 27, 32)
            fixed_diagnostic[fixed_crop] = (245, 245, 245)
            adaptive_diagnostic[adaptive_crop] = (245, 245, 245)
            transition = np.zeros_like(fixed_diagnostic)
            transition[quiet_crop] = (18, 21, 25)
            transition[fixed_crop & adaptive_crop] = (155, 160, 168)
            transition[fixed_crop & ~adaptive_crop] = (0, 224, 255)
            transition[adaptive_crop & ~fixed_crop] = (255, 145, 30)
            panels = (
                fixed_raw.convert("RGB"),
                adaptive_raw.convert("RGB"),
                Image.fromarray(fixed_diagnostic, mode="RGB"),
                Image.fromarray(adaptive_diagnostic, mode="RGB"),
                Image.fromarray(transition, mode="RGB"),
            )
            for column, panel in enumerate(panels):
                enlarged = panel.resize(
                    (tile_size, tile_size), Image.Resampling.NEAREST
                )
                canvas.paste(enlarged, (gutter + column * tile_size, row_y))
            row_y += row_stride
        y += section_height

    nonfocused_direction = non_focused["direction"]
    nonfocused_delta = non_focused["adaptive_v1_outliers"] - non_focused[
        "fixed_v1_outliers"
    ]
    disclosure = (
        "Disclosure · Focused anisotropic/subsurface: "
        f"{relative_change_label(focused['relative_reduction_fraction'])}.\n"
        f"Non-focused maps: {percentage(non_focused['fixed_v1_outlier_fraction'])} → "
        f"{percentage(non_focused['adaptive_v1_outlier_fraction'])}, "
        f"{nonfocused_direction}, {nonfocused_delta:+d} outliers; "
        f"{evidence['non_focused_regression_count']} profile/map rows regress.\n"
        "The result does not support a claim that every map improves."
    )
    draw.multiline_text(
        (38, y + 24),
        disclosure,
        font=_font(25, bold=True),
        fill=(255, 196, 103) if nonfocused_direction == "regression" else (220, 225, 232),
        spacing=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _overview_pixel_detail_layout(
    map_names: Sequence[str],
    captions: Sequence[str],
    *,
    display_size: int,
    canvas_width: int,
) -> dict[str, Any]:
    map_names = tuple(map_names)
    captions = tuple(captions)
    if (
        not map_names
        or len(map_names) != len(captions)
        or display_size <= 0
        or canvas_width <= 0
    ):
        raise ValueError("overview pixel-detail layout inputs are invalid")
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    title_font = _font(27, bold=True)
    variant_font = _font(20, bold=True)
    caption_font = _font(19, bold=True)
    variant_labels = ("Baseline", "Regularized")
    variant_widths = [
        draw.textbbox((0, 0), label, font=variant_font)[2]
        for label in variant_labels
    ]
    slot_width = max(display_size, max(variant_widths) + 24)
    content_width = 2 * slot_width + OVERVIEW_DETAIL_VARIANT_GAP
    caption_measurements = [
        draw.multiline_textbbox(
            (0, 0),
            caption,
            font=caption_font,
            spacing=5,
            align="center",
        )
        for caption in captions
    ]
    caption_height = max(
        72,
        max(bounds[3] - bounds[1] for bounds in caption_measurements) + 24,
    )
    group_widths = []
    for map_name, caption_bounds in zip(map_names, caption_measurements):
        title_bounds = draw.textbbox(
            (0, 0),
            _display_name(map_name),
            font=title_font,
        )
        group_widths.append(
            max(
                content_width + 32,
                title_bounds[2] - title_bounds[0] + 40,
                caption_bounds[2] - caption_bounds[0] + 40,
            )
        )
    total_width = sum(group_widths) + OVERVIEW_DETAIL_GROUP_GAP * (
        len(group_widths) - 1
    )
    if total_width > canvas_width - 72:
        raise ValueError(
            "overview pixel-detail cards do not fit inside the report canvas"
        )
    cursor = (canvas_width - total_width) // 2
    groups = []
    for map_name, caption, group_width in zip(
        map_names,
        captions,
        group_widths,
    ):
        group_left = cursor
        content_left = group_left + (group_width - content_width) // 2
        slots = []
        label_bounds = []
        image_boxes = []
        for variant_index, label in enumerate(variant_labels):
            slot_left = content_left + variant_index * (
                slot_width + OVERVIEW_DETAIL_VARIANT_GAP
            )
            slot = (slot_left, 0, slot_width, 32)
            slots.append(slot)
            label_bounds.append(
                _centered_text_bounds(draw, slot, label, variant_font)
            )
            image_left = slot_left + (slot_width - display_size) // 2
            image_boxes.append((image_left, 0, display_size, display_size))
        group_box = (group_left, 0, group_width, 1)
        groups.append(
            {
                "map": map_name,
                "caption": caption,
                "box": group_box,
                "title_bounds": _centered_text_bounds(
                    draw,
                    (group_left, 0, group_width, 34),
                    _display_name(map_name),
                    title_font,
                ),
                "variant_slots": slots,
                "variant_label_bounds": label_bounds,
                "image_boxes": image_boxes,
                "caption_bounds": _centered_multiline_text_bounds(
                    draw,
                    (group_left, 0, group_width, caption_height),
                    caption,
                    caption_font,
                    spacing=5,
                ),
            }
        )
        cursor += group_width + OVERVIEW_DETAIL_GROUP_GAP
    return {
        "display_size": display_size,
        "slot_width": slot_width,
        "variant_gap": OVERVIEW_DETAIL_VARIANT_GAP,
        "group_gap": OVERVIEW_DETAIL_GROUP_GAP,
        "caption_height": caption_height,
        "total_width": total_width,
        "groups": groups,
    }


def _frequency_overview_headlines(
    gate_report: dict[str, Any],
    relighting_line: str,
    adaptive_evidence: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    if not isinstance(gate_report, dict):
        raise ValueError("frequency overview requires cleanup qualification metadata")
    gates = gate_report.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("frequency overview qualification has no gates")

    def reduction_line(gate_name: str, label: str) -> str:
        gate = gates.get(gate_name)
        if not isinstance(gate, dict):
            raise ValueError(f"frequency overview is missing gate {gate_name}")
        values = (
            gate.get("baseline_value"),
            gate.get("regularized_value"),
            gate.get("observed"),
        )
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError(
                f"frequency overview gate {gate_name} has invalid headline values"
            )
        baseline, regularized, reduction = (float(value) for value in values)
        return (
            f"{label}: {100.0 * baseline:.2f}% → "
            f"{100.0 * regularized:.2f}% · "
            f"{100.0 * reduction:.1f}% reduction"
        )

    band_gate = gates.get("aggregate_guide_textured_band3_8_amplitude_ratio")
    if not isinstance(band_gate, dict):
        raise ValueError("frequency overview is missing the 3–8 px texture gate")
    band_ratio = band_gate.get("observed")
    band_threshold = band_gate.get("threshold")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in (band_ratio, band_threshold)
    ):
        raise ValueError("frequency overview texture gate has invalid values")
    try:
        status = str(gate_report["qualification_status"])
        thresholds_met = int(gate_report["thresholds_met"])
        gate_count = int(gate_report["gate_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("frequency overview qualification status is invalid") from exc
    if adaptive_evidence is not None:
        if adaptive_evidence.get("schema") != FREQUENCY_ADAPTIVE_EVIDENCE_SCHEMA:
            raise ValueError("frequency overview adaptive evidence is invalid")
        focused = adaptive_evidence["focused_summary"]
        all_maps = adaptive_evidence["all_map_summary"]
        non_focused = adaptive_evidence["non_focused_summary"]
        all_gate = gates["aggregate_quiet_outlier_relative_reduction"]
        edge_gate = gates["guide_edge_gradient_magnitude_ratio"]

        def fraction(value: float | None) -> str:
            return "n/a" if value is None else f"{100.0 * value:.2f}%"

        def relative_change_label(value: float | None) -> str:
            if value is None:
                return "n/a"
            return (
                f"{100.0 * value:.1f}% reduction"
                if value >= 0.0
                else f"{-100.0 * value:.1f}% increase"
            )

        nonfocused_delta = (
            int(non_focused["adaptive_v1_outliers"])
            - int(non_focused["fixed_v1_outliers"])
        )
        all_gate_threshold = float(all_gate["threshold"])
        return (
            (
                f"Map-cleanup qualification: {status} {thresholds_met}/{gate_count} · "
                "focused cleanup and safeguards are reported separately."
            ),
            (
                "Focused anisotropic + subsurface quiet outliers: "
                f"{fraction(focused['fixed_v1_outlier_fraction'])} → "
                f"{fraction(focused['adaptive_v1_outlier_fraction'])} · "
                f"{relative_change_label(focused['relative_reduction_fraction'])}"
            ),
            (
                "All-map quiet outliers: "
                f"{fraction(all_maps['fixed_v1_outlier_fraction'])} → "
                f"{fraction(all_maps['adaptive_v1_outlier_fraction'])} · "
                f"{100.0 * all_gate_threshold:.0f}% gate "
                f"{'met' if all_gate['meets_threshold'] else 'MISS'}"
            ),
            (
                "Non-focused maps: "
                f"{fraction(non_focused['fixed_v1_outlier_fraction'])} → "
                f"{fraction(non_focused['adaptive_v1_outlier_fraction'])} · "
                f"{non_focused['direction']} {nonfocused_delta:+d} outliers"
            ),
            (
                "Safeguards: 3–8 px amplitude retained "
                f"{100.0 * float(band_ratio):.2f}% · edge ratio "
                f"{100.0 * float(edge_gate['observed_minimum']):.1f}–"
                f"{100.0 * float(edge_gate['observed_maximum']):.1f}%"
            ),
            relighting_line,
            (
                "Scope: focused-map change does not mean every material "
                "map or the reconstruction improves."
            ),
        )
    return (
        (
            f"Map-cleanup qualification: {status} {thresholds_met}/{gate_count} · "
            "PASS = cleanup under relighting safeguards, not reconstruction improvement."
        ),
        reduction_line(
            "aggregate_quiet_outlier_relative_reduction",
            "Aggregate quiet outliers",
        ),
        reduction_line(
            "olat_fixed_hotspot_subsurface_outlier_relative_reduction",
            "Fixed OLAT subsurface-hotspot outliers",
        ),
        reduction_line(
            "olat_fixed_hotspot_specular_outlier_relative_reduction",
            "Fixed OLAT specular-hotspot outliers",
        ),
        (
            "Guide-textured 3–8 px amplitude retained: "
            f"{100.0 * float(band_ratio):.1f}% · safeguard "
            f"≥{100.0 * float(band_threshold):.0f}%"
        ),
        relighting_line,
        (
            "Scope: exported-map diagnostics do not establish overall material "
            "quality or preservation of all texture."
        ),
    )


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
    header_height = 537
    frequency_cleanup = summary.get("frequency_cleanup")
    adaptive_evidence = (
        frequency_cleanup.get("adaptive_cleanup_evidence")
        if isinstance(frequency_cleanup, dict)
        else None
    )
    if isinstance(adaptive_evidence, dict):
        adaptive_variant_labels = adaptive_evidence.get("variant_labels")
        if not isinstance(adaptive_variant_labels, dict) or set(
            adaptive_variant_labels
        ) != {"baseline", "regularized"}:
            raise ValueError("adaptive overview variant labels are invalid")
        fixed_variant_label = adaptive_variant_labels["baseline"]
        adaptive_variant_label = adaptive_variant_labels["regularized"]
        if not all(
            isinstance(label, str) and label
            for label in (fixed_variant_label, adaptive_variant_label)
        ):
            raise ValueError("adaptive overview variant labels are invalid")
        hero_maps = ("anisotropic", "subsurface", "roughness", "specular")
    else:
        fixed_variant_label = "Baseline"
        adaptive_variant_label = "Regularized"
        hero_maps = ("roughness", "specular", "subsurface", "anisotropic")
    material_left = OVERVIEW_MATERIAL_LEFT
    material_tile_width = 170
    with Image.open(
        baseline_camera / "material" / profiles[0] / "maps" / "roughness.png"
    ) as image:
        source_width, source_height = image.size
        material_tile_height = round(
            material_tile_width * image.height / image.width
        )
    material_heading_height = 128
    material_caption_height = 52
    material_row_height = material_tile_height + material_caption_height
    material_top = header_height
    material_end = (
        material_top
        + material_heading_height
        + material_row_height * len(profiles)
    )

    overview_profile = "mix" if "mix" in profiles else profiles[-1]
    detail_profile = overview_profile
    impulse_cleanup = summary.get("impulse_cleanup")
    impulse_profile_cleanup = None
    impulse_artifact = None
    frequency_profile_cleanup = None
    if (
        isinstance(impulse_cleanup, dict)
        and impulse_cleanup.get("available") is True
    ):
        impulse_profile_cleanup = impulse_cleanup["profiles"][overview_profile]
        detail_maps = _rank_impulse_detail_maps(impulse_profile_cleanup)
        acquisition = json.loads(
            (
                regularized_camera
                / "material"
                / overview_profile
                / "acquisition.json"
            ).read_text(encoding="utf-8")
        )
        impulse_artifact = _load_impulse_frozen_artifact(
            regularized_camera / "material" / overview_profile,
            acquisition,
            (source_height, source_width),
        )
    elif (
        isinstance(frequency_cleanup, dict)
        and frequency_cleanup.get("available") is True
    ):
        presentation = frequency_cleanup.get("presentation")
        if not isinstance(presentation, dict):
            raise ValueError("frequency overview is missing presentation metadata")
        detail_profile = presentation.get("profile")
        if not isinstance(detail_profile, str) or detail_profile not in profiles:
            raise ValueError("frequency overview profile is invalid")
        frequency_profile_cleanup = frequency_cleanup["profiles"][detail_profile]
        detail_maps = (
            tuple(FREQUENCY_ADAPTIVE_FOCUSED_MAPS)
            if isinstance(adaptive_evidence, dict)
            else ("subsurface", "specular")
        )
    else:
        ranked_detail_maps = sorted(
            (
                (
                    metrics["comparison"]["guide_edge_gradient_magnitude_ratio"],
                    map_name,
                )
                for map_name, metrics in map_metrics[overview_profile].items()
                if metrics["comparison"]["meaningful_guide_edge_gradient"]
                and metrics["comparison"]["guide_edge_gradient_magnitude_ratio"]
                is not None
            ),
            key=lambda item: item[0],
        )
        detail_maps = tuple(map_name for _, map_name in ranked_detail_maps[:2])
        if len(detail_maps) < 2:
            detail_maps += tuple(
                map_name
                for map_name in ("specular", "roughness")
                if map_name not in detail_maps
            )[: 2 - len(detail_maps)]
    if frequency_profile_cleanup is not None:
        detail_source_size = 96
        detail_display_size = 192
    else:
        detail_source_size = min(128, source_width, source_height)
        detail_display_size = 256
    detail_captions = []
    for map_name in detail_maps:
        comparison = map_metrics[detail_profile][map_name]["comparison"]
        if impulse_profile_cleanup is not None:
            cleanup_map = impulse_profile_cleanup["maps"][map_name]
            detail_captions.append(
                f"flags {cleanup_map['flagged_centers']}\n"
                "outside-own-mask max Δ "
                f"{cleanup_map['exported_change_outside_own_flags']['maximum_absolute']:.6f}"
            )
        elif frequency_profile_cleanup is not None:
            if isinstance(adaptive_evidence, dict):
                full = adaptive_evidence["profiles"][detail_profile]["maps"][
                    map_name
                ]
                crop = adaptive_evidence["crops"][map_name]["profiles"][
                    detail_profile
                ]
                detail_captions.append(
                    f"full {100.0 * full['fixed_v1_outlier_fraction']:.2f}% → "
                    f"{100.0 * full['adaptive_v1_outlier_fraction']:.2f}%\n"
                    f"crop removed {crop['removed']} · introduced {crop['introduced']}"
                )
            else:
                cleanup_map = frequency_profile_cleanup["maps"][map_name]
                count_labels = frequency_profile_cleanup.get(
                    "annotation_labels",
                    {"updated": "updates", "consensus": "consensus"},
                )
                detail_captions.append(
                    f"{count_labels['updated']} {cleanup_map['updated_entries']}\n"
                    f"{count_labels['consensus']} {cleanup_map['consensus_entries']}"
                )
        else:
            detail_captions.append(
                "gradient magnitude ratio "
                f"{_format_ratio(comparison['guide_edge_gradient_magnitude_ratio'])}\n"
                "signed cosine "
                f"{_format_cosine(comparison['guide_edge_signed_gradient_cosine'])}"
            )
    detail_layout = _overview_pixel_detail_layout(
        detail_maps,
        detail_captions,
        display_size=detail_display_size,
        canvas_width=width,
    )
    detail_heading_height = 208
    detail_caption_height = detail_layout["caption_height"]
    detail_top = material_end + 36
    detail_end = (
        detail_top
        + detail_heading_height
        + detail_display_size
        + detail_caption_height
    )

    olat_case = summary["displayed_evaluation_cases"]["olat"]["overview"][
        "case_id"
    ]
    with Image.open(
        baseline_camera / "evaluation" / "olat" / "cases" / olat_case / "reference.png"
    ) as image:
        evaluation_tile_width = 210
        evaluation_tile_height = round(
            evaluation_tile_width * image.height / image.width
        )
    evaluation_top = detail_end + 36
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
    if contract.get("active_frequency_upgrade"):
        baseline_weight = float(contract["baseline_tv_weight"])
        regularized_weight = float(contract["regularized_tv_weight"])
        weight_text = (
            f"λ={baseline_weight:g} both"
            if math.isclose(
                baseline_weight,
                regularized_weight,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            else f"λ={baseline_weight:g} → {regularized_weight:g}"
        )
        contract_text = (
            f"Controlled A/B · {fixed_variant_label} frequency consensus → "
            f"{adaptive_variant_label} · "
            f"{weight_text}"
        )
    else:
        contract_text = (
            "Controlled A/B · "
            f"{_display_regularizer(contract['regularization_kind'])} λ "
            f"{contract['baseline_tv_weight']:g} → "
            f"{contract['regularized_tv_weight']:g}"
        )
    draw.text(
        (42, 92),
        contract_text,
        font=_font_for_width(
            draw,
            contract_text,
            max_width=width - 84,
            preferred_size=30,
            minimum_size=24,
            bold=True,
        ),
        fill=(151, 205, 255),
    )
    spatial = summary["aggregate_map_spatial_statistics"]
    preservation = summary["guide_edge_preservation"]
    olat = summary["aggregate_evaluation"]["olat"]
    hdri = summary["aggregate_evaluation"]["hdri"]
    quiet_variation = spatial["quiet_region_neighbor_variation"]
    quiet_outliers = spatial[
        "quiet_region_median5_residual_outlier_fraction_gt_0p05"
    ]
    magnitude_distribution = preservation["gradient_magnitude_ratio"]
    cosine_distribution = preservation["signed_gradient_cosine"]

    def worst_metric_text(distribution: dict[str, Any], *, ratio: bool) -> str:
        worst = distribution["worst"]
        if worst is None:
            return "n/a"
        value = (
            _format_ratio(worst["value"], digits=1)
            if ratio
            else _format_cosine(worst["value"], digits=3)
        )
        return (
            f"{value} ({worst['profile'].upper()} "
            f"{_display_name(worst['map'])})"
        )

    def optional_decimal(value: float | None, digits: int) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    quiet_line = (
        f"Quiet variation {quiet_variation['baseline']:.4f} → "
        f"{quiet_variation['regularized']:.4f} "
        f"({_format_relative_change(quiet_variation['relative_change_fraction'])}) · "
        f"median-residual outliers {100.0 * quiet_outliers['baseline']:.2f}% → "
        f"{100.0 * quiet_outliers['regularized']:.2f}%"
    )
    gradient_line = (
        "Guide-edge magnitude ratio: median "
        f"{_format_ratio(magnitude_distribution['median'], digits=1)} · "
        f"worst {worst_metric_text(magnitude_distribution, ratio=True)}"
    )
    cosine_line = (
        "Guide-edge signed cosine: median "
        f"{_format_cosine(cosine_distribution['median'], digits=3)} · "
        f"worst {worst_metric_text(cosine_distribution, ratio=False)}"
    )
    relighting_line = (
        f"Acquisition aggregate: OLAT ΔPSNR {olat['mean_psnr']['delta']:+.3f} dB / "
        f"ΔSSIM {olat['mean_ssim_global']['delta']:+.4f} · HDRI "
        f"ΔPSNR {hdri['mean_psnr']['delta']:+.3f} dB / "
        f"ΔSSIM {hdri['mean_ssim_global']['delta']:+.4f}"
    )
    scope_line = (
        "Scope: exported-map diagnostics do not establish that all material "
        "texture is preserved."
    )
    cleanup = summary.get("impulse_cleanup")
    frequency = summary.get("frequency_cleanup")
    if isinstance(cleanup, dict) and cleanup.get("available") is True:
        cleanup_aggregate = cleanup["aggregate"]
        flagged = cleanup_aggregate["flagged"]
        distance = cleanup_aggregate["mean_absolute_distance_to_frozen_target"]
        resolved = cleanup_aggregate["resolved_to_dead_zone"]
        outside = cleanup_aggregate[
            "exported_change_outside_spatial_flag_union"
        ]
        lines = [
            quiet_line,
            (
                f"Impulse flags: {flagged['map_entries']} map entries "
                f"({_format_ratio(flagged['map_entry_fraction'], digits=3)}) · "
                f"resolved to dead zone {_format_ratio(resolved['fraction'], digits=1)}"
            ),
            (
                "Frozen-target MAE: "
                f"{optional_decimal(distance['baseline'], 5)} → "
                f"{optional_decimal(distance['candidate'], 5)} · "
                "outside flag-union change mean "
                f"{optional_decimal(outside['mean_absolute'], 6)} / "
                f"max {optional_decimal(outside['maximum_absolute'], 6)}"
            ),
            gradient_line,
            cosine_line,
            relighting_line,
            scope_line,
        ]
    elif isinstance(frequency, dict) and frequency.get("available") is True:
        gate_report = summary.get("scalar_map_cleanup_qualification")
        lines = list(
            _frequency_overview_headlines(
                gate_report,
                relighting_line,
                adaptive_evidence=(
                    adaptive_evidence
                    if isinstance(adaptive_evidence, dict)
                    else None
                ),
            )
        )
    else:
        outlier_line = (
            "Quiet-region median-residual outliers (>0.05): "
            f"{100.0 * quiet_outliers['baseline']:.2f}% → "
            f"{100.0 * quiet_outliers['regularized']:.2f}% "
            f"({100.0 * (quiet_outliers['regularized'] - quiet_outliers['baseline']):+.2f} pp)"
        )
        lines = [
            quiet_line,
            outlier_line,
            gradient_line,
            cosine_line,
            relighting_line,
            (
                "Cleanup-specific frozen-target metrics: "
                + cleanup["reason"]
                if isinstance(cleanup, dict)
                else "No cleanup-specific frozen-target artifact for this regularizer."
            ),
            scope_line,
        ]
    for index, line in enumerate(lines):
        line_font = _font_for_width(
            draw,
            line,
            max_width=width - 84,
            preferred_size=27,
            minimum_size=22,
            bold=True,
        )
        draw.text(
            (42, 140 + index * 52),
            line,
            font=line_font,
            fill="white",
        )

    draw.text(
        (36, material_top + 10),
        (
            f"Material maps · {fixed_variant_label} and "
            f"{adaptive_variant_label} side by side"
            if isinstance(adaptive_evidence, dict)
            else "Material maps · baseline and regularized side by side"
        ),
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
        variant_labels = (
            (fixed_variant_label, adaptive_variant_label)
            if isinstance(adaptive_evidence, dict)
            else ("Baseline", "Regularized")
        )
        for variant_index, variant_label in enumerate(variant_labels):
            _draw_centered_text(
                draw,
                (
                    group_x + variant_index * material_tile_width,
                    material_top + 96,
                    material_tile_width,
                    28,
                ),
                variant_label,
                _font(20, bold=True),
                (210, 210, 210),
            )
    for profile_index, profile in enumerate(profiles):
        y = material_top + material_heading_height + profile_index * material_row_height
        baseline_mean = np.mean(
            [
                map_metrics[profile][name]["baseline"][
                    "quiet_region_neighbor_variation"
                ]
                for name in hero_maps
            ]
        )
        regularized_mean = np.mean(
            [
                map_metrics[profile][name]["regularized"][
                    "quiet_region_neighbor_variation"
                ]
                for name in hero_maps
            ]
        )
        profile_entries = [
            (name, map_metrics[profile][name]["comparison"])
            for name in hero_maps
            if map_metrics[profile][name]["comparison"][
                "meaningful_guide_edge_gradient"
            ]
        ]
        profile_ratios = [
            entry["guide_edge_gradient_magnitude_ratio"]
            for _, entry in profile_entries
            if entry["guide_edge_gradient_magnitude_ratio"] is not None
        ]
        profile_cosines = [
            entry["guide_edge_signed_gradient_cosine"]
            for _, entry in profile_entries
            if entry["guide_edge_signed_gradient_cosine"] is not None
        ]
        worst_name, worst_entry = min(
            (
                (name, entry)
                for name, entry in profile_entries
                if entry["guide_edge_gradient_magnitude_ratio"] is not None
            ),
            key=lambda item: item[1]["guide_edge_gradient_magnitude_ratio"],
            default=(None, None),
        )
        profile_text = (
            f"{profile.upper()} fit\n"
            f"Quiet {baseline_mean:.3f} → {regularized_mean:.3f}\n"
            f"Magnitude median "
            f"{_format_ratio(float(np.median(profile_ratios)) if profile_ratios else None)}\n"
            f"Worst {_display_name(worst_name) if worst_name else 'n/a'} "
            f"{_format_ratio(worst_entry['guide_edge_gradient_magnitude_ratio'] if worst_entry else None)}\n"
            f"Cosine median "
            f"{_format_cosine(float(np.median(profile_cosines)) if profile_cosines else None)}"
        )
        profile_font = _font(21, bold=True)
        profile_bounds = draw.multiline_textbbox(
            (36, y + 18),
            profile_text,
            font=profile_font,
            spacing=8,
        )
        if profile_bounds[2] > material_left - 12:
            raise ValueError(
                f"overview metrics for profile {profile!r} overlap the material grid"
            )
        draw.multiline_text(
            (36, y + 18),
            profile_text,
            font=profile_font,
            fill="white",
            spacing=8,
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
            comparison = map_metrics[profile][map_name]["comparison"]
            _draw_centered_text(
                draw,
                (
                    material_left + map_index * 2 * material_tile_width,
                    y + material_tile_height + 4,
                    2 * material_tile_width,
                    material_caption_height - 8,
                ),
                (
                    "magnitude "
                    f"{_format_ratio(comparison['guide_edge_gradient_magnitude_ratio'])} · "
                    "cosine "
                    f"{_format_cosine(comparison['guide_edge_signed_gradient_cosine'])}"
                ),
                _font(19, bold=True),
                (225, 225, 225),
            )

    draw.text(
        (36, detail_top + 10),
        f"Pixel detail check · {detail_profile.upper()} fit",
        font=_font(43, bold=True),
        fill="white",
    )
    if frequency_profile_cleanup is not None:
        if isinstance(adaptive_evidence, dict):
            detail_description = (
                "Focused-map 96×96 crops are selected across all profiles by "
                "net removed → removed → fewest introduced → quiet support → top/left.\n"
                "Algorithmic shared crops; no manual cherry-picking. See material/adaptive_cleanup_evidence.png."
            )
        else:
            detail_description = (
                "Fixed 96×96 crop x=96:192, y=160:256 enlarged 2× with "
                "nearest-neighbor (1 source pixel = 2×2 display pixels).\n"
                "Native 1:1 crop and full-map audit sheets remain separate."
            )
    elif impulse_artifact is not None:
        detail_description = (
            f"Each {detail_source_size}×{detail_source_size} native crop is "
            "centered on its densest flagged window and enlarged with "
            "nearest-neighbor."
        )
    else:
        detail_description = (
            f"Centered {detail_source_size}×{detail_source_size} source crop "
            "shown with nearest-neighbor enlargement."
        )
    draw.multiline_text(
        (42, detail_top + 65),
        detail_description,
        font=_font(22),
        fill=(210, 210, 210),
        spacing=6,
    )
    centered_crop_box = (
        (source_width - detail_source_size) // 2,
        (source_height - detail_source_size) // 2,
        (source_width - detail_source_size) // 2 + detail_source_size,
        (source_height - detail_source_size) // 2 + detail_source_size,
    )
    detail_image_y = detail_top + detail_heading_height
    card_top = detail_top + 123
    card_bottom = (
        detail_image_y
        + detail_display_size
        + detail_caption_height
        - 4
    )
    for map_index, map_name in enumerate(detail_maps):
        group = detail_layout["groups"][map_index]
        group_x, _, group_width, _ = group["box"]
        draw.rounded_rectangle(
            (group_x, card_top, group_x + group_width, card_bottom),
            radius=12,
            fill=(20, 23, 28),
            outline=(72, 78, 88),
            width=2,
        )
        if frequency_profile_cleanup is not None:
            crop_box = (
                tuple(adaptive_evidence["crops"][map_name]["crop_box_xyxy"])
                if isinstance(adaptive_evidence, dict)
                else FREQUENCY_DETAIL_CROP_BOX
            )
        elif impulse_artifact is not None:
            crop_box = _densest_flagged_crop_box(
                impulse_artifact["maps"][map_name]["mask"],
                detail_source_size,
            )
        else:
            crop_box = centered_crop_box
        _draw_centered_text(
            draw,
            (group_x, detail_top + 129, group_width, 34),
            _display_name(map_name),
            _font(27, bold=True),
            "white",
        )
        first_slot = group["variant_slots"][0]
        second_slot = group["variant_slots"][1]
        divider_x = (
            first_slot[0]
            + first_slot[2]
            + second_slot[0]
        ) // 2
        draw.line(
            (
                divider_x,
                detail_top + 170,
                divider_x,
                detail_image_y + detail_display_size,
            ),
            fill=(62, 67, 76),
            width=2,
        )
        for variant_index, (variant_label, camera) in enumerate(
            (
                (
                    (fixed_variant_label, baseline_camera),
                    (adaptive_variant_label, regularized_camera),
                )
                if isinstance(adaptive_evidence, dict)
                else (("Baseline", baseline_camera), ("Regularized", regularized_camera))
            )
        ):
            slot_x, _, slot_width, _ = group["variant_slots"][variant_index]
            image_x, _, _, _ = group["image_boxes"][variant_index]
            _draw_centered_text(
                draw,
                (slot_x, detail_top + 168, slot_width, 32),
                variant_label,
                _font(20, bold=True),
                (220, 220, 220),
            )
            with Image.open(
                camera
                / "material"
                / detail_profile
                / "maps"
                / f"{map_name}.png"
            ) as image:
                crop = image.convert("RGB").crop(crop_box)
                if crop.size != (detail_display_size, detail_display_size):
                    crop = crop.resize(
                        (detail_display_size, detail_display_size),
                        Image.Resampling.NEAREST,
                    )
            canvas.paste(crop, (image_x, detail_image_y))
        _draw_centered_multiline_text(
            draw,
            (
                group_x,
                detail_image_y + detail_display_size + 6,
                group_width,
                detail_caption_height - 12,
            ),
            detail_captions[map_index],
            _font(19, bold=True),
            "white",
            spacing=5,
        )

    draw.text(
        (36, evaluation_top + 10),
        (
            f"Relighting spot-check · {overview_profile.upper()} fit · "
            "median ΔPSNR cases"
        ),
        font=_font(43, bold=True),
        fill="white",
    )
    for lighting_index, lighting in enumerate(EVALUATION_LIGHTING):
        block_y = (
            evaluation_top
            + evaluation_heading_height
            + lighting_index * evaluation_row_height
        )
        selection = summary["displayed_evaluation_cases"][lighting]["overview"]
        case_id = selection["case_id"]
        displayed_case_id = (
            case_id
            if len(case_id) <= 31
            else f"{case_id[:15]}…{case_id[-15:]}"
        )
        case_metrics = _case_png_row(
            summary["case_png_evaluation"],
            lighting=lighting,
            case_id=case_id,
            profile=overview_profile,
        )
        panels = _evaluation_case_panels(
            baseline_camera,
            regularized_camera,
            lighting=lighting,
            case_id=case_id,
            profile=overview_profile,
        )
        columns = [name for name, _, _ in panels]
        evaluation_left = width - len(columns) * evaluation_tile_width - 30
        metric_baseline = case_metrics["baseline"]
        metric_regularized = case_metrics["regularized"]
        draw.multiline_text(
            (36, block_y + 56),
            (
                f"{lighting.upper()} · MEDIAN ΔPSNR CASE\n"
                f"{displayed_case_id}\n"
                f"PSNR {metric_baseline['psnr']:.2f} → "
                f"{metric_regularized['psnr']:.2f} "
                f"({case_metrics['delta']['psnr']:+.3f} dB)\n"
                f"SSIM {metric_baseline['ssim_global']:.3f} → "
                f"{metric_regularized['ssim_global']:.3f} "
                f"({case_metrics['delta']['ssim_global']:+.4f})"
            ),
            font=_font(23, bold=True),
            fill="white",
            spacing=7,
        )
        for column, (name, image, contain) in enumerate(panels):
            x = evaluation_left + column * evaluation_tile_width
            _draw_centered_text(
                draw,
                (x, block_y, evaluation_tile_width, 48),
                name,
                _font(23, bold=True),
                "white",
            )
            panel = (
                _contain_image(
                    image,
                    (evaluation_tile_width, evaluation_tile_height),
                    fill=(12, 12, 12),
                )
                if contain
                else image.convert("RGB").resize(
                    (evaluation_tile_width, evaluation_tile_height),
                    Image.Resampling.LANCZOS,
                )
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


def _font_for_width(
    draw,
    text: str,
    *,
    max_width: int,
    preferred_size: int,
    minimum_size: int,
    bold: bool = False,
):
    """Return the largest requested report font that fits without clipping."""
    if max_width <= 0:
        raise ValueError("maximum text width must be positive")
    if preferred_size < minimum_size or minimum_size <= 0:
        raise ValueError("font size bounds are invalid")
    for size in range(preferred_size, minimum_size - 1, -1):
        font = _font(size, bold=bold)
        bounds = draw.textbbox((0, 0), text, font=font)
        if bounds[2] - bounds[0] <= max_width:
            return font
    raise ValueError(
        f"report text does not fit within {max_width}px at the minimum "
        f"font size {minimum_size}: {text}"
    )


def _centered_text_bounds(draw, box, text, font) -> tuple[float, float, float, float]:
    x, y, width, height = box
    bounds = draw.textbbox((0, 0), text, font=font)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    origin_x = x + (width - text_width) / 2 - bounds[0]
    origin_y = y + (height - text_height) / 2 - bounds[1]
    return (
        origin_x + bounds[0],
        origin_y + bounds[1],
        origin_x + bounds[2],
        origin_y + bounds[3],
    )


def _centered_multiline_text_bounds(
    draw,
    box,
    text,
    font,
    *,
    spacing: int,
) -> tuple[float, float, float, float]:
    x, y, width, height = box
    bounds = draw.multiline_textbbox(
        (0, 0),
        text,
        font=font,
        spacing=spacing,
        align="center",
    )
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    origin_x = x + (width - text_width) / 2 - bounds[0]
    origin_y = y + (height - text_height) / 2 - bounds[1]
    return (
        origin_x + bounds[0],
        origin_y + bounds[1],
        origin_x + bounds[2],
        origin_y + bounds[3],
    )


def _draw_centered_text(draw, box, text, font, fill) -> None:
    bounds = _centered_text_bounds(draw, box, text, font)
    raw_bounds = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (bounds[0] - raw_bounds[0], bounds[1] - raw_bounds[1]),
        text,
        font=font,
        fill=fill,
    )


def _draw_centered_multiline_text(
    draw,
    box,
    text,
    font,
    fill,
    *,
    spacing: int,
) -> None:
    bounds = _centered_multiline_text_bounds(
        draw,
        box,
        text,
        font,
        spacing=spacing,
    )
    raw_bounds = draw.multiline_textbbox(
        (0, 0),
        text,
        font=font,
        spacing=spacing,
        align="center",
    )
    draw.multiline_text(
        (bounds[0] - raw_bounds[0], bounds[1] - raw_bounds[1]),
        text,
        font=font,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def _display_name(name: str) -> str:
    return {
        "specularTint": "specular tint",
        "clearcoatGloss": "clearcoat gloss",
    }.get(name, name)


def _display_regularizer(kind: str) -> str:
    return {
        "l1": "L1 TV",
        "edge-charbonnier": "edge-aware Charbonnier",
        "impulse-median": "post-fit impulse proximal",
        "frequency-consensus": "post-fit frequency consensus",
        "frequency-consensus-adaptive": "adaptive frequency consensus",
    }.get(kind, kind)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _validate_report(
    stage: Path,
    profiles: Sequence[str],
    *,
    require_frequency_detail: bool = False,
    require_adaptive_cleanup_evidence: bool = False,
) -> None:
    required = [stage / "overview.png", stage / "summary.json", stage / "metrics.csv"]
    required.extend(stage / "material" / f"{profile}.png" for profile in profiles)
    required.extend(
        stage / "evaluation" / f"{lighting}.png"
        for lighting in EVALUATION_LIGHTING
    )
    if require_frequency_detail:
        required.extend(
            (
                stage / "material" / "frequency_hotspot_1to1.png",
                stage / "material" / "frequency_fullmaps_1to1.png",
            )
        )
    if require_adaptive_cleanup_evidence:
        evidence_path = stage / "material" / "adaptive_cleanup_evidence.png"
        required.append(evidence_path)
        summary = json.loads((stage / "summary.json").read_text(encoding="utf-8"))
        cleanup = summary.get("frequency_cleanup")
        evidence = (
            cleanup.get("adaptive_cleanup_evidence")
            if isinstance(cleanup, dict)
            else None
        )
        artifact = (
            summary.get("artifacts", {})
            .get("frequency_diagnostics", {})
            .get("adaptive_cleanup_evidence")
        )
        if (
            not isinstance(evidence, dict)
            or evidence.get("schema") != FREQUENCY_ADAPTIVE_EVIDENCE_SCHEMA
            or evidence.get("available") is not True
            or artifact != "material/adaptive_cleanup_evidence.png"
        ):
            raise RuntimeError(
                "adaptive cleanup report metadata is incomplete or inconsistent"
            )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"regularization report is incomplete: {missing}")
    if require_adaptive_cleanup_evidence:
        with Image.open(stage / "material" / "adaptive_cleanup_evidence.png") as image:
            expected_width = (
                350
                + 5
                * FREQUENCY_ADAPTIVE_EVIDENCE_CROP_SIZE
                * FREQUENCY_ADAPTIVE_EVIDENCE_SCALE
                + 40
            )
            expected_height = 430 + len(FREQUENCY_ADAPTIVE_FOCUSED_MAPS) * (
                130
                + 64
                + len(profiles)
                * (
                    FREQUENCY_ADAPTIVE_EVIDENCE_CROP_SIZE
                    * FREQUENCY_ADAPTIVE_EVIDENCE_SCALE
                    + 26
                )
                + 28
            ) + 164
            if image.mode != "RGB" or image.size != (
                expected_width,
                expected_height,
            ):
                raise RuntimeError(
                    "adaptive cleanup evidence sheet has an invalid image contract"
                )


if __name__ == "__main__":
    main()
