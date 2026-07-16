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
MATERIAL_LABEL_GUTTER = 600
OVERVIEW_MATERIAL_LEFT = 360
DEFAULT_GUIDE_ALBEDO_SIGMA = 0.05
DEFAULT_GUIDE_NORMAL_SIGMA = 0.02
QUIET_GUIDE_PERCENTILE = 50.0
EDGE_GUIDE_PERCENTILE = 80.0
MEANINGFUL_EDGE_GRADIENT_MIN = 1.0 / 255.0
IMPULSE_FROZEN_ARTIFACT_SCHEMA = "ictpolarreal.impulse-frozen-artifact.v1"
EXPORT_QUANTIZATION_TOLERANCE = 0.5 / 255.0
_LEGACY_CHECKPOINT_SCHEMA = "ictpolarreal.end2end-checkpoint.v8"
_EDGE_CHECKPOINT_SCHEMA = "ictpolarreal.end2end-checkpoint.v9"
_STAGED_IMPULSE_CHECKPOINT_SCHEMA = "ictpolarreal.end2end-checkpoint.v10"
_PROXIMAL_V11_CHECKPOINT_SCHEMA = "ictpolarreal.end2end-checkpoint.v11"
_PROXIMAL_V12_CHECKPOINT_SCHEMA = "ictpolarreal.end2end-checkpoint.v12"
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
            "Dataset root used to reconstruct the acquisition's capture × n-dot-v "
            "fit mask. Preferred over --mask."
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
            else f"Baseline · λ={baseline_weight:g}"
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
        guide_regions,
        guide_diagnostic,
        impulse_cleanup,
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

        _write_metrics_csv(
            stage / "metrics.csv",
            map_metrics,
            evaluation_metrics,
            impulse_cleanup,
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
    if baseline_weight == regularized_weight:
        raise ValueError(
            "baseline and regularized runs use the same regularization weight"
        )
    baseline_regularizer_inactive = baseline_weight == 0.0
    if not baseline_regularizer_inactive and (
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
        )
        transition_modes.add(transition)
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
            transition=transition,
        )
        regularized_signature = _normalized_comparison_signature(
            regularized,
            variable_regularizer=baseline_regularizer_inactive,
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
            else "scalar-map regularization weight λ"
        ),
        "base_color_source": "dataset_albedo",
        "baseline_regularization_kind": baseline_kind,
        "regularization_kind": regularized_kind,
        "regularized_parameters": list(baseline_parameters),
        "baseline_tv_weight": baseline_weight,
        "regularized_tv_weight": regularized_weight,
        "baseline_regularizer_inactive": baseline_regularizer_inactive,
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
    }.get(kind, kind)


def _comparison_signature_transition(
    baseline: dict[str, Any],
    regularized: dict[str, Any],
    *,
    allow_legacy_transition: bool,
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

    padded = np.pad(values, 2, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (5, 5))
    median = np.median(windows, axis=(-2, -1))
    residual = np.abs(values - median)[mask]
    quiet_residual = np.abs(values - median)[guide_masks["quiet_pixels"]]
    return {
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
        "guide_edge_gradient_magnitude": edge_gradient_magnitude,
    }


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
        guide_masks, guide_metadata[profile] = _guide_region_masks(
            _read_rgb_map(maps_dir / "baseColor.png"),
            _read_rgb_map(maps_dir / "normal.png"),
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
    proximal_acquisition_schemas = {
        _PROXIMAL_V11_IDENTITY: "ictpolarreal.end2end-disney.v11",
        _PROXIMAL_V12_IDENTITY: "ictpolarreal.end2end-disney.v12",
    }
    if len(candidate_identities) != 1:
        raise ValueError(
            "impulse cleanup report requires one supported candidate identity; "
            f"found {candidate_identities}"
        )
    candidate_identity = next(iter(candidate_identities))
    expected_acquisition_schema = proximal_acquisition_schemas.get(
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
    guide_regions: dict[str, dict[str, Any]],
    guide_diagnostic: dict[str, Any],
    impulse_cleanup: dict[str, Any] | None,
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
    return {
        "schema": "ictpolarreal.regularization-comparison.v3",
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
        "per_map_spatial_statistics": map_metrics,
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
    top = 190
    width = left + tile_width * len(columns)
    height = top + tile_height * len(profiles)
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (24, 18),
        f"{lighting.upper()} held-out relighting",
        font=_font(45, bold=True),
        fill="white",
    )
    case_label = f"Shared case · {case_id}"
    draw.text(
        (24, 72),
        case_label,
        font=_font_for_width(
            draw,
            case_label,
            max_width=width - 48,
            preferred_size=27,
            minimum_size=20,
            bold=True,
        ),
        fill="white",
    )
    draw.text(
        (24, 112),
        "Errors use the acquisition report's fixed visualization scale.",
        font=_font(24),
        fill=(210, 210, 210),
    )
    for column, name in enumerate(columns):
        _draw_centered_text(
            draw,
            (left + column * tile_width, 152, tile_width, 36),
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
    impulse_cleanup: dict[str, Any] | None,
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
    impulse_cleanup = summary.get("impulse_cleanup")
    impulse_profile_cleanup = None
    impulse_artifact = None
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
    detail_source_size = min(128, source_width, source_height)
    detail_display_size = 256
    detail_gap = 80
    detail_heading_height = 160
    detail_caption_height = 55
    detail_top = material_end + 36
    detail_end = (
        detail_top
        + detail_heading_height
        + detail_display_size
        + detail_caption_height
    )

    olat_case = summary["representative_cases"]["olat"]
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
    draw.text(
        (42, 92),
        (
            "Controlled A/B · "
            f"{_display_regularizer(contract['regularization_kind'])} λ "
            f"{contract['baseline_tv_weight']:g} → "
            f"{contract['regularized_tv_weight']:g}"
        ),
        font=_font(30, bold=True),
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
        f"Evaluation: OLAT ΔPSNR {olat['mean_psnr']['delta']:+.3f} dB / "
        f"ΔSSIM {olat['mean_ssim_global']['delta']:+.4f} · HDRI "
        f"ΔPSNR {hdri['mean_psnr']['delta']:+.3f} dB / "
        f"ΔSSIM {hdri['mean_ssim_global']['delta']:+.4f}"
    )
    scope_line = (
        "Scope: exported-map diagnostics do not establish that all material "
        "texture is preserved."
    )
    cleanup = summary.get("impulse_cleanup")
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
        draw.text(
            (42, 140 + index * 52),
            line,
            font=_font(27, bold=True),
            fill="white",
        )

    draw.text(
        (36, material_top + 10),
        "Material maps · baseline and regularized side by side",
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
        for variant_index, variant_label in enumerate(("Baseline", "Regularized")):
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
        f"Pixel detail check · {overview_profile.upper()} fit",
        font=_font(43, bold=True),
        fill="white",
    )
    draw.text(
        (42, detail_top + 65),
        (
            (
                f"Each {detail_source_size}x{detail_source_size} native crop is "
                "centered on its densest flagged window; "
            )
            if impulse_artifact is not None
            else (
                f"Centered {detail_source_size}x{detail_source_size} source crop "
                "shown with nearest-neighbor enlargement; "
            )
        )
        + "each block is one exported-map pixel.",
        font=_font(24),
        fill=(210, 210, 210),
    )
    detail_total_width = (
        len(detail_maps) * 2 * detail_display_size
        + (len(detail_maps) - 1) * detail_gap
    )
    detail_left = (width - detail_total_width) // 2
    centered_crop_box = (
        (source_width - detail_source_size) // 2,
        (source_height - detail_source_size) // 2,
        (source_width - detail_source_size) // 2 + detail_source_size,
        (source_height - detail_source_size) // 2 + detail_source_size,
    )
    detail_image_y = detail_top + detail_heading_height
    for map_index, map_name in enumerate(detail_maps):
        crop_box = (
            _densest_flagged_crop_box(
                impulse_artifact["maps"][map_name]["mask"],
                detail_source_size,
            )
            if impulse_artifact is not None
            else centered_crop_box
        )
        group_x = detail_left + map_index * (
            2 * detail_display_size + detail_gap
        )
        _draw_centered_text(
            draw,
            (group_x, detail_top + 100, 2 * detail_display_size, 30),
            _display_name(map_name),
            _font(27, bold=True),
            "white",
        )
        for variant_index, (variant_label, camera) in enumerate(
            (("Baseline", baseline_camera), ("Regularized", regularized_camera))
        ):
            x = group_x + variant_index * detail_display_size
            _draw_centered_text(
                draw,
                (x, detail_top + 130, detail_display_size, 27),
                variant_label,
                _font(20, bold=True),
                (220, 220, 220),
            )
            with Image.open(
                camera
                / "material"
                / overview_profile
                / "maps"
                / f"{map_name}.png"
            ) as image:
                crop = image.convert("RGB").crop(crop_box).resize(
                    (detail_display_size, detail_display_size),
                    Image.Resampling.NEAREST,
                )
            canvas.paste(crop, (x, detail_image_y))
        comparison = map_metrics[overview_profile][map_name]["comparison"]
        if impulse_profile_cleanup is not None:
            cleanup_map = impulse_profile_cleanup["maps"][map_name]
            detail_caption = (
                f"flags {cleanup_map['flagged_centers']} · outside-own-mask max Δ "
                f"{cleanup_map['exported_change_outside_own_flags']['maximum_absolute']:.6f}"
            )
        else:
            detail_caption = (
                "gradient magnitude ratio "
                f"{_format_ratio(comparison['guide_edge_gradient_magnitude_ratio'])} · "
                "signed cosine "
                f"{_format_cosine(comparison['guide_edge_signed_gradient_cosine'])}"
            )
        _draw_centered_text(
            draw,
            (
                group_x,
                detail_image_y + detail_display_size + 7,
                2 * detail_display_size,
                detail_caption_height - 10,
            ),
            detail_caption,
            _font(21, bold=True),
            "white",
        )

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


def _display_regularizer(kind: str) -> str:
    return {
        "l1": "L1 TV",
        "edge-charbonnier": "edge-aware Charbonnier",
        "impulse-median": "post-fit impulse proximal",
    }.get(kind, kind)


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
