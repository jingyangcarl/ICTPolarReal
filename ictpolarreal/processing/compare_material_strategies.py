from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from ictpolarreal.processing import compare_regularization as pair
from ictpolarreal.processing import end2end_acquisition as acquisition_core


REPORT_SCHEMA = "ictpolarreal.material-strategy-comparison.v1"
VARIANT_ROLES = (
    "data_only",
    "post_fit_cleanup",
    "train_time_regularizer",
)
VARIANT_LABELS = {
    "data_only": "Data-only fit",
    "post_fit_cleanup": "Fixed post-fit cleanup",
    "train_time_regularizer": "Train-time consensus regularizer",
}
TRAIN_REGULARIZER_KIND = "frequency-consensus-regularizer"
POST_FIT_KIND = "frequency-consensus"
OVERVIEW_WIDTH = 2100
OVERVIEW_BODY_FONT = 30
OVERVIEW_HERO_MAPS = (
    "roughness",
    "specular",
    "subsurface",
    "anisotropic",
)
OVERVIEW_DETAIL_MAPS = ("subsurface", "anisotropic")
OVERVIEW_DETAIL_CROP_SIZE = 96
OVERVIEW_DETAIL_SCALE = 2
REPORT_LIGHTING = pair.EVALUATION_LIGHTING
REPORT_MAPS = pair.SCALAR_MAPS


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compose a controlled three-way material-strategy report: data-only, "
            "fixed post-fit cleanup, and train-time regularization."
        )
    )
    parser.add_argument("--data-only", required=True)
    parser.add_argument("--post-fit-cleanup", required=True)
    parser.add_argument("--train-time-regularizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    compose_material_strategy_comparison(
        args.data_only,
        args.post_fit_cleanup,
        args.train_time_regularizer,
        args.output,
        mask_path=args.mask,
        data_root=args.data_root,
    )


def compose_material_strategy_comparison(
    data_only_camera: str | Path,
    post_fit_cleanup_camera: str | Path,
    train_time_regularizer_camera: str | Path,
    output_dir: str | Path,
    *,
    mask_path: str | Path | None = None,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    cameras = {
        "data_only": Path(data_only_camera).expanduser().resolve(),
        "post_fit_cleanup": Path(post_fit_cleanup_camera).expanduser().resolve(),
        "train_time_regularizer": Path(train_time_regularizer_camera)
        .expanduser()
        .resolve(),
    }
    output_dir = Path(output_dir).expanduser().resolve()
    _validate_paths(cameras, output_dir)

    manifests = {
        role: pair._load_complete_manifest(camera)
        for role, camera in cameras.items()
    }
    selected_cleanup = pair._comparison_profiles(
        manifests["data_only"],
        manifests["post_fit_cleanup"],
        requested=None,
    )
    selected_train = pair._comparison_profiles(
        manifests["data_only"],
        manifests["train_time_regularizer"],
        requested=None,
    )
    if selected_cleanup != selected_train:
        raise ValueError("the three material strategies do not share one profile list")
    profiles = selected_cleanup
    if tuple(profiles) != tuple(pair.QUALIFICATION_PROFILES):
        raise ValueError(
            "material strategy report requires exactly the olat, hdri, and mix profiles"
        )
    acquisitions = {
        role: pair._load_acquisitions(camera, profiles)
        for role, camera in cameras.items()
    }
    cleanup_pair = {
        "baseline": acquisitions["data_only"],
        "regularized": acquisitions["post_fit_cleanup"],
    }
    train_pair = {
        "baseline": acquisitions["data_only"],
        "regularized": acquisitions["train_time_regularizer"],
    }
    cleanup_contract = pair._validate_comparison_contract(cleanup_pair, profiles)
    train_contract = _validate_train_comparison_contract(train_pair, profiles)
    stage_contract = _validate_strategy_contracts(
        acquisitions,
        profiles,
        cleanup_contract=cleanup_contract,
        train_contract=train_contract,
    )
    train_audit = _validate_train_artifacts(cameras, acquisitions, profiles)

    sample_map = (
        cameras["data_only"]
        / "material"
        / profiles[0]
        / "maps"
        / "roughness.png"
    )
    with Image.open(sample_map) as image:
        sample_size = image.size
    mask, mask_source = _comparison_mask(
        cameras["data_only"],
        sample_size,
        mask_path=mask_path,
        data_root=data_root,
    )
    pair._validate_fit_mask_hash(mask, cleanup_pair, profiles)
    pair._validate_fit_mask_hash(mask, train_pair, profiles)
    interior_mask = pair._erode_mask(mask, radius=2)
    if not np.any(interior_mask):
        raise ValueError("comparison mask has no five-pixel interior")
    _validate_shared_guides(cameras, profiles)

    cleanup_guide = pair._guide_diagnostic_configuration(cleanup_contract)
    train_guide = pair._guide_diagnostic_configuration(train_contract)
    _validate_shared_guide_diagnostic(cleanup_guide, train_guide)
    guide_diagnostic = {
        **cleanup_guide,
        "source": "fixed_shared_report_diagnostic",
        "matches_active_regularizer_configuration": False,
    }
    cleanup_maps, cleanup_regions = pair._measure_material_maps(
        cameras["data_only"],
        cameras["post_fit_cleanup"],
        profiles,
        interior_mask,
        guide_diagnostic,
    )
    train_maps, train_regions = pair._measure_material_maps(
        cameras["data_only"],
        cameras["train_time_regularizer"],
        profiles,
        interior_mask,
        guide_diagnostic,
    )
    if cleanup_regions != train_regions:
        raise ValueError("three-way material diagnostics use different guide regions")
    material_metrics = _merge_pairwise_material_metrics(
        cleanup_maps,
        train_maps,
        profiles,
    )

    cleanup_audit = pair._measure_frequency_cleanup(
        cameras["data_only"],
        cameras["post_fit_cleanup"],
        profiles,
        mask,
        cleanup_pair,
        cleanup_contract,
    )
    if not isinstance(cleanup_audit, dict) or cleanup_audit.get("available") is not True:
        raise ValueError("post-fit cleanup is missing validated frozen-target evidence")

    cleanup_evaluation = pair._collect_evaluation_metrics(cleanup_pair, profiles)
    train_evaluation = pair._collect_evaluation_metrics(train_pair, profiles)
    evaluation_metrics = _merge_pairwise_evaluation_metrics(
        cleanup_evaluation,
        train_evaluation,
        profiles,
    )
    cleanup_cases = pair._collect_case_png_metrics(
        cameras["data_only"],
        cameras["post_fit_cleanup"],
        cleanup_pair,
        profiles,
        mask,
    )
    train_cases = pair._collect_case_png_metrics(
        cameras["data_only"],
        cameras["train_time_regularizer"],
        train_pair,
        profiles,
        mask,
    )
    case_metrics = _merge_pairwise_case_metrics(
        cleanup_cases,
        train_cases,
        profiles,
    )
    qualifications = {
        "post_fit_cleanup": pair._build_frequency_diagnostic_gates(
            cleanup_maps,
            cleanup_evaluation,
            cleanup_cases,
            profiles,
        ),
        "train_time_regularizer": pair._build_frequency_diagnostic_gates(
            train_maps,
            train_evaluation,
            train_cases,
            profiles,
        ),
    }
    aggregate = _aggregate_strategy_metrics(
        material_metrics,
        case_metrics,
        qualifications,
        profiles,
    )
    detail_crops = _select_overview_detail_crops(
        cameras["data_only"],
        profiles,
        interior_mask,
        guide_diagnostic,
    )
    summary: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "title": (
            f"{pair._camera_display_name(cameras['data_only'].name)} · "
            "Material strategy comparison"
        ),
        "variant_order": list(VARIANT_ROLES),
        "variants": {
            role: {
                "label": VARIANT_LABELS[role],
                "camera": str(cameras[role]),
                "stage": stage_contract[role],
            }
            for role in VARIANT_ROLES
        },
        "profiles": list(profiles),
        "maps": list(REPORT_MAPS),
        "evaluation_lighting": list(REPORT_LIGHTING),
        "mask_source": mask_source,
        "mask_pixels": int(np.count_nonzero(mask)),
        "guide_diagnostic": guide_diagnostic,
        "guide_regions": cleanup_regions,
        "comparison_contracts": {
            "data_only_to_post_fit_cleanup": cleanup_contract,
            "data_only_to_train_time_regularizer": train_contract,
        },
        "material_metrics": material_metrics,
        "evaluation_metrics": evaluation_metrics,
        "case_png_evaluation": case_metrics,
        "post_fit_cleanup_audit": cleanup_audit,
        "train_time_regularizer_audit": train_audit,
        "qualifications": qualifications,
        "aggregate": aggregate,
        "overview_detail_crops": detail_crops,
        "interpretation": (
            "The fixed cleanup is one deterministic update after the data-only fit; "
            "the train-time arm is an independent optimization trajectory with a "
            "frozen consensus target in its objective. Lower residual-outlier counts "
            "alone do not establish better recovered material."
        ),
        "artifacts": {
            "overview": {"path": "overview.png"},
            "metrics": {"path": "metrics.csv"},
            "material": {
                profile: {"path": f"material/{profile}.png"}
                for profile in profiles
            },
            "evaluation": {
                lighting: {"path": f"evaluation/{lighting}.png"}
                for lighting in REPORT_LIGHTING
            },
        },
    }

    stage = output_dir.with_name(f".{output_dir.name}.tmp")
    backup = output_dir.with_name(f".{output_dir.name}.previous")
    _recover_interrupted_report(output_dir, stage, backup)
    stage.mkdir(parents=True)
    try:
        for profile in profiles:
            _write_material_sheet(
                cameras,
                profile,
                material_metrics[profile],
                stage / "material" / f"{profile}.png",
            )
        for lighting in REPORT_LIGHTING:
            _write_evaluation_sheet(
                cameras,
                profiles,
                lighting,
                case_metrics,
                stage / "evaluation" / f"{lighting}.png",
            )
        csv_rows = _write_metrics_csv(
            stage / "metrics.csv",
            material_metrics,
            evaluation_metrics,
            case_metrics,
            qualifications,
            profiles,
        )
        summary["metrics_csv_rows"] = csv_rows
        _write_overview(
            cameras,
            profiles,
            material_metrics,
            evaluation_metrics,
            case_metrics,
            summary,
            interior_mask,
            stage / "overview.png",
        )
        _record_artifact_metadata(stage, summary["artifacts"])
        pair._write_json(stage / "summary.json", summary)
        _validate_report(stage, summary)

        _commit_staged_report(stage, output_dir, backup)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return summary


def _recover_interrupted_report(
    output_dir: Path,
    stage: Path,
    backup: Path,
) -> None:
    """Restore the last valid report before removing stale transaction state."""
    if backup.exists():
        if output_dir.exists():
            shutil.rmtree(backup)
        else:
            backup.replace(output_dir)
    if stage.exists():
        shutil.rmtree(stage)


def _commit_staged_report(stage: Path, output_dir: Path, backup: Path) -> None:
    if backup.exists():
        raise RuntimeError(f"stale report backup was not recovered: {backup}")
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


def _validate_paths(cameras: dict[str, Path], output_dir: Path) -> None:
    if len(set(cameras.values())) != len(VARIANT_ROLES):
        raise ValueError("the three strategy inputs must be distinct camera roots")
    for camera in cameras.values():
        if (
            output_dir == camera
            or output_dir in camera.parents
            or camera in output_dir.parents
        ):
            raise ValueError(
                "comparison output must not overlap an acquisition root: "
                f"output={output_dir}, acquisition={camera}"
            )


def _comparison_mask(
    data_only_camera: Path,
    sample_size: tuple[int, int],
    *,
    mask_path: str | Path | None,
    data_root: str | Path | None,
) -> tuple[np.ndarray, str]:
    if mask_path is not None and data_root is not None:
        raise ValueError("pass either --data-root or --mask, not both")
    if data_root is not None:
        root = Path(data_root).expanduser().resolve()
        mask = pair._fit_mask_from_data_root(
            root,
            data_only_camera.parent.name,
            data_only_camera.name,
            sample_size,
        )
        return mask, f"reconstructed acquisition fit mask from {root}"
    if mask_path is not None:
        path = Path(mask_path).expanduser().resolve()
        return pair._read_mask(path, sample_size), str(path)
    path = data_only_camera / "material" / "olat" / "maps" / "baseColor.png"
    if not path.is_file():
        profile = next((data_only_camera / "material").iterdir()).name
        path = data_only_camera / "material" / profile / "maps" / "baseColor.png"
    return pair._infer_foreground_mask(path), "fallback inferred from data-only baseColor"


def _validate_strategy_contracts(
    acquisitions: dict[str, dict[str, dict[str, Any]]],
    profiles: Sequence[str],
    *,
    cleanup_contract: dict[str, Any],
    train_contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if cleanup_contract.get("baseline_regularizer_inactive") is not True:
        raise ValueError("post-fit cleanup must branch from a data-only fit")
    if cleanup_contract.get("regularization_kind") != POST_FIT_KIND:
        raise ValueError("post-fit cleanup arm must use frequency-consensus")
    if train_contract.get("baseline_regularizer_inactive") is not True:
        raise ValueError("train-time regularizer must branch from a data-only fit")
    if train_contract.get("regularization_kind") != TRAIN_REGULARIZER_KIND:
        raise ValueError(
            "train-time arm must use frequency-consensus-regularizer"
        )

    data_stage: dict[str, Any] | None = None
    cleanup_stage: dict[str, Any] | None = None
    train_stage: dict[str, Any] | None = None
    for profile in profiles:
        data = acquisitions["data_only"][profile]
        cleanup = acquisitions["post_fit_cleanup"][profile]
        train = acquisitions["train_time_regularizer"][profile]
        data_regularization = data.get("regularization")
        cleanup_regularization = cleanup.get("regularization")
        train_regularization = train.get("regularization")
        if not all(
            isinstance(value, dict)
            for value in (
                data_regularization,
                cleanup_regularization,
                train_regularization,
            )
        ):
            raise ValueError("strategy acquisition is missing regularization provenance")
        if float(data_regularization["weight"]) != 0.0:
            raise ValueError("data-only arm must have zero regularization weight")
        data_plan = data_regularization.get("stage_plan")
        if (
            data.get("schema") != "ictpolarreal.end2end-disney.v13"
            or data_regularization.get("kind") != POST_FIT_KIND
            or data_regularization.get("data_objective_only") is not True
            or data_regularization.get("cleanup_applied") is not False
            or data_regularization.get("cleanup_diagnostic") is not None
            or not isinstance(data_plan, dict)
            or data_plan.get("enabled") is not False
            or data_plan.get("post_fit_updates") != 0
            or data_plan.get("cleanup_optimizer_steps") != 0
        ):
            raise ValueError(
                "data-only arm must record an inactive v13 frequency-consensus stage"
            )
        if cleanup_regularization.get("data_objective_only") is not True:
            raise ValueError("fixed cleanup must record a data-only optimization")
        if cleanup_regularization.get("cleanup_applied") is not True:
            raise ValueError("fixed cleanup arm did not record its post-fit update")
        cleanup_plan = cleanup_regularization.get("stage_plan")
        if not isinstance(cleanup_plan, dict):
            raise ValueError("fixed cleanup is missing its stage plan")
        if cleanup_plan.get("post_fit_updates") != 1:
            raise ValueError("fixed cleanup must perform exactly one post-fit update")
        if cleanup_plan.get("cleanup_optimizer_steps") != 0:
            raise ValueError("fixed cleanup must not run cleanup optimizer steps")

        if train_regularization.get("data_objective_only") is not False:
            raise ValueError("train-time regularizer must contribute to the objective")
        if train_regularization.get("cleanup_applied") is not False:
            raise ValueError("train-time regularizer must not apply post-fit cleanup")
        if train_regularization.get("weight_semantics") != "objective_coefficient":
            raise ValueError("train-time regularizer weight must be an objective coefficient")
        if "cleanup_diagnostic" in train_regularization:
            raise ValueError("train-time regularizer must not record a cleanup diagnostic")
        train_plan = train_regularization.get("stage_plan")
        if not isinstance(train_plan, dict):
            raise ValueError("train-time regularizer is missing its stage plan")
        total_steps = int(train.get("checkpoint_signature", {}).get("steps", 0))
        expected_settings = acquisition_core._regularization_settings(
            TRAIN_REGULARIZER_KIND
        )
        expected_plan = acquisition_core._frequency_consensus_regularizer_stage_plan(
            total_steps,
            enabled=True,
        )
        signature_regularization = train.get("checkpoint_signature", {}).get(
            "regularization"
        )
        if (
            train.get("schema") != "ictpolarreal.end2end-disney.v14"
            or train_regularization.get("kind") != TRAIN_REGULARIZER_KIND
            or train_regularization.get("settings") != expected_settings
            or train_plan != expected_plan
            or not isinstance(signature_regularization, dict)
            or any(
                train_regularization.get(key) != signature_regularization.get(key)
                for key in ("kind", "weight", "parameters", "settings", "stage_plan")
            )
        ):
            raise ValueError(
                "train-time arm is not the canonical v14 frequency-consensus regularizer"
            )
        warmup = _positive_int(train_plan.get("data_warmup_steps"), "data_warmup_steps")
        target_after = _positive_int(
            train_plan.get("target_after_data_step"),
            "target_after_data_step",
        )
        active = _positive_int(train_plan.get("regularized_steps"), "regularized_steps")
        if warmup != target_after or warmup + active != total_steps:
            raise ValueError("train-time regularizer stage arithmetic is inconsistent")
        if train_plan.get("post_fit_updates") != 0:
            raise ValueError("train-time regularizer must have zero post-fit updates")
        if train_plan.get("cleanup_optimizer_steps") != 0:
            raise ValueError("train-time regularizer must have zero cleanup optimizer steps")
        if train_plan.get("cleanup_optimizer_steps") != 0:
            raise ValueError("train-time regularizer must have zero cleanup optimizer steps")
        if train_plan.get("optimizer_reset") is not False:
            raise ValueError("train-time regularizer must preserve optimizer state")
        if train_regularization.get("post_fit_updates") != 0:
            raise ValueError("train-time result must record zero post-fit updates")
        if train_regularization.get("regularized_steps_completed") != active:
            raise ValueError(
                "train-time completed regularized steps differ from the stage plan"
            )
        frozen = train_regularization.get("frozen_bundle")
        if not isinstance(frozen, dict) or frozen.get("role") != "train_time_frozen_target":
            raise ValueError("train-time regularizer is missing its frozen-target role")
        if (
            frozen.get("schema") != "ictpolarreal.frequency-consensus-bundle.v1"
            or frozen.get("created_after_step") != target_after
            or frozen.get("strength") != 1.0
        ):
            raise ValueError("train-time regularizer frozen-target provenance is invalid")
        final_regularization = _finite_nonnegative(
            train_regularization.get("final_regularization_loss"),
            "final_regularization_loss",
        )
        weighted = _finite_nonnegative(
            train_regularization.get("weighted_final_regularization_loss"),
            "weighted_final_regularization_loss",
        )
        weight = float(train_regularization["weight"])
        if not math.isclose(weighted, weight * final_regularization, rel_tol=1e-6, abs_tol=1e-12):
            raise ValueError("train-time weighted regularization loss is inconsistent")
        if train_regularization.get("regularized_steps_completed") != active:
            raise ValueError("train-time regularized-step count is incomplete")

        profile_data_stage = {
            "optimization": "data_loss_only",
            "data_steps": total_steps,
            "regularization_weight": 0.0,
            "post_fit_updates": 0,
        }
        profile_cleanup_stage = {
            "optimization": "data_loss_only_then_fixed_cleanup",
            "data_steps": int(cleanup_plan["data_fit_steps"]),
            "regularization_weight": float(cleanup_regularization["weight"]),
            "post_fit_updates": 1,
            "cleanup_optimizer_steps": 0,
        }
        profile_train_stage = {
            "optimization": "data_warmup_then_joint_regularization",
            "regularization_weight": weight,
            "data_warmup_steps": warmup,
            "regularized_steps": active,
            "regularized_steps_completed": active,
            "post_fit_updates": 0,
            "optimizer_reset": False,
            "frozen_target_role": "train_time_frozen_target",
        }
        data_stage = _same_across_profiles(data_stage, profile_data_stage, "data-only stage")
        cleanup_stage = _same_across_profiles(
            cleanup_stage,
            profile_cleanup_stage,
            "post-fit stage",
        )
        train_stage = _same_across_profiles(train_stage, profile_train_stage, "train stage")
    assert data_stage is not None and cleanup_stage is not None and train_stage is not None
    return {
        "data_only": data_stage,
        "post_fit_cleanup": cleanup_stage,
        "train_time_regularizer": train_stage,
    }


def _validate_train_comparison_contract(
    acquisitions: dict[str, dict[str, dict[str, Any]]],
    profiles: Sequence[str],
) -> dict[str, Any]:
    """Validate the one explicit v13 data-only to v14 train-time transition."""
    configurations: dict[str, tuple[str, tuple[str, ...], float, str, str]] = {}
    for variant in ("baseline", "regularized"):
        per_profile = {
            profile: pair._regularization_configuration(
                acquisitions[variant][profile]
            )
            for profile in profiles
        }
        if len(set(per_profile.values())) != 1:
            raise ValueError(
                f"{variant} train-comparison configuration differs across profiles"
            )
        configurations[variant] = next(iter(per_profile.values()))
    baseline_kind, baseline_parameters, baseline_weight, _, _ = configurations[
        "baseline"
    ]
    train_kind, train_parameters, train_weight, train_settings, train_stage = (
        configurations["regularized"]
    )
    if baseline_kind != POST_FIT_KIND or baseline_weight != 0.0:
        raise ValueError(
            "train comparison requires the v13 zero-weight frequency-consensus data fit"
        )
    if train_kind != TRAIN_REGULARIZER_KIND or train_weight <= 0.0:
        raise ValueError(
            "train comparison requires an active frequency-consensus-regularizer"
        )
    if baseline_parameters != train_parameters:
        raise ValueError("data-only and train-time arms target different scalar maps")

    expected_baseline_identity = (
        "ictpolarreal.end2end-checkpoint.v13",
        "ictpolarreal.profile-acquisition-adapter.v7",
        "ictpolarreal-frequency-consensus-v1",
    )
    expected_train_identity = (
        "ictpolarreal.end2end-checkpoint.v14",
        "ictpolarreal.profile-acquisition-adapter.v8",
        "ictpolarreal-frequency-consensus-regularizer-v1",
    )

    def identity(signature: dict[str, Any]) -> tuple[Any, Any, Any]:
        adapter = signature.get("adapter")
        return (
            signature.get("schema"),
            adapter.get("schema") if isinstance(adapter, dict) else None,
            adapter.get("algorithm_version") if isinstance(adapter, dict) else None,
        )

    def normalized(signature: dict[str, Any]) -> dict[str, Any]:
        result = json.loads(json.dumps(signature))
        result["schema"] = "<v13-to-v14-frequency-regularizer-transition>"
        regularization = result.get("regularization")
        if not isinstance(regularization, dict):
            raise ValueError("train comparison signature lacks regularization")
        result["regularization"] = {
            "parameters": regularization.get("parameters"),
            "configuration": "<comparison-variable>",
        }
        adapter = result.get("adapter")
        if not isinstance(adapter, dict):
            raise ValueError("train comparison signature lacks adapter provenance")
        adapter["schema"] = "<compatible-frequency-regularizer-adapter>"
        adapter["algorithm_version"] = "<compatible-frequency-regularizer-algorithm>"
        adapter.pop("end2end_acquisition_sha256", None)
        return result

    for profile in profiles:
        baseline = acquisitions["baseline"][profile]
        regularized = acquisitions["regularized"][profile]
        if (
            baseline.get("schema") != "ictpolarreal.end2end-disney.v13"
            or regularized.get("schema") != "ictpolarreal.end2end-disney.v14"
        ):
            raise ValueError(
                "train comparison requires v13 data-only and v14 train acquisitions"
            )
        baseline_signature = baseline.get("checkpoint_signature")
        train_signature = regularized.get("checkpoint_signature")
        if not isinstance(baseline_signature, dict) or not isinstance(
            train_signature, dict
        ):
            raise ValueError("train comparison is missing checkpoint signatures")
        if identity(baseline_signature) != expected_baseline_identity:
            raise ValueError("data-only arm is not the supported v13/v7 identity")
        if identity(train_signature) != expected_train_identity:
            raise ValueError("train-time arm is not the supported v14/v8 identity")
        if normalized(baseline_signature) != normalized(train_signature):
            raise ValueError(
                f"{profile} train comparison differs beyond regularizer provenance"
            )
        if {
            baseline.get("base_color_source"),
            regularized.get("base_color_source"),
        } != {"dataset_albedo"}:
            raise ValueError("train comparison must use dataset_albedo in both arms")
    return {
        "controlled": True,
        "comparison_mode": "v13-data-only-to-v14-train-time-regularizer",
        "signature_compatibility": ["v13-frequency-v1-to-v14-frequency-regularizer-v1"],
        "only_intended_difference": (
            "frequency-consensus target participates in the final training stage "
            "instead of being inactive"
        ),
        "baseline_regularization_kind": baseline_kind,
        "regularization_kind": train_kind,
        "regularized_parameters": list(train_parameters),
        "baseline_tv_weight": baseline_weight,
        "regularized_tv_weight": train_weight,
        "baseline_regularizer_inactive": True,
        "active_frequency_upgrade": False,
        "regularizer_configuration_variable": True,
        "regularized_settings": json.loads(train_settings),
        "regularized_stage_plan": json.loads(train_stage),
        "base_color_source": "dataset_albedo",
    }


def _same_across_profiles(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    if previous is not None and previous != current:
        raise ValueError(f"{label} differs across profiles")
    return current


def _validate_train_artifacts(
    cameras: dict[str, Path],
    acquisitions: dict[str, dict[str, dict[str, Any]]],
    profiles: Sequence[str],
) -> dict[str, Any]:
    """Require the core acquisition validator to accept every frozen target."""
    camera = cameras["train_time_regularizer"]
    records: dict[str, Any] = {}
    for profile in profiles:
        material_dir = camera / "material" / profile
        acquisition = acquisitions["train_time_regularizer"][profile]
        if not acquisition_core._frequency_frozen_artifact_complete(
            material_dir,
            acquisition,
        ):
            raise ValueError(
                f"train-time regularizer frozen artifact is incomplete: {profile}"
            )
        frozen = acquisition["regularization"]["frozen_bundle"]
        artifact = frozen["artifact"]
        records[profile] = {
            "validated": True,
            "role": frozen["role"],
            "created_after_step": frozen["created_after_step"],
            "strength": frozen["strength"],
            "artifact": {
                "path": artifact["path"],
                "sha256": artifact["sha256"],
                "bytes": artifact["bytes"],
            },
        }
    return {
        "schema": "ictpolarreal.train-time-frequency-consensus-audit.v1",
        "validated": True,
        "profiles": records,
    }


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _validate_shared_guides(cameras: dict[str, Path], profiles: Sequence[str]) -> None:
    for profile in profiles:
        for filename in ("baseColor.png", "normal.png"):
            hashes = {
                _file_sha256(
                    camera / "material" / profile / "maps" / filename
                )
                for camera in cameras.values()
            }
            if len(hashes) != 1:
                raise ValueError(
                    f"the three strategies have different {profile}/{filename} guides"
                )


def _validate_shared_guide_diagnostic(
    cleanup: dict[str, Any], train: dict[str, Any]
) -> None:
    for name in ("guide", "score", "albedo_sigma", "normal_sigma", "input_precision"):
        if cleanup.get(name) != train.get(name):
            raise ValueError(f"candidate guide diagnostics differ for {name}")


def _merge_pairwise_material_metrics(
    cleanup: dict[str, Any],
    train: dict[str, Any],
    profiles: Sequence[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for profile in profiles:
        output[profile] = {}
        for map_name in REPORT_MAPS:
            left = cleanup[profile][map_name]
            right = train[profile][map_name]
            if left["baseline"] != right["baseline"]:
                raise ValueError(
                    f"pairwise material baselines disagree: {profile}/{map_name}"
                )
            output[profile][map_name] = {
                "data_only": left["baseline"],
                "post_fit_cleanup": left["regularized"],
                "train_time_regularizer": right["regularized"],
                "comparison_to_data_only": {
                    "post_fit_cleanup": left["comparison"],
                    "train_time_regularizer": right["comparison"],
                },
            }
    return output


def _merge_pairwise_evaluation_metrics(
    cleanup: dict[str, Any],
    train: dict[str, Any],
    profiles: Sequence[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for profile in profiles:
        output[profile] = {}
        for lighting in REPORT_LIGHTING:
            left = cleanup[profile][lighting]
            right = train[profile][lighting]
            if left["baseline"] != right["baseline"]:
                raise ValueError(
                    f"pairwise evaluation baselines disagree: {profile}/{lighting}"
                )
            output[profile][lighting] = {
                "data_only": left["baseline"],
                "post_fit_cleanup": left["regularized"],
                "train_time_regularizer": right["regularized"],
            }
    return output


def _merge_pairwise_case_metrics(
    cleanup: dict[str, Any],
    train: dict[str, Any],
    profiles: Sequence[str],
) -> dict[str, Any]:
    for key in ("mask_pixels", "profiles", "case_ids"):
        if cleanup.get(key) != train.get(key):
            raise ValueError(f"pairwise case-PNG reports disagree for {key}")
    if cleanup.get("profiles") != list(profiles):
        raise ValueError("case-PNG report profile order is inconsistent")
    cleanup_rows = {
        (row["lighting"], row["case_id"], row["profile"]): row
        for row in cleanup["rows"]
    }
    train_rows = {
        (row["lighting"], row["case_id"], row["profile"]): row
        for row in train["rows"]
    }
    if cleanup_rows.keys() != train_rows.keys():
        raise ValueError("pairwise case-PNG row coverage differs")
    rows = []
    for key in sorted(cleanup_rows):
        left = cleanup_rows[key]
        right = train_rows[key]
        if left["baseline"] != right["baseline"]:
            raise ValueError(f"pairwise case-PNG baselines disagree: {key}")
        left_files = left["files"]
        right_files = right["files"]
        for filename in (
            "reference",
            "reference_sha256",
            "baseline_prediction",
            "baseline_prediction_sha256",
            "baseline_error",
            "baseline_error_sha256",
            "baseline_error_pixel_sha256",
        ):
            if left_files.get(filename) != right_files.get(filename):
                raise ValueError(f"pairwise baseline case files disagree: {key}/{filename}")
        rows.append(
            {
                "lighting": left["lighting"],
                "case_id": left["case_id"],
                "profile": left["profile"],
                "files": {
                    "reference": left_files["reference"],
                    "reference_sha256": left_files["reference_sha256"],
                    "data_only_prediction": left_files["baseline_prediction"],
                    "data_only_prediction_sha256": left_files[
                        "baseline_prediction_sha256"
                    ],
                    "post_fit_cleanup_prediction": left_files[
                        "regularized_prediction"
                    ],
                    "post_fit_cleanup_prediction_sha256": left_files[
                        "regularized_prediction_sha256"
                    ],
                    "train_time_regularizer_prediction": right_files[
                        "regularized_prediction"
                    ],
                    "train_time_regularizer_prediction_sha256": right_files[
                        "regularized_prediction_sha256"
                    ],
                },
                "metrics": {
                    "data_only": left["baseline"],
                    "post_fit_cleanup": left["regularized"],
                    "train_time_regularizer": right["regularized"],
                },
                "delta_vs_data_only": {
                    "post_fit_cleanup": left["delta"],
                    "train_time_regularizer": right["delta"],
                },
            }
        )
    selected = {
        lighting: _select_shared_cases(rows, lighting, profiles)
        for lighting in REPORT_LIGHTING
    }
    return {
        "schema": "ictpolarreal.three-way-case-png-evaluation.v1",
        "mask_pixels": cleanup["mask_pixels"],
        "profiles": list(profiles),
        "case_ids": cleanup["case_ids"],
        "selected_cases": selected,
        "validated_reference_files": cleanup["validated_reference_files"],
        "validated_prediction_files": 3 * len(rows),
        "validated_error_files": 3 * len(rows),
        "selection_policy": (
            "representative uses baseline-only median mean PSNR; audit uses the "
            "worst mean PSNR delta over both candidate strategies"
        ),
        "rows": rows,
    }


def _select_shared_cases(
    rows: Sequence[dict[str, Any]],
    lighting: str,
    profiles: Sequence[str],
) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["lighting"] == lighting:
            by_case.setdefault(str(row["case_id"]), []).append(row)
    if not by_case:
        raise ValueError(f"no three-way case rows for {lighting}")
    ranked = []
    for case_id, case_rows in by_case.items():
        if sorted(row["profile"] for row in case_rows) != sorted(profiles):
            raise ValueError(f"{lighting}/{case_id} does not cover every profile")
        baseline_psnr = float(
            np.mean([row["metrics"]["data_only"]["psnr"] for row in case_rows])
        )
        deltas = {
            role: float(
                np.mean(
                    [row["delta_vs_data_only"][role]["psnr"] for row in case_rows]
                )
            )
            for role in ("post_fit_cleanup", "train_time_regularizer")
        }
        ranked.append(
            {
                "case_id": case_id,
                "mean_profile_data_only_psnr_db": baseline_psnr,
                "mean_profile_delta_psnr_db": deltas,
            }
        )
    baseline_ranked = sorted(
        ranked,
        key=lambda entry: (
            entry["mean_profile_data_only_psnr_db"],
            entry["case_id"],
        ),
    )
    representative_index = (len(baseline_ranked) - 1) // 2
    representative = {
        **baseline_ranked[representative_index],
        "role": "baseline_median_psnr",
        "rank_low_to_high": representative_index + 1,
    }
    audit_candidates = []
    for entry in ranked:
        for role in ("post_fit_cleanup", "train_time_regularizer"):
            audit_candidates.append(
                (
                    entry["mean_profile_delta_psnr_db"][role],
                    VARIANT_ROLES.index(role),
                    entry["case_id"],
                    role,
                    entry,
                )
            )
    _, _, _, trigger, worst = min(audit_candidates)
    audit = {
        **worst,
        "role": "joint_worst_candidate_delta_psnr",
        "trigger_variant": trigger,
    }
    return {
        "case_count": len(ranked),
        "representative": representative,
        "audit": audit,
    }


def _aggregate_strategy_metrics(
    material: dict[str, Any],
    cases: dict[str, Any],
    qualifications: dict[str, Any],
    profiles: Sequence[str],
) -> dict[str, Any]:
    quiet_pixels = {
        role: sum(
            int(material[p][m][role]["quiet_region_pixels"])
            for p in profiles
            for m in REPORT_MAPS
        )
        for role in VARIANT_ROLES
    }
    quiet_outliers = {
        role: sum(
            int(
                material[p][m][role][
                    "quiet_region_median5_residual_outlier_count_gt_0p05"
                ]
            )
            for p in profiles
            for m in REPORT_MAPS
        )
        for role in VARIANT_ROLES
    }
    quiet_fraction = {
        role: float(quiet_outliers[role] / quiet_pixels[role])
        for role in VARIANT_ROLES
    }
    baseline_band = sum(
        float(material[p][m]["data_only"]["guide_textured_band3_8_energy"])
        for p in profiles
        for m in REPORT_MAPS
    )
    texture_amplitude = {"data_only": 1.0}
    for role in ("post_fit_cleanup", "train_time_regularizer"):
        energy = sum(
            float(material[p][m][role]["guide_textured_band3_8_energy"])
            for p in profiles
            for m in REPORT_MAPS
        )
        texture_amplitude[role] = (
            float(math.sqrt(energy / baseline_band)) if baseline_band > 1e-18 else None
        )
    relighting = {}
    for lighting in REPORT_LIGHTING:
        relighting[lighting] = {}
        lighting_rows = [
            row
            for row in cases["rows"]
            if row["lighting"] == lighting and row["profile"] in profiles
        ]
        if not lighting_rows:
            raise ValueError(f"no validated case-PNG metrics for {lighting}")
        for role in VARIANT_ROLES:
            psnr = float(
                np.mean([row["metrics"][role]["psnr"] for row in lighting_rows])
            )
            ssim = float(
                np.mean(
                    [
                        row["metrics"][role]["ssim_global"]
                        for row in lighting_rows
                    ]
                )
            )
            relighting[lighting][role] = {
                "psnr": psnr,
                "ssim_global": ssim,
                "delta_psnr_vs_data_only": None,
                "delta_ssim_vs_data_only": None,
            }
        base = relighting[lighting]["data_only"]
        for role in ("post_fit_cleanup", "train_time_regularizer"):
            relighting[lighting][role]["delta_psnr_vs_data_only"] = (
                relighting[lighting][role]["psnr"] - base["psnr"]
            )
            relighting[lighting][role]["delta_ssim_vs_data_only"] = (
                relighting[lighting][role]["ssim_global"] - base["ssim_global"]
            )
    return {
        "quiet_outlier_fraction": quiet_fraction,
        "quiet_outlier_count": quiet_outliers,
        "quiet_pixel_count": quiet_pixels,
        "texture_band3_8_amplitude_ratio_vs_data_only": texture_amplitude,
        "relighting_source": "validated_case_png_all_cases_and_profiles",
        "relighting": relighting,
        "qualification": {
            "data_only": {"status": "REFERENCE", "thresholds_met": None, "gate_count": 28},
            **{
                role: {
                    "status": qualifications[role]["qualification_status"],
                    "thresholds_met": qualifications[role]["thresholds_met"],
                    "gate_count": qualifications[role]["gate_count"],
                }
                for role in ("post_fit_cleanup", "train_time_regularizer")
            },
        },
    }


def _select_overview_detail_crops(
    data_only_camera: Path,
    profiles: Sequence[str],
    interior_mask: np.ndarray,
    guide_diagnostic: dict[str, Any],
) -> dict[str, Any]:
    profile = "mix" if "mix" in profiles else profiles[-1]
    maps_dir = data_only_camera / "material" / profile / "maps"
    albedo = pair._read_rgb_map(maps_dir / "baseColor.png")
    normal = pair._read_rgb_map(maps_dir / "normal.png")
    guide_masks, _ = pair._guide_region_masks(
        albedo,
        normal,
        interior_mask,
        albedo_sigma=float(guide_diagnostic["albedo_sigma"]),
        normal_sigma=float(guide_diagnostic["normal_sigma"]),
    )
    output = {"profile": profile, "selection": "densest data-only quiet residual outliers"}
    output["maps"] = {}
    for map_name in OVERVIEW_DETAIL_MAPS:
        values = pair._read_scalar_map(maps_dir / f"{map_name}.png")
        flagged = (
            guide_masks["quiet_pixels"]
            & (pair._own_median5_residual(values) > 0.05)
        )
        box = pair._densest_flagged_crop_box(flagged, OVERVIEW_DETAIL_CROP_SIZE)
        left, top, right, bottom = box
        output["maps"][map_name] = {
            "crop_box_xyxy": list(box),
            "data_only_flagged_pixels_in_crop": int(
                np.count_nonzero(flagged[top:bottom, left:right])
            ),
        }
    return output


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_material_sheet(
    cameras: dict[str, Path],
    profile: str,
    metrics: dict[str, Any],
    output: Path,
) -> None:
    with Image.open(
        cameras["data_only"]
        / "material"
        / profile
        / "maps"
        / f"{REPORT_MAPS[0]}.png"
    ) as first:
        tile_width, tile_height = first.size
    left = pair.MATERIAL_LABEL_GUTTER
    top = 118
    group_header = 64
    caption_height = 58
    row_height = tile_height + caption_height
    groups = (REPORT_MAPS[:4], REPORT_MAPS[4:])
    group_height = group_header + len(VARIANT_ROLES) * row_height
    width = left + 4 * tile_width
    height = top + len(groups) * group_height
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (24, 20),
        f"{profile.upper()} fit · three material strategies",
        font=pair._font(48, bold=True),
        fill="white",
    )
    draw.text(
        (26, 78),
        "Outlier = quiet-region |map − own 5×5 median| > 0.05; band is 3–8 px amplitude vs data-only.",
        font=pair._font_for_width(
            draw,
            "Outlier = quiet-region |map − own 5×5 median| > 0.05; band is 3–8 px amplitude vs data-only.",
            max_width=width - 52,
            preferred_size=24,
            minimum_size=20,
        ),
        fill=(205, 210, 218),
    )
    for group_index, group in enumerate(groups):
        group_y = top + group_index * group_height
        for column, map_name in enumerate(group):
            pair._draw_centered_text(
                draw,
                (left + column * tile_width, group_y, tile_width, group_header),
                pair._display_name(map_name),
                pair._font(27, bold=True),
                "white",
            )
        for row_index, role in enumerate(VARIANT_ROLES):
            y = group_y + group_header + row_index * row_height
            group_quiet = sum(
                int(metrics[name][role]["quiet_region_pixels"]) for name in group
            )
            group_outliers = sum(
                int(
                    metrics[name][role][
                        "quiet_region_median5_residual_outlier_count_gt_0p05"
                    ]
                )
                for name in group
            )
            group_fraction = group_outliers / group_quiet
            label = (
                f"{VARIANT_LABELS[role]}\n"
                f"pooled group outliers {100.0 * group_fraction:.2f}%"
            )
            draw.multiline_text(
                (24, y + 24),
                label,
                font=pair._font_for_width(
                    draw,
                    max(label.splitlines(), key=len),
                    max_width=left - 48,
                    preferred_size=29,
                    minimum_size=23,
                    bold=True,
                ),
                fill="white",
                spacing=12,
            )
            for column, map_name in enumerate(group):
                x = left + column * tile_width
                path = (
                    cameras[role]
                    / "material"
                    / profile
                    / "maps"
                    / f"{map_name}.png"
                )
                with Image.open(path) as image:
                    panel = image.convert("RGB")
                if panel.size != (tile_width, tile_height):
                    raise ValueError(f"material map geometry differs: {path}")
                canvas.paste(panel, (x, y))
                values = metrics[map_name][role]
                outliers = 100.0 * values[
                    "quiet_region_median5_residual_outlier_fraction_gt_0p05"
                ]
                if role == "data_only":
                    band = "ref"
                else:
                    ratio = metrics[map_name]["comparison_to_data_only"][role][
                        "guide_textured_band3_8_amplitude_ratio"
                    ]
                    band = pair._format_ratio(ratio, digits=1)
                pair._draw_centered_multiline_text(
                    draw,
                    (x, y + tile_height, tile_width, caption_height),
                    f"out {outliers:.2f}%\nband {band}",
                    pair._font(17, bold=True),
                    "white",
                    spacing=2,
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _case_row(
    case_metrics: dict[str, Any],
    *,
    lighting: str,
    case_id: str,
    profile: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in case_metrics["rows"]
        if row["lighting"] == lighting
        and row["case_id"] == case_id
        and row["profile"] == profile
    ]
    if len(matches) != 1:
        raise ValueError(f"case row is missing or ambiguous: {lighting}/{case_id}/{profile}")
    return matches[0]


def _strategy_case_panels(
    cameras: dict[str, Path],
    *,
    lighting: str,
    case_id: str,
    profile: str,
) -> list[tuple[str, Image.Image, bool]]:
    baseline_case = (
        cameras["data_only"] / "evaluation" / lighting / "cases" / case_id
    )
    reference = pair._read_rgb_map(baseline_case / "reference.png")
    predictions = {
        role: pair._read_rgb_map(
            cameras[role]
            / "evaluation"
            / lighting
            / "cases"
            / case_id
            / "predictions"
            / f"{profile}.png"
        )
        for role in VARIANT_ROLES
    }

    def rgb_image(values: np.ndarray) -> Image.Image:
        return Image.fromarray(
            np.floor(np.clip(values, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        )

    panels: list[tuple[str, Image.Image, bool]] = []
    if lighting == "hdri":
        with Image.open(baseline_case / "lighting.png") as image:
            panels.append(("Lighting", image.convert("RGB"), True))
    panels.append(("Reference", rgb_image(reference), False))
    panels.extend(
        (
            ("Data-only", rgb_image(predictions["data_only"]), False),
            ("Post-fit", rgb_image(predictions["post_fit_cleanup"]), False),
            ("Train-time", rgb_image(predictions["train_time_regularizer"]), False),
            (
                "Data error",
                Image.fromarray(
                    pair._scalar_error_heatmap_u8(
                        predictions["data_only"], reference
                    )
                ),
                False,
            ),
            (
                "Post error",
                Image.fromarray(
                    pair._scalar_error_heatmap_u8(
                        predictions["post_fit_cleanup"], reference
                    )
                ),
                False,
            ),
            (
                "Train error",
                Image.fromarray(
                    pair._scalar_error_heatmap_u8(
                        predictions["train_time_regularizer"], reference
                    )
                ),
                False,
            ),
        )
    )
    return panels


def _write_evaluation_sheet(
    cameras: dict[str, Path],
    profiles: Sequence[str],
    lighting: str,
    case_metrics: dict[str, Any],
    output: Path,
) -> None:
    selection = case_metrics["selected_cases"][lighting]
    selected = (
        ("Baseline-median PSNR case", selection["representative"]),
        ("Joint worst candidate ΔPSNR case", selection["audit"]),
    )
    first_case = str(selection["representative"]["case_id"])
    with Image.open(
        cameras["data_only"]
        / "evaluation"
        / lighting
        / "cases"
        / first_case
        / "reference.png"
    ) as reference:
        tile_width, tile_height = reference.size
    columns = [name for name, _, _ in _strategy_case_panels(
        cameras,
        lighting=lighting,
        case_id=first_case,
        profile=profiles[0],
    )]
    left = 430
    top = 252
    header = 88
    gap = 28
    section_height = header + len(profiles) * tile_height
    width = max(1500, left + len(columns) * tile_width)
    height = top + len(selected) * section_height + gap + 24
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (24, 18),
        f"{lighting.upper()} held-out relighting · three material strategies",
        font=pair._font(45, bold=True),
        fill="white",
    )
    draw.text(
        (24, 78),
        (
            "Representative selection uses data-only PSNR only. Audit selection is "
            "the worst mean candidate ΔPSNR. Errors share the fixed 0–0.25 scale."
        ),
        font=pair._font_for_width(
            draw,
            "Representative selection uses data-only PSNR only. Audit selection is the worst mean candidate ΔPSNR. Errors share the fixed 0–0.25 scale.",
            max_width=width - 48,
            preferred_size=25,
            minimum_size=20,
        ),
        fill=(210, 215, 222),
    )
    for column, name in enumerate(columns):
        pair._draw_centered_text(
            draw,
            (left + column * tile_width, 196, tile_width, 42),
            name,
            pair._font(20, bold=True),
            "white",
        )
    for section_index, (role_label, selected_case) in enumerate(selected):
        section_y = top + section_index * (section_height + gap)
        case_id = str(selected_case["case_id"])
        trigger = selected_case.get("trigger_variant")
        trigger_text = f" · trigger {VARIANT_LABELS[trigger]}" if trigger else ""
        case_label = f"{role_label} · {case_id}{trigger_text}"
        draw.rounded_rectangle(
            (18, section_y + 4, width - 18, section_y + header - 4),
            radius=10,
            fill=(29, 34, 42),
        )
        draw.text(
            (30, section_y + 26),
            case_label,
            font=pair._font_for_width(
                draw,
                case_label,
                max_width=width - 60,
                preferred_size=26,
                minimum_size=18,
                bold=True,
            ),
            fill=(232, 236, 242),
        )
        for row_index, profile in enumerate(profiles):
            y = section_y + header + row_index * tile_height
            row = _case_row(
                case_metrics,
                lighting=lighting,
                case_id=case_id,
                profile=profile,
            )
            baseline = row["metrics"]["data_only"]
            post_delta = row["delta_vs_data_only"]["post_fit_cleanup"]
            train_delta = row["delta_vs_data_only"]["train_time_regularizer"]
            text = (
                f"{profile.upper()} fit\n"
                f"Data {baseline['psnr']:.2f} dB / {baseline['ssim_global']:.3f}\n"
                f"Post Δ {post_delta['psnr']:+.3f} / {post_delta['ssim_global']:+.4f}\n"
                f"Train Δ {train_delta['psnr']:+.3f} / {train_delta['ssim_global']:+.4f}"
            )
            draw.multiline_text(
                (24, y + 28),
                text,
                font=pair._font(22, bold=True),
                fill="white",
                spacing=9,
            )
            panels = _strategy_case_panels(
                cameras,
                lighting=lighting,
                case_id=case_id,
                profile=profile,
            )
            if [name for name, _, _ in panels] != columns:
                raise ValueError("evaluation panel columns differ from their header")
            for column, (_, image, contain) in enumerate(panels):
                panel = (
                    pair._contain_image(image, (tile_width, tile_height))
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
    material: dict[str, Any],
    evaluation: dict[str, Any],
    cases: dict[str, Any],
    qualifications: dict[str, Any],
    profiles: Sequence[str],
) -> int:
    rows: list[dict[str, Any]] = []

    def add(
        category: str,
        profile: str,
        target: str,
        metric: str,
        role: str,
        value: Any,
        *,
        baseline: Any = None,
        unit: str = "",
    ) -> None:
        delta: Any = ""
        ratio: Any = ""
        if (
            role != "data_only"
            and isinstance(value, (int, float))
            and isinstance(baseline, (int, float))
        ):
            delta = float(value) - float(baseline)
            ratio = (
                float(value) / float(baseline)
                if abs(float(baseline)) > 1e-12
                else ""
            )
        rows.append(
            {
                "category": category,
                "profile": profile,
                "target": target,
                "metric": metric,
                "variant": role,
                "value": value,
                "delta_vs_data_only": delta,
                "ratio_vs_data_only": ratio,
                "unit": unit,
            }
        )

    material_names = (
        "neighbor_variation",
        "median5_residual_mae",
        "median5_residual_outlier_fraction_gt_0p05",
        "quiet_region_neighbor_variation",
        "quiet_region_median5_residual_mae",
        "quiet_region_median5_residual_outlier_fraction_gt_0p05",
        "guide_edge_gradient_magnitude",
        "guide_textured_band3_8_energy",
    )
    for profile in profiles:
        for map_name in REPORT_MAPS:
            entry = material[profile][map_name]
            for metric in material_names:
                baseline = entry["data_only"][metric]
                for role in VARIANT_ROLES:
                    add(
                        "material",
                        profile,
                        map_name,
                        metric,
                        role,
                        entry[role][metric],
                        baseline=baseline,
                    )
            for role in ("post_fit_cleanup", "train_time_regularizer"):
                for metric, value in entry["comparison_to_data_only"][role].items():
                    if isinstance(value, (int, float)) or value is None:
                        add(
                            "material_comparison",
                            profile,
                            map_name,
                            metric,
                            role,
                            "" if value is None else value,
                        )
    for profile in profiles:
        for lighting in REPORT_LIGHTING:
            suites = evaluation[profile][lighting]
            shared = set.intersection(
                *(set(suites[role]) for role in VARIANT_ROLES)
            )
            for metric in sorted(shared):
                baseline = suites["data_only"][metric]
                for role in VARIANT_ROLES:
                    add(
                        "relighting",
                        profile,
                        lighting,
                        metric,
                        role,
                        suites[role][metric],
                        baseline=baseline,
                    )
    for row in cases["rows"]:
        for metric, baseline in row["metrics"]["data_only"].items():
            for role in VARIANT_ROLES:
                add(
                    "case_png_relighting",
                    row["profile"],
                    f"{row['lighting']}/{row['case_id']}",
                    metric,
                    role,
                    row["metrics"][role][metric],
                    baseline=baseline,
                )
    for role, qualification in qualifications.items():
        for gate_name, gate in qualification["gates"].items():
            if gate.get("operator") == "within_inclusive_range":
                for suffix in ("minimum", "maximum"):
                    add(
                        "qualification_gate",
                        "aggregate",
                        gate_name,
                        f"observed_{suffix}",
                        role,
                        gate.get(f"observed_{suffix}"),
                    )
            else:
                add(
                    "qualification_gate",
                    "aggregate",
                    gate_name,
                    "observed",
                    role,
                    gate.get("observed"),
                )
            add(
                "qualification_gate",
                "aggregate",
                gate_name,
                "meets_threshold",
                role,
                gate.get("meets_threshold"),
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
                "variant",
                "value",
                "delta_vs_data_only",
                "ratio_vs_data_only",
                "unit",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _write_overview(
    cameras: dict[str, Path],
    profiles: Sequence[str],
    material: dict[str, Any],
    evaluation: dict[str, Any],
    cases: dict[str, Any],
    summary: dict[str, Any],
    interior_mask: np.ndarray,
    output: Path,
) -> None:
    with Image.open(
        cameras["data_only"]
        / "material"
        / profiles[0]
        / "maps"
        / f"{OVERVIEW_HERO_MAPS[0]}.png"
    ) as first:
        source_width, source_height = first.size
    top_height = 650
    material_tile_width = 140
    material_tile_height = round(material_tile_width * source_height / source_width)
    material_heading = 132
    material_caption = 48
    material_row_height = material_tile_height + material_caption
    material_height = material_heading + len(profiles) * material_row_height + 42
    detail_height = 590
    evaluation_tile_width = 190
    evaluation_tile_height = round(
        evaluation_tile_width * source_height / source_width
    )
    evaluation_heading = 112
    evaluation_row_height = evaluation_tile_height + 72
    evaluation_height = evaluation_heading + len(REPORT_LIGHTING) * evaluation_row_height
    height = top_height + material_height + detail_height + evaluation_height + 50
    canvas = Image.new("RGB", (OVERVIEW_WIDTH, height), (15, 17, 21))
    draw = ImageDraw.Draw(canvas)
    draw.text((38, 24), summary["title"], font=pair._font(58, bold=True), fill="white")
    flow = (
        "Controlled inputs and held-out cases · post-fit cleanup changes maps · "
        "train-time regularization changes optimization"
    )
    draw.text(
        (42, 96),
        flow,
        font=pair._font_for_width(
            draw,
            flow,
            max_width=OVERVIEW_WIDTH - 84,
            preferred_size=30,
            minimum_size=26,
            bold=True,
        ),
        fill=(151, 205, 255),
    )
    card_left = 38
    card_top = 155
    card_gap = 22
    card_width = (OVERVIEW_WIDTH - 2 * card_left - 2 * card_gap) // 3
    card_height = 402
    colors = {
        "data_only": (115, 125, 142),
        "post_fit_cleanup": (57, 136, 220),
        "train_time_regularizer": (225, 142, 48),
    }
    aggregate = summary["aggregate"]
    base_outlier = aggregate["quiet_outlier_fraction"]["data_only"]
    for index, role in enumerate(VARIANT_ROLES):
        x = card_left + index * (card_width + card_gap)
        draw.rounded_rectangle(
            (x, card_top, x + card_width, card_top + card_height),
            radius=18,
            fill=(25, 29, 36),
            outline=colors[role],
            width=4,
        )
        draw.text(
            (x + 24, card_top + 22),
            VARIANT_LABELS[role],
            font=pair._font_for_width(
                draw,
                VARIANT_LABELS[role],
                max_width=card_width - 48,
                preferred_size=34,
                minimum_size=27,
                bold=True,
            ),
            fill="white",
        )
        stage = summary["variants"][role]["stage"]
        if role == "data_only":
            stage_line = f"{stage['data_steps']} data steps · no map update"
        elif role == "post_fit_cleanup":
            stage_line = "same fit · one deterministic update"
        else:
            stage_line = (
                f"{stage['data_warmup_steps']} warmup + "
                f"{stage['regularized_steps']} joint steps"
            )
        outlier = aggregate["quiet_outlier_fraction"][role]
        reduction = (
            None
            if role == "data_only"
            else (base_outlier - outlier) / base_outlier
        )
        texture = aggregate["texture_band3_8_amplitude_ratio_vs_data_only"][role]
        qualification = aggregate["qualification"][role]
        olat = aggregate["relighting"]["olat"][role]
        hdri = aggregate["relighting"]["hdri"][role]
        lines = [
            stage_line,
            (
                f"Quiet outliers  {100.0 * outlier:.2f}%"
                + (
                    "  (reference)"
                    if reduction is None
                    else f"  ({100.0 * reduction:+.1f}% reduction)"
                )
            ),
            (
                "3–8 px texture  "
                + ("100.00%" if texture is None else f"{100.0 * texture:.2f}%")
            ),
            (
                f"OLAT  {olat['psnr']:.2f} dB / {olat['ssim_global']:.4f}"
                if role == "data_only"
                else (
                    f"OLAT Δ  {olat['delta_psnr_vs_data_only']:+.3f} dB / "
                    f"{olat['delta_ssim_vs_data_only']:+.4f}"
                )
            ),
            (
                f"HDRI  {hdri['psnr']:.2f} dB / {hdri['ssim_global']:.4f}"
                if role == "data_only"
                else (
                    f"HDRI Δ  {hdri['delta_psnr_vs_data_only']:+.3f} dB / "
                    f"{hdri['delta_ssim_vs_data_only']:+.4f}"
                )
            ),
            (
                "Safeguards  reference"
                if role == "data_only"
                else (
                    f"Safeguards  {qualification['status']} · "
                    f"{qualification['thresholds_met']}/{qualification['gate_count']}"
                )
            ),
        ]
        draw.multiline_text(
            (x + 24, card_top + 86),
            "\n".join(lines),
            font=pair._font_for_width(
                draw,
                max(lines, key=len),
                max_width=card_width - 48,
                preferred_size=OVERVIEW_BODY_FONT,
                minimum_size=24,
                bold=True,
            ),
            fill=(232, 235, 241),
            spacing=18,
        )
    disclaimer = (
        "No winner is inferred from residual counts alone. Texture, edges, and "
        "held-out relighting are co-equal checks; detailed metrics are in metrics.csv."
    )
    draw.text(
        (42, 585),
        disclaimer,
        font=pair._font_for_width(
            draw,
            disclaimer,
            max_width=OVERVIEW_WIDTH - 84,
            preferred_size=28,
            minimum_size=24,
            bold=True,
        ),
        fill=(244, 211, 119),
    )

    material_top = top_height
    draw.text(
        (38, material_top + 10),
        "Material maps · four representative scalar maps, all three fit profiles",
        font=pair._font(43, bold=True),
        fill="white",
    )
    material_left = 360
    for map_index, map_name in enumerate(OVERVIEW_HERO_MAPS):
        group_x = material_left + map_index * len(VARIANT_ROLES) * material_tile_width
        pair._draw_centered_text(
            draw,
            (
                group_x,
                material_top + 62,
                len(VARIANT_ROLES) * material_tile_width,
                34,
            ),
            pair._display_name(map_name),
            pair._font(27, bold=True),
            "white",
        )
        for role_index, role in enumerate(VARIANT_ROLES):
            pair._draw_centered_text(
                draw,
                (
                    group_x + role_index * material_tile_width,
                    material_top + 98,
                    material_tile_width,
                    28,
                ),
                {"data_only": "Data", "post_fit_cleanup": "Post", "train_time_regularizer": "Train"}[role],
                pair._font(21, bold=True),
                (215, 218, 225),
            )
    for profile_index, profile in enumerate(profiles):
        y = material_top + material_heading + profile_index * material_row_height
        profile_outliers = {}
        for role in VARIANT_ROLES:
            pixels = sum(
                int(material[profile][name][role]["quiet_region_pixels"])
                for name in REPORT_MAPS
            )
            flagged = sum(
                int(
                    material[profile][name][role][
                        "quiet_region_median5_residual_outlier_count_gt_0p05"
                    ]
                )
                for name in REPORT_MAPS
            )
            profile_outliers[role] = flagged / pixels
        label = (
            f"{profile.upper()} fit\n"
            f"quiet outliers\n"
            f"Data  {100 * profile_outliers['data_only']:.2f}%\n"
            f"Post  {100 * profile_outliers['post_fit_cleanup']:.2f}%\n"
            f"Train {100 * profile_outliers['train_time_regularizer']:.2f}%"
        )
        draw.multiline_text(
            (40, y + 18),
            label,
            font=pair._font(24, bold=True),
            fill="white",
            spacing=7,
        )
        for map_index, map_name in enumerate(OVERVIEW_HERO_MAPS):
            group_x = material_left + map_index * len(VARIANT_ROLES) * material_tile_width
            for role_index, role in enumerate(VARIANT_ROLES):
                x = group_x + role_index * material_tile_width
                with Image.open(
                    cameras[role]
                    / "material"
                    / profile
                    / "maps"
                    / f"{map_name}.png"
                ) as image:
                    panel = image.convert("RGB").resize(
                        (material_tile_width, material_tile_height),
                        Image.Resampling.LANCZOS,
                    )
                canvas.paste(panel, (x, y))
            post = material[profile][map_name]["comparison_to_data_only"][
                "post_fit_cleanup"
            ]["guide_textured_band3_8_amplitude_ratio"]
            train = material[profile][map_name]["comparison_to_data_only"][
                "train_time_regularizer"
            ]["guide_textured_band3_8_amplitude_ratio"]
            pair._draw_centered_text(
                draw,
                (
                    group_x,
                    y + material_tile_height,
                    len(VARIANT_ROLES) * material_tile_width,
                    material_caption,
                ),
                f"texture: Post {pair._format_ratio(post, digits=1)} · Train {pair._format_ratio(train, digits=1)}",
                pair._font(19, bold=True),
                (220, 222, 228),
            )

    detail_top = material_top + material_height
    detail_profile = summary["overview_detail_crops"]["profile"]
    draw.text(
        (38, detail_top + 10),
        f"Native pixel evidence · {detail_profile.upper()} · crop selected from data-only outliers only",
        font=pair._font(41, bold=True),
        fill="white",
    )
    draw.text(
        (42, detail_top + 63),
        "Top: scalar map crop (nearest-neighbor 2×). Bottom: quiet median-residual outliers on the same fixed crop.",
        font=pair._font(25),
        fill=(210, 214, 222),
    )
    display = OVERVIEW_DETAIL_CROP_SIZE * OVERVIEW_DETAIL_SCALE
    group_width = len(VARIANT_ROLES) * display
    group_gap = 120
    detail_left = (OVERVIEW_WIDTH - 2 * group_width - group_gap) // 2
    maps_dir = cameras["data_only"] / "material" / detail_profile / "maps"
    guide_albedo = pair._read_rgb_map(maps_dir / "baseColor.png")
    guide_normal = pair._read_rgb_map(maps_dir / "normal.png")
    guide_masks, _ = pair._guide_region_masks(
        guide_albedo,
        guide_normal,
        interior_mask,
        albedo_sigma=float(summary["guide_diagnostic"]["albedo_sigma"]),
        normal_sigma=float(summary["guide_diagnostic"]["normal_sigma"]),
    )
    for map_index, map_name in enumerate(OVERVIEW_DETAIL_MAPS):
        group_x = detail_left + map_index * (group_width + group_gap)
        pair._draw_centered_text(
            draw,
            (group_x, detail_top + 98, group_width, 34),
            pair._display_name(map_name),
            pair._font(28, bold=True),
            "white",
        )
        box = tuple(
            summary["overview_detail_crops"]["maps"][map_name]["crop_box_xyxy"]
        )
        left, top, right, bottom = box
        for role_index, role in enumerate(VARIANT_ROLES):
            x = group_x + role_index * display
            pair._draw_centered_text(
                draw,
                (x, detail_top + 133, display, 28),
                {"data_only": "Data-only", "post_fit_cleanup": "Post-fit", "train_time_regularizer": "Train-time"}[role],
                pair._font(21, bold=True),
                (220, 223, 230),
            )
            path = (
                cameras[role]
                / "material"
                / detail_profile
                / "maps"
                / f"{map_name}.png"
            )
            with Image.open(path) as image:
                crop = image.convert("RGB").crop(box).resize(
                    (display, display), Image.Resampling.NEAREST
                )
            canvas.paste(crop, (x, detail_top + 164))
            values = pair._read_scalar_map(path)
            flagged = guide_masks["quiet_pixels"] & (
                pair._own_median5_residual(values) > 0.05
            )
            flagged_crop = flagged[top:bottom, left:right]
            mask_rgb = np.zeros(
                (OVERVIEW_DETAIL_CROP_SIZE, OVERVIEW_DETAIL_CROP_SIZE, 3),
                dtype=np.uint8,
            )
            mask_rgb[flagged_crop] = (240, 52, 68)
            mask_panel = Image.fromarray(mask_rgb).resize(
                (display, display), Image.Resampling.NEAREST
            )
            canvas.paste(mask_panel, (x, detail_top + 164 + display))
            pair._draw_centered_text(
                draw,
                (x, detail_top + 164 + 2 * display, display, 34),
                f"{int(np.count_nonzero(flagged_crop))} flagged",
                pair._font(20, bold=True),
                "white",
            )

    evaluation_top = detail_top + detail_height
    draw.text(
        (38, evaluation_top + 10),
        "Held-out relighting · candidate-independent representative cases",
        font=pair._font(43, bold=True),
        fill="white",
    )
    overview_profile = "mix" if "mix" in profiles else profiles[-1]
    for lighting_index, lighting in enumerate(REPORT_LIGHTING):
        block_y = evaluation_top + evaluation_heading + lighting_index * evaluation_row_height
        selection = cases["selected_cases"][lighting]["representative"]
        case_id = str(selection["case_id"])
        row = _case_row(
            cases,
            lighting=lighting,
            case_id=case_id,
            profile=overview_profile,
        )
        panels = _strategy_case_panels(
            cameras,
            lighting=lighting,
            case_id=case_id,
            profile=overview_profile,
        )
        panel_left = OVERVIEW_WIDTH - len(panels) * evaluation_tile_width - 28
        post = row["delta_vs_data_only"]["post_fit_cleanup"]
        train = row["delta_vs_data_only"]["train_time_regularizer"]
        short_case = case_id if len(case_id) <= 31 else f"{case_id[:15]}…{case_id[-15:]}"
        draw.multiline_text(
            (40, block_y + 50),
            (
                f"{lighting.upper()} · {overview_profile.upper()} fit\n"
                f"{short_case}\n"
                f"Post Δ {post['psnr']:+.3f} dB / {post['ssim_global']:+.4f}\n"
                f"Train Δ {train['psnr']:+.3f} dB / {train['ssim_global']:+.4f}"
            ),
            font=pair._font(23, bold=True),
            fill="white",
            spacing=8,
        )
        for column, (name, image, contain) in enumerate(panels):
            x = panel_left + column * evaluation_tile_width
            pair._draw_centered_text(
                draw,
                (x, block_y, evaluation_tile_width, 44),
                name,
                pair._font(20, bold=True),
                "white",
            )
            panel = (
                pair._contain_image(
                    image,
                    (evaluation_tile_width, evaluation_tile_height),
                    fill=(10, 10, 10),
                )
                if contain
                else image.convert("RGB").resize(
                    (evaluation_tile_width, evaluation_tile_height),
                    Image.Resampling.LANCZOS,
                )
            )
            canvas.paste(panel, (x, block_y + 46))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _record_artifact_metadata(stage: Path, artifacts: dict[str, Any]) -> None:
    entries = [artifacts["overview"]]
    entries.extend(artifacts["material"].values())
    entries.extend(artifacts["evaluation"].values())
    for entry in entries:
        path = stage / entry["path"]
        with Image.open(path) as image:
            entry.update(
                {
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "sha256": _file_sha256(path),
                }
            )
    metrics = artifacts["metrics"]
    metrics_path = stage / metrics["path"]
    metrics.update(
        {
            "bytes": int(metrics_path.stat().st_size),
            "sha256": _file_sha256(metrics_path),
        }
    )


def _validate_report(stage: Path, summary: dict[str, Any]) -> None:
    profiles = tuple(summary["profiles"])
    expected = {
        "overview.png",
        "summary.json",
        "metrics.csv",
        *(f"material/{profile}.png" for profile in profiles),
        *(f"evaluation/{lighting}.png" for lighting in REPORT_LIGHTING),
    }
    actual = {
        str(path.relative_to(stage))
        for path in stage.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise RuntimeError(
            "material strategy report file contract differs: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    stored = json.loads((stage / "summary.json").read_text(encoding="utf-8"))
    if stored != summary:
        raise RuntimeError("stored material strategy summary differs from memory")
    if stored.get("schema") != REPORT_SCHEMA:
        raise RuntimeError("material strategy report schema is invalid")
    if stored.get("variant_order") != list(VARIANT_ROLES):
        raise RuntimeError("material strategy report variant order is invalid")
    if set(stored.get("variants", {})) != set(VARIANT_ROLES):
        raise RuntimeError("material strategy report roles are incomplete")
    expected_paths = {
        "overview.png",
        *(f"material/{profile}.png" for profile in profiles),
        *(f"evaluation/{lighting}.png" for lighting in REPORT_LIGHTING),
    }
    image_entries = [stored["artifacts"]["overview"]]
    image_entries.extend(stored["artifacts"]["material"].values())
    image_entries.extend(stored["artifacts"]["evaluation"].values())
    if {entry.get("path") for entry in image_entries} != expected_paths:
        raise RuntimeError("material strategy artifact paths are inconsistent")
    for entry in image_entries:
        path = stage / entry["path"]
        with Image.open(path) as image:
            if image.mode != "RGB" or image.width <= 0 or image.height <= 0:
                raise RuntimeError(f"report image contract is invalid: {entry['path']}")
            if [image.width, image.height] != [entry["width"], entry["height"]]:
                raise RuntimeError(f"report image geometry metadata differs: {entry['path']}")
        if _file_sha256(path) != entry.get("sha256"):
            raise RuntimeError(f"report image hash metadata differs: {entry['path']}")
    if stored["artifacts"]["overview"]["width"] != OVERVIEW_WIDTH:
        raise RuntimeError("material strategy overview has the wrong width")
    metrics_path = stage / stored["artifacts"]["metrics"]["path"]
    if _file_sha256(metrics_path) != stored["artifacts"]["metrics"].get("sha256"):
        raise RuntimeError("material strategy metrics hash metadata differs")
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != stored.get("metrics_csv_rows") or not rows:
        raise RuntimeError("material strategy metrics CSV coverage is invalid")
    required_columns = {
        "category",
        "profile",
        "target",
        "metric",
        "variant",
        "value",
        "delta_vs_data_only",
        "ratio_vs_data_only",
        "unit",
    }
    if set(rows[0]) != required_columns:
        raise RuntimeError("material strategy metrics CSV columns are invalid")
    metric_roles = {row["variant"] for row in rows if row["category"] != "qualification_gate"}
    if metric_roles != set(VARIANT_ROLES):
        raise RuntimeError("material strategy metrics CSV roles are incomplete")
    if set(stored.get("qualifications", {})) != {
        "post_fit_cleanup",
        "train_time_regularizer",
    }:
        raise RuntimeError("material strategy qualifications are incomplete")
    train_audit = stored.get("train_time_regularizer_audit")
    if (
        not isinstance(train_audit, dict)
        or train_audit.get("schema")
        != "ictpolarreal.train-time-frequency-consensus-audit.v1"
        or train_audit.get("validated") is not True
        or set(train_audit.get("profiles", {})) != set(profiles)
        or any(
            record.get("validated") is not True
            or record.get("role") != "train_time_frozen_target"
            for record in train_audit.get("profiles", {}).values()
        )
    ):
        raise RuntimeError("train-time regularizer audit is incomplete")


if __name__ == "__main__":
    main()
