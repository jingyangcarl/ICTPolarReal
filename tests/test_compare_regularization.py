from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ictpolarreal.processing import compare_regularization


def test_long_evaluation_case_label_fits_report_width_without_clipping():
    canvas = Image.new("RGB", (1992, 190), "black")
    draw = ImageDraw.Draw(canvas)
    label = (
        "Shared case · "
        "hdriheaven_original_missile_launch_facility_01_2k_1k_22e59f24_rot000"
    )

    font = compare_regularization._font_for_width(
        draw,
        label,
        max_width=canvas.width - 48,
        preferred_size=27,
        minimum_size=20,
        bold=True,
    )
    bounds = draw.textbbox((0, 0), label, font=font)

    assert bounds[2] - bounds[0] <= canvas.width - 48
    assert font.size >= 20


def test_overview_profile_metrics_fit_before_material_grid():
    canvas = Image.new("RGB", (1800, 300), "black")
    draw = ImageDraw.Draw(canvas)
    text = (
        "MIX fit\n"
        "Quiet 0.045 → 0.044\n"
        "Magnitude median 100%\n"
        "Worst anisotropic 100%\n"
        "Cosine median 1.00"
    )
    bounds = draw.multiline_textbbox(
        (36, 18),
        text,
        font=compare_regularization._font(21, bold=True),
        spacing=8,
    )

    assert bounds[2] <= compare_regularization.OVERVIEW_MATERIAL_LEFT - 12


def _write_png(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8)).save(path)


def _write_synthetic_camera(root, *, tv_weight, noisy, profiles=("olat",)):
    manifest = {
        "profiles": list(profiles),
        "evaluation": {"status": "complete"},
    }
    (root / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    common = {
        "model": "Disney",
        "steps": 100,
        "learning_rate": 1e-3,
        "base_color_source": "dataset_albedo",
        "input_hashes": {"base_color_sha256": "same"},
        "hdri_condition_weights_sha256": "same",
        "scalar_initialization": {"roughness": 0.5},
        "surface_validity": {"front_facing_pixels": 10},
        "light_split": {"fit_light_indices": [0, 1, 2, 3]},
        "regularization": {
            "kind": "masked_l1_total_variation",
            "weight": tv_weight,
            "parameters": list(compare_regularization.SCALAR_MAPS),
        },
        "evaluation": {
            "evaluations": {
                lighting: {
                    "metrics": {
                        "psnr": 20.0 + tv_weight,
                        "ssim_global": 0.8 + tv_weight,
                        "mae": 0.05,
                    }
                }
                for lighting in ("olat", "hdri")
            }
        },
    }
    common["checkpoint_signature"] = {
        "model": common["model"],
        "steps": common["steps"],
        "learning_rate": common["learning_rate"],
        "base_color_source": common["base_color_source"],
        "input_hashes": common["input_hashes"],
        "regularization": {
            "kind": common["regularization"]["kind"],
            "weight": tv_weight,
            "parameters": common["regularization"]["parameters"],
        },
    }
    height, width = 32, 24
    base_color = np.full((height, width, 3), 80, dtype=np.uint8)
    base_color[:, width // 2 :] = 180
    normal = np.zeros((height, width, 3), dtype=np.uint8)
    normal[..., 0:2] = 128
    normal[..., 2] = 255
    for profile in profiles:
        acquisition = root / "material" / profile / "acquisition.json"
        acquisition.parent.mkdir(parents=True, exist_ok=True)
        acquisition.write_text(json.dumps(common), encoding="utf-8")
        maps = root / "material" / profile / "maps"
        _write_png(maps / "baseColor.png", base_color)
        _write_png(maps / "normal.png", normal)
        for index, name in enumerate(compare_regularization.SCALAR_MAPS):
            values = np.full((height, width), 80 + index, dtype=np.uint8)
            values[:, width // 2 :] += 60
            if noisy:
                values[8::4, 3 : width // 2 - 2 : 4] = 220
            _write_png(maps / f"{name}.png", values)

    evaluation_summary = {
        "suites": {
            lighting: {"representative_case": "shared_case"}
            for lighting in ("olat", "hdri")
        }
    }
    summary_path = root / "evaluation" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(evaluation_summary), encoding="utf-8")
    panel = np.full((height, width, 3), 120, dtype=np.uint8)
    for lighting in ("olat", "hdri"):
        case = root / "evaluation" / lighting / "cases" / "shared_case"
        _write_png(case / "reference.png", panel)
        _write_png(case / "lighting.png", panel)
        for profile in profiles:
            _write_png(case / "predictions" / f"{profile}.png", panel)
            _write_png(case / "errors" / f"{profile}.png", panel // 4)


def _array_sha256(values):
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_proximal_impulse_pair(baseline, candidate, *, profile="olat"):
    baseline_path = baseline / "material" / profile / "acquisition.json"
    candidate_path = candidate / "material" / profile / "acquisition.json"
    baseline_acquisition = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate_acquisition = json.loads(candidate_path.read_text(encoding="utf-8"))
    parameters = list(compare_regularization.SCALAR_MAPS)
    # The producer records Disney model order, not the report's display order.
    artifact_map_order = [
        "metallic",
        "subsurface",
        "specular",
        "roughness",
        "specularTint",
        "anisotropic",
        "clearcoat",
        "clearcoatGloss",
    ]
    baseline_regularization = {
        "kind": "masked_l1_total_variation",
        "weight": 0.0,
        "parameters": parameters,
    }
    baseline_signature = copy.deepcopy(baseline_acquisition["checkpoint_signature"])
    baseline_signature.update(
        {
            "schema": "ictpolarreal.end2end-checkpoint.v8",
            "regularization": copy.deepcopy(baseline_regularization),
            "adapter": {
                "schema": "ictpolarreal.profile-acquisition-adapter.v2",
                "algorithm_version": "ictpolarreal-masked-tv-v1",
                "lighting_profiles_sha256": "lighting",
                "optimizer": "Adam with cosine decay",
            },
        }
    )
    baseline_acquisition["regularization"] = baseline_regularization
    baseline_acquisition["checkpoint_signature"] = baseline_signature

    stage_plan = {
        "enabled": True,
        "data_fit_steps": 100,
        "detector_after_data_step": 100,
        "cleanup_iterations": 10,
        "cleanup_fraction": 0.1,
        "shrink_per_iteration": 0.01,
        "total_shrink": 0.1,
        "cleanup_optimizer_steps": 0,
    }
    impulse_settings = {
        "stage": "full_data_fit_then_post_fit_frozen_impulse_proximal",
        "target": "frozen_local_median_after_data_fit",
        "proximal_update": "soft_threshold_to_dead_zone",
        "detector": "absolute_center_minus_local_median",
        "mad_normalization": 1.4826,
        "mad_scale": 4.0,
        "minimum_deviation": 0.035,
        "normalized_robust_score": (
            "absolute_center_minus_median/max(mad_scale*mad_normalization*MAD,"
            "minimum_deviation)"
        ),
        "isolation_rule": "score_gt_1_and_equal_to_3x3_local_maximum",
        "local_maximum_tie_policy": "retain_all_equal_maxima",
        "dead_zone": 0.005,
        "unflagged_parameter_update": "none_bit_identical_to_data_fit",
    }
    artifact_payload = {
        "metadata": np.asarray(
            json.dumps(
                {
                    "schema": "ictpolarreal.impulse-frozen-artifact.v1",
                    "created_after_step": 100,
                    "maps": artifact_map_order,
                },
                sort_keys=True,
            )
        )
    }
    map_provenance = {}
    diagnostic_maps = {}
    for map_name in parameters:
        map_path = baseline / "material" / profile / "maps" / f"{map_name}.png"
        with Image.open(map_path) as image:
            baseline_u8 = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        target = baseline_u8.astype(np.float32) / 255.0
        flagged = np.zeros(target.shape, dtype=bool)
        flagged[10, 15] = True
        target_value_u8 = max(int(baseline_u8[10, 15]) - 20, 0)
        target[10, 15] = np.float32(target_value_u8 / 255.0)
        candidate_u8 = baseline_u8.copy()
        candidate_u8[10, 15] = target_value_u8
        _write_png(
            candidate / "material" / profile / "maps" / f"{map_name}.png",
            candidate_u8,
        )
        artifact_payload[f"{map_name}__target"] = target
        artifact_payload[f"{map_name}__mask"] = flagged
        map_provenance[map_name] = {
            "flagged_centers": 1,
            "target_sha256": _array_sha256(target),
            "mask_sha256": _array_sha256(flagged),
        }
        diagnostic_maps[map_name] = {
            "flagged_centers": 1,
            "moved_centers": 1,
            "mean_distance_before": 20.0 / 255.0,
            "mean_distance_after": 0.0,
            "max_distance_before": 20.0 / 255.0,
            "max_distance_after": 0.0,
        }
    artifact_path = (
        candidate / "material" / profile / "impulse_median_frozen.npz"
    )
    with artifact_path.open("wb") as stream:
        np.savez_compressed(stream, **artifact_payload)
    frozen_bundle = {
        "schema": "ictpolarreal.impulse-median-bundle.v1",
        "created_after_step": 100,
        "maps": map_provenance,
        "total_flagged_centers": len(parameters),
        "artifact": {
            "schema": "ictpolarreal.impulse-frozen-artifact.v1",
            "path": artifact_path.name,
            "sha256": _file_sha256(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "format": "numpy_npz_compressed",
        },
    }
    impulse_signature_regularization = {
        "kind": "impulse-median",
        "weight": 0.01,
        "parameters": parameters,
        "settings": impulse_settings,
        "stage_plan": stage_plan,
    }
    candidate_signature = copy.deepcopy(baseline_signature)
    candidate_signature.update(
        {
            "schema": "ictpolarreal.end2end-checkpoint.v12",
            "regularization": copy.deepcopy(impulse_signature_regularization),
        }
    )
    candidate_signature["adapter"].update(
        {
            "schema": "ictpolarreal.profile-acquisition-adapter.v6",
            "algorithm_version": "ictpolarreal-impulse-proximal-v5",
        }
    )
    candidate_acquisition["schema"] = "ictpolarreal.end2end-disney.v12"
    candidate_acquisition["regularization"] = {
        **impulse_signature_regularization,
        "weight_semantics": "constrained_shrink_per_cleanup_iteration",
        "frozen_bundle": frozen_bundle,
        "cleanup_applied": True,
        "cleanup_diagnostic": {
            "schema": "ictpolarreal.impulse-proximal-diagnostic.v1",
            "flagged_centers": len(parameters),
            "moved_centers": len(parameters),
            "total_shrink": 0.1,
            "dead_zone": 0.005,
            "mean_distance_before": 20.0 / 255.0,
            "mean_distance_after": 0.0,
            "max_distance_before": 20.0 / 255.0,
            "max_distance_after": 0.0,
            "maps": diagnostic_maps,
        },
        "data_objective_only": True,
    }
    candidate_acquisition["checkpoint_signature"] = candidate_signature
    baseline_path.write_text(json.dumps(baseline_acquisition), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate_acquisition), encoding="utf-8")
    return artifact_path


def test_map_spatial_metrics_separate_quiet_outliers_and_edge_correspondence():
    mask = np.ones((9, 9), dtype=bool)
    base_color = np.full((9, 9, 3), 0.2, dtype=np.float32)
    base_color[:, 5:] = 0.8
    normal = np.zeros((9, 9, 3), dtype=np.float32)
    normal[..., 0:2] = 0.5
    normal[..., 2] = 1.0
    guide_masks, metadata = compare_regularization._guide_region_masks(
        base_color, normal, mask
    )
    detailed = np.full((9, 9), 0.25, dtype=np.float32)
    detailed[:, 5:] = 0.75
    with_outlier = detailed.copy()
    with_outlier[4, 2] = 1.0
    reversed_edge = 1.0 - detailed

    clean = compare_regularization._map_spatial_metrics(
        detailed, mask, guide_masks
    )
    outlier = compare_regularization._map_spatial_metrics(
        with_outlier, mask, guide_masks
    )

    assert metadata["edge_pair_count"] > 0
    assert clean["quiet_region_neighbor_variation"] == pytest.approx(0.0)
    assert clean[
        "quiet_region_median5_residual_outlier_fraction_gt_0p05"
    ] == pytest.approx(0.0)
    assert clean["guide_edge_gradient_magnitude"] > 0.0
    assert outlier["quiet_region_neighbor_variation"] > 0.0
    assert outlier[
        "quiet_region_median5_residual_outlier_fraction_gt_0p05"
    ] > 0.0
    assert outlier["guide_edge_gradient_magnitude"] == pytest.approx(
        clean["guide_edge_gradient_magnitude"]
    )
    detailed_gradient = compare_regularization._signed_pair_gradients(
        detailed,
        guide_masks["edge_horizontal"],
        guide_masks["edge_vertical"],
    )
    reversed_gradient = compare_regularization._signed_pair_gradients(
        reversed_edge,
        guide_masks["edge_horizontal"],
        guide_masks["edge_vertical"],
    )
    assert compare_regularization._signed_gradient_cosine(
        detailed_gradient, detailed_gradient
    ) == pytest.approx(1.0)
    assert compare_regularization._signed_gradient_cosine(
        detailed_gradient, reversed_gradient
    ) == pytest.approx(-1.0)


def test_impulse_detail_map_ranking_uses_flags_and_display_order_for_ties():
    profile_cleanup = {
        "maps": {
            name: {"flagged_centers": 0}
            for name in compare_regularization.SCALAR_MAPS
        }
    }
    profile_cleanup["maps"]["roughness"]["flagged_centers"] = 4
    profile_cleanup["maps"]["specular"]["flagged_centers"] = 7
    profile_cleanup["maps"]["metallic"]["flagged_centers"] = 7

    assert compare_regularization._rank_impulse_detail_maps(profile_cleanup) == (
        "specular",
        "metallic",
    )


def test_densest_flagged_crop_is_centered_and_deterministic():
    flagged = np.zeros((20, 24), dtype=bool)
    flagged[2, 3] = True
    flagged[12:15, 16:19] = True
    assert compare_regularization._densest_flagged_crop_box(flagged, 5) == (
        15,
        11,
        20,
        16,
    )

    tied = np.zeros((20, 24), dtype=bool)
    tied[2:5, 3:6] = True
    tied[12:15, 16:19] = True
    assert compare_regularization._densest_flagged_crop_box(tied, 5) == (
        2,
        1,
        7,
        6,
    )
    assert compare_regularization._densest_flagged_crop_box(
        np.zeros((20, 24), dtype=bool), 5
    ) == (9, 7, 14, 12)


def test_comparison_contract_allows_only_regularization_weight_to_change():
    acquisition = {
        "model": "Disney",
        "steps": 100,
        "learning_rate": 1e-3,
        "base_color_source": "dataset_albedo",
        "input_hashes": {"base_color_sha256": "same"},
        "hdri_condition_weights_sha256": "same",
        "scalar_initialization": {"roughness": 0.5},
        "surface_validity": {"front_facing_pixels": 10},
        "light_split": {"fit_light_indices": [0, 1, 2, 3]},
        "regularization": {
            "kind": "masked_l1_total_variation",
            "weight": 0.0,
            "parameters": list(compare_regularization.SCALAR_MAPS),
        },
    }
    acquisition["checkpoint_signature"] = {
        "input_hashes": acquisition["input_hashes"],
        "regularization": copy.deepcopy(acquisition["regularization"]),
    }
    regularized = copy.deepcopy(acquisition)
    regularized["regularization"]["weight"] = 0.01
    regularized["checkpoint_signature"]["regularization"]["weight"] = 0.01
    acquisitions = {
        "baseline": {"olat": acquisition},
        "regularized": {"olat": regularized},
    }

    contract = compare_regularization._validate_comparison_contract(
        acquisitions, ("olat",)
    )

    assert contract["controlled"] is True
    assert contract["base_color_source"] == "dataset_albedo"


def test_comparison_contract_rejects_kind_change_without_known_adapter_identity():
    parameters = list(compare_regularization.SCALAR_MAPS)
    baseline = {
        "base_color_source": "dataset_albedo",
        "regularization": {
            "kind": "masked_l1_total_variation",
            "weight": 0.0,
            "parameters": parameters,
        },
        "checkpoint_signature": {
            "input": "same",
            "regularization": {
                "kind": "masked_l1_total_variation",
                "weight": 0.0,
                "parameters": parameters,
            },
        },
    }
    regularized = copy.deepcopy(baseline)
    regularized["regularization"] = {
        "kind": "edge-charbonnier",
        "weight": 0.0025,
        "parameters": parameters,
        "settings": {"guide": "normalized_albedo_and_normal"},
    }
    regularized["checkpoint_signature"]["regularization"] = copy.deepcopy(
        regularized["regularization"]
    )

    with pytest.raises(ValueError, match="known supported adapter"):
        compare_regularization._validate_comparison_contract(
            {
                "baseline": {"olat": baseline},
                "regularized": {"olat": regularized},
            },
            ("olat",),
        )


def test_comparison_contract_rejects_changed_input_hash():
    acquisition = {
        "model": "Disney",
        "steps": 100,
        "learning_rate": 1e-3,
        "base_color_source": "dataset_albedo",
        "input_hashes": {"base_color_sha256": "baseline"},
        "hdri_condition_weights_sha256": "same",
        "scalar_initialization": {"roughness": 0.5},
        "surface_validity": {"front_facing_pixels": 10},
        "light_split": {"fit_light_indices": [0, 1, 2, 3]},
        "regularization": {
            "kind": "masked_l1_total_variation",
            "weight": 0.0,
            "parameters": list(compare_regularization.SCALAR_MAPS),
        },
    }
    acquisition["checkpoint_signature"] = {
        "input_hashes": acquisition["input_hashes"],
        "regularization": copy.deepcopy(acquisition["regularization"]),
    }
    regularized = copy.deepcopy(acquisition)
    regularized["input_hashes"]["base_color_sha256"] = "different"
    regularized["checkpoint_signature"]["input_hashes"] = regularized[
        "input_hashes"
    ]
    regularized["regularization"]["weight"] = 0.01
    regularized["checkpoint_signature"]["regularization"]["weight"] = 0.01

    with pytest.raises(ValueError, match="checkpoint signatures"):
        compare_regularization._validate_comparison_contract(
            {
                "baseline": {"olat": acquisition},
                "regularized": {"olat": regularized},
            },
            ("olat",),
        )


def test_comparison_contract_accepts_zero_weight_v8_to_v9_regularizer_transition():
    parameters = list(compare_regularization.SCALAR_MAPS)
    baseline = {
        "base_color_source": "dataset_albedo",
        "regularization": {
            "kind": "masked_l1_total_variation",
            "weight": 0.0,
            "parameters": parameters,
        },
        "checkpoint_signature": {
            "schema": "ictpolarreal.end2end-checkpoint.v8",
            "normalized_targets_sha256": "targets",
            "foreground_sha256": "foreground",
            "normal_sha256": "normal",
            "fit_indices": [0, 1, 2],
            "regularization": {
                "kind": "masked_l1_total_variation",
                "weight": 0.0,
                "parameters": parameters,
            },
            "adapter": {
                "schema": "ictpolarreal.profile-acquisition-adapter.v2",
                "algorithm_version": "ictpolarreal-masked-tv-v1",
                "lighting_profiles_sha256": "lighting",
                "optimizer": "Adam with cosine decay",
            },
        },
    }
    settings = {
        "penalty": "sqrt(difference_squared + epsilon_squared) - epsilon",
        "guide": "normalized_albedo_and_normal",
        "albedo_difference": "mean_absolute_rgb",
        "normal_difference": "one_minus_cosine",
        "albedo_sigma": 0.05,
        "normal_sigma": 0.02,
    }
    regularized = copy.deepcopy(baseline)
    regularized["regularization"] = {
        "kind": "edge-charbonnier",
        "weight": 0.0025,
        "parameters": parameters,
        "settings": settings,
    }
    regularized["checkpoint_signature"]["schema"] = (
        "ictpolarreal.end2end-checkpoint.v9"
    )
    regularized["checkpoint_signature"]["regularization"] = copy.deepcopy(
        regularized["regularization"]
    )
    regularized["checkpoint_signature"]["adapter"].update(
        {
            "schema": "ictpolarreal.profile-acquisition-adapter.v3",
            "algorithm_version": "ictpolarreal-edge-aware-regularization-v2",
        }
    )

    contract = compare_regularization._validate_comparison_contract(
        {
            "baseline": {"olat": baseline},
            "regularized": {"olat": regularized},
        },
        ("olat",),
    )

    assert contract["baseline_regularizer_inactive"] is True
    assert contract["regularization_kind"] == "edge-charbonnier"
    assert contract["signature_compatibility"] == ["zero-weight-v8-to-v9"]
    diagnostic = compare_regularization._guide_diagnostic_configuration(contract)
    assert diagnostic["source"] == "active_regularizer_settings"
    assert diagnostic["matches_active_regularizer_configuration"] is True
    assert diagnostic["input_precision"] == "8-bit exported material maps"
    assert diagnostic["albedo_sigma"] == pytest.approx(0.05)
    assert diagnostic["normal_sigma"] == pytest.approx(0.02)

    changed_data = copy.deepcopy(regularized)
    changed_data["checkpoint_signature"]["normalized_targets_sha256"] = "changed"
    with pytest.raises(ValueError, match="checkpoint signatures"):
        compare_regularization._validate_comparison_contract(
            {
                "baseline": {"olat": baseline},
                "regularized": {"olat": changed_data},
            },
            ("olat",),
        )


@pytest.mark.parametrize(
    (
        "baseline_schema",
        "baseline_adapter",
        "baseline_kind",
        "candidate_schema",
        "candidate_adapter",
        "expected_mode",
    ),
    (
        (
            "ictpolarreal.end2end-checkpoint.v8",
            (
                "ictpolarreal.profile-acquisition-adapter.v2",
                "ictpolarreal-masked-tv-v1",
            ),
            "masked_l1_total_variation",
            "ictpolarreal.end2end-checkpoint.v10",
            (
                "ictpolarreal.profile-acquisition-adapter.v4",
                "ictpolarreal-impulse-median-v3",
            ),
            "zero-weight-v8-to-v10",
        ),
        (
            "ictpolarreal.end2end-checkpoint.v9",
            (
                "ictpolarreal.profile-acquisition-adapter.v3",
                "ictpolarreal-edge-aware-regularization-v2",
            ),
            "edge-charbonnier",
            "ictpolarreal.end2end-checkpoint.v10",
            (
                "ictpolarreal.profile-acquisition-adapter.v4",
                "ictpolarreal-impulse-median-v3",
            ),
            "zero-weight-v9-to-v10",
        ),
        (
            "ictpolarreal.end2end-checkpoint.v8",
            (
                "ictpolarreal.profile-acquisition-adapter.v2",
                "ictpolarreal-masked-tv-v1",
            ),
            "masked_l1_total_variation",
            "ictpolarreal.end2end-checkpoint.v11",
            (
                "ictpolarreal.profile-acquisition-adapter.v5",
                "ictpolarreal-impulse-proximal-v4",
            ),
            "zero-weight-v8-to-v11",
        ),
        (
            "ictpolarreal.end2end-checkpoint.v9",
            (
                "ictpolarreal.profile-acquisition-adapter.v3",
                "ictpolarreal-edge-aware-regularization-v2",
            ),
            "edge-charbonnier",
            "ictpolarreal.end2end-checkpoint.v11",
            (
                "ictpolarreal.profile-acquisition-adapter.v5",
                "ictpolarreal-impulse-proximal-v4",
            ),
            "zero-weight-v9-to-v11",
        ),
        (
            "ictpolarreal.end2end-checkpoint.v10",
            (
                "ictpolarreal.profile-acquisition-adapter.v4",
                "ictpolarreal-impulse-median-v3",
            ),
            "impulse-median",
            "ictpolarreal.end2end-checkpoint.v11",
            (
                "ictpolarreal.profile-acquisition-adapter.v5",
                "ictpolarreal-impulse-proximal-v4",
            ),
            "zero-weight-v10-to-v11",
        ),
        (
            "ictpolarreal.end2end-checkpoint.v8",
            (
                "ictpolarreal.profile-acquisition-adapter.v2",
                "ictpolarreal-masked-tv-v1",
            ),
            "masked_l1_total_variation",
            "ictpolarreal.end2end-checkpoint.v12",
            (
                "ictpolarreal.profile-acquisition-adapter.v6",
                "ictpolarreal-impulse-proximal-v5",
            ),
            "zero-weight-v8-to-v12",
        ),
        (
            "ictpolarreal.end2end-checkpoint.v11",
            (
                "ictpolarreal.profile-acquisition-adapter.v5",
                "ictpolarreal-impulse-proximal-v4",
            ),
            "impulse-median",
            "ictpolarreal.end2end-checkpoint.v12",
            (
                "ictpolarreal.profile-acquisition-adapter.v6",
                "ictpolarreal-impulse-proximal-v5",
            ),
            "zero-weight-v11-to-v12",
        ),
    ),
)
def test_comparison_contract_accepts_known_zero_weight_to_impulse_transition(
    baseline_schema,
    baseline_adapter,
    baseline_kind,
    candidate_schema,
    candidate_adapter,
    expected_mode,
):
    parameters = list(compare_regularization.SCALAR_MAPS)
    baseline_regularization = {
        "kind": baseline_kind,
        "weight": 0.0,
        "parameters": parameters,
    }
    if baseline_kind == "edge-charbonnier":
        baseline_regularization["settings"] = {
            "guide": "normalized_albedo_and_normal",
            "albedo_sigma": 0.05,
            "normal_sigma": 0.02,
        }
    elif baseline_kind == "impulse-median":
        baseline_regularization["settings"] = {
            "stage": "data_only_main_then_data_plus_frozen_impulse_cleanup",
        }
        baseline_regularization["stage_plan"] = {
            "enabled": False,
            "main_steps": 100,
            "cleanup_steps": 0,
            "cleanup_start_step": 100,
            "cleanup_fraction": 0.0,
        }
    baseline = {
        "base_color_source": "dataset_albedo",
        "regularization": baseline_regularization,
        "checkpoint_signature": {
            "schema": baseline_schema,
            "normalized_targets_sha256": "targets",
            "foreground_sha256": "foreground",
            "normal_sha256": "normal",
            "fit_indices": [0, 1, 2],
            "regularization": copy.deepcopy(baseline_regularization),
            "adapter": {
                "schema": baseline_adapter[0],
                "algorithm_version": baseline_adapter[1],
                "lighting_profiles_sha256": "lighting",
                "optimizer": "Adam with cosine decay",
            },
        },
    }
    stage_plan = {
        "enabled": True,
        "main_steps": 90,
        "cleanup_steps": 10,
        "cleanup_start_step": 90,
        "cleanup_fraction": 0.1,
    }
    impulse_regularization = {
        "kind": "impulse-median",
        "weight": 1.0,
        "parameters": parameters,
        "settings": {
            "stage": "data_only_main_then_data_plus_frozen_impulse_cleanup",
            "detector": "absolute_center_minus_local_median",
            "target": "frozen_local_median_at_main_cleanup_boundary",
        },
        "stage_plan": stage_plan,
    }
    regularized = copy.deepcopy(baseline)
    regularized["regularization"] = impulse_regularization
    regularized["checkpoint_signature"]["schema"] = candidate_schema
    regularized["checkpoint_signature"]["regularization"] = copy.deepcopy(
        impulse_regularization
    )
    regularized["checkpoint_signature"]["adapter"].update(
        {
            "schema": candidate_adapter[0],
            "algorithm_version": candidate_adapter[1],
        }
    )

    contract = compare_regularization._validate_comparison_contract(
        {
            "baseline": {"olat": baseline},
            "regularized": {"olat": regularized},
        },
        ("olat",),
    )

    assert contract["controlled"] is True
    assert contract["regularization_kind"] == "impulse-median"
    assert contract["regularized_stage_plan"] == stage_plan
    assert contract["signature_compatibility"] == [expected_mode]
    assert (
        compare_regularization._display_regularizer("impulse-median")
        == "post-fit impulse proximal"
    )
    diagnostic = compare_regularization._guide_diagnostic_configuration(contract)
    assert diagnostic["source"] == "fixed_report_diagnostic"
    assert diagnostic["matches_active_regularizer_configuration"] is False

    changed_data = copy.deepcopy(regularized)
    changed_data["checkpoint_signature"]["normalized_targets_sha256"] = "changed"
    with pytest.raises(ValueError, match="checkpoint signatures"):
        compare_regularization._validate_comparison_contract(
            {
                "baseline": {"olat": baseline},
                "regularized": {"olat": changed_data},
            },
            ("olat",),
        )


def test_impulse_regularization_requires_matching_stage_plan_provenance():
    parameters = list(compare_regularization.SCALAR_MAPS)
    acquisition = {
        "regularization": {
            "kind": "impulse-median",
            "weight": 1.0,
            "parameters": parameters,
            "settings": {"detector": "local_median"},
            "stage_plan": {"enabled": True, "cleanup_steps": 10},
        }
    }
    acquisition["checkpoint_signature"] = {
        "regularization": copy.deepcopy(acquisition["regularization"])
    }
    configuration = compare_regularization._regularization_configuration(acquisition)
    assert configuration[0] == "impulse-median"
    assert json.loads(configuration[4]) == {
        "enabled": True,
        "cleanup_steps": 10,
    }

    acquisition["checkpoint_signature"]["regularization"]["stage_plan"][
        "cleanup_steps"
    ] = 9
    with pytest.raises(ValueError, match="regularization provenance differ"):
        compare_regularization._regularization_configuration(acquisition)

    acquisition["regularization"]["stage_plan"] = "invalid"
    with pytest.raises(ValueError, match="invalid regularization stage plan"):
        compare_regularization._regularization_configuration(acquisition)


def test_guide_diagnostic_rejects_incompatible_edge_settings():
    contract = {
        "regularization_kind": "edge-charbonnier",
        "regularized_settings": {
            "guide": "unrelated_guide",
            "albedo_sigma": 0.05,
            "normal_sigma": 0.02,
        },
    }
    with pytest.raises(ValueError, match="does not support guide"):
        compare_regularization._guide_diagnostic_configuration(contract)

    contract["regularized_settings"]["guide"] = "normalized_albedo_and_normal"
    contract["regularized_settings"]["normal_sigma"] = 0.0
    with pytest.raises(ValueError, match="normal guide sigma"):
        compare_regularization._guide_diagnostic_configuration(contract)


def test_compose_regularization_comparison_writes_clean_report(tmp_path):
    baseline = tmp_path / "baseline" / "object" / "cam07"
    regularized = tmp_path / "regularized" / "object" / "cam07"
    _write_synthetic_camera(baseline, tv_weight=0.0, noisy=True)
    _write_synthetic_camera(regularized, tv_weight=0.01, noisy=False)
    mask = np.full((32, 24), 255, dtype=np.uint8)
    mask_path = tmp_path / "mask.png"
    _write_png(mask_path, mask)

    summary = compare_regularization.compose_regularization_comparison(
        baseline,
        regularized,
        tmp_path / "comparison",
        mask_path=mask_path,
    )

    report = tmp_path / "comparison"
    assert summary["comparison_contract"]["controlled"] is True
    assert summary["schema"] == "ictpolarreal.regularization-comparison.v3"
    assert summary["title"] == "Camera 07 · Regularization comparison"
    spatial = summary["aggregate_map_spatial_statistics"]
    assert spatial["quiet_region_neighbor_variation"][
        "relative_change_fraction"
    ] < 0
    assert spatial[
        "quiet_region_median5_residual_outlier_fraction_gt_0p05"
    ]["relative_change_fraction"] < 0
    assert spatial["guide_edge_gradient_magnitude"][
        "magnitude_ratio"
    ] == pytest.approx(1.0)
    preservation = summary["guide_edge_preservation"]
    assert preservation["meaningful_map_count"] == len(
        compare_regularization.SCALAR_MAPS
    )
    assert preservation["gradient_magnitude_ratio"]["median"] == pytest.approx(1.0)
    assert preservation["signed_gradient_cosine"]["median"] == pytest.approx(1.0)
    assert "aggregate_map_noise" not in summary
    serialized = json.dumps(summary).lower()
    assert "speckle" not in serialized
    assert "noise" not in serialized
    assert "retention" not in serialized
    assert (
        "do not establish that all material texture is preserved" in serialized
    )
    assert (report / "overview.png").is_file()
    with Image.open(report / "overview.png") as overview:
        assert overview.width == 1800
        assert overview.height > 2000
    assert (report / "material" / "olat.png").is_file()
    with Image.open(report / "material" / "olat.png") as material_sheet:
        assert material_sheet.width == (
            compare_regularization.MATERIAL_LABEL_GUTTER + 4 * 24
        )
    label_font = compare_regularization._font(29, bold=True)
    label_width = label_font.getbbox("edge-aware Charbonnier λ=0.00125")[2]
    assert 24 + label_width < compare_regularization.MATERIAL_LABEL_GUTTER
    impulse_label_width = label_font.getbbox(
        "post-fit impulse proximal λ=0.00125"
    )[2]
    assert 24 + impulse_label_width < compare_regularization.MATERIAL_LABEL_GUTTER
    assert (report / "evaluation" / "olat.png").is_file()
    assert (report / "evaluation" / "hdri.png").is_file()
    assert (report / "metrics.csv").is_file()
    metrics_text = (report / "metrics.csv").read_text(encoding="utf-8").lower()
    assert "guide_edge_signed_gradient_cosine" in metrics_text
    assert "speckle" not in metrics_text
    assert "noise" not in metrics_text
    assert not (tmp_path / ".comparison.tmp").exists()


def test_compose_proximal_impulse_report_accepts_model_map_order_and_metrics(
    tmp_path,
):
    baseline = tmp_path / "baseline" / "object" / "cam07"
    candidate = tmp_path / "candidate" / "object" / "cam07"
    _write_synthetic_camera(baseline, tv_weight=0.0, noisy=False)
    _write_synthetic_camera(candidate, tv_weight=0.0, noisy=False)
    artifact_path = _make_proximal_impulse_pair(baseline, candidate)
    mask_path = tmp_path / "mask.png"
    _write_png(mask_path, np.full((32, 24), 255, dtype=np.uint8))

    summary = compare_regularization.compose_regularization_comparison(
        baseline,
        candidate,
        tmp_path / "comparison",
        mask_path=mask_path,
    )

    assert summary["comparison_contract"]["signature_compatibility"] == [
        "zero-weight-v8-to-v12"
    ]
    settings = summary["comparison_contract"]["regularized_settings"]
    assert settings["mad_scale"] == pytest.approx(4.0)
    assert settings["isolation_rule"] == (
        "score_gt_1_and_equal_to_3x3_local_maximum"
    )
    assert settings["local_maximum_tie_policy"] == "retain_all_equal_maxima"
    cleanup = summary["impulse_cleanup"]
    assert cleanup["available"] is True
    aggregate = cleanup["aggregate"]
    assert aggregate["flagged"]["map_entries"] == len(
        compare_regularization.SCALAR_MAPS
    )
    assert aggregate["flagged"]["map_entry_fraction"] == pytest.approx(
        1.0 / (32 * 24)
    )
    distance = aggregate["mean_absolute_distance_to_frozen_target"]
    assert distance["baseline"] == pytest.approx(20.0 / 255.0)
    assert distance["candidate"] == pytest.approx(0.0)
    assert aggregate["resolved_to_dead_zone"]["fraction"] == pytest.approx(1.0)
    outside_union = aggregate["exported_change_outside_spatial_flag_union"]
    assert outside_union["mean_absolute"] == pytest.approx(0.0)
    assert outside_union["maximum_absolute"] == pytest.approx(0.0)
    outside_per_map = aggregate["exported_change_outside_per_map_flags"]
    assert outside_per_map["mean_absolute"] == pytest.approx(0.0)
    assert outside_per_map["maximum_absolute"] == pytest.approx(0.0)
    profile_cleanup = cleanup["profiles"]["olat"]
    assert profile_cleanup["artifact"]["validated"] is True
    assert profile_cleanup["flagged"]["spatial_union_pixels"] == 1
    metrics = (tmp_path / "comparison" / "metrics.csv").read_text(
        encoding="utf-8"
    )
    assert "flagged_map_entry_fraction" in metrics
    assert "mean_absolute_distance_to_frozen_target" in metrics
    assert "resolved_to_dead_zone_fraction" in metrics
    assert "outside_spatial_flag_union_maximum_absolute_exported_change" in metrics
    serialized = json.dumps(summary).lower()
    assert "do not establish that all material texture is preserved" in serialized

    artifact_bytes = bytearray(artifact_path.read_bytes())
    artifact_bytes[len(artifact_bytes) // 2] ^= 1
    artifact_path.write_bytes(artifact_bytes)
    with pytest.raises(ValueError, match="SHA-256 does not match provenance"):
        compare_regularization.compose_regularization_comparison(
            baseline,
            candidate,
            tmp_path / "corrupt-comparison",
            mask_path=mask_path,
        )


def test_compose_regularization_comparison_filters_to_shared_profile(tmp_path):
    baseline = tmp_path / "baseline" / "object" / "cam07"
    regularized = tmp_path / "regularized" / "object" / "cam07"
    _write_synthetic_camera(
        baseline,
        tv_weight=0.0,
        noisy=True,
        profiles=("olat", "mix"),
    )
    _write_synthetic_camera(regularized, tv_weight=0.01, noisy=False)
    mask_path = tmp_path / "mask.png"
    _write_png(mask_path, np.full((32, 24), 255, dtype=np.uint8))

    with pytest.raises(ValueError, match="--profiles"):
        compare_regularization.compose_regularization_comparison(
            baseline,
            regularized,
            tmp_path / "unfiltered",
            mask_path=mask_path,
        )

    summary = compare_regularization.compose_regularization_comparison(
        baseline,
        regularized,
        tmp_path / "comparison",
        mask_path=mask_path,
        profiles=("olat",),
    )

    assert summary["profiles"] == ["olat"]
    assert (tmp_path / "comparison" / "material" / "olat.png").is_file()
    assert not (tmp_path / "comparison" / "material" / "mix.png").exists()


def test_compose_regularization_comparison_rejects_overlapping_output(tmp_path):
    baseline = tmp_path / "baseline" / "object" / "cam07"
    regularized = tmp_path / "regularized" / "object" / "cam07"
    _write_synthetic_camera(baseline, tv_weight=0.0, noisy=True)
    _write_synthetic_camera(regularized, tv_weight=0.01, noisy=False)

    with pytest.raises(ValueError, match="must not overlap"):
        compare_regularization.compose_regularization_comparison(
            baseline,
            regularized,
            baseline / "comparison",
        )
