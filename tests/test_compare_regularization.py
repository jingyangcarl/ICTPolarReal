from __future__ import annotations

import copy
import hashlib
import json
import shutil

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ictpolarreal.processing import compare_regularization, end2end_acquisition


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


def test_frequency_overview_headlines_show_cleanup_evidence_at_native_width():
    def gate(baseline, regularized, reduction):
        return {
            "baseline_value": baseline,
            "regularized_value": regularized,
            "observed": reduction,
        }

    report = {
        "qualification_status": "INCOMPLETE",
        "thresholds_met": 28,
        "gate_count": 28,
        "gates": {
            "aggregate_quiet_outlier_relative_reduction": gate(
                0.1405378, 0.0504041, 0.6413486
            ),
            "olat_fixed_hotspot_subsurface_outlier_relative_reduction": gate(
                0.5267176, 0.2729008, 0.4818841
            ),
            "olat_fixed_hotspot_specular_outlier_relative_reduction": gate(
                0.4083969, 0.1641221, 0.5981308
            ),
            "aggregate_guide_textured_band3_8_amplitude_ratio": {
                "observed": 0.9828,
                "threshold": 0.90,
            },
        },
    }
    relighting = (
        "Acquisition aggregate: OLAT ΔPSNR +0.027 dB / ΔSSIM +0.0013 · "
        "HDRI ΔPSNR +0.016 dB / ΔSSIM +0.0007"
    )

    lines = compare_regularization._frequency_overview_headlines(
        report,
        relighting,
    )

    assert len(lines) == 7
    assert "PASS = cleanup under relighting safeguards" in lines[0]
    assert "not reconstruction improvement" in lines[0]
    assert "14.05% → 5.04% · 64.1% reduction" in lines[1]
    assert "52.67% → 27.29% · 48.2% reduction" in lines[2]
    assert "40.84% → 16.41% · 59.8% reduction" in lines[3]
    assert "3–8 px amplitude retained: 98.3%" in lines[4]
    assert lines[5] == relighting
    assert not any(
        phrase in line.lower()
        for line in lines
        for phrase in ("frequency updates", "frozen-target")
    )
    canvas = Image.new("RGB", (1800, 537), "black")
    draw = ImageDraw.Draw(canvas)
    font = compare_regularization._font(27, bold=True)
    for line in lines:
        bounds = draw.textbbox((42, 0), line, font=font)
        assert bounds[2] <= canvas.width - 42


def test_frequency_annotation_gutter_expands_beyond_all_count_text():
    profile_cleanup = {
        "maps": {
            map_name: {
                "updated_entries": 12345678901234567890,
                "consensus_entries": 98765432109876543210,
            }
            for map_name in compare_regularization.FREQUENCY_DETAIL_MAPS
        }
    }

    layout = compare_regularization._frequency_annotation_layout(
        profile_cleanup
    )

    assert layout["tile_left"] > compare_regularization.FREQUENCY_DETAIL_TILE_LEFT
    for row in layout["rows"]:
        left, _, right, _ = row["bounds_xyxy"]
        assert left >= layout["annotation_left"]
        assert right <= layout["tile_left"] - layout["annotation_gap"]


def test_overview_pixel_detail_cards_keep_labels_and_captions_separate():
    layout = compare_regularization._overview_pixel_detail_layout(
        ("subsurface", "specular"),
        (
            "updates 19130\nconsensus 7042",
            "updates 18313\nconsensus 6869",
        ),
        display_size=192,
        canvas_width=1800,
    )

    assert layout["display_size"] == 192
    first, second = layout["groups"]
    assert first["box"][0] + first["box"][2] < second["box"][0]
    for group in layout["groups"]:
        group_left, _, group_width, _ = group["box"]
        group_right = group_left + group_width
        baseline_bounds, regularized_bounds = group["variant_label_bounds"]
        assert baseline_bounds[2] < regularized_bounds[0]
        for bounds, slot in zip(
            group["variant_label_bounds"],
            group["variant_slots"],
        ):
            assert slot[0] <= bounds[0] < bounds[2] <= slot[0] + slot[2]
        caption_bounds = group["caption_bounds"]
        assert group_left <= caption_bounds[0]
        assert caption_bounds[2] <= group_right
    assert first["caption_bounds"][2] < second["caption_bounds"][0]


def test_adaptive_cleanup_crop_selector_uses_auditable_lexicographic_rule():
    removed = np.zeros((12, 12), dtype=bool)
    introduced = np.zeros_like(removed)
    quiet = np.zeros_like(removed)
    # Crop A has six removed and one introduced (net five). Crop B has five
    # removed and none introduced (also net five), so removed count selects A.
    for y, x in ((1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 2)):
        removed[y, x] = True
    introduced[2, 2] = True
    for y, x in ((7, 7), (7, 8), (7, 9), (8, 7), (8, 8)):
        removed[y, x] = True
    quiet[:] = True

    selected = compare_regularization._select_adaptive_cleanup_crop(
        [removed], [introduced], [quiet], crop_size=3
    )

    assert selected["crop_box_xyxy"] == [1, 1, 4, 4]
    assert selected["aggregate_net_removed"] == 5
    assert selected["aggregate_removed"] == 6
    assert selected["aggregate_introduced"] == 1
    assert selected["aggregate_quiet_pixels"] == 9
    assert selected["manual_selection"] is False
    assert selected["selection_rule"].startswith("maximize aggregate net removed")


def test_hdri_thumbnail_contain_fit_preserves_aspect_and_centers_bars():
    source = Image.new("RGB", (12, 4), (240, 20, 10))

    tile = compare_regularization._contain_image(
        source,
        (6, 8),
        fill=(3, 4, 5),
    )

    values = np.asarray(tile)
    assert tile.size == (6, 8)
    assert np.all(values[:3] == (3, 4, 5))
    assert np.all(values[3:5] == (240, 20, 10))
    assert np.all(values[5:] == (3, 4, 5))


@pytest.mark.parametrize(
    ("case_deltas", "expected_median", "expected_worst"),
    (
        ({"a": -2.0, "b": -1.0, "c": 0.0, "d": 1.0}, "b", "a"),
        (
            {"a": -2.0, "b": -1.0, "c": 0.0, "d": 1.0, "e": 2.0},
            "c",
            "a",
        ),
        ({"a": -2.0, "b": -1.0, "c": -1.0, "d": 1.0}, "b", "a"),
    ),
)
def test_case_png_selection_is_deterministic_lower_median(
    case_deltas,
    expected_median,
    expected_worst,
):
    rows = [
        {
            "lighting": "olat",
            "case_id": case_id,
            "profile": profile,
            "delta": {"psnr": delta + profile_index * 0.2},
        }
        for case_id, delta in reversed(tuple(case_deltas.items()))
        for profile_index, profile in enumerate(("olat", "mix"))
    ]

    selection = compare_regularization._select_case_png_examples(
        rows,
        lighting="olat",
        profiles=("olat", "mix"),
    )

    assert selection["median"]["case_id"] == expected_median
    assert selection["worst"]["case_id"] == expected_worst
    assert selection["median_policy"].startswith("lower_order_statistic")


def _write_png(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8)).save(path)


def _write_synthetic_camera(
    root,
    *,
    tv_weight,
    noisy,
    profiles=("olat",),
    shape=(32, 24),
):
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
                    "count": 2,
                    "metrics": {
                        "psnr": 20.0 + tv_weight,
                        "ssim_global": 0.8 + tv_weight,
                        "mae": 0.05,
                        "mean_intensity_ratio": 1.0,
                        "luminance_correlation": 0.9,
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
    height, width = shape
    base_color = np.full((height, width, 3), 80, dtype=np.uint8)
    base_color[:, width // 2 :] = 180
    normal = np.zeros((height, width, 3), dtype=np.uint8)
    normal[..., 0:2] = 128
    normal[..., 2] = 255
    for profile in profiles:
        profile_acquisition = copy.deepcopy(common)
        profile_acquisition["lighting_profile"] = profile
        profile_acquisition["checkpoint_signature"]["profile"] = profile
        profile_acquisition["evaluation"]["profile"] = profile
        acquisition = root / "material" / profile / "acquisition.json"
        acquisition.parent.mkdir(parents=True, exist_ok=True)
        acquisition.write_text(
            json.dumps(profile_acquisition),
            encoding="utf-8",
        )
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
    yy, xx = np.indices((height, width))
    panel = np.stack(
        (
            50 + (3 * xx + yy) % 150,
            60 + (xx + 2 * yy) % 140,
            70 + (2 * xx + 3 * yy) % 130,
        ),
        axis=-1,
    ).astype(np.uint8)
    for lighting in ("olat", "hdri"):
        for case_id in ("shared_case", "hard_case"):
            case = root / "evaluation" / lighting / "cases" / case_id
            _write_png(case / "reference.png", panel)
            _write_png(case / "lighting.png", panel)
            for profile in profiles:
                _write_png(case / "predictions" / f"{profile}.png", panel)
                error = compare_regularization._scalar_error_heatmap_u8(
                    panel.astype(np.float32) / 255.0,
                    panel.astype(np.float32) / 255.0,
                )
                _write_png(case / "errors" / f"{profile}.png", error)


@pytest.mark.parametrize(
    "binding",
    ("lighting_profile", "checkpoint_signature.profile", "evaluation.profile"),
)
def test_load_acquisitions_rejects_profile_binding_mismatch(tmp_path, binding):
    camera = tmp_path / "camera" / "object" / "cam07"
    _write_synthetic_camera(camera, tv_weight=0.0, noisy=False)
    path = camera / "material" / "olat" / "acquisition.json"
    acquisition = json.loads(path.read_text(encoding="utf-8"))
    if binding == "lighting_profile":
        acquisition["lighting_profile"] = "hdri"
    elif binding == "checkpoint_signature.profile":
        acquisition["checkpoint_signature"]["profile"] = "hdri"
    else:
        acquisition["evaluation"]["profile"] = "hdri"
    path.write_text(json.dumps(acquisition), encoding="utf-8")

    with pytest.raises(ValueError, match="profile binding"):
        compare_regularization._load_acquisitions(camera, ("olat",))


def _array_sha256(values):
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_proximal_impulse_pair(
    baseline,
    candidate,
    *,
    profile="olat",
    frequency_v2_adapter=False,
):
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
    if frequency_v2_adapter:
        baseline_signature.update(
            {
                "schema": "ictpolarreal.end2end-checkpoint.v13",
                "adapter": {
                    "schema": "ictpolarreal.profile-acquisition-adapter.v7",
                    "algorithm_version": "ictpolarreal-frequency-consensus-v2",
                    "lighting_profiles_sha256": "lighting",
                    "optimizer": "Adam with cosine decay",
                },
            }
        )
        baseline_acquisition["schema"] = "ictpolarreal.end2end-disney.v13"
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
            "schema": (
                "ictpolarreal.end2end-checkpoint.v13"
                if frequency_v2_adapter
                else "ictpolarreal.end2end-checkpoint.v12"
            ),
            "regularization": copy.deepcopy(impulse_signature_regularization),
        }
    )
    candidate_signature["adapter"].update(
        {
            "schema": (
                "ictpolarreal.profile-acquisition-adapter.v7"
                if frequency_v2_adapter
                else "ictpolarreal.profile-acquisition-adapter.v6"
            ),
            "algorithm_version": (
                "ictpolarreal-frequency-consensus-v2"
                if frequency_v2_adapter
                else "ictpolarreal-impulse-proximal-v5"
            ),
        }
    )
    candidate_acquisition["schema"] = (
        "ictpolarreal.end2end-disney.v13"
        if frequency_v2_adapter
        else "ictpolarreal.end2end-disney.v12"
    )
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


def _make_frequency_consensus_pair(
    baseline,
    candidate,
    *,
    profile="olat",
    fit_mask=None,
):
    torch = pytest.importorskip("torch")
    baseline_path = baseline / "material" / profile / "acquisition.json"
    candidate_path = candidate / "material" / profile / "acquisition.json"
    baseline_acquisition = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate_acquisition = json.loads(candidate_path.read_text(encoding="utf-8"))
    parameters = list(compare_regularization.SCALAR_MAPS)
    producer_maps = tuple(end2end_acquisition.DISNEY_TV_SCALAR_NAMES)

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

    source_values = {}
    for index, map_name in enumerate(producer_maps):
        baseline_map = (
            baseline / "material" / profile / "maps" / f"{map_name}.png"
        )
        with Image.open(baseline_map) as image:
            values_u8 = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        values_u8[180, 120] = np.uint8(220 - index)
        _write_png(baseline_map, values_u8)
        _write_png(
            candidate / "material" / profile / "maps" / f"{map_name}.png",
            values_u8,
        )
        source_values[map_name] = torch.as_tensor(
            values_u8.astype(np.float32) / 255.0
        )

    class ToyDisney(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for map_name, values in source_values.items():
                constrained = values.clamp(1e-5, 1.0 - 1e-5)
                setattr(
                    self,
                    f"{map_name}_un",
                    torch.nn.Parameter(torch.logit(constrained).unsqueeze(0)),
                )

        def _param_maps(self):
            return {
                map_name: torch.sigmoid(getattr(self, f"{map_name}_un"))[0]
                for map_name in producer_maps
            }

    model = ToyDisney()
    with Image.open(
        baseline / "material" / profile / "maps" / "baseColor.png"
    ) as image:
        albedo = torch.as_tensor(
            np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        )
    with Image.open(
        baseline / "material" / profile / "maps" / "normal.png"
    ) as image:
        normal = torch.as_tensor(
            np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0 * 2.0
            - 1.0
        )
    height, width = albedo.shape[:2]
    if fit_mask is None:
        fit_mask = np.ones((height, width), dtype=bool)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if fit_mask.shape != (height, width):
        raise ValueError("frequency test fit mask has the wrong shape")
    mask = torch.as_tensor(fit_mask.astype(np.float32)[..., None])
    stage_plan = end2end_acquisition._frequency_consensus_stage_plan(
        100,
        enabled=True,
        weight=0.00125,
    )
    bundle = end2end_acquisition._build_frequency_consensus_bundle(
        torch,
        model,
        mask,
        albedo,
        normal,
        created_after_step=100,
        strength=stage_plan["strength"],
    )
    diagnostic = end2end_acquisition._apply_frequency_consensus_update(
        torch,
        model,
        bundle,
    )
    for map_name, values in model._param_maps().items():
        _write_png(
            candidate / "material" / profile / "maps" / f"{map_name}.png",
            np.floor(
                np.clip(values.detach().numpy(), 0.0, 1.0) * 255.0 + 0.5
            ).astype(np.uint8),
        )
    provenance = end2end_acquisition._finalize_frequency_frozen_artifact(
        candidate / "material" / profile,
        bundle,
    )
    settings = end2end_acquisition._regularization_settings(
        "frequency-consensus"
    )
    signature_regularization = {
        "kind": "frequency-consensus",
        "weight": 0.00125,
        "parameters": parameters,
        "settings": settings,
        "stage_plan": stage_plan,
    }
    candidate_signature = copy.deepcopy(baseline_signature)
    candidate_signature.update(
        {
            "schema": "ictpolarreal.end2end-checkpoint.v13",
            "regularization": copy.deepcopy(signature_regularization),
        }
    )
    candidate_signature["adapter"].update(
        {
            "schema": "ictpolarreal.profile-acquisition-adapter.v7",
            "algorithm_version": "ictpolarreal-frequency-consensus-v1",
        }
    )
    candidate_acquisition["schema"] = "ictpolarreal.end2end-disney.v13"
    pre_cleanup = {
        "olat": [0.10, 0.11],
        "hdri": [0.20, 0.21],
    }
    post_cleanup = {
        "olat": [0.09, 0.10],
        "hdri": [0.19, 0.20],
    }
    evaluation_guard = end2end_acquisition._frequency_evaluation_guard(
        pre_cleanup,
        post_cleanup,
    )
    for lighting in compare_regularization.EVALUATION_LIGHTING:
        baseline_suite = baseline_acquisition["evaluation"]["evaluations"][
            lighting
        ]
        candidate_suite = candidate_acquisition["evaluation"]["evaluations"][
            lighting
        ]
        baseline_suite["count"] = len(pre_cleanup[lighting])
        candidate_suite["count"] = len(post_cleanup[lighting])
        baseline_suite["metrics"]["mse"] = evaluation_guard["suites"][
            lighting
        ]["pre_cleanup_mean_mse"]
        candidate_suite["metrics"]["mse"] = evaluation_guard["suites"][
            lighting
        ]["post_cleanup_mean_mse"]
    candidate_acquisition["regularization"] = {
        **signature_regularization,
        "weight_semantics": "normalized_post_fit_strength",
        "frozen_bundle": provenance,
        "cleanup_applied": True,
        "cleanup_diagnostic": diagnostic,
        "evaluation_guard": evaluation_guard,
        "data_objective_only": True,
    }
    candidate_acquisition["checkpoint_signature"] = candidate_signature
    baseline_path.write_text(json.dumps(baseline_acquisition), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate_acquisition), encoding="utf-8")
    return provenance["artifact"], bundle


def _make_active_frequency_adaptive_pair(
    baseline,
    candidate,
    *,
    profile="olat",
    fit_mask=None,
):
    torch = pytest.importorskip("torch")
    source = (
        baseline.parents[2]
        / "_adaptive_data_fit_source"
        / baseline.parent.name
        / baseline.name
    )
    if not source.exists():
        shutil.copytree(baseline, source)
    _make_frequency_consensus_pair(
        source,
        baseline,
        profile=profile,
        fit_mask=fit_mask,
    )

    baseline_path = baseline / "material" / profile / "acquisition.json"
    candidate_path = candidate / "material" / profile / "acquisition.json"
    baseline_acquisition = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate_acquisition = copy.deepcopy(baseline_acquisition)
    producer_maps = tuple(end2end_acquisition.DISNEY_TV_SCALAR_NAMES)
    source_values = {}
    for map_name in producer_maps:
        with Image.open(
            source / "material" / profile / "maps" / f"{map_name}.png"
        ) as image:
            values = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        source_values[map_name] = torch.as_tensor(values)

    class ToyDisney(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for map_name, values in source_values.items():
                constrained = values.clamp(1e-5, 1.0 - 1e-5)
                setattr(
                    self,
                    f"{map_name}_un",
                    torch.nn.Parameter(torch.logit(constrained).unsqueeze(0)),
                )

        def _param_maps(self):
            return {
                map_name: torch.sigmoid(getattr(self, f"{map_name}_un"))[0]
                for map_name in producer_maps
            }

    model = ToyDisney()
    with Image.open(
        source / "material" / profile / "maps" / "baseColor.png"
    ) as image:
        albedo = torch.as_tensor(
            np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        )
    with Image.open(
        source / "material" / profile / "maps" / "normal.png"
    ) as image:
        normal = torch.as_tensor(
            np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0 * 2.0
            - 1.0
        )
    height, width = albedo.shape[:2]
    if fit_mask is None:
        fit_mask = np.ones((height, width), dtype=bool)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    mask = torch.as_tensor(fit_mask.astype(np.float32)[..., None])
    stage_plan = end2end_acquisition._frequency_consensus_stage_plan(
        100,
        enabled=True,
        weight=0.00125,
    )
    bundle = end2end_acquisition._build_frequency_consensus_adaptive_bundle(
        torch,
        model,
        mask,
        albedo,
        normal,
        created_after_step=100,
        strength=stage_plan["strength"],
    )
    diagnostic = end2end_acquisition._apply_frequency_consensus_update(
        torch,
        model,
        bundle,
    )
    for map_name, values in model._param_maps().items():
        _write_png(
            candidate / "material" / profile / "maps" / f"{map_name}.png",
            np.floor(
                np.clip(values.detach().numpy(), 0.0, 1.0) * 255.0 + 0.5
            ).astype(np.uint8),
        )
    provenance = end2end_acquisition._finalize_frequency_frozen_artifact(
        candidate / "material" / profile,
        bundle,
    )
    signature_regularization = {
        "kind": "frequency-consensus-adaptive",
        "weight": 0.00125,
        "parameters": list(compare_regularization.SCALAR_MAPS),
        "settings": end2end_acquisition._regularization_settings(
            "frequency-consensus-adaptive"
        ),
        "stage_plan": stage_plan,
    }
    candidate_signature = copy.deepcopy(
        baseline_acquisition["checkpoint_signature"]
    )
    candidate_signature["regularization"] = copy.deepcopy(
        signature_regularization
    )
    candidate_signature["adapter"]["algorithm_version"] = (
        "ictpolarreal-frequency-consensus-adaptive-v1"
    )
    pre_cleanup = {"olat": [0.10, 0.11], "hdri": [0.20, 0.21]}
    post_cleanup = {"olat": [0.09, 0.10], "hdri": [0.19, 0.20]}
    evaluation_guard = end2end_acquisition._frequency_evaluation_guard(
        pre_cleanup,
        post_cleanup,
    )
    candidate_acquisition["regularization"] = {
        **signature_regularization,
        "weight_semantics": "normalized_post_fit_strength",
        "frozen_bundle": provenance,
        "cleanup_applied": True,
        "cleanup_diagnostic": diagnostic,
        "evaluation_guard": evaluation_guard,
        "data_objective_only": True,
    }
    candidate_acquisition["checkpoint_signature"] = candidate_signature
    for lighting in compare_regularization.EVALUATION_LIGHTING:
        suite = candidate_acquisition["evaluation"]["evaluations"][lighting]
        suite["count"] = len(post_cleanup[lighting])
        suite["metrics"]["mse"] = evaluation_guard["suites"][lighting][
            "post_cleanup_mean_mse"
        ]
    candidate_path.write_text(json.dumps(candidate_acquisition), encoding="utf-8")
    return provenance["artifact"], bundle


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


def test_frequency_transition_label_settings_and_baseline_state_are_versioned():
    parameters = list(compare_regularization.SCALAR_MAPS)
    baseline_regularization = {
        "kind": "masked_l1_total_variation",
        "weight": 0.0,
        "parameters": parameters,
    }
    baseline = {
        "base_color_source": "dataset_albedo",
        "regularization": baseline_regularization,
        "checkpoint_signature": {
            "schema": "ictpolarreal.end2end-checkpoint.v8",
            "normalized_targets_sha256": "targets",
            "regularization": copy.deepcopy(baseline_regularization),
            "adapter": {
                "schema": "ictpolarreal.profile-acquisition-adapter.v2",
                "algorithm_version": "ictpolarreal-masked-tv-v1",
            },
        },
    }
    stage_plan = end2end_acquisition._frequency_consensus_stage_plan(
        100,
        enabled=True,
        weight=0.00125,
    )
    regularization = {
        "kind": "frequency-consensus",
        "weight": 0.00125,
        "parameters": parameters,
        "settings": end2end_acquisition._regularization_settings(
            "frequency-consensus"
        ),
        "stage_plan": stage_plan,
    }
    candidate = copy.deepcopy(baseline)
    candidate["regularization"] = regularization
    candidate["checkpoint_signature"]["schema"] = (
        "ictpolarreal.end2end-checkpoint.v13"
    )
    candidate["checkpoint_signature"]["regularization"] = copy.deepcopy(
        regularization
    )
    candidate["checkpoint_signature"]["adapter"] = {
        "schema": "ictpolarreal.profile-acquisition-adapter.v7",
        "algorithm_version": "ictpolarreal-frequency-consensus-v1",
    }

    contract = compare_regularization._validate_comparison_contract(
        {
            "baseline": {"olat": baseline},
            "regularized": {"olat": candidate},
        },
        ("olat",),
    )

    assert contract["signature_compatibility"] == ["zero-weight-v8-to-v13"]
    assert contract["regularization_kind"] == "frequency-consensus"
    assert (
        compare_regularization._display_regularizer("frequency-consensus")
        == "post-fit frequency consensus"
    )
    compare_regularization._validate_frequency_baseline_state(
        baseline_regularization
    )
    compare_regularization._validate_frequency_settings(
        regularization["settings"]
    )
    bad_settings = copy.deepcopy(regularization["settings"])
    bad_settings["own_deviation_threshold"] = 0.1
    with pytest.raises(ValueError, match="settings differ"):
        compare_regularization._validate_frequency_settings(bad_settings)
    with pytest.raises(ValueError, match="stage must be disabled"):
        compare_regularization._validate_frequency_baseline_state(
            {
                **baseline_regularization,
                "stage_plan": {"enabled": True},
            }
        )
    with pytest.raises(ValueError, match="must not apply"):
        compare_regularization._validate_frequency_baseline_state(
            {**baseline_regularization, "cleanup_applied": True}
        )


def test_active_frequency_to_adaptive_contract_requires_equal_fit_and_weight():
    parameters = list(compare_regularization.SCALAR_MAPS)
    stage_plan = end2end_acquisition._frequency_consensus_stage_plan(
        100,
        enabled=True,
        weight=0.00125,
    )

    def acquisition(kind, algorithm_version):
        regularization = {
            "kind": kind,
            "weight": 0.00125,
            "parameters": parameters,
            "settings": end2end_acquisition._regularization_settings(kind),
            "stage_plan": copy.deepcopy(stage_plan),
        }
        return {
            "base_color_source": "dataset_albedo",
            "regularization": regularization,
            "checkpoint_signature": {
                "schema": "ictpolarreal.end2end-checkpoint.v13",
                "profile": "olat",
                "normalized_targets_sha256": "same-data-fit",
                "regularization": copy.deepcopy(regularization),
                "adapter": {
                    "schema": "ictpolarreal.profile-acquisition-adapter.v7",
                    "algorithm_version": algorithm_version,
                    "optimizer": "Adam with cosine decay",
                },
            },
        }

    baseline = acquisition(
        "frequency-consensus",
        "ictpolarreal-frequency-consensus-v1",
    )
    adaptive = acquisition(
        "frequency-consensus-adaptive",
        "ictpolarreal-frequency-consensus-adaptive-v1",
    )
    contract = compare_regularization._validate_comparison_contract(
        {
            "baseline": {"olat": baseline},
            "regularized": {"olat": adaptive},
        },
        ("olat",),
    )

    assert contract["comparison_mode"] == (
        "frequency-consensus-v1-to-adaptive-v1"
    )
    assert contract["signature_compatibility"] == [
        "active-frequency-v1-to-adaptive-v1"
    ]
    assert contract["baseline_tv_weight"] == contract["regularized_tv_weight"]
    assert contract["active_frequency_upgrade"] is True
    assert compare_regularization._display_regularizer(
        "frequency-consensus-adaptive"
    ) == "adaptive frequency consensus"

    baseline_v2 = acquisition(
        "frequency-consensus",
        "ictpolarreal-frequency-consensus-v2",
    )
    adaptive_v2 = acquisition(
        "frequency-consensus-adaptive",
        "ictpolarreal-frequency-consensus-adaptive-v2",
    )
    contract_v2 = compare_regularization._validate_comparison_contract(
        {
            "baseline": {"olat": baseline_v2},
            "regularized": {"olat": adaptive_v2},
        },
        ("olat",),
    )
    assert contract_v2["comparison_mode"] == (
        "frequency-consensus-v2-to-adaptive-v2"
    )
    assert contract_v2["signature_compatibility"] == [
        "active-frequency-v2-to-adaptive-v2"
    ]

    with pytest.raises(ValueError, match="unsupported schema/algorithm transition"):
        compare_regularization._validate_comparison_contract(
            {
                "baseline": {"olat": baseline_v2},
                "regularized": {"olat": adaptive},
            },
            ("olat",),
        )

    unequal_weight = copy.deepcopy(adaptive)
    unequal_weight["regularization"]["weight"] = 0.001
    unequal_weight["checkpoint_signature"]["regularization"]["weight"] = 0.001
    with pytest.raises(ValueError, match="requires equal weights"):
        compare_regularization._validate_comparison_contract(
            {
                "baseline": {"olat": baseline},
                "regularized": {"olat": unequal_weight},
            },
            ("olat",),
        )

    changed_fit = copy.deepcopy(adaptive)
    changed_fit["checkpoint_signature"]["normalized_targets_sha256"] = "changed"
    with pytest.raises(ValueError, match="checkpoint signatures"):
        compare_regularization._validate_comparison_contract(
            {
                "baseline": {"olat": baseline},
                "regularized": {"olat": changed_fit},
            },
            ("olat",),
        )


def test_data_root_fit_mask_reconstructs_recorded_legacy_or_clean_rule(
    tmp_path,
    monkeypatch,
):
    class Sample:
        def __init__(self, object_name, camera_name, camera_dir):
            self.camera_dir = camera_dir

        def image_path(self, kind):
            return self.camera_dir / f"{kind}.png"

    normal = np.asarray(
        [[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]],
        dtype=np.float32,
    )
    capture = np.ones((1, 2, 1), dtype=np.float32)
    view = np.zeros_like(normal)
    view[..., 2] = 1.0
    monkeypatch.setattr(compare_regularization, "CameraSample", Sample)
    monkeypatch.setattr(
        compare_regularization,
        "read_image",
        lambda path, channels=None: capture if channels == 1 else normal,
    )
    monkeypatch.setattr(
        compare_regularization,
        "load_end2end_view_directions",
        lambda data_root, sample, expected_shape: view,
    )

    clean = compare_regularization._fit_mask_from_data_root(
        tmp_path,
        "object",
        "cam07",
        (2, 1),
        fit_mask_rule=compare_regularization._CLEAN_CAPTURE_FIT_MASK_RULE,
    )
    legacy = compare_regularization._fit_mask_from_data_root(
        tmp_path,
        "object",
        "cam07",
        (2, 1),
        fit_mask_rule=compare_regularization._LEGACY_FRONT_FACING_FIT_MASK_RULE,
    )

    np.testing.assert_array_equal(clean, [[True, True]])
    np.testing.assert_array_equal(legacy, [[True, False]])


def _frequency_guard_acquisition(pre_cleanup, post_cleanup):
    guard = end2end_acquisition._frequency_evaluation_guard(
        pre_cleanup,
        post_cleanup,
    )
    return {
        "regularization": {"evaluation_guard": guard},
        "evaluation": {
            "evaluations": {
                lighting: {
                    "count": len(post_cleanup[lighting]),
                    "metrics": {
                        "mse": guard["suites"][lighting][
                            "post_cleanup_mean_mse"
                        ]
                    },
                }
                for lighting in compare_regularization.EVALUATION_LIGHTING
            }
        },
    }


@pytest.mark.parametrize(
    "post_cleanup",
    (
        {"olat": [0.09, 0.10], "hdri": [0.19, 0.20]},
        {"olat": [0.10, 0.11], "hdri": [0.20, 0.21]},
        {
            "olat": [
                0.10
                + 0.5
                * compare_regularization.FREQUENCY_EVALUATION_MEAN_MSE_TOLERANCE,
                0.11
                + 0.5
                * compare_regularization.FREQUENCY_EVALUATION_MEAN_MSE_TOLERANCE,
            ],
            "hdri": [0.20, 0.21],
        },
    ),
)
def test_frequency_evaluation_guard_accepts_improved_equal_and_sub_tolerance(
    post_cleanup,
):
    pre_cleanup = {"olat": [0.10, 0.11], "hdri": [0.20, 0.21]}
    acquisition = _frequency_guard_acquisition(pre_cleanup, post_cleanup)

    validated = compare_regularization._validate_frequency_evaluation_guard(
        acquisition,
        profile="olat",
    )

    assert (
        compare_regularization.FREQUENCY_EVALUATION_MEAN_MSE_TOLERANCE
        == end2end_acquisition.FREQUENCY_CONSENSUS_EVALUATION_MEAN_MSE_TOLERANCE
    )
    assert validated["validated"] is True
    assert validated["same_checkpoint"] is True
    assert all(
        suite["within_no_regression_tolerance"] is True
        for suite in validated["suites"].values()
    )


def test_frequency_evaluation_guard_rejects_missing_or_corrupt_provenance():
    pre_cleanup = {"olat": [0.10, 0.11], "hdri": [0.20, 0.21]}
    post_cleanup = {"olat": [0.09, 0.10], "hdri": [0.19, 0.20]}
    valid = _frequency_guard_acquisition(pre_cleanup, post_cleanup)

    missing = copy.deepcopy(valid)
    missing["regularization"].pop("evaluation_guard")
    with pytest.raises(ValueError, match="missing or invalid"):
        compare_regularization._validate_frequency_evaluation_guard(
            missing,
            profile="olat",
        )

    bad_arithmetic = copy.deepcopy(valid)
    bad_arithmetic["regularization"]["evaluation_guard"]["suites"]["olat"][
        "post_minus_pre_mse"
    ][0] += 0.01
    with pytest.raises(ValueError, match="per-case arithmetic"):
        compare_regularization._validate_frequency_evaluation_guard(
            bad_arithmetic,
            profile="olat",
        )

    bad_decision = copy.deepcopy(valid)
    bad_decision["regularization"]["evaluation_guard"]["suites"]["olat"][
        "worsened"
    ] = True
    with pytest.raises(ValueError, match="mean or decision"):
        compare_regularization._validate_frequency_evaluation_guard(
            bad_decision,
            profile="olat",
        )

    negative = copy.deepcopy(valid)
    negative["regularization"]["evaluation_guard"]["suites"]["olat"][
        "pre_cleanup_mse"
    ][0] = -0.1
    with pytest.raises(ValueError, match="per-case losses"):
        compare_regularization._validate_frequency_evaluation_guard(
            negative,
            profile="olat",
        )

    nonfinite = copy.deepcopy(valid)
    nonfinite["regularization"]["evaluation_guard"]["suites"]["olat"][
        "post_cleanup_mse"
    ][0] = float("nan")
    with pytest.raises(ValueError, match="per-case losses"):
        compare_regularization._validate_frequency_evaluation_guard(
            nonfinite,
            profile="olat",
        )

    list_count_mismatch = copy.deepcopy(valid)
    list_count_mismatch["regularization"]["evaluation_guard"]["suites"][
        "olat"
    ]["post_cleanup_mse"].pop()
    with pytest.raises(ValueError, match="per-case losses"):
        compare_regularization._validate_frequency_evaluation_guard(
            list_count_mismatch,
            profile="olat",
        )

    final_count_mismatch = copy.deepcopy(valid)
    final_count_mismatch["evaluation"]["evaluations"]["olat"]["count"] += 1
    with pytest.raises(ValueError, match="final evaluation count/MSE"):
        compare_regularization._validate_frequency_evaluation_guard(
            final_count_mismatch,
            profile="olat",
        )


def test_frequency_evaluation_guard_rejects_regression_beyond_tolerance():
    tolerance = compare_regularization.FREQUENCY_EVALUATION_MEAN_MSE_TOLERANCE
    pre_cleanup = {"olat": [0.10, 0.11], "hdri": [0.20, 0.21]}
    post_cleanup = {
        "olat": [0.10 + 2.0 * tolerance, 0.11 + 2.0 * tolerance],
        "hdri": [0.20, 0.21],
    }
    regressed = _frequency_guard_acquisition(pre_cleanup, post_cleanup)

    with pytest.raises(ValueError, match="regressed beyond tolerance"):
        compare_regularization._validate_frequency_evaluation_guard(
            regressed,
            profile="olat",
        )


def test_frequency_gates_reject_equal_bad_relighting_and_weak_edge_row():
    map_metrics = {"olat": {}}
    for map_name in compare_regularization.SCALAR_MAPS:
        map_metrics["olat"][map_name] = {
            "baseline": {
                "guide_textured_band3_8_energy": 1.0,
                "guide_edge_gradient_magnitude": 1.0,
                "quiet_region_pixels": 100,
                "quiet_region_median5_residual_outlier_count_gt_0p05": 10,
                "fixed_hotspot_median5_residual_outlier_fraction_gt_0p05": 0.5,
            },
            "regularized": {
                "guide_textured_band3_8_energy": 1.0,
                "guide_edge_gradient_magnitude": 1.0,
                "quiet_region_pixels": 100,
                "quiet_region_median5_residual_outlier_count_gt_0p05": 0,
                "fixed_hotspot_median5_residual_outlier_fraction_gt_0p05": 0.0,
            },
            "comparison": {
                "guide_edge_gradient_magnitude_ratio": 1.0,
                "guide_edge_signed_gradient_cosine": 1.0,
                "meaningful_guide_edge_gradient": False,
                "gaussian3_low_frequency_mae": 0.0,
            },
        }
    map_metrics["olat"]["clearcoat"]["baseline"][
        "guide_edge_gradient_magnitude"
    ] = 0.0
    map_metrics["olat"]["clearcoat"]["regularized"][
        "guide_edge_gradient_magnitude"
    ] = 0.001
    map_metrics["olat"]["clearcoat"]["comparison"].update(
        {
            "guide_edge_gradient_magnitude_ratio": None,
            "guide_edge_signed_gradient_cosine": None,
        }
    )
    bad = {
        "psnr": 10.0,
        "ssim_global": 0.3,
        "mean_intensity_ratio": 0.5,
        "luminance_correlation": 0.2,
    }
    evaluation_metrics = {
        "olat": {
            lighting: {
                "baseline": dict(bad),
                "regularized": dict(bad),
            }
            for lighting in compare_regularization.EVALUATION_LIGHTING
        }
    }
    case_png_metrics = {
        "schema": "ictpolarreal.case-png-evaluation.v2",
        "profiles": ["olat"],
        "validated_prediction_files": 2,
        "validated_error_files": 2,
        "selected_cases": {"olat": {"median": {}, "worst": {}}},
        "rows": [
            {
                "baseline": dict(bad),
                "regularized": dict(bad),
                "delta": {name: 0.0 for name in bad},
            }
        ],
    }

    gates = compare_regularization._build_frequency_diagnostic_gates(
        map_metrics,
        evaluation_metrics,
        case_png_metrics,
        ("olat",),
    )

    assert gates["gates"][
        "aggregate_relighting_delta_psnr_db"
    ]["meets_threshold"] is True
    assert gates["gates"][
        "aggregate_quiet_outlier_relative_reduction"
    ]["meets_threshold"] is True
    assert gates["gates"][
        "candidate_absolute_psnr_db"
    ]["meets_threshold"] is False
    assert gates["gates"][
        "candidate_absolute_mean_intensity_ratio"
    ]["meets_threshold"] is False
    assert gates["gates"][
        "guide_edge_gradient_magnitude_ratio"
    ]["meets_threshold"] is False
    assert gates["gates"]["guide_edge_gradient_magnitude_ratio"][
        "zero_baseline_nonzero_candidate_rows"
    ] == 1
    assert gates["all_thresholds_met"] is False

    map_metrics["olat"]["clearcoat"]["regularized"][
        "guide_edge_gradient_magnitude"
    ] = 0.0
    both_zero = compare_regularization._build_frequency_diagnostic_gates(
        map_metrics,
        evaluation_metrics,
        case_png_metrics,
        ("olat",),
    )
    edge_gate = both_zero["gates"]["guide_edge_gradient_magnitude_ratio"]
    cosine_gate = both_zero["gates"]["guide_edge_signed_gradient_cosine"]
    assert edge_gate["zero_baseline_nonzero_candidate_rows"] == 0
    assert edge_gate["meets_threshold"] is True
    assert cosine_gate["meets_threshold"] is True

    for map_name in compare_regularization.SCALAR_MAPS:
        map_metrics["olat"][map_name]["regularized"][
            "quiet_region_median5_residual_outlier_count_gt_0p05"
        ] = 9
    for map_name in ("subsurface", "specular"):
        map_metrics["olat"][map_name]["baseline"][
            "fixed_hotspot_median5_residual_outlier_fraction_gt_0p05"
        ] = 0.20
        map_metrics["olat"][map_name]["regularized"][
            "fixed_hotspot_median5_residual_outlier_fraction_gt_0p05"
        ] = 0.17
    for lighting in compare_regularization.EVALUATION_LIGHTING:
        evaluation_metrics["olat"][lighting]["regularized"]["psnr"] = 9.9
        evaluation_metrics["olat"][lighting]["regularized"]["ssim_global"] = 0.29
    insufficient = compare_regularization._build_frequency_diagnostic_gates(
        map_metrics,
        evaluation_metrics,
        case_png_metrics,
        ("olat",),
    )
    for gate_name in (
        "aggregate_quiet_outlier_relative_reduction",
        "olat_fixed_hotspot_subsurface_outlier_relative_reduction",
        "olat_fixed_hotspot_specular_outlier_relative_reduction",
        "aggregate_relighting_delta_psnr_db",
        "aggregate_relighting_delta_ssim_global",
    ):
        assert insufficient["gates"][gate_name]["meets_threshold"] is False


def test_evaluation_metrics_reject_nonfinite_absolute_sanity_input():
    metrics = {
        "psnr": float("nan"),
        "ssim_global": 0.8,
        "mean_intensity_ratio": 1.0,
        "luminance_correlation": 0.9,
    }
    acquisition = {
        "evaluation": {
            "evaluations": {
                lighting: {"metrics": dict(metrics)}
                for lighting in compare_regularization.EVALUATION_LIGHTING
            }
        }
    }
    with pytest.raises(ValueError, match="finite and complete"):
        compare_regularization._collect_evaluation_metrics(
            {
                "baseline": {"olat": acquisition},
                "regularized": {"olat": acquisition},
            },
            ("olat",),
        )


def test_case_png_metrics_accepts_variant_specific_representatives(tmp_path):
    baseline = tmp_path / "baseline" / "object" / "cam07"
    candidate = tmp_path / "candidate" / "object" / "cam07"
    for camera in (baseline, candidate):
        _write_synthetic_camera(camera, tv_weight=0.0, noisy=False)

    acquisitions = {"baseline": {}, "regularized": {}}
    for variant, camera, representative in (
        ("baseline", baseline, "shared_case"),
        ("regularized", candidate, "hard_case"),
    ):
        summary_path = camera / "evaluation" / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["suites"]["hdri"]["representative_case"] = representative
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        acquisition_path = camera / "material" / "olat" / "acquisition.json"
        acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
        acquisition["evaluation"]["evaluations"]["hdri"]["representative"] = {
            "condition_id": representative,
        }
        acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
        acquisitions[variant]["olat"] = acquisition

    result = compare_regularization._collect_case_png_metrics(
        baseline,
        candidate,
        acquisitions,
        ("olat",),
        np.ones((32, 24), dtype=bool),
    )

    assert result["selected_cases"]["hdri"]["median"]["case_id"] == "hard_case"
    assert result["selected_cases"]["hdri"]["worst"]["case_id"] == "hard_case"
    provenance = result["acquisition_provenance"]["hdri"]
    assert provenance["baseline"]["olat"]["recorded_representative"] == (
        "shared_case"
    )
    assert provenance["regularized"]["olat"]["recorded_representative"] == (
        "hard_case"
    )
    assert provenance["regularized"]["olat"][
        "camera_summary_representative"
    ] == "hard_case"


def test_case_png_metrics_rejects_stale_error_in_any_case_and_renderer_recomputes(
    tmp_path,
):
    baseline = tmp_path / "baseline" / "object" / "cam07"
    candidate = tmp_path / "candidate" / "object" / "cam07"
    for camera in (baseline, candidate):
        _write_synthetic_camera(camera, tv_weight=0.0, noisy=False)
    acquisitions = {
        variant: compare_regularization._load_acquisitions(camera, ("olat",))
        for variant, camera in (("baseline", baseline), ("regularized", candidate))
    }
    mask = np.ones((32, 24), dtype=bool)
    valid = compare_regularization._collect_case_png_metrics(
        baseline,
        candidate,
        acquisitions,
        ("olat",),
        mask,
    )
    assert valid["validated_error_files"] == 8
    assert (
        valid["error_heatmap_validation"]["maximum_error"]
        == end2end_acquisition.ERROR_HEATMAP_MAX
    )

    stale_path = (
        candidate
        / "evaluation"
        / "olat"
        / "cases"
        / "shared_case"
        / "errors"
        / "olat.png"
    )
    with Image.open(stale_path) as image:
        stale = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    stale[0, 0, 0] ^= np.uint8(1)
    _write_png(stale_path, stale)
    with pytest.raises(ValueError, match="stale or differs"):
        compare_regularization._collect_case_png_metrics(
            baseline,
            candidate,
            acquisitions,
            ("olat",),
            mask,
        )

    _write_png(
        stale_path,
        compare_regularization._scalar_error_heatmap_u8(
            compare_regularization._read_rgb_map(
                candidate
                / "evaluation"
                / "olat"
                / "cases"
                / "shared_case"
                / "predictions"
                / "olat.png"
            ),
            compare_regularization._read_rgb_map(
                baseline
                / "evaluation"
                / "olat"
                / "cases"
                / "shared_case"
                / "reference.png"
            ),
        ),
    )
    valid = compare_regularization._collect_case_png_metrics(
        baseline,
        candidate,
        acquisitions,
        ("olat",),
        mask,
    )
    for camera in (baseline, candidate):
        shutil.rmtree(camera / "evaluation" / "olat" / "cases" / "hard_case" / "errors")
    output = tmp_path / "evaluation.png"
    compare_regularization._write_evaluation_comparison(
        baseline,
        candidate,
        ("olat",),
        "olat",
        compare_regularization._collect_evaluation_metrics(
            acquisitions,
            ("olat",),
        ),
        valid,
        output,
    )
    assert output.is_file()


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
    assert summary["schema"] == "ictpolarreal.regularization-comparison.v4"
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


def test_impulse_cleanup_accepts_current_v13_frequency_v2_adapter(tmp_path):
    baseline = tmp_path / "baseline" / "object" / "cam07"
    candidate = tmp_path / "candidate" / "object" / "cam07"
    _write_synthetic_camera(baseline, tv_weight=0.0, noisy=False)
    _write_synthetic_camera(candidate, tv_weight=0.0, noisy=False)
    _make_proximal_impulse_pair(
        baseline,
        candidate,
        frequency_v2_adapter=True,
    )
    acquisitions = {
        variant: compare_regularization._load_acquisitions(camera, ("olat",))
        for variant, camera in (
            ("baseline", baseline),
            ("regularized", candidate),
        )
    }
    contract = compare_regularization._validate_comparison_contract(
        acquisitions,
        ("olat",),
    )

    cleanup = compare_regularization._measure_impulse_cleanup(
        baseline,
        candidate,
        ("olat",),
        np.ones((32, 24), dtype=bool),
        acquisitions,
        contract,
    )

    assert contract["signature_compatibility"] == ["exact-schema"]
    assert cleanup is not None
    assert cleanup["available"] is True
    assert cleanup["profiles"]["olat"]["artifact"]["validated"] is True


def test_compose_frequency_report_validates_writer_artifact_gates_and_native_panels(
    tmp_path,
):
    baseline = tmp_path / "baseline" / "object" / "cam07"
    candidate = tmp_path / "candidate" / "object" / "cam07"
    shape = (260, 200)
    _write_synthetic_camera(
        baseline,
        tv_weight=0.0,
        noisy=False,
        profiles=compare_regularization.QUALIFICATION_PROFILES,
        shape=shape,
    )
    _write_synthetic_camera(
        candidate,
        tv_weight=0.0,
        noisy=False,
        profiles=compare_regularization.QUALIFICATION_PROFILES,
        shape=shape,
    )
    fit_mask = np.ones(shape, dtype=bool)
    fit_mask[0, 0] = False
    fit_mask[160, 96] = False
    artifact_provenance = None
    bundle = None
    for profile in compare_regularization.QUALIFICATION_PROFILES:
        profile_artifact, profile_bundle = _make_frequency_consensus_pair(
            baseline,
            candidate,
            profile=profile,
            fit_mask=fit_mask,
        )
        if profile == "olat":
            artifact_provenance = profile_artifact
            bundle = profile_bundle
    mask_path = tmp_path / "mask.png"
    _write_png(mask_path, fit_mask.astype(np.uint8) * 255)

    summary = compare_regularization.compose_regularization_comparison(
        baseline,
        candidate,
        tmp_path / "comparison",
        mask_path=mask_path,
    )

    assert summary["comparison_contract"]["signature_compatibility"] == [
        "zero-weight-v8-to-v13"
    ]
    cleanup = summary["frequency_cleanup"]
    assert cleanup["available"] is True
    assert "adaptive_cleanup_evidence" not in cleanup
    assert cleanup["aggregate"]["updated"]["map_entries"] > 0
    assert cleanup["profiles"]["olat"]["artifact"]["validated"] is True
    assert cleanup["profiles"]["olat"]["evaluation_guard"]["validated"] is True
    assert cleanup["presentation"] == {
        "profile": "olat",
        "crop_box_xyxy": [96, 160, 192, 256],
        "crop_size_pixels": 96,
        "source_pixels_per_output_pixel": 1,
        "resampling": "none",
        "maps": list(compare_regularization.FREQUENCY_DETAIL_MAPS),
    }
    gates = summary["scalar_map_cleanup_qualification"]
    assert gates["gate_count"] == 28
    assert gates["material_gate_count"] == 10
    assert gates["map_improvement_gate_count"] == 3
    assert gates["material_safeguard_gate_count"] == 7
    assert gates["relighting_change_gate_count"] == 2
    assert gates["absolute_sanity_gate_count"] == 4
    assert gates["case_png_gate_count"] == 12
    assert gates["all_thresholds_met"] is True
    assert gates["qualification_status"] == "PASS"
    assert gates["qualification_coverage"]["complete"] is True
    assert gates["qualification_coverage"]["validated_case_rows"] == 12
    assert gates["qualification_coverage"]["validated_prediction_files"] == 24
    assert gates["qualification_coverage"]["validated_error_files"] == 24
    assert gates["gates"][
        "aggregate_quiet_outlier_relative_reduction"
    ]["threshold"] == pytest.approx(0.20)
    assert gates["gates"][
        "candidate_aggregate_quiet_outlier_fraction"
    ]["threshold"] == pytest.approx(0.092)
    assert gates["gates"][
        "candidate_olat_fixed_hotspot_subsurface_outlier_fraction"
    ]["threshold"] == pytest.approx(0.30)
    assert gates["gates"]["aggregate_relighting_delta_psnr_db"][
        "threshold"
    ] == pytest.approx(0.0)
    assert "not evidence of reconstruction" in gates["pass_meaning"]
    assert "does not establish reconstruction improvement" in gates[
        "interpretation"
    ]
    assert summary["case_png_evaluation"]["schema"] == (
        "ictpolarreal.case-png-evaluation.v2"
    )
    for lighting in compare_regularization.EVALUATION_LIGHTING:
        displayed = summary["displayed_evaluation_cases"][lighting]
        assert displayed["overview"]["role"] == "median_psnr_delta"
        assert set(displayed["detail"]) == {
            "median_psnr_delta",
            "worst_psnr_delta",
        }
        artifact = summary["artifacts"]["evaluation"][lighting]
        assert artifact["median_case"] == displayed["overview"]["case_id"]
        assert artifact["error_panels"].startswith("recomputed_after_pixel_exact")

    report = tmp_path / "comparison"
    hotspot = report / "material" / "frequency_hotspot_1to1.png"
    fullmaps = report / "material" / "frequency_fullmaps_1to1.png"
    assert hotspot.is_file()
    assert fullmaps.is_file()
    assert not (report / "material" / "adaptive_cleanup_evidence.png").exists()
    with Image.open(report / "evaluation" / "olat.png") as sheet:
        assert sheet.size == (1360, 1976)
    with Image.open(report / "evaluation" / "hdri.png") as sheet:
        assert sheet.size == (1560, 1976)
    with Image.open(
        baseline / "material" / "olat" / "maps" / "subsurface.png"
    ) as source:
        expected_pixel = source.convert("RGB").getpixel((96, 160))
        expected_adjacent_pixel = source.convert("RGB").getpixel((97, 160))
    annotation_layout = compare_regularization._frequency_annotation_layout(
        cleanup["profiles"]["olat"]
    )
    audit_tile_left = annotation_layout["tile_left"]
    with Image.open(hotspot) as sheet:
        assert sheet.getpixel(
            (
                audit_tile_left,
                compare_regularization.FREQUENCY_DETAIL_TILE_TOP,
            )
        ) == expected_pixel
        assert sheet.getpixel(
            (
                audit_tile_left + 2 * 96,
                compare_regularization.FREQUENCY_DETAIL_TILE_TOP,
            )
        ) == (0, 0, 0)
        assert sheet.width == audit_tile_left + 4 * 96 + 24
    with Image.open(fullmaps) as sheet:
        assert sheet.getpixel((audit_tile_left, 130)) == expected_pixel
        target_left = audit_tile_left + 2 * shape[1]
        assert sheet.getpixel((target_left, 130)) == (0, 0, 0)
        assert sheet.getpixel((target_left + 96, 130 + 160)) == (0, 0, 0)
        assert sheet.width == audit_tile_left + 4 * shape[1] + 24
    assert float(bundle["maps"]["subsurface"]["target"][160, 96]) > 0.0
    overview_captions = tuple(
        (
            f"updates {cleanup['profiles']['olat']['maps'][map_name]['updated_entries']}\n"
            "consensus "
            f"{cleanup['profiles']['olat']['maps'][map_name]['consensus_entries']}"
        )
        for map_name in ("subsurface", "specular")
    )
    overview_detail_layout = (
        compare_regularization._overview_pixel_detail_layout(
            ("subsurface", "specular"),
            overview_captions,
            display_size=192,
            canvas_width=1800,
        )
    )
    overview_image_x = overview_detail_layout["groups"][0]["image_boxes"][0][0]
    material_tile_height = round(170 * shape[0] / shape[1])
    material_end = 537 + 128 + (material_tile_height + 52) * 3
    overview_image_y = material_end + 36 + 208
    with Image.open(report / "overview.png") as sheet:
        assert {
            sheet.getpixel((overview_image_x + dx, overview_image_y + dy))
            for dx in (0, 1)
            for dy in (0, 1)
        } == {expected_pixel}
        assert sheet.getpixel((overview_image_x + 2, overview_image_y)) == (
            expected_adjacent_pixel
        )
    metrics = (report / "metrics.csv").read_text(encoding="utf-8")
    assert "frequency_cleanup" in metrics
    assert "updated_map_entries" in metrics
    assert "mean_absolute_distance_to_frozen_target" in metrics
    assert "scalar_map_cleanup_gate" in metrics
    assert "case_png_relighting" in metrics
    assert "scalar_map_cleanup_qualification" in metrics
    assert "evaluation_case_selection" in metrics
    assert "aggregate_guide_textured_band3_8_amplitude_ratio" in metrics
    serialized = json.dumps(summary).lower()
    assert "does not establish overall material quality" in serialized

    acquisition = json.loads(
        (candidate / "material" / "olat" / "acquisition.json").read_text(
            encoding="utf-8"
        )
    )
    loaded = compare_regularization._load_frequency_frozen_artifact(
        candidate / "material" / "olat",
        acquisition,
        shape,
    )
    assert loaded["median_chunk_rows"] == bundle["median_chunk_rows"]
    assert np.array_equal(
        loaded["edge_protected"],
        loaded["edge_protected_full_precision"]
        | loaded["edge_protected_png_quantized"],
    )
    tampered = copy.deepcopy(acquisition)
    tampered["regularization"]["frozen_bundle"]["tensor_hashes"]["root"][
        "update_safe"
    ] = "0" * 64
    with pytest.raises(ValueError, match="tensor hashes"):
        compare_regularization._load_frequency_frozen_artifact(
            candidate / "material" / "olat",
            tampered,
            shape,
        )
    assert artifact_provenance["path"] == "frequency_consensus_frozen.npz"

    acquisition_path = candidate / "material" / "olat" / "acquisition.json"
    missing_guard = copy.deepcopy(acquisition)
    missing_guard["regularization"].pop("evaluation_guard")
    acquisition_path.write_text(json.dumps(missing_guard), encoding="utf-8")
    with pytest.raises(ValueError, match="missing or invalid"):
        compare_regularization.compose_regularization_comparison(
            baseline,
            candidate,
            tmp_path / "missing-guard-comparison",
            mask_path=mask_path,
        )
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")

    partial = compare_regularization.compose_regularization_comparison(
        baseline,
        candidate,
        tmp_path / "partial-comparison",
        mask_path=mask_path,
        profiles=("olat",),
    )
    partial_gates = partial["scalar_map_cleanup_qualification"]
    assert partial_gates["all_numeric_thresholds_met"] is True
    assert partial_gates["all_thresholds_met"] is False
    assert partial_gates["qualification_status"] == "INCOMPLETE"
    assert partial_gates["qualification_coverage"]["complete"] is False

    corrupted_prediction = (
        candidate
        / "evaluation"
        / "olat"
        / "cases"
        / "shared_case"
        / "predictions"
        / "mix.png"
    )
    _write_png(corrupted_prediction, np.zeros((*shape, 3), dtype=np.uint8))
    end2end_acquisition._write_scalar_error_heatmap(
        corrupted_prediction,
        baseline
        / "evaluation"
        / "olat"
        / "cases"
        / "shared_case"
        / "reference.png",
        candidate
        / "evaluation"
        / "olat"
        / "cases"
        / "shared_case"
        / "errors"
        / "mix.png",
    )
    corrupted = compare_regularization.compose_regularization_comparison(
        baseline,
        candidate,
        tmp_path / "corrupted-comparison",
        mask_path=mask_path,
    )
    corrupted_gates = corrupted["scalar_map_cleanup_qualification"]
    assert corrupted_gates["qualification_status"] == "FAIL"
    assert corrupted_gates["gates"][
        "case_png_worst_psnr_db"
    ]["meets_threshold"] is False
    assert corrupted_gates["gates"][
        "case_png_worst_delta_psnr_db"
    ]["meets_threshold"] is False

    for camera in (baseline, candidate):
        shutil.rmtree(
            camera / "evaluation" / "hdri" / "cases" / "hard_case"
        )
    with pytest.raises(ValueError, match="recorded evaluation count"):
        compare_regularization.compose_regularization_comparison(
            baseline,
            candidate,
            tmp_path / "symmetric-deletion-comparison",
            mask_path=mask_path,
        )


def test_compose_active_frequency_to_adaptive_report_validates_schema_and_keeps_fail(
    tmp_path,
    monkeypatch,
):
    baseline = tmp_path / "baseline" / "object" / "cam07"
    candidate = tmp_path / "candidate" / "object" / "cam07"
    shape = (260, 200)
    for camera in (baseline, candidate):
        _write_synthetic_camera(
            camera,
            tv_weight=0.0,
            noisy=False,
            shape=shape,
        )
    # Force a high-dynamic-range v1 update where float32
    # source + 1.0 * (desired - source) is not bit-equal to desired.
    for map_name in compare_regularization.SCALAR_MAPS:
        map_path = baseline / "material" / "olat" / "maps" / f"{map_name}.png"
        with Image.open(map_path) as image:
            values = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        values[116:125, 36:45] = 0
        values[120, 40] = 255
        _write_png(map_path, values)
    fit_mask = np.ones(shape, dtype=bool)
    fit_mask[0, 0] = False
    fit_mask[160, 96] = False
    _, adaptive_bundle = _make_active_frequency_adaptive_pair(
        baseline,
        candidate,
        fit_mask=fit_mask,
    )
    update_safe = adaptive_bundle["update_safe"].detach().cpu().numpy()
    guide_core = adaptive_bundle["guide_texture_core"].detach().cpu().numpy()
    guide_halo = adaptive_bundle["guide_texture_halo"].detach().cpu().numpy()
    adaptive_arrays = {
        map_name: {
            name: values.detach().cpu().numpy()
            for name, values in entry.items()
            if hasattr(values, "detach")
        }
        for map_name, entry in adaptive_bundle["maps"].items()
    }
    v1_evidence = np.sum(
        np.stack(
            [
                np.abs(arrays["source"] - arrays["median7"])
                > compare_regularization.FREQUENCY_EVIDENCE_THRESHOLD
                for arrays in adaptive_arrays.values()
            ],
            axis=0,
        ),
        axis=0,
        dtype=np.int64,
    )
    unchanged_strong_eligible = 0
    expected_incremental_strong_changed = 0
    simplified_fixed_targets = {}
    rounding_sensitive_maps = []
    for map_name, arrays in adaptive_arrays.items():
        source = arrays["source"]
        median3 = arrays["median3"]
        median7 = arrays["median7"]
        fixed_target = arrays["fixed_target"]
        target = arrays["target"]
        consensus_mask = arrays["consensus_mask"]
        fixed_consensus = (
            (v1_evidence >= compare_regularization.FREQUENCY_MIN_EVIDENCE_MAPS)
            & (
                np.abs(source - median7)
                > compare_regularization.FREQUENCY_OWN_DEVIATION
            )
            & update_safe
        )
        fixed_base = source + compare_regularization.FREQUENCY_BASE_BLEND * (
            median3 - source
        )
        fixed_consensus_target = (
            compare_regularization.FREQUENCY_MEDIAN3_TARGET_WEIGHT * median3
            + compare_regularization.FREQUENCY_MEDIAN7_TARGET_WEIGHT * median7
        )
        fixed_desired = np.where(
            fixed_consensus,
            fixed_consensus_target,
            fixed_base,
        )
        exact_fixed = np.where(
            update_safe,
            source + 1.0 * (fixed_desired - source),
            source,
        )
        simplified_fixed = np.where(update_safe, fixed_desired, source)
        assert np.array_equal(fixed_target, exact_fixed)
        simplified_fixed_targets[map_name] = simplified_fixed
        if not np.array_equal(exact_fixed, simplified_fixed):
            rounding_sensitive_maps.append(map_name)
        protected = (
            guide_core
            if map_name in compare_regularization.FREQUENCY_ADAPTIVE_FOCUSED_MAPS
            else guide_halo
        )
        strong_eligible = update_safe & ~protected
        expected_consensus = strong_eligible & (target != source)
        assert np.array_equal(consensus_mask, expected_consensus)
        unchanged_strong_eligible += int(
            np.count_nonzero(strong_eligible & (target == source))
        )
        expected_incremental_strong_changed += int(
            np.count_nonzero(consensus_mask & (target != fixed_target))
        )
    assert unchanged_strong_eligible > 0
    assert rounding_sensitive_maps
    rounding_tamper_map = rounding_sensitive_maps[0]
    mask_path = tmp_path / "mask.png"
    _write_png(mask_path, fit_mask.astype(np.uint8) * 255)

    prediction = (
        candidate
        / "evaluation"
        / "olat"
        / "cases"
        / "shared_case"
        / "predictions"
        / "olat.png"
    )
    _write_png(prediction, np.zeros((*shape, 3), dtype=np.uint8))
    end2end_acquisition._write_scalar_error_heatmap(
        prediction,
        baseline
        / "evaluation"
        / "olat"
        / "cases"
        / "shared_case"
        / "reference.png",
        candidate
        / "evaluation"
        / "olat"
        / "cases"
        / "shared_case"
        / "errors"
        / "olat.png",
    )
    monkeypatch.setattr(compare_regularization, "QUALIFICATION_PROFILES", ("olat",))
    summary = compare_regularization.compose_regularization_comparison(
        baseline,
        candidate,
        tmp_path / "comparison",
        mask_path=mask_path,
    )

    contract = summary["comparison_contract"]
    assert contract["comparison_mode"] == (
        "frequency-consensus-v1-to-adaptive-v1"
    )
    assert contract["baseline_tv_weight"] == pytest.approx(0.00125)
    assert contract["regularized_tv_weight"] == pytest.approx(0.00125)
    assert summary["labels"]["baseline"].startswith(
        "Baseline · post-fit frequency consensus"
    )
    assert summary["labels"]["regularized"].startswith(
        "Regularized · adaptive frequency consensus"
    )
    cleanup = summary["frequency_cleanup"]
    assert cleanup["comparison_mode"] == (
        "frequency-consensus-v1-to-adaptive-v1"
    )
    assert "adaptive target differs" in cleanup["update_count_semantics"]
    assert cleanup["profiles"]["olat"]["artifact"]["validated"] is True
    assert cleanup["profiles"]["olat"]["artifact"][
        "consensus_entry_semantics"
    ] == compare_regularization.FREQUENCY_ADAPTIVE_CONSENSUS_ENTRY_SEMANTICS
    assert cleanup["profiles"]["olat"]["annotation_labels"] == {
        "updated": "changed",
        "consensus": "strong changed",
    }
    assert cleanup["profiles"]["olat"]["consensus"][
        "map_entries"
    ] == expected_incremental_strong_changed
    evidence = cleanup["adaptive_cleanup_evidence"]
    assert evidence["schema"] == (
        compare_regularization.FREQUENCY_ADAPTIVE_EVIDENCE_SCHEMA
    )
    assert evidence["focused_maps"] == list(
        compare_regularization.FREQUENCY_ADAPTIVE_FOCUSED_MAPS
    )
    assert evidence["definition"]["threshold_operator"] == ">"
    assert evidence["definition"]["outlier"] == (
        "abs(map - own median5_nearest) > 0.05"
    )
    assert evidence["presentation"]["manual_selection"] is False
    for map_name in compare_regularization.FREQUENCY_ADAPTIVE_FOCUSED_MAPS:
        selected = evidence["crops"][map_name]
        left, top, right, bottom = selected["crop_box_xyxy"]
        assert right - left == 96
        assert bottom - top == 96
        assert selected["manual_selection"] is False
        crop_counts = selected["profiles"]["olat"]
        assert crop_counts["fixed_v1_outliers"] == (
            crop_counts["removed"] + crop_counts["persistent"]
        )
        assert crop_counts["adaptive_v1_outliers"] == (
            crop_counts["introduced"] + crop_counts["persistent"]
        )
    for map_name, counts in evidence["profiles"]["olat"]["maps"].items():
        assert map_name in compare_regularization.SCALAR_MAPS
        assert counts["fixed_v1_outliers"] == (
            counts["removed"] + counts["persistent"]
        )
        assert counts["adaptive_v1_outliers"] == (
            counts["introduced"] + counts["persistent"]
        )
    assert summary["scalar_map_cleanup_qualification"][
        "qualification_status"
    ] == "FAIL"
    assert (tmp_path / "comparison" / "overview.png").is_file()
    assert (
        tmp_path / "comparison" / "material" / "frequency_hotspot_1to1.png"
    ).is_file()
    evidence_path = (
        tmp_path / "comparison" / "material" / "adaptive_cleanup_evidence.png"
    )
    assert evidence_path.is_file()
    assert summary["artifacts"]["frequency_diagnostics"][
        "adaptive_cleanup_evidence"
    ] == "material/adaptive_cleanup_evidence.png"
    with Image.open(evidence_path) as sheet:
        assert sheet.mode == "RGB"
        assert sheet.size == (1830, 1666)

    acquisition_path = candidate / "material" / "olat" / "acquisition.json"
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    bad_bundle = copy.deepcopy(acquisition)
    bad_bundle["regularization"]["frozen_bundle"]["schema"] = (
        compare_regularization.FREQUENCY_BUNDLE_SCHEMA
    )
    acquisition_path.write_text(json.dumps(bad_bundle), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen-bundle provenance"):
        compare_regularization.compose_regularization_comparison(
            baseline,
            candidate,
            tmp_path / "bad-bundle-comparison",
            mask_path=mask_path,
        )
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")

    bad_semantics = copy.deepcopy(acquisition)
    bad_semantics["regularization"]["frozen_bundle"][
        "consensus_entry_semantics"
    ] = "strong_policy_eligible"
    acquisition_path.write_text(json.dumps(bad_semantics), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen-bundle consensus semantics"):
        compare_regularization.compose_regularization_comparison(
            baseline,
            candidate,
            tmp_path / "bad-consensus-semantics-comparison",
            mask_path=mask_path,
        )
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")

    artifact_path = (
        candidate
        / "material"
        / "olat"
        / compare_regularization.FREQUENCY_FROZEN_ARTIFACT_NAME
    )
    original_artifact = artifact_path.read_bytes()
    with np.load(artifact_path, allow_pickle=False) as payload:
        artifact_payload = {
            name: np.array(payload[name], copy=True) for name in payload.files
        }
    artifact_metadata = json.loads(str(artifact_payload["metadata"].item()))
    artifact_metadata["consensus_entry_semantics"] = "strong_policy_eligible"
    artifact_payload["metadata"] = np.asarray(
        json.dumps(artifact_metadata, sort_keys=True)
    )
    with artifact_path.open("wb") as stream:
        np.savez_compressed(stream, **artifact_payload)
    bad_metadata = copy.deepcopy(acquisition)
    bad_metadata_artifact = bad_metadata["regularization"]["frozen_bundle"][
        "artifact"
    ]
    bad_metadata_artifact["sha256"] = _file_sha256(artifact_path)
    bad_metadata_artifact["bytes"] = artifact_path.stat().st_size
    acquisition_path.write_text(json.dumps(bad_metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata consensus semantics"):
        compare_regularization.compose_regularization_comparison(
            baseline,
            candidate,
            tmp_path / "bad-metadata-semantics-comparison",
            mask_path=mask_path,
        )
    artifact_path.write_bytes(original_artifact)
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")

    with np.load(artifact_path, allow_pickle=False) as payload:
        simplified_payload = {
            name: np.array(payload[name], copy=True) for name in payload.files
        }
    fixed_target_key = f"{rounding_tamper_map}__fixed_target"
    exact_fixed_target = simplified_payload[fixed_target_key]
    simplified_fixed_target = simplified_fixed_targets[rounding_tamper_map]
    assert not np.array_equal(exact_fixed_target, simplified_fixed_target)
    simplified_payload[fixed_target_key] = simplified_fixed_target
    simplified_hash = _array_sha256(simplified_fixed_target)
    simplified_metadata = json.loads(
        str(simplified_payload["metadata"].item())
    )
    simplified_metadata["tensor_hashes"]["maps"][rounding_tamper_map][
        "fixed_target"
    ] = simplified_hash
    simplified_payload["metadata"] = np.asarray(
        json.dumps(simplified_metadata, sort_keys=True)
    )
    with artifact_path.open("wb") as stream:
        np.savez_compressed(stream, **simplified_payload)
    simplified_acquisition = copy.deepcopy(acquisition)
    simplified_bundle = simplified_acquisition["regularization"][
        "frozen_bundle"
    ]
    simplified_bundle["tensor_hashes"]["maps"][rounding_tamper_map][
        "fixed_target"
    ] = simplified_hash
    simplified_bundle["maps"][rounding_tamper_map][
        "fixed_target_sha256"
    ] = simplified_hash
    simplified_bundle["artifact"]["sha256"] = _file_sha256(artifact_path)
    simplified_bundle["artifact"]["bytes"] = artifact_path.stat().st_size
    acquisition_path.write_text(
        json.dumps(simplified_acquisition), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="fixed v1 target is stale"):
        compare_regularization.compose_regularization_comparison(
            baseline,
            candidate,
            tmp_path / "simplified-fixed-target-comparison",
            mask_path=mask_path,
        )
    artifact_path.write_bytes(original_artifact)
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")

    bad_diagnostic = copy.deepcopy(acquisition)
    bad_diagnostic["regularization"]["cleanup_diagnostic"]["schema"] = (
        "ictpolarreal.frequency-consensus-adaptive-diagnostic.invalid"
    )
    acquisition_path.write_text(json.dumps(bad_diagnostic), encoding="utf-8")
    with pytest.raises(ValueError, match="cleanup diagnostics"):
        compare_regularization.compose_regularization_comparison(
            baseline,
            candidate,
            tmp_path / "bad-diagnostic-comparison",
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
