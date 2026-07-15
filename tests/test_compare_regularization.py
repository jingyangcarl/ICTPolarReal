from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from PIL import Image

from ictpolarreal.processing import compare_regularization


def _write_png(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8)).save(path)


def _write_synthetic_camera(root, *, tv_weight, noisy):
    profile = "olat"
    manifest = {
        "profiles": [profile],
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
    acquisition = root / "material" / profile / "acquisition.json"
    acquisition.parent.mkdir(parents=True, exist_ok=True)
    acquisition.write_text(json.dumps(common), encoding="utf-8")

    height, width = 32, 24
    base_color = np.full((height, width, 3), 128, dtype=np.uint8)
    _write_png(root / "material" / profile / "maps" / "baseColor.png", base_color)
    for index, name in enumerate(compare_regularization.SCALAR_MAPS):
        values = np.full((height, width), 80 + index, dtype=np.uint8)
        if noisy:
            values[8::4, 8::4] = 220
        _write_png(root / "material" / profile / "maps" / f"{name}.png", values)

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
        _write_png(case / "predictions" / f"{profile}.png", panel)
        _write_png(case / "errors" / f"{profile}.png", panel // 4)


def test_map_noise_metrics_distinguish_speckle_from_constant_map():
    mask = np.ones((9, 9), dtype=bool)
    constant = np.full((9, 9), 0.5, dtype=np.float32)
    speckled = constant.copy()
    speckled[4, 4] = 1.0

    clean = compare_regularization._map_noise_metrics(constant, mask)
    noisy = compare_regularization._map_noise_metrics(speckled, mask)

    assert clean == {
        "neighbor_tv": pytest.approx(0.0),
        "median5_residual_mae": pytest.approx(0.0),
        "speckle_fraction_gt_0p05": pytest.approx(0.0),
    }
    assert noisy["neighbor_tv"] > clean["neighbor_tv"]
    assert noisy["median5_residual_mae"] > clean["median5_residual_mae"]
    assert noisy["speckle_fraction_gt_0p05"] > 0.0


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
    assert summary["aggregate_map_noise"]["neighbor_tv"]["reduction_fraction"] > 0
    assert (report / "overview.png").is_file()
    assert (report / "material" / "olat.png").is_file()
    with Image.open(report / "material" / "olat.png") as material_sheet:
        assert material_sheet.width == (
            compare_regularization.MATERIAL_LABEL_GUTTER + 4 * 24
        )
    assert (report / "evaluation" / "olat.png").is_file()
    assert (report / "evaluation" / "hdri.png").is_file()
    assert (report / "metrics.csv").is_file()
    assert not (tmp_path / ".comparison.tmp").exists()


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
