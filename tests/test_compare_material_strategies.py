from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ictpolarreal.processing import compare_material_strategies as strategies
from ictpolarreal.processing import compare_regularization as pair
from ictpolarreal.processing import end2end_acquisition


PROFILES = ("olat", "hdri", "mix")


def _write_png(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8)).save(path)


def _foreground_hash(mask: np.ndarray) -> str:
    import hashlib

    values = np.ascontiguousarray(mask.astype(np.float32)[..., None])
    return hashlib.sha256(memoryview(values).cast("B")).hexdigest()


def _regularization_records(kind: str, weight: float, steps: int):
    parameters = list(end2end_acquisition.DISNEY_TV_SCALAR_NAMES)
    if kind == "frequency-consensus":
        settings = end2end_acquisition._regularization_settings(kind)
        stage = end2end_acquisition._frequency_consensus_stage_plan(
            steps,
            enabled=weight > 0.0,
            weight=weight,
        )
        signature = {
            "kind": kind,
            "weight": weight,
            "parameters": parameters,
            "settings": settings,
            "stage_plan": stage,
        }
        result = {
            **signature,
            "weight_semantics": "normalized_post_fit_strength",
            "cleanup_applied": weight > 0.0,
            "cleanup_diagnostic": {} if weight > 0.0 else None,
            "data_objective_only": True,
        }
        return signature, result
    if kind == strategies.TRAIN_REGULARIZER_KIND:
        settings = end2end_acquisition._regularization_settings(kind)
        stage = end2end_acquisition._frequency_consensus_regularizer_stage_plan(
            steps,
            enabled=True,
        )
        signature = {
            "kind": kind,
            "weight": weight,
            "parameters": parameters,
            "settings": settings,
            "stage_plan": stage,
        }
        final_loss = 0.2
        result = {
            **signature,
            "weight_semantics": "objective_coefficient",
            "frozen_bundle": {
                "schema": "ictpolarreal.frequency-consensus-bundle.v1",
                "role": "train_time_frozen_target",
                "created_after_step": stage["target_after_data_step"],
                "strength": 1.0,
                "artifact": {
                    "path": "frequency_consensus_frozen.npz",
                    "sha256": "synthetic",
                    "bytes": 1,
                },
            },
            "post_fit_updates": 0,
            "cleanup_applied": False,
            "regularized_steps_completed": stage["regularized_steps"],
            "data_objective_only": False,
            "final_regularization_loss": final_loss,
            "weighted_final_regularization_loss": weight * final_loss,
        }
        return signature, result
    raise AssertionError(kind)


def _write_strategy_camera(
    root: Path,
    *,
    role: str,
    mask: np.ndarray,
    steps: int = 100,
) -> None:
    if role == "train_time_regularizer":
        kind = strategies.TRAIN_REGULARIZER_KIND
        weight = 0.00125
        schema = "ictpolarreal.end2end-disney.v14"
        checkpoint_schema = "ictpolarreal.end2end-checkpoint.v14"
    else:
        kind = "frequency-consensus"
        weight = 0.00125 if role == "post_fit_cleanup" else 0.0
        schema = "ictpolarreal.end2end-disney.v13"
        checkpoint_schema = "ictpolarreal.end2end-checkpoint.v13"
    signature_regularization, regularization = _regularization_records(
        kind,
        weight,
        steps,
    )
    manifest = {
        "profiles": list(PROFILES),
        "evaluation": {"status": "complete"},
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    height, width = mask.shape
    yy, xx = np.indices(mask.shape)
    base_color = np.zeros((height, width, 3), dtype=np.uint8)
    base_color[..., :] = 78
    base_color[:, width // 2 :, :] = 178
    base_color[70:150, 45:120, 1] += ((xx[70:150, 45:120] % 9) * 3).astype(
        np.uint8
    )
    normal = np.zeros((height, width, 3), dtype=np.uint8)
    normal[..., :2] = 128
    normal[..., 2] = 255
    adapter = end2end_acquisition._adapter_provenance(kind)
    for profile_index, profile in enumerate(PROFILES):
        acquisition_regularization = copy.deepcopy(regularization)
        signature = {
            "schema": checkpoint_schema,
            "profile": profile,
            "model": "DisneyBRDFSimplifiedMultiLayer",
            "height": height,
            "width": width,
            "light_directions_sha256": "lights",
            "light_ids_sha256": "light-ids",
            "frame_ids_sha256": "frame-ids",
            "base_color_sha256": "base",
            "normal_sha256": "normal",
            "foreground_sha256": _foreground_hash(mask),
            "fit_indices": [0, 1, 2],
            "heldout_indices": [3, 4],
            "excluded_indices": [],
            "hdri_condition_weights_sha256": "hdri",
            "hdri_fit_condition_ids": ["fit"],
            "hdri_evaluation_condition_ids": ["case_a", "case_b"],
            "mix_schedule": "4_hdri_then_4_olat",
            "base_color_source": "dataset_albedo",
            "scalar_initialization": {"roughness": 0.5},
            "surface_validity": {"front_facing_pixels": int(mask.sum())},
            "steps": steps,
            "learning_rate": 0.001,
            "regularization": copy.deepcopy(signature_regularization),
            "disney_brdf_sha256": "disney",
            "adapter": copy.deepcopy(adapter),
        }
        evaluation = {
            "profile": profile,
            "evaluations": {
                lighting: {
                    "count": 2,
                    "metrics": {
                        "mse": 0.01,
                        "mae": 0.04,
                        "psnr": 24.0 + profile_index + (0.02 if role != "data_only" else 0.0),
                        "ssim_global": 0.86 + 0.01 * profile_index,
                        "mean_intensity_ratio": 1.0,
                        "luminance_correlation": 0.92,
                    },
                }
                for lighting in pair.EVALUATION_LIGHTING
            },
        }
        acquisition = {
            "schema": schema,
            "lighting_profile": profile,
            "base_color_source": "dataset_albedo",
            "input_hashes": {
                "base_color_sha256": "base",
                "normal_sha256": "normal",
                "foreground_sha256": _foreground_hash(mask),
            },
            "checkpoint_signature": signature,
            "regularization": acquisition_regularization,
            "evaluation": evaluation,
        }
        material = root / "material" / profile
        material.mkdir(parents=True, exist_ok=True)
        (material / "acquisition.json").write_text(
            json.dumps(acquisition), encoding="utf-8"
        )
        maps = material / "maps"
        _write_png(maps / "baseColor.png", base_color)
        _write_png(maps / "normal.png", normal)
        for map_index, map_name in enumerate(pair.SCALAR_MAPS):
            values = np.full(mask.shape, 70 + 8 * map_index, dtype=np.uint8)
            values[:, width // 2 :] += 45
            values += ((xx + 2 * yy + map_index) % 7).astype(np.uint8)
            # Identical deterministic impulse sites; each strategy controls their
            # strength so report metrics have visible but non-cherry-picked changes.
            impulse = (yy % 23 == 7) & (xx % 19 == 5) & mask
            values[impulse] = {
                "data_only": 230,
                "post_fit_cleanup": 150,
                "train_time_regularizer": 175,
            }[role]
            values[~mask] = 0
            _write_png(maps / f"{map_name}.png", values)

    summary = {
        "suites": {
            lighting: {"representative_case": "case_a"}
            for lighting in pair.EVALUATION_LIGHTING
        }
    }
    evaluation_root = root / "evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    (evaluation_root / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    reference = np.stack(
        (
            40 + (xx + yy) % 160,
            50 + (2 * xx + yy) % 150,
            60 + (xx + 2 * yy) % 140,
        ),
        axis=-1,
    ).astype(np.uint8)
    for lighting in pair.EVALUATION_LIGHTING:
        for case_index, case_id in enumerate(("case_a", "case_b")):
            case = evaluation_root / lighting / "cases" / case_id
            _write_png(case / "reference.png", reference)
            _write_png(case / "lighting.png", reference)
            for profile_index, profile in enumerate(PROFILES):
                offset = {
                    "data_only": 3,
                    "post_fit_cleanup": 2,
                    "train_time_regularizer": 4,
                }[role]
                prediction = np.clip(
                    reference.astype(np.int16)
                    + offset
                    + case_index
                    + profile_index,
                    0,
                    255,
                ).astype(np.uint8)
                _write_png(case / "predictions" / f"{profile}.png", prediction)
                error = pair._scalar_error_heatmap_u8(
                    prediction.astype(np.float32) / 255.0,
                    reference.astype(np.float32) / 255.0,
                )
                _write_png(case / "errors" / f"{profile}.png", error)


@pytest.fixture
def strategy_trees(tmp_path):
    shape = (260, 200)
    mask = np.ones(shape, dtype=bool)
    mask[:3] = False
    mask[-3:] = False
    mask[:, :3] = False
    mask[:, -3:] = False
    roots = {
        role: tmp_path / role / "subject" / "cam07"
        for role in strategies.VARIANT_ROLES
    }
    for role, root in roots.items():
        _write_strategy_camera(root, role=role, mask=mask)
    mask_path = tmp_path / "mask.png"
    _write_png(mask_path, mask.astype(np.uint8) * 255)
    return roots, mask_path


def test_train_contract_accepts_only_explicit_v13_to_v14_transition(strategy_trees):
    roots, _ = strategy_trees
    acquisitions = {
        variant: pair._load_acquisitions(root, PROFILES)
        for variant, root in (
            ("baseline", roots["data_only"]),
            ("regularized", roots["train_time_regularizer"]),
        )
    }
    contract = strategies._validate_train_comparison_contract(
        acquisitions, PROFILES
    )
    assert contract["controlled"] is True
    assert contract["regularization_kind"] == strategies.TRAIN_REGULARIZER_KIND

    acquisitions["regularized"]["olat"]["checkpoint_signature"][
        "learning_rate"
    ] = 0.002
    with pytest.raises(ValueError, match="differs beyond"):
        strategies._validate_train_comparison_contract(acquisitions, PROFILES)

    acquisitions = {
        variant: pair._load_acquisitions(root, PROFILES)
        for variant, root in (
            ("baseline", roots["data_only"]),
            ("regularized", roots["train_time_regularizer"]),
        )
    }
    acquisitions["regularized"]["olat"]["schema"] = "garbage.accepted.by.report"
    with pytest.raises(ValueError, match="requires v13 data-only and v14 train"):
        strategies._validate_train_comparison_contract(acquisitions, PROFILES)


def test_strategy_report_uses_one_recorded_fit_mask_rule(strategy_trees):
    roots, _ = strategy_trees
    acquisitions = {
        role: pair._load_acquisitions(root, PROFILES)
        for role, root in roots.items()
    }
    for role in strategies.VARIANT_ROLES:
        for profile in PROFILES:
            acquisition = acquisitions[role][profile]
            acquisition["surface_validity"] = {
                "fit_mask_rule": end2end_acquisition.FIT_MASK_RULE,
            }
            acquisition["adapter"] = {
                "fit_mask_rule": end2end_acquisition.FIT_MASK_RULE,
            }

    assert strategies._common_fit_mask_rule(acquisitions, PROFILES) == (
        end2end_acquisition.FIT_MASK_RULE
    )

    acquisitions["train_time_regularizer"]["mix"]["adapter"][
        "fit_mask_rule"
    ] = "legacy-different-rule"
    with pytest.raises(ValueError, match="different fitting-mask rules"):
        strategies._common_fit_mask_rule(acquisitions, PROFILES)


def test_strategy_contract_rejects_train_arm_with_cleanup(strategy_trees):
    roots, _ = strategy_trees
    loaded = {
        role: pair._load_acquisitions(root, PROFILES)
        for role, root in roots.items()
    }
    cleanup_contract = pair._validate_comparison_contract(
        {
            "baseline": loaded["data_only"],
            "regularized": loaded["post_fit_cleanup"],
        },
        PROFILES,
    )
    train_contract = strategies._validate_train_comparison_contract(
        {
            "baseline": loaded["data_only"],
            "regularized": loaded["train_time_regularizer"],
        },
        PROFILES,
    )
    loaded["train_time_regularizer"]["olat"]["regularization"][
        "cleanup_applied"
    ] = True
    with pytest.raises(ValueError, match="must not apply post-fit cleanup"):
        strategies._validate_strategy_contracts(
            loaded,
            PROFILES,
            cleanup_contract=cleanup_contract,
            train_contract=train_contract,
        )

    loaded = {
        role: pair._load_acquisitions(root, PROFILES)
        for role, root in roots.items()
    }
    loaded["data_only"]["olat"]["regularization"]["cleanup_applied"] = True
    loaded["data_only"]["olat"]["regularization"]["data_objective_only"] = False
    with pytest.raises(ValueError, match="inactive v13 frequency-consensus stage"):
        strategies._validate_strategy_contracts(
            loaded,
            PROFILES,
            cleanup_contract=cleanup_contract,
            train_contract=train_contract,
        )

    loaded = {
        role: pair._load_acquisitions(root, PROFILES)
        for role, root in roots.items()
    }
    loaded["train_time_regularizer"]["olat"]["regularization"]["settings"][
        "epsilon"
    ] = 0.5
    with pytest.raises(ValueError, match="canonical v14"):
        strategies._validate_strategy_contracts(
            loaded,
            PROFILES,
            cleanup_contract=cleanup_contract,
            train_contract=train_contract,
        )


def test_compose_requires_all_profiles_and_valid_train_artifact(
    strategy_trees,
    monkeypatch,
):
    roots, mask_path = strategy_trees
    monkeypatch.setattr(
        strategies.acquisition_core,
        "_frequency_frozen_artifact_complete",
        lambda *_args, **_kwargs: False,
    )
    with pytest.raises(ValueError, match="frozen artifact is incomplete"):
        strategies.compose_material_strategy_comparison(
            roots["data_only"],
            roots["post_fit_cleanup"],
            roots["train_time_regularizer"],
            roots["data_only"].parents[2] / "bad-artifact-report",
            mask_path=mask_path,
        )

    for root in roots.values():
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["profiles"] = ["olat", "hdri"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="requires exactly the olat, hdri, and mix"):
        strategies.compose_material_strategy_comparison(
            roots["data_only"],
            roots["post_fit_cleanup"],
            roots["train_time_regularizer"],
            roots["data_only"].parents[2] / "subset-report",
            mask_path=mask_path,
        )


def test_shared_case_representative_depends_only_on_data_only_metrics():
    rows = []
    for case_index, case_id in enumerate(("a", "b", "c")):
        for profile in ("olat",):
            rows.append(
                {
                    "lighting": "olat",
                    "case_id": case_id,
                    "profile": profile,
                    "metrics": {"data_only": {"psnr": 10.0 + case_index}},
                    "delta_vs_data_only": {
                        "post_fit_cleanup": {"psnr": -case_index},
                        "train_time_regularizer": {"psnr": case_index},
                    },
                }
            )
    first = strategies._select_shared_cases(rows, "olat", ("olat",))
    for row in rows:
        row["delta_vs_data_only"]["post_fit_cleanup"]["psnr"] *= -100.0
        row["delta_vs_data_only"]["train_time_regularizer"]["psnr"] *= 100.0
    second = strategies._select_shared_cases(rows, "olat", ("olat",))
    assert first["representative"]["case_id"] == "b"
    assert second["representative"]["case_id"] == "b"


def test_compose_material_strategy_report_is_atomic_and_has_exact_tree(
    strategy_trees,
    monkeypatch,
    tmp_path,
):
    roots, mask_path = strategy_trees
    monkeypatch.setattr(
        pair,
        "_measure_frequency_cleanup",
        lambda *_args, **_kwargs: {
            "available": True,
            "schema": "synthetic-validated-cleanup.v1",
        },
    )
    monkeypatch.setattr(
        strategies.acquisition_core,
        "_frequency_frozen_artifact_complete",
        lambda *_args, **_kwargs: True,
    )
    output = tmp_path / "comparison"
    summary = strategies.compose_material_strategy_comparison(
        roots["data_only"],
        roots["post_fit_cleanup"],
        roots["train_time_regularizer"],
        output,
        mask_path=mask_path,
    )
    expected = {
        "overview.png",
        "summary.json",
        "metrics.csv",
        "material/olat.png",
        "material/hdri.png",
        "material/mix.png",
        "evaluation/olat.png",
        "evaluation/hdri.png",
    }
    assert {
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file()
    } == expected
    assert summary["schema"] == strategies.REPORT_SCHEMA
    assert summary["variant_order"] == list(strategies.VARIANT_ROLES)
    assert [summary["variants"][role]["label"] for role in strategies.VARIANT_ROLES] == [
        "Data-only fit",
        "Fixed post-fit cleanup",
        "Train-time consensus regularizer",
    ]
    assert "preprocess" not in json.dumps(summary).lower()
    with Image.open(output / "overview.png") as overview:
        assert overview.mode == "RGB"
        assert overview.width == strategies.OVERVIEW_WIDTH
        assert overview.height > 2500
    with (output / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["variant"] for row in rows if row["category"] == "material"} == set(
        strategies.VARIANT_ROLES
    )
    assert not (tmp_path / ".comparison.tmp").exists()
    assert not (tmp_path / ".comparison.previous").exists()


def test_compose_failure_preserves_existing_report(
    strategy_trees,
    monkeypatch,
    tmp_path,
):
    roots, mask_path = strategy_trees
    monkeypatch.setattr(
        pair,
        "_measure_frequency_cleanup",
        lambda *_args, **_kwargs: {"available": True},
    )
    monkeypatch.setattr(
        strategies.acquisition_core,
        "_frequency_frozen_artifact_complete",
        lambda *_args, **_kwargs: True,
    )
    output = tmp_path / "comparison"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing", encoding="utf-8")

    def fail(*_args, **_kwargs):
        raise RuntimeError("writer failed")

    monkeypatch.setattr(strategies, "_write_material_sheet", fail)
    with pytest.raises(RuntimeError, match="writer failed"):
        strategies.compose_material_strategy_comparison(
            roots["data_only"],
            roots["post_fit_cleanup"],
            roots["train_time_regularizer"],
            output,
            mask_path=mask_path,
        )
    assert sentinel.read_text(encoding="utf-8") == "existing"
    assert not (tmp_path / ".comparison.tmp").exists()
    assert not (tmp_path / ".comparison.previous").exists()


def test_atomic_swap_failure_restores_previous_and_startup_recovers(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "comparison"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    stage = tmp_path / ".comparison.tmp"
    stage.mkdir()
    (stage / "new.txt").write_text("new", encoding="utf-8")
    backup = tmp_path / ".comparison.previous"
    original_replace = Path.replace

    def fail_stage_replace(self, target):
        if self == stage and Path(target) == output:
            raise OSError("simulated swap failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_stage_replace)
    with pytest.raises(OSError, match="simulated swap failure"):
        strategies._commit_staged_report(stage, output, backup)
    assert (output / "old.txt").read_text(encoding="utf-8") == "old"
    assert not backup.exists()

    monkeypatch.setattr(Path, "replace", original_replace)
    output.replace(backup)
    stale_stage = tmp_path / ".comparison.tmp"
    if stale_stage.exists():
        for child in stale_stage.iterdir():
            child.unlink()
    else:
        stale_stage.mkdir()
    (stale_stage / "partial.txt").write_text("partial", encoding="utf-8")
    strategies._recover_interrupted_report(output, stale_stage, backup)
    assert (output / "old.txt").read_text(encoding="utf-8") == "old"
    assert not stale_stage.exists()
    assert not backup.exists()
