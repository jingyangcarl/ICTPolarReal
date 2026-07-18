from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ictpolarreal.processing import render_saved_material
from ictpolarreal.utils.io import write_image


def _oriented_replay_contract():
    adapter = {
        "schema": "ictpolarreal.profile-acquisition-adapter.v7",
        "algorithm_version": "ictpolarreal-frequency-consensus-v2",
        "fit_mask_rule": render_saved_material.FIT_MASK_RULE,
        "normal_orientation_rule": render_saved_material.NORMAL_ORIENTATION_RULE,
        "minimum_oriented_n_dot_v": (
            render_saved_material.MIN_ORIENTED_N_DOT_V
        ),
    }
    surface = {
        "fit_mask_rule": render_saved_material.FIT_MASK_RULE,
        "normal_orientation_rule": render_saved_material.NORMAL_ORIENTATION_RULE,
        "minimum_n_dot_v": render_saved_material.MIN_ORIENTED_N_DOT_V,
        "capture_foreground_pixels": 12,
        "front_facing_pixels": 12,
        "excluded_back_facing_pixels": 0,
        "excluded_fraction": 0.0,
    }
    acquisition = {
        "schema": "ictpolarreal.end2end-disney.v13",
        "adapter": dict(adapter),
        "surface_validity": dict(surface),
        "input_hashes": {
            "capture_foreground_sha256": "a" * 64,
            "foreground_sha256": "a" * 64,
            "source_normal_sha256": "b" * 64,
        },
    }
    signature = {
        "schema": "ictpolarreal.end2end-checkpoint.v13",
        "adapter": dict(adapter),
        "surface_validity": dict(surface),
    }
    return acquisition, signature


def test_oriented_replay_contract_rejects_stale_or_partial_fit_provenance():
    acquisition, signature = _oriented_replay_contract()
    render_saved_material._validate_oriented_replay_contract(
        acquisition, signature
    )

    signature["adapter"]["algorithm_version"] = (
        "ictpolarreal-frequency-consensus-v1"
    )
    with pytest.raises(ValueError, match="adapter provenance differ"):
        render_saved_material._validate_oriented_replay_contract(
            acquisition, signature
        )

    acquisition, signature = _oriented_replay_contract()
    acquisition["surface_validity"]["front_facing_pixels"] = 11
    signature["surface_validity"]["front_facing_pixels"] = 11
    with pytest.raises(ValueError, match="every clean foreground pixel"):
        render_saved_material._validate_oriented_replay_contract(
            acquisition, signature
        )


def test_parse_condition_indices_preserves_batch_order_and_deduplicates():
    indices = render_saved_material.parse_condition_indices(
        "1:14:4, 9, 20", count=32
    )

    assert indices == [1, 5, 9, 13, 20]


@pytest.mark.parametrize("expression", ["", "1:", "1:4:0", "-1", "1,,2"])
def test_parse_condition_indices_rejects_ambiguous_or_unsafe_ranges(expression):
    with pytest.raises((ValueError, IndexError)):
        render_saved_material.parse_condition_indices(expression, count=8)


def test_load_recorded_lighting_keeps_absolute_rows_and_split_support(tmp_path):
    camera_dir = tmp_path / "object" / "cam07"
    assets = camera_dir / "evaluation" / "assets"
    assets.mkdir(parents=True)
    conditions = [
        {
            "condition_id": "fit_rot000",
            "split": "fit",
            "source": "fit.hdr",
            "rotation_degrees": 0,
        },
        {
            "condition_id": "heldout_rot000",
            "split": "heldout",
            "source": "heldout.hdr",
            "rotation_degrees": 0,
        },
    ]
    (assets / "conditions.json").write_text(
        json.dumps(
            {
                "schema": render_saved_material.EXPECTED_CONDITIONS_SCHEMA,
                "conditions": conditions,
                "projection": {
                    "fit_support_indices": [0, 2],
                    "evaluation_support_indices": [1],
                },
            }
        ),
        encoding="utf-8",
    )
    weights = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    np.savez(
        assets / "weights.npz",
        condition_ids=np.asarray(["fit_rot000", "heldout_rot000"]),
        splits=np.asarray(["fit", "heldout"]),
        weights=weights,
        fit_support_indices=np.asarray([0, 2], dtype=np.int64),
        evaluation_support_indices=np.asarray([1], dtype=np.int64),
    )

    lighting = render_saved_material.load_recorded_lighting(camera_dir)

    np.testing.assert_array_equal(lighting.weights, weights)
    np.testing.assert_array_equal(
        render_saved_material.support_for_condition(lighting, 0), [0, 2]
    )
    np.testing.assert_array_equal(
        render_saved_material.support_for_condition(lighting, 1), [1]
    )
    requests = render_saved_material.recorded_render_conditions(lighting, [1, 0])
    assert [request.record["absolute_index"] for request in requests] == [1, 0]
    assert requests[0].record["weight_sha256"] == (
        render_saved_material._array_sha256(weights[1])
    )


def test_reconstruct_light_ids_uses_saved_stack_order_not_split_order():
    acquisition = {
        "light_split": {
            "fit_stack_indices": [2, 0],
            "fit_light_indices": [12, 10],
            "heldout_stack_indices": [1],
            "heldout_light_indices": [11],
            "excluded_stack_indices": [3],
            "excluded_light_indices": [13],
        }
    }

    light_ids = render_saved_material.reconstruct_light_ids(acquisition, 4)

    np.testing.assert_array_equal(light_ids, [10, 11, 12, 13])


def test_reconstruct_light_ids_rejects_missing_stack_rows():
    acquisition = {
        "light_split": {
            "fit_stack_indices": [0],
            "fit_light_indices": [10],
            "heldout_stack_indices": [],
            "heldout_light_indices": [],
            "excluded_stack_indices": [],
            "excluded_light_indices": [],
        }
    }

    with pytest.raises(ValueError, match="does not cover"):
        render_saved_material.reconstruct_light_ids(acquisition, 2)


def test_reconstruct_frame_ids_uses_native_stack_provenance_order():
    acquisition = {
        "light_split": {
            "fit_stack_indices": [2, 0],
            "fit_frame_ids": [102, 100],
            "heldout_stack_indices": [1],
            "heldout_frame_ids": [101],
            "excluded_stack_indices": [3],
            "excluded_frame_ids": [103],
        }
    }

    frame_ids = render_saved_material.reconstruct_frame_ids(acquisition, 4)

    np.testing.assert_array_equal(frame_ids, [100, 101, 102, 103])


def test_load_parallel_targets_preserves_frame_order_and_acquisition_hash(
    monkeypatch, tmp_path,
):
    frame_ids = np.asarray([12, 4, 9], dtype=np.int64)
    for frame_id in frame_ids:
        (tmp_path / f"{int(frame_id)}.png").touch()
    sample = SimpleNamespace(
        light_path=lambda polarization, frame_id: tmp_path / f"{frame_id}.png"
    )

    def fake_read(path, *, channels):
        value = int(Path(path).stem)
        image = np.full((2, 1, 3), value, dtype=np.float32)
        if value == 4:
            image[0, 0, 0] = np.nan
            image[0, 0, 1] = -2.0
            image[0, 0, 2] = np.inf
            image[1, 0, 0] = -np.inf
        return image

    monkeypatch.setattr(render_saved_material, "read_image", fake_read)

    targets = render_saved_material.load_parallel_targets(
        sample, frame_ids, height=2, width=1
    )

    assert targets.dtype == np.float32
    np.testing.assert_array_equal(targets[:, 1, 0, 0], [12.0, 0.0, 9.0])
    np.testing.assert_array_equal(targets[1, 0, 0], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(targets[1, 1, 0], [0.0, 4.0, 4.0])
    assert render_saved_material._array_sha256(targets) == (
        render_saved_material._array_sha256(np.ascontiguousarray(targets))
    )


def test_load_parallel_targets_reports_missing_recorded_frame():
    sample = SimpleNamespace(light_path=lambda _polarization, _frame_id: None)

    with pytest.raises(FileNotFoundError, match="missing parallel OLAT frame 77"):
        render_saved_material.load_parallel_targets(
            sample, [77], height=2, width=1
        )


def test_replay_target_loader_matches_acquisition_polarization_order(
    monkeypatch, tmp_path,
):
    frame_ids = np.asarray([12, 4], dtype=np.int64)
    for polarization in ("cross", "parallel"):
        for frame_id in frame_ids:
            (tmp_path / f"{polarization}_{int(frame_id)}.png").touch()
    sample = SimpleNamespace(
        light_path=lambda polarization, frame_id: (
            tmp_path / f"{polarization}_{frame_id}.png"
        )
    )
    calls = []

    def fake_read(path, *, channels):
        calls.append(Path(path).stem)
        polarization, raw_frame_id = Path(path).stem.split("_")
        value = int(raw_frame_id) + (100 if polarization == "cross" else 0)
        return np.full((2, 1, channels), value, dtype=np.float32)

    monkeypatch.setattr(render_saved_material, "read_image", fake_read)

    capture = (
        render_saved_material._load_polarized_capture_in_acquisition_order(
            sample, frame_ids, height=2, width=1
        )
    )

    assert calls == ["cross_12", "parallel_12", "cross_4", "parallel_4"]
    assert len(capture.cross_images) == 2
    assert len(capture.parallel_images) == 2
    np.testing.assert_array_equal(
        capture.cross_stack[:, 0, 0, 0], [112.0, 104.0]
    )
    np.testing.assert_array_equal(
        capture.parallel_stack[:, 0, 0, 0], [12.0, 4.0]
    )

    calls.clear()
    targets = render_saved_material.load_parallel_targets_in_acquisition_order(
        sample, frame_ids, height=2, width=1
    )

    assert calls == ["cross_12", "parallel_12", "cross_4", "parallel_4"]
    np.testing.assert_array_equal(targets[:, 0, 0, 0], [12.0, 4.0])


def test_replay_target_loader_requires_both_polarizations(tmp_path):
    parallel = tmp_path / "parallel.png"
    parallel.touch()
    sample = SimpleNamespace(
        light_path=lambda polarization, _frame_id: (
            None if polarization == "cross" else parallel
        )
    )

    with pytest.raises(FileNotFoundError, match="missing cross OLAT frame 9"):
        render_saved_material.load_parallel_targets_in_acquisition_order(
            sample, [9], height=2, width=1
        )


def test_raw_parallel_targets_artifact_replays_exact_persisted_tensor(tmp_path):
    targets = np.linspace(0.0, 1.0, 2 * 3 * 4 * 3, dtype=np.float32).reshape(
        2, 3, 4, 3
    )
    path = tmp_path / "evaluation" / "assets" / "raw_parallel_targets.npz"
    path.parent.mkdir(parents=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, raw_parallel_targets=targets)
    record = {
        "schema": render_saved_material.RAW_PARALLEL_TARGETS_ARTIFACT_SCHEMA,
        "path": "evaluation/assets/raw_parallel_targets.npz",
        "format": "numpy_npz_compressed",
        "key": "raw_parallel_targets",
        "dtype": "float32",
        "shape": [2, 3, 4, 3],
        "array_sha256": render_saved_material._array_sha256(targets),
        "file_sha256": render_saved_material._file_sha256(path),
        "bytes": path.stat().st_size,
    }

    loaded, loaded_path = (
        render_saved_material._load_raw_parallel_targets_artifact(
            tmp_path,
            record,
            n_lights=2,
            height=3,
            width=4,
        )
    )

    assert loaded_path == path
    assert loaded.flags.c_contiguous
    np.testing.assert_array_equal(loaded, targets)

    escaping = dict(record, path="../raw_parallel_targets.npz")
    with pytest.raises(ValueError, match="escapes the camera result"):
        render_saved_material._load_raw_parallel_targets_artifact(
            tmp_path,
            escaping,
            n_lights=2,
            height=3,
            width=4,
        )


def test_replay_inputs_artifact_replays_every_persisted_tensor(tmp_path):
    arrays = {
        "raw_parallel_targets": np.zeros((2, 3, 4, 3), dtype=np.float32),
        "capture_foreground": np.ones((3, 4, 1), dtype=np.float32),
        "source_normal": np.full((3, 4, 3), 0.25, dtype=np.float32),
        "normal": np.full((3, 4, 3), 0.5, dtype=np.float32),
        "view_directions": np.full((3, 4, 3), 0.75, dtype=np.float32),
    }
    path = tmp_path / "evaluation" / "assets" / "replay_inputs.npz"
    path.parent.mkdir(parents=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    record = {
        "schema": render_saved_material.REPLAY_INPUTS_ARTIFACT_SCHEMA,
        "path": "evaluation/assets/replay_inputs.npz",
        "format": "numpy_npz_compressed",
        "arrays": {
            name: {
                "dtype": "float32",
                "shape": list(array.shape),
                "array_sha256": render_saved_material._array_sha256(array),
            }
            for name, array in arrays.items()
        },
        "file_sha256": render_saved_material._file_sha256(path),
        "bytes": path.stat().st_size,
    }

    loaded, loaded_path = render_saved_material._load_replay_inputs_artifact(
        tmp_path,
        record,
        n_lights=2,
        height=3,
        width=4,
    )

    assert loaded_path == path
    assert list(loaded) == list(render_saved_material.REPLAY_INPUT_ARRAY_NAMES)
    for name, array in arrays.items():
        assert loaded[name].flags.c_contiguous
        np.testing.assert_array_equal(loaded[name], array)

    tampered = json.loads(json.dumps(record))
    tampered["arrays"]["normal"]["array_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="replay input normal hash differs"):
        render_saved_material._load_replay_inputs_artifact(
            tmp_path,
            tampered,
            n_lights=2,
            height=3,
            width=4,
        )

    escaping = dict(record, path="../replay_inputs.npz")
    with pytest.raises(ValueError, match="escapes the camera result"):
        render_saved_material._load_replay_inputs_artifact(
            tmp_path,
            escaping,
            n_lights=2,
            height=3,
            width=4,
        )


def test_presentation_alpha_preserves_clean_soft_mask_values():
    mask = np.asarray(
        [[[0.0], [0.25]], [[0.75], [1.0]]], dtype=np.float32
    )

    alpha = render_saved_material._presentation_alpha(mask, 2, 2)

    np.testing.assert_array_equal(alpha, mask)
    assert alpha.flags.c_contiguous
    np.testing.assert_array_equal(
        render_saved_material._foreground_mask(mask, 2, 2)[..., 0],
        [[0.0, 0.0], [1.0, 1.0]],
    )


def test_install_staged_directory_replaces_only_after_complete_stage(tmp_path):
    output_dir = tmp_path / ".rendering_stage"
    staging_dir = tmp_path / ".rendering_stage.tmp-123"
    output_dir.mkdir()
    staging_dir.mkdir()
    (output_dir / "marker").write_text("old", encoding="utf-8")
    (staging_dir / "marker").write_text("new", encoding="utf-8")

    render_saved_material._install_staged_directory(
        staging_dir, output_dir, replace_output=True
    )

    assert (output_dir / "marker").read_text(encoding="utf-8") == "new"
    assert not staging_dir.exists()
    assert not list(tmp_path.glob(".rendering_stage.previous-*"))


def test_install_staged_directory_restores_previous_on_swap_failure(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / ".rendering_stage"
    staging_dir = tmp_path / ".rendering_stage.tmp-123"
    output_dir.mkdir()
    staging_dir.mkdir()
    (output_dir / "marker").write_text("old", encoding="utf-8")
    (staging_dir / "marker").write_text("new", encoding="utf-8")
    real_replace = render_saved_material.os.replace

    def fail_new_stage(source, destination):
        if Path(source) == staging_dir:
            raise OSError("synthetic stage swap failure")
        return real_replace(source, destination)

    monkeypatch.setattr(render_saved_material.os, "replace", fail_new_stage)
    with pytest.raises(OSError, match="synthetic stage swap failure"):
        render_saved_material._install_staged_directory(
            staging_dir, output_dir, replace_output=True
        )

    assert (output_dir / "marker").read_text(encoding="utf-8") == "old"
    assert (staging_dir / "marker").read_text(encoding="utf-8") == "new"
    assert not list(tmp_path.glob(".rendering_stage.previous-*"))


def test_require_slurm_cuda_rejects_local_session(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    fake_torch = SimpleNamespace(cuda=SimpleNamespace())

    with pytest.raises(RuntimeError, match="Slurm-only"):
        render_saved_material.require_slurm_cuda(fake_torch)


def test_compare_rendered_png_requires_decoded_pixel_exactness(tmp_path):
    exact = np.asarray([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    changed = np.asarray([[[0.0, 0.5 + 1.0 / 255.0, 1.0]]], dtype=np.float32)
    canonical_path = tmp_path / "canonical.png"
    exact_path = tmp_path / "exact.png"
    changed_path = tmp_path / "changed.png"
    write_image(canonical_path, exact)
    write_image(exact_path, exact)
    write_image(changed_path, changed)

    exact_result = render_saved_material._compare_rendered_png(
        exact_path, canonical_path, tolerance=0.0
    )
    changed_result = render_saved_material._compare_rendered_png(
        changed_path, canonical_path, tolerance=0.0
    )

    assert exact_result["passed"] is True
    assert exact_result["different_channel_values"] == 0
    assert changed_result["passed"] is False
    assert changed_result["different_channel_values"] == 1


def test_sample_sd_environment_matches_truncating_nearest_pixel_contract():
    torch = pytest.importorskip("torch")
    environment = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    directions = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    samples = render_saved_material.sample_sd_environment(
        torch, environment, directions
    )

    torch.testing.assert_close(samples[0], environment[0, 1])
    torch.testing.assert_close(samples[1], environment[0, 2])
    torch.testing.assert_close(samples[2], environment[0, 1])
    torch.testing.assert_close(samples[3], environment[0, 1])


def test_prepare_sd_fit_support_preset_is_calibration_then_exact_rank_order(
    monkeypatch, tmp_path
):
    torch = pytest.importorskip("torch")
    root = Path("/synthetic/hdr")
    paths = [root / name for name in render_saved_material.SD_OLAT_EXPECTED_TOP32]
    c04_path = Path(
        "/synthetic/imaginaire/OLATPipeClean/"
        "LightsLocationsRelativetoCamera/C04.txt"
    )
    monkeypatch.setattr(
        render_saved_material, "discover_environment_maps", lambda _root: paths
    )
    monkeypatch.setattr(
        render_saved_material,
        "load_sd_c04_directions",
        lambda _torch, _path, device: torch.cat(
            [
                torch.tensor([[1.0, 0.0, 0.0]]).repeat(155, 1),
                torch.zeros((9, 3)),
            ],
            dim=0,
        ),
    )
    monkeypatch.setattr(
        render_saved_material,
        "rank_sd_environment_maps",
        lambda _torch, candidates, _directions, count, device: [
            (path, float(1000 - rank))
            for rank, path in enumerate(candidates[:count])
        ],
    )
    monkeypatch.setattr(
        render_saved_material,
        "_read_environment",
        lambda _path: np.ones((2, 4, 3), dtype=np.float32),
    )

    def fake_hash(path):
        if Path(path) == c04_path:
            return render_saved_material.SD_C04_LIGHTS_SHA256
        return "a" * 64

    monkeypatch.setattr(render_saved_material, "_file_sha256", fake_hash)
    fit_support = np.asarray(
        [1, 2, 4, 5, 7, 8, 10, 11, 12, 14, 15, 17, 18, 20, 21, 23,
         24, 25, 27, 28, 30, 31, 33, 34, 36, 37, 38, 40, 41, 43, 44, 46,
         47, 48],
        dtype=np.int64,
    )
    angles = np.linspace(0.0, 2.0 * np.pi, 50, endpoint=False)
    z = np.linspace(-0.9, 0.9, 50)
    radius = np.sqrt(1.0 - z * z)
    light_directions = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles), z], axis=1
    ).astype(np.float32)
    acquisition = {
        "fit_conditions": {"olat": 34},
        "light_split": {
            "fit_stack_indices": fit_support.tolist(),
            "selection": {
                "fit_count": 34,
                "mode": "holdout_every_third_available_light",
            },
        },
    }
    acquisition_path = tmp_path / "material" / "olat" / "acquisition.json"
    acquisition_path.parent.mkdir(parents=True)
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    support_sha256 = render_saved_material._array_sha256(fit_support)
    replay = render_saved_material.ReplayInputs(
        acquisition=acquisition,
        acquisition_path=acquisition_path,
        model_path=acquisition_path.with_name("disney_brdf.pt"),
        height=2,
        width=4,
        light_ids=np.arange(50, dtype=np.int64),
        frame_ids=np.arange(50, dtype=np.int64),
        parallel_targets=np.zeros((50, 3, 2, 4), dtype=np.float32),
        light_directions=light_directions,
        view_directions=np.zeros((2, 4, 3), dtype=np.float32),
        fit_foreground=np.ones((2, 4, 1), dtype=np.float32),
        presentation_alpha=np.ones((2, 4, 1), dtype=np.float32),
        presentation_mask_path=Path("mask.png"),
        hash_checks={
            "fit_support_indices": {
                "actual_sha256": support_sha256,
                "expected_sha256": support_sha256,
                "passed": True,
            }
        },
    )
    lighting = render_saved_material.RecordedLighting(
        manifest={},
        conditions=(),
        condition_ids=np.asarray([]),
        splits=np.asarray([]),
        weights=np.zeros((0, 50, 3), dtype=np.float32),
        fit_support_indices=fit_support,
        evaluation_support_indices=np.setdiff1d(np.arange(50), fit_support),
    )

    preset = render_saved_material.prepare_sd_renderings_preset(
        torch=torch,
        replay=replay,
        lighting=lighting,
        hdri_root=root,
        c04_lights_path=c04_path,
        device="synthetic",
        projection_height=2,
        lighting_preset=render_saved_material.SD_RENDERINGS_FIT_SUPPORT_PRESET,
    )

    assert len(preset.conditions) == 36
    assert [entry.record["source"] for entry in preset.conditions[:4]] == [
        "calibration_w",
        "calibration_r",
        "calibration_g",
        "calibration_b",
    ]
    assert [entry.record["source"] for entry in preset.conditions[4:]] == list(
        render_saved_material.SD_OLAT_EXPECTED_TOP32
    )
    assert [entry.record["source_rank"] for entry in preset.conditions[4:]] == list(
        range(1, 33)
    )
    assert all(entry.weights.shape == (50, 3) for entry in preset.conditions)
    assert preset.provenance["schema"] == (
        render_saved_material.SD_RENDERINGS_FIT_SUPPORT_SCHEMA
    )
    assert preset.provenance["historical_lighting_exact"] is False
    assert preset.provenance["projection"]["support_count"] == 34
    assert preset.provenance["available_fit_support"]["fit_count"] == 34
    assert preset.provenance["orientation"]["horizontal_shift_fraction"] == 0.5

    with pytest.raises(ValueError, match="historical 164-light fit support"):
        render_saved_material.prepare_sd_renderings_preset(
            torch=torch,
            replay=replay,
            lighting=lighting,
            hdri_root=root,
            c04_lights_path=c04_path,
            device="synthetic",
            projection_height=2,
            lighting_preset=render_saved_material.SD_RENDERINGS_PRESET,
        )

    full_support = np.arange(164, dtype=np.int64)
    full_acquisition = {
        "fit_conditions": {"olat": 164},
        "light_split": {
            "fit_stack_indices": full_support.tolist(),
            "selection": {
                "fit_count": 164,
                "mode": "lsx_visible_hemisphere_164",
            },
        },
    }
    full_acquisition_path = tmp_path / "full" / "acquisition.json"
    full_acquisition_path.parent.mkdir()
    full_acquisition_path.write_text(
        json.dumps(full_acquisition), encoding="utf-8"
    )
    full_angles = np.linspace(0.0, 2.0 * np.pi, 164, endpoint=False)
    full_z = np.linspace(-0.99, 0.99, 164)
    full_radius = np.sqrt(1.0 - full_z * full_z)
    full_directions = np.stack(
        [
            full_radius * np.cos(full_angles),
            full_radius * np.sin(full_angles),
            full_z,
        ],
        axis=1,
    ).astype(np.float32)
    full_support_sha256 = render_saved_material._array_sha256(full_support)
    full_replay = SimpleNamespace(
        acquisition=full_acquisition,
        acquisition_path=full_acquisition_path,
        model_path=full_acquisition_path.with_name("disney_brdf.pt"),
        light_directions=full_directions,
        hash_checks={
            "fit_support_indices": {
                "actual_sha256": full_support_sha256,
                "expected_sha256": full_support_sha256,
                "passed": True,
            }
        },
    )
    full_lighting = render_saved_material.RecordedLighting(
        manifest={},
        conditions=(),
        condition_ids=np.asarray([]),
        splits=np.asarray([]),
        weights=np.zeros((0, 164, 3), dtype=np.float32),
        fit_support_indices=full_support,
        evaluation_support_indices=np.asarray([0], dtype=np.int64),
    )
    strict_preset = render_saved_material.prepare_sd_renderings_preset(
        torch=torch,
        replay=full_replay,
        lighting=full_lighting,
        hdri_root=root,
        c04_lights_path=c04_path,
        device="synthetic",
        projection_height=2,
        lighting_preset=render_saved_material.SD_RENDERINGS_PRESET,
    )
    assert strict_preset.provenance["schema"] == (
        render_saved_material.SD_RENDERINGS_SCHEMA
    )
    assert strict_preset.provenance["projection"]["support_count"] == 164
    assert "historical_lighting_exact" not in strict_preset.provenance


def test_historical_probe_parameters_match_exact_sd_constraints():
    torch = pytest.importorskip("torch")
    normal = torch.randn(2, 3, 3)
    base = torch.rand(2, 3, 3)
    scalar_names = (
        "metallic",
        "subsurface",
        "specular",
        "roughness",
        "specularTint",
        "anisotropic",
        "sheen",
        "sheenTint",
        "clearcoat",
        "clearcoatGloss",
    )
    maps = {"normal": normal, "baseColor": base}
    maps.update({name: torch.rand(2, 3) for name in scalar_names})
    model = SimpleNamespace(_param_maps=lambda: maps)

    grey = render_saved_material._historical_probe_parameters(
        torch, model, "greyball"
    )
    chrome = render_saved_material._historical_probe_parameters(
        torch, model, "chromeball"
    )

    assert grey["normal"] is normal
    assert chrome["normal"] is normal
    torch.testing.assert_close(grey["baseColor"], torch.ones_like(base))
    torch.testing.assert_close(chrome["baseColor"], torch.zeros_like(base))
    expected_grey = {"roughness": 1.0}
    expected_chrome = {"metallic": 0.8, "specular": 1.0, "roughness": 0.2}
    for name in scalar_names:
        torch.testing.assert_close(
            grey[name], torch.full_like(maps[name], expected_grey.get(name, 0.0))
        )
        torch.testing.assert_close(
            chrome[name],
            torch.full_like(maps[name], expected_chrome.get(name, 0.0)),
        )


def test_presentation_parameters_face_forward_only_negative_visible_normals():
    torch = pytest.importorskip("torch")
    normal = torch.tensor(
        [[[0.6, 0.0, -0.8], [0.0, 0.0, 1.0]]], dtype=torch.float32
    )
    model = SimpleNamespace(_param_maps=lambda: {"normal": normal})
    views = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], dtype=torch.float32
    )
    alpha = torch.tensor([[[1.0], [0.5]]], dtype=torch.float32)

    parameters, count = (
        render_saved_material._face_forward_presentation_parameters(
            torch, model, views, alpha
        )
    )

    torch.testing.assert_close(
        parameters["normal"],
        torch.tensor(
            [[[0.6, 0.0, 0.8], [0.0, 0.0, 1.0]]], dtype=torch.float32
        ),
    )
    assert count == 1


def test_render_condition_does_not_square_disney_soft_alpha():
    torch = pytest.importorskip("torch")
    alpha = torch.tensor([[[0.5], [1.0]]], dtype=torch.float32)

    class AlreadyMaskedModel:
        def __call__(self, **arguments):
            masked = arguments["mask"].permute(2, 0, 1).repeat(3, 1, 1)
            return masked, None, None

    rendered = render_saved_material._render_condition(
        torch=torch,
        model=AlreadyMaskedModel(),
        views=torch.zeros((1, 2, 3), dtype=torch.float32),
        lights=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        foreground=alpha,
        weights=np.ones((1, 3), dtype=np.float32),
        support_indices=np.asarray([0], dtype=np.int64),
        light_chunk=None,
    )

    np.testing.assert_array_equal(rendered[:, 0], np.full((1, 3), 0.5))
    np.testing.assert_array_equal(rendered[:, 1], np.ones((1, 3)))


def test_measured_gt_is_supported_weighted_sum_soft_mask_once_then_global_p995():
    torch = pytest.importorskip("torch")
    targets = torch.tensor(
        [
            [[[1.0, 4.0]], [[2.0, 5.0]], [[3.0, 6.0]]],
            [[[2.0, 8.0]], [[4.0, 10.0]], [[6.0, 12.0]]],
        ],
        dtype=torch.float32,
    )
    foreground = torch.tensor([[[1.0], [0.5]]], dtype=torch.float32)
    weights = np.asarray([[99.0, 99.0, 99.0], [1.0, 0.5, 0.25]], dtype=np.float32)

    actual = render_saved_material._render_measured_gt(
        torch=torch,
        parallel_targets=targets,
        presentation_alpha=foreground,
        weights=weights,
        support_indices=np.asarray([1], dtype=np.int64),
    )

    raw = targets[1] * torch.tensor([1.0, 0.5, 0.25]).reshape(3, 1, 1)
    mask = foreground.permute(2, 0, 1)
    expected = render_saved_material._normalize_presentation_render(raw, mask)
    expected = expected.permute(1, 2, 0)
    np.testing.assert_allclose(actual, expected.numpy(), rtol=0.0, atol=1e-7)


def test_presentation_gt_normalizer_does_not_square_soft_alpha():
    torch = pytest.importorskip("torch")
    render = torch.ones((3, 1, 2), dtype=torch.float32)
    alpha = torch.tensor([[[1.0, 0.5]]], dtype=torch.float32)

    normalized = render_saved_material._normalize_presentation_render(
        render, alpha
    )

    torch.testing.assert_close(normalized[:, :, 0], torch.ones((3, 1)))
    torch.testing.assert_close(
        normalized[:, :, 1], torch.full((3, 1), 0.5)
    )


def test_material_map_stage_has_exact_order_native_rgb_and_paths(tmp_path):
    torch = pytest.importorskip("torch")
    display = {
        name: torch.full((2, 3, 3), index, dtype=torch.uint8)
        for index, name in enumerate(render_saved_material.SD_SHEET_MAP_ORDER)
    }
    model = SimpleNamespace(to_display_maps=lambda gamma: display)
    camera_dir = tmp_path / "object" / "cam07"
    staging_dir = camera_dir / ".rendering_stage.tmp"
    output_dir = camera_dir / ".rendering_stage"

    records = render_saved_material._write_material_maps(
        model=model,
        staging_dir=staging_dir,
        output_dir=output_dir,
        camera_dir=camera_dir,
        height=2,
        width=3,
    )

    assert [record["name"] for record in records] == list(
        render_saved_material.SD_SHEET_MAP_ORDER
    )
    assert records[0]["render_path"] == (
        ".rendering_stage/material_maps/normal.png"
    )
    for record in records:
        path = staging_dir / "material_maps" / f"{record['name']}.png"
        assert path.is_file()
        assert record["render_sha256"] == render_saved_material._file_sha256(path)
        assert render_saved_material.read_image(path).shape == (2, 3, 3)


def test_sheet_contract_is_exact_historical_12_by_13_layout():
    raw_check = {
        "expected_sha256": "a" * 64,
        "actual_sha256": "a" * 64,
        "passed": True,
    }

    contract = render_saved_material.build_sheet_contract(raw_check)

    assert contract["reference_sha256"] == render_saved_material.SD_SHEET_REFERENCE_SHA256
    assert contract["historical_commit"] == render_saved_material.SD_SHEET_HISTORICAL_COMMIT
    assert (contract["columns"], contract["rows"], contract["tile_size"]) == (
        12,
        13,
        768,
    )
    assert contract["map_order"] == list(render_saved_material.SD_SHEET_MAP_ORDER)
    assert contract["panel_order"] == ["gt", "pred", "greyball", "chromeball"]
    assert contract["labels"]["templates"]["pred"] == "pred #{label_number}"
    assert contract["raw_parallel_targets"] == raw_check


def test_cli_lighting_source_is_mutually_exclusive():
    parser = render_saved_material.build_parser()
    common = [
        "--camera-dir",
        "/camera",
        "--data-root",
        "/data",
        "--object",
        "object",
        "--camera",
        "cam07",
        "--imaginaire-root",
        "/imaginaire",
        "--output-dir",
        "/camera/.rendering_stage",
    ]

    preset = parser.parse_args([*common, "--lighting-preset", "sd-renderings"])
    assert preset.lighting_preset == "sd-renderings"
    assert preset.condition_indices is None
    fit_support = parser.parse_args(
        [*common, "--lighting-preset", "sd-renderings-fit-support"]
    )
    assert fit_support.lighting_preset == "sd-renderings-fit-support"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                *common,
                "--lighting-preset",
                "sd-renderings",
                "--condition-indices",
                "1:4",
            ]
        )
