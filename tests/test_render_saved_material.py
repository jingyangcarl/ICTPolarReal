from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ictpolarreal.processing import render_saved_material
from ictpolarreal.utils.io import write_image


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


def test_prepare_sd_heads_preset_is_calibration_then_exact_rank_order(monkeypatch):
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
    replay = render_saved_material.ReplayInputs(
        acquisition={},
        model_path=Path("model.pt"),
        height=2,
        width=4,
        light_ids=np.arange(4, dtype=np.int64),
        light_directions=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        view_directions=np.zeros((2, 4, 3), dtype=np.float32),
        foreground=np.ones((2, 4, 1), dtype=np.float32),
        hash_checks={},
    )
    lighting = render_saved_material.RecordedLighting(
        manifest={},
        conditions=(),
        condition_ids=np.asarray([]),
        splits=np.asarray([]),
        weights=np.zeros((0, 4, 3), dtype=np.float32),
        fit_support_indices=np.arange(4, dtype=np.int64),
        evaluation_support_indices=np.arange(4, dtype=np.int64),
    )

    preset = render_saved_material.prepare_sd_olat_heads_preset(
        torch=torch,
        replay=replay,
        lighting=lighting,
        hdri_root=root,
        c04_lights_path=c04_path,
        device="synthetic",
        projection_height=2,
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
    assert all(entry.weights.shape == (4, 3) for entry in preset.conditions)
    assert preset.provenance["schema"] == render_saved_material.SD_OLAT_HEADS_SCHEMA
    assert preset.provenance["orientation"]["horizontal_shift_fraction"] == 0.5


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

    preset = parser.parse_args([*common, "--lighting-preset", "sd-olat-heads"])
    assert preset.lighting_preset == "sd-olat-heads"
    assert preset.condition_indices is None
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                *common,
                "--lighting-preset",
                "sd-olat-heads",
                "--condition-indices",
                "1:4",
            ]
        )
