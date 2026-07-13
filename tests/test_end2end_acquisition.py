from __future__ import annotations

import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ictpolarreal.data.dataset import CameraSample
from ictpolarreal.processing import (
    end2end_acquisition,
    lighting_profiles,
    material_decomposition,
    prepare_materials,
)
from ictpolarreal.utils.io import read_image, write_image


def test_split_light_indices_reserves_sphere_spread_holdout():
    train, heldout = end2end_acquisition.split_light_indices(346, 16)

    np.testing.assert_array_equal(heldout, np.arange(0, 346, 23))
    assert train.dtype == np.int64
    assert heldout.dtype == np.int64
    assert len(train) == 330
    assert len(heldout) == 16
    assert not np.intersect1d(train, heldout).size
    np.testing.assert_array_equal(
        np.sort(np.concatenate([train, heldout])), np.arange(346)
    )


def test_split_light_indices_keeps_minimum_training_set():
    train, heldout = end2end_acquisition.split_light_indices(4, 16)

    np.testing.assert_array_equal(train, np.arange(4))
    assert heldout.dtype == np.int64
    assert heldout.size == 0


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("olat,hdri,mix", ("olat", "hdri", "mix")),
        (" MIX, olat, mix ", ("olat", "mix")),
        (["HDRI", "olat"], ("olat", "hdri")),
        ("all", ("olat", "hdri", "mix")),
    ],
)
def test_parse_lighting_profiles_normalizes_to_canonical_order(requested, expected):
    assert lighting_profiles.parse_lighting_profiles(requested) == expected


@pytest.mark.parametrize("requested", ["", [], "olat,unknown", "all,mix"])
def test_parse_lighting_profiles_rejects_invalid_requests(requested):
    with pytest.raises(ValueError):
        lighting_profiles.parse_lighting_profiles(requested)


def test_mix_profile_uses_four_hdri_then_four_olat_iterations():
    kinds = [lighting_profiles.mix_condition_kind(step, rotations=4) for step in range(16)]

    assert kinds == ["hdri"] * 4 + ["olat"] * 4 + ["hdri"] * 4 + ["olat"] * 4


def test_hdri_sources_remain_grouped_across_rotations_and_project_to_olat_basis(
    monkeypatch, tmp_path
):
    hdri_root = tmp_path / "hdris"
    hdri_root.mkdir()
    for index in range(3):
        (hdri_root / f"studio_{index}.hdr").write_bytes(f"studio {index}".encode())

    def fake_read_environment(path):
        index = int(Path(path).stem.rsplit("_", 1)[1]) + 1
        rows = np.linspace(0.25, 1.0, 4, dtype=np.float32)[:, None]
        columns = np.linspace(0.5, 1.5, 8, dtype=np.float32)[None, :]
        luminance = index * rows * columns
        return np.stack(
            [luminance, luminance * 0.75 + 0.1, luminance * 0.5 + 0.2], axis=-1
        )

    monkeypatch.setattr(lighting_profiles, "_read_environment", fake_read_environment)
    light_dirs = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    conditions = lighting_profiles.prepare_environment_conditions(
        hdri_root,
        light_dirs,
        train_count=2,
        eval_count=1,
        rotations=4,
        projection_height=4,
        out_dir=tmp_path / "lighting",
        fit_support_indices=np.asarray([0, 2, 3], dtype=np.int64),
        evaluation_support_indices=np.asarray([1], dtype=np.int64),
    )

    assert len(conditions.train) == 24
    assert len(conditions.evaluation) == 4
    grouped = {}
    natural_conditions = [
        condition
        for condition in conditions.all
        if condition.source_kind == "environment_map"
    ]
    calibration_conditions = [
        condition
        for condition in conditions.all
        if condition.source_kind == "generated_calibration"
    ]
    assert len(natural_conditions) == 12
    assert len(calibration_conditions) == 16
    assert {condition.split for condition in calibration_conditions} == {"fit"}
    assert {condition.source_name for condition in calibration_conditions} == {
        "calibration_w",
        "calibration_r",
        "calibration_g",
        "calibration_b",
    }
    for condition in natural_conditions:
        grouped.setdefault(condition.source_name, []).append(condition)
    for condition in conditions.all:
        assert condition.weights.shape == (len(light_dirs), 3)
        assert np.isfinite(condition.weights).all()
        assert np.all(condition.weights >= 0.0)
        assert condition.weights.sum() > 0.0
        if condition.split == "fit":
            np.testing.assert_array_equal(condition.weights[1], 0.0)
        else:
            np.testing.assert_array_equal(condition.weights[[0, 2, 3]], 0.0)
    assert set(grouped) == {f"studio_{index}.hdr" for index in range(3)}
    for source_conditions in grouped.values():
        assert len(source_conditions) == 4
        assert len({condition.split for condition in source_conditions}) == 1
        assert {condition.rotation_degrees for condition in source_conditions} == {
            0,
            90,
            180,
            270,
        }

    archive = np.load(tmp_path / "lighting" / "weights.npz")
    assert archive["weights"].shape == (28, 4, 3)
    assert list(archive["splits"]).count("fit") == 24
    assert list(archive["splits"]).count("heldout") == 4
    manifest = json.loads((tmp_path / "lighting" / "conditions.json").read_text())
    assert manifest["target_origin"] == "synthesized_from_measured_olat"
    assert manifest["natural_fit_conditions"] == 8
    assert manifest["calibration_fit_conditions"] == 16
    assert manifest["projection"]["fit_support_indices"] == [0, 2, 3]
    assert manifest["projection"]["evaluation_support_indices"] == [1]
    assert len(manifest["conditions"]) == 28


def test_synthesized_hdri_targets_use_rgb_weights_and_requested_olat_support(tmp_path):
    torch = pytest.importorskip("torch")
    raw_targets = torch.arange(1, 19, dtype=torch.float32).reshape(3, 3, 1, 2)
    weights = np.asarray(
        [
            [[1.0, 0.5, 0.25], [999.0, 999.0, 999.0], [0.25, 0.5, 1.0]],
            [[0.1, 0.2, 0.3], [999.0, 999.0, 999.0], [0.4, 0.5, 0.6]],
        ],
        dtype=np.float32,
    )
    conditions = [
        lighting_profiles.EnvironmentCondition(
            condition_id=f"studio_rot{index * 90:03d}",
            source_path=tmp_path / "studio.hdr",
            source_name="studio.hdr",
            source_sha256="digest",
            rotation_degrees=index * 90,
            split="fit",
            variance_score=1.0,
            weights=condition_weights,
            preview=np.zeros((2, 4, 3), dtype=np.float32),
        )
        for index, condition_weights in enumerate(weights)
    ]
    support = np.asarray([0, 2], dtype=np.int64)

    synthesized = end2end_acquisition._synthesize_environment_targets(
        torch,
        raw_targets,
        conditions,
        support,
        np.ones((1, 2, 1), dtype=np.float32),
        torch.device("cpu"),
        storage_dtype=torch.float32,
    )

    weighted = torch.einsum(
        "bnc,nchw->bchw",
        torch.from_numpy(weights[:, support]),
        raw_targets.index_select(0, torch.from_numpy(support)),
    )
    expected = torch.stack(
        [
            (target / target.quantile(end2end_acquisition.PERCENTILE / 100.0)).clamp_max(1.0)
            for target in weighted
        ]
    )
    assert synthesized.dtype == torch.float32
    assert torch.allclose(synthesized, expected)


def test_end2end_initialization_and_optimizer_exclude_same_heldout_lights(
    monkeypatch, tmp_path
):
    camera_dir = tmp_path / "object" / "cam00"
    for kind in ("cross", "parallel"):
        directory = camera_dir / kind
        directory.mkdir(parents=True)
        for frame_id in range(6):
            (directory / f"{frame_id:06d}.png").touch()
    sample = CameraSample("object", "cam00", camera_dir)

    def fake_read(path, **_kwargs):
        return np.full((2, 2, 3), int(Path(path).stem), dtype=np.float32)

    def fake_lights(_root, indices, **_kwargs):
        indices = np.asarray(indices, dtype=np.float32)
        return np.stack([indices, np.zeros_like(indices), np.ones_like(indices)], axis=-1)

    initialization = {}

    def fake_decompose(cross, parallel, lights, **_kwargs):
        initialization["cross"] = cross.copy()
        initialization["parallel"] = parallel.copy()
        initialization["lights"] = lights.copy()
        image = np.zeros((2, 2, 3), dtype=np.float32)
        image[..., 2] = 1.0
        return SimpleNamespace(diffuse_albedo=image, diffuse_normal=image)

    acquisition = {}

    def fake_acquire(cross, parallel, lights, **kwargs):
        acquisition["cross"] = cross.copy()
        acquisition["parallel"] = parallel.copy()
        acquisition["lights"] = lights.copy()
        acquisition.update(kwargs)

    monkeypatch.setattr(material_decomposition, "read_image", fake_read)
    monkeypatch.setattr(material_decomposition, "load_light_directions", fake_lights)
    monkeypatch.setattr(
        material_decomposition,
        "load_view_directions",
        lambda _root, _sample, hw: np.broadcast_to(
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32), hw + (3,)
        ).copy(),
    )
    monkeypatch.setattr(
        material_decomposition, "decompose_polarized_olat", fake_decompose
    )
    monkeypatch.setattr(end2end_acquisition, "validate_end2end_runtime", lambda *_args: None)
    monkeypatch.setattr(end2end_acquisition, "acquire_disney_material", fake_acquire)

    used = material_decomposition.decompose_camera_sample(
        sample,
        data_root=tmp_path,
        out_root=tmp_path / "out",
        light_start=0,
        max_lights=None,
        light_root=None,
        backend="torch",
        device="cuda",
        noise=0.0,
        material_acquisition="end2end",
        imaginaire_root=tmp_path,
        end2end_steps=1,
        end2end_eval_lights=2,
    )

    assert used == 6
    np.testing.assert_array_equal(initialization["cross"][:, 0, 0, 0], [1, 2, 3, 4])
    np.testing.assert_array_equal(initialization["parallel"][:, 0, 0, 0], [1, 2, 3, 4])
    np.testing.assert_array_equal(acquisition["cross"][:, 0, 0, 0], np.arange(6))
    np.testing.assert_array_equal(acquisition["light_ids"], np.arange(6))
    np.testing.assert_array_equal(acquisition["frame_ids"], np.arange(6))
    assert acquisition["eval_lights"] == 2


def test_render_normalization_uses_foreground_and_masks_background():
    torch = pytest.importorskip("torch")
    render = torch.zeros((3, 2, 2), dtype=torch.float32)
    render[:, 0, 0] = 2.0
    render[:, 1, 1] = 1000.0
    mask = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])

    normalized = end2end_acquisition._normalize_render_foreground(render, mask)

    assert torch.allclose(normalized[:, 0, 0], torch.ones(3))
    assert torch.count_nonzero(normalized[:, 1, 1]) == 0


def test_safe_torch_quantile_strides_below_cuda_element_limit(monkeypatch):
    torch = pytest.importorskip("torch")
    values = torch.arange(100, dtype=torch.float32)
    monkeypatch.setattr(end2end_acquisition, "MAX_QUANTILE_ELEMENTS", 10)

    result = end2end_acquisition._safe_torch_quantile(values, 0.5)

    assert result == torch.quantile(values[::10], 0.5)


def test_hdri_gpu_preflight_rejects_low_memory_device():
    low_memory_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            get_device_properties=lambda _device: SimpleNamespace(
                name="test-gpu",
                total_memory=24 * (1 << 30),
            )
        )
    )

    with pytest.raises(RuntimeError, match="high-memory GPU"):
        end2end_acquisition._validate_hdri_gpu_memory(
            low_memory_torch,
            "cuda",
            n_lights=330,
            height=512,
            width=282,
        )


def test_disney_scalar_defaults_are_initialized_in_physical_space():
    torch = pytest.importorskip("torch")
    model = SimpleNamespace(
        **{
            f"{name}_un": torch.nn.Parameter(torch.zeros(1))
            for name in end2end_acquisition.DISNEY_PHYSICAL_DEFAULTS
        }
    )

    end2end_acquisition._initialize_disney_scalars(torch, model)

    for name, physical_value in end2end_acquisition.DISNEY_PHYSICAL_DEFAULTS.items():
        expected = min(
            max(physical_value, end2end_acquisition.SCALAR_INIT_EPS),
            1.0 - end2end_acquisition.SCALAR_INIT_EPS,
        )
        actual = float(torch.sigmoid(getattr(model, f"{name}_un")).item())
        assert actual == pytest.approx(expected, abs=1e-7)


def test_read_image_scales_integer_images_but_preserves_float_hdr(monkeypatch):
    imageio = pytest.importorskip("imageio.v3")
    integer_image = np.array([[[0, 32768, 65535]]], dtype=np.uint16)
    float_hdr = np.array([[[0.25, 1.0, 4.5]]], dtype=np.float32)

    monkeypatch.setattr(imageio, "imread", lambda _path: integer_image)
    integer_result = read_image("integer.exr")
    np.testing.assert_allclose(
        integer_result,
        integer_image.astype(np.float32) / np.iinfo(np.uint16).max,
    )

    monkeypatch.setattr(imageio, "imread", lambda _path: float_hdr)
    float_result = read_image("float.exr")
    np.testing.assert_array_equal(float_result, float_hdr)
    assert float_result.max() > 1.0


def test_end2end_targets_are_normalized_per_light_inside_foreground():
    targets = np.array(
        [
            [
                [[1.0, 1.0, 1.0], [1000.0, 1000.0, 1000.0]],
                [[2.0, 2.0, 2.0], [1000.0, 1000.0, 1000.0]],
            ],
            [
                [[10.0, 10.0, 10.0], [1000.0, 1000.0, 1000.0]],
                [[20.0, 20.0, 20.0], [1000.0, 1000.0, 1000.0]],
            ],
        ],
        dtype=np.float32,
    )
    foreground = np.array([[[1.0], [0.0]], [[1.0], [0.0]]], dtype=np.float32)

    normalized = end2end_acquisition._normalize_targets(targets, foreground)

    assert normalized.dtype == np.float32
    assert np.isfinite(normalized).all()
    np.testing.assert_allclose(normalized[:, 0, 0], 0.5)
    np.testing.assert_allclose(normalized[:, 1, 0], 1.0)
    # The bright background is excluded while estimating each light's scale.
    np.testing.assert_allclose(normalized[:, :, 1], 1.0)


def test_end2end_material_map_outputs_include_legacy_and_disney_aliases(
    monkeypatch, tmp_path
):
    foreground = np.array([[[1.0], [0.0]], [[1.0], [0.0]]], dtype=np.float32)
    maps = {
        "baseColor": np.full((2, 2, 3), 0.4, dtype=np.float32),
        "normal": np.zeros((2, 2, 3), dtype=np.float32),
        "specular": np.full((2, 2, 1), 0.2, dtype=np.float32),
        "roughness": np.full((2, 2, 1), 0.3, dtype=np.float32),
        "metallic": np.full((2, 2, 1), 0.1, dtype=np.float32),
        "specularTint": np.full((2, 2, 1), 0.25, dtype=np.float32),
        "subsurface": np.full((2, 2, 1), 0.05, dtype=np.float32),
        "anisotropic": np.full((2, 2, 1), 0.6, dtype=np.float32),
        "clearcoat": np.full((2, 2, 1), 0.7, dtype=np.float32),
        "clearcoatGloss": np.full((2, 2, 1), 0.8, dtype=np.float32),
    }
    written = {}

    def capture(path, image):
        written[Path(path).name] = np.asarray(image).copy()

    monkeypatch.setattr(end2end_acquisition, "write_image", capture)
    end2end_acquisition._write_material_maps(tmp_path, maps, foreground)

    assert set(written) == {
        "albedo.png",
        "anisotropic.png",
        "baseColor.png",
        "clearcoat.png",
        "clearcoatGloss.png",
        "metallic.png",
        "normal.png",
        "roughness.png",
        "specular.png",
        "specularTint.png",
        "subsurface.png",
    }
    np.testing.assert_array_equal(written["albedo.png"], written["baseColor.png"])
    np.testing.assert_allclose(written["normal.png"][:, 0], 0.5)
    np.testing.assert_allclose(written["normal.png"][:, 1], 0.0)


def test_write_olat_evaluation_writes_profile_scoped_artifacts_and_metrics(tmp_path):
    torch = pytest.importorskip("torch")
    target_chw = torch.stack(
        [torch.full((3, 2, 2), value) for value in (0.2, 0.3, 0.4)]
    )
    evaluation_indices = np.asarray([0, 2], dtype=np.int64)
    light_ids = np.asarray([10, 11, 12], dtype=np.int64)
    frame_ids = np.asarray([2, 3, 4], dtype=np.int64)
    foreground = np.asarray(
        [[[1.0], [1.0]], [[1.0], [0.0]]], dtype=np.float32
    )
    rendered = []

    def fake_renderer(stack_index):
        rendered.append(stack_index)
        return target_chw[stack_index] + 0.1

    evaluation_dir = (
        tmp_path
        / "olat"
        / end2end_acquisition.PROFILE_MODEL
        / "evaluation"
        / "olat"
    )
    cases_dir = evaluation_dir / "cases"
    stale_dir = cases_dir / "999999"
    stale_dir.mkdir(parents=True)
    (stale_dir / "stale.png").touch()
    summary, losses = end2end_acquisition._write_relighting_evaluation(
        fake_renderer,
        target_chw,
        evaluation_indices,
        light_ids,
        frame_ids,
        foreground,
        cases_dir,
        split="heldout_olat",
    )

    assert rendered == [0, 2]
    assert not stale_dir.exists()
    np.testing.assert_allclose(losses, [0.01, 0.01], rtol=1e-5)
    assert summary["split"] == "heldout_olat"
    assert summary["count"] == 2
    assert summary["frame_ids"] == [2, 4]
    assert summary["light_indices"] == [10, 12]
    assert summary["metrics"]["mse"] == pytest.approx(0.01)
    assert summary["metrics"]["mae"] == pytest.approx(0.1)
    assert summary["metrics"]["psnr"] == pytest.approx(20.0)
    assert 0.0 < summary["metrics"]["ssim_global"] < 1.0

    for frame_id in (2, 4):
        frame_dir = cases_dir / f"{frame_id:06d}"
        assert {path.name for path in frame_dir.iterdir()} == {
            "pred.png",
            "gt.png",
            "error.png",
            "comparison.png",
        }
    assert (evaluation_dir / "contact_sheet.png").is_file()

    metrics_path = evaluation_dir / "metrics.csv"
    summary_path = evaluation_dir / "summary.json"
    assert metrics_path.is_file()
    assert summary_path.is_file()
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["split"] for row in rows] == ["heldout_olat", "heldout_olat"]
    assert [int(row["frame_id"]) for row in rows] == [2, 4]
    assert [float(row["mse"]) for row in rows] == pytest.approx([0.01, 0.01])
    on_disk_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert on_disk_summary == summary
    assert summary["representative"]["gt_path"].startswith("cases/")


def test_camera_report_writes_three_by_two_profile_evaluation_matrix(tmp_path):
    profiles = ("olat", "hdri", "mix")
    profile_results = {}
    image = np.full((6, 10, 3), 0.4, dtype=np.float32)
    metric_names = {"mse": 0.01, "mae": 0.1, "psnr": 20.0, "ssim_global": 0.8}

    for profile_index, profile in enumerate(profiles):
        profile_dir = tmp_path / profile / end2end_acquisition.PROFILE_MODEL
        maps_dir = profile_dir / "material" / "maps"
        for name in ("baseColor", "normal", "roughness", "specular"):
            write_image(maps_dir / f"{name}.png", image + profile_index * 0.05)

        olat_case = profile_dir / "evaluation" / "olat" / "cases" / "000002"
        for name in ("gt", "pred", "error"):
            write_image(olat_case / f"{name}.png", image)
        hdri_case = profile_dir / "evaluation" / "hdri" / "cases" / "studio_rot000"
        for name in ("lighting", "gt", "pred", "error"):
            write_image(hdri_case / f"{name}.png", image)

        profile_results[profile] = {
            "fit_conditions": {
                "olat": 8 if profile in {"olat", "mix"} else 0,
                "hdri": 24 if profile in {"hdri", "mix"} else 0,
            },
            "evaluation": {
                "evaluations": {
                    "olat": {
                        "metrics": dict(metric_names),
                        "representative": {
                            "frame_id": 2,
                            "gt_path": "cases/000002/gt.png",
                            "pred_path": "cases/000002/pred.png",
                            "error_path": "cases/000002/error.png",
                        },
                    },
                    "hdri": {
                        "metrics": dict(metric_names),
                        "representative": {
                            "condition_id": "studio_rot000",
                            "lighting_path": "cases/studio_rot000/lighting.png",
                            "gt_path": "cases/studio_rot000/gt.png",
                            "pred_path": "cases/studio_rot000/pred.png",
                            "error_path": "cases/studio_rot000/error.png",
                        },
                    },
                }
            }
        }

    artifacts = end2end_acquisition._write_camera_report(
        tmp_path, profiles, profile_results
    )

    assert artifacts == {
        "overview": "report/overview.png",
        "summary": "report/summary.json",
        "metrics": "report/metrics.csv",
    }
    for relative_path in artifacts.values():
        assert (tmp_path / relative_path).is_file()
    summary = json.loads((tmp_path / artifacts["summary"]).read_text())
    assert summary["profiles"] == list(profiles)
    assert [row["training_profile"] for row in summary["rows"]] == list(profiles)
    assert all({"olat", "hdri"}.issubset(row) for row in summary["rows"])
    with (tmp_path / artifacts["metrics"]).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert {
        (row["training_profile"], row["evaluation_lighting"]) for row in rows
    } == {(profile, lighting) for profile in profiles for lighting in ("olat", "hdri")}


def test_end2end_provenance_records_git_state_and_exact_source_hash(
    monkeypatch, tmp_path
):
    source = tmp_path / "CookTorrance_IBL" / "disney_brdf.py"
    source.parent.mkdir()
    source_bytes = b"class DisneyBRDFSimplifiedMultiLayer:\n    pass\n"
    source.write_bytes(source_bytes)

    def fake_run(command, **_kwargs):
        if "status" in command:
            return SimpleNamespace(returncode=0, stdout=" M CookTorrance_IBL/disney_brdf.py\n")
        if "rev-parse" in command:
            return SimpleNamespace(returncode=0, stdout="abc123\n")
        raise AssertionError(f"unexpected git command: {command}")

    monkeypatch.setattr(end2end_acquisition.subprocess, "run", fake_run)
    provenance = end2end_acquisition._imaginaire_provenance(tmp_path, source)

    assert provenance == {
        "root": str(tmp_path),
        "commit": "abc123",
        "dirty": True,
        "disney_brdf_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }


@pytest.mark.parametrize(
    (
        "extra_args",
        "expected_mode",
        "expected_backend",
        "expected_steps",
        "expected_eval_lights",
    ),
    [
        ([], "default", "auto", 33000, 16),
        (
            [
                "--material-acquisition",
                "end2end",
                "--backend",
                "torch",
                "--end2end-steps",
                "17",
                "--end2end-learning-rate",
                "0.002",
                "--end2end-eval-lights",
                "7",
                "--end2end-profiles",
                "hdri,mix",
                "--end2end-hdri-root",
                "/tmp/hdris",
                "--end2end-hdri-count",
                "5",
                "--end2end-eval-hdris",
                "2",
                "--end2end-hdri-rotations",
                "6",
                "--end2end-primary-profile",
                "mix",
            ],
            "end2end",
            "torch",
            17,
            7,
        ),
    ],
)
def test_prepare_materials_dispatches_acquisition_mode(
    monkeypatch,
    tmp_path,
    extra_args,
    expected_mode,
    expected_backend,
    expected_steps,
    expected_eval_lights,
):
    sample = SimpleNamespace(object_name="object", camera="cam00")
    calls = []
    monkeypatch.setattr(prepare_materials, "iter_camera_samples", lambda _root: [sample])
    monkeypatch.setattr(prepare_materials, "tqdm", lambda iterable, **_kwargs: iterable)

    def fake_decompose(received_sample, **kwargs):
        assert received_sample is sample
        calls.append(kwargs)
        return 4

    monkeypatch.setattr(prepare_materials, "decompose_camera_sample", fake_decompose)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_materials",
            "--data-root",
            str(tmp_path / "data"),
            "--out-root",
            str(tmp_path / "materials"),
            *extra_args,
        ],
    )

    prepare_materials.main()

    assert len(calls) == 1
    assert calls[0]["material_acquisition"] == expected_mode
    assert calls[0]["backend"] == expected_backend
    assert calls[0]["end2end_steps"] == expected_steps
    assert calls[0]["end2end_eval_lights"] == expected_eval_lights
    assert calls[0]["end2end_learning_rate"] == pytest.approx(
        0.002 if expected_mode == "end2end" else 1e-3
    )
    if expected_mode == "end2end":
        assert calls[0]["end2end_profiles"] == "hdri,mix"
        assert calls[0]["end2end_hdri_root"] == "/tmp/hdris"
        assert calls[0]["end2end_hdri_count"] == 5
        assert calls[0]["end2end_eval_hdris"] == 2
        assert calls[0]["end2end_hdri_rotations"] == 6
        assert calls[0]["end2end_primary_profile"] == "mix"


def test_slurm_dry_run_propagates_end2end_selector_and_options(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    data_root = tmp_path / "data"
    output_root = tmp_path / "outputs"
    imaginaire_root = tmp_path / "imaginaire"
    imaginaire_root.mkdir()
    hdri_root = tmp_path / "hdris"
    hdri_root.mkdir()
    # This test covers shell argument serialization, not Python data validation. Keep
    # it independent of the active Conda installation and its compiled dependencies.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for executable in ("conda", "python"):
        path = fake_bin / executable
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
    environment = os.environ.copy()
    environment["ENV_NAME"] = "__ictpolarreal_pytest_missing_env__"
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    result = subprocess.run(
        [
            "bash",
            str(repo_root / "run.sh"),
            "process",
            "--data-root",
            str(data_root),
            "--output-root",
            str(output_root),
            "--material-acquisition",
            "end2end",
            "--imaginaire-root",
            str(imaginaire_root),
            "--end2end-steps",
            "17",
            "--end2end-learning-rate",
            "0.002",
            "--end2end-eval-lights",
            "7",
            "--end2end-profiles",
            "hdri,mix",
            "--end2end-hdri-root",
            str(hdri_root),
            "--end2end-hdri-count",
            "5",
            "--end2end-eval-hdris",
            "2",
            "--end2end-hdri-rotations",
            "6",
            "--end2end-primary-profile",
            "mix",
            "--max-lights",
            "4",
            "--min-lights",
            "4",
            "--backend",
            "torch",
            "--device",
            "cuda",
            "--slurm-dry-run",
            "--slurm-gpus",
            "1",
        ],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "--material-acquisition end2end" in result.stdout
    assert f"--material-root {output_root / 'material_acquisition_end2end'}" in result.stdout
    assert f"--imaginaire-root {imaginaire_root}" in result.stdout
    assert "--end2end-steps 17" in result.stdout
    assert "--end2end-learning-rate 0.002" in result.stdout
    assert "--end2end-eval-lights 7" in result.stdout
    command = shlex.split(result.stdout.partition(":")[2])
    assert command[command.index("--end2end-profiles") + 1] == "hdri,mix"
    assert f"--end2end-hdri-root {hdri_root}" in result.stdout
    assert "--end2end-hdri-count 5" in result.stdout
    assert "--end2end-eval-hdris 2" in result.stdout
    assert "--end2end-hdri-rotations 6" in result.stdout
    assert "--end2end-primary-profile mix" in result.stdout
    assert "--gpus-per-node=1" in result.stdout
