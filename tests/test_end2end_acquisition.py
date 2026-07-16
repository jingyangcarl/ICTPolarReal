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


def test_report_font_preserves_requested_scale():
    small = end2end_acquisition._report_font(16)
    large = end2end_acquisition._report_font(48, bold=True)

    small_box = small.getbbox("Readable")
    large_box = large.getbbox("Readable")
    small_height = small_box[3] - small_box[1]
    large_height = large_box[3] - large_box[1]

    assert large_height >= 30
    assert large_height >= 2 * small_height
    assert end2end_acquisition._report_camera_label(Path("cam07")) == "Camera 07"


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


def test_superdimension_selection_fits_164_visible_lights_and_holds_out_nine():
    train, heldout, excluded = (
        end2end_acquisition.select_superdimension_light_indices(
            np.arange(346, dtype=np.int64), eval_lights=16
        )
    )

    assert len(train) == 164
    np.testing.assert_array_equal(
        train, np.unique(np.linspace(0, 172, 164, dtype=np.int64))
    )
    np.testing.assert_array_equal(
        heldout, np.asarray([19, 38, 57, 76, 95, 114, 133, 152, 171])
    )
    assert set(train).isdisjoint(heldout)
    assert max(train) <= 172
    assert max(heldout) <= 172
    np.testing.assert_array_equal(excluded, np.arange(173, 346))


@pytest.mark.parametrize(
    (
        "eval_lights",
        "heldout_count",
        "excluded_count",
        "excluded_visible_count",
        "excluded_nonvisible_count",
    ),
    [
        (16, 9, 173, 0, 173),
        (0, 0, 182, 9, 173),
    ],
)
def test_lsx_selection_metadata_separates_visible_and_nonvisible_exclusions(
    eval_lights,
    heldout_count,
    excluded_count,
    excluded_visible_count,
    excluded_nonvisible_count,
):
    light_ids = np.arange(346, dtype=np.int64)
    train, heldout, excluded = (
        end2end_acquisition.select_superdimension_light_indices(
            light_ids, eval_lights=eval_lights
        )
    )

    metadata = end2end_acquisition._describe_light_selection(
        light_ids,
        train,
        heldout,
        excluded,
        requested_heldout_count=eval_lights,
    )

    assert metadata["mode"] == "lsx_visible_hemisphere_164"
    assert metadata["fit_count"] == 164
    assert metadata["heldout_count"] == heldout_count
    assert metadata["heldout_visible_count"] == heldout_count
    assert metadata["excluded_count"] == excluded_count
    assert metadata["excluded_visible_count"] == excluded_visible_count
    assert metadata["excluded_nonvisible_count"] == excluded_nonvisible_count
    assert "excluded_rear_or_unused_count" not in metadata


def test_reduced_capture_selection_metadata_reports_generic_fallback():
    light_ids = np.arange(40, dtype=np.int64)
    train, heldout, excluded = (
        end2end_acquisition.select_superdimension_light_indices(
            light_ids, eval_lights=5
        )
    )

    metadata = end2end_acquisition._describe_light_selection(
        light_ids,
        train,
        heldout,
        excluded,
        requested_heldout_count=5,
    )

    assert metadata == {
        "mode": "generic_sphere_spread",
        "policy": (
            "generic deterministic sphere-spread holdout over all selected "
            "calibrated lights"
        ),
        "requested_heldout_count": 5,
        "fit_count": 35,
        "heldout_count": 5,
        "excluded_count": 0,
    }


def test_zero_holdout_evaluation_samples_only_fitted_lsx_lights():
    train, heldout, excluded = (
        end2end_acquisition.select_superdimension_light_indices(
            np.arange(346, dtype=np.int64), eval_lights=0
        )
    )

    evaluation = end2end_acquisition._select_evaluation_light_indices(
        train, heldout
    )

    assert len(evaluation) == 16
    assert set(evaluation).issubset(set(train))
    assert set(evaluation).isdisjoint(set(excluded))


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
    assert all("preview" not in condition for condition in manifest["conditions"])
    assert not (tmp_path / "lighting" / "previews").exists()


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


def test_end2end_prefers_albedo_photometric_inputs_and_constant_view(
    monkeypatch, tmp_path
):
    camera_dir = tmp_path / "object" / "cam00"
    for kind in ("cross", "parallel"):
        directory = camera_dir / kind
        directory.mkdir(parents=True)
        for frame_id in range(4):
            (directory / f"{frame_id:06d}.png").touch()
    for name in ("albedo.png", "static.png", "normal.png", "mask.png"):
        (camera_dir / name).touch()
    sample = CameraSample("object", "cam00", camera_dir)

    albedo = np.full((2, 3, 3), 0.35, dtype=np.float32)
    static = np.full((2, 3, 3), 0.8, dtype=np.float32)
    photometric_normal = np.zeros((2, 3, 3), dtype=np.float32)
    photometric_normal[..., 2] = 1.0
    optical_axis = np.asarray([0.0, 0.6, 0.8], dtype=np.float32)
    constant_view = np.broadcast_to(optical_axis, (2, 3, 3)).copy()

    def fake_read(path, **_kwargs):
        stem = Path(path).stem
        if stem == "albedo":
            return albedo.copy()
        if stem == "static":
            return static.copy()
        if stem == "normal":
            return photometric_normal.copy()
        if stem == "mask":
            return np.ones((2, 3, 1), dtype=np.float32)
        return np.full((2, 3, 3), int(stem), dtype=np.float32)

    monkeypatch.setattr(material_decomposition, "read_image", fake_read)
    monkeypatch.setattr(
        material_decomposition,
        "load_light_directions",
        lambda _root, indices, **_kwargs: np.tile(
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            (len(indices), 1),
        ),
    )
    monkeypatch.setattr(
        material_decomposition,
        "load_view_directions",
        lambda _root, _sample, hw: np.broadcast_to(
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32), hw + (3,)
        ).copy(),
    )
    monkeypatch.setattr(
        material_decomposition,
        "load_end2end_view_directions",
        lambda _root, _sample, _hw: constant_view.copy(),
    )

    def unexpected_ward(*_args, **_kwargs):
        raise AssertionError("dataset albedo/normal inputs must bypass Ward initialization")

    captured = {}

    def fake_acquire(_cross, _parallel, _lights, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        material_decomposition, "decompose_polarized_olat", unexpected_ward
    )
    monkeypatch.setattr(
        end2end_acquisition, "validate_end2end_runtime", lambda *_args: None
    )
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
    )

    assert used == 4
    np.testing.assert_array_equal(captured["base_color"], albedo)
    assert captured["base_color_source"] == "dataset_albedo"
    np.testing.assert_array_equal(captured["normal"], photometric_normal)
    np.testing.assert_array_equal(captured["view_dirs"], constant_view)
    np.testing.assert_array_equal(captured["view_dirs"][0, 0], optical_axis)


def test_masked_disney_scalar_tv_penalizes_an_interior_spike():
    torch = pytest.importorskip("torch")
    scalar = torch.zeros((3, 3), dtype=torch.float32)
    scalar[1, 1] = 1.0
    scalar.requires_grad_()
    model = SimpleNamespace(
        _param_maps=lambda: {
            name: scalar for name in end2end_acquisition.DISNEY_TV_SCALAR_NAMES
        }
    )
    mask = torch.ones((3, 3, 1), dtype=torch.float32)

    total_variation = end2end_acquisition._masked_disney_scalar_total_variation(
        torch, model, mask
    )
    total_variation.backward()

    assert total_variation.item() > 0.0
    assert scalar.grad[1, 1].abs().item() > 0.0


def test_masked_disney_scalar_tv_ignores_differences_outside_fit_mask():
    torch = pytest.importorskip("torch")
    scalar = torch.tensor(
        [[0.25, 0.25, 0.0, 1.0], [0.25, 0.25, 1.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    model = SimpleNamespace(
        _param_maps=lambda: {
            name: scalar for name in end2end_acquisition.DISNEY_TV_SCALAR_NAMES
        }
    )
    mask = torch.zeros((2, 4, 1), dtype=torch.float32)
    mask[:, :2] = 1.0

    total_variation = end2end_acquisition._masked_disney_scalar_total_variation(
        torch, model, mask
    )
    total_variation.backward()

    assert total_variation.item() == pytest.approx(0.0)
    assert torch.count_nonzero(scalar.grad[:, 2:]) == 0


def _edge_regularizer_inputs(torch, scalar, mask=None, albedo=None, normal=None):
    height, width = scalar.shape
    if mask is None:
        mask = torch.ones((height, width, 1), dtype=torch.float32)
    if albedo is None:
        albedo = torch.full((height, width, 3), 0.5, dtype=torch.float32)
    if normal is None:
        normal = torch.zeros((height, width, 3), dtype=torch.float32)
        normal[..., 2] = 1.0
    model = SimpleNamespace(
        _param_maps=lambda: {
            name: scalar for name in end2end_acquisition.DISNEY_TV_SCALAR_NAMES
        }
    )
    pair_weights = end2end_acquisition._edge_aware_pair_weights(
        torch, albedo, normal, mask
    )
    return model, mask, pair_weights


def test_edge_charbonnier_is_zero_for_constant_scalar_maps():
    torch = pytest.importorskip("torch")
    scalar = torch.full((3, 4), 0.4, dtype=torch.float32, requires_grad=True)
    model, mask, pair_weights = _edge_regularizer_inputs(torch, scalar)

    loss = end2end_acquisition._disney_scalar_regularization(
        torch,
        model,
        mask,
        kind="edge-charbonnier",
        edge_pair_weights=pair_weights,
    )
    loss.backward()

    assert loss.item() == pytest.approx(0.0)
    assert torch.count_nonzero(scalar.grad) == 0


def test_edge_charbonnier_penalizes_an_interior_spike():
    torch = pytest.importorskip("torch")
    scalar = torch.zeros((3, 3), dtype=torch.float32)
    scalar[1, 1] = 1.0
    scalar.requires_grad_()
    model, mask, pair_weights = _edge_regularizer_inputs(torch, scalar)

    loss = end2end_acquisition._disney_scalar_regularization(
        torch,
        model,
        mask,
        kind="edge-charbonnier",
        edge_pair_weights=pair_weights,
    )
    loss.backward()

    assert loss.item() > 0.0
    assert scalar.grad[1, 1].item() > 0.0


@pytest.mark.parametrize("guide_kind", ["albedo", "normal"])
def test_edge_charbonnier_preserves_scalar_changes_at_guide_edges(guide_kind):
    torch = pytest.importorskip("torch")
    scalar = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    flat_albedo = torch.full((2, 4, 3), 0.5, dtype=torch.float32)
    edge_albedo = flat_albedo.clone()
    flat_normal = torch.zeros((2, 4, 3), dtype=torch.float32)
    flat_normal[..., 2] = 1.0
    edge_normal = flat_normal.clone()
    if guide_kind == "albedo":
        edge_albedo[:, 2:] = 1.0
    else:
        edge_normal[:, 2:, 0] = 1.0
        edge_normal[:, 2:, 2] = 0.0
    model, mask, flat_weights = _edge_regularizer_inputs(
        torch, scalar, albedo=flat_albedo, normal=flat_normal
    )
    _, _, edge_weights = _edge_regularizer_inputs(
        torch,
        scalar,
        mask=mask,
        albedo=edge_albedo,
        normal=edge_normal,
    )

    flat_loss = end2end_acquisition._disney_scalar_regularization(
        torch,
        model,
        mask,
        kind="edge-charbonnier",
        edge_pair_weights=flat_weights,
    )
    edge_loss = end2end_acquisition._disney_scalar_regularization(
        torch,
        model,
        mask,
        kind="edge-charbonnier",
        edge_pair_weights=edge_weights,
    )

    assert edge_loss.item() < flat_loss.item() * 0.1
    assert edge_weights["horizontal_weight"][:, 1].max().item() == pytest.approx(
        end2end_acquisition.EDGE_CHARBONNIER_WEIGHT_FLOOR, abs=1e-4
    )


def test_edge_charbonnier_uses_only_exact_fit_pairs():
    torch = pytest.importorskip("torch")
    scalar = torch.tensor(
        [[0.25, 0.25, 0.0, 1.0], [0.25, 0.25, 1.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    mask = torch.zeros((2, 4, 1), dtype=torch.float32)
    mask[:, :2] = 1.0
    model, mask, pair_weights = _edge_regularizer_inputs(
        torch, scalar, mask=mask
    )

    loss = end2end_acquisition._disney_scalar_regularization(
        torch,
        model,
        mask,
        kind="edge-charbonnier",
        edge_pair_weights=pair_weights,
    )
    loss.backward()

    assert loss.item() == pytest.approx(0.0)
    assert torch.count_nonzero(scalar.grad[:, 2:]) == 0


def test_edge_guide_pair_weights_are_detached_from_albedo_and_normal():
    torch = pytest.importorskip("torch")
    scalar = torch.zeros((2, 3), dtype=torch.float32, requires_grad=True)
    albedo = torch.rand((2, 3, 3), dtype=torch.float32, requires_grad=True)
    normal = torch.rand((2, 3, 3), dtype=torch.float32, requires_grad=True)
    model, mask, pair_weights = _edge_regularizer_inputs(
        torch, scalar, albedo=albedo, normal=normal
    )

    assert all(not value.requires_grad for value in pair_weights.values())
    loss = end2end_acquisition._disney_scalar_regularization(
        torch,
        model,
        mask,
        kind="edge-charbonnier",
        edge_pair_weights=pair_weights,
    )
    loss.backward()

    assert albedo.grad is None
    assert normal.grad is None


def _impulse_regularizer_inputs(torch, scalar, mask=None, albedo=None, normal=None):
    height, width = scalar.shape
    if mask is None:
        mask = torch.ones((height, width, 1), dtype=torch.float32)
    if albedo is None:
        albedo = torch.full((height, width, 3), 0.5, dtype=torch.float32)
    if normal is None:
        normal = torch.zeros((height, width, 3), dtype=torch.float32)
        normal[..., 2] = 1.0
    model = SimpleNamespace(
        _param_maps=lambda: {
            name: scalar for name in end2end_acquisition.DISNEY_TV_SCALAR_NAMES
        }
    )
    bundle = end2end_acquisition._build_impulse_median_bundle(
        torch,
        model,
        mask,
        albedo,
        normal,
        created_after_step=90,
    )
    return model, mask, bundle


def _toy_disney_scalar_model(torch, values):
    class ToyDisneyScalarModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in end2end_acquisition.DISNEY_TV_SCALAR_NAMES:
                constrained = values[name].clamp(1e-5, 1.0 - 1e-5)
                setattr(
                    self,
                    f"{name}_un",
                    torch.nn.Parameter(torch.logit(constrained).unsqueeze(0)),
                )

        def _param_maps(self):
            return {
                name: torch.sigmoid(getattr(self, f"{name}_un"))[0]
                for name in end2end_acquisition.DISNEY_TV_SCALAR_NAMES
            }

    return ToyDisneyScalarModel()


def _manual_impulse_bundle(torch, model, targets):
    maps = {}
    constrained = model._param_maps()
    for name in end2end_acquisition.DISNEY_TV_SCALAR_NAMES:
        target = constrained[name].detach().clone()
        mask = torch.zeros_like(target, dtype=torch.bool)
        for row, column, value in targets.get(name, []):
            mask[row, column] = True
            target[row, column] = value
        maps[name] = {
            "target": target,
            "mask": mask,
            "flagged_count": int(mask.sum().item()),
        }
    return {
        "schema": "ictpolarreal.impulse-median-bundle.v1",
        "created_after_step": 90,
        "maps": maps,
    }


def test_impulse_median_stage_plan_keeps_full_data_fit_then_derives_proximal():
    plan = end2end_acquisition._impulse_median_stage_plan(
        33000,
        enabled=True,
        shrink_per_iteration=0.00125,
    )

    assert plan == {
        "enabled": True,
        "data_fit_steps": 33000,
        "detector_after_data_step": 33000,
        "cleanup_iterations": 3300,
        "cleanup_fraction": pytest.approx(0.1),
        "shrink_per_iteration": pytest.approx(0.00125),
        "total_shrink": pytest.approx(4.125),
        "cleanup_optimizer_steps": 0,
    }
    assert end2end_acquisition._impulse_median_stage_plan(
        7, enabled=False
    ) == {
        "enabled": False,
        "data_fit_steps": 7,
        "detector_after_data_step": None,
        "cleanup_iterations": 0,
        "cleanup_fraction": 0.0,
        "shrink_per_iteration": 0.0,
        "total_shrink": 0.0,
        "cleanup_optimizer_steps": 0,
    }


def test_impulse_median_settings_record_frozen_detector_and_schedule():
    settings = end2end_acquisition._regularization_settings("impulse-median")
    adapter = end2end_acquisition._adapter_provenance()

    assert "impulse-median" in end2end_acquisition.TV_KINDS
    assert settings["cleanup_fraction"] == pytest.approx(0.1)
    assert settings["detector_window"] == 5
    assert settings["mad_scale"] == pytest.approx(4.0)
    assert settings["minimum_deviation"] == pytest.approx(0.035)
    assert settings["dead_zone"] == pytest.approx(0.005)
    assert settings["cleanup_optimizer_steps"] == 0
    assert settings["detector_boundary"] == "after_all_data_fit_steps"
    assert settings["unflagged_parameter_update"].startswith("none_bit_identical")
    assert settings["local_maximum_tie_policy"] == "retain_all_equal_maxima"
    assert adapter["schema"] == "ictpolarreal.profile-acquisition-adapter.v6"
    assert adapter["algorithm_version"] == "ictpolarreal-impulse-proximal-v5"


def test_zero_weight_final_regularization_does_not_require_impulse_bundle():
    def missing_bundle_regularizer():
        raise ValueError("impulse-median regularization requires a frozen bundle")

    assert end2end_acquisition._final_regularization_value(
        missing_bundle_regularizer,
        weight=0.0,
    ) == pytest.approx(0.0)


def test_impulse_median_flags_only_an_isolated_guide_safe_spike():
    torch = pytest.importorskip("torch")
    scalar = torch.zeros((9, 9), dtype=torch.float32)
    scalar[4, 4] = 1.0
    scalar.requires_grad_()
    _model, _mask, bundle = _impulse_regularizer_inputs(torch, scalar)

    for entry in bundle["maps"].values():
        assert entry["flagged_count"] == 1
        assert entry["mask"][4, 4]
        assert torch.count_nonzero(entry["mask"]) == 1
        assert entry["target"][4, 4].item() == pytest.approx(0.0)
        assert not entry["target"].requires_grad
        assert not entry["mask"].requires_grad


@pytest.mark.parametrize("rejection", ["albedo_edge", "mask_hole"])
def test_impulse_median_rejects_guide_or_foreground_unsafe_centers(rejection):
    torch = pytest.importorskip("torch")
    scalar = torch.zeros((9, 9), dtype=torch.float32)
    scalar[4, 4] = 1.0
    mask = torch.ones((9, 9, 1), dtype=torch.float32)
    albedo = torch.full((9, 9, 3), 0.5, dtype=torch.float32)
    if rejection == "albedo_edge":
        albedo[:, 5:] = 1.0
    else:
        mask[2, 2] = 0.0
    scalar.requires_grad_()

    _model, _mask, bundle = _impulse_regularizer_inputs(
        torch,
        scalar,
        mask=mask,
        albedo=albedo,
    )

    assert all(entry["flagged_count"] == 0 for entry in bundle["maps"].values())


@pytest.mark.parametrize(
    ("neighbor_value", "expected_columns"),
    [(0.8, [4]), (1.0, [4, 5])],
)
def test_impulse_median_uses_normalized_local_peaks_and_retains_ties(
    neighbor_value,
    expected_columns,
):
    torch = pytest.importorskip("torch")
    scalar = torch.zeros((9, 9), dtype=torch.float32)
    scalar[4, 4] = 1.0
    scalar[4, 5] = neighbor_value
    _model, _mask, bundle = _impulse_regularizer_inputs(torch, scalar)

    entry = bundle["maps"]["roughness"]
    assert entry["flagged_count"] == len(expected_columns)
    assert torch.nonzero(entry["mask"], as_tuple=False).tolist() == [
        [4, column] for column in expected_columns
    ]


def test_impulse_proximal_soft_thresholds_only_excess_beyond_dead_zone():
    torch = pytest.importorskip("torch")
    values = {
        name: torch.full((3, 3), 0.5, dtype=torch.float32)
        for name in end2end_acquisition.DISNEY_TV_SCALAR_NAMES
    }
    values["metallic"][1, 1] = 0.8
    values["roughness"][1, 1] = 0.52
    model = _toy_disney_scalar_model(torch, values)
    bundle = _manual_impulse_bundle(
        torch,
        model,
        {
            "metallic": [(1, 1, 0.2)],
            "roughness": [(1, 1, 0.5)],
        },
    )
    roughness_raw_before = model.roughness_un.detach().clone()

    diagnostic = end2end_acquisition._apply_impulse_median_proximal(
        torch,
        model,
        bundle,
        total_shrink=0.1,
        dead_zone=0.05,
    )

    assert model._param_maps()["metallic"][1, 1].item() == pytest.approx(0.7)
    assert torch.equal(model.roughness_un, roughness_raw_before)
    assert diagnostic["flagged_centers"] == 2
    assert diagnostic["moved_centers"] == 1
    assert diagnostic["maps"]["metallic"]["mean_distance_before"] == pytest.approx(
        0.6
    )
    assert diagnostic["maps"]["metallic"]["mean_distance_after"] == pytest.approx(
        0.5
    )


def test_impulse_proximal_preserves_every_unflagged_raw_entry_and_adam_state():
    torch = pytest.importorskip("torch")
    values = {
        name: torch.linspace(0.2, 0.8, 25, dtype=torch.float32).reshape(5, 5)
        for name in end2end_acquisition.DISNEY_TV_SCALAR_NAMES
    }
    baseline = _toy_disney_scalar_model(torch, values)
    candidate = _toy_disney_scalar_model(torch, values)
    candidate.load_state_dict(baseline.state_dict())
    baseline_optimizer = torch.optim.Adam(baseline.parameters(), lr=1e-3)
    candidate_optimizer = torch.optim.Adam(candidate.parameters(), lr=1e-3)

    def data_step(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        coupled = torch.stack(list(model._param_maps().values())).sum(dim=0)
        coupled.square().mean().backward()
        optimizer.step()

    data_step(baseline, baseline_optimizer)
    data_step(candidate, candidate_optimizer)
    bundle = _manual_impulse_bundle(
        torch,
        candidate,
        {"metallic": [(2, 2, 0.1)]},
    )
    end2end_acquisition._apply_impulse_median_proximal(
        torch,
        candidate,
        bundle,
        total_shrink=0.2,
        dead_zone=0.005,
    )

    flagged = bundle["maps"]["metallic"]["mask"].unsqueeze(0)
    assert torch.equal(candidate.metallic_un[~flagged], baseline.metallic_un[~flagged])
    assert not torch.equal(candidate.metallic_un[flagged], baseline.metallic_un[flagged])
    for name in end2end_acquisition.DISNEY_TV_SCALAR_NAMES:
        candidate_parameter = getattr(candidate, f"{name}_un")
        baseline_parameter = getattr(baseline, f"{name}_un")
        if name != "metallic":
            assert torch.equal(candidate_parameter, baseline_parameter)
        for state_name in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(
                candidate_optimizer.state[candidate_parameter][state_name],
                baseline_optimizer.state[baseline_parameter][state_name],
            )


def test_impulse_frozen_artifact_preserves_exact_masks_targets_and_provenance(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    scalar = torch.zeros((9, 9), dtype=torch.float32)
    scalar[4, 4] = 1.0
    model, _mask, bundle = _impulse_regularizer_inputs(torch, scalar)
    artifact_path = tmp_path / end2end_acquisition.IMPULSE_FROZEN_ARTIFACT_NAME
    artifact = end2end_acquisition._write_impulse_frozen_artifact(
        artifact_path,
        bundle,
    )
    with np.load(artifact_path, allow_pickle=False) as frozen:
        metadata = json.loads(str(frozen["metadata"].item()))
        assert metadata["created_after_step"] == 90
        for name in end2end_acquisition.DISNEY_TV_SCALAR_NAMES:
            np.testing.assert_array_equal(
                frozen[f"{name}__target"],
                bundle["maps"][name]["target"].detach().numpy(),
            )
            np.testing.assert_array_equal(
                frozen[f"{name}__mask"],
                bundle["maps"][name]["mask"].detach().numpy(),
            )
    signature_regularization = {
        "kind": "impulse-median",
        "weight": 0.00125,
        "parameters": list(end2end_acquisition.DISNEY_TV_SCALAR_NAMES),
        "settings": end2end_acquisition._regularization_settings("impulse-median"),
        "stage_plan": {"enabled": True},
    }
    acquisition = {
        "regularization": {
            **signature_regularization,
            "frozen_bundle": {"artifact": artifact},
        },
        "checkpoint_signature": {
            "regularization": dict(signature_regularization),
        },
    }
    assert end2end_acquisition._impulse_frozen_artifact_complete(
        tmp_path, acquisition
    )
    assert not end2end_acquisition._impulse_frozen_artifact_complete(
        tmp_path,
        {"checkpoint_signature": acquisition["checkpoint_signature"]},
    )
    tampered_top_level = json.loads(json.dumps(acquisition))
    tampered_top_level["regularization"]["weight"] = 0.5
    assert not end2end_acquisition._impulse_frozen_artifact_complete(
        tmp_path, tampered_top_level
    )
    missing_signature = {"regularization": acquisition["regularization"]}
    assert not end2end_acquisition._impulse_frozen_artifact_complete(
        tmp_path, missing_signature
    )
    assert not (tmp_path / f"{artifact_path.name}.tmp").exists()
    with artifact_path.open("ab") as stream:
        stream.write(b"tampered")
    assert not end2end_acquisition._impulse_frozen_artifact_complete(
        tmp_path, acquisition
    )
    disabled = {
        "regularization": {
            "kind": "impulse-median",
            "stage_plan": {"enabled": False},
        }
    }
    assert end2end_acquisition._impulse_frozen_artifact_complete(tmp_path, disabled)
    assert model._param_maps()["roughness"][4, 4].item() == pytest.approx(1.0)


def test_fit_without_frozen_bundle_unlinks_stale_impulse_artifact(tmp_path):
    stale = tmp_path / end2end_acquisition.IMPULSE_FROZEN_ARTIFACT_NAME
    stale.write_bytes(b"stale-enabled-run")

    assert end2end_acquisition._finalize_impulse_frozen_artifact(
        tmp_path, None
    ) is None
    assert not stale.exists()


def test_impulse_median_checkpoint_requires_exact_frozen_bundle_after_boundary():
    torch = pytest.importorskip("torch")
    scalar = torch.zeros((9, 9), dtype=torch.float32)
    scalar[4, 4] = 1.0
    _model, _mask, bundle = _impulse_regularizer_inputs(torch, scalar)
    stage_plan = end2end_acquisition._impulse_median_stage_plan(
        90,
        enabled=True,
        shrink_per_iteration=0.01,
    )

    assert end2end_acquisition._checkpoint_impulse_median_state(
        torch,
        {"impulse_median_bundle": None},
        stage_plan,
        next_step=89,
        expected_shape=(9, 9),
    ) == (None, False, None)
    pending = end2end_acquisition._checkpoint_impulse_median_state(
        torch,
        {
            "impulse_median_bundle": bundle,
            "impulse_cleanup_applied": False,
            "impulse_cleanup_diagnostic": None,
        },
        stage_plan,
        next_step=90,
        expected_shape=(9, 9),
    )
    assert pending[0] is bundle
    assert pending[1:] == (False, None)
    diagnostic = {"schema": "ictpolarreal.impulse-proximal-diagnostic.v1"}
    applied = end2end_acquisition._checkpoint_impulse_median_state(
        torch,
        {
            "impulse_median_bundle": bundle,
            "impulse_cleanup_applied": True,
            "impulse_cleanup_diagnostic": diagnostic,
        },
        stage_plan,
        next_step=90,
        expected_shape=(9, 9),
    )
    assert applied[0] is bundle
    assert applied[1:] == (True, diagnostic)
    with pytest.raises(ValueError, match="missing impulse-median frozen bundle"):
        end2end_acquisition._checkpoint_impulse_median_state(
            torch,
            {"impulse_median_bundle": None},
            stage_plan,
            next_step=90,
            expected_shape=(9, 9),
        )
    stale = dict(bundle)
    stale["created_after_step"] = 89
    with pytest.raises(ValueError, match="wrong stage boundary"):
        end2end_acquisition._checkpoint_impulse_median_state(
            torch,
            {"impulse_median_bundle": stale},
            stage_plan,
            next_step=90,
            expected_shape=(9, 9),
        )


def test_end2end_view_is_constant_optical_axis(tmp_path):
    camera_dir = tmp_path / "object" / "cam00"
    camera_dir.mkdir(parents=True)
    sample = CameraSample("object", "cam00", camera_dir)

    view = material_decomposition.load_end2end_view_directions(
        tmp_path, sample, (3, 5)
    )

    assert view.shape == (3, 5, 3)
    np.testing.assert_allclose(view, np.broadcast_to(view[0, 0], view.shape))
    np.testing.assert_allclose(np.linalg.norm(view, axis=-1), 1.0)


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


def test_disney_scalar_defaults_are_recorded_without_rewriting_unconstrained_values():
    torch = pytest.importorskip("torch")
    raw_defaults = {
        name: (0.5 if name in {"specular", "roughness", "sheenTint", "clearcoatGloss"} else 0.0)
        for name in end2end_acquisition.DISNEY_SCALAR_NAMES
    }
    model = SimpleNamespace(
        **{
            f"{name}_un": torch.nn.Parameter(torch.full((1,), value))
            for name, value in raw_defaults.items()
        }
    )

    initialization = end2end_acquisition._disney_scalar_initialization(torch, model)

    for name, raw_value in raw_defaults.items():
        assert float(getattr(model, f"{name}_un").item()) == raw_value
        assert initialization[name]["unconstrained"] == raw_value
        assert initialization[name]["constrained"] == pytest.approx(
            float(torch.sigmoid(torch.tensor(raw_value)).item())
        )


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


def test_end2end_targets_mask_background_then_use_whole_image_scale():
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
    # The bright background is removed before the whole-image renderer scale.
    np.testing.assert_allclose(normalized[:, :, 1], 0.0)


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


def test_write_olat_evaluation_writes_private_profile_staging_artifacts(tmp_path):
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

    evaluation_dir = tmp_path / "evaluation" / ".profiles" / "olat" / "olat"
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
    assert {
        "gt_mean_intensity",
        "pred_mean_intensity",
        "mean_intensity_ratio",
        "luminance_correlation",
    }.issubset(rows[0])
    assert [row["split"] for row in rows] == ["heldout_olat", "heldout_olat"]
    assert [int(row["frame_id"]) for row in rows] == [2, 4]
    assert [float(row["mse"]) for row in rows] == pytest.approx([0.01, 0.01])
    on_disk_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert on_disk_summary == summary
    assert summary["representative"]["gt_path"].startswith("cases/")


@pytest.mark.parametrize("source_layout", ["staging", "legacy"])
def test_camera_report_consolidates_clean_lighting_first_contract(
    tmp_path, source_layout
):
    profiles = ("olat", "hdri", "mix")
    profile_results = {}
    image = np.full((6, 10, 3), 0.4, dtype=np.float32)
    metric_names = {"mse": 0.01, "mae": 0.1, "psnr": 20.0, "ssim_global": 0.8}
    fit_adapter = {
        "schema": "ictpolarreal.profile-acquisition-adapter.v3",
        "algorithm_version": "recorded-fit-algorithm",
        "lighting_profiles_sha256": "recorded-fit-source",
    }

    for profile_index, profile in enumerate(profiles):
        material_dir = tmp_path / "material" / profile
        maps_dir = tmp_path / "material" / profile / "maps"
        for name in (
            "albedo",
            "anisotropic",
            "baseColor",
            "clearcoat",
            "clearcoatGloss",
            "metallic",
            "normal",
            "roughness",
            "specular",
            "specularTint",
            "subsurface",
        ):
            write_image(maps_dir / f"{name}.png", image + profile_index * 0.05)
        (material_dir / "disney_brdf.pt").touch()

        if source_layout == "staging":
            profile_evaluation = tmp_path / "evaluation" / ".profiles" / profile
        else:
            profile_evaluation = tmp_path / "evaluation" / profile

        olat_case = profile_evaluation / "olat" / "cases" / "000002"
        for name in ("gt", "pred", "error"):
            value = image if name == "gt" else image + profile_index * 0.05
            write_image(olat_case / f"{name}.png", value)
        end2end_acquisition._write_metric_rows(
            profile_evaluation / "olat" / "metrics.csv", [{"frame_id": 2}]
        )
        hdri_case = profile_evaluation / "hdri" / "cases" / "studio_rot000"
        for name in ("lighting", "gt", "pred", "error"):
            value = image if name in {"lighting", "gt"} else image + profile_index * 0.05
            write_image(hdri_case / f"{name}.png", value)
        end2end_acquisition._write_metric_rows(
            profile_evaluation / "hdri" / "metrics.csv",
            [{"condition_id": "studio_rot000"}],
        )

        profile_results[profile] = {
            "schema": "ictpolarreal.end2end-disney.v9",
            "lighting_profile": profile,
            "adapter": dict(fit_adapter),
            "checkpoint_signature": {"adapter": dict(fit_adapter)},
            "fit_conditions": {
                "olat": 8 if profile in {"olat", "mix"} else 0,
                "hdri": 24 if profile in {"hdri", "mix"} else 0,
            },
            "evaluation": {
                "evaluations": {
                    "olat": {
                        "split": "heldout_olat",
                        "count": 1,
                        "metrics": dict(metric_names),
                        "representative": {
                            "frame_id": 2,
                            "gt_path": "cases/000002/gt.png",
                            "pred_path": "cases/000002/pred.png",
                            "error_path": "cases/000002/error.png",
                        },
                    },
                    "hdri": {
                        "split": "heldout_hdri",
                        "count": 1,
                        "olat_support_split": "heldout_olat",
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
        (material_dir / "acquisition.json").write_text(
            json.dumps(profile_results[profile]), encoding="utf-8"
        )

    evaluation_root = tmp_path / "evaluation"
    provenance_dir = (
        evaluation_root / "lighting"
        if source_layout == "legacy"
        else evaluation_root / "assets"
    )
    provenance_dir.mkdir(parents=True, exist_ok=True)
    (provenance_dir / "conditions.json").write_text("{}\n", encoding="utf-8")
    np.savez(provenance_dir / "weights.npz", weights=np.ones((1,), dtype=np.float32))
    stale_report = evaluation_root / "report"
    stale_report.mkdir(parents=True)
    (stale_report / "obsolete.png").touch()

    artifacts = end2end_acquisition._write_camera_report(
        tmp_path, profiles, profile_results
    )

    assert artifacts["material_overview"] == "material/overview.png"
    assert artifacts["overview"] == "evaluation/overview.png"
    assert artifacts["summary"] == "evaluation/summary.json"
    assert artifacts["metrics"] == "evaluation/metrics.csv"
    assert artifacts["suites"] == {
        "olat": {"comparison": "evaluation/olat/comparison.png"},
        "hdri": {"comparison": "evaluation/hdri/comparison.png"},
    }
    assert artifacts["assets"] == {
        "conditions": "evaluation/assets/conditions.json",
        "weights": "evaluation/assets/weights.npz",
    }
    artifact_paths = [
        artifacts["material_overview"],
        artifacts["overview"],
        artifacts["summary"],
        artifacts["metrics"],
        *(entry["comparison"] for entry in artifacts["suites"].values()),
        *artifacts["assets"].values(),
    ]
    for relative_path in artifact_paths:
        assert (tmp_path / relative_path).is_file()
    material_overview = read_image(tmp_path / artifacts["material_overview"])
    evaluation_overview = read_image(tmp_path / artifacts["overview"])
    assert material_overview.shape[0] >= 1200
    assert material_overview.shape[1] >= 1200
    assert evaluation_overview.shape[0] >= 1600
    assert evaluation_overview.shape[1] >= 1600

    assert {path.name for path in (tmp_path / "material").iterdir()} == {
        "overview.png",
        *profiles,
    }
    assert {path.name for path in evaluation_root.iterdir()} == {
        "assets",
        "hdri",
        "metrics.csv",
        "olat",
        "overview.png",
        "summary.json",
    }
    assert {path.name for path in (evaluation_root / "assets").iterdir()} == {
        "conditions.json",
        "weights.npz",
    }
    assert not (evaluation_root / ".profiles").exists()
    assert not (evaluation_root / "report").exists()
    assert not (evaluation_root / "mix").exists()
    assert not (evaluation_root / "lighting").exists()

    for lighting, case_id, shared_files in (
        ("olat", "000002", {"reference.png"}),
        ("hdri", "studio_rot000", {"lighting.png", "reference.png"}),
    ):
        suite_dir = evaluation_root / lighting
        assert {path.name for path in suite_dir.iterdir()} == {
            "cases",
            "comparison.png",
        }
        case_dir = suite_dir / "cases" / case_id
        assert {path.name for path in case_dir.iterdir()} == {
            *shared_files,
            "comparison.png",
            "errors",
            "predictions",
        }
        assert {path.name for path in (case_dir / "predictions").iterdir()} == {
            f"{profile}.png" for profile in profiles
        }
        assert {path.name for path in (case_dir / "errors").iterdir()} == {
            f"{profile}.png" for profile in profiles
        }
        assert read_image(case_dir / "errors" / "mix.png").mean() > read_image(
            case_dir / "errors" / "olat.png"
        ).mean()
        for obsolete in ("gt.png", "pred.png", "error.png"):
            assert not (case_dir / obsolete).exists()

    summary = json.loads((evaluation_root / "summary.json").read_text())
    assert summary["profiles"] == list(profiles)
    assert [row["training_profile"] for row in summary["rows"]] == list(profiles)
    assert all({"olat", "hdri"}.issubset(row) for row in summary["rows"])
    with (evaluation_root / "metrics.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert {row["split"] for row in rows} == {"heldout_olat", "heldout_hdri"}
    assert {int(row["count"]) for row in rows} == {1}
    assert {
        (row["training_profile"], row["evaluation_lighting"]) for row in rows
    } == {(profile, lighting) for profile in profiles for lighting in ("olat", "hdri")}

    # Report-only presentation changes can recompose the canonical tree without
    # restoring profile-first artifacts or invoking the CUDA fitter.
    assert end2end_acquisition._write_camera_report(
        tmp_path, profiles, profile_results
    ) == artifacts

    for profile in profiles:
        acquisition = json.loads(
            (tmp_path / "material" / profile / "acquisition.json").read_text(
                encoding="utf-8"
            )
        )
        assert acquisition["schema"] == "ictpolarreal.end2end-disney.v9"
        assert acquisition["adapter"] == fit_adapter
        assert acquisition["checkpoint_signature"]["adapter"] == fit_adapter
        assert end2end_acquisition._profile_outputs_complete(
            tmp_path / "material" / profile,
            evaluation_root / ".profiles" / profile,
            profile_results[profile],
        )
    missing_prediction = (
        evaluation_root / "olat" / "cases" / "000002" / "predictions" / "mix.png"
    )
    missing_prediction.unlink()
    assert not end2end_acquisition._profile_outputs_complete(
        tmp_path / "material" / "mix",
        evaluation_root / ".profiles" / "mix",
        profile_results["mix"],
    )


def test_reorganize_preserves_recorded_numerical_adapter(monkeypatch, tmp_path):
    profiles = ("olat", "hdri", "mix")
    recorded = {
        "schema": "ictpolarreal.profile-acquisition-adapter.v3",
        "algorithm_version": "recorded-fit-v2",
        "lighting_profiles_sha256": "recorded-lighting",
    }
    current = {
        "schema": "ictpolarreal.profile-acquisition-adapter.v6",
        "algorithm_version": "current-checkout-v5",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"profiles": list(profiles), "lighting": {}}),
        encoding="utf-8",
    )
    for profile in profiles:
        material_dir = tmp_path / "material" / profile
        material_dir.mkdir(parents=True)
        (material_dir / "acquisition.json").write_text(
            json.dumps(
                {
                    "adapter": current,
                    "checkpoint_signature": {"adapter": recorded},
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        end2end_acquisition,
        "_write_camera_report",
        lambda *_args: end2end_acquisition._camera_report_artifacts(),
    )
    monkeypatch.setattr(
        end2end_acquisition,
        "_adapter_provenance",
        lambda: pytest.fail("reorganization must not stamp the current adapter"),
    )

    manifest = end2end_acquisition.reorganize_end2end_camera(tmp_path)

    assert manifest["adapter"] == recorded
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["adapter"] == recorded


def test_recorded_profile_adapter_rejects_disagreement():
    first = {
        "schema": "ictpolarreal.profile-acquisition-adapter.v3",
        "algorithm_version": "fit-a",
    }
    second = {**first, "algorithm_version": "fit-b"}
    with pytest.raises(ValueError, match="disagree.*numerical adapter"):
        end2end_acquisition._recorded_profile_adapter(
            {
                "olat": {"checkpoint_signature": {"adapter": first}},
                "hdri": {"checkpoint_signature": {"adapter": second}},
            }
        )


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


def test_checkpoint_signature_allows_presentation_only_source_hash_change():
    old = {
        "profile": "olat",
        "fit_indices": [0, 1, 2],
        "regularization": {
            "kind": "edge-charbonnier",
            "weight": 0.01,
            "settings": end2end_acquisition._regularization_settings(
                "edge-charbonnier"
            ),
        },
        "adapter": {
            "schema": "ictpolarreal.profile-acquisition-adapter.v3",
            "algorithm_version": "superdimension-parity-v2",
            "end2end_acquisition_sha256": "old-source-hash",
            "lighting_profiles_sha256": "lighting-hash",
        },
    }
    current = json.loads(json.dumps(old))
    current["adapter"]["end2end_acquisition_sha256"] = "report-only-source-hash"

    assert end2end_acquisition._checkpoint_signatures_match(old, current)

    changed_algorithm = json.loads(json.dumps(current))
    changed_algorithm["adapter"]["algorithm_version"] = "new-acquisition-algorithm"
    assert not end2end_acquisition._checkpoint_signatures_match(
        old, changed_algorithm
    )

    changed_fit = json.loads(json.dumps(current))
    changed_fit["fit_indices"] = [0, 2]
    assert not end2end_acquisition._checkpoint_signatures_match(old, changed_fit)

    changed_kind = json.loads(json.dumps(current))
    changed_kind["regularization"]["kind"] = "l1"
    assert not end2end_acquisition._checkpoint_signatures_match(old, changed_kind)

    changed_settings = json.loads(json.dumps(current))
    changed_settings["regularization"]["settings"]["epsilon"] = 0.01
    assert not end2end_acquisition._checkpoint_signatures_match(
        old, changed_settings
    )


def test_imaginaire_disney_import_treats_torchvision_as_debug_only(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(sys, "path", list(sys.path))
    package = tmp_path / "CookTorrance_IBL"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    source = package / "disney_brdf.py"
    source.write_text(
        "import torchvision\n"
        "imported_torchvision = torchvision\n"
        "class DisneyBRDFSimplifiedMultiLayer:\n"
        "    pass\n",
        encoding="utf-8",
    )

    monkeypatch.delitem(sys.modules, "CookTorrance_IBL.disney_brdf", raising=False)
    monkeypatch.delitem(sys.modules, "CookTorrance_IBL", raising=False)
    previous_torchvision = sys.modules.get("torchvision")
    real_import_module = end2end_acquisition.importlib.import_module

    def import_module(name, package=None):
        if name == "torchvision":
            raise ModuleNotFoundError(
                "No module named 'torchvision'", name="torchvision"
            )
        return real_import_module(name, package)

    monkeypatch.setattr(end2end_acquisition.importlib, "import_module", import_module)
    root, module, imported_source = end2end_acquisition._load_imaginaire_disney(
        tmp_path
    )

    assert root == tmp_path.resolve()
    assert imported_source == source.resolve()
    assert module.DisneyBRDFSimplifiedMultiLayer.__name__ == (
        "DisneyBRDFSimplifiedMultiLayer"
    )
    with pytest.raises(RuntimeError, match="shadow debug image export"):
        module.imported_torchvision.utils.save_image(None, tmp_path / "debug.png")
    if previous_torchvision is None:
        assert "torchvision" not in sys.modules
    else:
        assert sys.modules["torchvision"] is previous_torchvision


@pytest.mark.parametrize(
    (
        "extra_args",
        "expected_mode",
        "expected_backend",
        "expected_steps",
        "expected_eval_lights",
        "expected_regularizer",
    ),
    [
        ([], "default", "auto", 33000, 16, "impulse-median"),
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
                "--end2end-tv-weight",
                "0.025",
                "--end2end-tv-kind",
                "edge-charbonnier",
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
            "edge-charbonnier",
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
    expected_regularizer,
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
    assert calls[0]["end2end_tv_weight"] == pytest.approx(
        0.025 if expected_mode == "end2end" else 1.25e-3
    )
    assert calls[0]["end2end_tv_kind"] == expected_regularizer
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
            "--end2end-tv-weight",
            "0.025",
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
    assert "--end2end-tv-weight 0.025" in result.stdout
    assert "--end2end-tv-kind impulse-median" in result.stdout
    assert "--end2end-eval-lights 7" in result.stdout
    command = shlex.split(result.stdout.partition(":")[2])
    assert command[command.index("--end2end-profiles") + 1] == "hdri,mix"
    assert f"--end2end-hdri-root {hdri_root}" in result.stdout
    assert "--end2end-hdri-count 5" in result.stdout
    assert "--end2end-eval-hdris 2" in result.stdout
    assert "--end2end-hdri-rotations 6" in result.stdout
    assert "--end2end-primary-profile mix" in result.stdout
    assert "--gpus-per-node=1" in result.stdout
