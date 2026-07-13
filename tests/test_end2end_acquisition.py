from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ictpolarreal.processing import end2end_acquisition, prepare_materials
from ictpolarreal.utils.io import read_image


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
    ("extra_args", "expected_mode", "expected_backend", "expected_steps"),
    [
        ([], "default", "auto", 33000),
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
            ],
            "end2end",
            "torch",
            17,
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
):
    sample = object()
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
    assert calls[0]["end2end_learning_rate"] == pytest.approx(
        0.002 if expected_mode == "end2end" else 1e-3
    )


def test_slurm_dry_run_propagates_end2end_selector_and_options(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    data_root = tmp_path / "data"
    output_root = tmp_path / "outputs"
    imaginaire_root = tmp_path / "imaginaire"
    imaginaire_root.mkdir()
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
    assert "--gpus-per-node=1" in result.stdout
