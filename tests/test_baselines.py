import json
import os
from pathlib import Path

import numpy as np

from ictpolarreal.eval.baselines import (
    BASELINE_TASKS,
    _method_complete,
    _worker_environment,
    prepare_manifest,
)
from ictpolarreal.utils.io import read_image, write_image


def test_prepare_baseline_manifest_and_cache_contract(tmp_path):
    camera = tmp_path / "data" / "object" / "cam00"
    camera.mkdir(parents=True)
    image = np.full((8, 6, 3), 0.5, dtype=np.float32)
    write_image(camera / "static.png", image)

    out_root = tmp_path / "baselines"
    manifest_path, samples, forward_samples = prepare_manifest(
        camera.parents[1], out_root, max_samples=1
    )

    payload = json.loads(manifest_path.read_text())
    assert payload["samples"] == samples
    assert payload["stages"] == ["inverse"]
    assert payload["forward_samples"] == forward_samples == []
    assert samples[0]["object"] == "object"
    staged = read_image(samples[0]["input"])
    assert staged.shape == image.shape
    assert 0.0 <= staged.min() <= staged.max() <= 1.0

    for method, tasks in BASELINE_TASKS.items():
        assert not _method_complete(out_root, method, samples)
        for task in tasks:
            path = out_root / method / "object" / "cam00" / "static" / f"{task}.png"
            write_image(path, np.zeros_like(image))
        assert _method_complete(out_root, method, samples)


def test_forward_cache_contract(tmp_path):
    samples = [{"object": "object", "camera": "cam00", "input": "static.png"}]
    forward_samples = [
        {
            "object": "object",
            "camera": "cam00",
            "lighting_type": "olat",
            "lighting_name": "olat_000000",
            "environment": "olat.exr",
        },
        {
            "object": "object",
            "camera": "cam00",
            "lighting_type": "hdri",
            "lighting_name": "hdri_studio",
            "environment": "studio.exr",
        },
    ]
    root = tmp_path / "baselines"

    assert not _method_complete(
        root,
        "diffusion_renderer",
        samples,
        stages=("forward",),
        forward_samples=forward_samples,
    )
    for sample in forward_samples:
        write_image(
            root
            / "diffusion_renderer"
            / sample["object"]
            / sample["camera"]
            / sample["lighting_name"]
            / "forward_rgb.png",
            np.zeros((4, 4, 3), dtype=np.float32),
        )
    assert _method_complete(
        root,
        "diffusion_renderer",
        samples,
        stages=("forward",),
        forward_samples=forward_samples,
    )


def test_worker_environment_uses_selected_python_prefix(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    python = tmp_path / "env" / "bin" / "python"
    repo = tmp_path / "repo"
    environment = _worker_environment(repo, str(python))

    assert environment["CONDA_PREFIX"] == str(tmp_path / "env")
    assert environment["CUDA_HOME"] == str(tmp_path / "env")
    assert environment["PATH"].split(os.pathsep)[0] == str(python.parent)
    assert str(Path(__file__).resolve().parents[1]) in environment["PYTHONPATH"]
    assert str(repo) in environment["PYTHONPATH"]
