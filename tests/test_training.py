import argparse
import json

import numpy as np
import pytest

from ictpolarreal.data.training import ICTPolarRealTrainingDataset
from ictpolarreal.train.contracts import build_forward_condition, inverse_target_names
from ictpolarreal.train.diffusion import (
    _find_external_prediction,
    add_training_arguments,
    _parse_evaluation_baselines,
    _parse_evaluation_methods,
    _write_training_evaluation,
)
from ictpolarreal.utils.io import write_image


def _training_fixture(tmp_path):
    data_root = tmp_path / "data"
    material_root = tmp_path / "material_acquisition"
    camera = data_root / "object" / "cam00"
    (camera / "cross").mkdir(parents=True)
    (camera / "parallel").mkdir()

    image = np.full((8, 16, 3), 0.4, dtype=np.float32)
    write_image(camera / "static.png", image)
    write_image(camera / "static_cross.png", image * 0.5)
    write_image(camera / "static_parallel.png", image)
    write_image(camera / "mask.png", np.ones((8, 16, 1), dtype=np.float32))
    write_image(camera / "cross" / "000000.png", image * 0.25)
    write_image(camera / "parallel" / "000000.png", image * 0.75)

    brdf = material_root / "object" / "cam00" / "brdf"
    brdf.mkdir(parents=True)
    write_image(brdf / "albedo.png", image)
    write_image(brdf / "normal.png", np.dstack((np.full((8, 16), 0.5), np.full((8, 16), 0.5), np.ones((8, 16)))))
    write_image(brdf / "specular.png", np.full((8, 16, 1), 0.1, dtype=np.float32))
    return data_root, material_root


def test_rgb2x_training_dataset_contract(tmp_path):
    pytest.importorskip("torch")
    data_root, material_root = _training_fixture(tmp_path)
    dataset = ICTPolarRealTrainingDataset(
        data_root,
        material_root=material_root,
        resolution=32,
        max_lights=1,
        require_polarization_reference=True,
    )

    assert dataset.summary() == {
        "samples": 2,
        "cameras": 1,
        "static_samples": 1,
        "olat_samples": 1,
        "resolution": 32,
    }
    sample = dataset[1]
    for name in (
        "rgb",
        "albedo",
        "normal_inverse",
        "normal_forward",
        "specular",
        "cross",
        "parallel",
        "reference_cross",
        "reference_parallel",
        "irradiance",
    ):
        assert sample[name].shape == (3, 16, 32)
        assert sample[name].isfinite().all()
    assert sample["mask"].shape == (1, 16, 32)
    assert sample["light_index"] == 0
    assert sample["frame_id"] == 0


def test_rgb2x_inverse_and_forward_contracts(tmp_path):
    torch = pytest.importorskip("torch")
    functional = pytest.importorskip("torch.nn.functional")
    data_root, material_root = _training_fixture(tmp_path)
    sample = ICTPolarRealTrainingDataset(
        data_root,
        material_root=material_root,
        resolution=32,
        max_lights=1,
    )[1]
    batch = {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value for key, value in sample.items()}

    assert inverse_target_names("pbr") == ("albedo", "normal", "specular")
    assert inverse_target_names("polarization") == ("cross", "parallel")
    assert len(inverse_target_names("both")) == 5

    def encode(image):
        latent = functional.interpolate(image[:, :1], size=(2, 4), mode="nearest")
        return latent.repeat(1, 4, 1, 1)

    gbuffer = build_forward_condition(batch, mode="gbuffer", encode=encode, latent_hw=(2, 4))
    polarization = build_forward_condition(batch, mode="polarization", encode=encode, latent_hw=(2, 4))
    assert gbuffer.shape == (1, 19, 2, 4)
    assert polarization.shape == (1, 19, 2, 4)


def test_training_evaluation_writes_method_history(tmp_path):
    target_path = tmp_path / "target.png"
    write_image(target_path, np.full((8, 8, 3), 0.5, dtype=np.float32))
    rows = [
        {
            "step": 5,
            "stage": "inverse",
            "method": method,
            "task": "albedo",
            "object": "object",
            "camera": "cam00",
            "light": "static",
            "prediction": str(tmp_path / method / "albedo.png"),
            "target": str(target_path),
            "mse": value,
            "mae": value,
            "psnr": 20.0,
            "ssim": 0.8,
        }
        for method, value in (("rgb2x", 0.2), ("ours", 0.1))
    ]
    for row in rows:
        write_image(row["prediction"], np.full((8, 8, 3), 0.4, dtype=np.float32))
    step_root = tmp_path / "eval" / "step-000005"
    _write_training_evaluation(
        rows,
        step_root=step_root,
        output_dir=tmp_path,
        step=5,
        method_status={
            "rgb2x": {"status": "evaluated", "count": 1},
            "ours": {"status": "evaluated", "count": 1},
            "dsine": {"status": "skipped", "reason": "not configured"},
        },
    )

    assert (step_root / "metrics.csv").exists()
    assert (
        step_root
        / "comparisons"
        / "static"
        / "object__cam00__static__albedo.png"
    ).exists()
    video_path = step_root / "videos" / "object__cam00__albedo__static.mp4"
    assert video_path.stat().st_size > 0
    assert (step_root / "README.md").exists()
    summary = json.loads((step_root / "summary.json").read_text())
    assert summary["methods"]["rgb2x"]["label"] == "RGB2X"
    assert summary["methods"]["ours"]["label"] == "Ours"
    assert summary["methods"]["dsine"]["status"] == "skipped"
    assert summary["videos"] == {
        "object/cam00/albedo/static": "videos/object__cam00__albedo__static.mp4"
    }
    assert len((tmp_path / "eval" / "history.jsonl").read_text().splitlines()) == 1


def test_evaluation_method_and_baseline_parsing(tmp_path):
    methods = _parse_evaluation_methods(
        "pretrained,finetuned,diffusion-renderer,lotus,dsine,lotus"
    )
    assert methods == (
        "rgb2x",
        "ours",
        "diffusion_renderer",
        "lotus",
        "dsine",
    )
    assert _parse_evaluation_baselines([f"lotus={tmp_path}"]) == {"lotus": tmp_path}
    with pytest.raises(ValueError, match="Unknown evaluation method"):
        _parse_evaluation_methods("unknown")
    with pytest.raises(ValueError, match="METHOD=PATH"):
        _parse_evaluation_baselines(["lotus"])


def test_release_training_schedule_defaults():
    parser = add_training_arguments(argparse.ArgumentParser(), stage="inverse")
    args = parser.parse_args(
        [
            "--data-root",
            "data",
            "--material-root",
            "materials",
            "--out-dir",
            "outputs",
        ]
    )

    assert args.max_steps == 100000
    assert args.checkpointing_steps == 50000
    assert args.evaluation_steps == 50000
    assert args.evaluation_methods == "rgb2x,ours,diffusion_renderer,lotus,dsine"


def test_external_prediction_layouts(tmp_path):
    normal_path = tmp_path / "predictions" / "object" / "cam00" / "static" / "normal.png"
    write_image(normal_path, np.full((8, 8, 3), 0.5, dtype=np.float32))
    sample = {"object": "object", "camera": "cam00", "frame_id": -1}

    assert _find_external_prediction(tmp_path, sample=sample, task="normal") == normal_path
    assert _find_external_prediction(tmp_path, sample=sample, task="albedo") is None

    forward_path = (
        tmp_path
        / "object"
        / "cam00"
        / "hdri_studio"
        / "forward_rgb.png"
    )
    write_image(forward_path, np.full((8, 8, 3), 0.25, dtype=np.float32))
    forward_sample = {
        "object": "object",
        "camera": "cam00",
        "frame_id": -1,
        "lighting_name": "hdri_studio",
    }
    assert (
        _find_external_prediction(
            tmp_path,
            sample=forward_sample,
            task="forward_gbuffer",
        )
        == forward_path
    )
