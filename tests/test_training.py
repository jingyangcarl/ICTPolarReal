import numpy as np
import pytest

from ictpolarreal.data.training import ICTPolarRealTrainingDataset
from ictpolarreal.train.contracts import build_forward_condition, inverse_target_names
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
