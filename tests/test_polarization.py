import numpy as np
import pytest

from ictpolarreal.data.dataset import ICTPolarRealDataset
from ictpolarreal.data.polarization import separate_cross_parallel, synthesize_olat
from ictpolarreal.processing.material_decomposition import decompose_polarized_olat
from ictpolarreal.utils.io import write_image


def test_separate_cross_parallel():
    cross = np.ones((2, 2, 3), dtype=np.float32) * 0.2
    parallel = np.ones((2, 2, 3), dtype=np.float32) * 0.5
    diffuse, specular = separate_cross_parallel(cross, parallel)
    np.testing.assert_allclose(diffuse, 0.4)
    np.testing.assert_allclose(specular, 0.6)


def test_synthesize_olat():
    images = np.ones((2, 2, 2, 3), dtype=np.float32)
    weights = np.array([0.25, 0.75], dtype=np.float32)
    out = synthesize_olat(images, weights)
    np.testing.assert_allclose(out, 1.0)


def test_material_decomposition_shapes():
    cross = np.ones((4, 2, 2, 3), dtype=np.float32) * 0.1
    parallel = cross + 0.05
    lights = np.array(
        [
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, -1.0, 1.0],
        ],
        dtype=np.float32,
    )
    maps = decompose_polarized_olat(cross, parallel, lights, backend="cpu")
    assert maps.diffuse_albedo.shape == (2, 2, 3)
    assert maps.diffuse_normal.shape == (2, 2, 3)
    assert maps.specular_albedo.shape == (2, 2, 1)
    assert maps.roughness.shape == (2, 2, 1)
    assert np.isfinite(maps.sigma).all()


def test_dataset_polarization_and_gbuffer_modes(tmp_path):
    pytest.importorskip("torch")
    data_root = tmp_path / "data"
    material_root = tmp_path / "materials"
    cam = data_root / "object" / "cam00"
    (cam / "cross").mkdir(parents=True)
    (cam / "parallel").mkdir()
    image = np.ones((4, 4, 3), dtype=np.float32) * 0.4
    write_image(cam / "static.png", image)
    write_image(cam / "albedo.png", image)
    write_image(cam / "mask.png", np.ones((4, 4, 1), dtype=np.float32))
    write_image(cam / "cross" / "000000.png", image * 0.5)
    write_image(cam / "parallel" / "000000.png", image)

    mats = material_root / "object" / "cam00" / "material_properties"
    mats.mkdir(parents=True)
    write_image(mats / "albedo.png", image)
    write_image(mats / "normal.png", image)
    write_image(mats / "roughness.png", np.ones((4, 4, 1), dtype=np.float32) * 0.2)
    write_image(mats / "specular.png", np.ones((4, 4, 1), dtype=np.float32) * 0.1)

    inverse = ICTPolarRealDataset(data_root, input_mode="polarization", target_mode="image", target_name="albedo")
    assert inverse[0]["image"].shape[0] == 15
    assert inverse[0]["target"].shape[0] == 3

    forward = ICTPolarRealDataset(
        data_root,
        input_name="gbuffer",
        target_name="static",
        input_mode="gbuffer",
        target_mode="image",
        material_root=material_root,
    )
    assert forward[0]["image"].shape[0] == 8
    assert forward[0]["target"].shape[0] == 3
