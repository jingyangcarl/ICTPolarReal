import numpy as np

from ictpolarreal.data.forward_evaluation import (
    ICTPolarRealForwardEvaluationDataset,
    _sample_lightstage_environment,
)
from ictpolarreal.utils.io import read_image, write_image


def _forward_fixture(tmp_path):
    data_root = tmp_path / "data"
    material_root = tmp_path / "materials"
    hdri_root = tmp_path / "hdri"
    camera = data_root / "object" / "cam00"
    (camera / "cross").mkdir(parents=True)
    (camera / "parallel").mkdir()
    hdri_root.mkdir()

    image = np.full((8, 16, 3), 0.4, dtype=np.float32)
    write_image(camera / "static.png", image)
    write_image(camera / "static_cross.png", image * 0.5)
    write_image(camera / "static_parallel.png", image)
    write_image(camera / "mask.png", np.ones((8, 16, 1), dtype=np.float32))
    for light in range(3):
        write_image(camera / "cross" / f"{light:06d}.png", image * (0.1 + light * 0.1))
        write_image(camera / "parallel" / f"{light:06d}.png", image * (0.4 + light * 0.1))

    brdf = material_root / "object" / "cam00" / "brdf"
    brdf.mkdir(parents=True)
    write_image(brdf / "albedo.png", image)
    normal = np.dstack(
        (
            np.full((8, 16), 0.5),
            np.full((8, 16), 0.5),
            np.ones((8, 16)),
        )
    )
    write_image(brdf / "normal.png", normal)
    write_image(brdf / "specular.png", np.full((8, 16, 1), 0.1, dtype=np.float32))
    write_image(hdri_root / "studio_a.png", np.full((8, 16, 3), 0.25, dtype=np.float32))
    write_image(hdri_root / "studio_b.png", np.full((8, 16, 3), 0.75, dtype=np.float32))
    return data_root, material_root, hdri_root


def test_forward_evaluation_covers_fixed_olat_and_hdri(tmp_path):
    data_root, material_root, hdri_root = _forward_fixture(tmp_path)
    dataset = ICTPolarRealForwardEvaluationDataset(
        data_root,
        material_root=material_root,
        hdri_root=hdri_root,
        resolution=32,
        olat_samples=2,
        hdri_samples=2,
        hdri_olat_lights=2,
    )

    assert dataset.summary() == {
        "samples": 4,
        "cameras": 1,
        "olat_samples": 2,
        "hdri_samples": 2,
        "resolution": 32,
    }
    assert dataset.evaluation_indices(1) == [0, 1, 2, 3]
    assert {dataset[index]["lighting_type"] for index in range(len(dataset))} == {
        "olat",
        "hdri",
    }
    hdri_sample = dataset[2]
    assert hdri_sample["rgb"].shape == (3, 16, 32)
    assert hdri_sample["rgb"].isfinite().all()
    assert hdri_sample["irradiance"].isfinite().all()

    environment_path = dataset.export_environment(0, tmp_path / "cache")
    environment = read_image(environment_path, channels=3)
    assert environment.shape == (256, 512, 3)
    assert environment.max() == 1.0
    assert environment.min() == 0.0

    expected_label = dataset.light_order[0]
    np.testing.assert_array_equal(
        environment[..., 0] > 0.5,
        dataset.light_mapping == expected_label,
    )


def test_lightstage_environment_sampling_uses_z_spiral_mapping_labels():
    mapping = np.asarray([[1, 1, 2], [3, 3, 3]], dtype=np.int32)
    order = np.asarray([1, 3], dtype=np.int32)
    environment = np.asarray(
        [
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            [[0.0, 0.0, 2.0], [0.0, 0.0, 4.0], [0.0, 0.0, 6.0]],
        ],
        dtype=np.float32,
    )

    sampled = _sample_lightstage_environment(
        environment,
        np.asarray([0, 1]),
        mapping,
        order,
    )

    np.testing.assert_allclose(sampled, [[2.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
