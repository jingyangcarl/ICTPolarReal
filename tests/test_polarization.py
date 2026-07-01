import numpy as np
import pytest

from ictpolarreal.data.dataset import ICTPolarRealDataset
from ictpolarreal.data.olat import LightFrame, light_frames_from_ids, select_light_pairs
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


def test_raw_capture_frames_exclude_indicators_and_subsample_globally():
    frames = light_frames_from_ids(set(range(350)), "auto")
    assert len(frames) == 346
    assert frames[0] == LightFrame(frame_id=2, light_index=0)
    assert frames[-1] == LightFrame(frame_id=347, light_index=345)

    pairs = [(frame, frame) for frame in frames]
    selected = select_light_pairs(pairs, max_lights=8)
    assert selected[0][0].light_index == 0
    assert selected[-1][0].light_index == 345
    assert len({pair[0].light_index for pair in selected}) == 8


def test_material_decomposition_recovers_synthetic_diffuse_material():
    rng = np.random.default_rng(7)
    lights = rng.normal(size=(64, 3)).astype(np.float32)
    lights /= np.linalg.norm(lights, axis=1, keepdims=True)
    expected_normal = np.array([0.25, -0.15, 0.956], dtype=np.float32)
    expected_normal /= np.linalg.norm(expected_normal)
    expected_albedo = np.array([0.3, 0.18, 0.08], dtype=np.float32)
    cosine = np.maximum(lights @ expected_normal, 0.0)
    cross = 0.5 * cosine[:, None] * expected_albedo[None]

    half = lights + np.array([0.0, 0.0, 1.0], dtype=np.float32)
    half /= np.linalg.norm(half, axis=1, keepdims=True) + 1e-8
    specular = 0.03 * np.exp(-40.0 * (1.0 - np.maximum(half @ expected_normal, 0.0)))
    cross = np.broadcast_to(cross[:, None, None], (64, 2, 2, 3)).copy()
    parallel = cross + specular[:, None, None, None]

    maps = decompose_polarized_olat(cross, parallel, lights, backend="cpu", noise=1e-6)
    assert np.dot(maps.diffuse_normal[0, 0], expected_normal) > 0.99
    np.testing.assert_allclose(maps.diffuse_albedo[0, 0], expected_albedo, rtol=0.05, atol=0.01)
    assert maps.specular_albedo.max() > 0.0
    assert np.isfinite(maps.roughness).all()


def test_torch_material_optimizer_recovers_synthetic_normal():
    pytest.importorskip("torch")
    rng = np.random.default_rng(11)
    lights = rng.normal(size=(48, 3)).astype(np.float32)
    lights /= np.linalg.norm(lights, axis=1, keepdims=True)
    expected_normal = np.array([-0.2, 0.3, 0.933], dtype=np.float32)
    expected_normal /= np.linalg.norm(expected_normal)
    albedo = np.array([0.22, 0.3, 0.12], dtype=np.float32)
    cross = 0.5 * np.maximum(lights @ expected_normal, 0.0)[:, None] * albedo[None]
    cross = np.broadcast_to(cross[:, None, None], (48, 1, 1, 3)).copy()
    parallel = cross + 0.01
    maps = decompose_polarized_olat(
        cross,
        parallel,
        lights,
        backend="torch",
        device="cpu",
        noise=1e-6,
        normal_steps=12,
        sigma_steps=8,
        chunk_size=1,
    )
    assert np.dot(maps.diffuse_normal[0, 0], expected_normal) > 0.98
    assert np.isfinite(maps.sigma).all()


def test_dataset_polarization_and_gbuffer_modes(tmp_path):
    pytest.importorskip("torch")
    data_root = tmp_path / "data"
    material_root = tmp_path / "material_acquisition"
    cam = data_root / "object" / "cam00"
    (cam / "cross").mkdir(parents=True)
    (cam / "parallel").mkdir()
    image = np.ones((4, 4, 3), dtype=np.float32) * 0.4
    write_image(cam / "static.png", image)
    write_image(cam / "albedo.png", image)
    write_image(cam / "mask.png", np.ones((4, 4, 1), dtype=np.float32))
    write_image(cam / "cross" / "000000.png", image * 0.5)
    write_image(cam / "parallel" / "000000.png", image)

    mats = material_root / "object" / "cam00" / "brdf"
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
