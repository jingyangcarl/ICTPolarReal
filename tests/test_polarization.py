import numpy as np

from ictpolarreal.data.polarization import separate_cross_parallel, synthesize_olat


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

