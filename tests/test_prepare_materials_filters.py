from types import SimpleNamespace

from ictpolarreal.processing.prepare_materials import _filter_samples


def _sample(object_name: str, camera: str):
    return SimpleNamespace(object_name=object_name, camera=camera)


def test_filter_samples_selects_exact_object_and_camera_in_dataset_order():
    samples = [
        _sample("candle", "cam00"),
        _sample("candle", "cam07"),
        _sample("clorox", "cam00"),
        _sample("clorox", "cam07"),
    ]

    selected = _filter_samples(
        samples,
        object_names=["clorox"],
        camera_names=["cam07"],
    )

    assert [(item.object_name, item.camera) for item in selected] == [
        ("clorox", "cam07")
    ]


def test_filter_samples_allows_repeated_filters_and_unfiltered_dimension():
    samples = [
        _sample("candle", "cam00"),
        _sample("candle", "cam01"),
        _sample("clorox", "cam00"),
    ]

    selected = _filter_samples(
        samples,
        object_names=["candle", "candle"],
        camera_names=[],
    )

    assert [(item.object_name, item.camera) for item in selected] == [
        ("candle", "cam00"),
        ("candle", "cam01"),
    ]


def test_filter_samples_with_no_filters_preserves_all_samples():
    samples = [_sample("candle", "cam00"), _sample("clorox", "cam07")]

    assert _filter_samples(
        samples, object_names=[], camera_names=[]
    ) == samples
