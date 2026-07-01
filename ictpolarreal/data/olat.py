from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ictpolarreal.utils.io import IMAGE_EXTS


CALIBRATED_LIGHT_COUNT = 346
RAW_FRAME_COUNT = 350
RAW_FIRST_LIGHT_FRAME = 2
RAW_LAST_LIGHT_FRAME = 347
RAW_INDICATOR_FRAMES = frozenset({0, 1, 348, 349})


@dataclass(frozen=True)
class LightFrame:
    frame_id: int
    light_index: int


def numeric_image_ids(directory: str | Path) -> set[int]:
    directory = Path(directory)
    if not directory.exists():
        return set()
    return {
        int(path.stem)
        for path in directory.iterdir()
        if path.is_file() and path.stem.isdigit() and path.suffix.lower() in IMAGE_EXTS
    }


def infer_frame_layout(*id_sets: set[int]) -> str:
    ids = set().union(*id_sets)
    if len(ids) >= RAW_FRAME_COUNT - 2 or any(frame_id >= CALIBRATED_LIGHT_COUNT for frame_id in ids):
        return "raw"
    return "normalized"


def light_frames_from_ids(ids: set[int], frame_layout: str) -> list[LightFrame]:
    if frame_layout not in {"auto", "raw", "normalized"}:
        raise ValueError("frame_layout must be auto, raw, or normalized")
    if frame_layout == "auto":
        frame_layout = infer_frame_layout(ids)
    if frame_layout == "raw":
        return [
            LightFrame(frame_id, frame_id - RAW_FIRST_LIGHT_FRAME)
            for frame_id in sorted(ids)
            if RAW_FIRST_LIGHT_FRAME <= frame_id <= RAW_LAST_LIGHT_FRAME
        ]
    return [
        LightFrame(frame_id, frame_id)
        for frame_id in sorted(ids)
        if 0 <= frame_id < CALIBRATED_LIGHT_COUNT
    ]


def paired_light_frames(camera_dir: str | Path, frame_layout: str = "auto") -> tuple[str, list[tuple[LightFrame, LightFrame]]]:
    camera_dir = Path(camera_dir)
    cross_ids = numeric_image_ids(camera_dir / "cross")
    parallel_ids = numeric_image_ids(camera_dir / "parallel")
    layout = infer_frame_layout(cross_ids, parallel_ids) if frame_layout == "auto" else frame_layout
    cross = {frame.light_index: frame for frame in light_frames_from_ids(cross_ids, layout)}
    parallel = {frame.light_index: frame for frame in light_frames_from_ids(parallel_ids, layout)}
    return layout, [(cross[index], parallel[index]) for index in sorted(cross.keys() & parallel.keys())]


def select_light_pairs(
    pairs: list[tuple[LightFrame, LightFrame]],
    *,
    light_start: int = 0,
    max_lights: int | None = None,
) -> list[tuple[LightFrame, LightFrame]]:
    selected = [pair for pair in pairs if pair[0].light_index >= light_start]
    if max_lights is None or len(selected) <= max_lights:
        return selected
    if max_lights <= 0:
        return []
    if max_lights == 1:
        return [selected[len(selected) // 2]]

    last = len(selected) - 1
    indices = [round(index * last / (max_lights - 1)) for index in range(max_lights)]
    return [selected[index] for index in indices]
