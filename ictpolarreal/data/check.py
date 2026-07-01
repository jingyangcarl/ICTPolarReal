from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from ictpolarreal.data.dataset import iter_camera_samples
from ictpolarreal.data.olat import numeric_image_ids, paired_light_frames


SAMPLE_DRIVE_URL = "https://drive.google.com/drive/u/1/folders/1J2lfWe8rO1ZXpbeVW68u2RSqOocCs-S6"


def _min_max(values: list[int]) -> str:
    return f"{min(values) if values else 0} {max(values) if values else 0}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an ICTPolarReal data root before processing/training.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--min-objects", type=int, default=1)
    parser.add_argument("--min-cameras", type=int, default=1)
    parser.add_argument("--min-lights", type=int, default=16)
    parser.add_argument("--require-target", default=None, help="Optional training target, e.g. albedo, normal, specular.")
    parser.add_argument("--sample-url", default=SAMPLE_DRIVE_URL)
    args = parser.parse_args()

    root = Path(args.data_root)
    if not root.exists():
        print(f"[data-check] Missing data root: {root}")
        print("[data-check] Download or copy the sample dataset, then rerun with:")
        print(f"  bash run.sh all --data-root {root}")
        print(f"[data-check] Sample Google Drive folder: {args.sample_url}")
        raise SystemExit(2)

    samples = list(iter_camera_samples(root))
    objects = sorted({sample.object_name for sample in samples})
    cameras = Counter(sample.camera for sample in samples)
    missing_static = []
    missing_mask = []
    missing_target = []
    light_counts = []
    cross_counts = []
    parallel_counts = []
    layouts = Counter()

    for sample in samples:
        if sample.image_path("static") is None:
            missing_static.append(str(sample.camera_dir))
        if sample.image_path("mask") is None:
            missing_mask.append(str(sample.camera_dir))
        if args.require_target and sample.image_path(args.require_target) is None:
            missing_target.append(str(sample.camera_dir))
        layout, pairs = paired_light_frames(sample.camera_dir)
        layouts[layout] += 1
        light_counts.append(len(pairs))
        cross_counts.append(len(numeric_image_ids(sample.camera_dir / "cross")))
        parallel_counts.append(len(numeric_image_ids(sample.camera_dir / "parallel")))

    failures = []
    if len(objects) < args.min_objects:
        failures.append(f"found {len(objects)} object(s), need at least {args.min_objects}")
    if len(samples) < args.min_cameras:
        failures.append(f"found {len(samples)} camera sample(s), need at least {args.min_cameras}")
    if missing_static:
        failures.append(f"{len(missing_static)} camera folder(s) missing static image")
    if missing_mask:
        failures.append(f"{len(missing_mask)} camera folder(s) missing mask image")
    if missing_target:
        failures.append(f"{len(missing_target)} camera folder(s) missing target `{args.require_target}`")
    if not light_counts or max(light_counts) < args.min_lights:
        failures.append(f"found max {max(light_counts) if light_counts else 0} paired OLAT light(s), need at least {args.min_lights}")

    print(f"[data-check] data_root: {root.resolve()}")
    print(f"[data-check] objects: {len(objects)}")
    print(f"[data-check] camera_samples: {len(samples)}")
    print(f"[data-check] cameras: {dict(sorted(cameras.items()))}")
    print(f"[data-check] frame_layouts: {dict(sorted(layouts.items()))}")
    print(f"[data-check] cross_frames_min_max: {_min_max(cross_counts)}")
    print(f"[data-check] parallel_frames_min_max: {_min_max(parallel_counts)}")
    print(f"[data-check] paired_olat_lights_min_max: {min(light_counts) if light_counts else 0} {max(light_counts) if light_counts else 0}")

    if failures:
        print("[data-check] Not ready:")
        for failure in failures:
            print(f"  - {failure}")
        print(f"[data-check] Prepare data from the sample Google Drive folder: {args.sample_url}")
        raise SystemExit(2)

    print("[data-check] Ready.")


if __name__ == "__main__":
    main()
