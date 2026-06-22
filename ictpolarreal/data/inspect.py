from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from ictpolarreal.data.dataset import iter_camera_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an ICTPolarReal dataset root.")
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()

    samples = list(iter_camera_samples(args.data_root))
    objects = sorted({sample.object_name for sample in samples})
    cameras = Counter(sample.camera for sample in samples)
    light_counts = []
    for sample in samples:
        parallel_dir = sample.camera_dir / "parallel"
        if parallel_dir.exists():
            light_counts.append(len(list(parallel_dir.glob("*"))))

    print(f"data_root: {Path(args.data_root).resolve()}")
    print(f"objects: {len(objects)}")
    print(f"camera_samples: {len(samples)}")
    print(f"cameras: {dict(sorted(cameras.items()))}")
    if light_counts:
        print(f"parallel_lights_min_max: {min(light_counts)} {max(light_counts)}")


if __name__ == "__main__":
    main()

