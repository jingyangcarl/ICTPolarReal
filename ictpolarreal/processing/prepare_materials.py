from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from ictpolarreal.data.dataset import iter_camera_samples
from ictpolarreal.data.polarization import apply_mask, hdr_to_ldr, separate_cross_parallel
from ictpolarreal.utils.io import read_image, write_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare diffuse/specular previews from polarized OLAT captures.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--max-lights", type=int, default=None)
    parser.add_argument("--light-start", type=int, default=1)
    parser.add_argument("--preview", action="store_true", help="Tone-map outputs to PNG previews.")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    for sample in tqdm(list(iter_camera_samples(args.data_root)), desc="camera samples"):
        mask_path = sample.image_path("mask")
        mask = read_image(mask_path, channels=1) if mask_path else None
        light_ids = range(args.light_start, args.light_start + args.max_lights) if args.max_lights else range(args.light_start, 347)
        for light_id in light_ids:
            cross_path = sample.light_path("cross", light_id)
            parallel_path = sample.light_path("parallel", light_id)
            if cross_path is None or parallel_path is None:
                continue
            diffuse, specular = separate_cross_parallel(read_image(cross_path), read_image(parallel_path))
            if args.preview:
                diffuse = hdr_to_ldr(diffuse)
                specular = hdr_to_ldr(specular)
                suffix = ".png"
            else:
                suffix = ".exr"
            diffuse = apply_mask(diffuse, mask)
            specular = apply_mask(specular, mask)
            base = out_root / sample.object_name / sample.camera / f"{light_id:06d}"
            write_image(base / f"diffuse{suffix}", diffuse)
            write_image(base / f"specular{suffix}", specular)


if __name__ == "__main__":
    main()

