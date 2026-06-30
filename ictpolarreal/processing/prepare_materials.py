from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from ictpolarreal.data.dataset import iter_camera_samples
from ictpolarreal.processing.material_decomposition import decompose_camera_sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose polarized OLAT captures into diffuse/specular material maps and previews."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--max-lights", type=int, default=None)
    parser.add_argument("--light-start", type=int, default=0)
    parser.add_argument("--light-root", default=None, help="Optional folder containing LSX3_light_positions.txt and LSX3_light_z_spiral.txt.")
    parser.add_argument("--preview", action="store_true", help="Write PNG previews next to EXR material maps.")
    parser.add_argument("--backend", choices=["auto", "cpu", "torch"], default="auto")
    parser.add_argument("--device", default="cuda", help="Torch device used when --backend torch/auto can use PyTorch.")
    parser.add_argument("--save-aggregate", action="store_true", help="Save mean diffuse/specular OLAT images per camera.")
    parser.add_argument("--noise", type=float, default=1.5e-3, help="Radiance threshold used for robust normal and roughness fitting.")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    samples = list(iter_camera_samples(args.data_root))
    processed = 0
    light_count = 0
    for sample in tqdm(samples, desc="decompose cameras"):
        used = decompose_camera_sample(
            sample,
            data_root=args.data_root,
            out_root=out_root,
            light_start=args.light_start,
            max_lights=args.max_lights,
            light_root=args.light_root,
            backend=args.backend,
            device=args.device,
            preview=args.preview,
            save_aggregate=args.save_aggregate,
            noise=args.noise,
        )
        if used:
            processed += 1
            light_count += used
    print(f"[process] decomposed cameras: {processed}")
    print(f"[process] paired OLAT images used: {light_count}")
    print(f"[process] material maps: {out_root}")


if __name__ == "__main__":
    main()
