from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from ictpolarreal.data.dataset import iter_camera_samples
from ictpolarreal.processing.material_decomposition import decompose_camera_sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize polarized OLAT captures into diffuse/specular material maps."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--max-lights", type=int, default=None)
    parser.add_argument("--light-start", type=int, default=0)
    parser.add_argument("--light-root", default=None, help="Optional folder containing LSX3_light_positions.txt and LSX3_light_z_spiral.txt.")
    parser.add_argument("--backend", choices=["auto", "cpu", "torch"], default="auto")
    parser.add_argument("--device", default="cuda", help="Torch device used when --backend torch/auto can use PyTorch.")
    parser.add_argument("--noise", type=float, default=1.5e-3, help="Radiance threshold used for robust normal and roughness fitting.")
    parser.add_argument("--frame-layout", choices=["auto", "raw", "normalized"], default="auto")
    parser.add_argument("--normal-steps", type=int, default=30)
    parser.add_argument("--sigma-steps", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=4096)
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
            noise=args.noise,
            frame_layout=args.frame_layout,
            normal_steps=args.normal_steps,
            sigma_steps=args.sigma_steps,
            chunk_size=args.chunk_size,
        )
        if used:
            processed += 1
            light_count += used
    print(f"[process] decomposed cameras: {processed}")
    print(f"[process] paired OLAT images used: {light_count}")
    print(f"[process] material maps: {out_root}")


if __name__ == "__main__":
    main()
