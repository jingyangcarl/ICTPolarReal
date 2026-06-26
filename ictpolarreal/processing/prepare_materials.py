from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ictpolarreal.data.dataset import iter_camera_samples
from ictpolarreal.data.polarization import apply_mask, hdr_to_ldr, separate_cross_parallel
from ictpolarreal.utils.io import read_image, write_image


def _separate_backend(cross, parallel, backend: str, device: str):
    if backend == "cpu":
        return separate_cross_parallel(cross, parallel)
    try:
        import torch
    except ModuleNotFoundError:
        if backend == "torch":
            raise
        return separate_cross_parallel(cross, parallel)

    if backend == "auto" and device.startswith("cuda") and not torch.cuda.is_available():
        return separate_cross_parallel(cross, parallel)

    torch_device = torch.device(device if backend == "torch" or torch.cuda.is_available() else "cpu")
    cross_t = torch.from_numpy(cross.astype(np.float32)).to(torch_device)
    parallel_t = torch.from_numpy(parallel.astype(np.float32)).to(torch_device)
    diffuse_t = 2.0 * cross_t
    specular_t = 2.0 * torch.clamp(parallel_t - cross_t, min=0.0)
    return diffuse_t.cpu().numpy(), specular_t.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare diffuse/specular previews from polarized OLAT captures.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--max-lights", type=int, default=None)
    parser.add_argument("--light-start", type=int, default=1)
    parser.add_argument("--preview", action="store_true", help="Tone-map outputs to PNG previews.")
    parser.add_argument("--backend", choices=["auto", "cpu", "torch"], default="auto")
    parser.add_argument("--device", default="cuda", help="Torch device used when --backend torch/auto can use PyTorch.")
    parser.add_argument("--save-aggregate", action="store_true", help="Save mean diffuse/specular material-property previews per camera.")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    for sample in tqdm(list(iter_camera_samples(args.data_root)), desc="camera samples"):
        mask_path = sample.image_path("mask")
        mask = read_image(mask_path, channels=1) if mask_path else None
        light_ids = range(args.light_start, args.light_start + args.max_lights) if args.max_lights else range(args.light_start, 347)
        diffuse_accum = []
        specular_accum = []
        for light_id in light_ids:
            cross_path = sample.light_path("cross", light_id)
            parallel_path = sample.light_path("parallel", light_id)
            if cross_path is None or parallel_path is None:
                continue
            diffuse, specular = _separate_backend(read_image(cross_path), read_image(parallel_path), args.backend, args.device)
            if args.save_aggregate:
                diffuse_accum.append(diffuse)
                specular_accum.append(specular)
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
        if args.save_aggregate and diffuse_accum:
            mean_diffuse = np.mean(np.stack(diffuse_accum, axis=0), axis=0)
            mean_specular = np.mean(np.stack(specular_accum, axis=0), axis=0)
            if args.preview:
                mean_diffuse = hdr_to_ldr(mean_diffuse)
                mean_specular = hdr_to_ldr(mean_specular)
            mean_diffuse = apply_mask(mean_diffuse, mask)
            mean_specular = apply_mask(mean_specular, mask)
            material_dir = out_root / sample.object_name / sample.camera / "material_properties"
            write_image(material_dir / "mean_diffuse.png", mean_diffuse)
            write_image(material_dir / "mean_specular.png", mean_specular)


if __name__ == "__main__":
    main()
