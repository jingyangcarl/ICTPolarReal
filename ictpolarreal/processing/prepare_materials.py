from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from ictpolarreal.data.dataset import iter_camera_samples
from ictpolarreal.processing.lighting_profiles import parse_lighting_profiles
from ictpolarreal.processing.material_decomposition import decompose_camera_sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize polarized OLAT captures into diffuse/specular material maps."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--max-lights", type=int, default=None)
    parser.add_argument("--light-start", type=int, default=0)
    parser.add_argument(
        "--light-root",
        default=None,
        help=(
            "Optional folder containing LSX3_light_positions.txt and "
            "LSX3_light_z_spiral.txt."
        ),
    )
    parser.add_argument("--backend", choices=["auto", "cpu", "torch"], default="auto")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device used when --backend torch/auto can use PyTorch.",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=1.5e-3,
        help="Radiance threshold used for robust normal and roughness fitting.",
    )
    parser.add_argument("--frame-layout", choices=["auto", "raw", "normalized"], default="auto")
    parser.add_argument("--normal-steps", type=int, default=30)
    parser.add_argument("--sigma-steps", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument(
        "--material-acquisition",
        choices=["default", "end2end"],
        default="default",
        help="default polarized Ward fit or Imaginaire Disney end-to-end fit.",
    )
    parser.add_argument(
        "--imaginaire-root",
        default=str(Path(__file__).resolve().parents[3] / "imaginaire"),
        help="External Imaginaire checkout used only by end2end acquisition.",
    )
    parser.add_argument("--end2end-steps", type=int, default=33000)
    parser.add_argument("--end2end-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--end2end-eval-lights",
        type=int,
        default=16,
        help="OLATs held out from end2end initialization/fitting for relighting evaluation.",
    )
    parser.add_argument(
        "--end2end-profiles",
        default="olat,hdri,mix",
        help="Comma-separated independent Disney fits: olat,hdri,mix; 'all' is an alias.",
    )
    parser.add_argument(
        "--end2end-hdri-root",
        default=None,
        help="Folder of HDR/EXR environment maps projected onto the ICT light basis.",
    )
    parser.add_argument(
        "--end2end-hdri-count",
        type=int,
        default=100,
        help=(
            "Natural HDRI identities used for fitting; w/r/g/b calibrations "
            "are added separately."
        ),
    )
    parser.add_argument("--end2end-eval-hdris", type=int, default=4)
    parser.add_argument("--end2end-hdri-rotations", type=int, default=4)
    parser.add_argument(
        "--end2end-primary-profile",
        choices=["olat", "hdri", "mix"],
        default="olat",
        help="Profile exposed to downstream material consumers through manifest.json.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    samples = list(iter_camera_samples(args.data_root))
    profiles: tuple[str, ...] = ()
    if args.material_acquisition == "end2end":
        if args.end2end_steps <= 0:
            parser.error("--end2end-steps must be positive")
        if args.end2end_learning_rate <= 0:
            parser.error("--end2end-learning-rate must be positive")
        if args.end2end_eval_lights < 0:
            parser.error("--end2end-eval-lights must be non-negative")
        if args.end2end_hdri_count <= 0:
            parser.error("--end2end-hdri-count must be positive")
        if args.end2end_eval_hdris < 0:
            parser.error("--end2end-eval-hdris must be non-negative")
        if args.end2end_hdri_rotations <= 0:
            parser.error("--end2end-hdri-rotations must be positive")
        profiles = parse_lighting_profiles(args.end2end_profiles)
        if args.end2end_primary_profile not in profiles:
            parser.error(
                "--end2end-primary-profile must be included in --end2end-profiles"
            )
        if args.end2end_hdri_root is None:
            parser.error("--end2end-hdri-root is required for end2end acquisition")
    run_manifest = {
        "schema": "ictpolarreal.material-acquisition-run.v1",
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "material_acquisition": args.material_acquisition,
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "profiles": list(profiles),
        "primary_profile": (
            args.end2end_primary_profile
            if args.material_acquisition == "end2end"
            else None
        ),
        "settings": {
            "backend": args.backend,
            "device": args.device,
            "frame_layout": args.frame_layout,
            "max_lights": args.max_lights,
            "light_start": args.light_start,
            "imaginaire_root": str(Path(args.imaginaire_root).expanduser().resolve()),
            "end2end_steps": args.end2end_steps,
            "end2end_learning_rate": args.end2end_learning_rate,
            "end2end_eval_lights": args.end2end_eval_lights,
            "end2end_hdri_root": (
                str(Path(args.end2end_hdri_root).expanduser().resolve())
                if args.end2end_hdri_root is not None
                else None
            ),
            "end2end_hdri_count": args.end2end_hdri_count,
            "end2end_eval_hdris": args.end2end_eval_hdris,
            "end2end_hdri_rotations": args.end2end_hdri_rotations,
        },
        "cameras": [],
    }
    _write_run_manifest(out_root, run_manifest)
    print(f"[process] material acquisition: {args.material_acquisition}")
    processed = 0
    light_count = 0
    current_sample = None
    try:
        for sample in tqdm(samples, desc="decompose cameras"):
            current_sample = sample
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
                material_acquisition=args.material_acquisition,
                imaginaire_root=args.imaginaire_root,
                end2end_steps=args.end2end_steps,
                end2end_learning_rate=args.end2end_learning_rate,
                end2end_eval_lights=args.end2end_eval_lights,
                end2end_profiles=args.end2end_profiles,
                end2end_hdri_root=args.end2end_hdri_root,
                end2end_hdri_count=args.end2end_hdri_count,
                end2end_eval_hdris=args.end2end_eval_hdris,
                end2end_hdri_rotations=args.end2end_hdri_rotations,
                end2end_primary_profile=args.end2end_primary_profile,
            )
            if used:
                processed += 1
                light_count += used
                camera_record = {
                    "object": sample.object_name,
                    "camera": sample.camera,
                    "paired_olat_images": used,
                }
                if args.material_acquisition == "end2end":
                    camera_record["manifest"] = (
                        f"{sample.object_name}/{sample.camera}/manifest.json"
                    )
                run_manifest["cameras"].append(camera_record)
                _write_run_manifest(out_root, run_manifest)
    except BaseException as exc:
        run_manifest["status"] = "failed"
        run_manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        run_manifest["error"] = f"{type(exc).__name__}: {exc}"
        if current_sample is not None:
            failed_camera = {
                "object": current_sample.object_name,
                "camera": current_sample.camera,
            }
            camera_manifest = (
                out_root
                / current_sample.object_name
                / current_sample.camera
                / "manifest.json"
            )
            if camera_manifest.is_file():
                failed_camera["manifest"] = (
                    f"{current_sample.object_name}/{current_sample.camera}/manifest.json"
                )
            run_manifest["failed_camera"] = failed_camera
        _write_run_manifest(out_root, run_manifest)
        raise
    run_manifest["status"] = "complete"
    run_manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    run_manifest["processed_cameras"] = processed
    run_manifest["paired_olat_images"] = light_count
    _write_run_manifest(out_root, run_manifest)
    print(f"[process] decomposed cameras: {processed}")
    print(f"[process] paired OLAT images used: {light_count}")
    print(f"[process] material maps: {out_root}")


def _write_run_manifest(out_root: Path, payload: dict[str, object]) -> None:
    path = out_root / "run.json"
    temporary = out_root / "run.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
