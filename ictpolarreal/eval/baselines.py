from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from ictpolarreal.data.forward_evaluation import (
    LIGHTING_CONVENTION,
    ICTPolarRealForwardEvaluationDataset,
)
from ictpolarreal.data.dataset import iter_camera_samples
from ictpolarreal.eval.cache import forward_prediction_is_current
from ictpolarreal.utils.io import read_image, write_image


BASELINE_TASKS = {
    "diffusion_renderer": ("albedo", "normal", "specular"),
    "lotus": ("normal",),
    "dsine": ("normal",),
}


def _tone_map(image: np.ndarray) -> np.ndarray:
    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if image.max(initial=0.0) <= 1.0:
        return np.clip(image, 0.0, 1.0)
    finite = image[np.isfinite(image)]
    scale = float(np.percentile(finite, 99.5)) if finite.size else 1.0
    normalized = np.maximum(image / max(scale, 1e-8), 0.0)
    mapped = normalized * (1.0 + normalized / 11.2**2) / (1.0 + normalized + 1e-8)
    return np.clip(mapped, 0.0, 1.0).astype(np.float32)


def prepare_manifest(
    data_root: str | Path,
    out_root: str | Path,
    *,
    max_samples: int,
    stages: tuple[str, ...] = ("inverse",),
    material_root: str | Path | None = None,
    hdri_root: str | Path | None = None,
    light_root: str | Path | None = None,
    frame_layout: str = "auto",
    max_lights: int | None = None,
    light_start: int = 0,
    forward_olat_samples: int = 20,
    forward_hdri_samples: int = 20,
    forward_hdri_olat_lights: int = 20,
) -> tuple[Path, list[dict[str, str]], list[dict[str, str]]]:
    out_root = Path(out_root)
    samples = []
    for camera in iter_camera_samples(data_root):
        input_path = camera.image_path("static")
        if input_path is None:
            continue
        staged_path = out_root / "inputs" / camera.object_name / camera.camera / "static.png"
        write_image(staged_path, _tone_map(read_image(input_path, channels=3)))
        samples.append(
            {
                "object": camera.object_name,
                "camera": camera.camera,
                "input": str(staged_path.resolve()),
            }
        )
        if len(samples) >= max_samples:
            break
    if not samples:
        raise FileNotFoundError(f"No camera with a static image was found under {data_root}")
    forward_samples = []
    if "forward" in stages:
        if material_root is None:
            raise ValueError("Forward baselines require --material-root")
        if hdri_root is None:
            raise ValueError("Forward baselines require --hdri-root")
        dataset = ICTPolarRealForwardEvaluationDataset(
            data_root,
            material_root=material_root,
            hdri_root=hdri_root,
            max_lights=max_lights,
            light_start=light_start,
            frame_layout=frame_layout,
            light_root=light_root,
            olat_samples=forward_olat_samples,
            hdri_samples=forward_hdri_samples,
            hdri_olat_lights=forward_hdri_olat_lights,
        )
        for index in dataset.evaluation_indices(max_samples):
            record = dataset.records[index]
            forward_samples.append(
                {
                    "object": record.camera_key[0],
                    "camera": record.camera_key[1],
                    "lighting_type": record.lighting_type,
                    "lighting_name": record.lighting_name,
                    "environment": str(dataset.export_environment(index, out_root).resolve()),
                    "lighting_convention": LIGHTING_CONVENTION,
                    "gbuffer_convention": "diffusion-renderer-display-normal-v1",
                    "sampling_convention": "fixed-seed-v1",
                    "environment_flip": False,
                    "environment_rotation_degrees": 180.0,
                    "environment_strength": 10.0 if record.lighting_type == "olat" else 1.0,
                }
            )

    manifest_path = out_root / "baseline_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "stages": list(stages),
                "samples": samples,
                "forward_samples": forward_samples,
            },
            indent=2,
        )
        + "\n"
    )
    return manifest_path.resolve(), samples, forward_samples


def _method_complete(
    root: Path,
    method: str,
    samples: list[dict[str, str]],
    *,
    stages: tuple[str, ...] = ("inverse",),
    forward_samples: list[dict[str, str]] | None = None,
) -> bool:
    inverse_complete = "inverse" not in stages or all(
        (root / method / sample["object"] / sample["camera"] / "static" / f"{task}.png").is_file()
        for sample in samples
        for task in BASELINE_TASKS[method]
    )
    forward_complete = (
        "forward" not in stages
        or method != "diffusion_renderer"
        or all(
            forward_prediction_is_current(
                root
                / method
                / sample["object"]
                / sample["camera"]
                / sample["lighting_name"]
                / "forward_rgb.png",
                sample,
            )
            for sample in forward_samples or []
        )
    )
    return inverse_complete and forward_complete


def _worker_environment(repo: Path | None, python: str) -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[2]
    paths = [str(project_root)]
    if repo is not None:
        paths.append(str(repo))
    existing = os.environ.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    python_path = Path(python).expanduser()
    if python_path.parent.name == "bin":
        environment_prefix = python_path.parent.parent
        environment["CONDA_PREFIX"] = str(environment_prefix)
        environment["PATH"] = os.pathsep.join(
            (str(python_path.parent), environment.get("PATH", ""))
        )
        environment.setdefault("CUDA_HOME", str(environment_prefix))
        environment.setdefault("CUDA_PATH", str(environment_prefix))
        library_paths = [str(environment_prefix / "lib")]
        if environment.get("LD_LIBRARY_PATH"):
            library_paths.append(environment["LD_LIBRARY_PATH"])
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths)
    return environment


def _run_method(
    method: str,
    *,
    python: str,
    repo: str | None,
    manifest: Path,
    out_root: Path,
    args: argparse.Namespace,
) -> None:
    repo_path = Path(repo).expanduser().resolve() if repo else None
    if method in {"lotus", "diffusion_renderer"} and (repo_path is None or not repo_path.is_dir()):
        option = "--lotus-repo" if method == "lotus" else "--diffusion-renderer-repo"
        raise FileNotFoundError(f"{method} source checkout not found; set {option} PATH")
    command = [
        python,
        "-m",
        "ictpolarreal.eval.baseline_worker",
        method,
        "--manifest",
        str(manifest),
        "--out-root",
        str(out_root / method),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    if repo_path is not None:
        command.extend(("--repo", str(repo_path)))
    if args.local_files_only:
        command.append("--local-files-only")
    if args.force:
        command.append("--force")
    if method == "diffusion_renderer":
        command.extend(
            (
                "--inference-steps",
                str(args.diffusion_renderer_steps),
                "--height",
                str(args.diffusion_renderer_height),
                "--width",
                str(args.diffusion_renderer_width),
            )
        )
        if args.diffusion_renderer_checkpoint_dir:
            command.extend(("--checkpoint-dir", args.diffusion_renderer_checkpoint_dir))
        if args.diffusion_renderer_offload:
            command.append("--offload")
    print(f"[baselines] running {method} with {python}")
    subprocess.run(
        command,
        check=True,
        cwd=str(repo_path or Path.cwd()),
        env=_worker_environment(repo_path, python),
    )


def run(args: argparse.Namespace) -> dict[str, str]:
    methods = tuple(dict.fromkeys(item.strip() for item in args.methods.split(",") if item.strip()))
    unknown = set(methods) - set(BASELINE_TASKS)
    if unknown:
        raise ValueError(f"Unknown baseline method(s): {', '.join(sorted(unknown))}")
    stages = tuple(dict.fromkeys(item.strip() for item in args.stages.split(",") if item.strip()))
    unknown_stages = set(stages) - {"inverse", "forward"}
    if unknown_stages or not stages:
        raise ValueError("--stages must contain inverse, forward, or both")
    out_root = Path(args.out_root).expanduser().resolve()
    manifest, samples, forward_samples = prepare_manifest(
        args.data_root,
        out_root,
        max_samples=args.max_samples,
        stages=stages,
        material_root=args.material_root,
        hdri_root=args.hdri_root,
        light_root=args.light_root,
        frame_layout=args.frame_layout,
        max_lights=args.max_lights,
        light_start=args.light_start,
        forward_olat_samples=args.forward_olat_samples,
        forward_hdri_samples=args.forward_hdri_samples,
        forward_hdri_olat_lights=args.forward_hdri_olat_lights,
    )
    settings = {
        "lotus": (args.lotus_python, args.lotus_repo),
        "dsine": (args.dsine_python, args.dsine_repo),
        "diffusion_renderer": (
            args.diffusion_renderer_python,
            args.diffusion_renderer_repo,
        ),
    }
    roots = {}
    for method in methods:
        roots[method] = str(out_root / method)
        if _method_complete(
            out_root,
            method,
            samples,
            stages=stages,
            forward_samples=forward_samples,
        ) and not args.force:
            print(f"[baselines] using cached {method} predictions under {out_root / method}")
            continue
        python, repo = settings[method]
        _run_method(
            method,
            python=python,
            repo=repo,
            manifest=manifest,
            out_root=out_root,
            args=args,
        )
        if not _method_complete(
            out_root,
            method,
            samples,
            stages=stages,
            forward_samples=forward_samples,
        ):
            raise RuntimeError(f"{method} finished without writing every expected prediction")
    roots_path = out_root / "roots.json"
    if roots_path.exists():
        roots = {**json.loads(roots_path.read_text()), **roots}
    roots_path.write_text(json.dumps(roots, indent=2, sort_keys=True) + "\n")
    print(f"[baselines] ready: {roots_path}")
    return roots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute ICTPolarReal benchmark methods.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--stages", default="inverse")
    parser.add_argument("--material-root", default=None)
    parser.add_argument("--hdri-root", default=None)
    parser.add_argument("--light-root", default=None)
    parser.add_argument("--frame-layout", choices=["auto", "raw", "normalized"], default="auto")
    parser.add_argument("--max-lights", type=int, default=None)
    parser.add_argument("--light-start", type=int, default=0)
    parser.add_argument("--forward-olat-samples", type=int, default=20)
    parser.add_argument("--forward-hdri-samples", type=int, default=20)
    parser.add_argument("--forward-hdri-olat-lights", type=int, default=20)
    parser.add_argument("--methods", default="diffusion_renderer,lotus,dsine")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lotus-python", default=sys.executable)
    parser.add_argument("--lotus-repo", default=None)
    parser.add_argument("--dsine-python", default=sys.executable)
    parser.add_argument("--dsine-repo", default=None)
    parser.add_argument("--diffusion-renderer-python", default=sys.executable)
    parser.add_argument("--diffusion-renderer-repo", default=None)
    parser.add_argument("--diffusion-renderer-checkpoint-dir", default=None)
    parser.add_argument("--diffusion-renderer-steps", type=int, default=15)
    parser.add_argument("--diffusion-renderer-height", type=int, default=704)
    parser.add_argument("--diffusion-renderer-width", type=int, default=1280)
    parser.add_argument("--diffusion-renderer-offload", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
