from __future__ import annotations

import argparse
import gc
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from ictpolarreal.eval.cache import (
    forward_prediction_is_current,
    write_forward_cache_signature,
)
from ictpolarreal.utils.io import read_image, write_image


def _load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _output_path(root: str | Path, sample: dict[str, str], task: str) -> Path:
    return Path(root) / sample["object"] / sample["camera"] / "static" / f"{task}.png"


def _gbuffer_path(root: str | Path, sample: dict[str, str], task: str) -> Path:
    return Path(root) / sample["object"] / sample["camera"] / "_gbuffer" / f"{task}.png"


def _forward_output_path(root: str | Path, sample: dict[str, str]) -> Path:
    return (
        Path(root)
        / sample["object"]
        / sample["camera"]
        / sample["lighting_name"]
        / "forward_rgb.png"
    )


def _run_lotus(args: argparse.Namespace, samples: list[dict[str, str]]) -> None:
    import torch
    from PIL import Image

    if not args.repo:
        raise ValueError("Lotus requires --repo pointing to an EnVision-Research/Lotus checkout")
    sys.path.insert(0, str(Path(args.repo).resolve()))
    from pipeline import LotusGPipeline

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipeline = LotusGPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    for sample in samples:
        output_path = _output_path(args.out_root, sample, "normal")
        if output_path.exists() and not args.force:
            continue
        image = np.asarray(Image.open(sample["input"]).convert("RGB"), dtype=np.float32)
        rgb = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        rgb = rgb.to(device=device, dtype=torch.float32) / 127.5 - 1.0
        task_embedding = torch.tensor([[1.0, 0.0]], device=device)
        task_embedding = torch.cat(
            (torch.sin(task_embedding), torch.cos(task_embedding)),
            dim=-1,
        )
        autocast = torch.autocast(device_type="cuda", dtype=dtype) if device.type == "cuda" else nullcontext()
        with torch.inference_mode(), autocast:
            prediction = pipeline(
                rgb_in=rgb,
                task_emb=task_embedding,
                prompt="",
                num_inference_steps=1,
                timesteps=[999],
                generator=generator,
                output_type="np",
            ).images[0]
        write_image(output_path, np.clip(prediction, 0.0, 1.0))
        print(f"[baseline:lotus] wrote {output_path}")


def _run_dsine(args: argparse.Namespace, samples: list[dict[str, str]]) -> None:
    import cv2
    import torch

    hub_source = "local" if args.repo else "github"
    hub_repo = args.repo or "hugoycj/DSINE-hub"
    load_kwargs = {"source": hub_source, "trust_repo": True}
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / "dsine.pt"
    if checkpoint.exists():
        load_kwargs["local_file_path"] = str(checkpoint)
    elif args.local_files_only:
        raise FileNotFoundError(f"DSINE checkpoint not found in the torch hub cache: {checkpoint}")
    predictor = torch.hub.load(hub_repo, "DSINE", **load_kwargs)

    for sample in samples:
        output_path = _output_path(args.out_root, sample, "normal")
        if output_path.exists() and not args.force:
            continue
        image = cv2.imread(sample["input"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read baseline input: {sample['input']}")
        with torch.inference_mode():
            prediction = predictor.infer_cv2(image)[0]
        prediction = ((prediction + 1.0) * 0.5).permute(1, 2, 0).float().cpu().numpy()
        write_image(output_path, np.clip(prediction, 0.0, 1.0))
        print(f"[baseline:dsine] wrote {output_path}")


def _diffusion_renderer_batch(image_path: str, *, height: int, width: int):
    import torch
    import torch.nn.functional as functional
    from PIL import Image

    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    original_height, original_width = image.shape[:2]
    rgb = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    rgb = functional.interpolate(rgb, size=(height, width), mode="bilinear", align_corners=False)
    rgb = (rgb * 2.0 - 1.0).unsqueeze(2)
    return {
        "clip_name": [Path(image_path).stem],
        "chunk_index": ["0000"],
        "num_frames": torch.ones((1,), dtype=torch.float32),
        "image_size": torch.tensor([[height, width]]),
        "fps": torch.tensor([24.0]),
        "padding_mask": torch.zeros((1, 1, height, width)),
        "t5_text_embeddings": torch.zeros((1, 512, 1024)),
        "t5_text_mask": torch.nn.functional.one_hot(torch.tensor([1]), num_classes=512).float(),
        "context_index": torch.zeros((1, 1), dtype=torch.long),
        "rgb": rgb,
        "in_res": torch.tensor([[original_height, original_width]]),
    }


def _diffusion_renderer_prediction(output, *, output_hw: tuple[int, int]):
    import torch
    import torch.nn.functional as functional

    prediction = torch.as_tensor(output).permute(0, 3, 1, 2).float()
    prediction = functional.interpolate(
        prediction,
        size=output_hw,
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0).cpu().numpy()
    if prediction.max(initial=0.0) > 2.0:
        prediction = prediction / 255.0
    return np.clip(prediction[..., :3], 0.0, 1.0)


def _diffusion_renderer_forward_batch(
    sample: dict[str, str],
    *,
    out_root: str | Path,
    height: int,
    width: int,
):
    import torch
    import torch.nn.functional as functional

    batch = _diffusion_renderer_batch(sample["input"], height=height, width=width)
    for task in ("basecolor", "normal", "metallic", "roughness", "depth"):
        image = read_image(_gbuffer_path(out_root, sample, task), channels=3)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        tensor = functional.interpolate(
            tensor,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        batch[task] = (tensor * 2.0 - 1.0).unsqueeze(2)
    return batch


def _run_diffusion_renderer(args: argparse.Namespace, manifest: dict) -> None:
    import torch

    if not args.repo:
        raise ValueError("Diffusion Renderer requires --repo pointing to a Cosmos Diffusion Renderer checkout")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Diffusion Renderer inverse/forward inference requires a CUDA GPU")
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    from cosmos_predict1.diffusion.inference.diffusion_renderer_pipeline import (
        DiffusionRendererPipeline,
    )
    from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.rendering_utils import (
        GBUFFER_INDEX_MAPPING,
        envmap_vec,
    )
    from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.utils_env_proj import (
        process_environment_map,
    )

    checkpoint_dir = Path(args.checkpoint_dir or repo / "checkpoints")
    samples = list(manifest["samples"])
    forward_samples = list(manifest.get("forward_samples", []))
    stages = set(manifest.get("stages", ["inverse"]))
    passes = {
        "diffuse_albedo": "albedo",
        "normal": "normal",
        "specular_albedo": "specular",
    }
    required_sources = []
    if "inverse" in stages:
        required_sources.extend(passes)
    if forward_samples:
        required_sources.extend(("basecolor", "normal", "metallic", "roughness", "depth"))
    required_sources = list(dict.fromkeys(required_sources))

    inverse_pipeline = None
    for sample in samples:
        batch = None
        original_hw = None
        for source in required_sources:
            gbuffer_path = _gbuffer_path(args.out_root, sample, source)
            if args.force or not gbuffer_path.exists():
                if inverse_pipeline is None:
                    inverse_pipeline = DiffusionRendererPipeline(
                        checkpoint_dir=str(checkpoint_dir),
                        checkpoint_name="Diffusion_Renderer_Inverse_Cosmos_7B",
                        offload_network=args.offload,
                        offload_tokenizer=args.offload,
                        offload_text_encoder_model=args.offload,
                        offload_guardrail_models=args.offload,
                        guidance=0,
                        num_steps=args.inference_steps,
                        height=args.height,
                        width=args.width,
                        fps=24,
                        num_video_frames=1,
                        seed=args.seed,
                    )
                if batch is None:
                    batch = _diffusion_renderer_batch(
                        sample["input"],
                        height=args.height,
                        width=args.width,
                    )
                    original_hw = tuple(int(value) for value in batch["in_res"][0])
                batch.pop("video", None)
                batch["context_index"].fill_(GBUFFER_INDEX_MAPPING[source])
                output = inverse_pipeline.generate_video(
                    data_batch=batch,
                    normalize_normal=False,
                )
                prediction = _diffusion_renderer_prediction(output, output_hw=original_hw)
                write_image(gbuffer_path, prediction)
                print(f"[baseline:diffusion_renderer] wrote {gbuffer_path}")
            else:
                prediction = read_image(gbuffer_path, channels=3)

            target = passes.get(source)
            if target is None or "inverse" not in stages:
                continue
            output_path = _output_path(args.out_root, sample, target)
            if output_path.exists() and not args.force:
                continue
            display = prediction.copy()
            if source == "normal":
                display[..., 0] = 1.0 - display[..., 0]
            write_image(output_path, display)
            print(f"[baseline:diffusion_renderer] wrote {output_path}")

    if inverse_pipeline is not None:
        del inverse_pipeline
        gc.collect()
        torch.cuda.empty_cache()

    pending_forward = [
        sample
        for sample in forward_samples
        if args.force
        or not forward_prediction_is_current(
            _forward_output_path(args.out_root, sample),
            sample,
        )
    ]
    if not pending_forward:
        return

    input_by_camera = {(sample["object"], sample["camera"]): sample for sample in samples}
    forward_pipeline = DiffusionRendererPipeline(
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_name="Diffusion_Renderer_Forward_Cosmos_7B",
        offload_network=args.offload,
        offload_tokenizer=args.offload,
        offload_text_encoder_model=args.offload,
        offload_guardrail_models=args.offload,
        guidance=0,
        num_steps=args.inference_steps,
        height=args.height,
        width=args.width,
        fps=24,
        num_video_frames=1,
        seed=args.seed,
    )
    device = torch.device("cuda")
    for index, forward_sample in enumerate(pending_forward):
        camera_key = (forward_sample["object"], forward_sample["camera"])
        if camera_key not in input_by_camera:
            raise KeyError(f"No staged static input for forward sample {camera_key}")
        input_sample = input_by_camera[camera_key]
        batch = _diffusion_renderer_forward_batch(
            input_sample,
            out_root=args.out_root,
            height=args.height,
            width=args.width,
        )
        environment = process_environment_map(
            forward_sample["environment"],
            resolution=(args.height, args.width),
            num_frames=1,
            fixed_pose=True,
            rotate_envlight=False,
            env_flip=bool(forward_sample["environment_flip"]),
            env_rot=float(forward_sample["environment_rotation_degrees"]),
            env_format=["proj"],
            device=device,
        )
        batch["env_ldr"] = environment["env_ldr"].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
        batch["env_log"] = environment["env_log"].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1
        environment_normal = envmap_vec([args.height, args.width], device=device)
        batch["env_nrm"] = (
            environment_normal.unsqueeze(0)
            .unsqueeze(0)
            .permute(0, 4, 1, 2, 3)
            .expand_as(batch["env_ldr"])
        )
        output = forward_pipeline.generate_video(
            data_batch=batch,
            seed=args.seed + index,
        )
        original_hw = tuple(int(value) for value in batch["in_res"][0])
        prediction = _diffusion_renderer_prediction(output, output_hw=original_hw)
        output_path = _forward_output_path(args.out_root, forward_sample)
        write_image(output_path, prediction)
        write_forward_cache_signature(output_path, forward_sample)
        print(f"[baseline:diffusion_renderer] wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one ICTPolarReal benchmark baseline worker.")
    parser.add_argument("method", choices=["diffusion_renderer", "lotus", "dsine"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--repo", default=None)
    parser.add_argument("--model", default="jingheya/lotus-normal-g-v1-1")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference-steps", type=int, default=15)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    samples = list(manifest["samples"])
    if args.method == "lotus":
        _run_lotus(args, samples)
    elif args.method == "dsine":
        _run_dsine(args, samples)
    else:
        _run_diffusion_renderer(args, manifest)


if __name__ == "__main__":
    main()
