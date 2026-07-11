from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from ictpolarreal.utils.io import write_image


def _load_samples(path: str | Path) -> list[dict[str, str]]:
    payload = json.loads(Path(path).read_text())
    return list(payload["samples"])


def _output_path(root: str | Path, sample: dict[str, str], task: str) -> Path:
    return Path(root) / sample["object"] / sample["camera"] / "static" / f"{task}.png"


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


def _run_diffusion_renderer(args: argparse.Namespace, samples: list[dict[str, str]]) -> None:
    import torch
    import torch.nn.functional as functional

    if not args.repo:
        raise ValueError("Diffusion Renderer requires --repo pointing to a Cosmos Diffusion Renderer checkout")
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    from cosmos_predict1.diffusion.inference.diffusion_renderer_pipeline import (
        DiffusionRendererPipeline,
    )
    from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.rendering_utils import (
        GBUFFER_INDEX_MAPPING,
    )

    checkpoint_dir = Path(args.checkpoint_dir or repo / "checkpoints")
    pipeline = DiffusionRendererPipeline(
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
    passes = {
        "diffuse_albedo": "albedo",
        "normal": "normal",
        "specular_albedo": "specular",
    }

    for sample in samples:
        pending = {
            source: target
            for source, target in passes.items()
            if args.force or not _output_path(args.out_root, sample, target).exists()
        }
        if not pending:
            continue
        batch = _diffusion_renderer_batch(sample["input"], height=args.height, width=args.width)
        original_hw = tuple(int(value) for value in batch["in_res"][0])
        for source, target in pending.items():
            batch["context_index"].fill_(GBUFFER_INDEX_MAPPING[source])
            output = pipeline.generate_video(
                data_batch=batch,
                normalize_normal=False,
            )
            prediction = torch.as_tensor(output).permute(0, 3, 1, 2).float()
            prediction = functional.interpolate(
                prediction,
                size=original_hw,
                mode="bilinear",
                align_corners=False,
            )[0].permute(1, 2, 0).cpu().numpy()
            if prediction.max(initial=0.0) > 2.0:
                prediction = prediction / 255.0
            prediction = np.clip(prediction, 0.0, 1.0)
            if source == "normal":
                prediction[..., 0] = 1.0 - prediction[..., 0]
            output_path = _output_path(args.out_root, sample, target)
            write_image(output_path, prediction)
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

    samples = _load_samples(args.manifest)
    if args.method == "lotus":
        _run_lotus(args, samples)
    elif args.method == "dsine":
        _run_dsine(args, samples)
    else:
        _run_diffusion_renderer(args, samples)


if __name__ == "__main__":
    main()
