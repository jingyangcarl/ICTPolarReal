from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from ictpolarreal.data.training import ICTPolarRealTrainingDataset
from ictpolarreal.train.contracts import (
    INVERSE_PROMPTS,
    build_forward_condition,
    inverse_target,
    inverse_target_names,
)
from ictpolarreal.utils.io import write_image


def add_training_arguments(parser: argparse.ArgumentParser, *, stage: str) -> argparse.ArgumentParser:
    default_model = "zheng95z/rgb-to-x" if stage == "inverse" else "zheng95z/x-to-rgb"
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--material-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pred-dir", default=None)
    parser.add_argument("--model-name", default=default_model)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max-lights", type=int, default=None)
    parser.add_argument("--light-start", type=int, default=0)
    parser.add_argument("--frame-layout", choices=["auto", "raw", "normalized"], default="auto")
    parser.add_argument("--light-root", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--mixed-precision", choices=["auto", "no", "fp16", "bf16"], default="auto")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--checkpointing-steps", type=int, default=1000)
    parser.add_argument("--resume-from-checkpoint", default=None, help="Checkpoint path or 'latest'.")
    parser.add_argument("--evaluation-steps", type=int, default=5000)
    parser.add_argument("--evaluation-samples", type=int, default=4)
    parser.add_argument("--evaluation-methods", default="pretrained,finetuned")
    parser.add_argument("--eval-data-root", default=None)
    parser.add_argument("--eval-material-root", default=None)
    parser.add_argument("--preview-samples", type=int, default=1)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate data tensors without loading model weights.")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    if stage == "inverse":
        parser.add_argument("--workflow", choices=["pbr", "polarization", "both"], default="both")
    else:
        parser.add_argument("--conditioning", choices=["gbuffer", "polarization"], default="gbuffer")
    return parser


def run_diffusion_training(args: argparse.Namespace, *, stage: str) -> None:
    if stage not in {"inverse", "forward"}:
        raise ValueError("stage must be inverse or forward")

    dataset = ICTPolarRealTrainingDataset(
        args.data_root,
        material_root=args.material_root,
        resolution=args.resolution,
        max_lights=args.max_lights,
        light_start=args.light_start,
        frame_layout=args.frame_layout,
        light_root=args.light_root,
        require_polarization_reference=stage == "forward" and args.conditioning == "polarization",
        max_samples=args.max_samples,
    )
    print(f"[train:{stage}] dataset: {dataset.summary()}")
    if args.dry_run:
        _print_sample_contract(dataset[0], stage=stage, args=args)
        return

    evaluation_methods = tuple(method.strip() for method in args.evaluation_methods.split(",") if method.strip())
    invalid_methods = set(evaluation_methods) - {"pretrained", "finetuned"}
    if invalid_methods:
        raise ValueError(f"Unknown evaluation method(s): {', '.join(sorted(invalid_methods))}")
    evaluation_dataset = None
    if args.evaluation_samples > 0 and evaluation_methods:
        evaluation_dataset = ICTPolarRealTrainingDataset(
            args.eval_data_root or args.data_root,
            material_root=args.eval_material_root or args.material_root,
            resolution=args.resolution,
            max_lights=args.max_lights,
            light_start=args.light_start,
            frame_layout=args.frame_layout,
            light_root=args.light_root,
            require_polarization_reference=stage == "forward" and args.conditioning == "polarization",
        )
        print(f"[eval:{stage}] dataset: {evaluation_dataset.summary()}")

    try:
        import torch
        import torch.nn.functional as functional
        from accelerate import Accelerator
        from accelerate.utils import set_seed
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
        from peft import LoraConfig
        from torch.utils.data import DataLoader
        from transformers import CLIPTextModel, CLIPTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit("Diffusion training dependencies are missing. Run `bash run.sh setup`.") from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA training was requested but CUDA is unavailable. Use `--device cpu` only for diagnostics.")
    mixed_precision = args.mixed_precision
    if mixed_precision == "auto":
        mixed_precision = "fp16" if args.device == "cuda" else "no"
    accelerator = Accelerator(
        cpu=args.device == "cpu",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    set_seed(args.seed)
    output_dir = Path(args.out_dir)
    resume_dir = _resolve_resume_dir(output_dir, args.resume_from_checkpoint)

    load_kwargs = {"local_files_only": args.local_files_only}
    noise_scheduler = DDIMScheduler.from_pretrained(args.model_name, subfolder="scheduler", **load_kwargs)
    noise_scheduler = DDIMScheduler.from_config(
        noise_scheduler.config,
        prediction_type="v_prediction",
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing",
    )
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name, subfolder="tokenizer", **load_kwargs)
    text_encoder = CLIPTextModel.from_pretrained(args.model_name, subfolder="text_encoder", **load_kwargs)
    vae = AutoencoderKL.from_pretrained(args.model_name, subfolder="vae", **load_kwargs)
    if args.full_finetune and resume_dir is not None:
        unet = UNet2DConditionModel.from_pretrained(resume_dir / "unet", **load_kwargs)
    else:
        unet = UNet2DConditionModel.from_pretrained(args.model_name, subfolder="unet", **load_kwargs)

    expected_channels = 8 if stage == "inverse" else 23
    if unet.config.in_channels != expected_channels:
        raise ValueError(
            f"{stage} model {args.model_name} must accept {expected_channels} channels; "
            f"its UNet accepts {unet.config.in_channels}."
        )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if not args.full_finetune:
        unet.requires_grad_(False)
        unet.add_adapter(
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            ),
            adapter_name="ictpolarreal",
        )
        if resume_dir is not None:
            _load_lora_adapter(unet, resume_dir / "adapter")
        for parameter in unet.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if args.full_finetune and "pretrained" in evaluation_methods:
        evaluation_methods = tuple(method for method in evaluation_methods if method != "pretrained")
        print("[eval] skipping pretrained comparison because --full-finetune cannot disable an adapter")

    trainable_parameters = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable UNet parameters were selected")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate)
    global_step = 0
    if resume_dir is not None:
        trainer_state = torch.load(resume_dir / "trainer_state.pt", map_location="cpu")
        optimizer.load_state_dict(trainer_state["optimizer"])
        global_step = int(trainer_state["global_step"])
        print(f"[train:{stage}] resumed from {resume_dir} at step {global_step}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    unet, optimizer, loader = accelerator.prepare(unet, optimizer, loader)

    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype).eval()
    text_encoder.to(accelerator.device, dtype=weight_dtype).eval()
    unet.train()

    prompts = tuple(INVERSE_PROMPTS.values()) if stage == "inverse" else ("",)
    prompt_embeddings = _encode_prompts(
        prompts,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=accelerator.device,
    )
    target_rng = random.Random(args.seed)
    inverse_names = inverse_target_names(args.workflow) if stage == "inverse" else ()
    if stage == "inverse":
        for _ in range(global_step):
            target_rng.choice(inverse_names)

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_run_config(output_dir, args, stage=stage, dataset_summary=dataset.summary())
    accelerator.wait_for_everyone()

    last_evaluation_step = -1
    while global_step < args.max_steps:
        for batch in loader:
            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    def encode(image):
                        return _encode_images(image, vae=vae, dtype=weight_dtype)

                    target_name = None
                    if stage == "inverse":
                        target_name = target_rng.choice(inverse_names)
                        target_image, prompt = inverse_target(batch, target_name)
                        target_latents = encode(target_image)
                        condition = encode(batch["rgb"])
                    else:
                        prompt = ""
                        target_latents = encode(batch["rgb"])
                        condition = build_forward_condition(
                            batch,
                            mode=args.conditioning,
                            encode=encode,
                            latent_hw=target_latents.shape[-2:],
                        )

                    noise = torch.randn_like(target_latents)
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (target_latents.shape[0],),
                        device=target_latents.device,
                    ).long()
                    noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)
                    model_input = torch.cat((noisy_latents, condition), dim=1)
                    embeddings = prompt_embeddings[prompt].repeat(target_latents.shape[0], 1, 1)
                    model_prediction = unet(
                        model_input,
                        timesteps,
                        encoder_hidden_states=embeddings,
                        return_dict=False,
                    )[0]
                    velocity = noise_scheduler.get_velocity(target_latents, noise, timesteps)
                    valid = _latent_valid_mask(batch["mask"], target_latents.shape[-2:])
                    loss = functional.mse_loss(model_prediction[valid], velocity[valid], reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                label = f" target={target_name}" if target_name else f" conditioning={args.conditioning}"
                accelerator.print(f"[train:{stage}] step={global_step} loss={loss.detach().item():.6f}{label}")
                if args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
                    _save_model(
                        accelerator,
                        unet,
                        optimizer,
                        output_dir / f"checkpoint-{global_step:06d}",
                        args,
                        global_step=global_step,
                    )
                if (
                    evaluation_dataset is not None
                    and args.evaluation_steps > 0
                    and global_step % args.evaluation_steps == 0
                ):
                    _run_periodic_evaluation(
                        evaluation_dataset,
                        stage=stage,
                        args=args,
                        methods=evaluation_methods,
                        step=global_step,
                        accelerator=accelerator,
                        unet=unet,
                        vae=vae,
                        noise_scheduler=noise_scheduler,
                        prompt_embeddings=prompt_embeddings,
                        dtype=weight_dtype,
                    )
                    last_evaluation_step = global_step
            if global_step >= args.max_steps:
                break

    accelerator.wait_for_everyone()
    _save_model(accelerator, unet, optimizer, output_dir / "final", args, global_step=global_step)
    if evaluation_dataset is not None and last_evaluation_step != global_step:
        _run_periodic_evaluation(
            evaluation_dataset,
            stage=stage,
            args=args,
            methods=evaluation_methods,
            step=global_step,
            accelerator=accelerator,
            unet=unet,
            vae=vae,
            noise_scheduler=noise_scheduler,
            prompt_embeddings=prompt_embeddings,
            dtype=weight_dtype,
        )
    if args.preview_samples > 0 and args.pred_dir and accelerator.is_main_process:
        _write_previews(
            dataset,
            stage=stage,
            args=args,
            unet=accelerator.unwrap_model(unet),
            vae=vae,
            noise_scheduler=noise_scheduler,
            prompt_embeddings=prompt_embeddings,
            device=accelerator.device,
            dtype=weight_dtype,
        )
    accelerator.wait_for_everyone()


def _print_sample_contract(sample: dict, *, stage: str, args: argparse.Namespace) -> None:
    tensor_shapes = {key: tuple(value.shape) for key, value in sample.items() if hasattr(value, "shape")}
    print(f"[train:{stage}] tensor contract: {tensor_shapes}")
    if stage == "inverse":
        print(f"[train:{stage}] targets: {inverse_target_names(args.workflow)}")
    else:
        print(f"[train:{stage}] conditioning: {args.conditioning}")


def _encode_prompts(prompts, *, tokenizer, text_encoder, device) -> dict[str, object]:
    import torch

    encoded = {}
    with torch.no_grad():
        for prompt in prompts:
            tokens = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            encoded[prompt] = text_encoder(tokens.input_ids.to(device), return_dict=False)[0]
    return encoded


def _encode_images(images, *, vae, dtype, sample: bool = True):
    import torch

    with torch.no_grad():
        distribution = vae.encode(images.to(dtype=dtype)).latent_dist
        latents = distribution.sample() if sample else distribution.mean
    return latents * vae.config.scaling_factor


def _latent_valid_mask(mask, latent_hw: tuple[int, int]):
    import torch
    import torch.nn.functional as functional

    invalid = 1.0 - mask.clamp(0.0, 1.0)
    invalid = functional.adaptive_max_pool2d(invalid, latent_hw)
    valid = (invalid < 0.5).expand(-1, 4, -1, -1)
    if not torch.any(valid):
        valid = torch.ones_like(valid, dtype=torch.bool)
    return valid


def _resolve_resume_dir(output_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    if value == "latest":
        candidates = [path for path in output_dir.glob("checkpoint-*") if (path / "trainer_state.pt").exists()]
        if (output_dir / "final" / "trainer_state.pt").exists():
            candidates.append(output_dir / "final")
        if not candidates:
            raise FileNotFoundError(f"No checkpoints found under {output_dir}")

        import torch

        return max(
            candidates,
            key=lambda path: int(torch.load(path / "trainer_state.pt", map_location="cpu")["global_step"]),
        )
    checkpoint = Path(value)
    if not checkpoint.exists() and not checkpoint.is_absolute():
        checkpoint = output_dir / checkpoint
    if not (checkpoint / "trainer_state.pt").exists():
        raise FileNotFoundError(f"Missing trainer_state.pt under {checkpoint}")
    return checkpoint


def _load_lora_adapter(unet, adapter_dir: Path) -> None:
    from diffusers.utils.state_dict_utils import convert_unet_state_dict_to_peft
    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file

    weights_path = adapter_dir / "pytorch_lora_weights.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing LoRA weights: {weights_path}")
    state = load_file(weights_path)
    state = {key.removeprefix("unet."): value for key, value in state.items()}
    peft_state = convert_unet_state_dict_to_peft(state)
    incompatible = set_peft_model_state_dict(unet, peft_state, adapter_name="ictpolarreal")
    if incompatible.unexpected_keys:
        raise ValueError(f"Unexpected LoRA keys in {weights_path}: {incompatible.unexpected_keys[:3]}")


def _save_model(
    accelerator,
    unet,
    optimizer,
    output_dir: Path,
    args: argparse.Namespace,
    *,
    global_step: int,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        model = accelerator.unwrap_model(unet)
        if args.full_finetune:
            model.save_pretrained(output_dir / "unet", safe_serialization=True)
        else:
            from diffusers import StableDiffusionPipeline
            from diffusers.utils import convert_state_dict_to_diffusers
            from peft.utils import get_peft_model_state_dict

            lora_state = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(model, adapter_name="ictpolarreal")
            )
            StableDiffusionPipeline.save_lora_weights(
                save_directory=output_dir / "adapter",
                unet_lora_layers=lora_state,
                safe_serialization=True,
            )
        accelerator.save(
            {"global_step": global_step, "optimizer": optimizer.state_dict()},
            output_dir / "trainer_state.pt",
        )
        print(f"[train] saved checkpoint to {output_dir}")
    accelerator.wait_for_everyone()


def _write_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    *,
    stage: str,
    dataset_summary: dict[str, object],
) -> None:
    payload = {"stage": stage, "dataset": dataset_summary, "arguments": vars(args)}
    (output_dir / "run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run_periodic_evaluation(
    dataset: ICTPolarRealTrainingDataset,
    *,
    stage: str,
    args: argparse.Namespace,
    methods: tuple[str, ...],
    step: int,
    accelerator,
    unet,
    vae,
    noise_scheduler,
    prompt_embeddings,
    dtype,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and methods:
        import torch

        from ictpolarreal.utils.metrics import mae, mse, psnr, ssim_global

        model = accelerator.unwrap_model(unet)
        model.eval()
        step_root = Path(args.out_dir) / "eval" / f"step-{step:06d}"
        rows = []
        indices = _evaluation_indices(dataset, stage=stage, count=args.evaluation_samples)
        try:
            for method in methods:
                if not args.full_finetune:
                    if method == "pretrained":
                        model.disable_adapters()
                    else:
                        model.enable_adapters()
                for index in indices:
                    sample = dataset[index]
                    batch = _sample_to_batch(sample, device=_model_device(model), torch_module=torch)

                    def encode(image):
                        return _encode_images(image, vae=vae, dtype=dtype, sample=False)

                    if stage == "inverse":
                        condition = encode(batch["rgb"])
                        tasks = inverse_target_names(args.workflow)
                    else:
                        latent_hw = (batch["rgb"].shape[-2] // 8, batch["rgb"].shape[-1] // 8)
                        condition = build_forward_condition(
                            batch,
                            mode=args.conditioning,
                            encode=encode,
                            latent_hw=latent_hw,
                        )
                        tasks = (f"forward_{args.conditioning}",)

                    for task in tasks:
                        prompt = INVERSE_PROMPTS[task] if stage == "inverse" else ""
                        prediction = _sample_image(
                            model,
                            vae,
                            noise_scheduler,
                            condition=condition,
                            prompt_embedding=prompt_embeddings[prompt],
                            output_hw=batch["rgb"].shape[-2:],
                            inference_steps=args.inference_steps,
                            seed=args.seed + index,
                            device=_model_device(model),
                            dtype=dtype,
                        )
                        target_tensor = inverse_target(batch, task)[0] if stage == "inverse" else batch["rgb"]
                        target = _tensor_image(target_tensor[0])
                        mask = batch["mask"][0].float().cpu().permute(1, 2, 0).numpy()
                        light = "static" if sample["frame_id"] < 0 else f"{sample['frame_id']:06d}"
                        prediction_path = (
                            step_root
                            / method
                            / "predictions"
                            / sample["object"]
                            / sample["camera"]
                            / light
                            / f"{task}.png"
                        )
                        target_path = (
                            step_root
                            / "ground_truth"
                            / sample["object"]
                            / sample["camera"]
                            / light
                            / f"{task}.png"
                        )
                        write_image(prediction_path, prediction)
                        write_image(target_path, target)
                        rows.append(
                            {
                                "step": step,
                                "stage": stage,
                                "method": method,
                                "task": task,
                                "object": sample["object"],
                                "camera": sample["camera"],
                                "light": light,
                                "prediction": str(prediction_path),
                                "mse": mse(prediction, target, mask),
                                "mae": mae(prediction, target, mask),
                                "psnr": psnr(prediction, target, mask),
                                "ssim": ssim_global(prediction, target, mask),
                            }
                        )
        finally:
            if not args.full_finetune:
                model.enable_adapters()
            model.train()
        _write_training_evaluation(rows, step_root=step_root, output_dir=Path(args.out_dir), step=step)
    accelerator.wait_for_everyone()


def _write_training_evaluation(rows: list[dict], *, step_root: Path, output_dir: Path, step: int) -> None:
    if not rows:
        return
    step_root.mkdir(parents=True, exist_ok=True)
    with (step_root / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {"step": step, "methods": {}}
    for method in sorted({row["method"] for row in rows}):
        method_summary = {}
        for task in sorted({row["task"] for row in rows if row["method"] == method}):
            selected = [row for row in rows if row["method"] == method and row["task"] == task]
            method_summary[task] = {
                "count": len(selected),
                **{
                    metric: float(sum(row[metric] for row in selected) / len(selected))
                    for metric in ("mse", "mae", "psnr", "ssim")
                },
            }
        summary["methods"][method] = method_summary
    summary_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (step_root / "summary.json").write_text(summary_text)
    history_path = output_dir / "eval" / "history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a") as file:
        file.write(json.dumps(summary, sort_keys=True) + "\n")
    print(f"[eval:training] step={step} wrote {step_root / 'metrics.csv'}")


def _evaluation_indices(dataset: ICTPolarRealTrainingDataset, *, stage: str, count: int) -> list[int]:
    indices = []
    for index, record in enumerate(dataset.records):
        if stage == "forward" and record.light_index is None:
            continue
        indices.append(index)
        if len(indices) >= count:
            break
    return indices


def _sample_to_batch(sample: dict, *, device, torch_module) -> dict:
    return {
        key: value.unsqueeze(0).to(device) if isinstance(value, torch_module.Tensor) else [value]
        for key, value in sample.items()
    }


def _tensor_image(tensor) -> object:
    import numpy as np

    image = tensor.detach().float().cpu().permute(1, 2, 0).numpy()
    return np.clip(image * 0.5 + 0.5, 0.0, 1.0)


def _model_device(model):
    return next(model.parameters()).device


def _write_previews(
    dataset: ICTPolarRealTrainingDataset,
    *,
    stage: str,
    args: argparse.Namespace,
    unet,
    vae,
    noise_scheduler,
    prompt_embeddings,
    device,
    dtype,
) -> None:
    import torch

    unet.eval()
    prediction_root = Path(args.pred_dir)
    records = _preview_indices(dataset, stage=stage, count=args.preview_samples)
    with torch.no_grad():
        for index in records:
            sample = dataset[index]
            batch = {
                key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else [value]
                for key, value in sample.items()
            }

            def encode(image):
                return _encode_images(image, vae=vae, dtype=dtype)

            if stage == "inverse":
                condition = encode(batch["rgb"])
                for target_name in inverse_target_names(args.workflow):
                    prompt = INVERSE_PROMPTS[target_name]
                    prediction = _sample_image(
                        unet,
                        vae,
                        noise_scheduler,
                        condition=condition,
                        prompt_embedding=prompt_embeddings[prompt],
                        output_hw=batch["rgb"].shape[-2:],
                        inference_steps=args.inference_steps,
                        seed=args.seed,
                        device=device,
                        dtype=dtype,
                    )
                    write_image(
                        prediction_root / sample["object"] / sample["camera"] / f"{target_name}.png",
                        prediction,
                    )
            else:
                latent_hw = (batch["rgb"].shape[-2] // 8, batch["rgb"].shape[-1] // 8)
                condition = build_forward_condition(
                    batch,
                    mode=args.conditioning,
                    encode=encode,
                    latent_hw=latent_hw,
                )
                prediction = _sample_image(
                    unet,
                    vae,
                    noise_scheduler,
                    condition=condition,
                    prompt_embedding=prompt_embeddings[""],
                    output_hw=batch["rgb"].shape[-2:],
                    inference_steps=args.inference_steps,
                    seed=args.seed,
                    device=device,
                    dtype=dtype,
                )
                light = "static" if sample["frame_id"] < 0 else f"{sample['frame_id']:06d}"
                write_image(
                    prediction_root / sample["object"] / sample["camera"] / light / "pred.png",
                    prediction,
                )
    print(f"[train:{stage}] wrote previews to {prediction_root}")


def _preview_indices(dataset: ICTPolarRealTrainingDataset, *, stage: str, count: int) -> list[int]:
    selected = []
    seen_cameras = set()
    for index, record in enumerate(dataset.records):
        camera_key = (record.camera.object_name, record.camera.camera)
        if camera_key in seen_cameras:
            continue
        if stage == "forward" and record.light_index is None:
            continue
        seen_cameras.add(camera_key)
        selected.append(index)
        if len(selected) >= count:
            break
    return selected


def _sample_image(
    unet,
    vae,
    scheduler,
    *,
    condition,
    prompt_embedding,
    output_hw: tuple[int, int],
    inference_steps: int,
    seed: int,
    device,
    dtype,
):
    import numpy as np
    import torch
    from diffusers import DDIMScheduler

    with torch.inference_mode():
        sample_scheduler = DDIMScheduler.from_config(scheduler.config)
        sample_scheduler.set_timesteps(inference_steps, device=device)
        generator = torch.Generator(device=device).manual_seed(seed)
        latent_hw = condition.shape[-2:]
        latents = torch.randn((1, 4, *latent_hw), generator=generator, device=device, dtype=dtype)
        latents *= sample_scheduler.init_noise_sigma
        embeddings = prompt_embedding.repeat(latents.shape[0], 1, 1)
        for timestep in sample_scheduler.timesteps:
            scaled = sample_scheduler.scale_model_input(latents, timestep)
            model_input = torch.cat((scaled, condition), dim=1)
            prediction = unet(
                model_input,
                timestep,
                encoder_hidden_states=embeddings,
                return_dict=False,
            )[0]
            latents = sample_scheduler.step(prediction, timestep, latents, return_dict=False)[0]
        decoded = vae.decode(
            (latents / vae.config.scaling_factor).to(dtype=dtype),
            return_dict=False,
        )[0]
        image = (decoded.float().clamp(-1.0, 1.0) * 0.5 + 0.5)[0].permute(1, 2, 0).cpu().numpy()
    if image.shape[:2] != output_hw:
        import cv2

        image = cv2.resize(image, (output_hw[1], output_hw[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(image, 0.0, 1.0)
