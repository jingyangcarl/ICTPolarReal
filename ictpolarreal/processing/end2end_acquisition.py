from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ictpolarreal.utils.io import write_image
from ictpolarreal.utils.metrics import mae, mse, psnr, ssim_global


PERCENTILE = 99.5
MIN_LR_RATIO = 0.01
MIN_TRAIN_LIGHTS = 4
SCALAR_INIT_EPS = 1e-4
DISNEY_PHYSICAL_DEFAULTS = {
    "metallic": 0.0,
    "subsurface": 0.0,
    "specular": 0.5,
    "roughness": 0.5,
    "specularTint": 0.0,
    "anisotropic": 0.0,
    "sheen": 0.0,
    "sheenTint": 0.5,
    "clearcoat": 0.0,
    "clearcoatGloss": 0.5,
}


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(values).cast("B")).hexdigest()


def _initialize_disney_scalars(torch, model) -> None:
    """Initialize physical scalar defaults in the model's sigmoid-logit space."""
    with torch.no_grad():
        for name, physical_value in DISNEY_PHYSICAL_DEFAULTS.items():
            value = min(max(float(physical_value), SCALAR_INIT_EPS), 1.0 - SCALAR_INIT_EPS)
            unconstrained = math.log(value / (1.0 - value))
            getattr(model, f"{name}_un").fill_(unconstrained)


def split_light_indices(
    n_lights: int, eval_lights: int
) -> tuple[np.ndarray, np.ndarray]:
    if n_lights < MIN_TRAIN_LIGHTS:
        raise ValueError(
            f"end2end acquisition needs at least {MIN_TRAIN_LIGHTS} calibrated lights"
        )
    if eval_lights < 0:
        raise ValueError("end2end evaluation light count must be non-negative")
    heldout_count = min(int(eval_lights), max(n_lights - MIN_TRAIN_LIGHTS, 0))
    if heldout_count == 0:
        return np.arange(n_lights, dtype=np.int64), np.zeros((0,), dtype=np.int64)
    heldout = np.linspace(0, n_lights - 1, heldout_count, dtype=np.int64)
    heldout = np.unique(heldout)
    train = np.setdiff1d(np.arange(n_lights, dtype=np.int64), heldout)
    return train, heldout


def validate_end2end_runtime(imaginaire_root: str | Path, device: str = "cuda") -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The end2end acquisition requires PyTorch and the Imaginaire runtime dependencies."
        ) from exc
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "end2end material acquisition is CUDA-only. Submit it with `run.sh process "
            "--material-acquisition end2end --slurm --backend torch --device cuda`."
        )
    _load_imaginaire_disney(imaginaire_root)


def acquire_disney_material(
    cross_stack: np.ndarray,
    parallel_stack: np.ndarray,
    light_dirs: np.ndarray,
    *,
    light_ids: np.ndarray | None = None,
    frame_ids: np.ndarray | None = None,
    base_color: np.ndarray,
    normal: np.ndarray,
    mask: np.ndarray | None,
    view_dirs: np.ndarray,
    out_dir: str | Path,
    imaginaire_root: str | Path,
    device: str = "cuda",
    steps: int = 33000,
    learning_rate: float = 1e-3,
    eval_lights: int = 16,
) -> dict[str, Any]:
    """Fit Imaginaire's per-pixel Disney material model to ICTPolarReal OLATs.

    The SuperDimension data loader is deliberately not used. ICTPolarReal supplies
    calibrated light/view directions and polarization-separated observations, while
    the Disney renderer and optimizer behavior come from the external Imaginaire
    checkout selected by ``imaginaire_root``.
    """
    import torch

    if steps <= 0:
        raise ValueError("end2end steps must be a positive integer")
    if learning_rate <= 0:
        raise ValueError("end2end learning rate must be positive")
    if eval_lights < 0:
        raise ValueError("end2end evaluation light count must be non-negative")
    torch_device = torch.device(device)
    validate_end2end_runtime(imaginaire_root, device)

    root, disney_module, source_path = _load_imaginaire_disney(imaginaire_root)
    model_class = disney_module.DisneyBRDFSimplifiedMultiLayer
    param_config = disney_module.DisneyParamConfig(per_pixel=True, height_mode="none")

    cross = _clean_stack(cross_stack)
    parallel = _clean_stack(parallel_stack)
    if cross.shape != parallel.shape or cross.ndim != 4 or cross.shape[-1] != 3:
        raise ValueError("cross and parallel OLAT stacks must share shape (N,H,W,3)")
    if len(light_dirs) != len(cross):
        raise ValueError("light direction count must match the OLAT stacks")

    n_lights, height, width, _ = cross.shape
    _validate_spatial_inputs(base_color, normal, mask, view_dirs, height, width)
    light_dirs = np.asarray(light_dirs, dtype=np.float32)
    if light_dirs.shape != (n_lights, 3) or not np.isfinite(light_dirs).all():
        raise ValueError(f"light directions must be finite with shape ({n_lights},3)")
    if np.any(np.linalg.norm(light_dirs, axis=-1) <= 1e-8):
        raise ValueError("end2end acquisition received a zero-length light direction")
    if light_ids is None:
        light_ids = np.arange(n_lights, dtype=np.int64)
    light_ids = np.asarray(light_ids, dtype=np.int64)
    if light_ids.shape != (n_lights,) or len(np.unique(light_ids)) != n_lights:
        raise ValueError(f"light ids must be unique with shape ({n_lights},)")
    if frame_ids is None:
        frame_ids = light_ids.copy()
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    if frame_ids.shape != (n_lights,) or len(np.unique(frame_ids)) != n_lights:
        raise ValueError(f"frame ids must be unique with shape ({n_lights},)")
    train_indices, heldout_indices = split_light_indices(n_lights, eval_lights)
    evaluation_indices = (
        heldout_indices
        if len(heldout_indices)
        else np.linspace(0, n_lights - 1, min(n_lights, 16), dtype=np.int64)
    )
    evaluation_split = "heldout_olat" if len(heldout_indices) else "fitted_olat"
    print(
        f"[end2end] light split: {len(train_indices)} fit, "
        f"{len(heldout_indices)} held out for relighting evaluation",
        flush=True,
    )
    foreground = _foreground_mask(mask, height, width)
    if not np.any(foreground > 0.5):
        raise ValueError("end2end acquisition requires a non-empty foreground mask")
    target_stack = 2.0 * cross + 2.0 * np.maximum(parallel - cross, 0.0)
    target_stack = _normalize_targets(target_stack, foreground)
    base_color = _normalize_base_color(base_color, foreground)
    normal = _normalize_vectors(normal)
    view_dirs = _normalize_vectors(view_dirs)
    input_hashes = {
        "normalized_targets_sha256": _array_sha256(target_stack),
        "foreground_sha256": _array_sha256(foreground),
        "base_color_sha256": _array_sha256(base_color),
        "normal_sha256": _array_sha256(normal),
        "view_directions_sha256": _array_sha256(view_dirs),
    }

    model = model_class(height, width, device=torch_device, cfg=param_config).to(torch_device)
    _initialize_disney_scalars(torch, model)
    model.init_basecolor_from_image(
        torch.as_tensor(base_color, device=torch_device), require_grad=False
    )
    model.init_normal_from_image(
        torch.as_tensor(normal, device=torch_device),
        in_range="m11",
        require_grad=False,
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    lights = torch.as_tensor(
        np.ascontiguousarray(_normalize_vectors(light_dirs)), device=torch_device
    )
    views = torch.as_tensor(np.ascontiguousarray(view_dirs), device=torch_device)
    mask_hwc = torch.as_tensor(foreground, device=torch_device)
    mask_chw = mask_hwc.permute(2, 0, 1).contiguous()
    light_rgb = torch.ones((1, 3), dtype=torch.float32, device=torch_device)
    foreground_values = mask_chw.sum().clamp_min(1.0) * 3.0
    target_chw = torch.from_numpy(
        np.ascontiguousarray(target_stack.transpose(0, 3, 1, 2))
    )
    del target_stack

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = _imaginaire_provenance(root, source_path)
    checkpoint_path = out_dir / "end2end_checkpoint.pt"
    checkpoint_temp_path = out_dir / "end2end_checkpoint.tmp"
    signature = {
        "schema": "ictpolarreal.end2end-checkpoint.v4",
        "height": height,
        "width": width,
        "lights": n_lights,
        "light_directions_sha256": _array_sha256(light_dirs),
        "light_ids_sha256": _array_sha256(light_ids),
        "frame_ids_sha256": _array_sha256(frame_ids),
        **input_hashes,
        "train_indices": [int(index) for index in train_indices],
        "heldout_indices": [int(index) for index in heldout_indices],
        "scalar_initialization": {
            "physical_values": DISNEY_PHYSICAL_DEFAULTS,
            "boundary_epsilon": SCALAR_INIT_EPS,
        },
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "disney_brdf_sha256": provenance["disney_brdf_sha256"],
    }

    def render_prediction(light_index: int):
        prediction, _, _ = model(
            V=views,
            L_dir=lights[light_index : light_index + 1],
            L_rgb=light_rgb,
            mask=mask_hwc,
            return_hdr=True,
        )
        return _normalize_render_foreground(prediction, mask_chw)

    def render_loss(light_index: int):
        target = target_chw[light_index].to(device=torch_device, non_blocking=True)
        prediction = render_prediction(light_index)
        residual = (prediction - target) * mask_chw
        return residual.square().sum() / foreground_values

    start_step = 0
    initial_evaluation_losses = None
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=torch_device)
        if checkpoint.get("signature") != signature:
            raise RuntimeError(
                f"Existing checkpoint {checkpoint_path} does not match this acquisition. "
                "Use a different --material-root or remove the stale checkpoint."
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["next_step"])
        initial_evaluation_losses = checkpoint["initial_evaluation_losses"]
        print(
            f"[end2end] resuming {checkpoint_path} at iteration {start_step}/{steps}",
            flush=True,
        )
    if initial_evaluation_losses is None:
        with torch.no_grad():
            initial_evaluation_losses = [
                float(render_loss(int(index)).cpu()) for index in evaluation_indices
            ]
    initial_loss = float(np.mean(initial_evaluation_losses))

    def save_checkpoint(next_step: int) -> None:
        torch.save(
            {
                "signature": signature,
                "next_step": int(next_step),
                "initial_evaluation_losses": initial_evaluation_losses,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            checkpoint_temp_path,
        )
        checkpoint_temp_path.replace(checkpoint_path)

    log_every = max(1, min(500, steps // 20))
    checkpoint_every = max(1000, n_lights * 10)
    final_loss = initial_loss
    model.train()
    for step in range(start_step, steps):
        progress = step / max(steps - 1, 1)
        lr_scale = MIN_LR_RATIO + 0.5 * (1.0 - MIN_LR_RATIO) * (
            1.0 + math.cos(math.pi * progress)
        )
        optimizer.param_groups[0]["lr"] = learning_rate * lr_scale
        optimizer.zero_grad(set_to_none=True)
        light_index = int(train_indices[step % len(train_indices)])
        loss = render_loss(light_index)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite end2end loss at iteration {step + 1}")
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step == 0 or (step + 1) % log_every == 0 or step + 1 == steps:
            print(
                f"[end2end] iteration {step + 1}/{steps} "
                f"light={light_ids[light_index]} loss={final_loss:.7f} "
                f"lr={optimizer.param_groups[0]['lr']:.3e}",
                flush=True,
            )
        if (step + 1) % checkpoint_every == 0 or step + 1 == steps:
            save_checkpoint(step + 1)

    model.eval()
    with torch.no_grad():
        relighting_summary, final_evaluation_losses = _write_relighting_evaluation(
            render_prediction,
            target_chw,
            evaluation_indices,
            light_ids,
            frame_ids,
            foreground,
            out_dir / "relighting",
            split=evaluation_split,
        )
        maps = _material_maps_numpy(model)
        if start_step >= steps:
            final_loss = float(render_loss(int(train_indices[-1])).cpu())

    _write_material_maps(out_dir, maps, foreground)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(state, out_dir / "disney_brdf.pt")

    metrics = {
        "schema": "ictpolarreal.end2end-disney.v3",
        "material_acquisition": "end2end",
        "model": "DisneyBRDFSimplifiedMultiLayer",
        "target": "2*cross + 2*max(parallel-cross, 0)",
        "target_percentile": PERCENTILE,
        "lights": int(n_lights),
        "fit_lights": int(len(train_indices)),
        "heldout_lights": int(len(heldout_indices)),
        "steps": int(steps),
        "resumed_from_step": int(start_step),
        "learning_rate": float(learning_rate),
        "initial_evaluation_mse": initial_loss,
        "final_iteration_mse": final_loss,
        "evaluation_split": evaluation_split,
        "evaluation_indices": [int(index) for index in evaluation_indices],
        "evaluation_light_indices": [int(light_ids[index]) for index in evaluation_indices],
        "evaluation_frame_ids": [int(frame_ids[index]) for index in evaluation_indices],
        "initial_evaluation_mse_by_light": initial_evaluation_losses,
        "final_evaluation_mse_by_light": final_evaluation_losses,
        "final_evaluation_mse": float(np.mean(final_evaluation_losses)),
        "evaluation_mse_improvement": initial_loss - float(np.mean(final_evaluation_losses)),
        "light_split": {
            "requested_heldout_lights": int(eval_lights),
            "fit_stack_indices": [int(index) for index in train_indices],
            "fit_light_indices": [int(light_ids[index]) for index in train_indices],
            "fit_frame_ids": [int(frame_ids[index]) for index in train_indices],
            "heldout_stack_indices": [int(index) for index in heldout_indices],
            "heldout_light_indices": [int(light_ids[index]) for index in heldout_indices],
            "heldout_frame_ids": [int(frame_ids[index]) for index in heldout_indices],
        },
        "input_hashes": input_hashes,
        "scalar_initialization": {
            "physical_values": DISNEY_PHYSICAL_DEFAULTS,
            "boundary_epsilon": SCALAR_INIT_EPS,
        },
        "relighting": relighting_summary,
        "imaginaire": provenance,
    }
    (out_dir / "acquisition.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if provenance["dirty"]:
        print(
            "[end2end] Warning: the selected Imaginaire checkout is dirty; "
            "the exact disney_brdf.py hash was recorded in acquisition.json.",
            flush=True,
        )
    checkpoint_path.unlink(missing_ok=True)
    checkpoint_temp_path.unlink(missing_ok=True)
    return metrics


def _load_imaginaire_disney(imaginaire_root: str | Path):
    root = Path(imaginaire_root).expanduser().resolve()
    source_path = root / "CookTorrance_IBL" / "disney_brdf.py"
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Could not find Imaginaire Disney BRDF implementation at {source_path}. "
            "Pass --imaginaire-root PATH."
        )
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    try:
        module = importlib.import_module("CookTorrance_IBL.disney_brdf")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import Imaginaire's Disney BRDF runtime. The selected Python "
            "environment must provide torch, torchvision, scipy, numpy, and Pillow."
        ) from exc
    imported_path = Path(module.__file__).resolve()
    if imported_path != source_path:
        raise RuntimeError(
            f"Imported Disney BRDF from {imported_path}, expected {source_path}. "
            "Use a clean Python process or correct --imaginaire-root."
        )
    return root, module, source_path


def _normalize_targets(targets: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    targets = np.maximum(np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    selected = foreground[..., 0] > 0.5
    flat = targets[:, selected, :].reshape(len(targets), -1)
    scales = np.quantile(flat, PERCENTILE / 100.0, axis=1).astype(np.float32)
    scales = np.maximum(scales, 1e-8)
    return np.clip(targets / scales[:, None, None, None], 0.0, 1.0).astype(np.float32)


def _normalize_render_foreground(render, mask_chw):
    """Apply the target's foreground-only p99.5 scale to a rendered HDR tensor."""
    finite = render.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    valid = mask_chw.expand_as(render) > 0.5
    values = finite.masked_select(valid)
    if values.numel() == 0:
        raise ValueError("cannot normalize a render without finite foreground pixels")
    scale = values.float().quantile(PERCENTILE / 100.0).to(render.dtype).clamp_min(1e-8)
    return (finite / scale).clamp_max(1.0) * mask_chw


def _normalize_base_color(base_color: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    image = np.maximum(np.nan_to_num(base_color, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    selected = foreground[..., 0] > 0.5
    values = image[selected]
    scale = float(np.quantile(values, PERCENTILE / 100.0)) if values.size else 1.0
    return np.clip(image / max(scale, 1e-8), 0.0, 1.0).astype(np.float32)


def _foreground_mask(mask: np.ndarray | None, height: int, width: int) -> np.ndarray:
    if mask is None:
        return np.ones((height, width, 1), dtype=np.float32)
    if mask.ndim == 2:
        mask = mask[..., None]
    return (mask[..., :1] > 0.5).astype(np.float32)


def _material_maps_numpy(model) -> dict[str, np.ndarray]:
    constrained = model._param_maps()
    maps = {}
    for name, value in constrained.items():
        if name in {"height", "height2normal"}:
            continue
        array = value.detach().float().cpu().numpy()
        if array.ndim == 2:
            array = array[..., None]
        maps[name] = array.astype(np.float32)
    return maps


def _write_material_maps(
    out_dir: Path, maps: dict[str, np.ndarray], foreground: np.ndarray
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    aliases = {
        "albedo": "baseColor",
        "normal": "normal",
        "specular": "specular",
        "roughness": "roughness",
        "metallic": "metallic",
        "specularTint": "specularTint",
        "subsurface": "subsurface",
        "anisotropic": "anisotropic",
        "clearcoat": "clearcoat",
        "clearcoatGloss": "clearcoatGloss",
        "baseColor": "baseColor",
    }
    for output_name, map_name in aliases.items():
        image = maps[map_name]
        if map_name == "normal":
            image = image * 0.5 + 0.5
        write_image(out_dir / f"{output_name}.png", np.clip(image, 0.0, 1.0) * foreground)


def _write_relighting_evaluation(
    render_prediction,
    target_chw,
    evaluation_indices: np.ndarray,
    light_ids: np.ndarray,
    frame_ids: np.ndarray,
    foreground: np.ndarray,
    relighting_dir: Path,
    *,
    split: str,
) -> tuple[dict[str, Any], list[float]]:
    if relighting_dir.exists():
        shutil.rmtree(relighting_dir)
    relighting_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    comparisons = []
    for stack_index in evaluation_indices:
        stack_index = int(stack_index)
        prediction = (
            render_prediction(stack_index)
            .detach()
            .float()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )
        target = target_chw[stack_index].permute(1, 2, 0).numpy()
        prediction = np.clip(prediction, 0.0, 1.0)
        target = np.clip(target, 0.0, 1.0)
        error = np.abs(prediction - target) * foreground

        frame_id = int(frame_ids[stack_index])
        light_id = int(light_ids[stack_index])
        frame_dir = relighting_dir / f"{frame_id:06d}"
        pred_path = frame_dir / "pred.png"
        gt_path = frame_dir / "gt.png"
        error_path = frame_dir / "error.png"
        comparison_path = frame_dir / "comparison.png"
        write_image(pred_path, prediction * foreground)
        write_image(gt_path, target * foreground)
        write_image(error_path, error)
        comparison = np.concatenate(
            [target * foreground, prediction * foreground, np.clip(error * 4.0, 0.0, 1.0)],
            axis=1,
        )
        write_image(comparison_path, comparison)
        comparisons.append((f"frame {frame_id:06d}", comparison))
        rows.append(
            {
                "split": split,
                "stack_index": stack_index,
                "light_index": light_id,
                "frame_id": frame_id,
                "mse": mse(prediction, target, foreground),
                "mae": mae(prediction, target, foreground),
                "psnr": psnr(prediction, target, foreground),
                "ssim_global": ssim_global(prediction, target, foreground),
                "pred_path": str(pred_path),
                "gt_path": str(gt_path),
                "error_path": str(error_path),
            }
        )

    metrics_path = relighting_dir.parent / "relighting_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    metric_names = ("mse", "mae", "psnr", "ssim_global")
    summary = {
        "schema": "ictpolarreal.relighting-evaluation.v1",
        "split": split,
        "count": len(rows),
        "normalization": "independent foreground p99.5 linear clipping",
        "target": "2*cross + 2*max(parallel-cross, 0)",
        "metrics": {
            name: float(np.mean([row[name] for row in rows])) for name in metric_names
        },
        "frame_ids": [row["frame_id"] for row in rows],
        "light_indices": [row["light_index"] for row in rows],
        "metrics_csv": str(metrics_path),
        "contact_sheet": str(relighting_dir.parent / "relighting_contact_sheet.png"),
        "panels": ["ground_truth", "prediction", "absolute_error_x4"],
    }
    summary_path = relighting_dir.parent / "relighting_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_relighting_contact_sheet(
        comparisons, relighting_dir.parent / "relighting_contact_sheet.png"
    )
    print(
        f"[end2end] {split} relighting: count={len(rows)} "
        f"PSNR={summary['metrics']['psnr']:.3f} "
        f"SSIM-global={summary['metrics']['ssim_global']:.4f}",
        flush=True,
    )
    return summary, [float(row["mse"]) for row in rows]


def _write_relighting_contact_sheet(
    comparisons: list[tuple[str, np.ndarray]], path: Path
) -> None:
    from PIL import Image, ImageDraw

    thumbnails = []
    for label, comparison in comparisons:
        array = (np.clip(comparison, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        image = Image.fromarray(array)
        target_width = 720
        target_height = max(1, round(image.height * target_width / image.width))
        image = image.resize((target_width, target_height), Image.Resampling.BILINEAR)
        header_height = 38
        tile = Image.new("RGB", (target_width, target_height + header_height), color="black")
        tile.paste(image, (0, header_height))
        draw = ImageDraw.Draw(tile)
        draw.text((6, 3), label, fill="white")
        panel_width = target_width // 3
        draw.text((6, 20), "ground truth", fill="white")
        draw.text((panel_width + 6, 20), "prediction", fill="white")
        draw.text((2 * panel_width + 6, 20), "absolute error x4", fill="white")
        thumbnails.append(tile)

    columns = 2
    rows = math.ceil(len(thumbnails) / columns)
    tile_width = max(image.width for image in thumbnails)
    tile_height = max(image.height for image in thumbnails)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), color="black")
    for index, image in enumerate(thumbnails):
        sheet.paste(image, ((index % columns) * tile_width, (index // columns) * tile_height))
    sheet.save(path)


def _imaginaire_provenance(root: Path, source_path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()

    def git_output(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    status = git_output("status", "--porcelain", "--untracked-files=no")
    return {
        "root": str(root),
        "commit": git_output("rev-parse", "HEAD"),
        "dirty": bool(status and status != "unknown"),
        "disney_brdf_sha256": digest,
    }


def _clean_stack(stack: np.ndarray) -> np.ndarray:
    values = np.asarray(stack, dtype=np.float32)
    return np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, 1e-8)


def _validate_spatial_inputs(
    base_color: np.ndarray,
    normal: np.ndarray,
    mask: np.ndarray | None,
    view_dirs: np.ndarray,
    height: int,
    width: int,
) -> None:
    expected_vector = (height, width, 3)
    if np.asarray(base_color).shape != expected_vector:
        raise ValueError(f"base color must have shape {expected_vector}")
    if np.asarray(normal).shape != expected_vector:
        raise ValueError(f"normal must have shape {expected_vector}")
    if np.asarray(view_dirs).shape != expected_vector:
        raise ValueError(f"view directions must have shape {expected_vector}")
    if mask is not None and np.asarray(mask).shape not in {
        (height, width),
        (height, width, 1),
    }:
        raise ValueError(f"mask must have shape ({height},{width}) or ({height},{width},1)")
