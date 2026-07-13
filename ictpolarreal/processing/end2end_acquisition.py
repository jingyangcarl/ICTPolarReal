from __future__ import annotations

import hashlib
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ictpolarreal.utils.io import write_image


PERCENTILE = 99.5
MIN_LR_RATIO = 0.01


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
    base_color: np.ndarray,
    normal: np.ndarray,
    mask: np.ndarray | None,
    view_dirs: np.ndarray,
    out_dir: str | Path,
    imaginaire_root: str | Path,
    device: str = "cuda",
    steps: int = 33000,
    learning_rate: float = 1e-3,
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
    foreground = _foreground_mask(mask, height, width)
    if not np.any(foreground > 0.5):
        raise ValueError("end2end acquisition requires a non-empty foreground mask")
    target_stack = 2.0 * cross + 2.0 * np.maximum(parallel - cross, 0.0)
    target_stack = _normalize_targets(target_stack, foreground)
    base_color = _normalize_base_color(base_color, foreground)
    normal = _normalize_vectors(normal)
    view_dirs = _normalize_vectors(view_dirs)

    model = model_class(height, width, device=torch_device, cfg=param_config).to(torch_device)
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
        "schema": "ictpolarreal.end2end-checkpoint.v1",
        "height": height,
        "width": width,
        "lights": n_lights,
        "light_directions_sha256": hashlib.sha256(
            np.ascontiguousarray(light_dirs).tobytes()
        ).hexdigest(),
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "disney_brdf_sha256": provenance["disney_brdf_sha256"],
    }

    def render_loss(light_index: int):
        target = target_chw[light_index].to(device=torch_device, non_blocking=True)
        prediction, _, _ = model(
            V=views,
            L_dir=lights[light_index : light_index + 1],
            L_rgb=light_rgb,
            mask=mask_hwc,
            tone_map_method="linear",
        )
        residual = (prediction - target) * mask_chw
        return residual.square().sum() / foreground_values

    evaluation_indices = np.linspace(
        0, n_lights - 1, min(n_lights, 16), dtype=np.int64
    )
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
        loss = render_loss(step % n_lights)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite end2end loss at iteration {step + 1}")
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step == 0 or (step + 1) % log_every == 0 or step + 1 == steps:
            print(
                f"[end2end] iteration {step + 1}/{steps} "
                f"light={step % n_lights} loss={final_loss:.7f} "
                f"lr={optimizer.param_groups[0]['lr']:.3e}",
                flush=True,
            )
        if (step + 1) % checkpoint_every == 0 or step + 1 == steps:
            save_checkpoint(step + 1)

    model.eval()
    with torch.no_grad():
        final_evaluation_losses = [
            float(render_loss(int(index)).cpu()) for index in evaluation_indices
        ]
        maps = _material_maps_numpy(model)
    if start_step >= steps:
        final_loss = float(final_evaluation_losses[-1])

    _write_material_maps(out_dir, maps, foreground)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(state, out_dir / "disney_brdf.pt")

    metrics = {
        "schema": "ictpolarreal.end2end-disney.v1",
        "material_acquisition": "end2end",
        "model": "DisneyBRDFSimplifiedMultiLayer",
        "target": "2*cross + 2*max(parallel-cross, 0)",
        "target_percentile": PERCENTILE,
        "lights": int(n_lights),
        "steps": int(steps),
        "resumed_from_step": int(start_step),
        "learning_rate": float(learning_rate),
        "initial_evaluation_mse": initial_loss,
        "final_iteration_mse": final_loss,
        "evaluation_indices": [int(index) for index in evaluation_indices],
        "initial_evaluation_mse_by_light": initial_evaluation_losses,
        "final_evaluation_mse_by_light": final_evaluation_losses,
        "final_evaluation_mse": float(np.mean(final_evaluation_losses)),
        "evaluation_mse_improvement": initial_loss - float(np.mean(final_evaluation_losses)),
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
