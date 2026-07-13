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
from typing import Any, Sequence

import numpy as np

from ictpolarreal.utils.io import read_image, write_image
from ictpolarreal.utils.metrics import mae, mse, psnr, ssim_global
from ictpolarreal.processing.lighting_profiles import (
    EnvironmentCondition,
    mix_condition_kind,
    parse_lighting_profiles,
    prepare_environment_conditions,
)


PERCENTILE = 99.5
MIN_LR_RATIO = 0.01
MIN_TRAIN_LIGHTS = 4
SCALAR_INIT_EPS = 1e-4
MAX_QUANTILE_ELEMENTS = 1 << 23
HDRI_AUTOGRAD_BYTES_PER_LIGHT_PIXEL = 216
MIN_HDRI_GPU_MEMORY_BYTES = 40 * (1 << 30)
PROFILE_MODEL = "simplified-multilayer"
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
    lighting_profiles: str | Sequence[str] = "olat,hdri,mix",
    hdri_root: str | Path | None = None,
    hdri_count: int = 100,
    eval_hdris: int = 4,
    hdri_rotations: int = 4,
    primary_profile: str = "olat",
) -> dict[str, Any]:
    """Fit independent OLAT, HDRI, and mixed Disney material profiles.

    ICTPolarReal supplies the calibrated 346-light basis and polarized observations.
    The HDRI targets are synthesized from those measured observations with the same
    spherical-Voronoi construction used by Imaginaire's SuperDimension pipeline.
    """
    import torch

    if steps <= 0:
        raise ValueError("end2end steps must be a positive integer")
    if learning_rate <= 0:
        raise ValueError("end2end learning rate must be positive")
    if eval_lights < 0:
        raise ValueError("end2end evaluation light count must be non-negative")
    if hdri_count <= 0:
        raise ValueError("end2end HDRI count must be positive")
    if eval_hdris < 0:
        raise ValueError("end2end HDRI evaluation count must be non-negative")
    if hdri_rotations <= 0:
        raise ValueError("end2end HDRI rotations must be positive")
    profiles = parse_lighting_profiles(lighting_profiles)
    primary_profile = primary_profile.strip().lower()
    if primary_profile not in profiles:
        raise ValueError(
            f"primary end2end profile {primary_profile!r} is not in requested profiles {profiles}"
        )
    if hdri_root is None:
        raise ValueError(
            "end2end profile evaluation requires --end2end-hdri-root; HDRI targets "
            "are synthesized from measured ICTPolarReal OLATs"
        )
    torch_device = torch.device(device)
    validate_end2end_runtime(imaginaire_root, device)

    root, disney_module, source_path = _load_imaginaire_disney(imaginaire_root)

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
    if any(profile in {"hdri", "mix"} for profile in profiles):
        _validate_hdri_gpu_memory(
            torch,
            torch_device,
            len(train_indices),
            height,
            width,
        )
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
    raw_target_stack = np.maximum(
        2.0 * cross + 2.0 * np.maximum(parallel - cross, 0.0), 0.0
    ).astype(np.float32)
    target_stack = _normalize_targets(raw_target_stack, foreground)
    base_color = _normalize_base_color(base_color, foreground)
    normal = _normalize_vectors(normal)
    view_dirs = _normalize_vectors(view_dirs)
    input_hashes = {
        "normalized_targets_sha256": _array_sha256(target_stack),
        "foreground_sha256": _array_sha256(foreground),
        "base_color_sha256": _array_sha256(base_color),
        "normal_sha256": _array_sha256(normal),
        "view_directions_sha256": _array_sha256(view_dirs),
        "raw_polarized_targets_sha256": _array_sha256(raw_target_stack),
    }
    target_chw = torch.from_numpy(
        np.ascontiguousarray(target_stack.transpose(0, 3, 1, 2))
    )
    raw_target_chw = torch.from_numpy(
        np.ascontiguousarray(raw_target_stack.transpose(0, 3, 1, 2))
    )
    del target_stack, raw_target_stack

    camera_dir = Path(out_dir)
    camera_dir.mkdir(parents=True, exist_ok=True)
    lighting_dir = camera_dir / "lighting"
    print(
        f"[end2end] preparing {hdri_count} fit and {eval_hdris} held-out HDRIs "
        f"with {hdri_rotations} rotations",
        flush=True,
    )
    heldout_support = heldout_indices if len(heldout_indices) else train_indices
    projection_evaluation_support = (
        heldout_support if eval_hdris > 0 else train_indices
    )
    environments = prepare_environment_conditions(
        hdri_root,
        light_dirs,
        train_count=hdri_count,
        eval_count=eval_hdris,
        rotations=hdri_rotations,
        out_dir=lighting_dir,
        fit_support_indices=train_indices,
        evaluation_support_indices=projection_evaluation_support,
    )
    evaluation_environments = environments.evaluation
    evaluation_support = projection_evaluation_support
    hdri_evaluation_split = "heldout_hdri"
    if not evaluation_environments:
        natural_fit_environments = [
            condition
            for condition in environments.train
            if condition.source_kind == "environment_map"
        ]
        fallback_environments = natural_fit_environments or environments.train
        evaluation_environments = fallback_environments[
            : min(4, len(fallback_environments))
        ]
        evaluation_support = train_indices
        hdri_evaluation_split = "fitted_hdri"
    hdri_fit_targets = _synthesize_environment_targets(
        torch,
        raw_target_chw,
        environments.train,
        train_indices,
        foreground,
        torch_device,
        storage_dtype=torch.float16,
    )
    hdri_evaluation_targets = _synthesize_environment_targets(
        torch,
        raw_target_chw,
        evaluation_environments,
        evaluation_support,
        foreground,
        torch_device,
        storage_dtype=torch.float32,
    )
    del raw_target_chw
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()

    provenance = _imaginaire_provenance(root, source_path)
    adapter_provenance = _adapter_provenance()
    profile_results = {}
    for profile in profiles:
        profile_dir = camera_dir / profile / PROFILE_MODEL
        print(
            f"[end2end] fitting lighting profile {profile} -> {profile_dir}",
            flush=True,
        )
        profile_results[profile] = _fit_disney_profile(
            torch=torch,
            disney_module=disney_module,
            profile=profile,
            profile_dir=profile_dir,
            camera_dir=camera_dir,
            target_chw=target_chw,
            hdri_fit_targets=hdri_fit_targets,
            hdri_evaluation_targets=hdri_evaluation_targets,
            hdri_fit_conditions=environments.train,
            hdri_evaluation_conditions=evaluation_environments,
            hdri_evaluation_split=hdri_evaluation_split,
            light_dirs=light_dirs,
            light_ids=light_ids,
            frame_ids=frame_ids,
            train_indices=train_indices,
            heldout_indices=heldout_indices,
            evaluation_indices=evaluation_indices,
            evaluation_split=evaluation_split,
            evaluation_support=evaluation_support,
            base_color=base_color,
            normal=normal,
            foreground=foreground,
            view_dirs=view_dirs,
            device=torch_device,
            steps=steps,
            learning_rate=learning_rate,
            hdri_rotations=hdri_rotations,
            eval_lights=eval_lights,
            input_hashes=input_hashes,
            provenance=provenance,
            adapter_provenance=adapter_provenance,
        )

    manifest = {
        "schema": "ictpolarreal.material-profiles.v1",
        "material_acquisition": "end2end",
        "model": "DisneyBRDFSimplifiedMultiLayer",
        "profiles": list(profiles),
        "primary_profile": primary_profile,
        "primary_material_dir": f"{primary_profile}/{PROFILE_MODEL}/material/maps",
        "lighting": {
            "conditions": "lighting/conditions.json",
            "weights": "lighting/weights.npz",
            "fit_hdri_conditions": len(environments.train),
            "fit_natural_hdri_identities": int(hdri_count),
            "fit_calibration_conditions": int(4 * hdri_rotations),
            "evaluation_hdri_conditions": len(evaluation_environments),
            "evaluation_natural_hdri_identities": int(eval_hdris),
            "target_origin": "synthesized_from_measured_olat",
        },
        "evaluation_matrix": "each profile evaluated on the same OLAT and HDRI suites",
        "report": {"status": "pending"},
        "adapter": adapter_provenance,
        "profile_acquisitions": {
            profile: f"{profile}/{PROFILE_MODEL}/acquisition.json" for profile in profiles
        },
    }
    manifest_path = camera_dir / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    try:
        report = _write_camera_report(camera_dir, profiles, profile_results)
    except Exception as exc:
        manifest["report"] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json_atomic(manifest_path, manifest)
        raise
    manifest["report"] = {"status": "complete", **report}
    _write_json_atomic(manifest_path, manifest)
    if provenance["dirty"]:
        print(
            "[end2end] Warning: the selected Imaginaire checkout is dirty; "
            "exact source hashes were recorded in each acquisition.json.",
            flush=True,
        )
    return manifest


def _fit_disney_profile(
    *,
    torch,
    disney_module,
    profile: str,
    profile_dir: Path,
    camera_dir: Path,
    target_chw,
    hdri_fit_targets,
    hdri_evaluation_targets,
    hdri_fit_conditions: Sequence[EnvironmentCondition],
    hdri_evaluation_conditions: Sequence[EnvironmentCondition],
    hdri_evaluation_split: str,
    light_dirs: np.ndarray,
    light_ids: np.ndarray,
    frame_ids: np.ndarray,
    train_indices: np.ndarray,
    heldout_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    evaluation_split: str,
    evaluation_support: np.ndarray,
    base_color: np.ndarray,
    normal: np.ndarray,
    foreground: np.ndarray,
    view_dirs: np.ndarray,
    device,
    steps: int,
    learning_rate: float,
    hdri_rotations: int,
    eval_lights: int,
    input_hashes: dict[str, str],
    provenance: dict[str, Any],
    adapter_provenance: dict[str, Any],
) -> dict[str, Any]:
    height, width = foreground.shape[:2]
    model_class = disney_module.DisneyBRDFSimplifiedMultiLayer
    param_config = disney_module.DisneyParamConfig(per_pixel=True, height_mode="none")
    model = model_class(height, width, device=device, cfg=param_config).to(device)
    _initialize_disney_scalars(torch, model)
    model.init_basecolor_from_image(torch.as_tensor(base_color, device=device), require_grad=False)
    model.init_normal_from_image(
        torch.as_tensor(normal, device=device), in_range="m11", require_grad=False
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    lights = torch.as_tensor(
        np.ascontiguousarray(_normalize_vectors(light_dirs)), device=device
    )
    views = torch.as_tensor(np.ascontiguousarray(view_dirs), device=device)
    mask_hwc = torch.as_tensor(foreground, device=device)
    mask_chw = mask_hwc.permute(2, 0, 1).contiguous()
    foreground_values = mask_chw.sum().clamp_min(1.0) * 3.0
    white_light = torch.ones((1, 3), dtype=torch.float32, device=device)
    fit_support = torch.as_tensor(train_indices, dtype=torch.long, device=device)
    eval_support = torch.as_tensor(evaluation_support, dtype=torch.long, device=device)
    fit_integration_weights = torch.ones(len(train_indices), device=device)
    eval_integration_weights = torch.ones(len(evaluation_support), device=device)
    fit_hdri_weights = torch.as_tensor(
        np.stack([condition.weights for condition in hdri_fit_conditions]), device=device
    )
    eval_hdri_weights = torch.as_tensor(
        np.stack([condition.weights for condition in hdri_evaluation_conditions]), device=device
    )

    def render_olat(stack_index: int):
        prediction, _, _ = model(
            V=views,
            L_dir=lights[stack_index : stack_index + 1],
            L_rgb=white_light,
            mask=mask_hwc,
            return_hdr=True,
        )
        return _normalize_render_foreground(prediction, mask_chw)

    def render_hdri(condition_index: int, *, evaluation: bool):
        support = eval_support if evaluation else fit_support
        weights = eval_hdri_weights if evaluation else fit_hdri_weights
        integration_weights = (
            eval_integration_weights if evaluation else fit_integration_weights
        )
        prediction, _, _ = model(
            V=views,
            L_dir=lights.index_select(0, support),
            L_rgb=weights[condition_index].index_select(0, support),
            light_weights=integration_weights,
            mask=mask_hwc,
            return_hdr=True,
        )
        return _normalize_render_foreground(prediction, mask_chw)

    def olat_loss(stack_index: int):
        target = target_chw[stack_index].to(device=device, non_blocking=True)
        residual = (render_olat(stack_index) - target) * mask_chw
        return residual.square().sum() / foreground_values

    def hdri_loss(condition_index: int, *, evaluation: bool):
        targets = hdri_evaluation_targets if evaluation else hdri_fit_targets
        target = targets[condition_index].to(device=device, dtype=torch.float32, non_blocking=True)
        residual = (render_hdri(condition_index, evaluation=evaluation) - target) * mask_chw
        return residual.square().sum() / foreground_values

    environment_hash = _array_sha256(
        np.stack(
            [
                condition.weights
                for condition in [
                    *hdri_fit_conditions,
                    *hdri_evaluation_conditions,
                ]
            ]
        )
    )
    checkpoint_dir = profile_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "latest.pt"
    checkpoint_temp_path = checkpoint_dir / "latest.tmp"
    signature = {
        "schema": "ictpolarreal.end2end-checkpoint.v6",
        "profile": profile,
        "model": PROFILE_MODEL,
        "height": height,
        "width": width,
        "light_directions_sha256": _array_sha256(light_dirs),
        "light_ids_sha256": _array_sha256(light_ids),
        "frame_ids_sha256": _array_sha256(frame_ids),
        **input_hashes,
        "fit_indices": [int(index) for index in train_indices],
        "heldout_indices": [int(index) for index in heldout_indices],
        "hdri_condition_weights_sha256": environment_hash,
        "hdri_fit_condition_ids": [condition.condition_id for condition in hdri_fit_conditions],
        "hdri_evaluation_condition_ids": [
            condition.condition_id for condition in hdri_evaluation_conditions
        ],
        "mix_schedule": f"{hdri_rotations}_hdri_then_{hdri_rotations}_olat",
        "scalar_initialization": {
            "physical_values": DISNEY_PHYSICAL_DEFAULTS,
            "boundary_epsilon": SCALAR_INIT_EPS,
        },
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "disney_brdf_sha256": provenance["disney_brdf_sha256"],
        "adapter": adapter_provenance,
    }
    acquisition_path = profile_dir / "acquisition.json"
    if acquisition_path.is_file():
        try:
            completed = json.loads(acquisition_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            completed = None
        if (
            completed is not None
            and completed.get("checkpoint_signature") == signature
            and _profile_outputs_complete(profile_dir, completed)
        ):
            print(
                f"[end2end:{profile}] reusing completed profile with matching provenance",
                flush=True,
            )
            return completed

    start_step = 0
    initial_evaluation_losses = None
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get("signature") != signature:
            raise RuntimeError(
                f"Existing checkpoint {checkpoint_path} does not match this "
                f"{profile} acquisition. "
                "Use a different --material-root or remove the stale checkpoint."
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["next_step"])
        initial_evaluation_losses = checkpoint["initial_evaluation_losses"]
        print(
            f"[end2end:{profile}] resuming at iteration {start_step}/{steps}", flush=True
        )
    if initial_evaluation_losses is None:
        with torch.no_grad():
            initial_evaluation_losses = {
                "olat": [
                    float(olat_loss(int(index)).cpu()) for index in evaluation_indices
                ],
                "hdri": [
                    float(hdri_loss(index, evaluation=True).cpu())
                    for index in range(len(hdri_evaluation_conditions))
                ],
            }

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

    def training_condition(step: int) -> tuple[str, int, str]:
        if profile == "olat":
            position = step % len(train_indices)
            stack_index = int(train_indices[position])
            return "olat", stack_index, f"light={int(light_ids[stack_index])}"
        if profile == "hdri":
            position = step % len(hdri_fit_conditions)
            return "hdri", position, f"hdri={hdri_fit_conditions[position].condition_id}"
        kind = mix_condition_kind(step, hdri_rotations)
        block_offset = (step // (2 * hdri_rotations)) * hdri_rotations
        position_in_block = step % hdri_rotations
        sequence_index = block_offset + position_in_block
        if kind == "hdri":
            position = sequence_index % len(hdri_fit_conditions)
            return "hdri", position, f"hdri={hdri_fit_conditions[position].condition_id}"
        position = sequence_index % len(train_indices)
        stack_index = int(train_indices[position])
        return "olat", stack_index, f"light={int(light_ids[stack_index])}"

    log_every = max(1, min(500, steps // 20))
    checkpoint_every = max(1000, len(train_indices) * 10)
    final_loss = float(np.mean(initial_evaluation_losses["olat"]))
    final_kind, final_index, final_label = training_condition(max(start_step - 1, 0))
    model.train()
    for step in range(start_step, steps):
        progress = step / max(steps - 1, 1)
        lr_scale = MIN_LR_RATIO + 0.5 * (1.0 - MIN_LR_RATIO) * (
            1.0 + math.cos(math.pi * progress)
        )
        optimizer.param_groups[0]["lr"] = learning_rate * lr_scale
        optimizer.zero_grad(set_to_none=True)
        final_kind, final_index, final_label = training_condition(step)
        if final_kind == "olat":
            loss = olat_loss(final_index)
        else:
            loss = hdri_loss(final_index, evaluation=False)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(
                f"Non-finite {profile} end2end loss at iteration {step + 1}"
            )
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step == 0 or (step + 1) % log_every == 0 or step + 1 == steps:
            print(
                f"[end2end:{profile}] iteration {step + 1}/{steps} "
                f"kind={final_kind} {final_label} loss={final_loss:.7f} "
                f"lr={optimizer.param_groups[0]['lr']:.3e}",
                flush=True,
            )
        if (step + 1) % checkpoint_every == 0 or step + 1 == steps:
            save_checkpoint(step + 1)

    model.eval()
    evaluation_dir = profile_dir / "evaluation"
    hdri_support_split = (
        "heldout_olat"
        if len(heldout_indices)
        and np.array_equal(evaluation_support, heldout_indices)
        else "fitted_olat"
    )
    with torch.no_grad():
        olat_summary, olat_losses = _write_relighting_evaluation(
            render_olat,
            target_chw,
            evaluation_indices,
            light_ids,
            frame_ids,
            foreground,
            evaluation_dir / "olat" / "cases",
            split=evaluation_split,
        )
        hdri_summary, hdri_losses = _write_hdri_evaluation(
            lambda index: render_hdri(index, evaluation=True),
            hdri_evaluation_targets,
            hdri_evaluation_conditions,
            foreground,
            evaluation_dir / "hdri",
            split=hdri_evaluation_split,
            olat_support_split=hdri_support_split,
        )
        maps = _material_maps_numpy(model)
        if start_step >= steps:
            if final_kind == "olat":
                final_loss = float(olat_loss(final_index).cpu())
            else:
                final_loss = float(hdri_loss(final_index, evaluation=False).cpu())

    material_dir = profile_dir / "material"
    maps_dir = material_dir / "maps"
    _write_material_maps(maps_dir, maps, foreground)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(state, material_dir / "disney_brdf.pt")
    evaluation_summary = {
        "schema": "ictpolarreal.profile-evaluation.v1",
        "profile": profile,
        "evaluations": {"olat": olat_summary, "hdri": hdri_summary},
    }
    (evaluation_dir / "summary.json").write_text(
        json.dumps(evaluation_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_combined_evaluation_csv(evaluation_dir, profile, olat_summary, hdri_summary)

    metrics = {
        "schema": "ictpolarreal.end2end-disney.v4",
        "material_acquisition": "end2end",
        "lighting_profile": profile,
        "model": "DisneyBRDFSimplifiedMultiLayer",
        "target": {
            "olat": "2*cross + 2*max(parallel-cross, 0)",
            "hdri": "spherical-Voronoi weighted sum of polarized OLAT targets",
            "hdri_origin": "synthesized_from_measured_olat",
            "percentile": PERCENTILE,
        },
        "steps": int(steps),
        "resumed_from_step": int(start_step),
        "learning_rate": float(learning_rate),
        "final_training_condition_mse": final_loss,
        "fit_conditions": {
            "olat": int(len(train_indices)) if profile in {"olat", "mix"} else 0,
            "hdri": int(len(hdri_fit_conditions)) if profile in {"hdri", "mix"} else 0,
        },
        "schedule": (
            f"{hdri_rotations} HDRI then {hdri_rotations} OLAT iterations"
            if profile == "mix"
            else f"{profile} only"
        ),
        "light_split": {
            "requested_heldout_lights": int(eval_lights),
            "fit_stack_indices": [int(index) for index in train_indices],
            "fit_light_indices": [int(light_ids[index]) for index in train_indices],
            "fit_frame_ids": [int(frame_ids[index]) for index in train_indices],
            "heldout_stack_indices": [int(index) for index in heldout_indices],
            "heldout_light_indices": [int(light_ids[index]) for index in heldout_indices],
            "heldout_frame_ids": [int(frame_ids[index]) for index in heldout_indices],
        },
        "initial_evaluation_mse": {
            name: float(np.mean(losses))
            for name, losses in initial_evaluation_losses.items()
        },
        "final_evaluation_mse": {
            "olat": float(np.mean(olat_losses)),
            "hdri": float(np.mean(hdri_losses)),
        },
        "evaluation_mse_improvement": {
            "olat": float(np.mean(initial_evaluation_losses["olat"]) - np.mean(olat_losses)),
            "hdri": float(np.mean(initial_evaluation_losses["hdri"]) - np.mean(hdri_losses)),
        },
        "paths": {
            "material_maps": _relative_path(maps_dir, camera_dir),
            "model": _relative_path(material_dir / "disney_brdf.pt", camera_dir),
            "evaluation": _relative_path(evaluation_dir, camera_dir),
        },
        "input_hashes": input_hashes,
        "hdri_condition_weights_sha256": environment_hash,
        "scalar_initialization": {
            "physical_values": DISNEY_PHYSICAL_DEFAULTS,
            "boundary_epsilon": SCALAR_INIT_EPS,
        },
        "evaluation": evaluation_summary,
        "imaginaire": provenance,
        "adapter": adapter_provenance,
        "renderer_integration": (
            "L_rgb stores solid-angle-integrated cell radiance; unit explicit "
            "light_weights prevent a second 4pi/N factor"
        ),
        "checkpoint_signature": signature,
    }
    _write_json_atomic(acquisition_path, metrics)
    checkpoint_path.unlink(missing_ok=True)
    checkpoint_temp_path.unlink(missing_ok=True)
    try:
        checkpoint_dir.rmdir()
    except OSError:
        pass
    return metrics


def _synthesize_environment_targets(
    torch,
    raw_target_chw,
    conditions: Sequence[EnvironmentCondition],
    support_indices: np.ndarray,
    foreground: np.ndarray,
    device,
    *,
    storage_dtype,
):
    if not conditions:
        raise ValueError("cannot synthesize HDRI targets without environment conditions")
    support = torch.as_tensor(support_indices, dtype=torch.long)
    raw_support = raw_target_chw.index_select(0, support).to(device=device)
    weights = torch.as_tensor(
        np.stack([condition.weights for condition in conditions]), device=device
    ).index_select(1, support.to(device=device))
    mask_chw = torch.as_tensor(foreground, device=device).permute(2, 0, 1).contiguous()
    output = torch.empty(
        (len(conditions), *raw_target_chw.shape[1:]),
        dtype=storage_dtype,
        device="cpu",
    )
    batch_size = 4
    with torch.no_grad():
        for start in range(0, len(conditions), batch_size):
            stop = min(start + batch_size, len(conditions))
            synthesized = torch.einsum(
                "bnc,nchw->bchw", weights[start:stop], raw_support
            )
            for offset in range(stop - start):
                normalized = _normalize_render_foreground(
                    synthesized[offset], mask_chw
                )
                output[start + offset].copy_(
                    normalized.to(device="cpu", dtype=storage_dtype)
                )
    del raw_support, weights
    if getattr(device, "type", None) == "cuda":
        torch.cuda.empty_cache()
    return output


def _write_hdri_evaluation(
    render_prediction,
    targets,
    conditions: Sequence[EnvironmentCondition],
    foreground: np.ndarray,
    evaluation_dir: Path,
    *,
    split: str,
    olat_support_split: str,
) -> tuple[dict[str, Any], list[float]]:
    if evaluation_dir.exists():
        shutil.rmtree(evaluation_dir)
    cases_dir = evaluation_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    comparisons = []
    for index, condition in enumerate(conditions):
        prediction = (
            render_prediction(index).detach().float().cpu().permute(1, 2, 0).numpy()
        )
        target = targets[index].float().permute(1, 2, 0).numpy()
        prediction = np.clip(prediction, 0.0, 1.0)
        target = np.clip(target, 0.0, 1.0)
        error = np.abs(prediction - target) * foreground
        case_dir = cases_dir / condition.condition_id
        lighting_path = case_dir / "lighting.png"
        pred_path = case_dir / "pred.png"
        gt_path = case_dir / "gt.png"
        error_path = case_dir / "error.png"
        comparison_path = case_dir / "comparison.png"
        write_image(lighting_path, condition.preview)
        write_image(pred_path, prediction * foreground)
        write_image(gt_path, target * foreground)
        write_image(error_path, error)
        lighting = _resize_panel_numpy(condition.preview, target.shape[:2])
        comparison = np.concatenate(
            [
                lighting,
                target * foreground,
                prediction * foreground,
                np.clip(error * 4.0, 0.0, 1.0),
            ],
            axis=1,
        )
        write_image(comparison_path, comparison)
        comparisons.append((condition.condition_id, comparison))
        rows.append(
            {
                "split": split,
                "condition_id": condition.condition_id,
                "source": condition.source_name,
                "source_kind": condition.source_kind,
                "rotation_degrees": condition.rotation_degrees,
                "mse": mse(prediction, target, foreground),
                "mae": mae(prediction, target, foreground),
                "psnr": psnr(prediction, target, foreground),
                "ssim_global": ssim_global(prediction, target, foreground),
                "lighting_path": _relative_path(lighting_path, evaluation_dir),
                "pred_path": _relative_path(pred_path, evaluation_dir),
                "gt_path": _relative_path(gt_path, evaluation_dir),
                "error_path": _relative_path(error_path, evaluation_dir),
                "comparison_path": _relative_path(comparison_path, evaluation_dir),
            }
        )
    metrics_path = evaluation_dir / "metrics.csv"
    _write_metric_rows(metrics_path, rows)
    summary = _evaluation_summary(
        rows,
        split=split,
        target="spherical-Voronoi weighted sum of polarized OLAT targets",
        panels=["lighting", "ground_truth", "prediction", "absolute_error_x4"],
        metrics_path="metrics.csv",
        contact_sheet="contact_sheet.png",
        target_origin="synthesized_from_measured_olat",
    )
    summary["olat_support_split"] = olat_support_split
    (evaluation_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_condition_contact_sheet(
        comparisons,
        evaluation_dir / "contact_sheet.png",
        ["lighting", "ground truth", "prediction", "absolute error x4"],
    )
    print(
        f"[end2end] {split} evaluation: count={len(rows)} "
        f"PSNR={summary['metrics']['psnr']:.3f} "
        f"SSIM-global={summary['metrics']['ssim_global']:.4f}",
        flush=True,
    )
    return summary, [float(row["mse"]) for row in rows]


def _write_combined_evaluation_csv(
    evaluation_dir: Path,
    profile: str,
    olat_summary: dict[str, Any],
    hdri_summary: dict[str, Any],
) -> None:
    rows = []
    for lighting, summary in (("olat", olat_summary), ("hdri", hdri_summary)):
        rows.append(
            {
                "training_profile": profile,
                "evaluation_lighting": lighting,
                "split": summary["split"],
                "count": summary["count"],
                **summary["metrics"],
            }
        )
    _write_metric_rows(evaluation_dir / "metrics.csv", rows)


def _profile_outputs_complete(
    profile_dir: Path,
    acquisition: dict[str, Any],
) -> bool:
    maps_dir = profile_dir / "material" / "maps"
    required = [
        profile_dir / "material" / "disney_brdf.pt",
        profile_dir / "evaluation" / "summary.json",
        profile_dir / "evaluation" / "metrics.csv",
    ]
    required.extend(
        maps_dir / f"{name}.png"
        for name in (
            "albedo",
            "baseColor",
            "normal",
            "specular",
            "roughness",
            "metallic",
            "specularTint",
            "subsurface",
            "anisotropic",
            "clearcoat",
            "clearcoatGloss",
        )
    )
    try:
        evaluations = acquisition["evaluation"]["evaluations"]
        for lighting in ("olat", "hdri"):
            suite_dir = profile_dir / "evaluation" / lighting
            required.extend(
                [
                    suite_dir / "summary.json",
                    suite_dir / "metrics.csv",
                    suite_dir / "contact_sheet.png",
                ]
            )
            representative = evaluations[lighting]["representative"]
            required.extend(
                suite_dir / value
                for key, value in representative.items()
                if key.endswith("_path")
            )
    except (KeyError, TypeError):
        return False
    return all(path.is_file() for path in required)


def _write_camera_report(
    camera_dir: Path,
    profiles: Sequence[str],
    profile_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    report_dir = camera_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "profile / metrics",
        "baseColor",
        "normal",
        "roughness",
        "specular",
        "OLAT ground truth",
        "OLAT prediction",
        "OLAT error x4",
        "HDRI lighting",
        "HDRI ground truth",
        "HDRI prediction",
        "HDRI error x4",
    ]
    label_width = 300
    cell_width, cell_height = 240, 136
    header_height = 70
    row_height = 196
    canvas = Image.new(
        "RGB",
        (
            label_width + (len(columns) - 1) * cell_width,
            header_height + len(profiles) * row_height,
        ),
        color=(12, 12, 12),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 16), columns[0], fill="white")
    for index, label in enumerate(columns[1:]):
        draw.text((label_width + index * cell_width + 6, 16), label, fill="white")
    reference_evaluations = profile_results[profiles[0]]["evaluation"]["evaluations"]
    shared_olat = reference_evaluations["olat"]["representative"]
    shared_hdri = reference_evaluations["hdri"]["representative"]
    shared_frame_id = int(shared_olat["frame_id"])
    shared_condition_id = str(shared_hdri["condition_id"])
    hdri_support_split = reference_evaluations["hdri"].get(
        "olat_support_split",
        "unspecified_olat",
    )
    draw.text(
        (8, 32),
        f"shared OLAT frame {shared_frame_id:06d} / HDRI {shared_condition_id}",
        fill=(190, 190, 190),
    )
    draw.text(
        (8, 49),
        f"HDRI ground truth is synthesized from {hdri_support_split} support",
        fill=(190, 190, 190),
    )
    report_rows = []
    for row_index, profile in enumerate(profiles):
        result = profile_results[profile]
        y = header_height + row_index * row_height
        evaluations = result["evaluation"]["evaluations"]
        olat_summary = evaluations["olat"]
        hdri_summary = evaluations["hdri"]
        fit_conditions = result.get("fit_conditions", {})
        draw.text((8, y + 8), f"{profile.upper()}-trained", fill=(255, 220, 80))
        draw.text(
            (8, y + 30),
            f"OLAT  PSNR {olat_summary['metrics']['psnr']:.2f}  "
            f"SSIM {olat_summary['metrics']['ssim_global']:.3f}",
            fill="white",
        )
        draw.text(
            (8, y + 48),
            f"HDRI  PSNR {hdri_summary['metrics']['psnr']:.2f}  "
            f"SSIM {hdri_summary['metrics']['ssim_global']:.3f}",
            fill="white",
        )
        draw.text(
            (8, y + 68),
            f"fit: {fit_conditions.get('olat', 0)} OLAT / "
            f"{fit_conditions.get('hdri', 0)} HDRI",
            fill=(190, 190, 190),
        )
        profile_dir = camera_dir / profile / PROFILE_MODEL
        maps_dir = profile_dir / "material" / "maps"
        image_specs = [
            (maps_dir / "baseColor.png", 1.0),
            (maps_dir / "normal.png", 1.0),
            (maps_dir / "roughness.png", 1.0),
            (maps_dir / "specular.png", 1.0),
        ]
        olat_root = profile_dir / "evaluation" / "olat"
        hdri_root = profile_dir / "evaluation" / "hdri"
        olat_case = olat_root / "cases" / f"{shared_frame_id:06d}"
        hdri_case = hdri_root / "cases" / shared_condition_id
        image_specs.extend(
            [
                (olat_case / "gt.png", 1.0),
                (olat_case / "pred.png", 1.0),
                (olat_case / "error.png", 4.0),
                (hdri_case / "lighting.png", 1.0),
                (hdri_case / "gt.png", 1.0),
                (hdri_case / "pred.png", 1.0),
                (hdri_case / "error.png", 4.0),
            ]
        )
        for column_index, (path, scale) in enumerate(image_specs):
            image = np.clip(read_image(path) * scale, 0.0, 1.0)
            tile = _pil_fit_image(image, cell_width, cell_height)
            x = label_width + column_index * cell_width
            canvas.paste(tile, (x, y))
        report_rows.append(
            {
                "training_profile": profile,
                "olat": olat_summary["metrics"],
                "hdri": hdri_summary["metrics"],
                "representative_olat": {
                    "frame_id": shared_frame_id,
                    "case": _relative_path(olat_case, camera_dir),
                },
                "representative_hdri": {
                    "condition_id": shared_condition_id,
                    "case": _relative_path(hdri_case, camera_dir),
                },
            }
        )
    overview_path = report_dir / "overview.png"
    canvas.save(overview_path)
    summary = {
        "schema": "ictpolarreal.material-profile-report.v1",
        "profiles": list(profiles),
        "evaluation_matrix": (
            "one model per training profile, evaluated on common OLAT and HDRI suites"
        ),
        "hdri_target_origin": "synthesized_from_measured_olat",
        "representative_policy": (
            f"shared cases selected from {profiles[0]} profile nearest median PSNR"
        ),
        "panels": columns,
        "rows": report_rows,
        "overview": "overview.png",
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metric_rows = []
    for row in report_rows:
        for lighting in ("olat", "hdri"):
            metric_rows.append(
                {
                    "training_profile": row["training_profile"],
                    "evaluation_lighting": lighting,
                    **row[lighting],
                }
            )
    _write_metric_rows(report_dir / "metrics.csv", metric_rows)
    return {
        "overview": "report/overview.png",
        "summary": "report/summary.json",
        "metrics": "report/metrics.csv",
    }


def _evaluation_summary(
    rows: list[dict[str, Any]],
    *,
    split: str,
    target: str,
    panels: list[str],
    metrics_path: str,
    contact_sheet: str,
    target_origin: str,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"cannot summarize empty {split} evaluation")
    metric_names = ("mse", "mae", "psnr", "ssim_global")
    psnr_values = np.asarray([float(row["psnr"]) for row in rows])
    median = float(np.median(psnr_values))
    representative_index = int(np.argmin(np.abs(psnr_values - median)))
    representative_row = rows[representative_index]
    path_keys = [key for key in representative_row if key.endswith("_path")]
    identifier_keys = (
        "condition_id",
        "stack_index",
        "light_index",
        "frame_id",
        "source",
        "rotation_degrees",
    )
    representative = {
        key: representative_row[key]
        for key in [*identifier_keys, *path_keys]
        if key in representative_row
    }
    return {
        "schema": "ictpolarreal.relighting-evaluation.v2",
        "split": split,
        "count": len(rows),
        "normalization": "independent foreground p99.5 linear clipping",
        "target": target,
        "target_origin": target_origin,
        "metrics": {
            name: float(np.mean([float(row[name]) for row in rows]))
            for name in metric_names
        },
        "metrics_csv": metrics_path,
        "contact_sheet": contact_sheet,
        "panels": panels,
        "representative_policy": "condition nearest median PSNR",
        "representative": representative,
    }


def _write_metric_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty metrics table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_condition_contact_sheet(
    comparisons: list[tuple[str, np.ndarray]], path: Path, panel_labels: list[str]
) -> None:
    from PIL import Image, ImageDraw

    thumbnails = []
    for label, comparison in comparisons:
        array = (np.clip(comparison, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        image = Image.fromarray(array)
        target_width = 960
        target_height = max(1, round(image.height * target_width / image.width))
        image = image.resize((target_width, target_height), Image.Resampling.BILINEAR)
        header_height = 40
        tile = Image.new("RGB", (target_width, target_height + header_height), color="black")
        tile.paste(image, (0, header_height))
        draw = ImageDraw.Draw(tile)
        draw.text((6, 3), label, fill="white")
        panel_width = target_width // len(panel_labels)
        for index, panel_label in enumerate(panel_labels):
            draw.text((index * panel_width + 6, 21), panel_label, fill="white")
        thumbnails.append(tile)
    columns = 2
    rows = math.ceil(len(thumbnails) / columns)
    tile_width = max(tile.width for tile in thumbnails)
    tile_height = max(tile.height for tile in thumbnails)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), color="black")
    for index, tile in enumerate(thumbnails):
        sheet.paste(tile, ((index % columns) * tile_width, (index // columns) * tile_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _resize_panel_numpy(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    array = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    resized = Image.fromarray(array).resize(
        (target_hw[1], target_hw[0]), Image.Resampling.BILINEAR
    )
    return np.asarray(resized).astype(np.float32) / 255.0


def _pil_fit_image(image: np.ndarray, width: int, height: int):
    from PIL import Image

    array = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    source = Image.fromarray(array)
    scale = min(width / source.width, height / source.height)
    resized = source.resize(
        (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
        Image.Resampling.BILINEAR,
    )
    tile = Image.new("RGB", (width, height), color="black")
    tile.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return tile


def _relative_path(path: str | Path, root: str | Path) -> str:
    path = Path(path)
    root = Path(root)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _adapter_provenance() -> dict[str, Any]:
    acquisition_path = Path(__file__).resolve()
    lighting_path = acquisition_path.with_name("lighting_profiles.py")
    return {
        "schema": "ictpolarreal.profile-acquisition-adapter.v1",
        "algorithm_version": "disney-profile-matrix-v1",
        "end2end_acquisition_sha256": hashlib.sha256(
            acquisition_path.read_bytes()
        ).hexdigest(),
        "lighting_profiles_sha256": hashlib.sha256(
            lighting_path.read_bytes()
        ).hexdigest(),
        "percentile": PERCENTILE,
        "minimum_learning_rate_ratio": MIN_LR_RATIO,
        "maximum_quantile_elements": MAX_QUANTILE_ELEMENTS,
        "hdri_autograd_bytes_per_light_pixel": HDRI_AUTOGRAD_BYTES_PER_LIGHT_PIXEL,
        "minimum_hdri_gpu_memory_bytes": MIN_HDRI_GPU_MEMORY_BYTES,
        "optimizer": "Adam with cosine decay",
        "hdri_integration": (
            "solid-angle-integrated Voronoi cell radiance with explicit unit "
            "renderer weights"
        ),
    }


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
    scale = _safe_torch_quantile(values, PERCENTILE / 100.0).clamp_min(1e-8)
    return (finite / scale).clamp_max(1.0) * mask_chw


def _safe_torch_quantile(values, quantile: float):
    """Evaluate a scalar quantile below torch's 2^24 CUDA element limit."""
    flat = values.reshape(-1)
    step = max(
        1,
        (flat.numel() + MAX_QUANTILE_ELEMENTS - 1) // MAX_QUANTILE_ELEMENTS,
    )
    return flat[::step].float().quantile(quantile).to(values.dtype)


def _validate_hdri_gpu_memory(
    torch,
    device,
    n_lights: int,
    height: int,
    width: int,
) -> None:
    properties = torch.cuda.get_device_properties(device)
    total_bytes = int(properties.total_memory)
    saved_tensor_bytes = (
        HDRI_AUTOGRAD_BYTES_PER_LIGHT_PIXEL * n_lights * height * width
    )
    required_bytes = max(
        MIN_HDRI_GPU_MEMORY_BYTES,
        math.ceil(saved_tensor_bytes * 1.5),
    )
    gib = float(1 << 30)
    print(
        f"[end2end] HDRI GPU preflight: {properties.name}, "
        f"{total_bytes / gib:.1f} GiB total, "
        f"~{saved_tensor_bytes / gib:.1f} GiB saved tensors per fit render",
        flush=True,
    )
    if total_bytes < required_bytes:
        raise RuntimeError(
            "HDRI/mix material acquisition requires a high-memory GPU for the "
            f"{n_lights}x{height}x{width} Disney autograd graph: found "
            f"{total_bytes / gib:.1f} GiB, require at least "
            f"{required_bytes / gib:.1f} GiB. Request a 40/48 GiB+ Slurm GPU, "
            "reduce --max-lights for a smoke test, or select --end2end-profiles olat."
        )


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
    evaluation_dir = relighting_dir.parent
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
                "pred_path": _relative_path(pred_path, evaluation_dir),
                "gt_path": _relative_path(gt_path, evaluation_dir),
                "error_path": _relative_path(error_path, evaluation_dir),
                "comparison_path": _relative_path(comparison_path, evaluation_dir),
            }
        )

    metrics_path = evaluation_dir / "metrics.csv"
    _write_metric_rows(metrics_path, rows)
    summary = _evaluation_summary(
        rows,
        split=split,
        target="2*cross + 2*max(parallel-cross, 0)",
        panels=["ground_truth", "prediction", "absolute_error_x4"],
        metrics_path="metrics.csv",
        contact_sheet="contact_sheet.png",
        target_origin="measured_polarized_olat",
    )
    summary["frame_ids"] = [row["frame_id"] for row in rows]
    summary["light_indices"] = [row["light_index"] for row in rows]
    summary_path = evaluation_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_relighting_contact_sheet(
        comparisons, evaluation_dir / "contact_sheet.png"
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
    trainer_path = (
        root
        / "imaginaire"
        / "trainers"
        / "portrait_relighting"
        / "relighting_switchlight_pretrain.py"
    )

    def git_output(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    status = git_output("status", "--porcelain", "--untracked-files=no")
    provenance = {
        "root": str(root),
        "commit": git_output("rev-parse", "HEAD"),
        "dirty": bool(status and status != "unknown"),
        "disney_brdf_sha256": digest,
    }
    if trainer_path.is_file():
        provenance["superdimension_profile_reference"] = str(trainer_path)
        provenance["superdimension_profile_reference_sha256"] = hashlib.sha256(
            trainer_path.read_bytes()
        ).hexdigest()
    return provenance


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
