from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import shutil
import subprocess
import sys
import types
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
MIN_N_DOT_V = 1e-4
LSX_VISIBLE_HEMISPHERE_LIGHTS = 173
SUPERDIMENSION_FIT_LIGHTS = 164
MAX_QUANTILE_ELEMENTS = 1 << 23
HDRI_AUTOGRAD_BYTES_PER_LIGHT_PIXEL = 216
MIN_HDRI_GPU_MEMORY_BYTES = 40 * (1 << 30)
MODEL_NAME = "DisneyBRDFSimplifiedMultiLayer"
DISNEY_SCALAR_NAMES = (
    "metallic",
    "subsurface",
    "specular",
    "roughness",
    "specularTint",
    "anisotropic",
    "sheen",
    "sheenTint",
    "clearcoat",
    "clearcoatGloss",
)
REPORT_PROFILES = ("olat", "hdri", "mix")
ERROR_HEATMAP_MAX = 0.25


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(values).cast("B")).hexdigest()


def _disney_scalar_initialization(torch, model) -> dict[str, dict[str, float]]:
    """Record, without rewriting, Imaginaire's unconstrained scalar defaults.

    ``DisneyParamConfig`` values are copied directly into ``*_un`` parameters by
    Imaginaire and constrained with sigmoid only when rendering.  Treating those
    config values as physical values and applying logit a second time pins all
    zero-valued defaults close to zero, which is not the SuperDimension flow.
    """
    initialization = {}
    with torch.no_grad():
        for name in DISNEY_SCALAR_NAMES:
            raw = getattr(model, f"{name}_un").detach().float()
            initialization[name] = {
                "unconstrained": float(raw.mean().cpu()),
                "constrained": float(torch.sigmoid(raw).mean().cpu()),
            }
    return initialization


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


def select_superdimension_light_indices(
    light_ids: np.ndarray,
    eval_lights: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select Imaginaire's 164 OLATs from LSX's visible 173-light hemisphere.

    The nine visible lights omitted by the evenly spaced 164-light fit become a
    natural held-out set.  Rear-hemisphere captures are excluded because their
    independently normalized near-black observations otherwise become
    full-strength noise targets.  Non-LSX or incomplete inputs use the generic
    sphere-spread split.
    """
    ids = np.asarray(light_ids, dtype=np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError("light ids must be a one-dimensional unique array")
    if eval_lights < 0:
        raise ValueError("end2end evaluation light count must be non-negative")
    lookup = {int(light_id): index for index, light_id in enumerate(ids)}
    visible_ids = np.arange(LSX_VISIBLE_HEMISPHERE_LIGHTS, dtype=np.int64)
    if _has_complete_lsx_visible_hemisphere(ids):
        fit_ids = np.linspace(
            0,
            LSX_VISIBLE_HEMISPHERE_LIGHTS - 1,
            SUPERDIMENSION_FIT_LIGHTS,
            dtype=np.int64,
        )
        fit_ids = np.unique(fit_ids)
        heldout_ids = np.setdiff1d(visible_ids, fit_ids)
        if eval_lights < len(heldout_ids):
            if eval_lights == 0:
                heldout_ids = np.zeros((0,), dtype=np.int64)
            else:
                positions = np.linspace(
                    0, len(heldout_ids) - 1, eval_lights, dtype=np.int64
                )
                heldout_ids = heldout_ids[positions]
        train = np.asarray([lookup[int(light_id)] for light_id in fit_ids], dtype=np.int64)
        heldout = np.asarray(
            [lookup[int(light_id)] for light_id in heldout_ids], dtype=np.int64
        )
        used = np.concatenate([train, heldout])
        excluded = np.setdiff1d(np.arange(len(ids), dtype=np.int64), used)
        return train, heldout, excluded

    train, heldout = split_light_indices(len(ids), eval_lights)
    return train, heldout, np.zeros((0,), dtype=np.int64)


def _has_complete_lsx_visible_hemisphere(light_ids: np.ndarray) -> bool:
    available = {int(light_id) for light_id in np.asarray(light_ids).reshape(-1)}
    return all(
        light_id in available for light_id in range(LSX_VISIBLE_HEMISPHERE_LIGHTS)
    )


def _describe_light_selection(
    light_ids: np.ndarray,
    train_indices: np.ndarray,
    heldout_indices: np.ndarray,
    excluded_indices: np.ndarray,
    *,
    requested_heldout_count: int,
) -> dict[str, Any]:
    """Return truthful, machine-readable provenance for either split policy."""
    ids = np.asarray(light_ids, dtype=np.int64)
    description: dict[str, Any] = {
        "requested_heldout_count": int(requested_heldout_count),
        "fit_count": int(len(train_indices)),
        "heldout_count": int(len(heldout_indices)),
        "excluded_count": int(len(excluded_indices)),
    }
    if not _has_complete_lsx_visible_hemisphere(ids):
        return {
            "mode": "generic_sphere_spread",
            "policy": (
                "generic deterministic sphere-spread holdout over all selected "
                "calibrated lights"
            ),
            **description,
        }

    visible = (ids >= 0) & (ids < LSX_VISIBLE_HEMISPHERE_LIGHTS)
    heldout_visible_count = int(np.count_nonzero(visible[heldout_indices]))
    excluded_visible_count = int(np.count_nonzero(visible[excluded_indices]))
    return {
        "mode": "lsx_visible_hemisphere_164",
        "policy": (
            "fit 164 evenly spaced LSX visible indices 0..172; select requested "
            "holdouts from the nine omitted visible indices; exclude unselected "
            "omitted-visible and non-visible inputs"
        ),
        **description,
        "heldout_visible_count": heldout_visible_count,
        "excluded_visible_count": excluded_visible_count,
        "excluded_nonvisible_count": int(len(excluded_indices))
        - excluded_visible_count,
    }


def _select_evaluation_light_indices(
    train_indices: np.ndarray,
    heldout_indices: np.ndarray,
    max_cases: int = 16,
) -> np.ndarray:
    if len(heldout_indices):
        return heldout_indices
    positions = np.linspace(
        0,
        len(train_indices) - 1,
        min(len(train_indices), max_cases),
        dtype=np.int64,
    )
    return train_indices[positions]


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
    train_indices, heldout_indices, excluded_indices = (
        select_superdimension_light_indices(light_ids, eval_lights)
    )
    light_selection = _describe_light_selection(
        light_ids,
        train_indices,
        heldout_indices,
        excluded_indices,
        requested_heldout_count=eval_lights,
    )
    if any(profile in {"hdri", "mix"} for profile in profiles):
        _validate_hdri_gpu_memory(
            torch,
            torch_device,
            len(train_indices),
            height,
            width,
        )
    evaluation_indices = _select_evaluation_light_indices(
        train_indices, heldout_indices
    )
    evaluation_split = "heldout_olat" if len(heldout_indices) else "fitted_olat"
    if light_selection["mode"] == "lsx_visible_hemisphere_164":
        print(
            "[end2end] light split (LSX visible-hemisphere adaptation): "
            f"{light_selection['fit_count']} fit, "
            f"{light_selection['heldout_visible_count']} visible held out, "
            f"{light_selection['excluded_visible_count']} omitted visible excluded, "
            f"{light_selection['excluded_nonvisible_count']} non-visible excluded",
            flush=True,
        )
    else:
        print(
            "[end2end] light split (generic sphere-spread): "
            f"{light_selection['fit_count']} fit, "
            f"{light_selection['heldout_count']} held out, "
            f"{light_selection['excluded_count']} excluded",
            flush=True,
        )
    capture_foreground = _foreground_mask(mask, height, width)
    if not np.any(capture_foreground > 0.5):
        raise ValueError("end2end acquisition requires a non-empty foreground mask")
    # SuperDimension optimizes the measured OLAT photograph directly.  For the
    # polarized ICT capture, the parallel branch is the corresponding full
    # reflection observation (diffuse + specular).  Do not collapse the two
    # polarization branches into a synthetic target: that changes the fitting
    # objective and amplifies polarization noise where cross > parallel.
    normal = _normalize_vectors(normal)
    view_dirs = _normalize_vectors(view_dirs)
    n_dot_v = np.sum(normal * view_dirs, axis=-1, keepdims=True)
    front_facing = (n_dot_v > MIN_N_DOT_V).astype(np.float32)
    foreground = capture_foreground * front_facing
    capture_pixels = int(np.count_nonzero(capture_foreground > 0.5))
    valid_pixels = int(np.count_nonzero(foreground > 0.5))
    if valid_pixels == 0:
        raise ValueError(
            "end2end acquisition has no front-facing foreground pixels; check "
            "the normal/view coordinate convention"
        )
    surface_validity = {
        "minimum_n_dot_v": MIN_N_DOT_V,
        "capture_foreground_pixels": capture_pixels,
        "front_facing_pixels": valid_pixels,
        "excluded_back_facing_pixels": capture_pixels - valid_pixels,
        "excluded_fraction": float((capture_pixels - valid_pixels) / capture_pixels),
    }
    print(
        "[end2end] surface validity: "
        f"{valid_pixels}/{capture_pixels} foreground pixels are front-facing "
        f"({100.0 * surface_validity['excluded_fraction']:.2f}% excluded)",
        flush=True,
    )
    raw_target_stack = np.maximum(parallel, 0.0).astype(np.float32)
    target_stack = _normalize_targets(raw_target_stack, foreground)
    base_color = _normalize_base_color(base_color, capture_foreground)
    input_hashes = {
        "normalized_targets_sha256": _array_sha256(target_stack),
        "capture_foreground_sha256": _array_sha256(capture_foreground),
        "foreground_sha256": _array_sha256(foreground),
        "base_color_sha256": _array_sha256(base_color),
        "normal_sha256": _array_sha256(normal),
        "view_directions_sha256": _array_sha256(view_dirs),
        "raw_parallel_targets_sha256": _array_sha256(raw_target_stack),
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
    material_root = camera_dir / "material"
    evaluation_root = camera_dir / "evaluation"
    lighting_dir = evaluation_root / "assets"
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
        material_dir = material_root / profile
        # The renderer writes one self-contained profile at a time.  Keep those
        # destructive writers in a private staging area, then consolidate them
        # into the public lighting-first tree in one atomic report transaction.
        evaluation_dir = evaluation_root / ".profiles" / profile
        print(
            f"[end2end] fitting lighting profile {profile} -> {material_dir}",
            flush=True,
        )
        profile_results[profile] = _fit_disney_profile(
            torch=torch,
            disney_module=disney_module,
            profile=profile,
            material_dir=material_dir,
            evaluation_dir=evaluation_dir,
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
            excluded_indices=excluded_indices,
            evaluation_indices=evaluation_indices,
            evaluation_split=evaluation_split,
            evaluation_support=evaluation_support,
            base_color=base_color,
            normal=normal,
            foreground=foreground,
            material_foreground=capture_foreground,
            view_dirs=view_dirs,
            device=torch_device,
            steps=steps,
            learning_rate=learning_rate,
            hdri_rotations=hdri_rotations,
            eval_lights=eval_lights,
            input_hashes=input_hashes,
            surface_validity=surface_validity,
            provenance=provenance,
            adapter_provenance=adapter_provenance,
        )

    manifest = {
        "schema": "ictpolarreal.material-profiles.v3",
        "material_acquisition": "end2end",
        "model": MODEL_NAME,
        "profiles": list(profiles),
        "primary_profile": primary_profile,
        "primary_material_dir": f"material/{primary_profile}/maps",
        "olat_selection": light_selection,
        "lighting": {
            "conditions": "evaluation/assets/conditions.json",
            "weights": "evaluation/assets/weights.npz",
            "fit_hdri_conditions": len(environments.train),
            "fit_natural_hdri_identities": int(hdri_count),
            "fit_calibration_conditions": int(4 * hdri_rotations),
            "evaluation_hdri_conditions": len(evaluation_environments),
            "evaluation_natural_hdri_identities": int(eval_hdris),
            "target_origin": "synthesized_from_measured_olat",
        },
        "evaluation_matrix": "each profile evaluated on the same OLAT and HDRI suites",
        "surface_validity": surface_validity,
        "material": {"overview": "material/overview.png"},
        "evaluation": {"status": "pending"},
        "adapter": adapter_provenance,
        "profile_acquisitions": {
            profile: f"material/{profile}/acquisition.json" for profile in profiles
        },
    }
    manifest_path = camera_dir / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    try:
        report = _write_camera_report(camera_dir, profiles, profile_results)
    except Exception as exc:
        manifest["evaluation"] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json_atomic(manifest_path, manifest)
        raise
    manifest["material"] = {"overview": report.pop("material_overview")}
    manifest["evaluation"] = {"status": "complete", **report}
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
    material_dir: Path,
    evaluation_dir: Path,
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
    excluded_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    evaluation_split: str,
    evaluation_support: np.ndarray,
    base_color: np.ndarray,
    normal: np.ndarray,
    foreground: np.ndarray,
    material_foreground: np.ndarray,
    view_dirs: np.ndarray,
    device,
    steps: int,
    learning_rate: float,
    hdri_rotations: int,
    eval_lights: int,
    input_hashes: dict[str, str],
    surface_validity: dict[str, Any],
    provenance: dict[str, Any],
    adapter_provenance: dict[str, Any],
) -> dict[str, Any]:
    height, width = foreground.shape[:2]
    model_class = disney_module.DisneyBRDFSimplifiedMultiLayer
    param_config = disney_module.DisneyParamConfig(per_pixel=True, height_mode="none")
    model = model_class(height, width, device=device, cfg=param_config).to(device)
    scalar_initialization = _disney_scalar_initialization(torch, model)
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
        )
        return prediction

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
        )
        return prediction

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
    checkpoint_dir = material_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "latest.pt"
    checkpoint_temp_path = checkpoint_dir / "latest.tmp"
    signature = {
        "schema": "ictpolarreal.end2end-checkpoint.v7",
        "profile": profile,
        "model": MODEL_NAME,
        "height": height,
        "width": width,
        "light_directions_sha256": _array_sha256(light_dirs),
        "light_ids_sha256": _array_sha256(light_ids),
        "frame_ids_sha256": _array_sha256(frame_ids),
        **input_hashes,
        "fit_indices": [int(index) for index in train_indices],
        "heldout_indices": [int(index) for index in heldout_indices],
        "excluded_indices": [int(index) for index in excluded_indices],
        "hdri_condition_weights_sha256": environment_hash,
        "hdri_fit_condition_ids": [condition.condition_id for condition in hdri_fit_conditions],
        "hdri_evaluation_condition_ids": [
            condition.condition_id for condition in hdri_evaluation_conditions
        ],
        "mix_schedule": f"{hdri_rotations}_hdri_then_{hdri_rotations}_olat",
        "scalar_initialization": scalar_initialization,
        "surface_validity": surface_validity,
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "disney_brdf_sha256": provenance["disney_brdf_sha256"],
        "adapter": adapter_provenance,
    }
    acquisition_path = material_dir / "acquisition.json"
    if acquisition_path.is_file():
        try:
            completed = json.loads(acquisition_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            completed = None
        if (
            completed is not None
            and _checkpoint_signatures_match(
                completed.get("checkpoint_signature"), signature
            )
            and _profile_outputs_complete(material_dir, evaluation_dir, completed)
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
        if not _checkpoint_signatures_match(checkpoint.get("signature"), signature):
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

    maps_dir = material_dir / "maps"
    _write_material_maps(maps_dir, maps, material_foreground)
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
        "schema": "ictpolarreal.end2end-disney.v7",
        "material_acquisition": "end2end",
        "lighting_profile": profile,
        "model": MODEL_NAME,
        "target": {
            "olat": "measured parallel-polarized OLAT (diffuse + specular)",
            "hdri": "spherical-Voronoi weighted sum of measured parallel OLAT targets",
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
            "selection": _describe_light_selection(
                light_ids,
                train_indices,
                heldout_indices,
                excluded_indices,
                requested_heldout_count=eval_lights,
            ),
            "requested_heldout_lights": int(eval_lights),
            "fit_stack_indices": [int(index) for index in train_indices],
            "fit_light_indices": [int(light_ids[index]) for index in train_indices],
            "fit_frame_ids": [int(frame_ids[index]) for index in train_indices],
            "heldout_stack_indices": [int(index) for index in heldout_indices],
            "heldout_light_indices": [int(light_ids[index]) for index in heldout_indices],
            "heldout_frame_ids": [int(frame_ids[index]) for index in heldout_indices],
            "excluded_stack_indices": [int(index) for index in excluded_indices],
            "excluded_light_indices": [int(light_ids[index]) for index in excluded_indices],
            "excluded_frame_ids": [int(frame_ids[index]) for index in excluded_indices],
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
            "evaluation": "evaluation",
            "evaluation_suites": {
                "olat": "evaluation/olat",
                "hdri": "evaluation/hdri",
            },
        },
        "input_hashes": input_hashes,
        "hdri_condition_weights_sha256": environment_hash,
        "scalar_initialization": scalar_initialization,
        "surface_validity": surface_validity,
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
                **_appearance_metrics(prediction, target, foreground),
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
        target="spherical-Voronoi weighted sum of measured parallel OLAT targets",
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
        f"SSIM-global={summary['metrics']['ssim_global']:.4f} "
        f"intensity-ratio={summary['metrics']['mean_intensity_ratio']:.3f} "
        f"corr={summary['metrics']['luminance_correlation']:.3f}",
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
    material_dir: Path,
    evaluation_dir: Path,
    acquisition: dict[str, Any],
) -> bool:
    maps_dir = material_dir / "maps"
    required = [material_dir / "disney_brdf.pt"]
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
    if not all(path.is_file() for path in required):
        return False

    staged_required = [evaluation_dir / "summary.json", evaluation_dir / "metrics.csv"]
    try:
        evaluations = acquisition["evaluation"]["evaluations"]
        for lighting in ("olat", "hdri"):
            suite_dir = evaluation_dir / lighting
            staged_required.extend(
                [
                    suite_dir / "summary.json",
                    suite_dir / "metrics.csv",
                    suite_dir / "contact_sheet.png",
                ]
            )
            representative = evaluations[lighting]["representative"]
            staged_required.extend(
                suite_dir / value
                for key, value in representative.items()
                if key.endswith("_path")
            )
    except (KeyError, TypeError):
        return False
    if all(path.is_file() for path in staged_required):
        return True

    evaluation_root = (
        evaluation_dir.parent.parent
        if evaluation_dir.parent.name == ".profiles"
        else evaluation_dir.parent
    )
    profile = str(acquisition.get("lighting_profile", evaluation_dir.name))
    public_required = [
        evaluation_root / "overview.png",
        evaluation_root / "summary.json",
        evaluation_root / "metrics.csv",
    ]
    for lighting in ("olat", "hdri"):
        suite_dir = evaluation_root / lighting
        public_required.append(suite_dir / "comparison.png")
        expected = int(evaluations[lighting].get("count", 0))
        predictions = list(
            (suite_dir / "cases").glob(f"*/predictions/{profile}.png")
        )
        errors = list((suite_dir / "cases").glob(f"*/errors/{profile}.png"))
        if expected <= 0 or len(predictions) != expected or len(errors) != expected:
            return False
    return all(path.is_file() for path in public_required)


def _write_camera_report(
    camera_dir: Path,
    profiles: Sequence[str],
    profile_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compose the public lighting-first report from private profile outputs.

    The whole evaluation tree is built as a sibling and swapped only after
    validation.  This keeps a failed presentation pass from damaging a
    completed material fit and also makes v2 report migration GPU-free.
    """
    camera_dir = Path(camera_dir)
    profiles = tuple(profiles)
    if not profiles or len(set(profiles)) != len(profiles):
        raise ValueError("report profiles must be a non-empty unique sequence")
    missing = [profile for profile in profiles if profile not in profile_results]
    if missing:
        raise KeyError(f"missing profile results for: {', '.join(missing)}")

    evaluation_root = camera_dir / "evaluation"
    artifacts = _camera_report_artifacts()
    sources = {
        profile: _profile_evaluation_source(evaluation_root, profile)
        for profile in profiles
    }
    if all(source is None for source in sources.values()):
        if not _clean_camera_report_complete(camera_dir, profiles, profile_results):
            raise FileNotFoundError(
                "No private or legacy profile evaluations were found, and the clean "
                "lighting-first report is incomplete."
            )

    stage = camera_dir / ".evaluation.clean.tmp"
    backup = camera_dir / ".evaluation.previous.tmp"
    material_tmp = camera_dir / ".material-overview.tmp.png"
    for stale in (stage, backup):
        if stale.exists():
            shutil.rmtree(stale)
    material_tmp.unlink(missing_ok=True)
    stage.mkdir(parents=True)
    try:
        asset_source = (
            evaluation_root / "assets"
            if (evaluation_root / "assets").is_dir()
            else evaluation_root / "lighting"
        )
        if not asset_source.is_dir():
            raise FileNotFoundError(
                f"missing evaluation lighting assets under {evaluation_root}"
            )
        shutil.copytree(asset_source, stage / "assets")

        suite_reports = {
            lighting: _consolidate_evaluation_suite(
                evaluation_root,
                stage,
                lighting,
                profiles,
                sources,
                profile_results,
            )
            for lighting in ("olat", "hdri")
        }
        report_rows = []
        for profile in profiles:
            row = {"training_profile": profile}
            for lighting in ("olat", "hdri"):
                metrics = profile_results[profile]["evaluation"]["evaluations"][
                    lighting
                ]["metrics"]
                row[lighting] = {
                    name: metrics[name]
                    for name in (
                        "psnr",
                        "ssim_global",
                        "mean_intensity_ratio",
                        "luminance_correlation",
                    )
                    if name in metrics
                }
            report_rows.append(row)
        metric_rows = [
            {
                "training_profile": row["training_profile"],
                "evaluation_lighting": lighting,
                "split": profile_results[row["training_profile"]]["evaluation"][
                    "evaluations"
                ][lighting].get("split", "unspecified"),
                "count": profile_results[row["training_profile"]]["evaluation"][
                    "evaluations"
                ][lighting].get("count", 0),
                **profile_results[row["training_profile"]]["evaluation"][
                    "evaluations"
                ][lighting]["metrics"],
            }
            for row in report_rows
            for lighting in ("olat", "hdri")
        ]
        _write_metric_rows(stage / "metrics.csv", metric_rows)
        summary = {
            "schema": "ictpolarreal.material-profile-report.v2",
            "profiles": list(profiles),
            "evaluation_matrix": (
                "rows are training-light profiles; columns are common OLAT and "
                "HDRI test-light suites"
            ),
            "hdri_target_origin": "synthesized_from_measured_olat",
            "representative_policy": (
                f"shared cases selected from {profiles[0]}-trained predictions "
                "nearest median PSNR"
            ),
            "rows": report_rows,
            "overview": "overview.png",
            "metrics_csv": "metrics.csv",
            "suites": {
                lighting: {
                    "comparison": f"{lighting}/comparison.png",
                    "representative_case": suite_reports[lighting][
                        "representative_case"
                    ],
                }
                for lighting in ("olat", "hdri")
            },
        }
        _write_json_atomic(stage / "summary.json", summary)
        _write_evaluation_overview(
            stage,
            profiles,
            profile_results,
            suite_reports["olat"]["representative_case"],
            suite_reports["hdri"]["representative_case"],
        )
        _write_material_overview(camera_dir, profiles, material_tmp)
        _validate_clean_report_tree(stage, profiles, suite_reports)

        if evaluation_root.exists():
            evaluation_root.replace(backup)
        try:
            stage.replace(evaluation_root)
        except Exception:
            if backup.exists() and not evaluation_root.exists():
                backup.replace(evaluation_root)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        material_overview = camera_dir / "material" / "overview.png"
        material_overview.parent.mkdir(parents=True, exist_ok=True)
        material_tmp.replace(material_overview)
        _update_profile_acquisition_reports(
            camera_dir, profiles, profile_results, suite_reports
        )
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        material_tmp.unlink(missing_ok=True)
        raise
    return artifacts


def reorganize_end2end_camera(camera_dir: str | Path) -> dict[str, Any]:
    """Migrate a completed v2 camera report without importing PyTorch."""
    camera_dir = Path(camera_dir)
    manifest_path = camera_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profiles = tuple(manifest.get("profiles", REPORT_PROFILES))
    profile_results = {
        profile: json.loads(
            (camera_dir / "material" / profile / "acquisition.json").read_text(
                encoding="utf-8"
            )
        )
        for profile in profiles
    }
    report = _write_camera_report(camera_dir, profiles, profile_results)
    manifest["schema"] = "ictpolarreal.material-profiles.v3"
    manifest["lighting"]["conditions"] = "evaluation/assets/conditions.json"
    manifest["lighting"]["weights"] = "evaluation/assets/weights.npz"
    manifest["material"] = {"overview": report.pop("material_overview")}
    manifest["evaluation"] = {"status": "complete", **report}
    manifest["adapter"] = _adapter_provenance()
    manifest.pop("report", None)
    manifest.pop("profile_evaluations", None)
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _camera_report_artifacts() -> dict[str, Any]:
    return {
        "material_overview": "material/overview.png",
        "overview": "evaluation/overview.png",
        "summary": "evaluation/summary.json",
        "metrics": "evaluation/metrics.csv",
        "suites": {
            lighting: {
                "comparison": f"evaluation/{lighting}/comparison.png",
            }
            for lighting in ("olat", "hdri")
        },
        "assets": {
            "conditions": "evaluation/assets/conditions.json",
            "weights": "evaluation/assets/weights.npz",
        },
    }


def _profile_evaluation_source(evaluation_root: Path, profile: str) -> Path | None:
    staged = evaluation_root / ".profiles" / profile
    if (staged / "olat" / "cases").is_dir() and (staged / "hdri" / "cases").is_dir():
        return staged
    legacy = evaluation_root / profile
    if (legacy / "olat" / "cases").is_dir() and (legacy / "hdri" / "cases").is_dir():
        return legacy
    return None


def _clean_camera_report_complete(
    camera_dir: Path,
    profiles: Sequence[str],
    profile_results: dict[str, dict[str, Any]],
) -> bool:
    artifacts = _camera_report_artifacts()
    required = [
        camera_dir / artifacts["material_overview"],
        camera_dir / artifacts["overview"],
        camera_dir / artifacts["summary"],
        camera_dir / artifacts["metrics"],
        camera_dir / artifacts["assets"]["conditions"],
        camera_dir / artifacts["assets"]["weights"],
    ]
    for lighting in ("olat", "hdri"):
        required.extend(
            camera_dir / value
            for value in artifacts["suites"][lighting].values()
        )
        cases_dir = camera_dir / "evaluation" / lighting / "cases"
        for profile in profiles:
            expected = int(
                profile_results[profile]["evaluation"]["evaluations"][lighting][
                    "count"
                ]
            )
            if len(list(cases_dir.glob(f"*/predictions/{profile}.png"))) != expected:
                return False
            if len(list(cases_dir.glob(f"*/errors/{profile}.png"))) != expected:
                return False
    return all(path.is_file() for path in required)


def _consolidate_evaluation_suite(
    evaluation_root: Path,
    stage: Path,
    lighting: str,
    profiles: Sequence[str],
    sources: dict[str, Path | None],
    profile_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    suite_dir = stage / lighting
    cases_dir = suite_dir / "cases"
    cases_dir.mkdir(parents=True)
    case_ids_by_profile: dict[str, list[str]] = {}
    for profile in profiles:
        source = sources[profile]
        if source is not None:
            rows = _read_csv_rows(source / lighting / "metrics.csv")
        else:
            rows = [
                row
                for row in _read_csv_rows(evaluation_root / "metrics.csv")
                if row.get("training_profile") == profile
                and row.get("evaluation_lighting") == lighting
            ]
            # A clean tree has aggregate metrics only.  Its case identifiers
            # come directly from the canonical case folders.
            if rows:
                case_ids_by_profile[profile] = sorted(
                    path.name for path in (evaluation_root / lighting / "cases").iterdir()
                    if path.is_dir()
                )
                continue
        if not rows:
            raise ValueError(f"no {lighting} metric rows for profile {profile}")
        case_ids_by_profile[profile] = [
            _metric_case_id(lighting, row) for row in rows
        ]

    reference_ids = case_ids_by_profile[profiles[0]]
    if not reference_ids or len(reference_ids) != len(set(reference_ids)):
        raise ValueError(f"invalid or duplicate {lighting} case identifiers")
    for profile in profiles[1:]:
        if set(case_ids_by_profile[profile]) != set(reference_ids):
            raise ValueError(
                f"{lighting} cases differ between {profiles[0]} and {profile}"
            )

    for case_id in reference_ids:
        reference_sources = []
        lighting_sources = []
        predictions = {}
        for profile in profiles:
            source = sources[profile]
            if source is not None:
                source_case = source / lighting / "cases" / case_id
                reference_sources.append(source_case / "gt.png")
                predictions[profile] = source_case / "pred.png"
                if lighting == "hdri":
                    lighting_sources.append(source_case / "lighting.png")
            else:
                source_case = evaluation_root / lighting / "cases" / case_id
                reference_sources.append(source_case / "reference.png")
                predictions[profile] = source_case / "predictions" / f"{profile}.png"
                if lighting == "hdri":
                    lighting_sources.append(source_case / "lighting.png")
        _assert_identical_files(reference_sources, f"{lighting}/{case_id} reference")
        if lighting_sources:
            _assert_identical_files(lighting_sources, f"{lighting}/{case_id} lighting")

        output_case = cases_dir / case_id
        output_case.mkdir(parents=True)
        shutil.copy2(reference_sources[0], output_case / "reference.png")
        if lighting_sources:
            shutil.copy2(lighting_sources[0], output_case / "lighting.png")
        for profile, prediction in predictions.items():
            prediction_path = output_case / "predictions" / f"{profile}.png"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(prediction, prediction_path)
            _write_scalar_error_heatmap(
                prediction_path,
                output_case / "reference.png",
                output_case / "errors" / f"{profile}.png",
            )
        _write_case_comparison(output_case, lighting, profiles, case_id)

    first_summary = profile_results[profiles[0]]["evaluation"]["evaluations"][
        lighting
    ]
    representative = first_summary.get("representative", {})
    representative_case = (
        f"{int(representative['frame_id']):06d}"
        if lighting == "olat" and "frame_id" in representative
        else str(representative.get("condition_id", reference_ids[0]))
    )
    if representative_case not in reference_ids:
        representative_case = reference_ids[len(reference_ids) // 2]
    _write_suite_comparison(
        [cases_dir / case_id / "comparison.png" for case_id in reference_ids],
        suite_dir / "comparison.png",
        f"{lighting.upper()} relighting · shared references · models side by side",
    )
    return {"case_ids": reference_ids, "representative_case": representative_case}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _metric_case_id(lighting: str, row: dict[str, Any]) -> str:
    if lighting == "olat":
        return f"{int(row['frame_id']):06d}"
    return str(row["condition_id"])


def _assert_identical_files(paths: Sequence[Path], label: str) -> None:
    if not paths or any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"missing {label} inputs: {missing}")
    digests = {hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if len(digests) != 1:
        raise ValueError(
            f"cannot deduplicate {label}: profile copies are not byte-identical"
        )


def _write_scalar_error_heatmap(
    prediction_path: Path, reference_path: Path, output_path: Path
) -> None:
    prediction = np.clip(read_image(prediction_path), 0.0, 1.0)
    reference = np.clip(read_image(reference_path), 0.0, 1.0)
    if prediction.shape != reference.shape:
        raise ValueError(
            f"error inputs differ in shape: {prediction.shape} vs {reference.shape}"
        )
    error = np.mean(np.abs(prediction[..., :3] - reference[..., :3]), axis=-1)
    normalized = np.clip(error / ERROR_HEATMAP_MAX, 0.0, 1.0)
    positions = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    colors = np.asarray(
        [
            [0, 0, 0],
            [15, 32, 110],
            [0, 180, 220],
            [255, 220, 35],
            [220, 25, 25],
        ],
        dtype=np.float32,
    ) / 255.0
    heatmap = np.stack(
        [np.interp(normalized, positions, colors[:, channel]) for channel in range(3)],
        axis=-1,
    ).astype(np.float32)
    write_image(output_path, heatmap)


def _report_font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    names = (
        (
            "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        )
        if bold
        else (
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _fit_pil_path(path: Path, width: int, height: int):
    from PIL import Image

    with Image.open(path) as source_file:
        source = source_file.convert("RGB")
        scale = min(width / source.width, height / source.height)
        resized = source.resize(
            (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
            Image.Resampling.LANCZOS,
        )
    tile = Image.new("RGB", (width, height), (12, 12, 14))
    tile.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return tile


def _centered_text(draw, bounds: tuple[int, int, int, int], text: str, font, fill) -> None:
    left, top, right, bottom = bounds
    box = draw.textbbox((0, 0), text, font=font)
    width, height = box[2] - box[0], box[3] - box[1]
    draw.text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        text,
        font=font,
        fill=fill,
    )


def _write_case_comparison(
    case_dir: Path, lighting: str, profiles: Sequence[str], case_id: str
) -> None:
    from PIL import Image, ImageDraw

    panel_width, panel_height = 260, 390
    gap, margin = 16, 36
    header_height, label_height = 90, 44
    leading = []
    if lighting == "hdri":
        leading.append(("Lighting", case_dir / "lighting.png"))
    leading.append(("Reference", case_dir / "reference.png"))
    columns = [
        *leading,
        *[
            (f"{profile.upper()}-trained", case_dir / "predictions" / f"{profile}.png")
            for profile in profiles
        ],
    ]
    width = 2 * margin + len(columns) * panel_width + (len(columns) - 1) * gap
    height = header_height + 2 * (label_height + panel_height) + 42 + margin
    canvas = Image.new("RGB", (width, height), (10, 10, 12))
    draw = ImageDraw.Draw(canvas)
    title_font = _report_font(30, bold=True)
    label_font = _report_font(22, bold=True)
    note_font = _report_font(19)
    draw.text((margin, 18), f"{lighting.upper()} · {case_id}", font=title_font, fill="white")
    draw.text(
        (margin, 56),
        "Top: relighting result  ·  Bottom: fixed-scale mean absolute RGB error",
        font=note_font,
        fill=(176, 181, 190),
    )
    top_y = header_height
    for index, (label, path) in enumerate(columns):
        x = margin + index * (panel_width + gap)
        _centered_text(
            draw,
            (x, top_y, x + panel_width, top_y + label_height),
            label,
            label_font,
            (235, 238, 242),
        )
        canvas.paste(
            _fit_pil_path(path, panel_width, panel_height),
            (x, top_y + label_height),
        )
    error_y = top_y + label_height + panel_height + 42
    model_offset = len(leading)
    draw.multiline_text(
        (margin, error_y + label_height + panel_height // 2 - 22),
        f"Error scale\n0 to {ERROR_HEATMAP_MAX:g}",
        font=note_font,
        fill=(176, 181, 190),
        spacing=8,
    )
    for profile_index, profile in enumerate(profiles):
        index = model_offset + profile_index
        x = margin + index * (panel_width + gap)
        _centered_text(
            draw,
            (x, error_y, x + panel_width, error_y + label_height),
            f"{profile.upper()} error",
            label_font,
            (235, 238, 242),
        )
        canvas.paste(
            _fit_pil_path(
                case_dir / "errors" / f"{profile}.png", panel_width, panel_height
            ),
            (x, error_y + label_height),
        )
    canvas.save(case_dir / "comparison.png")


def _write_suite_comparison(case_paths: Sequence[Path], output: Path, title: str) -> None:
    from PIL import Image, ImageDraw

    thumbnails = []
    for path in case_paths:
        with Image.open(path) as source_file:
            source = source_file.convert("RGB")
            target_width = 860
            target_height = max(1, round(source.height * target_width / source.width))
            thumbnails.append(
                source.resize((target_width, target_height), Image.Resampling.LANCZOS)
            )
    columns = 2 if len(thumbnails) > 1 else 1
    rows = math.ceil(len(thumbnails) / columns)
    gap, margin, header = 24, 40, 90
    tile_width = max(tile.width for tile in thumbnails)
    tile_height = max(tile.height for tile in thumbnails)
    canvas = Image.new(
        "RGB",
        (
            2 * margin + columns * tile_width + (columns - 1) * gap,
            header + margin + rows * tile_height + (rows - 1) * gap,
        ),
        (8, 8, 10),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 24), title, font=_report_font(30, bold=True), fill="white")
    for index, tile in enumerate(thumbnails):
        x = margin + (index % columns) * (tile_width + gap)
        y = header + (index // columns) * (tile_height + gap)
        canvas.paste(tile, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _write_material_overview(
    camera_dir: Path, profiles: Sequence[str], output: Path
) -> None:
    from PIL import Image, ImageDraw

    map_names = ("baseColor", "normal", "roughness", "specular")
    panel_width, panel_height = 270, 440
    label_width, margin, gap = 190, 44, 18
    header = 108
    width = 2 * margin + label_width + len(map_names) * panel_width + 3 * gap
    height = header + len(profiles) * panel_height + (len(profiles) - 1) * gap + margin
    canvas = Image.new("RGB", (width, height), (10, 10, 12))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 17),
        "Disney BRDF material maps",
        font=_report_font(34, bold=True),
        fill="white",
    )
    draw.text(
        (margin, 60),
        "Rows: acquisition lighting profile  ·  Columns: fitted material parameter",
        font=_report_font(20),
        fill=(176, 181, 190),
    )
    for index, name in enumerate(map_names):
        x = margin + label_width + index * (panel_width + gap)
        _centered_text(
            draw,
            (x, 62, x + panel_width, header),
            name,
            _report_font(22, bold=True),
            (235, 238, 242),
        )
    for row, profile in enumerate(profiles):
        y = header + row * (panel_height + gap)
        _centered_text(
            draw,
            (margin, y, margin + label_width - gap, y + panel_height),
            f"{profile.upper()}-trained",
            _report_font(24, bold=True),
            (255, 216, 90),
        )
        maps_dir = camera_dir / "material" / profile / "maps"
        for column, name in enumerate(map_names):
            x = margin + label_width + column * (panel_width + gap)
            canvas.paste(
                _fit_pil_path(
                    maps_dir / f"{name}.png", panel_width, panel_height
                ),
                (x, y),
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _write_evaluation_overview(
    stage: Path,
    profiles: Sequence[str],
    profile_results: dict[str, dict[str, Any]],
    olat_case_id: str,
    hdri_case_id: str,
) -> None:
    from PIL import Image, ImageDraw

    width, margin = 1800, 58
    matrix_top, matrix_header, metric_height = 125, 64, 138
    matrix_left, metric_width = 300, 690
    hero_panel_width, hero_panel_height, hero_label = 290, 410, 44
    olat_top = matrix_top + matrix_header + len(profiles) * metric_height + 80
    hdri_top = olat_top + 70 + hero_label + hero_panel_height + 70
    height = hdri_top + 70 + hero_label + hero_panel_height + 70
    canvas = Image.new("RGB", (width, height), (9, 9, 11))
    draw = ImageDraw.Draw(canvas)
    title_font = _report_font(40, bold=True)
    section_font = _report_font(28, bold=True)
    label_font = _report_font(22, bold=True)
    metric_font = _report_font(28, bold=True)
    note_font = _report_font(19)
    draw.text(
        (margin, 25),
        "Material acquisition · relighting evaluation",
        font=title_font,
        fill="white",
    )
    draw.text(
        (margin, 78),
        "Rows are training-light profiles; columns are test-light suites. Higher PSNR / SSIM is better.",
        font=note_font,
        fill=(176, 181, 190),
    )
    draw.text(
        (margin, matrix_top + 17),
        "TRAINED ON",
        font=label_font,
        fill=(160, 166, 176),
    )
    for column, lighting in enumerate(("OLAT test", "HDRI test")):
        x = matrix_left + column * metric_width
        _centered_text(
            draw,
            (x, matrix_top, x + metric_width, matrix_top + matrix_header),
            lighting,
            label_font,
            (235, 238, 242),
        )
    best = {
        lighting: max(
            float(
                profile_results[profile]["evaluation"]["evaluations"][lighting][
                    "metrics"
                ]["psnr"]
            )
            for profile in profiles
        )
        for lighting in ("olat", "hdri")
    }
    for row, profile in enumerate(profiles):
        y = matrix_top + matrix_header + row * metric_height
        _centered_text(
            draw,
            (margin, y, matrix_left - 16, y + metric_height),
            f"{profile.upper()}-trained",
            label_font,
            (255, 216, 90),
        )
        for column, lighting in enumerate(("olat", "hdri")):
            metrics = profile_results[profile]["evaluation"]["evaluations"][lighting][
                "metrics"
            ]
            x = matrix_left + column * metric_width
            is_best = float(metrics["psnr"]) == best[lighting]
            fill = (18, 52, 55) if is_best else (22, 22, 26)
            outline = (70, 215, 190) if is_best else (56, 56, 64)
            draw.rounded_rectangle(
                (x + 8, y + 8, x + metric_width - 8, y + metric_height - 8),
                radius=14,
                fill=fill,
                outline=outline,
                width=3 if is_best else 1,
            )
            draw.text(
                (x + 28, y + 22),
                f"{float(metrics['psnr']):.2f} dB   ·   {float(metrics['ssim_global']):.3f} SSIM",
                font=metric_font,
                fill="white",
            )
            ratio = float(metrics.get("mean_intensity_ratio", float("nan")))
            corr = float(metrics.get("luminance_correlation", float("nan")))
            draw.text(
                (x + 28, y + 78),
                f"brightness pred/ref {ratio:.3f}    ·    luminance corr {corr:.3f}",
                font=note_font,
                fill=(185, 190, 199),
            )
    _draw_hero_row(
        canvas,
        draw,
        stage / "olat" / "cases" / olat_case_id,
        olat_top,
        "Representative OLAT relighting",
        [("Reference", "reference.png")]
        + [
            (f"{profile.upper()}-trained", f"predictions/{profile}.png")
            for profile in profiles
        ],
        hero_panel_width,
        hero_panel_height,
        hero_label,
        section_font,
        label_font,
    )
    _draw_hero_row(
        canvas,
        draw,
        stage / "hdri" / "cases" / hdri_case_id,
        hdri_top,
        "Representative HDRI relighting · reference synthesized from measured OLAT",
        [("Lighting", "lighting.png"), ("Reference", "reference.png")]
        + [
            (f"{profile.upper()}-trained", f"predictions/{profile}.png")
            for profile in profiles
        ],
        hero_panel_width,
        hero_panel_height,
        hero_label,
        section_font,
        label_font,
    )
    canvas.save(stage / "overview.png")


def _draw_hero_row(
    canvas,
    draw,
    case_dir: Path,
    top: int,
    title: str,
    panels: Sequence[tuple[str, str]],
    panel_width: int,
    panel_height: int,
    label_height: int,
    title_font,
    label_font,
) -> None:
    gap = 16
    total_width = len(panels) * panel_width + (len(panels) - 1) * gap
    start_x = (canvas.width - total_width) // 2
    draw.text((start_x, top), title, font=title_font, fill="white")
    y = top + 58
    for index, (label, relative_path) in enumerate(panels):
        x = start_x + index * (panel_width + gap)
        _centered_text(
            draw,
            (x, y, x + panel_width, y + label_height),
            label,
            label_font,
            (235, 238, 242),
        )
        canvas.paste(
            _fit_pil_path(case_dir / relative_path, panel_width, panel_height),
            (x, y + label_height),
        )


def _validate_clean_report_tree(
    stage: Path, profiles: Sequence[str], suite_reports: dict[str, dict[str, Any]]
) -> None:
    required = [
        stage / "overview.png",
        stage / "summary.json",
        stage / "metrics.csv",
        stage / "assets" / "conditions.json",
        stage / "assets" / "weights.npz",
    ]
    for lighting, report in suite_reports.items():
        suite = stage / lighting
        required.append(suite / "comparison.png")
        for case_id in report["case_ids"]:
            case = suite / "cases" / case_id
            required.extend([case / "reference.png", case / "comparison.png"])
            if lighting == "hdri":
                required.append(case / "lighting.png")
            for profile in profiles:
                required.extend(
                    [
                        case / "predictions" / f"{profile}.png",
                        case / "errors" / f"{profile}.png",
                    ]
                )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"clean report validation failed; missing: {missing[:8]}")
    unexpected = [name for name in ("report", ".profiles") if (stage / name).exists()]
    if unexpected:
        raise RuntimeError(f"clean report contains private/legacy directories: {unexpected}")


def _update_profile_acquisition_reports(
    camera_dir: Path,
    profiles: Sequence[str],
    profile_results: dict[str, dict[str, Any]],
    suite_reports: dict[str, dict[str, Any]],
) -> None:
    adapter = _adapter_provenance()
    for profile in profiles:
        result = profile_results[profile]
        evaluations = result["evaluation"]["evaluations"]
        clean_evaluations = {}
        for lighting in ("olat", "hdri"):
            old = evaluations[lighting]
            case_id = suite_reports[lighting]["representative_case"]
            representative = {
                "reference_path": f"evaluation/{lighting}/cases/{case_id}/reference.png",
                "prediction_path": f"evaluation/{lighting}/cases/{case_id}/predictions/{profile}.png",
                "error_path": f"evaluation/{lighting}/cases/{case_id}/errors/{profile}.png",
                "comparison_path": f"evaluation/{lighting}/cases/{case_id}/comparison.png",
            }
            if lighting == "olat":
                representative["frame_id"] = int(case_id)
            else:
                representative["condition_id"] = case_id
                representative["lighting_path"] = (
                    f"evaluation/hdri/cases/{case_id}/lighting.png"
                )
            clean_evaluations[lighting] = {
                **{
                    key: value
                    for key, value in old.items()
                    if key
                    not in {"representative", "metrics_csv", "contact_sheet", "panels"}
                },
                "schema": "ictpolarreal.relighting-evaluation.v4",
                "metrics_csv": "evaluation/metrics.csv",
                "comparison": f"evaluation/{lighting}/comparison.png",
                "panels": (
                    ["reference", f"{profile}-trained", "fixed_scale_error"]
                    if lighting == "olat"
                    else [
                        "lighting",
                        "OLAT-synthesized reference",
                        f"{profile}-trained",
                        "fixed_scale_error",
                    ]
                ),
                "representative": representative,
            }
        result["schema"] = "ictpolarreal.end2end-disney.v7"
        result["evaluation"] = {
            "schema": "ictpolarreal.profile-evaluation.v2",
            "profile": profile,
            "evaluations": clean_evaluations,
        }
        result.setdefault("paths", {})["evaluation"] = "evaluation"
        result["paths"]["evaluation_suites"] = {
            "olat": "evaluation/olat",
            "hdri": "evaluation/hdri",
        }
        result["adapter"] = adapter
        if isinstance(result.get("checkpoint_signature"), dict):
            result["checkpoint_signature"]["adapter"] = adapter
        acquisition_path = camera_dir / "material" / profile / "acquisition.json"
        if acquisition_path.is_file():
            _write_json_atomic(acquisition_path, result)


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
    metric_names = (
        "mse",
        "mae",
        "psnr",
        "ssim_global",
        "gt_mean_intensity",
        "pred_mean_intensity",
        "mean_intensity_ratio",
        "luminance_correlation",
    )
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
        "schema": "ictpolarreal.relighting-evaluation.v3",
        "split": split,
        "count": len(rows),
        "normalization": (
            "masked whole-image p99.5 linear clipping "
            "(Imaginaire renderer parity)"
        ),
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


def _appearance_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    foreground: np.ndarray,
) -> dict[str, float]:
    selected = foreground[..., 0] > 0.5
    if not np.any(selected):
        raise ValueError("appearance metrics require a non-empty foreground")
    pred_values = prediction[selected]
    target_values = target[selected]
    pred_mean = float(np.mean(pred_values))
    target_mean = float(np.mean(target_values))
    luma = np.asarray([0.2989, 0.5870, 0.1140], dtype=np.float32)
    pred_luma = pred_values @ luma
    target_luma = target_values @ luma
    if float(np.std(pred_luma)) <= 1e-8 or float(np.std(target_luma)) <= 1e-8:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(pred_luma, target_luma)[0, 1])
    return {
        "gt_mean_intensity": target_mean,
        "pred_mean_intensity": pred_mean,
        "mean_intensity_ratio": pred_mean / max(target_mean, 1e-8),
        "luminance_correlation": correlation,
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


def _checkpoint_signatures_match(
    existing: dict[str, Any] | None, expected: dict[str, Any]
) -> bool:
    """Compare numerical-fit provenance while ignoring presentation-only edits.

    Older adapters hashed this entire module, so changing a report label made a
    valid CUDA fit look stale.  ``algorithm_version`` remains the explicit
    compatibility boundary; only that obsolete whole-file digest is ignored.
    """
    if not isinstance(existing, dict):
        return False

    def normalized(signature: dict[str, Any]) -> dict[str, Any]:
        result = json.loads(json.dumps(signature))
        adapter = result.get("adapter")
        if isinstance(adapter, dict):
            adapter.pop("end2end_acquisition_sha256", None)
        return result

    return normalized(existing) == normalized(expected)


def _adapter_provenance() -> dict[str, Any]:
    lighting_path = Path(__file__).resolve().with_name("lighting_profiles.py")
    return {
        "schema": "ictpolarreal.profile-acquisition-adapter.v2",
        "algorithm_version": "superdimension-parity-v2",
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

    # disney_brdf imports torchvision at module scope, but only uses
    # torchvision.utils.save_image inside optional shadow-debug branches.  The
    # production acquisition never enables those branches, so do not make a
    # large compiled vision package a hard runtime dependency.  Keep this shim
    # local to the import: the loaded Disney module retains its reference while
    # unrelated imports in this process continue to see the real environment.
    torchvision_shim = None
    had_torchvision_module = "torchvision" in sys.modules
    previous_torchvision_module = sys.modules.get("torchvision")
    try:
        importlib.import_module("torchvision")
    except ModuleNotFoundError as exc:
        if exc.name != "torchvision":
            raise

        def _missing_torchvision_save_image(*_args, **_kwargs):
            raise RuntimeError(
                "Imaginaire shadow debug image export requires torchvision; "
                "end2end material acquisition itself does not."
            )

        torchvision_shim = types.ModuleType("torchvision")
        torchvision_shim.utils = types.SimpleNamespace(
            save_image=_missing_torchvision_save_image
        )
        sys.modules["torchvision"] = torchvision_shim
    try:
        module = importlib.import_module("CookTorrance_IBL.disney_brdf")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import Imaginaire's Disney BRDF runtime. The selected Python "
            "environment must provide torch, scipy, numpy, and Pillow."
        ) from exc
    finally:
        if (
            torchvision_shim is not None
            and sys.modules.get("torchvision") is torchvision_shim
        ):
            if had_torchvision_module:
                sys.modules["torchvision"] = previous_torchvision_module
            else:
                del sys.modules["torchvision"]
    imported_path = Path(module.__file__).resolve()
    if imported_path != source_path:
        raise RuntimeError(
            f"Imported Disney BRDF from {imported_path}, expected {source_path}. "
            "Use a clean Python process or correct --imaginaire-root."
        )
    return root, module, source_path


def _normalize_targets(targets: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    targets = np.maximum(np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    # Match the renderer: mask first, then compute one p99.5 scale over the
    # complete CHW image (including the now-zero background).
    targets = targets * foreground[None, ...]
    flat = targets.reshape(len(targets), -1)
    scales = np.quantile(flat, PERCENTILE / 100.0, axis=1).astype(np.float32)
    scales = np.maximum(scales, 1e-8)
    return np.clip(targets / scales[:, None, None, None], 0.0, 1.0).astype(np.float32)


def _normalize_render_foreground(render, mask_chw):
    """Mask an HDR tensor, then apply Imaginaire's whole-image p99.5 scale."""
    finite = (
        render.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        * mask_chw
    )
    if not bool((mask_chw > 0.5).any()):
        raise ValueError("cannot normalize a render without foreground pixels")
    scale = _safe_torch_quantile(
        finite.reshape(-1), PERCENTILE / 100.0
    ).clamp_min(1e-8)
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
    image = image * foreground
    values = image.reshape(-1)
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
                **_appearance_metrics(prediction, target, foreground),
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
        target="measured parallel-polarized OLAT (diffuse + specular)",
        panels=["ground_truth", "prediction", "absolute_error_x4"],
        metrics_path="metrics.csv",
        contact_sheet="contact_sheet.png",
        target_origin="measured_parallel_olat",
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
        f"SSIM-global={summary['metrics']['ssim_global']:.4f} "
        f"intensity-ratio={summary['metrics']['mean_intensity_ratio']:.3f} "
        f"corr={summary['metrics']['luminance_correlation']:.3f}",
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
