from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ictpolarreal.data.dataset import CameraSample
from ictpolarreal.data.olat import paired_light_frames, select_light_pairs
from ictpolarreal.data.polarization import hdr_to_ldr
from ictpolarreal.utils.io import read_image, write_image


LUMA = np.asarray([0.2989, 0.5870, 0.1140], dtype=np.float32)
EPS = 1e-8
MIN_FIT_LIGHTS = 4


@dataclass
class MaterialMaps:
    diffuse_albedo: np.ndarray
    diffuse_normal: np.ndarray
    specular_albedo: np.ndarray
    specular_normal: np.ndarray
    roughness: np.ndarray
    anisotropy: np.ndarray
    sigma: np.ndarray
    tangent: np.ndarray
    bitangent: np.ndarray
    occlusion: np.ndarray
    interreflection: np.ndarray
    mean_diffuse: np.ndarray
    mean_specular: np.ndarray


def load_light_directions(
    data_root: str | Path,
    light_indices: Iterable[int],
    *,
    light_root: str | Path | None = None,
) -> np.ndarray:
    indices = list(light_indices)
    if not indices:
        return np.zeros((0, 3), dtype=np.float32)
    table = _load_light_table(data_root, light_root)
    if table is None:
        raise FileNotFoundError(
            "Missing LSX light calibration. Expected LSX3_light_positions.txt and "
            "LSX3_light_z_spiral.txt under metadata/, DATA_ROOT/calibration, or --light-root."
        )
    if min(indices) < 0 or max(indices) >= len(table):
        raise ValueError(f"Light index range {min(indices)}..{max(indices)} exceeds calibration size {len(table)}")
    return table[indices].astype(np.float32)


def decompose_polarized_olat(
    cross_stack: np.ndarray,
    parallel_stack: np.ndarray,
    light_dirs: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    view_dirs: np.ndarray | None = None,
    backend: str = "cpu",
    device: str = "cuda",
    noise: float = 1.5e-3,
    normal_steps: int = 30,
    sigma_steps: int = 50,
    chunk_size: int = 4096,
) -> MaterialMaps:
    if cross_stack.shape != parallel_stack.shape:
        raise ValueError("cross and parallel stacks must have the same shape")
    if cross_stack.ndim != 4 or cross_stack.shape[-1] != 3:
        raise ValueError("OLAT stacks must have shape (N,H,W,3)")
    if cross_stack.shape[0] != light_dirs.shape[0]:
        raise ValueError("light direction count must match OLAT image count")
    if cross_stack.shape[0] < MIN_FIT_LIGHTS:
        raise ValueError(f"Material fitting needs at least {MIN_FIT_LIGHTS} calibrated OLAT pairs")
    if backend not in {"auto", "cpu", "torch"}:
        raise ValueError("backend must be auto, cpu, or torch")

    use_torch = backend == "torch" or (backend == "auto" and _torch_cuda_available(device))
    if use_torch:
        return _decompose_torch(
            cross_stack,
            parallel_stack,
            light_dirs,
            mask,
            view_dirs,
            device,
            noise,
            normal_steps,
            sigma_steps,
            chunk_size,
        )
    return _decompose_numpy(cross_stack, parallel_stack, light_dirs, mask, view_dirs, noise, chunk_size)


def decompose_camera_sample(
    sample: CameraSample,
    *,
    data_root: str | Path,
    out_root: str | Path,
    light_start: int,
    max_lights: int | None,
    light_root: str | Path | None,
    backend: str,
    device: str,
    noise: float,
    frame_layout: str = "auto",
    normal_steps: int = 30,
    sigma_steps: int = 50,
    chunk_size: int = 4096,
) -> int:
    layout, available_pairs = paired_light_frames(sample.camera_dir, frame_layout)
    pairs = select_light_pairs(available_pairs, light_start=light_start, max_lights=max_lights)
    if len(pairs) < MIN_FIT_LIGHTS:
        raise ValueError(
            f"{sample.camera_dir} has {len(available_pairs)} valid paired OLAT lights "
            f"({layout} layout), but {len(pairs)} remain after selection; need at least {MIN_FIT_LIGHTS}. "
            "Raw 350-frame captures must include frames 000002 through 000347."
        )

    cross_images = []
    parallel_images = []
    light_indices = []
    for cross_frame, parallel_frame in pairs:
        cross_path = sample.light_path("cross", cross_frame.frame_id)
        parallel_path = sample.light_path("parallel", parallel_frame.frame_id)
        if cross_path is None or parallel_path is None:
            continue
        cross_images.append(read_image(cross_path))
        parallel_images.append(read_image(parallel_path))
        light_indices.append(cross_frame.light_index)

    print(
        f"[process] {sample.object_name}/{sample.camera}: optimizing {len(cross_images)} "
        f"calibrated OLAT pairs ({layout} frame layout)"
    )
    cross_stack = np.stack(cross_images, axis=0).astype(np.float32)
    parallel_stack = np.stack(parallel_images, axis=0).astype(np.float32)
    light_dirs = load_light_directions(data_root, light_indices, light_root=light_root)
    mask_path = sample.image_path("mask")
    mask = read_image(mask_path, channels=1) if mask_path else None
    view_dirs = load_view_directions(data_root, sample, cross_stack.shape[1:3])
    maps = decompose_polarized_olat(
        cross_stack,
        parallel_stack,
        light_dirs,
        mask=mask,
        view_dirs=view_dirs,
        backend=backend,
        device=device,
        noise=noise,
        normal_steps=normal_steps,
        sigma_steps=sigma_steps,
        chunk_size=chunk_size,
    )
    _write_material_maps(out_root, sample, maps)
    return len(cross_images)


def load_view_directions(data_root: str | Path, sample: CameraSample, hw: tuple[int, int]) -> np.ndarray:
    camera_path = _find_camera_file(data_root, sample)
    if camera_path is None:
        view = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        return np.broadcast_to(view, hw + (3,)).copy()
    try:
        return _lightstage_view_dirs(camera_path, hw)
    except (IndexError, TypeError, ValueError):
        view = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        return np.broadcast_to(view, hw + (3,)).copy()


def _decompose_numpy(
    cross_stack: np.ndarray,
    parallel_stack: np.ndarray,
    light_dirs: np.ndarray,
    mask: np.ndarray | None,
    view_dirs: np.ndarray | None,
    noise: float,
    chunk_size: int,
) -> MaterialMaps:
    cross = _clean_stack(cross_stack)
    parallel = _clean_stack(parallel_stack)
    dirs = _normalize(light_dirs.astype(np.float32), axis=-1)
    raw_specular = np.maximum(parallel - cross, 0.0)
    height, width = cross.shape[1:3]
    mask_arr = _mask_array(mask, height, width)
    view = _prepare_view_dirs(view_dirs, height, width)

    albedo, normal, occlusion, interreflection = _fit_diffuse_numpy(
        cross, dirs, mask_arr, noise, chunk_size
    )
    specular, specular_normal, sigma, tangent, bitangent = _fit_specular_numpy(
        raw_specular, dirs, view, normal, mask_arr, noise, chunk_size
    )
    roughness = np.sqrt(np.maximum(sigma[..., 0:1] ** 2 + sigma[..., 1:2] ** 2, 0.0))
    anisotropy = (sigma[..., 0:1] - sigma[..., 1:2]) / (
        sigma[..., 0:1] + sigma[..., 1:2] + EPS
    )
    return MaterialMaps(
        diffuse_albedo=albedo,
        diffuse_normal=normal,
        specular_albedo=specular,
        specular_normal=specular_normal,
        roughness=_apply_foreground(roughness, mask_arr),
        anisotropy=_apply_foreground(anisotropy, mask_arr),
        sigma=sigma,
        tangent=tangent,
        bitangent=bitangent,
        occlusion=occlusion,
        interreflection=interreflection,
        mean_diffuse=_apply_foreground(2.0 * cross.mean(axis=0), mask_arr),
        mean_specular=_apply_foreground(2.0 * raw_specular.mean(axis=0), mask_arr),
    )


def _fit_diffuse_numpy(
    cross: np.ndarray,
    light_dirs: np.ndarray,
    mask: np.ndarray,
    noise: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_lights, height, width, _ = cross.shape
    pixels = height * width
    foreground = np.flatnonzero(mask.reshape(-1) > 0.5)
    cross_flat = cross.reshape(n_lights, pixels, 3)
    y_flat = np.einsum("npc,c->np", cross_flat, LUMA)
    albedo = np.zeros((pixels, 3), dtype=np.float32)
    normal = np.zeros((pixels, 3), dtype=np.float32)
    occlusion = np.zeros((pixels, 1), dtype=np.float32)
    interreflection = np.zeros((pixels, 3), dtype=np.float32)

    for start in range(0, len(foreground), chunk_size):
        indices = foreground[start : start + chunk_size]
        images = np.transpose(cross_flat[:, indices], (1, 0, 2))
        y = np.maximum(y_flat[:, indices].T, 0.0)
        initial = np.maximum(y - noise, 0.0) @ light_dirs
        n = _normalize_with_fallback(initial, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))

        for _ in range(4):
            raw_dot = n @ light_dirs.T
            visible = raw_dot > 0.0
            design = np.maximum(raw_dot, 0.0)
            rho = np.sum(design * y, axis=1, keepdims=True) / (
                np.sum(design**2, axis=1, keepdims=True) + EPS
            )
            residual = np.abs(y - rho * design)
            scale = np.mean(residual * visible, axis=1, keepdims=True) + noise
            robust = np.minimum(1.0, 2.5 * scale / (residual + EPS))
            weights = visible.astype(np.float32) * robust
            system = np.einsum("bn,ni,nj->bij", weights, light_dirs, light_dirs)
            rhs = np.einsum("bn,bn,ni->bi", weights, y, light_dirs)
            system += np.eye(3, dtype=np.float32)[None] * 1e-4
            candidate = np.linalg.solve(system, rhs[..., None])[..., 0]
            n = _normalize_with_fallback(candidate, n)

        raw_dot = n @ light_dirs.T
        design = np.maximum(raw_dot, 0.0)
        denominator = np.sum(design**2, axis=1, keepdims=True) + EPS
        albedo[indices] = np.maximum(
            2.0 * np.sum(images * design[..., None], axis=1) / denominator,
            0.0,
        )
        active = y > noise
        occlusion[indices] = np.sum(active * design, axis=1, keepdims=True) / max(n_lights / 4.0, 1.0)
        interreflection[indices] = np.sum(
            active[..., None] * np.maximum(-raw_dot, 0.0)[..., None] * images,
            axis=1,
        )
        normal[indices] = n

    occlusion /= float(occlusion.max()) + EPS
    return (
        albedo.reshape(height, width, 3),
        normal.reshape(height, width, 3),
        occlusion.reshape(height, width, 1),
        interreflection.reshape(height, width, 3),
    )


def _fit_specular_numpy(
    specular_stack: np.ndarray,
    light_dirs: np.ndarray,
    view_dirs: np.ndarray,
    diffuse_normal: np.ndarray,
    mask: np.ndarray,
    noise: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_lights, height, width, _ = specular_stack.shape
    pixels = height * width
    foreground = np.flatnonzero(mask.reshape(-1) > 0.5)
    spec_flat = specular_stack.reshape(n_lights, pixels, 3)
    y_flat = np.einsum("npc,c->np", spec_flat, LUMA)
    view_flat = view_dirs.reshape(pixels, 3)
    diffuse_flat = diffuse_normal.reshape(pixels, 3)
    albedo = np.zeros((pixels, 1), dtype=np.float32)
    normal = np.zeros((pixels, 3), dtype=np.float32)
    sigma = np.zeros((pixels, 3), dtype=np.float32)
    tangent = np.zeros((pixels, 3), dtype=np.float32)
    bitangent = np.zeros((pixels, 3), dtype=np.float32)

    for start in range(0, len(foreground), chunk_size):
        indices = foreground[start : start + chunk_size]
        y = np.maximum(y_flat[:, indices].T, 0.0)
        target = np.maximum(y - noise, 0.0)
        view = view_flat[indices]
        half = _normalize(light_dirs[None] + view[:, None], axis=-1)
        weighted_half = np.sum(target[..., None] * half, axis=1)
        n = _normalize_with_fallback(weighted_half, diffuse_flat[indices])
        valid = np.count_nonzero(target > 0.0, axis=1) >= MIN_FIT_LIGHTS
        n = np.where(valid[:, None], n, diffuse_flat[indices])
        t, b = _basis_from_normal(n)

        hdot_t = np.sum(half * t[:, None], axis=-1)
        hdot_b = np.sum(half * b[:, None], axis=-1)
        hdot_n = np.maximum(np.sum(half * n[:, None], axis=-1), 0.0)
        weight_sum = np.sum(target, axis=1, keepdims=True) + EPS
        sx = 2.0 * np.sqrt(
            np.sum(target * hdot_t**2 / (1.0 + hdot_n), axis=1, keepdims=True) / weight_sum
        )
        sy = 2.0 * np.sqrt(
            np.sum(target * hdot_b**2 / (1.0 + hdot_n), axis=1, keepdims=True) / weight_sum
        )
        fitted_sigma = np.clip(np.concatenate([sx, sy], axis=1), 0.02, 10.0)
        fitted_sigma = np.where(valid[:, None], fitted_sigma, 0.0)
        response = _ward_response_numpy(light_dirs, view, n, t, b, np.maximum(fitted_sigma, 0.02))
        rho = np.sum(response * y, axis=1, keepdims=True) / (
            np.sum(response**2, axis=1, keepdims=True) + EPS
        )

        albedo[indices] = np.where(valid[:, None], np.maximum(rho, 0.0), 0.0)
        normal[indices] = n
        sigma[indices, :2] = fitted_sigma
        tangent[indices] = t
        bitangent[indices] = b

    return (
        albedo.reshape(height, width, 1),
        normal.reshape(height, width, 3),
        sigma.reshape(height, width, 3),
        tangent.reshape(height, width, 3),
        bitangent.reshape(height, width, 3),
    )


def _decompose_torch(
    cross_stack: np.ndarray,
    parallel_stack: np.ndarray,
    light_dirs: np.ndarray,
    mask: np.ndarray | None,
    view_dirs: np.ndarray | None,
    device: str,
    noise: float,
    normal_steps: int,
    sigma_steps: int,
    chunk_size: int,
) -> MaterialMaps:
    try:
        import torch
        import torch.nn.functional as functional
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("The torch backend requires PyTorch. Run `bash run.sh setup`.") from exc

    torch_device = _resolve_torch_device(torch, device)
    cross = _clean_stack(cross_stack)
    parallel = _clean_stack(parallel_stack)
    raw_specular = np.maximum(parallel - cross, 0.0)
    dirs = torch.as_tensor(_normalize(light_dirs.astype(np.float32), axis=-1), device=torch_device)
    luma = torch.as_tensor(LUMA, device=torch_device)
    n_lights, height, width, _ = cross.shape
    pixels = height * width
    mask_arr = _mask_array(mask, height, width)
    foreground = np.flatnonzero(mask_arr.reshape(-1) > 0.5)
    view = _prepare_view_dirs(view_dirs, height, width).reshape(pixels, 3)
    cross_flat = cross.reshape(n_lights, pixels, 3)
    specular_flat = raw_specular.reshape(n_lights, pixels, 3)

    outputs = {
        "albedo": np.zeros((pixels, 3), dtype=np.float32),
        "normal": np.zeros((pixels, 3), dtype=np.float32),
        "specular": np.zeros((pixels, 1), dtype=np.float32),
        "specular_normal": np.zeros((pixels, 3), dtype=np.float32),
        "sigma": np.zeros((pixels, 3), dtype=np.float32),
        "tangent": np.zeros((pixels, 3), dtype=np.float32),
        "bitangent": np.zeros((pixels, 3), dtype=np.float32),
        "occlusion": np.zeros((pixels, 1), dtype=np.float32),
        "interreflection": np.zeros((pixels, 3), dtype=np.float32),
    }

    for start in range(0, len(foreground), chunk_size):
        indices = foreground[start : start + chunk_size]
        cross_pixels = torch.as_tensor(
            np.ascontiguousarray(np.transpose(cross_flat[:, indices], (1, 0, 2))),
            device=torch_device,
        )
        specular_pixels = torch.as_tensor(
            np.ascontiguousarray(np.transpose(specular_flat[:, indices], (1, 0, 2))),
            device=torch_device,
        )
        view_pixels = torch.as_tensor(np.ascontiguousarray(view[indices]), device=torch_device)
        y_diffuse = torch.einsum("bnc,c->bn", cross_pixels, luma).clamp_min(0.0)
        diffuse_normal = _optimize_diffuse_normal_torch(
            torch, functional, y_diffuse, dirs, noise, normal_steps
        )
        raw_dot = diffuse_normal @ dirs.T
        design = raw_dot.clamp_min(0.0)
        denominator = design.square().sum(dim=1, keepdim=True).clamp_min(EPS)
        diffuse_albedo = (
            2.0 * (cross_pixels * design[..., None]).sum(dim=1) / denominator
        ).clamp_min(0.0)
        active = y_diffuse > noise
        occlusion = (active * design).sum(dim=1, keepdim=True) / max(n_lights / 4.0, 1.0)
        interreflection = (
            active[..., None] * (-raw_dot).clamp_min(0.0)[..., None] * cross_pixels
        ).sum(dim=1)

        y_specular = torch.einsum("bnc,c->bn", specular_pixels, luma).clamp_min(0.0)
        specular_normal, valid_specular = _optimize_specular_normal_torch(
            torch,
            functional,
            y_specular,
            dirs,
            view_pixels,
            diffuse_normal,
            noise,
            normal_steps,
        )
        specular_sigma, tangent, bitangent = _optimize_sigma_torch(
            torch,
            functional,
            y_specular,
            dirs,
            view_pixels,
            specular_normal,
            valid_specular,
            sigma_steps,
        )
        response = _ward_response_torch(
            torch, functional, dirs, view_pixels, specular_normal, tangent, bitangent, specular_sigma.clamp_min(0.02)
        )
        specular_albedo = (
            (response * y_specular).sum(dim=1, keepdim=True)
            / response.square().sum(dim=1, keepdim=True).clamp_min(EPS)
        ).clamp_min(0.0)
        specular_albedo = torch.where(valid_specular[:, None], specular_albedo, 0.0)

        chunk_outputs = {
            "albedo": diffuse_albedo,
            "normal": diffuse_normal,
            "specular": specular_albedo,
            "specular_normal": specular_normal,
            "sigma": functional.pad(specular_sigma, (0, 1)),
            "tangent": tangent,
            "bitangent": bitangent,
            "occlusion": occlusion,
            "interreflection": interreflection,
        }
        for name, value in chunk_outputs.items():
            outputs[name][indices] = value.detach().cpu().numpy()

    outputs["occlusion"] /= float(outputs["occlusion"].max()) + EPS
    sigma = outputs["sigma"].reshape(height, width, 3)
    roughness = np.sqrt(np.maximum(sigma[..., 0:1] ** 2 + sigma[..., 1:2] ** 2, 0.0))
    anisotropy = (sigma[..., 0:1] - sigma[..., 1:2]) / (
        sigma[..., 0:1] + sigma[..., 1:2] + EPS
    )
    return MaterialMaps(
        diffuse_albedo=outputs["albedo"].reshape(height, width, 3),
        diffuse_normal=outputs["normal"].reshape(height, width, 3),
        specular_albedo=outputs["specular"].reshape(height, width, 1),
        specular_normal=outputs["specular_normal"].reshape(height, width, 3),
        roughness=roughness,
        anisotropy=anisotropy,
        sigma=sigma,
        tangent=outputs["tangent"].reshape(height, width, 3),
        bitangent=outputs["bitangent"].reshape(height, width, 3),
        occlusion=outputs["occlusion"].reshape(height, width, 1),
        interreflection=outputs["interreflection"].reshape(height, width, 3),
        mean_diffuse=_apply_foreground(2.0 * cross.mean(axis=0), mask_arr),
        mean_specular=_apply_foreground(2.0 * raw_specular.mean(axis=0), mask_arr),
    )


def _optimize_diffuse_normal_torch(torch, functional, y, light_dirs, noise, steps):
    initial = y.clamp_min(0.0) @ light_dirs
    fallback = torch.tensor([0.0, 0.0, 1.0], device=y.device).expand_as(initial)
    initial = _normalize_torch_with_fallback(functional, initial, fallback)
    hemisphere = initial @ light_dirs.T > 0.0
    target = y.clamp_min(0.0) * hemisphere
    active = (y > noise) & hemisphere
    valid = active.sum(dim=1) >= MIN_FIT_LIGHTS
    if steps <= 0 or not bool(valid.any()):
        return initial

    parameter = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([parameter], lr=0.05)
    target_unit = functional.normalize(target, dim=1, eps=EPS)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        normal = functional.normalize(parameter, dim=1, eps=EPS)
        response = (normal @ light_dirs.T) * hemisphere
        response = functional.normalize(response, dim=1, eps=EPS)
        correlation = (active * response * target_unit).sum(dim=1)
        loss = (1.0 - correlation[valid]).mean()
        loss.backward()
        optimizer.step()
    return functional.normalize(parameter.detach(), dim=1, eps=EPS)


def _optimize_specular_normal_torch(
    torch,
    functional,
    y,
    light_dirs,
    view_dirs,
    diffuse_normal,
    noise,
    steps,
):
    target = y.clamp_min(0.0)
    reflect_dir = functional.normalize(target @ light_dirs, dim=1, eps=EPS)
    initial = _normalize_torch_with_fallback(functional, reflect_dir + view_dirs, diffuse_normal)
    hemisphere = initial @ light_dirs.T > 0.0
    target = target * hemisphere
    active = (y > noise) & hemisphere
    valid = active.sum(dim=1) >= MIN_FIT_LIGHTS
    initial = torch.where(valid[:, None], initial, diffuse_normal)
    if steps <= 0 or not bool(valid.any()):
        return initial, valid

    parameter = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([parameter], lr=0.04)
    target_unit = functional.normalize(target, dim=1, eps=EPS)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        normal = functional.normalize(parameter, dim=1, eps=EPS)
        ndotl = normal @ light_dirs.T
        reflected = 2.0 * ndotl[..., None] * normal[:, None] - light_dirs[None]
        response = (reflected * view_dirs[:, None]).sum(dim=-1) * hemisphere
        response = functional.normalize(response, dim=1, eps=EPS)
        correlation = (active * response * target_unit).sum(dim=1)
        loss = (1.0 - correlation[valid]).mean()
        loss.backward()
        optimizer.step()
    normal = functional.normalize(parameter.detach(), dim=1, eps=EPS)
    return torch.where(valid[:, None], normal, diffuse_normal), valid


def _optimize_sigma_torch(
    torch,
    functional,
    y,
    light_dirs,
    view_dirs,
    normal,
    valid,
    steps,
):
    tangent, bitangent = _basis_from_normal_torch(torch, functional, normal)
    half = functional.normalize(light_dirs[None] + view_dirs[:, None], dim=-1, eps=EPS)
    hdot_t = (half * tangent[:, None]).sum(dim=-1)
    hdot_b = (half * bitangent[:, None]).sum(dim=-1)
    hdot_n = (half * normal[:, None]).sum(dim=-1).clamp_min(0.0)
    weight_sum = y.sum(dim=1, keepdim=True).clamp_min(EPS)
    sx = 2.0 * torch.sqrt((y * hdot_t.square() / (1.0 + hdot_n)).sum(dim=1, keepdim=True) / weight_sum)
    sy = 2.0 * torch.sqrt((y * hdot_b.square() / (1.0 + hdot_n)).sum(dim=1, keepdim=True) / weight_sum)
    initial = torch.cat([sx, sy], dim=1).clamp(0.02, 10.0)
    initial = torch.where(valid[:, None], initial, torch.zeros_like(initial))
    if steps <= 0 or not bool(valid.any()):
        return initial, tangent, bitangent

    minimum, maximum = 0.02, 10.0
    unit = ((initial.clamp_min(minimum + 1e-4) - minimum) / (maximum - minimum)).clamp(1e-5, 1.0 - 1e-5)
    parameter = torch.logit(unit).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([parameter], lr=0.08)
    target_unit = functional.normalize(y, dim=1, eps=EPS)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        sigma = minimum + (maximum - minimum) * torch.sigmoid(parameter)
        response = _ward_response_torch(
            torch, functional, light_dirs, view_dirs, normal, tangent, bitangent, sigma
        )
        response_unit = functional.normalize(response, dim=1, eps=EPS)
        residual = (response_unit - target_unit).square().sum(dim=1)
        loss = residual[valid].mean()
        loss.backward()
        optimizer.step()
    sigma = minimum + (maximum - minimum) * torch.sigmoid(parameter.detach())
    return torch.where(valid[:, None], sigma, torch.zeros_like(sigma)), tangent, bitangent


def _ward_response_numpy(light_dirs, view_dirs, normal, tangent, bitangent, sigma):
    half = _normalize(light_dirs[None] + view_dirs[:, None], axis=-1)
    ndotl = np.sum(normal[:, None] * light_dirs[None], axis=-1)
    ndotv = np.sum(normal * view_dirs, axis=-1, keepdims=True)
    hdot_n = np.sum(half * normal[:, None], axis=-1)
    hdot_t = np.sum(half * tangent[:, None], axis=-1)
    hdot_b = np.sum(half * bitangent[:, None], axis=-1)
    sx, sy = sigma[:, 0:1], sigma[:, 1:2]
    log_response = (
        -0.5 * np.log(np.maximum(ndotv * ndotl, EPS))
        - np.log(4.0 * np.pi * sx * sy + EPS)
        - 2.0 * ((hdot_t / sx) ** 2 + (hdot_b / sy) ** 2) / (1.0 + hdot_n + EPS)
    )
    response = np.exp(np.clip(log_response, -50.0, 20.0))
    return np.where((ndotl > 0.0) & (ndotv > 0.0), response, 0.0).astype(np.float32)


def _ward_response_torch(torch, functional, light_dirs, view_dirs, normal, tangent, bitangent, sigma):
    half = functional.normalize(light_dirs[None] + view_dirs[:, None], dim=-1, eps=EPS)
    ndotl = (normal[:, None] * light_dirs[None]).sum(dim=-1)
    ndotv = (normal * view_dirs).sum(dim=-1, keepdim=True)
    hdot_n = (half * normal[:, None]).sum(dim=-1)
    hdot_t = (half * tangent[:, None]).sum(dim=-1)
    hdot_b = (half * bitangent[:, None]).sum(dim=-1)
    sx, sy = sigma[:, 0:1], sigma[:, 1:2]
    log_response = (
        -0.5 * torch.log((ndotv * ndotl).clamp_min(EPS))
        - torch.log(4.0 * torch.pi * sx * sy + EPS)
        - 2.0 * ((hdot_t / sx).square() + (hdot_b / sy).square()) / (1.0 + hdot_n + EPS)
    )
    response = torch.exp(log_response.clamp(-50.0, 20.0))
    return torch.where((ndotl > 0.0) & (ndotv > 0.0), response, 0.0)


def _write_material_maps(out_root: str | Path, sample: CameraSample, maps: MaterialMaps) -> None:
    material_dir = Path(out_root) / sample.object_name / sample.camera / "brdf"
    foreground = np.linalg.norm(maps.diffuse_normal, axis=-1, keepdims=True) > EPS
    outputs = {
        "albedo": maps.diffuse_albedo,
        "normal": maps.diffuse_normal,
        "specular": maps.specular_albedo,
        "roughness": maps.roughness,
        "anisotropy": maps.anisotropy,
        "sigma": maps.sigma,
        "tangent": maps.tangent,
        "bitangent": maps.bitangent,
    }
    for name, image in outputs.items():
        write_image(material_dir / f"{name}.png", _preview_map(name, image) * foreground)


def _preview_map(name: str, image: np.ndarray) -> np.ndarray:
    if name in {"normal", "tangent", "bitangent"}:
        return np.clip(image * 0.5 + 0.5, 0.0, 1.0)
    if name == "anisotropy":
        return np.clip(image * 0.5 + 0.5, 0.0, 1.0)
    if name in {"roughness", "sigma"}:
        return np.clip(image / (1.0 + np.maximum(image, 0.0)), 0.0, 1.0)
    return hdr_to_ldr(image)


def _load_light_table(data_root: str | Path, light_root: str | Path | None) -> np.ndarray | None:
    roots = []
    if light_root:
        roots.append(Path(light_root))
    data_root = Path(data_root)
    roots.extend(
        [
            data_root / "calibration",
            data_root / "metadata",
            data_root / "LSX",
            Path(__file__).resolve().parents[2] / "metadata",
            Path(__file__).resolve().parents[4] / "data" / "LSX",
        ]
    )
    for root in roots:
        positions_path = root / "LSX3_light_positions.txt"
        order_path = root / "LSX3_light_z_spiral.txt"
        if positions_path.exists() and order_path.exists():
            positions = np.genfromtxt(positions_path).astype(np.float32)
            order = np.genfromtxt(order_path).astype(np.int32)
            positions = _normalize(positions, axis=-1)
            rotation_y_180 = np.asarray(
                [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                dtype=np.float32,
            )
            return (positions @ rotation_y_180.T)[order - 1]
    return None


def _find_camera_file(data_root: str | Path, sample: CameraSample) -> Path | None:
    camera_id = sample.camera.replace("cam", "")
    filename = f"camera{camera_id}.txt"
    candidates = [
        Path(data_root) / "cameras" / filename,
        Path(data_root) / "cameras" / f"{sample.camera}.txt",
        sample.camera_dir / "camera.txt",
        Path(__file__).resolve().parents[2] / "metadata" / "cameras" / filename,
    ]
    return next((path for path in candidates if path.exists()), None)


def _lightstage_view_dirs(camera_path: Path, hw: tuple[int, int]) -> np.ndarray:
    text = camera_path.read_text().splitlines()
    focal = np.asarray(text[1].split(), dtype=np.float32)
    principal = np.asarray(text[3].split(), dtype=np.float32)
    resolution = np.asarray(text[5].split(), dtype=np.float32)
    rotation = np.asarray([line.split() for line in text[12:15]], dtype=np.float32)[:, :3]
    height, width = hw
    scale_x = width / max(resolution[0], EPS)
    scale_y = height / max(resolution[1], EPS)
    focal_x = focal[0] * scale_x
    principal_x = principal[0] * scale_x
    principal_y = principal[1] * scale_y
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    directions = np.stack(
        ((x - principal_x) / focal_x, (y - principal_y) / focal_x, -np.ones_like(x)),
        axis=-1,
    )
    return _normalize(-(directions @ rotation.T), axis=-1)


def _prepare_view_dirs(view_dirs: np.ndarray | None, height: int, width: int) -> np.ndarray:
    if view_dirs is None:
        view_dirs = np.broadcast_to(
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            (height, width, 3),
        )
    return _normalize(view_dirs.astype(np.float32), axis=-1)


def _mask_array(mask: np.ndarray | None, height: int, width: int) -> np.ndarray:
    if mask is None:
        return np.ones((height, width, 1), dtype=np.float32)
    if mask.ndim == 2:
        mask = mask[..., None]
    return (mask[..., :1] > 0.5).astype(np.float32)


def _clean_stack(stack: np.ndarray) -> np.ndarray:
    return np.nan_to_num(stack.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _apply_foreground(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) * mask


def _basis_from_normal(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = _normalize(normal.astype(np.float32), axis=-1)
    up = np.zeros_like(normal)
    up[..., 2] = 1.0
    alternate = np.zeros_like(normal)
    alternate[..., 1] = 1.0
    near_parallel = np.abs(np.sum(normal * up, axis=-1, keepdims=True)) > 0.95
    up = np.where(near_parallel, alternate, up)
    tangent = _normalize(up - np.sum(up * normal, axis=-1, keepdims=True) * normal, axis=-1)
    bitangent = _normalize(np.cross(normal, tangent), axis=-1)
    return tangent.astype(np.float32), bitangent.astype(np.float32)


def _basis_from_normal_torch(torch, functional, normal):
    up = torch.zeros_like(normal)
    up[:, 2] = 1.0
    alternate = torch.zeros_like(normal)
    alternate[:, 1] = 1.0
    near_parallel = (normal * up).sum(dim=-1, keepdim=True).abs() > 0.95
    up = torch.where(near_parallel, alternate, up)
    tangent = functional.normalize(up - (up * normal).sum(dim=-1, keepdim=True) * normal, dim=-1, eps=EPS)
    bitangent = functional.normalize(torch.cross(normal, tangent, dim=-1), dim=-1, eps=EPS)
    return tangent, bitangent


def _normalize_with_fallback(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    normalized = values / (norm + EPS)
    fallback = np.broadcast_to(fallback, values.shape)
    fallback = _normalize(fallback.astype(np.float32), axis=-1)
    return np.where(norm > 1e-6, normalized, fallback).astype(np.float32)


def _normalize_torch_with_fallback(functional, values, fallback):
    norm = values.norm(dim=-1, keepdim=True)
    normalized = functional.normalize(values, dim=-1, eps=EPS)
    fallback = functional.normalize(fallback, dim=-1, eps=EPS)
    return normalized.where(norm > 1e-6, fallback)


def _normalize(values: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(values, axis=axis, keepdims=True)
    return values / (norm + EPS)


def _resolve_torch_device(torch, device: str):
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("[process] CUDA is unavailable; running the PyTorch optimizer on CPU.")
        return torch.device("cpu")
    return torch.device(device)


def _torch_cuda_available(device: str) -> bool:
    if not str(device).startswith("cuda"):
        return False
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return bool(torch.cuda.is_available())
