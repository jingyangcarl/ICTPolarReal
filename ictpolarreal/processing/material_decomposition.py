from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ictpolarreal.data.dataset import CameraSample
from ictpolarreal.data.polarization import apply_mask, hdr_to_ldr, separate_cross_parallel
from ictpolarreal.utils.io import find_first_existing, read_image, write_image


LUMA = np.asarray([0.2989, 0.5870, 0.1140], dtype=np.float32)
EPS = 1e-8


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


def paired_light_ids(sample: CameraSample, light_start: int, max_lights: int | None) -> list[int]:
    cross_dir = sample.camera_dir / "cross"
    parallel_dir = sample.camera_dir / "parallel"
    if not cross_dir.exists() or not parallel_dir.exists():
        return []
    cross = {int(path.stem) for path in cross_dir.iterdir() if path.stem.isdigit()}
    parallel = {int(path.stem) for path in parallel_dir.iterdir() if path.stem.isdigit()}
    ids = [light_id for light_id in sorted(cross & parallel) if light_id >= light_start]
    if max_lights is not None:
        ids = ids[:max_lights]
    return ids


def load_light_directions(
    data_root: str | Path,
    light_ids: Iterable[int],
    *,
    light_root: str | Path | None = None,
) -> np.ndarray:
    ids = list(light_ids)
    if not ids:
        return np.zeros((0, 3), dtype=np.float32)
    table = _load_light_table(data_root, light_root)
    if table is None:
        return _fibonacci_sphere(max(ids) + 1)[ids]
    if max(ids) >= len(table):
        raise ValueError(f"Light id {max(ids)} exceeds available calibrated light count {len(table)}")
    return table[ids].astype(np.float32)


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
) -> MaterialMaps:
    if cross_stack.shape != parallel_stack.shape:
        raise ValueError("cross and parallel stacks must have the same shape")
    if cross_stack.ndim != 4 or cross_stack.shape[-1] != 3:
        raise ValueError("OLAT stacks must have shape (N,H,W,3)")
    if cross_stack.shape[0] != light_dirs.shape[0]:
        raise ValueError("light direction count must match OLAT image count")

    if backend == "torch" or (backend == "auto" and _torch_cuda_available(device)):
        return _decompose_torch(cross_stack, parallel_stack, light_dirs, mask, view_dirs, device, noise)
    return _decompose_numpy(cross_stack, parallel_stack, light_dirs, mask, view_dirs, noise)


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
    preview: bool,
    save_aggregate: bool,
    noise: float,
) -> int:
    light_ids = paired_light_ids(sample, light_start, max_lights)
    if not light_ids:
        return 0

    cross_images = []
    parallel_images = []
    for light_id in light_ids:
        cross_path = sample.light_path("cross", light_id)
        parallel_path = sample.light_path("parallel", light_id)
        if cross_path is None or parallel_path is None:
            continue
        cross = read_image(cross_path)
        parallel = read_image(parallel_path)
        cross_images.append(cross)
        parallel_images.append(parallel)
        diffuse, specular = separate_cross_parallel(cross, parallel)
        _write_light_preview(out_root, sample, light_id, diffuse, specular, preview)

    if not cross_images:
        return 0

    cross_stack = np.stack(cross_images, axis=0).astype(np.float32)
    parallel_stack = np.stack(parallel_images, axis=0).astype(np.float32)
    light_dirs = load_light_directions(data_root, light_ids, light_root=light_root)
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
    )
    _write_material_maps(out_root, sample, maps, mask=mask, preview=preview, save_aggregate=save_aggregate)
    return len(cross_images)


def load_view_directions(data_root: str | Path, sample: CameraSample, hw: tuple[int, int]) -> np.ndarray:
    camera_path = _find_camera_file(data_root, sample)
    if camera_path is None:
        view = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        return np.broadcast_to(view, hw + (3,)).copy()
    try:
        return _lightstage_view_dirs(camera_path, hw)
    except Exception:
        view = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        return np.broadcast_to(view, hw + (3,)).copy()


def _decompose_numpy(
    cross_stack: np.ndarray,
    parallel_stack: np.ndarray,
    light_dirs: np.ndarray,
    mask: np.ndarray | None,
    view_dirs: np.ndarray | None,
    noise: float,
) -> MaterialMaps:
    cross_stack = np.nan_to_num(cross_stack.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    parallel_stack = np.nan_to_num(parallel_stack.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    light_dirs = _normalize(light_dirs.astype(np.float32), axis=-1)
    diffuse_stack = 2.0 * cross_stack
    specular_stack = 2.0 * np.maximum(parallel_stack - cross_stack, 0.0)
    raw_specular_stack = np.maximum(parallel_stack - cross_stack, 0.0)
    n_lights, height, width, _ = cross_stack.shape
    mask_arr = _mask_array(mask, height, width)

    diffuse_albedo, diffuse_normal, occlusion, interreflection = _diffuse_maps(cross_stack, light_dirs, mask_arr, noise)
    specular_albedo, specular_normal, sigma, tangent, bitangent = _specular_maps(
        raw_specular_stack,
        light_dirs,
        view_dirs,
        mask_arr,
        noise,
    )
    roughness = np.sqrt(np.maximum(sigma[..., 0:1] ** 2 + sigma[..., 1:2] ** 2, 0.0))
    anisotropy = (sigma[..., 0:1] - sigma[..., 1:2]) / (sigma[..., 0:1] + sigma[..., 1:2] + EPS)

    if n_lights < 3:
        default_normal = np.broadcast_to(np.asarray([0.5, 0.5, 1.0], dtype=np.float32), (height, width, 3))
        diffuse_normal = _normalize(default_normal, axis=-1)
    if n_lights < 2:
        specular_normal = diffuse_normal.copy()
        tangent, bitangent = _basis_from_normal(specular_normal)

    return MaterialMaps(
        diffuse_albedo=_apply_foreground(diffuse_albedo, mask_arr),
        diffuse_normal=_apply_foreground(diffuse_normal, mask_arr),
        specular_albedo=_apply_foreground(specular_albedo, mask_arr),
        specular_normal=_apply_foreground(specular_normal, mask_arr),
        roughness=_apply_foreground(roughness, mask_arr),
        anisotropy=_apply_foreground(anisotropy, mask_arr),
        sigma=_apply_foreground(sigma, mask_arr),
        tangent=_apply_foreground(tangent, mask_arr),
        bitangent=_apply_foreground(bitangent, mask_arr),
        occlusion=_apply_foreground(occlusion, mask_arr),
        interreflection=_apply_foreground(interreflection, mask_arr),
        mean_diffuse=_apply_foreground(diffuse_stack.mean(axis=0), mask_arr),
        mean_specular=_apply_foreground(specular_stack.mean(axis=0), mask_arr),
    )


def _decompose_torch(
    cross_stack: np.ndarray,
    parallel_stack: np.ndarray,
    light_dirs: np.ndarray,
    mask: np.ndarray | None,
    view_dirs: np.ndarray | None,
    device: str,
    noise: float,
) -> MaterialMaps:
    try:
        import torch
    except ModuleNotFoundError:
        return _decompose_numpy(cross_stack, parallel_stack, light_dirs, mask, view_dirs, noise)

    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    cross = torch.from_numpy(np.nan_to_num(cross_stack.astype(np.float32))).to(torch_device)
    parallel = torch.from_numpy(np.nan_to_num(parallel_stack.astype(np.float32))).to(torch_device)
    dirs = torch.from_numpy(_normalize(light_dirs.astype(np.float32), axis=-1)).to(torch_device)
    luma = torch.tensor(LUMA, device=torch_device)
    raw_spec = torch.clamp(parallel - cross, min=0.0)
    diffuse = 2.0 * cross
    specular = 2.0 * raw_spec
    y = torch.einsum("nhwc,c->nhw", cross, luma)
    s = torch.einsum("nhwc,c->nhw", raw_spec, luma)
    n_lights, height, width, _ = cross.shape
    mask_arr = _mask_array(mask, height, width)
    mask_t = torch.from_numpy(mask_arr[..., 0]).to(torch_device)

    l0 = cross.sum(dim=0)
    if n_lights >= 4:
        albedo = (2.0 * torch.clamp(l0, min=0.0)) / max(n_lights / 2.0, 1.0) * (np.pi**2)
    else:
        albedo = diffuse.mean(dim=0)
    y_flat = y.permute(1, 2, 0).reshape(-1, n_lights)
    normal_flat = torch.matmul(torch.clamp(y_flat - noise, min=0.0), dirs)
    normal = torch.nn.functional.normalize(normal_flat.reshape(height, width, 3), dim=-1, eps=EPS)
    if n_lights < 3:
        normal = torch.nn.functional.normalize(torch.tensor([0.5, 0.5, 1.0], device=torch_device).expand(height, width, 3), dim=-1, eps=EPS)
    ndotl = torch.clamp(torch.einsum("hwc,nc->hwn", normal, dirs), min=0.0)
    denom = torch.sum(ndotl**2, dim=-1, keepdim=True).clamp_min(EPS)
    rho = 2.0 * torch.sum(cross.permute(1, 2, 0, 3) * ndotl[..., None], dim=2) / denom
    albedo = torch.where(denom > EPS, rho, albedo)
    active = (y.permute(1, 2, 0) > noise).float()
    occlusion = torch.sum(active * ndotl, dim=-1, keepdim=True) / max(n_lights / 4.0, 1.0)
    occlusion = occlusion / occlusion.max().clamp_min(EPS)
    interreflection = torch.sum(active[..., None] * torch.clamp(-torch.einsum("hwc,nc->hwn", normal, dirs), min=0.0)[..., None] * cross.permute(1, 2, 0, 3), dim=2)

    view = torch.from_numpy(view_dirs.astype(np.float32) if view_dirs is not None else np.broadcast_to([0.0, 0.0, 1.0], (height, width, 3)).astype(np.float32)).to(torch_device)
    s_flat = s.permute(1, 2, 0).reshape(-1, n_lights)
    rdir = torch.nn.functional.normalize(torch.matmul(torch.clamp(s_flat - noise, min=0.0), dirs).reshape(height, width, 3), dim=-1, eps=EPS)
    spec_normal = torch.nn.functional.normalize(rdir + view, dim=-1, eps=EPS)
    if n_lights < 2:
        spec_normal = normal
    spec_l0 = raw_spec.sum(dim=0)
    spec_albedo = 4.0 * np.pi * torch.einsum("hwc,c->hw", torch.clamp(spec_l0, min=0.0), luma)[..., None] / max(n_lights, 1)
    tangent, bitangent = _basis_from_normal(spec_normal.detach().cpu().numpy())
    sigma = _estimate_sigma_numpy(raw_spec.detach().cpu().numpy(), light_dirs, spec_normal.detach().cpu().numpy(), tangent, bitangent, view.detach().cpu().numpy(), noise)
    roughness = np.sqrt(np.maximum(sigma[..., 0:1] ** 2 + sigma[..., 1:2] ** 2, 0.0))
    anisotropy = (sigma[..., 0:1] - sigma[..., 1:2]) / (sigma[..., 0:1] + sigma[..., 1:2] + EPS)

    mask_np = mask_t.detach().cpu().numpy()[..., None]
    return MaterialMaps(
        diffuse_albedo=_apply_foreground(albedo.detach().cpu().numpy(), mask_np),
        diffuse_normal=_apply_foreground(normal.detach().cpu().numpy(), mask_np),
        specular_albedo=_apply_foreground(spec_albedo.detach().cpu().numpy(), mask_np),
        specular_normal=_apply_foreground(spec_normal.detach().cpu().numpy(), mask_np),
        roughness=_apply_foreground(roughness, mask_np),
        anisotropy=_apply_foreground(anisotropy, mask_np),
        sigma=_apply_foreground(sigma, mask_np),
        tangent=_apply_foreground(tangent, mask_np),
        bitangent=_apply_foreground(bitangent, mask_np),
        occlusion=_apply_foreground(occlusion.detach().cpu().numpy(), mask_np),
        interreflection=_apply_foreground(interreflection.detach().cpu().numpy(), mask_np),
        mean_diffuse=_apply_foreground(diffuse.mean(dim=0).detach().cpu().numpy(), mask_np),
        mean_specular=_apply_foreground(specular.mean(dim=0).detach().cpu().numpy(), mask_np),
    )


def _diffuse_maps(
    cross_stack: np.ndarray,
    light_dirs: np.ndarray,
    mask: np.ndarray,
    noise: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_lights = cross_stack.shape[0]
    diffuse_stack = 2.0 * cross_stack
    l0 = cross_stack.sum(axis=0)
    if n_lights >= 4:
        diffuse_albedo = (2.0 * np.maximum(l0, 0.0)) / max(n_lights / 2.0, 1.0) * (np.pi**2)
    else:
        diffuse_albedo = diffuse_stack.mean(axis=0)
    y = np.einsum("nhwc,c->nhw", cross_stack, LUMA)
    y_flat = np.moveaxis(y, 0, -1).reshape(-1, n_lights)
    normal = _normalize(np.maximum(y_flat - noise, 0.0) @ light_dirs, axis=-1).reshape(cross_stack.shape[1:3] + (3,))
    ndotl = np.maximum(np.einsum("hwc,nc->hwn", normal, light_dirs), 0.0)
    denom = ndotl.square().sum(axis=-1, keepdims=True) if hasattr(ndotl, "square") else np.sum(ndotl**2, axis=-1, keepdims=True)
    rho = 2.0 * np.sum(np.moveaxis(cross_stack, 0, 2) * ndotl[..., None], axis=2) / (denom + EPS)
    diffuse_albedo = np.where(denom > EPS, rho, diffuse_albedo)
    active = np.moveaxis(y > noise, 0, -1).astype(np.float32)
    occlusion = np.sum(active * ndotl, axis=-1, keepdims=True) / max(n_lights / 4.0, 1.0)
    occlusion = occlusion / (float(occlusion.max()) + EPS)
    raw_ndotl = np.einsum("hwc,nc->hwn", normal, light_dirs)
    interreflection = np.sum(active[..., None] * np.maximum(-raw_ndotl, 0.0)[..., None] * np.moveaxis(cross_stack, 0, 2), axis=2)
    return diffuse_albedo, normal, occlusion, interreflection


def _specular_maps(
    specular_stack: np.ndarray,
    light_dirs: np.ndarray,
    view_dirs: np.ndarray | None,
    mask: np.ndarray,
    noise: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_lights, height, width, _ = specular_stack.shape
    view = view_dirs if view_dirs is not None else np.broadcast_to(np.asarray([0.0, 0.0, 1.0], dtype=np.float32), (height, width, 3))
    view = _normalize(view.astype(np.float32), axis=-1)
    y = np.einsum("nhwc,c->nhw", specular_stack, LUMA)
    y_flat = np.moveaxis(y, 0, -1).reshape(-1, n_lights)
    reflect_dir = _normalize(np.maximum(y_flat - noise, 0.0) @ light_dirs, axis=-1).reshape(height, width, 3)
    specular_normal = _normalize(reflect_dir + view, axis=-1)
    specular_albedo = 4.0 * np.pi * np.einsum("hwc,c->hw", np.maximum(specular_stack.sum(axis=0), 0.0), LUMA)[..., None] / max(n_lights, 1)
    tangent, bitangent = _basis_from_normal(specular_normal)
    sigma = _estimate_sigma_numpy(specular_stack, light_dirs, specular_normal, tangent, bitangent, view, noise)
    return specular_albedo, specular_normal, sigma, tangent, bitangent


def _estimate_sigma_numpy(
    specular_stack: np.ndarray,
    light_dirs: np.ndarray,
    normal: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
    view_dirs: np.ndarray,
    noise: float,
    chunk: int = 65536,
) -> np.ndarray:
    n_lights, height, width, _ = specular_stack.shape
    y = np.einsum("nhwc,c->nhw", specular_stack, LUMA)
    weights = np.maximum(np.moveaxis(y, 0, -1).reshape(-1, n_lights) - noise, 0.0)
    normal_f = normal.reshape(-1, 3)
    tangent_f = tangent.reshape(-1, 3)
    bitangent_f = bitangent.reshape(-1, 3)
    view_f = view_dirs.reshape(-1, 3)
    sigma = np.zeros((normal_f.shape[0], 3), dtype=np.float32)
    for start in range(0, normal_f.shape[0], chunk):
        end = min(start + chunk, normal_f.shape[0])
        n = normal_f[start:end]
        t = tangent_f[start:end]
        b = bitangent_f[start:end]
        v = view_f[start:end]
        w = weights[start:end]
        half_vec = _normalize(light_dirs[None, :, :] + v[:, None, :], axis=-1)
        hdot_t = np.sum(half_vec * t[:, None, :], axis=-1)
        hdot_b = np.sum(half_vec * b[:, None, :], axis=-1)
        hdot_n = np.maximum(np.sum(half_vec * n[:, None, :], axis=-1), 0.0)
        w = w * hdot_n
        denom = np.sum(w, axis=-1, keepdims=True) + EPS
        sx = np.sqrt(np.sum(w * hdot_t**2, axis=-1, keepdims=True) / denom)
        sy = np.sqrt(np.sum(w * hdot_b**2, axis=-1, keepdims=True) / denom)
        sigma[start:end, 0:1] = np.clip(sx, 0.02, 1.0)
        sigma[start:end, 1:2] = np.clip(sy, 0.02, 1.0)
    return sigma.reshape(height, width, 3)


def _write_light_preview(out_root: str | Path, sample: CameraSample, light_id: int, diffuse: np.ndarray, specular: np.ndarray, preview: bool) -> None:
    suffix = ".png" if preview else ".exr"
    if preview:
        diffuse = hdr_to_ldr(diffuse)
        specular = hdr_to_ldr(specular)
    base = Path(out_root) / sample.object_name / sample.camera / f"{light_id:06d}"
    write_image(base / f"diffuse{suffix}", diffuse)
    write_image(base / f"specular{suffix}", specular)


def _write_material_maps(
    out_root: str | Path,
    sample: CameraSample,
    maps: MaterialMaps,
    *,
    mask: np.ndarray | None,
    preview: bool,
    save_aggregate: bool,
) -> None:
    material_dir = Path(out_root) / sample.object_name / sample.camera / "material_properties"
    outputs = {
        "diffuse_albedo": maps.diffuse_albedo,
        "diffuse_normal": maps.diffuse_normal,
        "specular_albedo": maps.specular_albedo,
        "specular_normal": maps.specular_normal,
        "roughness": maps.roughness,
        "anisotropy": maps.anisotropy,
        "sigma": maps.sigma,
        "tangent": maps.tangent,
        "bitangent": maps.bitangent,
        "occlusion": maps.occlusion,
        "interreflection": maps.interreflection,
        "albedo": maps.diffuse_albedo,
        "normal": maps.diffuse_normal,
        "specular": maps.specular_albedo,
    }
    for name, image in outputs.items():
        write_image(material_dir / f"{name}.exr", image.astype(np.float32))
        if preview:
            write_image(material_dir / f"{name}.png", _preview_map(name, image))
    if save_aggregate:
        write_image(material_dir / "mean_diffuse.exr", maps.mean_diffuse.astype(np.float32))
        write_image(material_dir / "mean_specular.exr", maps.mean_specular.astype(np.float32))
        if preview:
            write_image(material_dir / "mean_diffuse.png", apply_mask(hdr_to_ldr(maps.mean_diffuse), mask))
            write_image(material_dir / "mean_specular.png", apply_mask(hdr_to_ldr(maps.mean_specular), mask))


def _preview_map(name: str, image: np.ndarray) -> np.ndarray:
    if name.endswith("normal") or name in {"normal", "tangent", "bitangent"}:
        return np.clip(image * 0.5 + 0.5, 0.0, 1.0)
    if name in {"anisotropy"}:
        return np.clip(image * 0.5 + 0.5, 0.0, 1.0)
    return hdr_to_ldr(image) if float(np.nanmax(image)) > 1.0 else np.clip(image, 0.0, 1.0)


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
        pos_file = root / "LSX3_light_positions.txt"
        order_file = root / "LSX3_light_z_spiral.txt"
        if pos_file.exists() and order_file.exists():
            positions = np.genfromtxt(pos_file).astype(np.float32)
            order = np.genfromtxt(order_file).astype(np.int32)
            positions = _normalize(positions, axis=-1)
            rotation_y_180 = np.asarray([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
            positions = positions @ rotation_y_180.T
            return positions[order - 1]
    return None


def _fibonacci_sphere(count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    i = np.arange(count, dtype=np.float32)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (2.0 * i / max(count - 1, 1))
    radius = np.sqrt(np.maximum(1.0 - y * y, 0.0))
    theta = phi * i
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    return _normalize(np.stack([x, y, z], axis=-1).astype(np.float32), axis=-1)


def _find_camera_file(data_root: str | Path, sample: CameraSample) -> Path | None:
    cam_id = sample.camera.replace("cam", "")
    candidates = [
        Path(data_root) / "cameras" / f"camera{cam_id}.txt",
        Path(data_root) / "cameras" / f"{sample.camera}.txt",
        sample.camera_dir / "camera.txt",
    ]
    return next((path for path in candidates if path.exists()), None)


def _lightstage_view_dirs(camera_path: Path, hw: tuple[int, int]) -> np.ndarray:
    text = camera_path.read_text().splitlines()
    focal = np.asarray(text[1].split(), dtype=np.float32)
    pp = np.asarray(text[3].split(), dtype=np.float32)
    resolution = np.asarray(text[5].split(), dtype=np.float32)
    rt = np.asarray([line.split() for line in text[12:15]], dtype=np.float32)
    height, width = hw
    sx = width / max(resolution[0], EPS)
    sy = height / max(resolution[1], EPS)
    focal_x = focal[0] * sx
    pp_x = pp[0] * sx
    pp_y = pp[1] * sy
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    dirs = np.stack(((x - pp_x) / focal_x, (y - pp_y) / focal_x, -np.ones_like(x)), axis=-1)
    dirs = -(dirs @ rt[:3, :3].T)
    return _normalize(dirs, axis=-1)


def _mask_array(mask: np.ndarray | None, height: int, width: int) -> np.ndarray:
    if mask is None:
        return np.ones((height, width, 1), dtype=np.float32)
    if mask.ndim == 2:
        mask = mask[..., None]
    return (mask[..., :1] > 0.5).astype(np.float32)


def _apply_foreground(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) * mask


def _basis_from_normal(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = _normalize(normal.astype(np.float32), axis=-1)
    up = np.zeros_like(normal)
    up[..., 2] = 1.0
    near_parallel = np.abs(np.sum(normal * up, axis=-1, keepdims=True)) > 0.95
    alt = np.zeros_like(normal)
    alt[..., 1] = 1.0
    up = np.where(near_parallel, alt, up)
    tangent = up - np.sum(up * normal, axis=-1, keepdims=True) * normal
    tangent = _normalize(tangent, axis=-1)
    bitangent = _normalize(np.cross(normal, tangent), axis=-1)
    return tangent.astype(np.float32), bitangent.astype(np.float32)


def _normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + EPS)


def _torch_cuda_available(device: str) -> bool:
    if not str(device).startswith("cuda"):
        return False
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return bool(torch.cuda.is_available())
