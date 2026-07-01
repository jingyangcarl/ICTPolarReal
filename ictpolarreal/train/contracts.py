from __future__ import annotations

from collections.abc import Callable


INVERSE_PROMPTS = {
    "albedo": "Albedo (diffuse basecolor)",
    "normal": "Camera-space Normal",
    "specular": "Specular Albedo",
    "cross": "Cross Polarization",
    "parallel": "Parallel Polarization",
}

INVERSE_BATCH_KEYS = {
    "albedo": "albedo",
    "normal": "normal_inverse",
    "specular": "specular",
    "cross": "cross",
    "parallel": "parallel",
}


def inverse_target_names(workflow: str) -> tuple[str, ...]:
    if workflow == "pbr":
        return "albedo", "normal", "specular"
    if workflow == "polarization":
        return "cross", "parallel"
    if workflow == "both":
        return "albedo", "normal", "specular", "cross", "parallel"
    raise ValueError("inverse workflow must be pbr, polarization, or both")


def inverse_target(batch: dict, name: str):
    try:
        return batch[INVERSE_BATCH_KEYS[name]], INVERSE_PROMPTS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported inverse target: {name}") from exc


def build_forward_condition(
    batch: dict,
    *,
    mode: str,
    encode: Callable,
    latent_hw: tuple[int, int],
):
    import torch
    import torch.nn.functional as functional

    if mode == "gbuffer":
        first = encode(batch["albedo"])
        encoded = [first, encode(batch["normal_forward"]), encode(batch["specular"])]
    elif mode == "polarization":
        first = encode(batch["reference_cross"])
        encoded = [first, encode(batch["reference_parallel"])]
    else:
        raise ValueError("forward mode must be gbuffer or polarization")

    black = torch.zeros_like(first)
    if mode == "gbuffer":
        encoded.append(black)
    else:
        encoded.extend((black, black))

    irradiance = functional.interpolate(
        batch["irradiance"].to(dtype=first.dtype),
        size=latent_hw,
        mode="bilinear",
        align_corners=False,
    )
    condition = torch.cat((*encoded, irradiance[:, :3]), dim=1)
    if condition.shape[1] != 19:
        raise ValueError(f"RGB2X forward conditioning must have 19 channels, got {condition.shape[1]}")
    return condition
