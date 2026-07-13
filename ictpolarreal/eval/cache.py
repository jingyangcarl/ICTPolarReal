from __future__ import annotations

import json
from pathlib import Path


def forward_cache_signature(sample: dict[str, object]) -> dict[str, object]:
    return {
        "lighting_convention": sample["lighting_convention"],
        "environment_flip": bool(sample["environment_flip"]),
        "environment_rotation_degrees": float(sample["environment_rotation_degrees"]),
    }


def forward_prediction_is_current(path: str | Path, sample: dict[str, object]) -> bool:
    path = Path(path)
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return metadata == forward_cache_signature(sample)


def write_forward_cache_signature(path: str | Path, sample: dict[str, object]) -> None:
    Path(path).with_suffix(".json").write_text(
        json.dumps(forward_cache_signature(sample), indent=2, sort_keys=True) + "\n"
    )
