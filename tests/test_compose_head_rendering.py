from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from ictpolarreal.processing import compose_head_rendering as compositor
from ictpolarreal.processing.compose_head_rendering import (
    EXPECTED_HEAD_COUNT,
    GAP,
    IMAGE_HEIGHT,
    LABEL_HEIGHT,
    PADDING,
    SHEET_COLUMNS,
    TILE_WIDTH,
    compose_head_rendering,
)


def _color(position: int) -> tuple[int, int, int]:
    return (
        (17 * (position + 1)) % 256,
        (53 * (position + 1)) % 256,
        (97 * (position + 1)) % 256,
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _preset_condition(position: int) -> dict:
    if position < 4:
        color_names = ("w", "r", "g", "b")
        source = f"calibration_{color_names[position]}"
        condition_id = f"{source}_rot090"
        source_kind = "generated_calibration"
        source_rank = None
        variance_score = 0.0
    else:
        source_rank = position - 3
        source = f"sd_c04_rank_{source_rank:02d}_source_exact.hdr"
        condition_id = f"sd_c04_rank_{source_rank:02d}_rot090"
        source_kind = "environment_map"
        variance_score = float(1000 - source_rank)
    return {
        "absolute_index": None,
        "condition_id": condition_id,
        "orientation": {"horizontal_flip": False, "rotation_degrees": 90},
        "preset_index": position,
        "rotation_degrees": 90,
        "source": source,
        "source_kind": source_kind,
        "source_rank": source_rank,
        "source_sha256": _digest(source),
        "split": "fit",
        "support_count": 164,
        "variance_score": variance_score,
        "weight_sha256": _digest(f"weights:{condition_id}"),
    }


def _write_renderer_fixture(
    camera_dir: Path,
    *,
    omit_position: int | None = None,
    corrupt_hash_position: int | None = None,
    bad_rank: bool = False,
    omit_preset_provenance: bool = False,
) -> list[dict]:
    material_dir = camera_dir / "material" / "olat"
    stage_dir = material_dir / ".rendering_stage"
    conditions_dir = stage_dir / "conditions"
    conditions_dir.mkdir(parents=True, exist_ok=True)

    rendered_conditions = []
    for position in range(EXPECTED_HEAD_COUNT):
        record = _preset_condition(position)
        render_path = conditions_dir / f"{record['condition_id']}.png"
        if position != omit_position:
            Image.new("RGB", (20, 36), _color(position)).save(render_path)
        record["render_path"] = (
            f"material/olat/.rendering_stage/conditions/{record['condition_id']}.png"
        )
        record["render_sha256"] = (
            _sha256(render_path) if render_path.is_file() else "0" * 64
        )
        if position == corrupt_hash_position:
            record["render_sha256"] = "f" * 64
        rendered_conditions.append(record)
    if bad_rank:
        rendered_conditions[12]["source_rank"] = 99

    material_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = material_dir / "disney_brdf.pt"
    checkpoint_path.write_bytes(b"synthetic saved Disney state")
    validation_condition = rendered_conditions[4]
    manifest = {
        "conditions": rendered_conditions,
        "input_hash_checks": {
            "condition_weights": {
                "actual_sha256": _digest("condition weights"),
                "expected_sha256": _digest("condition weights"),
                "passed": True,
            },
            "material_state": {
                "actual_sha256": _sha256(checkpoint_path),
                "expected_sha256": _sha256(checkpoint_path),
                "passed": True,
            },
        },
        "lighting_preset": "sd-olat-heads",
        "material": {
            "path": "material/olat/disney_brdf.pt",
            "sha256": _sha256(checkpoint_path),
        },
        "preset_provenance": (
            None
            if omit_preset_provenance
            else {
                "calibration_count": 4,
                "candidate_count": 256,
                "environment_count": 32,
                "hdri_root": "/datasets/HDR/hdr_maps_1k",
                "orientation": {
                    "cumulative_quarter_rolls": 2,
                    "label_rotation_degrees": 90,
                },
                "projection": {
                    "support": "ICTPolarReal OLAT fit support",
                    "support_count": 164,
                },
                "ranking": {
                    "camera": "C04",
                    "expected_top32_guard": [
                        record["source"] for record in rendered_conditions[4:]
                    ],
                    "lights_path": "/datasets/SuperDimension/LSX3/C04.txt",
                    "lights_sha256": _digest("C04 lights"),
                },
                "sampler_source": {
                    "path": "/imaginaire/CookTorrance_IBL/CookTorrance.py",
                    "sha256": _digest("sampler source"),
                },
                "schema": "ictpolarreal.sd-olat-heads-preset.v1",
                "trainer_source": {
                    "path": "/imaginaire/trainers/relighting_switchlight_pretrain.py",
                    "sha256": _digest("trainer source"),
                },
            }
        ),
        "profile": "olat",
        "renderer": {
            "device": "cuda",
            "exact_original_summation_order": True,
            "gpu": {
                "name": "Synthetic A100",
                "total_memory_bytes": 80 * 1024**3,
                "total_memory_gib": 80.0,
            },
            "integration": "solid-angle Voronoi",
            "light_chunk": None,
            "model": "DisneyBRDFSimplifiedMultiLayer",
            "slurm_job_id": "12345",
            "tone_map": "Imaginaire p99.5 then output clamp",
        },
        "schema": "ictpolarreal.saved-disney-render.v1",
        "selected_absolute_indices": None,
        "validation": {
            "absolute_index": 428,
            "byte_identical": True,
            "canonical_path": "evaluation/hdri/cases/canonical/predictions/olat.png",
            "canonical_sha256": validation_condition["render_sha256"],
            "condition_id": "recorded_validation_row_428",
            "different_channel_values": 0,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "passed": True,
            "render_path": "material/olat/.rendering_stage/validation/row428.png",
            "render_sha256": validation_condition["render_sha256"],
            "tolerance": 0.0,
        },
    }
    (stage_dir / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation_dir = stage_dir / "validation"
    validation_dir.mkdir(exist_ok=True)
    Image.new("RGB", (4, 2), (128, 128, 128)).save(validation_dir / "row428.png")

    # Deliberately unrelated and far too short: the renderer preset, not this
    # recorded ICT manifest, owns the contact-sheet order.
    conditions_path = camera_dir / "evaluation" / "assets" / "conditions.json"
    conditions_path.parent.mkdir(parents=True, exist_ok=True)
    conditions_path.write_text(
        json.dumps(
            {
                "conditions": [{"condition_id": "unrelated_ict_condition"}],
                "schema": "ictpolarreal.hdri-conditions.v1",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return rendered_conditions


def test_compose_uses_source_exact_renderer_order_and_is_deterministic(tmp_path):
    camera_dir = tmp_path / "dragondruit" / "cam07"
    expected = _write_renderer_fixture(camera_dir)

    rendering_path, manifest_path = compose_head_rendering(camera_dir)
    first_rendering = rendering_path.read_bytes()
    first_manifest = manifest_path.read_bytes()

    payload = json.loads(first_manifest)
    assert payload["schema"] == "ictpolarreal.sd-olat-head-rendering.v1"
    assert payload["presentation"] == "predicted object renderings only"
    assert payload["selection"] == {
        "authority": "render_manifest.conditions order",
        "count": 36,
        "lighting_preset": "sd-olat-heads",
    }
    assert payload["input"]["selection_authority"] == (
        "ordered conditions in saved renderer manifest"
    )
    assert payload["input"]["selected_condition_tiles_retained"] is False
    assert [item["condition_id"] for item in payload["conditions"]] == [
        item["condition_id"] for item in expected
    ]
    assert [item["source_rank"] for item in payload["conditions"]] == (
        [None] * 4 + list(range(1, 33))
    )
    assert "manifest_index" not in payload["conditions"][0]
    assert payload["render_provenance"]["lighting_preset"] == "sd-olat-heads"
    assert payload["render_provenance"]["preset_provenance"]["ranking"][
        "camera"
    ] == "C04"
    assert payload["render_provenance"]["selected_absolute_indices"] is None
    assert payload["render_provenance"]["validation"]["passed"] is True
    assert "render_path" not in payload["render_provenance"]["validation"]
    assert payload["render_provenance"]["renderer"]["slurm_job_id"] == "12345"
    assert ".rendering_stage" not in first_manifest.decode("utf-8")
    assert rendering_path == camera_dir / "material" / "olat" / "rendering.png"
    assert manifest_path == camera_dir / "material" / "olat" / "rendering.json"

    rows = 6
    tile_height = LABEL_HEIGHT + IMAGE_HEIGHT
    expected_size = (
        2 * PADDING + SHEET_COLUMNS * TILE_WIDTH + (SHEET_COLUMNS - 1) * GAP,
        2 * PADDING + rows * tile_height + (rows - 1) * GAP,
    )
    with Image.open(rendering_path) as sheet:
        assert sheet.size == expected_size
        assert sheet.mode == "RGB"
        for position in range(EXPECTED_HEAD_COUNT):
            column = position % SHEET_COLUMNS
            row = position // SHEET_COLUMNS
            x = PADDING + column * (TILE_WIDTH + GAP) + TILE_WIDTH // 2
            y = (
                PADDING
                + row * (tile_height + GAP)
                + LABEL_HEIGHT
                + IMAGE_HEIGHT // 2
            )
            assert sheet.getpixel((x, y)) == _color(position)

    assert not (
        camera_dir / "material" / "olat" / ".rendering.png.pending"
    ).exists()
    assert not (
        camera_dir / "material" / "olat" / ".rendering.json.pending"
    ).exists()
    assert not (camera_dir / "material" / "olat" / ".rendering_stage").exists()
    _write_renderer_fixture(camera_dir)
    compose_head_rendering(camera_dir)
    assert rendering_path.read_bytes() == first_rendering
    assert manifest_path.read_bytes() == first_manifest


def test_compose_rejects_partial_renderer_output(tmp_path):
    camera_dir = tmp_path / "dragondruit" / "cam07"
    _write_renderer_fixture(camera_dir, omit_position=7)

    with pytest.raises(FileNotFoundError, match="all-or-nothing") as error:
        compose_head_rendering(camera_dir)

    assert "position 8" in str(error.value)
    output_dir = camera_dir / "material" / "olat"
    assert not (output_dir / "rendering.png").exists()
    assert not (output_dir / "rendering.json").exists()
    assert (output_dir / ".rendering_stage").is_dir()


@pytest.mark.parametrize(
    ("fixture_kwargs", "message"),
    [
        ({"corrupt_hash_position": 9}, "hash does not match"),
        ({"bad_rank": True}, "source ranks 1..32"),
        ({"omit_preset_provenance": True}, "missing preset_provenance"),
    ],
)
def test_compose_rejects_unverifiable_renderer_provenance(
    tmp_path, fixture_kwargs, message
):
    camera_dir = tmp_path / "dragondruit" / "cam07"
    _write_renderer_fixture(camera_dir, **fixture_kwargs)

    with pytest.raises(ValueError, match=message):
        compose_head_rendering(camera_dir)

    output_dir = camera_dir / "material" / "olat"
    assert not (output_dir / "rendering.png").exists()
    assert not (output_dir / "rendering.json").exists()
    assert (output_dir / ".rendering_stage").is_dir()


def test_failed_pending_validation_preserves_previous_pickup_and_stage(
    tmp_path, monkeypatch
):
    camera_dir = tmp_path / "dragondruit" / "cam07"
    _write_renderer_fixture(camera_dir)
    output_dir = camera_dir / "material" / "olat"
    old_rendering = b"previous validated rendering"
    old_manifest = b"previous validated manifest"
    (output_dir / "rendering.png").write_bytes(old_rendering)
    (output_dir / "rendering.json").write_bytes(old_manifest)

    def reject_pending(*args, **kwargs):
        raise RuntimeError("synthetic final validation failure")

    monkeypatch.setattr(compositor, "_validate_final_outputs", reject_pending)

    with pytest.raises(RuntimeError, match="synthetic final validation failure"):
        compose_head_rendering(camera_dir)

    assert (output_dir / "rendering.png").read_bytes() == old_rendering
    assert (output_dir / "rendering.json").read_bytes() == old_manifest
    assert (output_dir / ".rendering_stage").is_dir()
    assert not (output_dir / ".rendering.png.pending").exists()
    assert not (output_dir / ".rendering.json.pending").exists()
