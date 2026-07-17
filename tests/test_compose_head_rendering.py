from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import pytest
from PIL import Image

from ictpolarreal.processing import compose_head_rendering as compositor
from ictpolarreal.processing.compose_head_rendering import (
    EXPECTED_HEAD_COUNT,
    PREDICTION_LABEL_NUMBERS,
    SHEET_COLUMNS,
    SHEET_ROWS,
    compose_head_rendering,
)


TILE_SIZE = (128, 96)


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


def _orientation() -> dict:
    return {
        "cumulative_quarter_rolls": 2,
        "effective_raw_shift_degrees": 180,
        "horizontal_shift_fraction": 0.5,
        "label_rotation_degrees": 90,
        "trainer_rotation_index": 1,
    }


def _preset_condition(position: int) -> dict:
    if position < 4:
        color_names = ("w", "r", "g", "b")
        source = f"calibration_{color_names[position]}"
        condition_id = f"{source}_sd_c04_rot090"
        source_kind = "generated_calibration"
        source_rank = None
        variance_score = 0.0
    else:
        source_rank = position - 3
        source = f"sd_c04_rank_{source_rank:02d}_source_exact.hdr"
        condition_id = f"sd_c04_rank_{source_rank:02d}_sd_c04_rot090"
        source_kind = "environment_map"
        variance_score = float(1000 - source_rank)
    return {
        "absolute_index": None,
        "condition_id": condition_id,
        "orientation": _orientation(),
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
    mismatched_size_position: int | None = None,
    non_rgb_position: int | None = None,
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
            size = (129, 96) if position == mismatched_size_position else TILE_SIZE
            mode = "L" if position == non_rgb_position else "RGB"
            color = 127 if mode == "L" else _color(position)
            Image.new(mode, size, color).save(render_path)
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

    validation_dir = stage_dir / "validation"
    validation_dir.mkdir(exist_ok=True)
    validation_path = validation_dir / "recorded_validation_row_428.png"
    Image.new("RGB", (8, 8), (128, 96, 64)).save(validation_path)
    canonical_path = (
        camera_dir
        / "evaluation"
        / "hdri"
        / "cases"
        / "recorded_validation_row_428"
        / "predictions"
        / "olat.png"
    )
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(validation_path, canonical_path)
    validation_sha256 = _sha256(validation_path)

    preset_provenance = None
    if not omit_preset_provenance:
        preset_provenance = {
            "calibration_count": 4,
            "candidate_count": 256,
            "environment_count": 32,
            "hdri_root": "/datasets/HDR/hdr_maps_1k",
            "orientation": _orientation(),
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
                "recorded_direction_count": 155,
                "xyz_table_runtime_sha256": _digest("runtime xyz table"),
                "xyz_table_trainer_cpu_audit_sha256": _digest(
                    "trainer CPU xyz table"
                ),
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
        "preset_provenance": preset_provenance,
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
            "canonical_path": (
                "evaluation/hdri/cases/recorded_validation_row_428/"
                "predictions/olat.png"
            ),
            "canonical_sha256": validation_sha256,
            "condition_id": "recorded_validation_row_428",
            "different_channel_values": 0,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "passed": True,
            "render_path": (
                "material/olat/.rendering_stage/validation/"
                "recorded_validation_row_428.png"
            ),
            "render_sha256": validation_sha256,
            "tolerance": 0.0,
        },
    }
    (stage_dir / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rendered_conditions


def test_compose_writes_only_native_gapless_12x3_sd_grid(tmp_path, monkeypatch):
    camera_dir = tmp_path / "dragondruit" / "cam07"
    _write_renderer_fixture(camera_dir)
    material_dir = camera_dir / "material" / "olat"
    (material_dir / "rendering.json").write_text("stale sidecar", encoding="utf-8")

    calls = []
    original_put_text = cv2.putText

    def record_put_text(image, text, origin, font, scale, color, thickness, line):
        calls.append((text, origin, font, scale, color, thickness, line))
        return original_put_text(
            image, text, origin, font, scale, color, thickness, line
        )

    monkeypatch.setattr(cv2, "putText", record_put_text)
    rendering_path = compose_head_rendering(camera_dir)
    first_rendering = rendering_path.read_bytes()

    assert rendering_path == material_dir / "rendering.png"
    assert not (material_dir / "rendering.json").exists()
    assert not (material_dir / ".rendering_stage").exists()
    assert not (material_dir / ".rendering.png.pending").exists()
    assert [call[0] for call in calls] == [
        f"pred #{number}" for number in PREDICTION_LABEL_NUMBERS
    ]
    font_size = max(10, int(TILE_SIZE[1] * 0.07))
    assert all(call[1] == (4, TILE_SIZE[1] - 4) for call in calls)
    assert all(call[2] == cv2.FONT_HERSHEY_SIMPLEX for call in calls)
    assert all(call[3] == font_size / 30.0 for call in calls)
    assert all(call[4] == (255, 255, 0) for call in calls)
    assert all(call[5] == max(1, font_size // 15) for call in calls)
    assert all(call[6] == cv2.LINE_AA for call in calls)

    with Image.open(rendering_path) as sheet:
        assert sheet.format == "PNG"
        assert sheet.mode == "RGB"
        assert sheet.size == (
            SHEET_COLUMNS * TILE_SIZE[0],
            SHEET_ROWS * TILE_SIZE[1],
        )
        for position in range(EXPECTED_HEAD_COUNT):
            row, column = divmod(position, SHEET_COLUMNS)
            # Upper-right stays outside the lower-left label.  Sampling directly
            # across tile boundaries also proves there are no gutters or padding.
            assert sheet.getpixel(
                (
                    (column + 1) * TILE_SIZE[0] - 1,
                    row * TILE_SIZE[1],
                )
            ) == _color(position)

    _write_renderer_fixture(camera_dir)
    calls.clear()
    compose_head_rendering(camera_dir)
    assert rendering_path.read_bytes() == first_rendering


def test_keep_heads_retains_only_private_stage_not_a_public_sidecar(tmp_path):
    camera_dir = tmp_path / "dragondruit" / "cam07"
    _write_renderer_fixture(camera_dir)
    material_dir = camera_dir / "material" / "olat"
    (material_dir / "rendering.json").write_text("stale sidecar", encoding="utf-8")

    compose_head_rendering(camera_dir, keep_heads=True)

    assert (material_dir / "rendering.png").is_file()
    assert not (material_dir / "rendering.json").exists()
    assert (material_dir / ".rendering_stage" / "render_manifest.json").is_file()


def test_compose_rejects_partial_renderer_output(tmp_path):
    camera_dir = tmp_path / "dragondruit" / "cam07"
    _write_renderer_fixture(camera_dir, omit_position=7)

    with pytest.raises(FileNotFoundError, match="all-or-nothing") as error:
        compose_head_rendering(camera_dir)

    assert "position 8" in str(error.value)
    material_dir = camera_dir / "material" / "olat"
    assert not (material_dir / "rendering.png").exists()
    assert (material_dir / ".rendering_stage").is_dir()


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

    material_dir = camera_dir / "material" / "olat"
    assert not (material_dir / "rendering.png").exists()
    assert (material_dir / ".rendering_stage").is_dir()


@pytest.mark.parametrize(
    ("fixture_kwargs", "message"),
    [
        ({"mismatched_size_position": 5}, "identical native RGB dimensions"),
        ({"non_rgb_position": 5}, "must already be RGB"),
    ],
)
def test_compose_rejects_nonuniform_or_non_rgb_native_tiles(
    tmp_path, fixture_kwargs, message
):
    camera_dir = tmp_path / "dragondruit" / "cam07"
    _write_renderer_fixture(camera_dir, **fixture_kwargs)

    with pytest.raises(ValueError, match=message):
        compose_head_rendering(camera_dir)

    material_dir = camera_dir / "material" / "olat"
    assert not (material_dir / "rendering.png").exists()
    assert (material_dir / ".rendering_stage").is_dir()


def test_failed_pending_validation_preserves_previous_pickup_and_stage(
    tmp_path, monkeypatch
):
    camera_dir = tmp_path / "dragondruit" / "cam07"
    _write_renderer_fixture(camera_dir)
    material_dir = camera_dir / "material" / "olat"
    old_rendering = b"previous validated rendering"
    old_sidecar = b"previous sidecar"
    (material_dir / "rendering.png").write_bytes(old_rendering)
    (material_dir / "rendering.json").write_bytes(old_sidecar)

    def reject_pending(*args, **kwargs):
        raise RuntimeError("synthetic final validation failure")

    monkeypatch.setattr(compositor, "_validate_rendering", reject_pending)

    with pytest.raises(RuntimeError, match="synthetic final validation failure"):
        compose_head_rendering(camera_dir)

    assert (material_dir / "rendering.png").read_bytes() == old_rendering
    assert (material_dir / "rendering.json").read_bytes() == old_sidecar
    assert (material_dir / ".rendering_stage").is_dir()
    assert not (material_dir / ".rendering.png.pending").exists()
