from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from ictpolarreal.processing import compose_sd_rendering as compositor


NATIVE_SIZE = (48, 96)  # width, height; deliberately portrait like ICT cam07
TEST_TILE_SIZE = 128


def _color(index: int) -> tuple[int, int, int]:
    return (
        1 + (17 * (index + 1)) % 254,
        1 + (53 * (index + 1)) % 254,
        1 + (97 * (index + 1)) % 254,
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
        "safe_condition_id": condition_id,
        "label_number": compositor.CONDITION_LABEL_NUMBERS[position],
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


def _camera_relative(camera_dir: Path, path: Path) -> str:
    return path.relative_to(camera_dir).as_posix()


def _write_rgb(path: Path, color: tuple[int, int, int], *, mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = color if mode == "RGB" else color[0]
    Image.new(mode, NATIVE_SIZE, value).save(path)


def _write_renderer_fixture(camera_dir: Path) -> dict[str, tuple[int, int, int]]:
    material_dir = camera_dir / "material" / "olat"
    stage_dir = camera_dir / ".rendering_stage"
    maps_dir = stage_dir / "material_maps"
    conditions_dir = stage_dir / "conditions"
    maps_dir.mkdir(parents=True, exist_ok=True)
    conditions_dir.mkdir(parents=True, exist_ok=True)

    expected_colors: dict[str, tuple[int, int, int]] = {}
    material_maps = []
    for position, name in enumerate(compositor.MATERIAL_MAP_ORDER):
        color = _color(position)
        path = maps_dir / f"{name}.png"
        _write_rgb(path, color)
        expected_colors[name] = color
        material_maps.append(
            {
                "name": name,
                "render_path": _camera_relative(camera_dir, path),
                "render_sha256": _sha256(path),
            }
        )

    rendered_conditions = []
    color_index = len(compositor.MATERIAL_MAP_ORDER)
    for position in range(compositor.EXPECTED_CONDITION_COUNT):
        record = _preset_condition(position)
        panel_dir = conditions_dir / record["safe_condition_id"]
        panels = {}
        for panel_name in compositor.PANEL_ORDER:
            color = _color(color_index)
            color_index += 1
            path = panel_dir / f"{panel_name}.png"
            _write_rgb(path, color)
            label = f"{panel_name} #{record['label_number']}"
            expected_colors[label] = color
            panels[panel_name] = {
                "render_path": _camera_relative(camera_dir, path),
                "render_sha256": _sha256(path),
            }
        record["panels"] = panels
        rendered_conditions.append(record)

    material_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = material_dir / "disney_brdf.pt"
    checkpoint_path.write_bytes(b"synthetic fixed-post-fit Disney state")

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
        "schema": "ictpolarreal.sd-renderings-preset.v1",
        "trainer_source": {
            "path": "/imaginaire/trainers/relighting_switchlight_pretrain.py",
            "sha256": _digest("trainer source"),
        },
    }

    raw_targets_digest = _digest("raw measured parallel OLAT targets")
    raw_parallel_check = {
        "actual_sha256": raw_targets_digest,
        "expected_sha256": raw_targets_digest,
        "passed": True,
    }
    mask_path = camera_dir / "clean_mask.png"
    mask_u8 = np.asarray([[0, 64], [192, 255]], dtype=np.uint8)
    Image.fromarray(mask_u8, mode="L").save(mask_path)
    presentation_alpha = np.ascontiguousarray(
        (mask_u8.astype(np.float32) / 255.0)[..., None]
    )
    capture_foreground = np.ascontiguousarray(
        (presentation_alpha > 0.5).astype(np.float32)
    )
    fit_foreground = np.zeros_like(capture_foreground)
    fit_foreground[1, 1, 0] = 1.0
    capture_digest = compositor._array_sha256(capture_foreground)
    fit_digest = compositor._array_sha256(fit_foreground)
    manifest = {
        "conditions": rendered_conditions,
        "input_hash_checks": {
            "capture_foreground": {
                "actual_sha256": capture_digest,
                "expected_sha256": capture_digest,
                "passed": True,
            },
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
            "foreground": {
                "actual_sha256": fit_digest,
                "expected_sha256": fit_digest,
                "passed": True,
            },
            "raw_parallel_targets": raw_parallel_check,
        },
        "lighting_preset": "sd-renderings",
        "material": {
            "path": "material/olat/disney_brdf.pt",
            "sha256": _sha256(checkpoint_path),
        },
        "material_maps": material_maps,
        "preset_provenance": preset_provenance,
        "profile": "olat",
        "presentation_mask": {
            "schema": compositor.PRESENTATION_MASK_SCHEMA,
            "fit_rule": compositor.FIT_MASK_RULE,
            "fit_foreground_sha256": fit_digest,
            "presentation_rule": compositor.PRESENTATION_MASK_RULE,
            "presentation_normal_rule": compositor.PRESENTATION_NORMAL_RULE,
            "presentation_mask_path": str(mask_path.resolve()),
            "presentation_mask_file_sha256": _sha256(mask_path),
            "presentation_alpha_sha256": compositor._array_sha256(
                presentation_alpha
            ),
            "capture_foreground_sha256": capture_digest,
            "capture_foreground_pixels": 2,
            "fit_foreground_pixels": 1,
            "restored_foreground_pixels": 1,
            "fractional_alpha_pixels": 2,
            "faceforwarded_foreground_pixels": 1,
        },
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
        "schema": compositor.RENDER_MANIFEST_SCHEMA,
        "selected_absolute_indices": None,
        "sheet_contract": compositor._expected_sheet_contract(raw_parallel_check),
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
                ".rendering_stage/validation/recorded_validation_row_428.png"
            ),
            "render_sha256": validation_sha256,
            "tolerance": 0.0,
        },
    }
    (stage_dir / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return expected_colors


def _load_manifest(camera_dir: Path) -> dict:
    return json.loads(
        (camera_dir / ".rendering_stage" / "render_manifest.json").read_text(
            encoding="utf-8"
        )
    )


def _save_manifest(camera_dir: Path, manifest: dict) -> None:
    (camera_dir / ".rendering_stage" / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def small_sheet(monkeypatch):
    monkeypatch.setattr(compositor, "SHEET_TILE_SIZE", TEST_TILE_SIZE)


def test_production_contract_is_the_identified_reference():
    assert compositor.RENDER_MANIFEST_SCHEMA == "ictpolarreal.saved-disney-render.v3"
    assert compositor.SHEET_COLUMNS == 12
    assert compositor.SHEET_ROWS == 13
    assert compositor.SHEET_TILE_SIZE == 768
    assert compositor.EXPECTED_SHEET_TILE_COUNT == 156
    assert 12 * compositor.SHEET_TILE_SIZE == 9216
    assert 13 * compositor.SHEET_TILE_SIZE == 9984
    assert compositor.CONDITION_LABEL_NUMBERS == tuple(range(1, 142, 4))
    assert compositor.MATERIAL_MAP_ORDER == (
        "normal",
        "baseColor",
        "metallic",
        "roughness",
        "specular",
        "specularTint",
        "subsurface",
        "anisotropic",
        "sheen",
        "sheenTint",
        "clearcoat",
        "clearcoatGloss",
    )
    assert compositor.PANEL_ORDER == ("gt", "pred", "greyball", "chromeball")
    contract = compositor._expected_sheet_contract()
    assert contract["reference_sha256"] == compositor.REFERENCE_RENDERINGS_SHA256
    assert contract["historical_commit"] == "877b6f0732cdadd55ed621771d327a574a077137"


def test_square_fit_contains_portrait_without_crop_or_stretch():
    color = np.asarray([31, 97, 211], dtype=np.uint8)
    source = np.broadcast_to(color, (512, 282, 3)).copy()

    fitted = compositor._fit_square_rgb(source, tile_size=768)

    # 512 -> 768 is a 1.5x uniform scale; 282 -> 423, centered as 172/173.
    assert fitted.shape == (768, 768, 3)
    np.testing.assert_array_equal(fitted[:, :172], 0)
    np.testing.assert_array_equal(fitted[:, 595:], 0)
    np.testing.assert_array_equal(fitted[0, 172], color)
    np.testing.assert_array_equal(fitted[-1, 594], color)


def test_historical_label_uses_exact_768px_metrics(monkeypatch):
    calls = []
    original_put_text = cv2.putText

    def record_put_text(image, text, origin, font, scale, color, thickness, line):
        calls.append((text, origin, font, scale, color, thickness, line))
        return original_put_text(
            image, text, origin, font, scale, color, thickness, line
        )

    monkeypatch.setattr(cv2, "putText", record_put_text)
    compositor._overlay_historical_label(
        np.zeros((768, 768, 3), dtype=np.uint8), "chromeball #141"
    )

    assert calls == [
        (
            "chromeball #141",
            (4, 764),
            cv2.FONT_HERSHEY_SIMPLEX,
            61 / 30.0,
            (255, 255, 0),
            4,
            cv2.LINE_AA,
        )
    ]


def test_compose_writes_full_gapless_sheet_in_exact_order(
    tmp_path, monkeypatch, small_sheet
):
    camera_dir = tmp_path / "dragondruit" / "cam07"
    expected_colors = _write_renderer_fixture(camera_dir)
    material_root = camera_dir / "material"
    olat_dir = material_root / "olat"
    (camera_dir / "rendering.json").write_text("obsolete", encoding="utf-8")
    (material_root / "rendering.png").write_bytes(b"obsolete material pickup")
    (material_root / "rendering.json").write_text("obsolete", encoding="utf-8")
    (olat_dir / "rendering.png").write_bytes(b"obsolete OLAT pickup")
    (olat_dir / "rendering.json").write_text("obsolete", encoding="utf-8")

    calls = []
    original_put_text = cv2.putText

    def record_put_text(image, text, origin, font, scale, color, thickness, line):
        calls.append((text, origin, font, scale, color, thickness, line))
        return original_put_text(
            image, text, origin, font, scale, color, thickness, line
        )

    monkeypatch.setattr(cv2, "putText", record_put_text)
    rendering_path = compositor.compose_sd_rendering(camera_dir)
    first_rendering = rendering_path.read_bytes()

    expected_labels = list(compositor.MATERIAL_MAP_ORDER)
    expected_labels.extend(
        f"{panel} #{number}"
        for number in compositor.CONDITION_LABEL_NUMBERS
        for panel in compositor.PANEL_ORDER
    )
    assert [call[0] for call in calls] == expected_labels
    font_size = int(TEST_TILE_SIZE * compositor.LABEL_FONT_SIZE_RATIO)
    assert all(call[1] == (4, TEST_TILE_SIZE - 4) for call in calls)
    assert all(call[2] == cv2.FONT_HERSHEY_SIMPLEX for call in calls)
    assert all(call[3] == font_size / 30.0 for call in calls)
    assert all(call[4] == (255, 255, 0) for call in calls)
    assert all(call[5] == max(1, font_size // 15) for call in calls)
    assert all(call[6] == cv2.LINE_AA for call in calls)

    assert rendering_path == camera_dir / "rendering.png"
    assert rendering_path.is_file()
    assert not (camera_dir / ".rendering_stage").exists()
    assert not (camera_dir / ".rendering.png.pending").exists()
    assert not (camera_dir / ".rendering.png.previous").exists()
    assert not (camera_dir / "rendering.json").exists()
    assert not (material_root / "rendering.png").exists()
    assert not (material_root / "rendering.json").exists()
    assert not (olat_dir / "rendering.png").exists()
    assert not (olat_dir / "rendering.json").exists()

    with Image.open(rendering_path) as sheet:
        assert sheet.format == "PNG"
        assert sheet.mode == "RGB"
        assert sheet.size == (
            compositor.SHEET_COLUMNS * TEST_TILE_SIZE,
            compositor.SHEET_ROWS * TEST_TILE_SIZE,
        )
        # Sources are 1:2 portrait, so their fit is 64x128 with 32px black bars.
        for position, label in enumerate(expected_labels):
            row, column = divmod(position, compositor.SHEET_COLUMNS)
            y = row * TEST_TILE_SIZE
            x = column * TEST_TILE_SIZE
            assert sheet.getpixel((x + 31, y)) == (0, 0, 0)
            assert sheet.getpixel((x + 32, y)) == expected_colors[label]
            assert sheet.getpixel((x + 95, y)) == expected_colors[label]
            assert sheet.getpixel((x + 96, y)) == (0, 0, 0)

    _write_renderer_fixture(camera_dir)
    calls.clear()
    compositor.compose_sd_rendering(camera_dir)
    assert rendering_path.read_bytes() == first_rendering


def test_keep_stage_retains_only_private_inputs(tmp_path, small_sheet):
    camera_dir = tmp_path / "dragondruit" / "cam07"
    _write_renderer_fixture(camera_dir)

    compositor.compose_sd_rendering(camera_dir, keep_stage=True)

    assert (camera_dir / "rendering.png").is_file()
    assert (camera_dir / ".rendering_stage" / "render_manifest.json").is_file()
    assert not (camera_dir / "rendering.json").exists()


def test_missing_panel_preserves_previous_pickup_stage_and_legacy_files(
    tmp_path, small_sheet
):
    camera_dir = tmp_path / "dragondruit" / "cam07"
    _write_renderer_fixture(camera_dir)
    manifest = _load_manifest(camera_dir)
    missing_record = manifest["conditions"][7]["panels"]["chromeball"]
    missing_path = camera_dir / missing_record["render_path"]
    missing_path.unlink()
    old_rendering = b"previous validated camera-level rendering"
    (camera_dir / "rendering.png").write_bytes(old_rendering)
    legacy = camera_dir / "material" / "olat" / "rendering.png"
    legacy.write_bytes(b"legacy remains until success")

    with pytest.raises(FileNotFoundError, match="all-or-nothing"):
        compositor.compose_sd_rendering(camera_dir)

    assert (camera_dir / "rendering.png").read_bytes() == old_rendering
    assert legacy.read_bytes() == b"legacy remains until success"
    assert (camera_dir / ".rendering_stage").is_dir()


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("reference_sha256", "0" * 64),
        ("historical_commit", "deadbeef" * 5),
        ("columns", 3),
        ("rows", 12),
        ("tile_size", 512),
        ("map_order", ["baseColor", "normal"]),
        ("panel_order", ["pred", "gt", "greyball", "chromeball"]),
    ],
)
def test_compose_rejects_sheet_contract_drift(
    tmp_path, small_sheet, key, bad_value
):
    camera_dir = tmp_path / "dragondruit" / "cam07"
    _write_renderer_fixture(camera_dir)
    manifest = _load_manifest(camera_dir)
    manifest["sheet_contract"][key] = bad_value
    _save_manifest(camera_dir, manifest)

    with pytest.raises(ValueError, match=f"sheet_contract.{key}"):
        compositor.compose_sd_rendering(camera_dir)

    assert not (camera_dir / "rendering.png").exists()
    assert (camera_dir / ".rendering_stage").is_dir()


def test_compose_rejects_extra_sheet_contract_key(tmp_path, small_sheet):
    camera_dir = tmp_path / "dragondruit" / "cam07"
    _write_renderer_fixture(camera_dir)
    manifest = _load_manifest(camera_dir)
    manifest["sheet_contract"]["invented"] = True
    _save_manifest(camera_dir, manifest)

    with pytest.raises(ValueError, match="keys differ"):
        compositor.compose_sd_rendering(camera_dir)


@pytest.mark.parametrize("kind", ["maps", "panels"])
def test_compose_rejects_reordered_inputs(tmp_path, small_sheet, kind):
    camera_dir = tmp_path / "dragondruit" / "cam07"
    _write_renderer_fixture(camera_dir)
    manifest = _load_manifest(camera_dir)
    if kind == "maps":
        manifest["material_maps"][0], manifest["material_maps"][1] = (
            manifest["material_maps"][1],
            manifest["material_maps"][0],
        )
        message = "historical map order"
    else:
        panels = manifest["conditions"][0]["panels"]
        manifest["conditions"][0]["panels"] = {
            "pred": panels["pred"],
            "gt": panels["gt"],
            "greyball": panels["greyball"],
            "chromeball": panels["chromeball"],
        }
        message = "panels must preserve order"
    _save_manifest(camera_dir, manifest)

    with pytest.raises(ValueError, match=message):
        compositor.compose_sd_rendering(camera_dir)


def test_compose_rejects_wrong_label_number_and_panel_hash(tmp_path, small_sheet):
    camera_dir = tmp_path / "dragondruit" / "cam07"
    _write_renderer_fixture(camera_dir)
    manifest = _load_manifest(camera_dir)
    manifest["conditions"][4]["label_number"] = 999
    _save_manifest(camera_dir, manifest)
    with pytest.raises(ValueError, match="label_number"):
        compositor.compose_sd_rendering(camera_dir)

    manifest = _load_manifest(camera_dir)
    manifest["conditions"][4]["label_number"] = compositor.CONDITION_LABEL_NUMBERS[4]
    manifest["conditions"][4]["panels"]["pred"]["render_sha256"] = "f" * 64
    _save_manifest(camera_dir, manifest)
    with pytest.raises(ValueError, match="hash does not match"):
        compositor.compose_sd_rendering(camera_dir)


def test_compose_requires_passing_raw_parallel_hash(tmp_path, small_sheet):
    camera_dir = tmp_path / "dragondruit" / "cam07"
    _write_renderer_fixture(camera_dir)
    manifest = _load_manifest(camera_dir)
    manifest["input_hash_checks"].pop("raw_parallel_targets")
    _save_manifest(camera_dir, manifest)

    with pytest.raises(ValueError, match="raw_parallel_targets is required"):
        compositor.compose_sd_rendering(camera_dir)


def test_compose_rejects_hash_mismatch_even_when_marked_passing(
    tmp_path, small_sheet
):
    camera_dir = tmp_path / "dragondruit" / "cam07"
    _write_renderer_fixture(camera_dir)
    manifest = _load_manifest(camera_dir)
    check = manifest["input_hash_checks"]["condition_weights"]
    check["actual_sha256"] = "f" * 64
    check["passed"] = True
    _save_manifest(camera_dir, manifest)

    with pytest.raises(ValueError, match="input hash check failed"):
        compositor.compose_sd_rendering(camera_dir)


def test_compose_rejects_fit_mask_as_presentation_policy(tmp_path, small_sheet):
    camera_dir = tmp_path / "dragondruit" / "cam07"
    _write_renderer_fixture(camera_dir)
    manifest = _load_manifest(camera_dir)
    manifest["presentation_mask"]["presentation_rule"] = (
        "incorrectly_reuse_fit_foreground"
    )
    _save_manifest(camera_dir, manifest)

    with pytest.raises(ValueError, match="wrong presentation rule"):
        compositor.compose_sd_rendering(camera_dir)


def test_compose_rejects_changed_clean_mask_source(tmp_path, small_sheet):
    camera_dir = tmp_path / "dragondruit" / "cam07"
    _write_renderer_fixture(camera_dir)
    manifest = _load_manifest(camera_dir)
    mask_path = Path(manifest["presentation_mask"]["presentation_mask_path"])
    Image.new("L", (2, 2), 255).save(mask_path)

    with pytest.raises(ValueError, match="source does not match provenance"):
        compositor.compose_sd_rendering(camera_dir)


def test_compose_rejects_non_rgb_source(tmp_path, small_sheet):
    camera_dir = tmp_path / "dragondruit" / "cam07"
    _write_renderer_fixture(camera_dir)
    manifest = _load_manifest(camera_dir)
    record = manifest["material_maps"][3]
    path = camera_dir / record["render_path"]
    _write_rgb(path, (93, 93, 93), mode="L")
    record["render_sha256"] = _sha256(path)
    _save_manifest(camera_dir, manifest)

    with pytest.raises(ValueError, match="must already be RGB"):
        compositor.compose_sd_rendering(camera_dir)

    assert (camera_dir / ".rendering_stage").is_dir()


def test_pending_validation_failure_preserves_previous_output(tmp_path, monkeypatch, small_sheet):
    camera_dir = tmp_path / "dragondruit" / "cam07"
    _write_renderer_fixture(camera_dir)
    old_rendering = b"previous validated camera-level rendering"
    (camera_dir / "rendering.png").write_bytes(old_rendering)

    def reject_pending(*args, **kwargs):
        raise RuntimeError("synthetic pending validation failure")

    monkeypatch.setattr(compositor, "_validate_rendering", reject_pending)
    with pytest.raises(RuntimeError, match="pending validation failure"):
        compositor.compose_sd_rendering(camera_dir)

    assert (camera_dir / "rendering.png").read_bytes() == old_rendering
    assert (camera_dir / ".rendering_stage").is_dir()
    assert not (camera_dir / ".rendering.png.pending").exists()
    assert not (camera_dir / ".rendering.png.previous").exists()


def test_final_validation_failure_restores_previous_output(tmp_path, monkeypatch, small_sheet):
    camera_dir = tmp_path / "dragondruit" / "cam07"
    _write_renderer_fixture(camera_dir)
    old_rendering = b"previous validated camera-level rendering"
    (camera_dir / "rendering.png").write_bytes(old_rendering)
    original_validate = compositor._validate_rendering
    calls = 0

    def reject_final(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic final validation failure")
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(compositor, "_validate_rendering", reject_final)
    with pytest.raises(RuntimeError, match="final validation failure"):
        compositor.compose_sd_rendering(camera_dir)

    assert calls == 2
    assert (camera_dir / "rendering.png").read_bytes() == old_rendering
    assert (camera_dir / ".rendering_stage").is_dir()
    assert not (camera_dir / ".rendering.png.pending").exists()
    assert not (camera_dir / ".rendering.png.previous").exists()


def test_cli_names_full_sheet_and_keep_stage():
    parser = compositor._build_parser()
    args = parser.parse_args(["--camera-dir", "/camera", "--keep-stage"])

    assert args.keep_stage is True
    assert "full historical SD material-rendering sheet" in parser.description
