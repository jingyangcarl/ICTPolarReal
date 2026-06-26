from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ictpolarreal.data.dataset import iter_camera_samples
from ictpolarreal.utils.io import read_image
from ictpolarreal.utils.metrics import mae, mse, psnr, ssim_global


@dataclass
class EvalPair:
    dataset: str
    task: str
    object_name: str
    camera: str
    light: str
    target: str
    pred_path: Path
    gt_path: Path
    mask_path: Path | None = None


def _load_manifest(path: str | None) -> list[dict]:
    if not path:
        return []
    with Path(path).open() as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return list(payload.get("samples", []))
    if isinstance(payload, list):
        return payload
    raise ValueError("Manifest must be a JSON list or an object with a `samples` list.")


def _format_template(template: str, **values: object) -> Path:
    return Path(template.format(**values))


def _first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _ictpolarreal_pairs(args: argparse.Namespace) -> tuple[list[EvalPair], int]:
    pairs = []
    skipped = 0
    pred_root = Path(args.pred_root)
    for sample in iter_camera_samples(args.gt_root):
        if args.max_samples is not None and len(pairs) >= args.max_samples:
            break
        if args.task == "decomposition":
            gt_path = sample.image_path(args.target)
            light = ""
            default_pred = pred_root / sample.object_name / sample.camera / f"{args.target}.png"
        else:
            gt_path = sample.light_path(args.gt_kind, args.light_id)
            light = f"{args.light_id:06d}"
            default_pred = pred_root / sample.object_name / sample.camera / light / args.pred_name
        if gt_path is None:
            skipped += 1
            continue
        if args.pred_template:
            pred_path = _format_template(
                args.pred_template,
                pred_root=pred_root,
                object=sample.object_name,
                camera=sample.camera,
                target=args.target,
                light=light,
                frame="",
            )
        else:
            pred_path = default_pred
        if not pred_path.exists():
            skipped += 1
            continue
        pairs.append(
            EvalPair(
                dataset="ictpolarreal",
                task=args.task,
                object_name=sample.object_name,
                camera=sample.camera,
                light=light,
                target=args.target,
                pred_path=pred_path,
                gt_path=gt_path,
                mask_path=sample.image_path("mask"),
            )
        )
    return pairs, skipped


def _objaverse_objects(gt_root: Path, manifest: list[dict]) -> list[dict]:
    if manifest:
        return manifest
    render_root = gt_root / "renderings" if (gt_root / "renderings").exists() else gt_root
    return [{"object": path.name, "frames": [1], "lights": ["all_white"]} for path in sorted(render_root.iterdir()) if path.is_dir()]


def _objaverse_pairs(args: argparse.Namespace) -> tuple[list[EvalPair], int]:
    pairs = []
    skipped = 0
    gt_root = Path(args.gt_root)
    pred_root = Path(args.pred_root)
    render_root = gt_root / "renderings" if (gt_root / "renderings").exists() else gt_root
    for sample in _objaverse_objects(gt_root, _load_manifest(args.manifest)):
        object_name = sample["object"]
        frames = sample.get("frames", [1])
        lights = sample.get("lights", ["all_white"])
        for frame in frames:
            if args.max_samples is not None and len(pairs) >= args.max_samples:
                break
            frame_id = f"{int(frame):04d}"
            object_root = render_root / object_name
            if args.task == "decomposition":
                gt_path = object_root / "gbuffers" / args.target / f"Image{frame_id}.exr"
                light = ""
                pred_candidates = [
                    pred_root / object_name / "gbuffers" / args.target / f"Image{frame_id}.png",
                    pred_root / object_name / args.target / f"Image{frame_id}.png",
                    pred_root / object_name / f"{args.target}.png",
                ]
            else:
                for light_name in lights:
                    gt_path = object_root / "lighting" / light_name / f"frame_{frame_id}.png"
                    if args.pred_template:
                        pred_path = _format_template(
                            args.pred_template,
                            pred_root=pred_root,
                            object=object_name,
                            camera="cam00",
                            target=args.target,
                            light=light_name,
                            frame=frame_id,
                        )
                    else:
                        pred_path = _first_existing(
                            [
                                pred_root / object_name / "lighting" / light_name / f"frame_{frame_id}.png",
                                pred_root / object_name / light_name / f"frame_{frame_id}.png",
                                pred_root / object_name / f"{light_name}.png",
                            ]
                        )
                    if gt_path.exists() and pred_path is not None and pred_path.exists():
                        pairs.append(
                            EvalPair("objaverse", args.task, object_name, "cam00", light_name, args.target, pred_path, gt_path, None)
                        )
                    else:
                        skipped += 1
                continue

            if args.pred_template:
                pred_path = _format_template(
                    args.pred_template,
                    pred_root=pred_root,
                    object=object_name,
                    camera="cam00",
                    target=args.target,
                    light="",
                    frame=frame_id,
                )
            else:
                pred_path = _first_existing(pred_candidates)
            mask_path = object_root / "gbuffers" / "mask" / f"Image{frame_id}.exr"
            if gt_path.exists() and pred_path is not None and pred_path.exists():
                pairs.append(
                    EvalPair(
                        "objaverse",
                        args.task,
                        object_name,
                        "cam00",
                        "",
                        args.target,
                        pred_path,
                        gt_path,
                        mask_path if mask_path.exists() else None,
                    )
                )
            else:
                skipped += 1
    return pairs, skipped


def _resize_like(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.shape[:2] == gt.shape[:2]:
        return pred
    from PIL import Image

    pred_u8 = np.clip(pred, 0.0, 1.0)
    pred_u8 = (pred_u8 * 255.0 + 0.5).astype(np.uint8)
    resized = Image.fromarray(pred_u8).resize((gt.shape[1], gt.shape[0]), Image.BILINEAR)
    return np.asarray(resized).astype(np.float32) / 255.0


def _evaluate_pairs(pairs: list[EvalPair]) -> list[dict]:
    rows = []
    for pair in pairs:
        pred = read_image(pair.pred_path)
        gt = read_image(pair.gt_path)
        pred = _resize_like(pred, gt)
        mask = read_image(pair.mask_path, channels=1) if pair.mask_path else None
        rows.append(
            {
                "dataset": pair.dataset,
                "task": pair.task,
                "object": pair.object_name,
                "camera": pair.camera,
                "light": pair.light,
                "target": pair.target,
                "pred_path": str(pair.pred_path),
                "gt_path": str(pair.gt_path),
                "mse": mse(pred, gt, mask),
                "mae": mae(pred, gt, mask),
                "psnr": psnr(pred, gt, mask),
                "ssim": ssim_global(pred, gt, mask),
            }
        )
    return rows


def _write_outputs(rows: list[dict], out_dir: Path, task: str, skipped: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{task}_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "count": len(rows),
        "skipped": skipped,
        "mse": float(np.mean([row["mse"] for row in rows])),
        "mae": float(np.mean([row["mae"] for row in rows])),
        "psnr": float(np.mean([row["psnr"] for row in rows])),
        "ssim": float(np.mean([row["ssim"] for row in rows])),
    }
    summary_path = out_dir / f"{task}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[eval] wrote {csv_path}")
    print(f"[eval] wrote {summary_path}")
    print(f"[eval] summary: {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ICTPolarReal or Objaverse-style predictions.")
    parser.add_argument("--dataset-mode", choices=["ictpolarreal", "objaverse"], default="ictpolarreal")
    parser.add_argument("--task", choices=["decomposition", "relighting"], default="decomposition")
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target", default="albedo")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--pred-template", default=None, help="Optional Python format string for prediction paths.")
    parser.add_argument("--pred-name", default="pred.png")
    parser.add_argument("--gt-kind", choices=["cross", "parallel"], default="parallel")
    parser.add_argument("--light-id", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    if args.dataset_mode == "ictpolarreal":
        pairs, skipped = _ictpolarreal_pairs(args)
    else:
        pairs, skipped = _objaverse_pairs(args)
    if not pairs:
        raise SystemExit(f"No evaluation pairs found. Skipped candidates: {skipped}. Check --pred-root, --gt-root, and --manifest.")
    rows = _evaluate_pairs(pairs)
    _write_outputs(rows, Path(args.out_dir), args.task, skipped)


if __name__ == "__main__":
    main()
