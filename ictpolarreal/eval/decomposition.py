from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ictpolarreal.data.dataset import iter_camera_samples
from ictpolarreal.utils.io import read_image
from ictpolarreal.utils.metrics import mae, mse, psnr


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted material maps.")
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--target", default="albedo")
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    rows = []
    pred_root = Path(args.pred_root)
    for sample in iter_camera_samples(args.gt_root):
        gt_path = sample.image_path(args.target)
        pred_path = pred_root / sample.object_name / sample.camera / f"{args.target}.png"
        if gt_path is None or not pred_path.exists():
            continue
        mask_path = sample.image_path("mask")
        mask = read_image(mask_path, channels=1) if mask_path else None
        pred = read_image(pred_path)
        gt = read_image(gt_path)
        rows.append(
            {
                "object": sample.object_name,
                "camera": sample.camera,
                "mse": mse(pred, gt, mask),
                "mae": mae(pred, gt, mask),
                "psnr": psnr(pred, gt, mask),
            }
        )

    if not rows:
        raise SystemExit("No prediction/ground-truth pairs found.")
    out_csv = Path(args.out_csv or pred_root / f"eval_{args.target}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()

