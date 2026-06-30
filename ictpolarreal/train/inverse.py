from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from ictpolarreal.data.dataset import ICTPolarRealDataset
from ictpolarreal.utils.io import write_image


def build_parser(description: str = "Train a compact inverse material decomposition baseline.") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--input", default="polarization")
    parser.add_argument("--target", default="albedo")
    parser.add_argument("--input-mode", choices=["image", "polarization", "gbuffer"], default="polarization")
    parser.add_argument("--target-mode", choices=["image", "polarization", "gbuffer"], default="image")
    parser.add_argument("--material-root", default=None, help="Processed material root written by `bash run.sh process`.")
    parser.add_argument("--light-id", type=int, default=None, help="OLAT light id for polarization-mode training inputs.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--pred-dir", default=None, help="Optional directory for predicted PNGs after training.")
    parser.add_argument("--checkpoint-name", default="baseline_inverse.pt")
    return parser


def run_training(args: argparse.Namespace, *, stage_name: str = "inverse") -> None:
    import torch
    from torch.utils.data import DataLoader

    from ictpolarreal.models.baseline import SmallConvNet

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = ICTPolarRealDataset(
        args.data_root,
        input_name=args.input,
        target_name=args.target,
        input_mode=args.input_mode,
        target_mode=args.target_mode,
        material_root=args.material_root,
        light_id=args.light_id,
        max_samples=args.max_samples,
    )
    if len(dataset) == 0:
        raise SystemExit(f"No training samples found under {args.data_root}")
    first = dataset[0]
    in_channels = int(first["image"].shape[0])
    out_channels = int(first["target"].shape[0])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = SmallConvNet(in_channels=in_channels, out_channels=out_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(
        f"[train:{stage_name}] input_mode={args.input_mode} target_mode={args.target_mode} "
        f"in_channels={in_channels} out_channels={out_channels}"
    )
    step = 0
    pbar = tqdm(total=args.max_steps, desc=f"train {stage_name}")
    while step < args.max_steps:
        for batch in loader:
            image = batch["image"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            pred = model(image)
            loss = (((pred - target) ** 2) * mask).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.5f}")
            if step >= args.max_steps:
                break
    pbar.close()
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "stage": stage_name,
            "in_channels": in_channels,
            "out_channels": out_channels,
        },
        out_dir / args.checkpoint_name,
    )

    if args.pred_dir:
        _write_predictions(model, dataset, Path(args.pred_dir), device)


def _write_predictions(model, dataset: ICTPolarRealDataset, pred_root: Path, device) -> None:
    import torch

    target_slices = dataset.target_slices()
    model.eval()
    with torch.no_grad():
        for idx, sample in enumerate(dataset.samples):
            item = dataset[idx]
            image = item["image"].unsqueeze(0).to(device)
            pred = model(image).squeeze(0).cpu().numpy().transpose(1, 2, 0)
            for name, channel_slice in target_slices:
                write_image(pred_root / sample.object_name / sample.camera / f"{name}.png", pred[..., channel_slice])
    print(f"[train] wrote predictions to {pred_root}")


def main() -> None:
    run_training(build_parser().parse_args(), stage_name="inverse")


if __name__ == "__main__":
    main()
