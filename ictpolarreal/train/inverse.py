from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from ictpolarreal.data.dataset import ICTPolarRealDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact inverse decomposition baseline.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--input", default="static")
    parser.add_argument("--target", default="albedo")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from ictpolarreal.models.baseline import SmallConvNet

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ICTPolarRealDataset(
        args.data_root,
        input_name=args.input,
        target_name=args.target,
        max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = SmallConvNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    pbar = tqdm(total=args.max_steps, desc="train")
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
    torch.save({"model": model.state_dict(), "args": vars(args)}, out_dir / "baseline_inverse.pt")


if __name__ == "__main__":
    main()
