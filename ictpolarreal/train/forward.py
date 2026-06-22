from __future__ import annotations

import argparse

from ictpolarreal.train.inverse import main as inverse_main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forward relighting baseline. Use --input to choose a prepared material image and --target for a relit image."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--input", default="albedo")
    parser.add_argument("--target", default="static")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    args, unknown = parser.parse_known_args()
    import sys

    sys.argv = [
        sys.argv[0],
        "--data-root",
        args.data_root,
        "--out-dir",
        args.out_dir,
        "--input",
        args.input,
        "--target",
        args.target,
        "--batch-size",
        str(args.batch_size),
        "--max-steps",
        str(args.max_steps),
        "--lr",
        str(args.lr),
    ]
    if args.max_samples is not None:
        sys.argv.extend(["--max-samples", str(args.max_samples)])
    sys.argv.extend(unknown)
    inverse_main()


if __name__ == "__main__":
    main()

