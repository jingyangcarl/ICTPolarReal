from __future__ import annotations

from ictpolarreal.train.inverse import build_parser, run_training


def main() -> None:
    parser = build_parser("Train a compact forward relighting baseline from material g-buffers.")
    parser.set_defaults(
        input="gbuffer",
        target="static",
        input_mode="gbuffer",
        target_mode="image",
        checkpoint_name="baseline_forward.pt",
    )
    run_training(parser.parse_args(), stage_name="forward")


if __name__ == "__main__":
    main()
