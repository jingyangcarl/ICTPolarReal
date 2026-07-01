from __future__ import annotations

import argparse

from ictpolarreal.train.diffusion import add_training_arguments, run_diffusion_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune RGB2X for ICTPolarReal forward rendering from PBR or polarization conditions."
    )
    return add_training_arguments(parser, stage="forward")


def main() -> None:
    run_diffusion_training(build_parser().parse_args(), stage="forward")


if __name__ == "__main__":
    main()
