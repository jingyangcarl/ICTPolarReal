"""Deprecated compatibility wrapper for :mod:`compose_sd_rendering`."""

from __future__ import annotations

from pathlib import Path

from ictpolarreal.processing.compose_sd_rendering import compose_sd_rendering, main


def compose_head_rendering(
    camera_dir: str | Path,
    *,
    render_dir: str | Path | None = None,
    keep_heads: bool = False,
) -> Path:
    """Call the full SD compositor through its former short-lived API."""

    return compose_sd_rendering(
        camera_dir,
        render_dir=render_dir,
        keep_stage=keep_heads,
    )


if __name__ == "__main__":
    raise SystemExit(main())
