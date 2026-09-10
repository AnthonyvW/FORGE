#!/usr/bin/env python3
"""
Generate a pyramidal TIFF from an input image using libvips.

libvips processes the image as a demand-driven pipeline rather than
decoding it fully into RAM, so peak memory stays roughly constant
regardless of image size -- unlike a numpy/tifffile approach, which has
to hold the whole decoded raster (and every pyramid level) resident at
once. This is what vips's own tiffsave does, equivalent to running:

    vips tiffsave INPUT OUTPUT --tile --pyramid --compression jpeg

Requires libvips itself, not just the pyvips Python binding:
    Debian/Ubuntu: sudo apt install libvips
    macOS:         brew install vips
    pip:           pip install pyvips

Usage:
    python generate_pyramid_tiff.py INPUT_IMAGE OUTPUT.tiff
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyvips

DEFAULT_TILE_SIZE = 256
DEFAULT_QUALITY = 90


def write_pyramid_tiff(
    input_path: Path, output_path: Path, tile_size: int, compression: str, quality: int,
) -> None:
    image = pyvips.Image.new_from_file(str(input_path), access="sequential")
    image.tiffsave(
        str(output_path),
        tile=True,
        tile_width=tile_size,
        tile_height=tile_size,
        pyramid=True,
        compression=compression,
        Q=quality,
        bigtiff=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a pyramidal TIFF from an input image using libvips")
    parser.add_argument("input", type=Path, help="Path to the source image")
    parser.add_argument("output", type=Path, help="Path to write the pyramidal TIFF")
    parser.add_argument(
        "--tile-size", type=int, default=DEFAULT_TILE_SIZE,
        help="Tile size for every pyramid level (default: %(default)s)",
    )
    parser.add_argument(
        "--compression", default="jpeg",
        help="libvips TIFF compression, e.g. jpeg, deflate, lzw, none (default: %(default)s)",
    )
    parser.add_argument(
        "--quality", type=int, default=DEFAULT_QUALITY,
        help="JPEG quality, ignored for other compressions (default: %(default)s)",
    )
    args = parser.parse_args()

    write_pyramid_tiff(args.input, args.output, args.tile_size, args.compression, args.quality)
    print(f"Wrote pyramidal TIFF to {args.output}")


if __name__ == "__main__":
    main()
