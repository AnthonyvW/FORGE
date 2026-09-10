#!/usr/bin/env python3
"""
Generate a pyramidal TIFF from an input image, using tifffile's SubIFD
pyramid layout -- the same layout main.py's PyramidTiffBackend and
UI/widgets/preview_overlay/large_image_source.py's _PyramidTiffBackend
already know how to read (series.is_pyramidal / series.levels).

Usage:
    python generate_pyramid_tiff.py INPUT_IMAGE OUTPUT.tiff
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

DEFAULT_TILE_SIZE = 512
DEFAULT_MIN_LEVEL_SIZE = 512


def build_levels(image: Image.Image, min_level_size: int) -> list[np.ndarray]:
    if image.mode != "RGB":
        image = image.convert("RGB")
    levels = [np.asarray(image)]
    while max(levels[-1].shape[:2]) > min_level_size:
        previous = Image.fromarray(levels[-1])
        next_size = (max(1, previous.width // 2), max(1, previous.height // 2))
        levels.append(np.asarray(previous.resize(next_size, Image.Resampling.LANCZOS)))
    return levels


def write_pyramid_tiff(levels: list[np.ndarray], output_path: Path, tile_size: int, compression: str) -> None:
    options = dict(photometric="rgb", tile=(tile_size, tile_size), compression=compression)
    with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
        tif.write(levels[0], subifds=len(levels) - 1, **options)
        for level in levels[1:]:
            tif.write(level, subfiletype=1, **options)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a pyramidal TIFF from an input image")
    parser.add_argument("input", type=Path, help="Path to the source image")
    parser.add_argument("output", type=Path, help="Path to write the pyramidal TIFF")
    parser.add_argument(
        "--tile-size", type=int, default=DEFAULT_TILE_SIZE,
        help="Tile size for every pyramid level, must be a multiple of 16 (default: %(default)s)",
    )
    parser.add_argument(
        "--min-level-size", type=int, default=DEFAULT_MIN_LEVEL_SIZE,
        help="Stop halving once the coarsest level's longest side is at or below this size (default: %(default)s)",
    )
    parser.add_argument(
        "--compression", default="deflate",
        help="tifffile compression codec, e.g. deflate, lzw, none. 'jpeg' gives smaller files "
             "but requires the imagecodecs package (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.tile_size % 16 != 0:
        parser.error("--tile-size must be a multiple of 16")

    image = Image.open(args.input)
    levels = build_levels(image, args.min_level_size)
    write_pyramid_tiff(levels, args.output, args.tile_size, args.compression)

    native_h, native_w = levels[0].shape[:2]
    print(f"Wrote {len(levels)} level(s), {native_w}x{native_h} native, to {args.output}")


if __name__ == "__main__":
    main()
