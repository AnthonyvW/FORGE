#!/usr/bin/env python3
"""
Generate a pyramidal TIFF from an input image using libvips.

libvips processes the image as a demand-driven pipeline rather than
decoding it fully into RAM, so peak memory stays roughly constant
regardless of image size -- unlike a numpy/tifffile approach, which has
to hold the whole decoded raster (and every pyramid level) resident at
once. This is what vips's own tiffsave does, equivalent to running:

    vips tiffsave INPUT OUTPUT --tile --pyramid --compression deflate

Defaults to deflate: lossless (jpeg's default produced visible blocking
on sharp edges -- it discards data) and, unlike zstd, built directly
into libtiff rather than linked as an optional external codec, so it
works on every libvips build without needing one compiled with zstd
support (not guaranteed -- the prebuilt pyvips wheel on Windows lacks
it, for one).

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

# Matches TILE_SIZE in UI/widgets/preview_overlay/large_image_source.py --
# when they agree, most of that reader's virtual tile requests land on
# exactly one on-disk segment instead of needing several to stitch
# together, which under concurrent load means far fewer separate
# lock-protected file reads to decode the same view.
DEFAULT_TILE_SIZE = 512
DEFAULT_QUALITY = 90


def _report_progress(image: pyvips.Image, progress: pyvips.VipsProgress) -> None:
    print(f"\rGenerating pyramidal TIFF: {progress.percent:3d}%", end="", flush=True)


def write_pyramid_tiff(
    input_path: Path, output_path: Path, tile_size: int, compression: str, quality: int,
) -> None:
    image = pyvips.Image.new_from_file(str(input_path), access="sequential")
    image.set_progress(True)
    image.signal_connect("eval", _report_progress)
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
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a pyramidal TIFF from an input image using libvips")
    parser.add_argument("input", type=Path, help="Path to the source image")
    parser.add_argument("output", type=Path, help="Path to write the pyramidal TIFF")
    parser.add_argument(
        "--tile-size", type=int, default=DEFAULT_TILE_SIZE,
        help="Tile size for every pyramid level (default: %(default)s)",
    )
    parser.add_argument(
        "--compression", default="deflate",
        help="libvips TIFF compression: deflate, lzw, none, zstd (all lossless -- zstd may "
             "not be available on every libvips build), or jpeg (lossy -- smaller files but "
             "visible artifacts on sharp edges) (default: %(default)s)",
    )
    parser.add_argument(
        "--quality", type=int, default=DEFAULT_QUALITY,
        help="JPEG quality, only used with --compression jpeg (default: %(default)s)",
    )
    args = parser.parse_args()

    write_pyramid_tiff(args.input, args.output, args.tile_size, args.compression, args.quality)
    print(f"Wrote pyramidal TIFF to {args.output}")


if __name__ == "__main__":
    main()
