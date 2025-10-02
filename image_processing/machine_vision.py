import time
from collections import defaultdict
from typing import Iterable, Set, Tuple, Dict

import numpy as np
import pygame

from image_processing.analyzers import find_focused_areas


class MachineVision:
    """
    Camera capture + focus analysis + hot-pixel (invalid tile) management.

    Responsibilities pulled out of FocusOverlay:
      - frame capture -> NumPy BGR
      - focused tile computation (hard/soft bands, thresholds)
      - invalid tile map build/clear/query (hot-pixel map)
      - tile index/rect helpers & filtering by invalid map
    """

    def __init__(
        self,
        camera,
        *,
        tile_size: int = 48,
        stride: int = 48,
        top_percent: float = 0.15,
        min_score: float | None = None,
        soft_min_score: float | None = None,
    ):
        self.camera = camera

        # Analysis parameters
        self.tile_size = int(tile_size)
        self.stride = int(stride)
        self.top_percent = float(top_percent)
        self.min_score = min_score
        self.soft_min_score = soft_min_score

        # Hot-pixel / invalid tiles as grid indices
        self._invalid_tiles: Set[Tuple[int, int]] = set()

    # --------------- public properties ---------------
    @property
    def invalid_tiles(self) -> Set[Tuple[int, int]]:
        return self._invalid_tiles

    # --------------- frame I/O ---------------
    def capture_current_frame_bgr(self) -> np.ndarray | None:
        """
        Grab the current camera frame and return it as a NumPy BGR array (H,W,3).
        Returns None if no frame is available.
        """
        frame_surface = self.camera.get_frame()
        if not frame_surface:
            return None

        pg_pixels = pygame.surfarray.array3d(frame_surface)  # (W,H,3) RGB
        img_rgb = np.transpose(pg_pixels, (1, 0, 2))         # (H,W,3)
        img_bgr = img_rgb[:, :, ::-1]                        # RGB->BGR
        return img_bgr

    # --------------- tile helpers ---------------
    def tile_index_from_xy(self, x: int, y: int) -> Tuple[int, int]:
        """Top-left pixel (overlay space) → (col,row) index using stride."""
        col = max(0, int(x) // self.stride)
        row = max(0, int(y) // self.stride)
        return (col, row)

    def tile_rect_from_index(self, col: int, row: int, w: int | None = None, h: int | None = None) -> pygame.Rect:
        """(col,row) grid index → pygame.Rect in overlay space."""
        rw = int(w if w is not None else self.tile_size)
        rh = int(h if h is not None else self.tile_size)
        rx = int(col * self.stride)
        ry = int(row * self.stride)
        return pygame.Rect(rx, ry, rw, rh)

    # --------------- invalid map ---------------
    def clear_hot_pixel_map(self) -> None:
        self._invalid_tiles.clear()

    def is_tile_invalid(self, col: int, row: int) -> bool:
        return (col, row) in self._invalid_tiles

    def filter_tiles(self, tiles: Iterable) -> list:
        """
        Drop any tiles whose top-left falls on an invalid (col,row).
        Assumes tiles have x,y members in overlay coordinates.
        """
        out = []
        for t in tiles:
            col, row = self.tile_index_from_xy(int(t.x), int(t.y))
            if (col, row) not in self._invalid_tiles:
                out.append(t)
        return out

    # --------------- focused tile computation ---------------
    def compute_focused_tiles(
        self,
        *,
        include_soft: bool = True,
        filter_invalid: bool = True,
    ) -> dict:
        """
        Returns a dict with lists of tiles:
        {
          "all":  [...],
          "hard": [...],  # >= min_score
          "soft": [...],  # soft_min_score <= score < min_score (if enabled)
        }
        """
        img_bgr = self.capture_current_frame_bgr()
        if img_bgr is None:
            return {"all": [], "hard": [], "soft": []}

        tiles_all = find_focused_areas(
            img_bgr,
            tile_size=self.tile_size,
            stride=self.stride,
            top_percent=self.top_percent,
            min_score=self.min_score,
            soft_min_score=(self.soft_min_score if include_soft else None),
        ) or []

        # Split soft/hard by score (if soft band enabled)
        soft_tiles, hard_tiles = [], []
        if include_soft and self.soft_min_score is not None:
            hard_cut = self.min_score if self.min_score is not None else float("inf")
            for t in tiles_all:
                s = getattr(t, "score", None)
                if s is not None and self.soft_min_score <= s < hard_cut:
                    soft_tiles.append(t)
                else:
                    hard_tiles.append(t)
        else:
            hard_tiles = tiles_all

        if filter_invalid:
            soft_tiles = self.filter_tiles(soft_tiles)
            hard_tiles = self.filter_tiles(hard_tiles)

        return {
            "all":  (hard_tiles + soft_tiles) if include_soft else hard_tiles,
            "hard": hard_tiles,
            "soft": soft_tiles,
        }

    def get_in_focus_tiles(
        self,
        *,
        band: str = "all",       # "all" | "hard" | "soft"
        as_rects: bool = False   # pygame.Rects in overlay coords
    ):
        result = self.compute_focused_tiles(include_soft=(band in ("all", "soft")), filter_invalid=True)
        tiles = result.get(band, result["all"]) if band in ("all", "hard", "soft") else result["all"]

        if not as_rects:
            return tiles

        rects = []
        for t in tiles:
            rects.append(pygame.Rect(int(t.x), int(t.y), int(t.w), int(t.h)))
        return rects

    # --------------- hot-pixel sampler ---------------
    def build_hot_pixel_map(
        self,
        duration_sec: float = 1.0,
        *,
        dilate: int = 0,
        min_hits: int = 1,
        max_fps: int = 30,
        include_soft: bool = True,
    ) -> Dict[str, int]:
        """
        Sample the camera with the lens cap on. Any tile that appears 'in focus'
        ≥ min_hits times during duration_sec is marked INVALID (hot).
        """
        t_end = time.monotonic() + max(0.05, float(duration_sec))
        min_hits = max(1, int(min_hits))
        stride = int(self.stride)

        hits: Dict[Tuple[int, int], int] = defaultdict(int)

        # Simple FPS throttle
        min_dt = 1.0 / float(max(1, int(max_fps)))
        t_next = 0.0

        frames = 0
        while time.monotonic() < t_end:
            now = time.monotonic()
            if now < t_next:
                time.sleep(max(0.0, t_next - now))
            t_next = time.monotonic() + min_dt

            res = self.compute_focused_tiles(include_soft=include_soft, filter_invalid=False)
            for t in res["all"]:
                col = int(t.x) // stride
                row = int(t.y) // stride
                hits[(col, row)] += 1

            frames += 1

        candidates: Set[Tuple[int, int]] = {ij for ij, n in hits.items() if n >= min_hits}

        if dilate > 0 and candidates:
            expanded: Set[Tuple[int, int]] = set()
            for (c, r) in candidates:
                for dc in range(-dilate, dilate + 1):
                    for dr in range(-dilate, dilate + 1):
                        expanded.add((c + dc, r + dr))
            candidates = expanded

        before = len(self._invalid_tiles)
        self._invalid_tiles |= candidates
        after = len(self._invalid_tiles)

        return {
            "frames": frames,
            "candidates": len(candidates),
            "newly_marked": after - before,
            "total_invalid": after,
        }
