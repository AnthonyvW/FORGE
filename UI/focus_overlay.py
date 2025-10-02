import pygame
import numpy as np
import time
from collections import defaultdict
from typing import Iterable, Set, Tuple, Dict

from UI.frame import Frame
from UI.text import Text, TextStyle

from image_processing.analyzers import find_focused_areas
from UI.camera_view import CameraView



class FocusOverlay(Frame):
    def __init__(
        self,
        camera_view: "CameraView",
        *,
        visible = True,
        tile_size: int = 48,
        stride: int = 48,
        top_percent: float = 0.15,
        alpha_hard: int = 100,
        border_alpha_hard: int = 200,
        alpha_soft: int = 50,
        border_alpha_soft: int = 120,
        min_score: float | None = None,        # NEW
        soft_min_score: float | None = None,   # NEW
    ):
        super().__init__(parent=camera_view, x=0, y=0, width=1, height=1,
                         x_is_percent=True, y_is_percent=True,
                         width_is_percent=True, height_is_percent=True,
                         z_index=camera_view.z_index + 1,
                         background_color=None)
        self.camera_view = camera_view
        self.tile_size = tile_size
        self.stride = stride
        self.top_percent = top_percent
        self.alpha_hard = alpha_hard
        self.border_alpha_hard = border_alpha_hard
        self.alpha_soft = alpha_soft
        self.border_alpha_soft = border_alpha_soft
        self.min_score = min_score
        self.soft_min_score = soft_min_score

        # Hot-pixel map storage (tile indices as (col, row))
        self._invalid_tiles: Set[Tuple[int, int]] = set()
        self.draw_invalid = True

        # Red style for invalid tiles
        self.invalid_fill = pygame.Color(255, 0, 0, 90)      # translucent red
        self.invalid_border = pygame.Color(255, 0, 0, 180)   # darker border
        self.invalid_border_w = 2

        self.soft_fill = pygame.Color(255, 180, 0, self.alpha_soft)
        self.soft_border = pygame.Color(255, 180, 0, self.border_alpha_soft)
        self.soft_border_w = 1

        self.hard_fill = pygame.Color(0, 200, 255, self.alpha_hard)
        self.hard_border = pygame.Color(0, 200, 255, self.border_alpha_hard)
        self.hard_border_w = 2

        self._frame_counter = 0
        self._cached_tiles = None
        self.visible = visible
        
    def capture_current_frame_bgr(self):
        """
        Grab the current camera frame and return it as a NumPy BGR array (H,W,3).
        Returns None if no frame is available.
        """
        frame_surface = self.camera_view.camera.get_frame()
        if not frame_surface:
            return None
        pg_pixels = pygame.surfarray.array3d(frame_surface)   # (W,H,3) RGB
        img_rgb = np.transpose(pg_pixels, (1, 0, 2))          # (H,W,3)
        img_bgr = img_rgb[:, :, ::-1]                         # BGR
        return img_bgr
    
    
    def compute_focused_tiles(
        self,
        *,
        include_soft: bool = True,
        filter_invalid: bool = True,
    ) -> dict:
        """
        Compute focused tiles for the current frame without drawing.

        Returns:
            {
              "all":  [tiles...],
              "hard": [tiles...],  # >= min_score
              "soft": [tiles...],  # soft_min_score <= score < min_score (if enabled)
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

        # Split into soft/hard bands if soft is enabled and tiles have .score
        soft_tiles, hard_tiles = [], []
        if include_soft and self.soft_min_score is not None:
            for t in tiles_all:
                if t.score and self.soft_min_score <= t.score < (self.min_score or float("inf")):
                    soft_tiles.append(t)
                else:
                    hard_tiles.append(t)
        else:
            hard_tiles = tiles_all

        if filter_invalid:
            soft_tiles = self.filter_tiles(soft_tiles)
            hard_tiles = self.filter_tiles(hard_tiles)

        return {
            "all":  hard_tiles + soft_tiles if include_soft else hard_tiles,
            "hard": hard_tiles,
            "soft": soft_tiles,
        }

    def get_in_focus_tiles(
        self,
        *,
        band: str = "all",          # "all" | "hard" | "soft"
        as_rects: bool = False      # return pygame.Rects instead of tile objects
    ):
        """
        Convenience accessor for focused tiles on the current frame.

        band:
          - "all":  hard (+ soft if configured)
          - "hard": only >= min_score
          - "soft": only soft band (requires soft_min_score)

        as_rects:
          - False: return the original tile objects from find_focused_areas
          - True:  return pygame.Rects in overlay coordinates (x,y,w,h)
        """
        result = self.compute_focused_tiles(include_soft=(band in ("all", "soft")), filter_invalid=True)
        tiles = result.get(band, result["all"]) if band in ("all", "hard", "soft") else result["all"]

        if not as_rects:
            return tiles

        rects = []
        for t in tiles:
            rects.append(pygame.Rect(int(t.x), int(t.y), int(t.w), int(t.h)))
        return rects

    def _tile_index_from_rect(self, x: int, y: int) -> Tuple[int, int]:
        """
        Convert a tile's top-left pixel (overlay space) to (col,row) index
        based on stride. Assumes tiles are laid on a stride grid.
        """
        col = max(0, x // self.stride)
        row = max(0, y // self.stride)
        return (int(col), int(row))

    def _tile_rect_from_index(self, col: int, row: int, w: int | None = None, h: int | None = None) -> pygame.Rect:
        """
        Convert a (col,row) tile index back to a rect in overlay space.
        Uses tile_size unless w/h are provided.
        """
        rw = int(w if w is not None else self.tile_size)
        rh = int(h if h is not None else self.tile_size)
        rx = int(col * self.stride)
        ry = int(row * self.stride)
        return pygame.Rect(rx, ry, rw, rh)

    def clear_hot_pixel_map(self) -> None:
        """Remove all invalid (hot) tiles."""
        self._invalid_tiles.clear()

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
        Sample the camera for ~duration_sec while the lens cap is on.
        Any tile that appears 'in focus' in any sampled frame is marked INVALID.
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

            # Use analysis pipeline without drawing
            res = self.compute_focused_tiles(include_soft=include_soft, filter_invalid=False)
            tiles = res["all"]
            # Count appearances
            for t in tiles:
                col = int(t.x) // stride
                row = int(t.y) // stride
                hits[(col, row)] += 1

            frames += 1

        # Select tiles meeting the hit threshold
        candidates: Set[Tuple[int, int]] = {ij for ij, n in hits.items() if n >= min_hits}

        # Optional dilation
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

    def is_tile_invalid(self, col: int, row: int) -> bool:
        """Check whether a (col,row) tile index is marked invalid."""
        return (col, row) in self._invalid_tiles

    def filter_tiles(self, tiles: Iterable) -> list:
        """
        Given tiles from find_focused_areas, drop any that overlap an invalid tile index.
        """
        out = []
        for t in tiles:
            col, row = self._tile_index_from_rect(t.x, t.y)
            if (col, row) not in self._invalid_tiles:
                out.append(t)
        return out

    def set_enabled(self, value: bool) -> None:
        self.enabled = bool(value)

    def toggle_overlay(self) -> None:
        self.visible = not self.visible

    def draw(self, surface: pygame.Surface) -> None:
        """
        Draw focus overlay:
        1) soft tiles (below min_score, above soft_min_score)
        2) hard tiles (>= min_score)
        3) invalid tiles (hot-pixel map) in red, on top
        """
        if not self.visible:
            return

        fr = self.camera_view.get_frame_rect()
        if not fr:
            return
        dx, dy, fw, fh = fr

        # Compute tiles once (no capture/draw duplication)
        res = self.compute_focused_tiles(include_soft=True, filter_invalid=True)
        soft_tiles = res["soft"]
        hard_tiles = res["hard"]

        # Draw helper
        def _blit_rect(fill_color, border_color, border_w, rx, ry, rw, rh):
            if fill_color:
                rect_surf = pygame.Surface((rw, rh), pygame.SRCALPHA)
                rect_surf.fill(fill_color)
                surface.blit(rect_surf, (rx, ry))
            if border_color and border_w > 0:
                pygame.draw.rect(surface, border_color, pygame.Rect(rx, ry, rw, rh), width=border_w)

        # Soft tiles first
        for t in soft_tiles:
            rx, ry = dx + int(t.x), dy + int(t.y)
            _blit_rect(self.soft_fill, self.soft_border, self.soft_border_w, rx, ry, int(t.w), int(t.h))

        # Hard tiles next
        for t in hard_tiles:
            rx, ry = dx + int(t.x), dy + int(t.y)
            _blit_rect(self.hard_fill, self.hard_border, self.hard_border_w, rx, ry, int(t.w), int(t.h))

        # Invalid (hot) tiles on top
        if self.draw_invalid and self._invalid_tiles:
            for (col, row) in self._invalid_tiles:
                r = self._tile_rect_from_index(col, row, self.tile_size, self.tile_size)
                _blit_rect(self.invalid_fill, self.invalid_border, self.invalid_border_w, dx + r.x, dy + r.y, r.w, r.h)

