import pygame
from typing import Tuple, Set

from UI.frame import Frame
from UI.camera_view import CameraView

from image_processing.machine_vision import MachineVision


class FocusOverlay(Frame):
    """
    UI overlay that renders the focused tiles and (optionally) the invalid/hot tiles
    produced by MachineVision.
    """
    def __init__(
        self,
        camera_view: "CameraView",
        mv: MachineVision,   # Injected machine vision instance (optional)
        visible: bool = False,

        # Visual styles (overlay-only concerns)
        alpha_hard: int = 100,
        border_alpha_hard: int = 200,
        alpha_soft: int = 50,
        border_alpha_soft: int = 120,
        invalid_alpha_fill: int = 90,
        invalid_alpha_border: int = 180,
        invalid_border_w: int = 2,
        soft_border_w: int = 1,
        hard_border_w: int = 2,
        draw_invalid: bool = True,
    ):
        super().__init__(parent=camera_view, x=0, y=0, width=1, height=1,
                         x_is_percent=True, y_is_percent=True,
                         width_is_percent=True, height_is_percent=True,
                         z_index=camera_view.z_index + 1,
                         background_color=None)

        self.camera_view = camera_view
        self.mv = mv

        # Visuals
        self.visible = visible
        self.draw_invalid = draw_invalid

        self.soft_fill   = pygame.Color(255, 180, 0, alpha_soft)
        self.soft_border = pygame.Color(255, 180, 0, border_alpha_soft)
        self.soft_border_w = soft_border_w

        self.hard_fill   = pygame.Color(0, 200, 255, alpha_hard)
        self.hard_border = pygame.Color(0, 200, 255, border_alpha_hard)
        self.hard_border_w = hard_border_w

        self.invalid_fill     = pygame.Color(255, 0, 0, invalid_alpha_fill)
        self.invalid_border   = pygame.Color(255, 0, 0, invalid_alpha_border)
        self.invalid_border_w = invalid_border_w

        # Edge margin overlays (translucent red)
        self.draw_edge_margins = True
        self.edge_fill = pygame.Color(255, 0, 0, 80)      # translucent red fill
        self.edge_border = pygame.Color(255, 0, 0, 160)   # slightly less translucent border
        self.edge_border_w = 1
        
        # cache overlay to avoid realloc each frame
        self._overlay = None
        self._overlay_size = None

    # ---------- convenience pass-throughs for callers ----------
    def toggle_overlay(self) -> None:
        self.visible = not self.visible

    def set_enabled(self, value: bool) -> None:
        self.enabled = bool(value)

    def clear_hot_pixel_map(self) -> None:
        self.mv.clear_hot_pixel_map()

    def build_hot_pixel_map(
        self,
        duration_sec: float = 1.0,
        *,
        dilate: int = 0,
        min_hits: int = 1,
        max_fps: int = 30,
        include_soft: bool = True,
    ):
        return self.mv.build_hot_pixel_map(
            duration_sec=duration_sec,
            dilate=dilate,
            min_hits=min_hits,
            max_fps=max_fps,
            include_soft=include_soft,
        )

    def is_tile_invalid(self, col: int, row: int) -> bool:
        return self.mv.is_tile_invalid(col, row)


    def _get_overlay(self, surface_size: tuple[int, int]) -> pygame.Surface:
        """Return an RGBA overlay the size of the target surface (recreate on resize)."""
        if self._overlay is None or self._overlay_size != surface_size:
            self._overlay_size = surface_size
            self._overlay = pygame.Surface(surface_size, flags=pygame.SRCALPHA)
        else:
            # Clear with fully transparent color
            self._overlay.fill((0, 0, 0, 0))
        return self._overlay

    # ------------------------------- draw -------------------------------
    def draw(self, surface: pygame.Surface) -> None:
        if not self.visible:
            return

        # Build/resize overlay and clear it fully transparent
        overlay = self._get_overlay(surface.get_size())
        overlay.fill((0, 0, 0, 0))

        fr = self.camera_view.get_frame_rect()
        if not fr:
            return

        fx, fy, fw, fh = fr

        # Get raw frame shape so we can map RAW → GUI coordinates
        raw = self.mv.capture_current_frame(color="rgb", source="latest")
        if raw is None:
            return
        h, w = raw.shape[:2]

        # Scale factors from RAW pixel space to the drawn frame rectangle
        sx = float(fw) / float(w) if w else 1.0
        sy = float(fh) / float(h) if h else 1.0

        # Compute focused tiles once in RAW
        res = self.mv.compute_focused_tiles(include_soft=True, filter_invalid=True)
        soft_tiles = res["soft"]
        hard_tiles = res["hard"]

        # All blits/draws go to overlay (supports alpha)
        def _blit_rect(fill_color, border_color, border_w, rect):
            if rect.width <= 0 or rect.height <= 0:
                return
            pygame.draw.rect(overlay, fill_color, rect)
            pygame.draw.rect(overlay, border_color, rect, border_w)

        def rect_from_raw(tile, fx, fy, sx, sy):
            # Snap both edges (shared boundaries) and derive size from them
            left   = fx + int(round(tile.x * sx))
            top    = fy + int(round(tile.y * sy))
            right  = fx + int(round((tile.x + tile.w) * sx))
            bottom = fy + int(round((tile.y + tile.h) * sy))
            w = max(0, right - left)
            h = max(0, bottom - top)
            return pygame.Rect(left, top, w, h)

        # --- draw edge margin overlays (percent insets from each edge) ---
        if self.draw_edge_margins and self.mv is not None:
            l_pct, r_pct, t_pct, b_pct = self.mv.get_edge_margins()

            # Clamp
            l_pct = max(0.0, min(1.0, float(l_pct)))
            r_pct = max(0.0, min(1.0, float(r_pct)))
            t_pct = max(0.0, min(1.0, float(t_pct)))
            b_pct = max(0.0, min(1.0, float(b_pct)))

            # Sizes in screen pixels
            left_w   = int(round(fw * l_pct))
            right_w  = int(round(fw * r_pct))
            top_h    = int(round(fh * t_pct))
            bottom_h = int(round(fh * b_pct))

            # Interior (safe) rect
            inner_x = fx + left_w
            inner_y = fy + top_h
            inner_w = max(0, fw - left_w - right_w)
            inner_h = max(0, fh - top_h - bottom_h)

            # TOP (owns corners)
            if top_h > 0:
                pygame.draw.rect(overlay, self.edge_fill, (fx, fy, fw, top_h), 0)

            # BOTTOM (owns corners)
            if bottom_h > 0:
                pygame.draw.rect(overlay, self.edge_fill, (fx, fy + fh - bottom_h, fw, bottom_h), 0)

            # LEFT (trimmed to avoid overlap)
            usable_h = max(0, fh - top_h - bottom_h)
            if left_w > 0 and usable_h > 0:
                pygame.draw.rect(overlay, self.edge_fill, (fx, fy + top_h, left_w, usable_h), 0)

            # RIGHT (trimmed)
            if right_w > 0 and usable_h > 0:
                pygame.draw.rect(overlay, self.edge_fill, (fx + fw - right_w, fy + top_h, right_w, usable_h), 0)

            # Single border around the central (non-red) region
            if inner_w > 0 and inner_h > 0 and self.edge_border_w > 0:
                pygame.draw.rect(
                    overlay,
                    self.edge_border,
                    pygame.Rect(inner_x, inner_y, inner_w, inner_h),
                    width=self.edge_border_w
                )

        # Soft tiles
        for t in soft_tiles:
            r = rect_from_raw(t, fx, fy, sx, sy)
            _blit_rect(self.soft_fill, self.soft_border, self.soft_border_w, r)

        for t in hard_tiles:
            r = rect_from_raw(t, fx, fy, sx, sy)
            _blit_rect(self.hard_fill, self.hard_border, self.hard_border_w, r)

        # Invalid (hot) tiles
        if self.draw_invalid and self.mv.invalid_tiles:
            interior_raw = self.mv.get_interior_rect_pixels(w, h)  # RAW coords
            for (col, row) in self.mv.invalid_tiles:
                t = self.mv.tile_rect_from_index(col, row)  # RAW rect
                # only draw if fully inside the interior
                if (t.left   >= interior_raw.left and
                    t.top    >= interior_raw.top  and
                    t.right  <= interior_raw.right and
                    t.bottom <= interior_raw.bottom):
                    r = rect_from_raw(t, fx, fy, sx, sy)
                    _blit_rect(self.invalid_fill, self.invalid_border, self.invalid_border_w, r)

        # Composite overlay (with alpha) onto the actual screen surface
        surface.blit(overlay, (0, 0))