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

    # ------------------------------- draw -------------------------------
    def draw(self, surface: pygame.Surface) -> None:
        if not self.visible:
            return

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

        def _blit_rect(fill_color, border_color, border_w, rx, ry, rw, rh):
            if fill_color:
                rect_surf = pygame.Surface((rw, rh), pygame.SRCALPHA)
                rect_surf.fill(fill_color)
                surface.blit(rect_surf, (rx, ry))
            if border_color and border_w > 0:
                pygame.draw.rect(surface, border_color, pygame.Rect(rx, ry, rw, rh), width=border_w)

        # Soft tiles
        for t in soft_tiles:
            rx = fx + int(round(t.x * sx))
            ry = fy + int(round(t.y * sy))
            rw = int(round(t.w * sx))
            rh = int(round(t.h * sy))
            _blit_rect(self.soft_fill, self.soft_border, self.soft_border_w, rx, ry, rw, rh)

        # Hard tiles
        for t in hard_tiles:
            rx = fx + int(round(t.x * sx))
            ry = fy + int(round(t.y * sy))
            rw = int(round(t.w * sx))
            rh = int(round(t.h * sy))
            _blit_rect(self.hard_fill, self.hard_border, self.hard_border_w, rx, ry, rw, rh)

        # Invalid (hot) tiles
        if self.draw_invalid and self.mv.invalid_tiles:
            for (col, row) in self.mv.invalid_tiles:
                r = self.mv.tile_rect_from_index(col, row)  # RAW rect
                rx = fx + int(round(r.x * sx))
                ry = fy + int(round(r.y * sy))
                rw = int(round(r.w * sx))
                rh = int(round(r.h * sy))
                _blit_rect(self.invalid_fill, self.invalid_border, self.invalid_border_w, rx, ry, rw, rh)
