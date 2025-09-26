import pygame
from typing import Optional
import numpy as np

from UI.frame import Frame
from UI.text import Text, TextStyle

from image_processing.analyzers import find_focused_areas

class CameraView(Frame):
    """
    A Frame that renders a camera feed inside itself, respecting pixel margins.
    - Maintains aspect ratio with letterboxing
    - Reacts to parent resize automatically
    - Can be put behind everything via z_index
    """
    def __init__(
        self,
        camera,
        parent: Optional[Frame] = None,
        *,
        x: float = 0,
        y: float = 0,
        width: float = 1.0,
        height: float = 1.0,
        x_is_percent: bool = True,
        y_is_percent: bool = True,
        width_is_percent: bool = True,
        height_is_percent: bool = True,
        z_index: int = -100,  # keep it behind panels/modals
        x_align: str = "left",
        y_align: str = "top",
        background_color: Optional[pygame.Color] = None,
        mouse_passthrough: bool = True,
        left_margin_px: int = 0,
        right_margin_px: int = 0,
        top_margin_px: int = 0,
        bottom_margin_px: int = 0,
    ):
        self.camera = camera
        self.mouse_passthrough = mouse_passthrough
        self.background_color = background_color

        # margins (in pixels) to reserve for other UI
        self.left_margin_px = left_margin_px
        self.right_margin_px = right_margin_px
        self.top_margin_px = top_margin_px
        self.bottom_margin_px = bottom_margin_px

        # track last applied size to avoid redundant resizes
        self._last_draw_w = None
        self._last_draw_h = None
        self._last_frame_rect = None  # (dx, dy, fw, fh)

        super().__init__(
            parent=parent,
            x=x, y=y, width=width, height=height,
            x_is_percent=x_is_percent, y_is_percent=y_is_percent,
            width_is_percent=width_is_percent, height_is_percent=height_is_percent,
            z_index=z_index, x_align=x_align, y_align=y_align,
            background_color=None,  # we fill manually to keep margins clean
        )

        self.no_camera_text = Text(
            text="No Camera Detected",
            parent=self,
            x=0.5, y=0.5,
            x_is_percent=True, y_is_percent=True,
            x_align="center", y_align="center",
            style=TextStyle(font_size=32),
        )

        # Show/hide based on current init state
        if self.camera.initialized:
            self.no_camera_text.add_hidden_reason("SYSTEM")

    def _compute_inner_rect(self):
        # Base geometry from normal frame rules
        abs_x, abs_y, abs_w, abs_h = super().get_absolute_geometry()
        # Apply pixel margins to shrink usable area
        ix = abs_x + self.left_margin_px
        iy = abs_y + self.top_margin_px
        iw = max(0, abs_w - self.left_margin_px - self.right_margin_px)
        ih = max(0, abs_h - self.top_margin_px - self.bottom_margin_px)
        return ix, iy, iw, ih

    def get_frame_rect(self):
        """
        Returns (x, y, w, h) of the currently drawn camera frame within the surface,
        accounting for letterboxing. May be None if nothing drawn yet.
        """
        return self._last_frame_rect

    def get_absolute_geometry(self):
        # Expose the *inner* rect as the geometry of this view
        return self._compute_inner_rect()

    def draw(self, surface: pygame.Surface) -> None:
        if self.is_effectively_hidden:
            return

        ix, iy, iw, ih = self._compute_inner_rect()

        # Optional background fill behind letterboxed image
        if self.background_color:
            pygame.draw.rect(surface, self.background_color, (ix, iy, iw, ih))

        # If our draw size changed, notify the camera so its scaler is updated
        if iw != self._last_draw_w or ih != self._last_draw_h:
            self._last_draw_w, self._last_draw_h = iw, ih
            if iw > 0 and ih > 0:
                self.camera.resize(iw, ih)

        # Get the scaled frame and letterbox it inside (ix, iy, iw, ih)
        frame = self.camera.get_frame()
        fw, fh = frame.get_size() if frame else (0, 0)

        # Center the frame (already scaled in camera.get_frame())
        dx = ix + (iw - fw) // 2
        dy = iy + (ih - fh) // 2
        self._last_frame_rect = (dx, dy, fw, fh)

        if frame and iw > 0 and ih > 0:
            surface.blit(frame, (dx, dy))

        # Draw overlays/children on top
        for child in reversed(self.children):
            child.draw(surface)

class FocusOverlay(Frame):
    def __init__(
        self,
        camera_view: "CameraView",
        *,
        enabled = True,
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

        self._frame_counter = 0
        self._cached_tiles = None
        self.enabled = enabled
        
    def set_enabled(self, value: bool) -> None:
        self.enabled = bool(value)

    def toggle_overlay(self) -> None:
        self.enabled = not self.enabled

    def draw(self, surface: pygame.Surface) -> None:
        if not self.enabled:
            return

        fr = self.camera_view.get_frame_rect()
        if not fr:
            return
        dx, dy, fw, fh = fr
        if fw <= 0 or fh <= 0:
            return

        frame_surface = self.camera_view.camera.get_frame()
        if not frame_surface:
            return

        pg_pixels = pygame.surfarray.array3d(frame_surface)
        img_rgb = np.transpose(pg_pixels, (1, 0, 2))
        img_bgr = img_rgb[:, :, ::-1]

        self._frame_counter = (self._frame_counter + 1) % 2
        if self._frame_counter == 0 or self._cached_tiles is None:
            self._cached_tiles = find_focused_areas(
                img_bgr,
                tile_size=self.tile_size,
                stride=self.stride,
                top_percent=self.top_percent,
                min_score=self.min_score,
                soft_min_score=self.soft_min_score,
            )

        tiles = self._cached_tiles or []

        # Style: hard = cyan-ish, soft = amber-ish
        fill_hard = pygame.Color(0, 200, 255, self.alpha_hard)
        border_hard = pygame.Color(0, 200, 255, self.border_alpha_hard)
        fill_soft = pygame.Color(255, 180, 0, self.alpha_soft)
        border_soft = pygame.Color(255, 180, 0, self.border_alpha_soft)

        for t in tiles:
            rx = dx + t.x
            ry = dy + t.y
            rw = t.w
            rh = t.h

            if t.band == "hard":
                fill, border = fill_hard, border_hard
                border_w = 2
            else:
                fill, border = fill_soft, border_soft
                border_w = 1

            rect_surf = pygame.Surface((rw, rh), pygame.SRCALPHA)
            rect_surf.fill(fill)
            surface.blit(rect_surf, (rx, ry))
            pygame.draw.rect(surface, border, pygame.Rect(rx, ry, rw, rh), width=border_w)

