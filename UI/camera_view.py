from typing import Optional

import pygame
import numpy as np

from UI.frame import Frame
from UI.text import Text, TextStyle


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
        if self.background_color:
            pygame.draw.rect(surface, self.background_color, (ix, iy, iw, ih))

        # --- fetch NumPy frame (prefer still, fallback to last live stream) ---
        arr = self.camera.get_last_frame(prefer="latest", wait_for_still=False)

        if arr is None or iw <= 0 or ih <= 0:
            self._last_frame_rect = (ix, iy, 0, 0)
            # (Optionally draw a subtle "no signal" background here)
            for child in reversed(self.children):
                child.draw(surface)
            return

        # Ensure contiguous uint8 RGB
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        h, w, c = arr.shape
        assert c in (3, 4)

        # --- fit to (iw, ih) with letterboxing ---
        scale = min(iw / w, ih / h)
        tw, th = max(1, int(round(w * scale))), max(1, int(round(h * scale)))

        # Convert NumPy → Surface then scale
        # Use frombuffer on a contiguous copy to avoid strides issues
        buf = arr[:, :, :3].copy(order="C").tobytes()  # RGB only for pygame
        frame_surf = pygame.image.frombuffer(buf, (w, h), "RGB")
        if (tw, th) != (w, h):
            frame_surf = pygame.transform.smoothscale(frame_surf, (tw, th))

        dx = ix + (iw - tw) // 2
        dy = iy + (ih - th) // 2
        self._last_frame_rect = (dx, dy, tw, th)
        surface.blit(frame_surf, (dx, dy))

        # Draw overlays/children
        for child in reversed(self.children):
            child.draw(surface)
