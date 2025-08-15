import pygame
from typing import Optional
from UI.frame import Frame

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

        super().__init__(
            parent=parent,
            x=x, y=y, width=width, height=height,
            x_is_percent=x_is_percent, y_is_percent=y_is_percent,
            width_is_percent=width_is_percent, height_is_percent=height_is_percent,
            z_index=z_index, x_align=x_align, y_align=y_align,
            background_color=None,  # we fill manually to keep margins clean
        )

    def _compute_inner_rect(self):
        # Base geometry from normal frame rules
        abs_x, abs_y, abs_w, abs_h = super().get_absolute_geometry()
        # Apply pixel margins to shrink usable area
        ix = abs_x + self.left_margin_px
        iy = abs_y + self.top_margin_px
        iw = max(0, abs_w - self.left_margin_px - self.right_margin_px)
        ih = max(0, abs_h - self.top_margin_px - self.bottom_margin_px)
        return ix, iy, iw, ih

    def get_absolute_geometry(self):
        # Expose the *inner* rect as the geometry of this view
        return self._compute_inner_rect()

    def draw(self, surface: pygame.Surface) -> None:
        if self.is_hidden:
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
        if frame and iw > 0 and ih > 0:
            surface.blit(frame, (dx, dy))

        # Draw overlays/children on top
        for child in reversed(self.children):
            child.draw(surface)
