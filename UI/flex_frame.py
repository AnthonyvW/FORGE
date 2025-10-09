import pygame
from typing import Tuple
from UI.frame import Frame

class FlexFrame(Frame):
    """
    Simple flex-like layout container.

    Direction: column only (for now).
    - Packs visible children from top to bottom.
    - Uses each child's *current* height (so collapsed Sections shrink naturally).
    - Optionally fills child widths to container width.
    """

    def __init__(
        self,
        *,
        parent=None,
        x=0, y=0, width=100, height=100,
        x_is_percent=False, y_is_percent=False,
        width_is_percent=False, height_is_percent=False,
        z_index=0,
        background_color: pygame.Color | None = None,

        # Flex options
        padding: Tuple[int, int, int, int] = (0, 0, 0, 0),  # (left, top, right, bottom)
        gap: int = 8,
        fill_child_width: bool = True,
        align_horizontal: str = "left",  # "left" | "center" | "right"
        auto_height_to_content: bool = False,  # If True, grow/shrink this frame's height to fit children
        **kwargs
    ):
        super().__init__(
            parent=parent, x=x, y=y, width=width, height=height,
            x_is_percent=x_is_percent, y_is_percent=y_is_percent,
            width_is_percent=width_is_percent, height_is_percent=height_is_percent,
            z_index=z_index, background_color=background_color,
            padding=padding,
            **kwargs
        )
        self.gap = gap
        self.fill_child_width = fill_child_width
        self.align_horizontal = align_horizontal
        self.auto_height_to_content = auto_height_to_content

        self._layout_dirty = True

    # --- Public: you can call this if you change padding/gap/etc dynamically
    def request_layout(self) -> None:
        self._layout_dirty = True

    # --- Core layout: compute child positions/sizes in absolute container space
    def _layout(self) -> None:

        y_cursor = 0  # content-local pixels (0 is top of content box)
        visible_children = [ch for ch in self.children if not ch.is_effectively_hidden]

        for i, child in enumerate(visible_children):
            # --- Fill width if requested: since base Frame uses the content box,
            #     percent widths are now relative to content width. So 1.0 == fill.
            if self.fill_child_width:
                child.width_is_percent = True
                child.width = 1.0
            # else: leave child's width props as-is

            # Horizontal alignment inside content box
            if self.align_horizontal == "left":
                child.x_align = "left"
            elif self.align_horizontal == "center":
                child.x_align = "center"
            elif self.align_horizontal == "right":
                child.x_align = "right"
            else:
                child.x_align = "left"

            # Position vertically (content-local); base Frame will offset by content origin
            child.y_is_percent = False
            child.y = y_cursor

            # For left/center/right we want x as an offset from alignment anchor
            child.x_is_percent = False
            child.x = 0  # use alignment only

            # After setting width/position hints, read child's computed height
            # (this uses current width/height props relative to content box)
            _, _, ch_w, ch_h = child.get_absolute_geometry()

            # Advance cursor
            y_cursor += ch_h
            if i != len(visible_children) - 1:
                y_cursor += self.gap

        # Auto-size this FlexFrame’s OUTER height to fit children + padding
        if self.auto_height_to_content and not self.height_is_percent:
            pt, pr, pb, pl = self.padding  # base Frame’s padding (top,right,bottom,left)
            self.height = max(0, y_cursor + pt + pb)

        self._layout_dirty = False

    # Ensure layout is up-to-date in all the usual passes
    def draw(self, surface: pygame.Surface) -> None:
        self._layout()
        super().draw(surface)

    def process_mouse_move(self, px, py):
        self._layout()
        super().process_mouse_move(px, py)

    def process_mouse_press(self, px, py, button):
        self._layout()
        super().process_mouse_press(px, py, button)

    def process_mouse_release(self, px, py, button):
        self._layout()
        super().process_mouse_release(px, py, button)

    def process_mouse_wheel(self, px: int, py: int, *, dx: int, dy: int) -> bool:
        self._layout()
        return super().process_mouse_wheel(px, py, dx=dx, dy=dy)
