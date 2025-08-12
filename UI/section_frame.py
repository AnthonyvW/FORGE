import pygame
from UI.frame import Frame
from UI.text import Text, TextStyle

class Section(Frame):
    def __init__(
        self, *,
        parent,
        title: str,
        x=0, y=0, width=100, height=100,
        x_is_percent=False, y_is_percent=False,
        width_is_percent=False, height_is_percent=False,
        z_index=0,
        background_color=pygame.Color("#ffffff"),
        header_height=32,
        header_bg=pygame.Color("#dbdbdb"),
        title_style: TextStyle | None = None,
        title_align: str = "left"
    ):
        super().__init__(
            parent=parent, x=x, y=y, width=width, height=height,
            x_is_percent=x_is_percent, y_is_percent=y_is_percent,
            width_is_percent=width_is_percent, height_is_percent=height_is_percent,
            z_index=z_index, background_color=background_color
        )

        # Header bar
        self.header = Frame(
            parent=self,
            x=0, y=0,
            width=1.0, height=header_height,
            width_is_percent=True,
            background_color=header_bg,
            z_index=z_index + 1
        )
        super().add_child(self.header)

        # Title text
        if title_style is None:
            title_style = TextStyle(
                color=pygame.Color("#7a7a7a"),
                font_size=24,
                font_name="assets/fonts/SofiaSans-Regular.ttf",
            )
        self.title = Text(
            text=title,
            x=8 if title_align == "left" else self.header.width // 2,
            y=self.header.height // 2,
            x_align=title_align,
            y_align="center",
            style=title_style,
        )
        self.header.add_child(self.title)

        # Body (content area) — fills remaining vertical space below header
        self.body = Frame(
            parent=self,
            x=0, y=self.header.height,
            width=1.0,
            width_is_percent=True,
            height=max(0, height - header_height) if not height_is_percent else 0,
            height_is_percent=not height_is_percent,  # if parent height is % we’ll size body each frame
            z_index=z_index
        )
        super().add_child(self.body)

    # Default: when you "add_child" to a Section, it goes into the body.
    def add_child(self, child):
        if getattr(self, "_initializing", False) or not hasattr(self, "body"):
            # during bootstrap, attach to the Section itself (not the body)
            return super().add_child(child)
        return self.body.add_child(child)

    # If you ever need to add to the header explicitly:
    def add_to_header(self, child):
        self.header.add_child(child)

    # Convenience: absolute geometry of just the content area
    def get_content_geometry(self):
        sec_x, sec_y, sec_w, sec_h = self.get_absolute_geometry()
        return sec_x, sec_y + self.header.height, sec_w, max(0, sec_h - self.header.height)

    # Keep the body sized correctly when parent is percent-based or resizes
    def _layout(self):
        # Use absolute sizes to set a pixel height on the body
        _, _, sec_w, sec_h = self.get_absolute_geometry()
        self.body.y = self.header.height
        self.body.height_is_percent = False
        self.body.height = max(0, sec_h - self.header.height)

    # Ensure layout is up to date before drawing/dispatching
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
