import pygame

from UI.frame import Frame
from UI.text import Text, TextStyle

from UI.input.button import Button, ButtonColors

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
        title_align: str = "left",
        collapsible: bool = False,
        **kwargs
    ):
        self._initializing = True
        self.collapsible = collapsible
        self.collapsed = False

        body_padding = kwargs.pop("padding", (0, 0, 0, 0))

        super().__init__(
            parent=parent, x=x, y=y, width=width, height=height,
            x_is_percent=x_is_percent, y_is_percent=y_is_percent,
            width_is_percent=width_is_percent, height_is_percent=height_is_percent,
            z_index=z_index, background_color=background_color,
            padding=(0, 0, 0, 0),
            **kwargs
        )

        # Save original (expanded) height config so we can restore it
        self._saved_height = height
        self._saved_height_is_percent = height_is_percent

        # Header bar
        self.header = Frame(
            parent=self,
            x=0, y=0,
            width=1.0, height=header_height,
            width_is_percent=True,
            background_color=header_bg,
            z_index=z_index + 1
        )

        # Title text
        if title_style is None:
            title_style = TextStyle(
                color=pygame.Color("#7a7a7a"),
                font_size=24,
                font_name="assets/fonts/SofiaSans-Regular.ttf",
            )
        self.title = Text(
            text=title,
            parent=self.header,
            x=(0.5 if title_align == "center" else 8),
            y=self.header.height // 2,
            x_is_percent=(title_align == "center"),
            y_is_percent=False,
            x_align=title_align,
            y_align="center",
            style=title_style,
        )

        # Collapse toggle button
        if self.collapsible:
            self.toggle_btn = Button(
                self.toggle_collapse,
                x=0, y=0,
                width=header_height, height=header_height,
                text="-",
                parent=self.header,
                x_align="right", y_align="top",
                colors=ButtonColors(
                    background=header_bg,
                    foreground=header_bg,
                    hover_background=pygame.Color("#b3b4b6"),
                    disabled_background=header_bg,
                    disabled_foreground=header_bg
                ),
                text_style=TextStyle(
                    color=pygame.Color("#7a7a7a"),
                    font_size=min(20, header_height - 8),
                )
            )
        else:
            self.toggle_btn = None

        # Body (content area)
        self.body = Frame(
            parent=self,
            x=0, y=self.header.height,
            width=1.0,
            width_is_percent=True,
            height=max(0, height - header_height) if not height_is_percent else 0,
            height_is_percent=not height_is_percent,
            z_index=z_index,
            padding=body_padding
        )

        self._initializing = False

    # --- public helper (optional) ---
    def set_collapsed(self, value: bool):
        if value == self.collapsed:
            return

        self.collapsed = value
        if self.toggle_btn:
            self.toggle_btn.set_text("+" if self.collapsed else "-")

        if self.collapsed:
            # Save current size/fill config
            self._saved_height = self.height
            self._saved_height_is_percent = self.height_is_percent
            self._saved_fill_remaining_height = getattr(self, "_saved_fill_remaining_height", self.fill_remaining_height)

            # Clamp Section to header-only and stop filling parent
            self.fill_remaining_height = False
            self.height_is_percent = False
            self.height = self.header.height

            # Hide only the body subtree
            self._for_each_in_body(lambda f: f.add_hidden_reason("COLLAPSED"))

        else:
            # Restore original size/fill config
            self.height_is_percent = self._saved_height_is_percent
            self.height = self._saved_height
            self.fill_remaining_height = getattr(self, "_saved_fill_remaining_height", self.fill_remaining_height)

            # Unhide the body subtree
            self._for_each_in_body(lambda f: f.remove_hidden_reason("COLLAPSED"))

    def toggle_collapse(self):
        self.set_collapsed(not self.collapsed)

    def add_child(self, child):
        if getattr(self, "_initializing", False) or not hasattr(self, "body"):
            return super().add_child(child)
        return self.body.add_child(child)

    def _for_each_in_body(self, fn):
        stack = [self.body]
        while stack:
            node = stack.pop()
            fn(node)
            stack.extend(node.children)

    def add_to_header(self, child):
        self.header.add_child(child)

    def get_content_geometry(self):
        # Use the section's outer rect (no header offset). Padding lives on body now.
        abs_x, abs_y, abs_w, abs_h = Frame.get_absolute_geometry(self)
        # Section keeps zero padding; body owns padding. If you ever want chrome padding, set it here.
        pad_top, pad_right, pad_bottom, pad_left = self.padding
        inner_x = abs_x + pad_left
        inner_y = abs_y + pad_top
        inner_w = max(0, abs_w - pad_left - pad_right)
        inner_h = max(0, abs_h - pad_top - pad_bottom)
        return inner_x, inner_y, inner_w, inner_h

    def _layout(self):
        _, _, sec_w, sec_h = self.get_absolute_geometry()
        self.body.y = self.header.height
        self.body.height_is_percent = False
        # With the section height now clamped to header.height when collapsed,
        # this naturally becomes 0. Otherwise, it's the remaining space.
        self.body.height = max(0, sec_h - self.header.height)

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
