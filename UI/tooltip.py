import pygame
from UI.frame import Frame
from UI.text import Text
from UI.styles import make_display_text_style


class Tooltip(Frame):
    """
    Self-attaching, multi-line tooltip using Forge's global make_display_text_style().

    Example:
        tip = Tooltip.attach(some_frame, "Hello\nWorld")
        tip.set_text("Updated text")
        tip.detach()
    """

    def __init__(
        self,
        parent,
        text: str,
        *,
        font_size: int = 20,
        padding: int = 6,
        bg_color=(255, 255, 224),
        border_color=(0, 0, 0),
        follow_cursor: bool = True,
        margin: int = 6,
        cursor_offset: int = 12,
        z_index: int = 10_000,
        line_spacing: int = 2,
        style_overrides: dict | None = None,
    ):
        super().__init__(parent=parent, x=0, y=0, width=0, height=0, z_index=z_index)
        self.mouse_passthrough = True

        self._padding = padding
        self._bg_color = pygame.Color(*bg_color)
        self._border_color = pygame.Color(*border_color)
        self._follow_cursor = follow_cursor
        self._margin = margin
        self._cursor_offset = cursor_offset
        self._line_spacing = line_spacing
        self._font_size = font_size

        # base style from global Forge styling system
        self._text_style = make_display_text_style(font_size=font_size)
        if style_overrides:
            for k, v in style_overrides.items():
                if hasattr(self._text_style, k):
                    setattr(self._text_style, k, v)

        self._target = None
        self._orig_enter = None
        self._orig_leave = None
        self._orig_hover = None

        self._line_widgets: list[Text] = []
        self.set_text(text)
        self.hide()

    # ---------------- attach / detach ----------------
    @classmethod
    def attach(cls, target: Frame, text: str, **kwargs) -> "Tooltip":
        """Attach to any Frame without editing its class."""
        root = target
        while root.parent is not None:
            root = root.parent
        tip = cls(parent=root, text=text, **kwargs)
        tip._bind(target)
        return tip

    def detach(self) -> None:
        """Unbind and remove from scene."""
        if self._target:
            self._target.on_hover_enter = self._orig_enter
            self._target.on_hover_leave = self._orig_leave
            self._target.on_hover = self._orig_hover
            self._target = None
        if self.parent and self in self.parent.children:
            self.parent.children.remove(self)

    # ---------------- event wrapping ----------------
    def _bind(self, target: Frame) -> None:
        self._target = target
        self._orig_enter = target.on_hover_enter
        self._orig_leave = target.on_hover_leave
        self._orig_hover = target.on_hover

        def wrapped_enter():
            self._orig_enter()
            self._show_at_mouse()

        def wrapped_leave():
            self._orig_leave()
            self.hide()

        def wrapped_hover():
            self._orig_hover()
            if self._follow_cursor and not self.is_effectively_hidden:
                self._reposition_to_mouse()

        target.on_hover_enter = wrapped_enter
        target.on_hover_leave = wrapped_leave
        target.on_hover = wrapped_hover

    # ---------------- text layout ----------------
    def set_text(self, text: str):
        """Supports multiple lines using '\\n'."""
        for w in self._line_widgets:
            if w in self.children:
                self.children.remove(w)
        self._line_widgets.clear()

        lines = text.split("\n") if text else [""]
        y_cursor = self._padding
        max_w = 0

        for line in lines:
            tw = Text(
                text=line,
                x=self._padding,
                y=y_cursor,
                style=self._text_style,
            )
            tw.mouse_passthrough = True
            self.add_child(tw)
            self._line_widgets.append(tw)

            lw, lh = tw.size
            y_cursor += lh + self._line_spacing
            max_w = max(max_w, lw)

        if self._line_widgets:
            y_cursor -= self._line_spacing

        self.width = max_w + self._padding * 2
        self.height = y_cursor + self._padding

    # ---------------- positioning ----------------
    def _bring_to_front(self):
        if not self.parent:
            return
        max_z = max((getattr(ch, "z_index", 0) for ch in self.parent.children), default=0)
        if self.z_index <= max_z:
            self.z_index = max_z + 1
            self.parent.children.sort(key=lambda c: c.z_index, reverse=True)

    def _show_at_mouse(self):
        self._reposition_to_mouse()
        self._bring_to_front()
        self.show()

    def _reposition_to_mouse(self):
        mx, my = pygame.mouse.get_pos()
        sw, sh = pygame.display.get_surface().get_size()
        w, h = self.width, self.height

        x = mx + self._cursor_offset
        y = my + self._cursor_offset

        if x + w > sw - self._margin:
            x = mx - w - self._cursor_offset
        if y + h > sh - self._margin:
            y = my - h - self._cursor_offset

        x = max(self._margin, min(x, sw - w - self._margin))
        y = max(self._margin, min(y, sh - h - self._margin))

        self.x, self.y = x, y
        self._bring_to_front()

    # ---------------- draw ----------------
    def draw(self, surface: pygame.Surface) -> None:
        if self.is_effectively_hidden:
            return
        abs_x, abs_y, w, h = self.get_absolute_geometry()
        pygame.draw.rect(surface, self._bg_color, (abs_x, abs_y, w, h))
        pygame.draw.rect(surface, self._border_color, (abs_x, abs_y, w, h), 1)
        for child in reversed(self.children):
            child.draw(surface)
