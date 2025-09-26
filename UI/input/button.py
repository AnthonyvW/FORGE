import pygame
from typing import Callable, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from UI.text import Text, TextStyle
from UI.frame import Frame

class ButtonShape(Enum):
    RECTANGLE = "rectangle"
    DIAMOND = "diamond"

def default_background() -> pygame.Color:
    return pygame.Color("#dbdbdb")

def default_foreground() -> pygame.Color:  # used for BORDER only
    return pygame.Color("#b3b4b6")

def default_hover_background() -> pygame.Color:
    return pygame.Color("#b3b4b6")

def default_disabled_background() -> pygame.Color:
    return pygame.Color(128, 128, 128)

def default_disabled_foreground() -> pygame.Color:
    return pygame.Color(192, 192, 192)

@dataclass
class ButtonColors:
    # Backgrounds
    background: pygame.Color = field(default_factory=default_background)
    hover_background: pygame.Color = field(default_factory=default_hover_background)
    disabled_background: pygame.Color = field(default_factory=default_disabled_background)

    # Borders ("foreground")
    foreground: pygame.Color = field(default_factory=default_foreground)
    hover_foreground: Optional[pygame.Color] = None
    disabled_foreground: pygame.Color = field(default_factory=default_disabled_foreground)


class Button(Frame):
    def __init__(
        self,
        function_to_call: Callable,
        x: int,
        y: int,
        width: int,
        height: int,
        text: str = "",
        colors: Optional[ButtonColors] = None,
        text_style: Optional[TextStyle] = None,
        args: Optional[Tuple] = None,
        args_provider: Optional[Callable[[], Tuple]] = None,
        shape: ButtonShape = ButtonShape.RECTANGLE,
        **frame_kwargs
    ):
        super().__init__(x=x, y=y, width=width, height=height, **frame_kwargs)

        self.function_to_call = function_to_call
        self.args = args or ()
        self.args_provider = args_provider
        self.shape = shape

        self.is_hover = False
        self.is_enabled = True
        self.colors = colors or ButtonColors()
        self.text_style = text_style or TextStyle(font_size=min(height - 4, 32))

        # If hover border isn't given, keep border consistent with normal foreground.
        if self.colors.hover_foreground is None:
            self.colors.hover_foreground = self.colors.foreground

        # Create text child if provided
        if text:
            self.text = Text(
                text,
                x=0.5, y=0.5,
                x_is_percent=True,
                y_is_percent=True,
                x_align="center",
                y_align="center",
                style=self.text_style
            )
            # Inherit initial state
            self.text.set_is_enabled(self.is_enabled)
            self.text.set_is_hover(self.is_hover)
            self.add_child(self.text)
        else:
            self.text = None

    @property
    def debug_outline_color(self) -> pygame.Color:
        return pygame.Color(0, 255, 0)

    def _point_in_diamond(self, x: int, y: int) -> bool:
        abs_x, abs_y, abs_w, abs_h = self.get_absolute_geometry()
        rel_x = x - abs_x
        rel_y = y - abs_y
        cx = abs_w / 2
        cy = abs_h / 2
        return (abs(rel_x - cx) / cx + abs(rel_y - cy) / cy) <= 1.0

    def _get_diamond_points(self) -> list:
        abs_x, abs_y, abs_w, abs_h = self.get_absolute_geometry()
        cx = abs_x + abs_w / 2
        cy = abs_y + abs_h / 2
        return [
            (cx, abs_y),               # top
            (abs_x + abs_w, cy),       # right
            (cx, abs_y + abs_h),       # bottom
            (abs_x, cy)                # left
        ]

    def contains_point(self, x: int, y: int) -> bool:
        if self.shape == ButtonShape.DIAMOND:
            return self._point_in_diamond(x, y)
        return super().contains_point(x, y)

    def on_click(self, button=None):
        if not self.is_enabled:
            return
        args = self.args_provider() if self.args_provider else self.args
        self.function_to_call(*args)

    def on_hover_enter(self):
        self.is_hover = True
        if self.text:
            self.text.set_is_hover(True)

    def on_hover_leave(self):
        self.is_hover = False
        if self.text:
            self.text.set_is_hover(False)

    def set_enabled(self, enabled: bool) -> None:
        if self.is_enabled != enabled:
            self.is_enabled = enabled
            if self.text:
                self.text.set_is_enabled(enabled)

    def set_text(self, text: str) -> None:
        if self.text:
            self.text.set_text(text)
        elif text:
            self.text = Text(
                text,
                x=0.5, y=0.5,
                x_is_percent=True,
                y_is_percent=True,
                x_align="center",
                y_align="center",
                style=self.text_style,
            )
            self.text.set_is_enabled(self.is_enabled)
            self.text.set_is_hover(self.is_hover)
            self.add_child(self.text)

    def set_shape(self, shape: ButtonShape) -> None:
        self.shape = shape

    def _resolve_colors(self):
        """Compute bg and border colors based on state (text handled by Text)."""
        if not self.is_enabled:
            return self.colors.disabled_background, self.colors.disabled_foreground
        if self.is_hover:
            return self.colors.hover_background, (self.colors.hover_foreground or self.colors.foreground)
        return self.colors.background, self.colors.foreground

    def draw(self, surface: pygame.Surface) -> None:
        if self.is_effectively_hidden:
            return

        abs_x, abs_y, abs_w, abs_h = self.get_absolute_geometry()
        bg_color, border_color = self._resolve_colors()

        # Draw geometry
        if self.shape == ButtonShape.DIAMOND:
            points = self._get_diamond_points()
            pygame.draw.polygon(surface, bg_color, points)
            pygame.draw.polygon(surface, border_color, points, 2)
        else:
            pygame.draw.rect(surface, bg_color, (abs_x, abs_y, abs_w, abs_h))
            pygame.draw.rect(surface, border_color, (abs_x, abs_y, abs_w, abs_h), 2)

        # Draw children
        for child in self.children:
            child.draw(surface)
