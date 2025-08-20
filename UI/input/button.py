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

def default_foreground() -> pygame.Color:  # used for BORDER unless text colors are unset
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

    # Text colors
    text: Optional[pygame.Color] = None
    hover_text: Optional[pygame.Color] = None
    disabled_text: Optional[pygame.Color] = None


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

        # If hover border isn't given, keep border consistent with normal foreground.
        if self.colors.hover_foreground is None:
            self.colors.hover_foreground = self.colors.foreground

        # Create text child (if provided) using explicit text color,
        # falling back to border color only when not defined.
        if text:
            if not text_style:
                base_text_color = self.colors.text if self.colors.text is not None else self.colors.foreground
                text_style = TextStyle(
                    color=base_text_color,
                    font_size=min(height - 4, 32)
                )
            self.text = Text(
                text,
                x=0.5, y=0.5,
                x_is_percent=True,
                y_is_percent=True,
                x_align="center",
                y_align="center",
                style=text_style
            )
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

    def on_hover_leave(self):
        self.is_hover = False

    def set_text(self, text: str) -> None:
        if self.text:
            self.text.set_text(text)
        elif text:
            # Respect explicit text color, fall back to border if not provided
            base_text_color = self.colors.text if self.colors.text is not None else self.colors.foreground
            self.text = Text(
                text,
                x=0.5, y=0.5,
                x_is_percent=True,
                y_is_percent=True,
                x_align="center",
                y_align="center",
                style=TextStyle(color=base_text_color)
            )
            self.add_child(self.text)

    def set_shape(self, shape: ButtonShape) -> None:
        self.shape = shape

    def _resolve_colors(self):
        """Compute bg, border, and text colors based on state with the new text semantics."""
        if not self.is_enabled:
            bg = self.colors.disabled_background
            border = self.colors.disabled_foreground
            # text falls back to disabled_foreground only if disabled_text is unset
            txt = self.colors.disabled_text if self.colors.disabled_text is not None else self.colors.disabled_foreground
            return bg, border, txt

        if self.is_hover:
            bg = self.colors.hover_background
            border = self.colors.hover_foreground if self.colors.hover_foreground is not None else self.colors.foreground

            # Hover text: prefer explicit hover_text, else normal text if set,
            # else fall back to border color to preserve legacy behavior.
            if self.colors.hover_text is not None:
                txt = self.colors.hover_text
            elif self.colors.text is not None:
                txt = self.colors.text
            else:
                txt = self.colors.foreground
            return bg, border, txt

        # Normal state
        bg = self.colors.background
        border = self.colors.foreground
        txt = self.colors.text if self.colors.text is not None else self.colors.foreground
        return bg, border, txt

    def draw(self, surface: pygame.Surface) -> None:
        if self.is_effectively_hidden:
            return

        abs_x, abs_y, abs_w, abs_h = self.get_absolute_geometry()
        bg_color, border_color, text_color = self._resolve_colors()

        # Draw geometry
        if self.shape == ButtonShape.DIAMOND:
            points = self._get_diamond_points()
            pygame.draw.polygon(surface, bg_color, points)
            pygame.draw.polygon(surface, border_color, points, 2)
        else:
            pygame.draw.rect(surface, bg_color, (abs_x, abs_y, abs_w, abs_h))
            pygame.draw.rect(surface, border_color, (abs_x, abs_y, abs_w, abs_h), 2)

        # Update text color just-in-time so hovering/disabled reflects immediately
        if self.text:
            self.text.set_color(text_color)

        # Draw children
        for child in self.children:
            child.draw(surface)
