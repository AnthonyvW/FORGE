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

def default_foreground() -> pygame.Color:
    return pygame.Color("#b3b4b6")

def default_hover_background() -> pygame.Color:
    return pygame.Color("#b3b4b6")

def default_disabled_background() -> pygame.Color:
    return pygame.Color(128, 128, 128)

def default_disabled_foreground() -> pygame.Color:
    return pygame.Color(192, 192, 192)

@dataclass
class ButtonColors:
    background: pygame.Color = field(default_factory=default_background)
    foreground: pygame.Color = field(default_factory=default_foreground)
    hover_background: pygame.Color = field(default_factory=default_hover_background)
    disabled_background: pygame.Color = field(default_factory=default_disabled_background)
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

        if text:
            if not text_style:
                text_style = TextStyle(
                    color=self.colors.foreground,
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
        return pygame.Color(0, 255, 0)  # Default: red

    def _point_in_diamond(self, x: int, y: int) -> bool:
        """Check if a point is inside the diamond shape"""
        abs_x, abs_y, abs_w, abs_h = self.get_absolute_geometry()
        
        # Convert to relative coordinates within the button
        rel_x = x - abs_x
        rel_y = y - abs_y
        
        # Diamond center
        center_x = abs_w / 2
        center_y = abs_h / 2
        
        # Check if point is within diamond bounds using Manhattan distance
        # Diamond equation: |x - cx|/half_width + |y - cy|/half_height <= 1
        return (abs(rel_x - center_x) / center_x + abs(rel_y - center_y) / center_y) <= 1.0

    def _get_diamond_points(self) -> list:
        """Get the four corner points of the diamond"""
        abs_x, abs_y, abs_w, abs_h = self.get_absolute_geometry()
        
        center_x = abs_x + abs_w / 2
        center_y = abs_y + abs_h / 2
        
        # Diamond points: top, right, bottom, left
        points = [
            (center_x, abs_y),                    # top
            (abs_x + abs_w, center_y),           # right
            (center_x, abs_y + abs_h),           # bottom
            (abs_x, center_y)                    # left
        ]
        return points

    def contains_point(self, x: int, y: int) -> bool:
        """Override to handle diamond shape hit detection"""
        if self.shape == ButtonShape.DIAMOND:
            return self._point_in_diamond(x, y)
        else:
            # Use default rectangular hit detection
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
            self.text = Text(
                text,
                x=0.5, y=0.5,
                x_is_percent=True,
                y_is_percent=True,
                style=TextStyle(color=self.colors.foreground)
            )
            self.add_child(self.text)

    def set_shape(self, shape: ButtonShape) -> None:
        """Change the button shape"""
        self.shape = shape

    def draw(self, surface: pygame.Surface) -> None:
        if self.is_hidden:
            return
            
        abs_x, abs_y, abs_w, abs_h = self.get_absolute_geometry()
        
        if not self.is_enabled:
            bg_color = self.colors.disabled_background
            fg_color = self.colors.disabled_foreground
        elif self.is_hover:
            bg_color = self.colors.hover_background
            fg_color = self.colors.foreground
        else:
            bg_color = self.colors.background
            fg_color = self.colors.foreground

        if self.shape == ButtonShape.DIAMOND:
            # Draw diamond shape
            points = self._get_diamond_points()
            pygame.draw.polygon(surface, bg_color, points)
            pygame.draw.polygon(surface, fg_color, points, 2)
        else:
            # Draw rectangle (default behavior)
            pygame.draw.rect(surface, bg_color, (abs_x, abs_y, abs_w, abs_h))
            pygame.draw.rect(surface, fg_color, (abs_x, abs_y, abs_w, abs_h), 2)

        for child in self.children:
            child.draw(surface)