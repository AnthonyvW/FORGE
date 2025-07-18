import pygame
from typing import Callable, Optional, Tuple
from dataclasses import dataclass, field
from UI.text import Text, TextStyle
from UI.frame import Frame

def default_background() -> pygame.Color:
    return pygame.Color(32, 128, 32)

def default_foreground() -> pygame.Color:
    return pygame.Color(64, 255, 64)

def default_hover_background() -> pygame.Color:
    return pygame.Color(128, 128, 255)

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
        **frame_kwargs
    ):
        super().__init__(x=x, y=y, width=width, height=height, **frame_kwargs)

        self.function_to_call = function_to_call
        self.args = args or ()
        self.args_provider = args_provider

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
                style=text_style
            )
            self.add_child(self.text)
        else:
            self.text = None

    def on_click(self):
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

    def draw(self, surface: pygame.Surface) -> None:
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

        pygame.draw.rect(surface, bg_color, (abs_x, abs_y, abs_w, abs_h))
        pygame.draw.rect(surface, fg_color, (abs_x, abs_y, abs_w, abs_h), 2)

        for child in self.children:
            child.draw(surface)
