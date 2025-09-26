import pygame
from dataclasses import dataclass
from typing import Callable, Optional

from UI.input.button import Button, ButtonColors
from UI.text import TextStyle

@dataclass
class ToggledColors:
    background: Optional[pygame.Color] = None
    hover_background: Optional[pygame.Color] = None
    foreground: Optional[pygame.Color] = None
    hover_foreground: Optional[pygame.Color] = None

class ToggleButton(Button):
    """
    Two-state button (ON/OFF).
    - `on_click` is optional.
    - `on_change` receives (state: bool, button: ToggleButton).
    - Color palette can be overridden when ON via ToggledColors.
    """
    def __init__(
        self,
        function_to_call: Optional[Callable] = None,   # optional click callback
        x: int = 0, y: int = 0, width: int = 100, height: int = 30,
        text: str = "",
        *,
        toggled: bool = False,
        on_change: Optional[Callable[[bool, "ToggleButton"], None]] = None,
        colors: Optional[ButtonColors] = None,
        toggled_colors: Optional[ToggledColors] = None,
        text_style: Optional[TextStyle] = None,
        **frame_kwargs
    ):
        super().__init__(
            function_to_call=function_to_call,
            x=x, y=y, width=width, height=height,
            text=text, colors=colors, text_style=text_style,
            **frame_kwargs
        )
        self._is_on = bool(toggled)
        self._on_change = on_change
        self._toggled_colors = toggled_colors or ToggledColors()

    @property
    def is_on(self) -> bool:
        return self._is_on

    def set_toggled(self, on: bool, fire: bool = True) -> None:
        if self._is_on == on:
            return
        self._is_on = on
        if fire and self._on_change:
            self._on_change(self._is_on, self)

    def toggle(self, fire: bool = True) -> None:
        self.set_toggled(not self._is_on, fire=fire)

    def on_click(self, button=None):
        if not self.is_enabled:
            return
        # Flip state first
        self.toggle(fire=True)
        # Only call per-click handler if provided
        if self.function_to_call:
            super().on_click(button)

    def _resolve_colors(self):
        base_bg, base_border = super()._resolve_colors()
        if not self._is_on:
            return base_bg, base_border

        bg = self._toggled_colors.background or base_bg
        fg = self._toggled_colors.foreground or base_border
        if self.is_hover:
            bg = self._toggled_colors.hover_background or bg
            fg = self._toggled_colors.hover_foreground or fg
        return bg, fg
