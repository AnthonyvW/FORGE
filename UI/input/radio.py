import pygame
from typing import Callable, Optional, Any, List
from dataclasses import dataclass, field

from UI.input.button import Button, ButtonColors
from UI.text import TextStyle

@dataclass
class SelectedColors:
    """Optional override palette when a RadioButton is selected."""
    background: Optional[pygame.Color] = None
    hover_background: Optional[pygame.Color] = None
    foreground: Optional[pygame.Color] = None  # border color when selected
    hover_foreground: Optional[pygame.Color] = None

class RadioGroup:
    """
    Manages an exclusive group of RadioButtons.
    - Only one RadioButton can be selected at a time.
    - Optionally allow deselection by clicking the already-selected button.
    """
    def __init__(
        self,
        allow_deselect: bool = False,
        on_change: Optional[Callable[[Optional["RadioButton"]], None]] = None
    ):
        self._buttons: List["RadioButton"] = []
        self._selected: Optional["RadioButton"] = None
        self.allow_deselect = allow_deselect
        self.on_change = on_change

    def add(self, btn: "RadioButton") -> None:
        if btn not in self._buttons:
            self._buttons.append(btn)
            btn._group = self
            if btn.is_selected:
                self.select(btn, fire=False)

    def remove(self, btn: "RadioButton") -> None:
        if btn in self._buttons:
            was_selected = (btn is self._selected)
            self._buttons.remove(btn)
            btn._group = None
            if was_selected:
                self._selected = None
                if self.on_change:
                    self.on_change(None)

    def select(self, btn: Optional["RadioButton"], fire: bool = True) -> None:
        if btn is None:
            if self._selected:
                self._selected._set_selected(False)
                self._selected = None
                if fire and self.on_change:
                    self.on_change(None)
            return

        if btn not in self._buttons:
            self.add(btn)

        if self._selected is btn:
            if self.allow_deselect:
                # Deselect current
                btn._set_selected(False)
                self._selected = None
                if fire and self.on_change:
                    self.on_change(None)
            # else: clicking again does nothing
            return

        # Deselect previous
        if self._selected:
            self._selected._set_selected(False)

        # Select new
        btn._set_selected(True)
        self._selected = btn
        if fire and self.on_change:
            self.on_change(btn)

    def get_selected(self) -> Optional["RadioButton"]:
        return self._selected

    def get_value(self) -> Optional[Any]:
        return self._selected.value if self._selected else None

    def set_value(self, value: Any, fire: bool = True) -> None:
        for b in self._buttons:
            if b.value == value:
                self.select(b, fire=fire)
                return
        # If value not found, deselect all
        self.select(None, fire=fire)


class RadioButton(Button):
    """
    A Button that participates in a RadioGroup.
    - Clicking selects this button in its group (exclusive).
    - Appearance changes when selected (via SelectedColors).
    - You can still pass a per-button callback (function_to_call).
    """
    def __init__(
        self,
        function_to_call,
        x, y, width, height,
        text: str = "",
        *,
        value: Any = None,
        group: Optional[RadioGroup] = None,
        selected: bool = False,
        colors: Optional[ButtonColors] = None,
        selected_colors: Optional[SelectedColors] = None,
        text_style: Optional[TextStyle] = None,
        **frame_kwargs
    ):
        super().__init__(
            function_to_call=function_to_call,
            x=x, y=y, width=width, height=height,
            text=text, colors=colors, text_style=text_style, **frame_kwargs
        )
        self.value = value if value is not None else text
        self._group: Optional[RadioGroup] = None
        self._is_selected: bool = False
        self._selected_colors = selected_colors or SelectedColors()

        if group:
            group.add(self)
        if selected:
            # defer to group to enforce exclusivity
            (group or self._group).select(self, fire=False)

    # --- Public API ---
    @property
    def is_selected(self) -> bool:
        return self._is_selected

    def set_selected(self, selected: bool, fire: bool = True) -> None:
        """
        Ask the group to set selection (preferred).
        If no group, sets locally.
        """
        if self._group:
            self._group.select(self if selected else None, fire=fire)
        else:
            self._set_selected(selected)

    # --- Internal state change (no group notifications) ---
    def _set_selected(self, selected: bool) -> None:
        if self._is_selected == selected:
            return
        self._is_selected = selected
        # If you want to adjust child Text style on selection, you can do it here.

    # --- Button overrides ---
    def on_click(self, button=None):
        if not self.is_enabled:
            return
        # Selection logic first (so callbacks see the new state)
        if self._group:
            self._group.select(self)
        else:
            self._set_selected(True)

        # Optional per-button callback
        super().on_click(button)

    def _resolve_colors(self):
        """
        Extend Button color resolution to account for selected state.
        We only override backgrounds/borders—your Text color stays independent.
        """
        base_bg, base_border = super()._resolve_colors()
        if not self._is_selected:
            return base_bg, base_border

        # Selected palette with graceful fallback to base colors
        bg = self._selected_colors.background or base_bg
        fg = self._selected_colors.foreground or base_border

        if self.is_hover:
            bg = self._selected_colors.hover_background or bg
            fg = self._selected_colors.hover_foreground or fg

        return bg, fg
