from __future__ import annotations
import pygame
from UI.text import TextStyle
from UI.input.button import ButtonColors
from UI.input.radio import SelectedColors


# ---- Text Styles -----------------------------------------------------------

def make_button_text_style() -> TextStyle:
    return TextStyle(color=pygame.Color("#5a5a5a"), font_size=20)

def make_display_text_style(font_size = 18) -> TextStyle:
    return TextStyle(
    color=pygame.Color(32, 32, 32),
    font_size=font_size,
    font_name="assets/fonts/SofiaSans-Regular.ttf",
)

def make_settings_text_style() -> TextStyle:
    return TextStyle(
    color=pygame.Color(32, 32, 32),
    font_size=20,
    font_name="assets/fonts/SofiaSans-Regular.ttf",
)


# ---- Radio / Button shared styling ----------------------------------------


# Base (unselected) colors for radio buttons
BASE_BUTTON_COLORS = ButtonColors(
    hover_foreground=pygame.Color("#5a5a5a")
)

# Colors when a radio is selected
SELECTED_RADIO_COLORS = SelectedColors(
    background=pygame.Color("#b3b4b6"),
    hover_background=pygame.Color("#b3b4b6"),
    foreground=pygame.Color("#b3b4b6"),
    hover_foreground=pygame.Color("#5a5a5a"),
)

# Text style used by radios
RADIO_TEXT_STYLE = TextStyle(
    font_size=16,
    color=pygame.Color("#5a5a5a"),
    hover_color=pygame.Color("#5a5a5a"),
    disabled_color=pygame.Color("#5a5a5a"),
)