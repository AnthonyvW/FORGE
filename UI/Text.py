import pygame
from typing import Tuple, Optional
from dataclasses import dataclass, field
from UI.frame import Frame

def default_color() -> pygame.Color:
    return pygame.Color(255, 255, 255)

@dataclass
class TextStyle:
    """Style configuration for text rendering"""
    color: pygame.Color = field(default_factory=default_color)
    font_size: int = 32
    font_name: Optional[str] = None
    bold: bool = False
    italic: bool = False
    anti_alias: bool = True

class Text(Frame):
    def __init__(self, text: str, x: int, y: int, style: Optional[TextStyle] = None, **frame_kwargs):
        super().__init__(x=x, y=y, width=0, height=0, **frame_kwargs)

        self.text = text
        self.style = style or TextStyle()
        self._font = self._create_font()
        self._surface = None
        self._update_surface()

    def _create_font(self) -> pygame.font.Font:
        """Create pygame font object based on style"""
        if self.style.font_name:
            try:
                font = pygame.font.Font(self.style.font_name, self.style.font_size)
            except pygame.error:
                font = pygame.font.Font(None, self.style.font_size)
        else:
            font = pygame.font.Font(None, self.style.font_size)
            
        font.bold = self.style.bold
        font.italic = self.style.italic
        return font

    def _update_surface(self) -> None:
        """Update the text surface and rectangle"""
        self._surface = self._font.render(
            self.text,
            self.style.anti_alias,
            self.style.color
        )

    @property
    def size(self) -> Tuple[int, int]:
        return self._surface.get_size() if self._surface else (0, 0)

    def set_text(self, text: str) -> None:
        """Update the displayed text"""
        if self.text != text:
            self.text = text
            self._update_surface()
    
    def set_style(self, style: TextStyle) -> None:
        """Update the text style"""
        self.style = style
        self._font = self._create_font()
        self._update_surface()
    
    def draw(self, surface: pygame.Surface) -> None:
        if self._surface:
            abs_x, abs_y, abs_w, abs_h = self.get_absolute_geometry()
            rect = self._surface.get_rect(center=(abs_x + abs_w / 2, abs_y + abs_h / 2))
            surface.blit(self._surface, rect)