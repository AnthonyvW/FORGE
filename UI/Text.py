import pygame
from typing import Tuple, Optional
from dataclasses import dataclass, field
from typing import Callable

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

class Text:
    def __init__(
        self,
        text: str,
        x: int,
        y: int,
        style: Optional[TextStyle] = None
    ):
        """
        Initialize a text object with position and styling.
        
        Args:
            text: The text to display
            x: X-coordinate of text position
            y: Y-coordinate of text position
            style: Optional custom text style
        """
        self.text = text
        self.x = x
        self.y = y
        self.style = style or TextStyle()
        
        self._font = self._create_font()
        self._surface = None
        self._rect = None
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
        self._rect = self._surface.get_rect(center=(self.x, self.y))
    
    @property
    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)
    
    @position.setter
    def position(self, pos: Tuple[int, int]) -> None:
        self.x, self.y = pos
        if self._rect:
            self._rect.center = pos
    
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
    
    def update_position(self, x_offset: int, y_offset: int) -> None:
        """Update text position by the given offset"""
        self.x += x_offset
        self.y += y_offset
        if self._rect:
            self._rect.center = (self.x, self.y)
    
    def draw(self, surface: pygame.Surface) -> None:
        """Draw the text on the given surface"""
        if self._surface and self._rect:
            surface.blit(self._surface, self._rect)