import pygame
import pygame.freetype as freetype
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

class Text(Frame):
    def __init__(self, 
        text: str, 
        x: int, y: int, 
        style: Optional[TextStyle] = None,
        x_align: str = "left", y_align: str = "top", 
        **frame_kwargs):

        super().__init__(x=x, y=y, width=0, height=0, **frame_kwargs)

        self.text = text
        self.style = style or TextStyle()
        self.x_align = x_align
        self.y_align = y_align
        self._font = self._create_font()
        self._surface = None
        self._update_surface()

    @property
    def debug_outline_color(self) -> pygame.Color:
        return pygame.Color(0, 0, 255)  # Default: red

    def _create_font(self) -> freetype.Font:
        """Create a FreeType font object based on style"""
        try:
            font = freetype.Font(self.style.font_name, self.style.font_size)
        except Exception:
            font = freetype.SysFont(None, self.style.font_size)

        font.strong = self.style.bold
        font.oblique = self.style.italic
        return font

    def _update_surface(self) -> None:
        """Render the text to a surface using FreeType"""
        if self._font:
            self._surface, _ = self._font.render(
                self.text,
                fgcolor=self.style.color,
                size=self.style.font_size,
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

    def get_absolute_geometry(self):
        parent_x, parent_y, parent_w, parent_h = (
            self.parent.get_absolute_geometry() if self.parent else (0, 0, *pygame.display.get_surface().get_size())
        )

        draw_x = self.x * parent_w if self.x_is_percent else self.x
        draw_y = self.y * parent_h if self.y_is_percent else self.y
        draw_x += parent_x
        draw_y += parent_y

        text_w, text_h = self._surface.get_size() if self._surface else (0, 0)

        # Apply alignment like in draw()
        if self.x_align == "center":
            draw_x -= text_w // 2
        elif self.x_align == "right":
            draw_x -= text_w

        if self.y_align == "center":
            draw_y -= text_h // 2
        elif self.y_align == "bottom":
            draw_y -= text_h

        return draw_x, draw_y, text_w, text_h

    
    def draw(self, surface: pygame.Surface) -> None:
        if not self._surface:
            return

        parent_x, parent_y, parent_w, parent_h = (
            self.parent.get_absolute_geometry() if self.parent else (0, 0, *surface.get_size())
        )

        # Resolve anchor point in absolute coordinates
        draw_x = self.x * parent_w if self.x_is_percent else self.x
        draw_y = self.y * parent_h if self.y_is_percent else self.y
        draw_x += parent_x
        draw_y += parent_y

        text_w, text_h = self._surface.get_size()

        # Apply alignment
        if self.x_align == "center":
            draw_x -= text_w // 2
        elif self.x_align == "right":
            draw_x -= text_w

        if self.y_align == "center":
            draw_y -= text_h // 2
        elif self.y_align == "bottom":
            draw_y -= text_h

        surface.blit(self._surface, (draw_x, draw_y))

