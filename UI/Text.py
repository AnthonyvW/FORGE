import pygame
import pygame.freetype as freetype
from typing import Tuple, Optional
from dataclasses import dataclass, field
from UI.frame import Frame

def default_color() -> pygame.Color:
    return pygame.Color("#b3b4b6")

@dataclass
class TextStyle:
    """Style configuration for text rendering"""
    color: pygame.Color = field(default_factory=default_color)
    font_size: int = 32
    font_name: Optional[str] = None
    bold: bool = False
    italic: bool = False

    hover_color: Optional[pygame.Color] = None
    disabled_color: Optional[pygame.Color] = None

class Text(Frame):
    def __init__(self, 
        text: str, 
        x: int, y: int, 
        style: Optional[TextStyle] = None,
        x_align: str = "left", y_align: str = "top",
        max_width: Optional[int] = None,
        truncate_mode: str = "none",
        show_tooltip_on_hover: bool = True,
        **frame_kwargs):

        super().__init__(x=x, y=y, width=0, height=0, **frame_kwargs)

        self.mouse_passthrough = True
        self.text = text
        self.style = style or TextStyle()
        self.x_align = x_align
        self.y_align = y_align
        self.max_width = max_width
        self.truncate_mode = truncate_mode
        self.show_tooltip_on_hover = show_tooltip_on_hover

        self._is_hover = False
        self._is_enabled = True

        self._font = self._create_font()
        self._surface = None
        self._render_text = text
        self._update_surface()

    @property
    def debug_outline_color(self) -> pygame.Color:
        return pygame.Color(0, 0, 255)

    def _create_font(self) -> freetype.Font:
        """Create a FreeType font object based on style"""
        try:
            font = freetype.Font(self.style.font_name, self.style.font_size)
        except Exception:
            font = freetype.SysFont(None, self.style.font_size)

        font.strong = self.style.bold
        font.oblique = self.style.italic
        return font

    def _current_color(self) -> pygame.Color:
        # Resolve stateful color with sensible fallbacks
        if not self._is_enabled and self.style.disabled_color is not None:
            return self.style.disabled_color
        if self._is_hover and self.style.hover_color is not None:
            return self.style.hover_color
        return self.style.color

    def _update_surface(self) -> None:
        """Render the text to a surface using FreeType"""
        if not self._font:
            return
        self._render_text = self.text
        if self.max_width and self.truncate_mode != "none":
            self._render_text = self._ellipsize(self.text, self.max_width, self.truncate_mode)
        self._surface, _ = self._font.render(
            self._render_text,
            fgcolor=self._current_color(),
            size=self.style.font_size,
        )

    @property
    def size(self) -> Tuple[int, int]:
        return self._surface.get_size() if self._surface else (0, 0)

    def contains_point(self, px, py):
        # allow hover detection
        abs_x, abs_y, w, h = self.get_absolute_geometry()
        return abs_x <= px <= abs_x + w and abs_y <= py <= abs_y + h

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

    def set_color(self, color) -> None:
        """Set the base text color and re-render the surface."""
        if not isinstance(color, pygame.Color):
            color = pygame.Color(color)
        if self.style.color != color:
            self.style.color = color
            self._update_surface()

    def get_color(self) -> pygame.Color:
        """Return the current resolved text color."""
        return self._current_color()

    def set_is_hover(self, is_hover: bool) -> None:
        if self._is_hover != is_hover:
            self._is_hover = is_hover
            self._update_surface()

    def set_is_enabled(self, is_enabled: bool) -> None:
        if self._is_enabled != is_enabled:
            self._is_enabled = is_enabled
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

    # --- Truncation helpers ---
    def _measure_width(self, s: str) -> int:
        return self._font.get_rect(s, size=self.style.font_size).width

    def _ellipsize(self, s: str, max_w: int, mode: str) -> str:
        if max_w is None:
            return s
        if self._measure_width(s) <= max_w:
            return s

        ell = "…"
        ell_w = self._measure_width(ell)
        if ell_w > max_w:
            return ""

        def fit_end(prefix: str) -> str:
            lo, hi = 0, len(prefix)
            best = ""
            while lo <= hi:
                mid = (lo + hi) // 2
                cand = prefix[:mid] + ell
                if self._measure_width(cand) <= max_w:
                    best = cand
                    lo = mid + 1
                else:
                    hi = mid - 1
            return best

        def fit_start(suffix: str) -> str:
            lo, hi = 0, len(suffix)
            best = ""
            while lo <= hi:
                mid = (lo + hi) // 2
                cand = ell + suffix[-mid:] if mid > 0 else ell
                if self._measure_width(cand) <= max_w:
                    best = cand
                    lo = mid + 1
                else:
                    hi = mid - 1
            return best

        if mode == "end":
            return fit_end(s)
        if mode == "start":
            return fit_start(s)
        if mode == "middle":
            left, right = 0, 0
            best = ell
            while left + right < len(s):
                cand = s[:left+1] + ell + (s[-right:] if right else "")
                if self._measure_width(cand) <= max_w:
                    left += 1
                    best = cand
                else:
                    break
                cand = s[:left] + ell + s[-(right+1):]
                if self._measure_width(cand) <= max_w:
                    right += 1
                    best = cand
                else:
                    break
            return best

        return s

    # --- Rendering ---
    def draw(self, surface: pygame.Surface) -> None:
        if not self._surface or self.is_effectively_hidden:
            return

        parent_x, parent_y, parent_w, parent_h = (
            self.parent.get_absolute_geometry() if self.parent else (0, 0, *surface.get_size())
        )

        draw_x = self.x * parent_w if self.x_is_percent else self.x
        draw_y = self.y * parent_h if self.y_is_percent else self.y
        draw_x += parent_x
        draw_y += parent_y

        text_w, text_h = self._surface.get_size()

        if self.x_align == "center":
            draw_x -= text_w // 2
        elif self.x_align == "right":
            draw_x -= text_w

        if self.y_align == "center":
            draw_y -= text_h // 2
        elif self.y_align == "bottom":
            draw_y -= text_h

        surface.blit(self._surface, (draw_x, draw_y))

        # Tooltip rendering
        if (
            self.show_tooltip_on_hover
            and self._render_text != self.text
        ):
            mx, my = pygame.mouse.get_pos()
            if self.contains_point(mx, my):
                self._draw_tooltip(surface, self.text, mx, my)

    def _draw_tooltip(self, surface, text, x, y):
        tooltip_font = self._create_font()
        tip_surface, _ = tooltip_font.render(text, fgcolor=pygame.Color("black"))
        padding = 6
        cursor_offset = 12
        margin = 6

        tw, th = tip_surface.get_size()
        rect = pygame.Rect(x + cursor_offset, y + cursor_offset, tw + padding * 2, th + padding * 2)
        sw, sh = surface.get_size()

        if rect.right > sw - margin:
            rect.x = x - cursor_offset - rect.w
        if rect.right > sw - margin:
            rect.x = sw - rect.w - margin
        if rect.x < margin:
            rect.x = margin

        if rect.bottom > sh - margin:
            rect.y = y - cursor_offset - rect.h
        if rect.bottom > sh - margin:
            rect.y = sh - rect.h - margin
        if rect.y < margin:
            rect.y = margin

        pygame.draw.rect(surface, pygame.Color(255, 255, 224), rect)
        pygame.draw.rect(surface, pygame.Color("black"), rect, 1)
        surface.blit(tip_surface, (rect.x + padding, rect.y + padding))
