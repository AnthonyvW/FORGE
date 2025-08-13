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
        max_width: Optional[int] = None,          # NEW: constrain width for rendering
        truncate_mode: str = "none",              # NEW: 'none' | 'end' | 'middle' | 'start'
        **frame_kwargs):

        super().__init__(x=x, y=y, width=0, height=0, **frame_kwargs)

        self.text = text
        self.style = style or TextStyle()
        self.x_align = x_align
        self.y_align = y_align
        self.max_width = max_width
        self.truncate_mode = truncate_mode
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
        if not self._font:
            return
        render_text = self.text
        if self.max_width and self.truncate_mode != "none":
            render_text = self._ellipsize(render_text, self.max_width, self.truncate_mode)
        self._surface, _ = self._font.render(
            render_text,
            fgcolor=self.style.color,
            size=self.style.font_size,
        )

    @property
    def size(self) -> Tuple[int, int]:
        return self._surface.get_size() if self._surface else (0, 0)

    def contains_point(self, px, py):
        return False

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

    # --- Truncation Properties ---
    def _measure_width(self, s: str) -> int:
        # pygame.freetype returns metrics via get_rect
        return self._font.get_rect(s, size=self.style.font_size).width

    def _ellipsize(self, s: str, max_w: int, mode: str) -> str:
        if max_w is None:
            return s
        if self._measure_width(s) <= max_w:
            return s

        ell = "…"
        ell_w = self._measure_width(ell)
        if ell_w > max_w:
            return ""  # degenerate case

        # helpers
        def fit_end(prefix: str) -> str:
            # keep from start, cut at end
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
            # keep from end, cut at start
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
            # keep from both ends, cut the middle
            left, right = 0, 0
            best = ell
            # two-pointer growth until it no longer fits
            while left + right < len(s):
                # try grow left
                cand = s[:left+1] + ell + (s[-right:] if right else "")
                if self._measure_width(cand) <= max_w:
                    left += 1
                    best = cand
                else:
                    break
                # try grow right
                cand = s[:left] + ell + s[-(right+1):]
                if self._measure_width(cand) <= max_w:
                    right += 1
                    best = cand
                else:
                    break
            return best

        # default
        return s

    # --- Rendering Properties ---
    
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

