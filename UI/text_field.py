import pygame
from UI.frame import Frame
from UI.text import Text, TextStyle

class TextField(Frame):
    def __init__(self, parent=None, x=0, y=0, width=200, height=30,
                 placeholder="", style=None,
                 background_color=pygame.Color("white"),
                 text_color=pygame.Color("black"),
                 padding=5,
                 **kwargs):
        super().__init__(parent=parent, x=x, y=y, width=width, height=height,
                         background_color=background_color, **kwargs)

        self.active = False
        self.placeholder = placeholder
        self.text = ""
        self.style = style or TextStyle(color=text_color, font_size=18)

        # Rendered text element inside the field
        self._text = Text(self.placeholder, parent=self, x=padding, y=height // 2,
                          x_align="left", y_align="center", style=self.style)

        # Caret state
        self._padding = padding
        self._caret_index = 0
        self._caret_visible = True
        self._blink_interval_ms = 500
        self._last_blink_ms = pygame.time.get_ticks()

    # --- focus management: respond to global clicks anywhere ---
    def on_global_mouse_press(self, px, py, button):
        was_active = self.active
        self.active = self.contains_point(px, py)

        # Place caret at end on focus change in (simple behavior).
        if self.active and not was_active:
            self._caret_index = len(self.text)
            self._reset_blink()
        elif not self.active and was_active:
            self._caret_visible = False  # hide when unfocused

    # --- typing: react only when active ---
    def on_key_event(self, event):
        print("TextField.on_key_event", event.key, repr(getattr(event, "unicode", "")), "active=", self.active, "id=", id(self))
        if not self.active or event.type != pygame.KEYDOWN:
            return

        if event.key == pygame.K_RETURN:
            self.active = False
            self._caret_visible = False
            return

        if event.key == pygame.K_BACKSPACE:
            if self._caret_index > 0:
                self.text = self.text[:self._caret_index - 1] + self.text[self._caret_index:]
                self._caret_index -= 1
                self._refresh(); self._reset_blink()
            return

        if event.key == pygame.K_DELETE:
            if self._caret_index < len(self.text):
                self.text = self.text[:self._caret_index] + self.text[self._caret_index + 1:]
                self._refresh(); self._reset_blink()
            return

        if event.key == pygame.K_LEFT:
            if self._caret_index > 0:
                self._caret_index -= 1; self._reset_blink()
            return

        if event.key == pygame.K_RIGHT:
            if self._caret_index < len(self.text):
                self._caret_index += 1; self._reset_blink()
            return

        if event.key == pygame.K_HOME:
            self._caret_index = 0; self._reset_blink(); return

        if event.key == pygame.K_END:
            self._caret_index = len(self.text); self._reset_blink(); return

        # Printable char insertion (KEYDOWN-only)
        if event.unicode and event.unicode.isprintable():
            self.text = self.text[:self._caret_index] + event.unicode + self.text[self._caret_index:]
            self._caret_index += 1
            self._refresh(); self._reset_blink()

    def _refresh(self):
        if self.text:
            self._text.set_text(self.text)
        else:
            self._text.set_text(self.placeholder)
        # Clamp caret to bounds
        self._caret_index = max(0, min(self._caret_index, len(self.text)))

    def _reset_blink(self):
        self._last_blink_ms = pygame.time.get_ticks()
        self._caret_visible = True

    def _update_blink(self):
        now = pygame.time.get_ticks()
        if now - self._last_blink_ms >= self._blink_interval_ms:
            self._caret_visible = not self._caret_visible
            self._last_blink_ms = now

    def _measure_text_prefix_width(self, prefix: str) -> int:
        """
        Measure pixel width of a substring using the Text's FreeType font.
        Falls back gracefully if font unavailable.
        """
        font = getattr(self._text, "_font", None)
        if not font:
            # crude fallback: render via Text then read width
            temp = Text(prefix, parent=self, x=0, y=0, style=self.style)
            w, _ = temp.size
            # don't add temp to tree permanently
            self.children.remove(temp)
            return w
        # With pygame.freetype, render() returns (surface, rect)
        surf, _ = font.render(prefix, fgcolor=self.style.color, size=self.style.font_size)
        return surf.get_width()

    def draw(self, surface):
        # Background + border
        abs_x, abs_y, abs_w, abs_h = self.get_absolute_geometry()
        pygame.draw.rect(surface, self.background_color, (abs_x, abs_y, abs_w, abs_h))
        border = pygame.Color("dodgerblue") if self.active else pygame.Color("black")
        pygame.draw.rect(surface, border, (abs_x, abs_y, abs_w, abs_h), 2)

        # Clip text to inner rect (simple visual safety)
        clip_rect = pygame.Rect(abs_x + 2, abs_y + 2, max(0, abs_w - 4), max(0, abs_h - 4))
        prev_clip = surface.get_clip()
        surface.set_clip(clip_rect)

        # Draw the text
        self._text.draw(surface)

        # Caret
        if self.active:
            self._update_blink()
            if self._caret_visible:
                # Compute caret x at the insertion point
                prefix = self.text[:self._caret_index]
                prefix_w = self._measure_text_prefix_width(prefix)
                # Align with our internal Text baseline
                text_x, text_y, text_w, text_h = self._text.get_absolute_geometry()
                caret_x = text_x + prefix_w
                caret_y = text_y
                caret_h = text_h
                pygame.draw.line(surface, pygame.Color("black"),
                                 (caret_x, caret_y),
                                 (caret_x, caret_y + caret_h), 1)

        # Restore clip
        surface.set_clip(prev_clip)
