import unicodedata
import re

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
        
        # --- key repeat state ---
        self._repeat_key = None
        self._repeat_delay_ms = 350     # first repeat delay (ms)
        self._repeat_interval_ms = 40   # subsequent repeats (ms)
        self._next_repeat_ms = 0

    # --- focus management: respond to global clicks anywhere ---
    def on_global_mouse_press(self, px, py, button):
        was_active = self.active
        self.active = self.contains_point(px, py)

        if self.active and not was_active:
            self._caret_index = len(self.text)
            self._reset_blink()
        elif not self.active and was_active:
            self._caret_visible = False
            self._repeat_key = None      # NEW: stop repeating when losing focus

    # --- helper for a single backspace action ---
    def _do_backspace(self):
        if self._caret_index > 0:
            self.text = self.text[:self._caret_index - 1] + self.text[self._caret_index:]
            self._caret_index -= 1
            self._refresh(); self._reset_blink()

    # --- per-frame repeat tick ---
    def _update_key_repeat(self):
        if not self.active or self._repeat_key is None:
            return
        now = pygame.time.get_ticks()
        while now >= self._next_repeat_ms:
            if self._repeat_key == pygame.K_BACKSPACE:
                self._do_backspace()
            # (extend here for DELETE, arrows, etc. if you want)
            self._next_repeat_ms += self._repeat_interval_ms

    def _decode_clip_bytes(self, raw: bytes) -> str:
        """Best-effort decode for clipboard bytes from pygame.scrap/SDL."""
        # BOMs first
        if raw.startswith(b'\xff\xfe') or raw.startswith(b'\xfe\xff'):
            try:
                return raw.decode('utf-16')
            except Exception:
                pass
        # Heuristic: lots of NULs => UTF-16 without BOM (Windows CF_UNICODETEXT)
        if raw and raw[1:2] == b'\x00' or raw.count(b'\x00') >= max(1, len(raw)//4):
            for enc in ('utf-16-le', 'utf-16-be'):
                try:
                    return raw.decode(enc)
                except Exception:
                    continue
        # Try utf-8, then latin1 as last resort
        for enc in ('utf-8', 'latin1'):
            try:
                return raw.decode(enc)
            except Exception:
                continue
        # Fallback replace to avoid exceptions
        return raw.decode('utf-8', errors='replace')

    def _sanitize_paste_text(self, s: str) -> str:
        """Normalize, strip invisibles, and make it safe for a single-line field."""
        # Normalize
        s = unicodedata.normalize('NFC', s)

        # Replace non-breaking spaces and other common whitespace oddities with space
        s = s.replace('\u00A0', ' ')  # nbsp
        s = s.replace('\u2007', ' ')  # figure space
        s = s.replace('\u202F', ' ')  # narrow nbsp

        # Drop zero-width and format chars (keep newline handling for now)
        # Cf (format), Cc (control) — but allow \n and \t to survive for next step
        s = ''.join(ch for ch in s if not (
            (unicodedata.category(ch) in ('Cf', 'Cc')) and ch not in ('\n', '\t')
        ))

        # Optional: map curly quotes if your font lacks them
        # Comment these out if your font supports U+2018/2019 properly
        s = s.replace('‘', "'").replace('’', "'").replace('“', '"').replace('”', '"')

        # Normalize line endings and collapse to single line for this widget
        s = s.replace('\r\n', '\n').replace('\r', '\n').replace('\t', ' ')
        s = " ".join(s.splitlines())  # turns any newlines into single spaces

        # Trim trailing exotic whitespace that might still linger
        s = re.sub(r'\s+$', ' ', s)  # keep one space if user had one at end
        return s

    # --- Paste functionality ---
    def _insert_text(self, s: str):
        if not s:
            return
        s = self._sanitize_paste_text(s)
        if not s:
            return
        self.text = self.text[:self._caret_index] + s + self.text[self._caret_index:]
        self._caret_index += len(s)
        self._refresh()
        self._reset_blink()
        if getattr(self, "_repeat_key", None) == pygame.K_BACKSPACE:
            self._repeat_key = None

    def _get_clipboard_text(self) -> str:
        """Clipboard : try pygame.scrap, then pygame.clipboard; robust decoding."""
        # 1) pygame.scrap
        try:
            if hasattr(pygame, "scrap"):
                if not pygame.scrap.get_init():
                    pygame.scrap.init()
                raw = (
                    pygame.scrap.get("text/plain;charset=utf-8")
                    or pygame.scrap.get("text/plain")
                    or pygame.scrap.get(getattr(pygame, "SCRAP_TEXT", "text/plain"))
                )
                if raw:
                    if isinstance(raw, bytes):
                        return self._decode_clip_bytes(raw)
                    return str(raw)
        except Exception:
            pass

        # 2) SDL clipboard
        try:
            if hasattr(pygame, "clipboard"):
                txt = pygame.clipboard.get_text()
                if isinstance(txt, bytes):
                    return self._decode_clip_bytes(txt)
                return txt or ""
        except Exception:
            pass

        return ""

    # --- typing: react to KEYDOWN + KEYUP ---
    def on_key_event(self, event):
        if not self.active:
            return

        # handle KEYUP to stop repeat
        if event.type == pygame.KEYUP:
            if event.key == self._repeat_key:
                self._repeat_key = None
            return

        if event.type != pygame.KEYDOWN:
            return

        mods = getattr(event, "mod", 0)
        is_ctrl_or_cmd = bool(mods & (pygame.KMOD_CTRL | pygame.KMOD_META))
        is_shift = bool(mods & pygame.KMOD_SHIFT)

        # Paste: Ctrl/Cmd+V or Shift+Insert
        if (event.key == pygame.K_v and is_ctrl_or_cmd) or (event.key == pygame.K_INSERT and is_shift):
            pasted = self._get_clipboard_text()
            if pasted:
                self._insert_text(pasted)
            return

        if event.key == pygame.K_RETURN:
            self.active = False
            self._caret_visible = False
            self._repeat_key = None
            return

        if event.key == pygame.K_BACKSPACE:
            self._do_backspace()
            # start repeat timing
            now = pygame.time.get_ticks()
            self._repeat_key = pygame.K_BACKSPACE
            self._next_repeat_ms = now + self._repeat_delay_ms
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

        if event.unicode and event.unicode.isprintable():
            self.text = self.text[:self._caret_index] + event.unicode + self.text[self._caret_index:]
            self._caret_index += 1
            self._refresh(); self._reset_blink()
            # printable keys cancel backspace repeat
            if self._repeat_key == pygame.K_BACKSPACE:
                self._repeat_key = None
    
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
        
        self._update_key_repeat()

        # Background + border
        abs_x, abs_y, abs_w, abs_h = self.get_absolute_geometry()
        pygame.draw.rect(surface, self.background_color, (abs_x, abs_y, abs_w, abs_h))
        border = pygame.Color("dodgerblue") if self.active else pygame.Color("black")
        pygame.draw.rect(surface, border, (abs_x, abs_y, abs_w, abs_h), 2)

        # Clip text to inner rect
        clip_rect = pygame.Rect(abs_x + 2, abs_y + 2, max(0, abs_w - 4), max(0, abs_h - 4))
        prev_clip = surface.get_clip()
        surface.set_clip(clip_rect)

        # Calculate horizontal scroll if text exceeds field width
        text_width = self._measure_text_prefix_width(self.text)
        inner_width = abs_w - 2 * self._padding

        if text_width > inner_width:
            # Shift so right edge of text is visible
            offset_x = inner_width - text_width
        else:
            offset_x = 0

        # Draw the text with horizontal offset
        text_abs_x = abs_x + self._padding + offset_x
        self._text.x = text_abs_x - abs_x  # keep relative to parent
        self._text.draw(surface)

        # Caret
        if self.active:
            self._update_blink()
            if self._caret_visible:
                prefix = self.text[:self._caret_index]
                prefix_w = self._measure_text_prefix_width(prefix)
                text_x, text_y, text_w, text_h = self._text.get_absolute_geometry()
                caret_x = text_x + prefix_w
                caret_y = text_y
                caret_h = text_h
                pygame.draw.line(surface, pygame.Color("black"),
                                (caret_x, caret_y),
                                (caret_x, caret_y + caret_h), 1)

        surface.set_clip(prev_clip)

