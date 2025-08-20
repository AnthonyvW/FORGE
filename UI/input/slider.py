# slider.py
import pygame
from typing import Callable, Optional
from UI.frame import Frame
from UI.text import TextStyle
from UI.input.button import Button, ButtonColors

class Slider(Frame):
    def __init__(
        self,
        min_value: float,
        max_value: float,
        x: int,
        y: int,
        width: int,
        height: int,
        initial_value: Optional[float] = None,
        on_change: Optional[Callable[[float], None]] = None,
        tick_count: int = 2,  # 0 or >=2; 1 coerced to 2
        track_color: Optional[pygame.Color] = None,
        tick_color: Optional[pygame.Color] = None,
        knob_fill: Optional[pygame.Color] = None,
        knob_border_color: Optional[pygame.Color] = None,
        knob_hover_fill: Optional[pygame.Color] = None,
        knob_hover_border_color: Optional[pygame.Color] = None,
        with_buttons: bool = False,
        step: float = 1.0,
        **frame_kwargs
    ):
        super().__init__(x=x, y=y, width=width, height=height, **frame_kwargs)

        # --- values / callback ---
        self.min_value = min_value
        self.max_value = max_value
        self.value = min_value if initial_value is None else max(min_value, min(max_value, initial_value))
        self.on_change = on_change

        # --- ticks ---
        if tick_count < 0:
            tick_count = 0
        if tick_count == 1:
            tick_count = 2
        self.tick_count = tick_count

        # --- visuals ---
        self.track_height = 4
        self.knob_width = 10
        self.knob_border = 2

        self.track_color = track_color or pygame.Color("#b3b4b6")
        self.tick_color = tick_color or self.track_color
        self.knob_fill = knob_fill or pygame.Color("#b3b4b6")
        self.knob_border_color = pygame.Color(knob_border_color) if knob_border_color else pygame.Color("#5a5a5a")

        # hover styles
        self.knob_hover_fill = knob_hover_fill or pygame.Color("#5a5a5a")
        self.knob_hover_border_color = pygame.Color(knob_hover_border_color) if knob_hover_border_color else pygame.Color("#5a5a5a")

        # --- interaction ---
        self.dragging = False
        self.knob_hover = False

        # --- optional +/- buttons (fixed layout, no resize adjustments) ---
        self.with_buttons = with_buttons
        self.step = step
        self.btn_w = 0
        self.btn_margin = 4
        self.left_button: Optional[Button] = None
        self.right_button: Optional[Button] = None

        if self.with_buttons:
            self.btn_w = min(height, 16)

            # White background & border, keep text black
            btn_colors = ButtonColors(
                background=pygame.Color("#ffffff"),  
                hover_background=pygame.Color("#ffffff"),  
                disabled_background=pygame.Color("#ffffff"),

                foreground=pygame.Color("#FFFFFF"),          
                hover_foreground=pygame.Color("#ffffff"),    
                disabled_foreground=pygame.Color("#ffffff"),

                text=pygame.Color("#b3b4b6"),
                hover_text=pygame.Color("#5a5a5a"),
                disabled_text=pygame.Color("#ffffff")
            )

            self.left_button = Button(
                self._decrement, 0, 0, self.btn_w, height,
                text="-", colors=btn_colors, parent=self
            )
            self.right_button = Button(
                self._increment, width - self.btn_w, 0, self.btn_w, height,
                text="+", colors=btn_colors, parent=self
            )

    # ===== Public helpers (so external buttons can also drive the slider) =====
    def increment(self, amount: Optional[float] = None):
        self._bump(+ (amount if amount is not None else self.step))

    def decrement(self, amount: Optional[float] = None):
        self._bump(- (amount if amount is not None else self.step))

    # ===== Internal helpers =====
    def _bump(self, delta: float):
        old = self.value
        self.value = max(self.min_value, min(self.max_value, self.value + delta))
        if self.on_change and self.value != old:
            self.on_change(self.value)

    def _increment(self):
        self._bump(self.step)

    def _decrement(self):
        self._bump(-self.step)

    def _track_rect(self):
        """Return (track_x, track_y, track_w, abs_h) for the inner track.
           If buttons are enabled, we reserve left/right gutters."""
        abs_x, abs_y, abs_w, abs_h = self.get_absolute_geometry()
        if self.with_buttons:
            track_x = abs_x + self.btn_w + self.btn_margin
            track_w = max(1, abs_w - 2 * (self.btn_w + self.btn_margin))
        else:
            track_x = abs_x
            track_w = max(1, abs_w)
        track_y = abs_y + abs_h // 2 - self.track_height // 2
        return track_x, track_y, track_w, abs_h

    # --- value <-> position mapped to inner track ---
    def _value_to_pos(self) -> int:
        track_x, _, track_w, _ = self._track_rect()
        if self.max_value == self.min_value:
            return track_x
        ratio = (self.value - self.min_value) / (self.max_value - self.min_value)
        return track_x + int(ratio * track_w)

    def _pos_to_value(self, px: int) -> float:
        track_x, _, track_w, _ = self._track_rect()
        if track_w <= 0 or self.max_value == self.min_value:
            return self.min_value
        ratio = (px - track_x) / track_w
        ratio = max(0.0, min(1.0, ratio))
        return self.min_value + ratio * (self.max_value - self.min_value)

    # ===== Events =====
    def on_mouse_press(self, button):
        if button == "left":
            self.dragging = True

    def on_mouse_release(self, button):
        if button == "left":
            self.dragging = False

    def process_mouse_move(self, px, py):
        # Keep your existing hover bookkeeping
        super().process_mouse_move(px, py)

        # Purely visual hover for the knob
        if self.is_hovered:
            track_x, _, _, abs_h = self._track_rect()
            knob_center = self._value_to_pos()
            knob_x = knob_center - self.knob_width // 2
            abs_x, abs_y, _, _ = self.get_absolute_geometry()
            self.knob_hover = (knob_x <= px <= knob_x + self.knob_width) and (abs_y <= py <= abs_y + abs_h)
        else:
            self.knob_hover = False

    def on_hover_leave(self):
        # Only clear highlight; do not cancel a live drag
        self.knob_hover = False
        if not pygame.mouse.get_pressed()[0]:
            self.dragging = False

    # ===== Drawing =====
    def draw(self, surface: pygame.Surface) -> None:
        if self.is_effectively_hidden:
            return

        # Keep dragging even when mouse leaves the slider, stop on global button up
        if self.dragging:
            left_down = pygame.mouse.get_pressed()[0]
            if not left_down:
                self.dragging = False
            else:
                mx, _ = pygame.mouse.get_pos()
                old = self.value
                self.value = self._pos_to_value(mx)
                if self.on_change and self.value != old:
                    self.on_change(self.value)

        # Draw track
        track_x, track_y, track_w, abs_h = self._track_rect()
        pygame.draw.rect(surface, self.track_color, (track_x, track_y, track_w, self.track_height))

        # Ticks
        if self.tick_count >= 2:
            base_tick_h = max(6, min(12, abs_h // 2))
            end_tick_h = int(base_tick_h * 1.5)
            for i in range(self.tick_count):
                t = i / (self.tick_count - 1)
                tx = track_x + int(t * track_w)
                tick_h = end_tick_h if i in (0, self.tick_count - 1) else base_tick_h
                top = track_y - tick_h // 2
                bottom = top + tick_h
                pygame.draw.line(
                    surface,
                    self.tick_color,
                    (tx, top),
                    (tx, bottom + self.track_height),
                    int(self.track_height * 0.75)
                )

        # Knob (rectangle) with hover style
        knob_x_center = self._value_to_pos()
        knob_x = knob_x_center - self.knob_width // 2
        abs_x, abs_y, _, _ = self.get_absolute_geometry()
        knob_y = abs_y

        if self.knob_hover:
            fill_color = self.knob_hover_fill
            border_color = self.knob_hover_border_color
        else:
            fill_color = self.knob_fill
            border_color = self.knob_border_color

        # Draw children (the +/- buttons)
        for child in self.children:
            child.draw(surface)

        pygame.draw.rect(surface, fill_color, (knob_x, knob_y, self.knob_width, abs_h))
        pygame.draw.rect(surface, border_color, (knob_x, knob_y, self.knob_width, abs_h), self.knob_border)
