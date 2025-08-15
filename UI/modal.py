import pygame
from UI.frame import Frame
from UI.button import Button, ButtonColors
from UI.section_frame import Section
from UI.text import TextStyle

class _Scrim(Frame):
    """Full-screen overlay that blocks interaction with underlying UI."""
    def __init__(self, *, parent, z_index=10_000, alpha=160):
        # Cover the whole root; percent sizing + centered at (0,0) top-left.
        super().__init__(
            parent=parent, x=0, y=0, width=1.0, height=1.0,
            x_is_percent=True, y_is_percent=True,
            width_is_percent=True, height_is_percent=True,
            z_index=z_index
        )
        self._alpha = alpha
        self._capture_drag = None       # set by Modal while dragging
        self._capture_release = None    # set by Modal while dragging
        self.hide()  # start hidden

    def draw(self, surface: pygame.Surface) -> None:
        if self.is_effectively_hidden:
            return
        abs_x, abs_y, abs_w, abs_h = self.get_absolute_geometry()
        # Semi-transparent scrim
        scrim = pygame.Surface((abs_w, abs_h), pygame.SRCALPHA)
        scrim.fill((0, 0, 0, self._alpha))
        surface.blit(scrim, (abs_x, abs_y))

        # Draw children (e.g., the modal panel)
        for ch in reversed(self.children):
            ch.draw(surface)

    # Clicking the scrim closes the modal (if no child handled the click)
    def on_click(self, button=None):
        # Delegate: the Modal sets this from its constructor
        if hasattr(self, "_request_close"):
            self._request_close()

    def process_mouse_move(self, px, py):
        # If a modal is dragging, forward move regardless of where the mouse is.
        if callable(self._capture_drag):
            self._capture_drag(px, py)
        super().process_mouse_move(px, py)

    def process_mouse_release(self, px, py, button):
        # If a modal is dragging, ensure the release reaches it.
        if callable(self._capture_release):
            self._capture_release(px, py, button)
        super().process_mouse_release(px, py, button)

class _DragCapture(Frame):
    """Full-screen invisible mouse-capture layer used during modal drag in floating mode."""
    def __init__(self, *, parent, on_move, on_release, z_index):
        super().__init__(
            parent=parent, x=0, y=0, width=1.0, height=1.0,
            x_is_percent=True, y_is_percent=True,
            width_is_percent=True, height_is_percent=True,
            z_index=z_index
        )
        self._on_move = on_move
        self._on_release = on_release
        # No background; purely for event capture

    def process_mouse_move(self, px, py):
        if callable(self._on_move):
            self._on_move(px, py)
        super().process_mouse_move(px, py)

    def process_mouse_release(self, px, py, button):
        if callable(self._on_release):
            self._on_release(px, py, button)
        super().process_mouse_release(px, py, button)

class Modal(Section):
    """
    Modal with two modes:
      - overlay=True  : uses a full-screen scrim (blocks clicks behind)
      - overlay=False : no scrim; modal floats and does NOT block other buttons
    """
    def __init__(self, *, parent, title: str, width: int = 480, height: int = 320, header_height: int = 32,
                 on_close=None, z_index: int = 10_001,
                 header_bg: pygame.Color = pygame.Color("#dbdbdb"),
                 background_color: pygame.Color = pygame.Color("#ffffff"),
                 title_style: TextStyle | None = None,
                 overlay: bool = True,
                 scrim_alpha: int = 160):
        super().__init__(
            parent=(parent if not overlay else _Scrim(parent=parent, z_index=z_index - 1, alpha=scrim_alpha)),
            title=title,
            width=width, height=height,
            header_height=header_height,
            header_bg=header_bg,
            background_color=background_color,
            title_style=title_style,
            z_index=z_index,
            x=0.5, y=0.5,
            x_is_percent=True, y_is_percent=True
        )

        self.x_align = "center"
        self.y_align = "center"

        self._on_close = on_close
        self._overlay = overlay        
        self._dragging = False
        self._drag_offset = (0, 0)
        self._drag_capture_layer = None  # only used when overlay == False

        if overlay:
            self._scrim = self.parent
            self._scrim.hide(True)
            self._scrim._request_close = self.close
        else:
            self._scrim = None
            self.hide(True)

        close_btn_style = TextStyle(
            color=pygame.Color("#4a4a4a"),
            font_size=min(20, header_height - 8),
        )
        self._close_btn = Button(
            self.close,
            x=8, y=(header_height - 24) // 2,
            width=24, height=24,
            text="X",
            text_style=close_btn_style,
            x_align="right",
            parent=self.header,
            colors=ButtonColors(
                background=header_bg,
                foreground=header_bg,
                hover_background=pygame.Color("#b3b4b6"),
                disabled_background=header_bg,
                disabled_foreground=header_bg
            ),
        )

    # Public API
    def open(self):
        if self._scrim is not None:
            self._scrim.show(True)   # shows panel via scrim
        else:
            self.show(True)          # floating mode: just show panel

    def close(self):
        if self._scrim is not None:
            self._scrim.hide(True)
        else:
            self.hide(True)
        if callable(self._on_close):
            self._on_close()

    # ESC to close (works in both modes)
    def on_key_event(self, event):
        try:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close()
        except Exception:
            pass

    # --- Dragging support ---

    def _begin_drag(self, px, py):
        # Anchor the pointer offset from the modal's top-left corner
        abs_x, abs_y, _, _ = self.get_absolute_geometry()
        self._dragging = True
        self._drag_offset = (px - abs_x, py - abs_y)

        # While dragging, switch to absolute pixel positioning
        self.x_is_percent = False
        self.y_is_percent = False
        self.x_align = "left"
        self.y_align = "top"

        if self._scrim is not None:
            # Overlay mode: use scrim as move/release forwarder
            self._scrim._capture_drag = self._on_drag_move
            self._scrim._capture_release = self._on_drag_release
        else:
            # Floating mode: create a temporary full-screen capture layer
            self._drag_capture_layer = _DragCapture(
                parent=self.parent,
                on_move=self._on_drag_move,
                on_release=self._on_drag_release,
                z_index=self.z_index + 10_000
            )

    def _end_drag(self):
        self._dragging = False
        if self._scrim is not None:
            self._scrim._capture_drag = None
            self._scrim._capture_release = None
        if self._drag_capture_layer is not None:
            # Remove the capture layer
            try:
                self.parent.children.remove(self._drag_capture_layer)
            except ValueError:
                pass
            self._drag_capture_layer = None

    def _on_drag_move(self, px, py):
        if not self._dragging:
            return
        parent_x, parent_y, parent_w, parent_h = self.parent.get_absolute_geometry()

        new_x = px - self._drag_offset[0] - parent_x
        new_y = py - self._drag_offset[1] - parent_y

        # Optional: keep modal inside parent bounds (comment out if you don’t want clamping)
        w, h = self.size
        new_x = max(0, min(new_x, max(0, parent_w - w)))
        new_y = max(0, min(new_y, max(0, parent_h - h)))

        self.x = new_x
        self.y = new_y

    def _on_drag_release(self, px, py, button):
        if button == "left":
            self._end_drag()

    # Hook into your existing event methods

    def process_mouse_press(self, px, py, button):
        super().process_mouse_press(px, py, button)

        if button != "left":
            return

        # Start dragging if press was on the header (but not on the close button)
        if self.header.contains_point(px, py) and not self._close_btn.contains_point(px, py):
            self._begin_drag(px, py)

    def process_mouse_move(self, px, py):
        # If dragging, update first so the UI feels snappy
        if self._dragging:
            self._on_drag_move(px, py)
        super().process_mouse_move(px, py)

    def process_mouse_release(self, px, py, button):
        # Ensure we end dragging even if the release happens off the modal
        if self._dragging:
            self._on_drag_release(px, py, button)
        super().process_mouse_release(px, py, button)