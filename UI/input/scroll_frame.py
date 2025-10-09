# UI/input/scroll_frame.py
import pygame
from UI.frame import Frame

class ScrollbarV(Frame):
    """Vertical scrollbar child for a ScrollFrame (right-side strip)."""
    def __init__(
        self,
        *,
        parent: "ScrollFrame",
        width: int = 12,
        track_color: pygame.Color = pygame.Color("#d7d7d7"),
        thumb_color: pygame.Color = pygame.Color("#9a9a9a"),
        thumb_min_px: int = 24,
        z_index: int = 1000,
    ):
        super().__init__(
            parent=parent,
            x=0, y=0,
            width=width, height=1.0,
            width_is_percent=False, height_is_percent=True,
            x_align="right", y_align="top",
            z_index=z_index,
            background_color=None,
        )
        self._dragging = False
        self._drag_offset = 0
        self.track_color = track_color
        self.thumb_color = thumb_color
        self.thumb_min_px = thumb_min_px

    # --- helpers that query parent metrics ---
    @property
    def _sf(self) -> "ScrollFrame":
        return self.parent  # type: ignore[return-value]

    def _viewport_h(self) -> int:
        return self._sf._viewport_height()

    def _content_h(self) -> int:
        return self._sf._content_height()

    def _track_rect(self) -> pygame.Rect:
        x, y, w, h = self.get_absolute_geometry()
        return pygame.Rect(x, y, w, h)

    def _thumb_h(self) -> int:
        vh = self._viewport_h()
        ch = self._content_h()
        if ch <= vh:
            return vh
        return max(int(vh * (vh / ch)), self.thumb_min_px)

    def _thumb_rect(self) -> pygame.Rect:
        track = self._track_rect()
        vh = self._viewport_h()
        ch = self._content_h()

        if ch <= vh:
            # No overflow: fill track (thumb == track)
            return pygame.Rect(track.x, track.y, track.w, track.h)

        thumb_h = self._thumb_h()
        track_h = track.h - thumb_h
        ratio = self._sf.scroll_y / (ch - vh) if ch > vh else 0.0
        thumb_y = track.y + int(ratio * track_h)
        return pygame.Rect(track.x, thumb_y, track.w, thumb_h)

    def _set_scroll_from_thumb_y(self, thumb_y: int):
        track = self._track_rect()
        vh = self._viewport_h()
        ch = self._content_h()
        if ch <= vh:
            self._sf._set_scroll(0)
            return
        thumb_h = self._thumb_h()
        track_h = track.h - thumb_h
        ratio = (thumb_y - track.y) / track_h if track_h > 0 else 0.0
        self._sf._set_scroll(ratio * (ch - vh))

    # --- input handling ---
    def process_mouse_press(self, px, py, button):
        if self.is_effectively_hidden:
            return
        if button == "left":
            thumb = self._thumb_rect()
            track = self._track_rect()
            if thumb.collidepoint(px, py):
                self._dragging = True
                self._drag_offset = py - thumb.y
                return
            if track.collidepoint(px, py):
                # Jump to click position and begin drag
                new_y = py - self._thumb_h() // 2
                new_y = max(track.y, min(new_y, track.bottom - self._thumb_h()))
                self._set_scroll_from_thumb_y(new_y)
                self._dragging = True
                self._drag_offset = py - self._thumb_rect().y
                return
        # Not on the scrollbar; no need to route further because ScrollFrame will.

    def process_mouse_move(self, px, py):
        if self._dragging:
            track = self._track_rect()
            th = self._thumb_h()
            new_y = max(track.y, min(py - self._drag_offset, track.bottom - th))
            self._set_scroll_from_thumb_y(new_y)

    def process_mouse_release(self, px, py, button):
        self._dragging = False

    # --- drawing ---
    def draw(self, surface: pygame.Surface) -> None:
        if self.is_effectively_hidden:
            return

        # track
        track = self._track_rect()
        pygame.draw.rect(surface, self.track_color, track)

        # thumb
        thumb = self._thumb_rect()
        pygame.draw.rect(surface, self.thumb_color, thumb)
        pygame.draw.rect(surface, pygame.Color(0, 0, 0), thumb, 1)


class ScrollFrame(Frame):
    """Vertical-only scrollable container with a dedicated right-side ScrollbarV child."""
    def __init__(
        self,
        *,
        parent,
        x=0, y=0, width=100, height=100,
        scroll_speed=30,
        scrollbar_width=12,
        background_color=pygame.Color("#f5f5f5"),
        track_color=pygame.Color("#d7d7d7"),
        thumb_color=pygame.Color("#9a9a9a"),
        thumb_min_px=24,
        z_index=0,
    ):
        # Prevent overridden add_child from running before content exists
        self._initializing = True
        super().__init__(
            parent=parent, x=x, y=y, width=width, height=height,
            background_color=background_color, z_index=z_index
        )

        self.scroll_y = 0
        self.scroll_speed = scroll_speed
        self.scrollbar_width = scrollbar_width

        # Content container (everything the user adds goes here)
        self.content = Frame(
            parent=None,
            x=0, y=0,
            width=1.0, height=1.0,
            width_is_percent=True, height_is_percent=True,
            z_index=z_index  # below scrollbar
        )
        # Attach directly then sort
        self.content.parent = self
        self.children.append(self.content)

        # Scrollbar child (high z so it wins hit-testing)
        self.scrollbar = ScrollbarV(
            parent=self,
            width=scrollbar_width,
            track_color=track_color,
            thumb_color=thumb_color,
            thumb_min_px=thumb_min_px,
            z_index=z_index + 999  # ensure on top
        )

        self._initializing = False

    # Route user children into the content frame
    def add_child(self, child):
        if self._initializing:
            return super().add_child(child)
        return self.content.add_child(child)

    # --- geometry + layout ---
    def _viewport_rect(self) -> pygame.Rect:
        x, y, w, h = self.get_absolute_geometry()
        return pygame.Rect(x, y, max(0, w - self.scrollbar_width), h)

    def _viewport_height(self) -> int:
        return self.get_absolute_geometry()[3]

    def _layout(self):
        """Ensure content matches viewport width and logical scroll position."""
        abs_x, abs_y, abs_w, abs_h = self.get_absolute_geometry()
        viewport_w = max(0, abs_w - self.scrollbar_width)

        # Content occupies only viewport area (not under the scrollbar)
        self.content.width_is_percent = False
        self.content.height_is_percent = False
        self.content.x = 0
        self.content.width = viewport_w
        self.content.height = abs_h
        self.content.y = -self.scroll_y  # logical scroll

    def _content_height(self) -> int:
        """Total vertical extent of content children relative to content's top."""
        if not self.content.children:
            return self._viewport_height()

        _, content_abs_top, _, _ = self.content.get_absolute_geometry()
        max_bottom_rel = 0
        for ch in self.content.children:
            _, ch_abs_y, _, ch_h = ch.get_absolute_geometry()
            bottom_rel = (ch_abs_y + ch_h) - content_abs_top
            if bottom_rel > max_bottom_rel:
                max_bottom_rel = bottom_rel

        return max(max_bottom_rel, self._viewport_height())

    # --- scroll core ---
    def _set_scroll(self, value: int | float):
        max_scroll = max(0, self._content_height() - self._viewport_height())
        self.scroll_y = max(0, min(int(value), max_scroll))
        self.content.y = -self.scroll_y

    def on_wheel(self, delta_y: int):
        """(Hook this up later from your event loop.)"""
        self._set_scroll(self.scroll_y - delta_y * self.scroll_speed)

    # --- input routing: run layout first so geometry is up-to-date ---
    def process_mouse_press(self, px, py, button):
        self._layout()
        # Children (including scrollbar) handle their own input via z-index ordering
        super().process_mouse_press(px, py, button)

    def process_mouse_move(self, px, py):
        self._layout()
        # If the scrollbar is dragging, capture the move
        if self.scrollbar._dragging:
            self.scrollbar.process_mouse_move(px, py)
            return
        super().process_mouse_move(px, py)

    def process_mouse_release(self, px, py, button):
        self._layout()
        # If the scrollbar started a drag, ensure it gets the release
        if self.scrollbar._dragging:
            self.scrollbar.process_mouse_release(px, py, button)
            return
        super().process_mouse_release(px, py, button)

    # --- drawing ---
    def draw(self, surface: pygame.Surface) -> None:
        if self.is_effectively_hidden:
            return

        self._layout()
        abs_x, abs_y, abs_w, abs_h = self.get_absolute_geometry()

        # Background
        if self.background_color:
            pygame.draw.rect(surface, self.background_color, (abs_x, abs_y, abs_w, abs_h))

        # Clip and draw content
        old_clip = surface.get_clip()
        surface.set_clip(self._viewport_rect())
        self.content.draw(surface)
        surface.set_clip(old_clip)

        # Draw scrollbar (child)
        self.scrollbar.draw(surface)
