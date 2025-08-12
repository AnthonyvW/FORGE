import pygame
from typing import Callable, Optional, Tuple

def default_frame_background() -> Optional[pygame.Color]:
    return None

class Frame():
    def __init__(
        self, parent=None, x=0, y=0, width=100, height=100, 
        x_is_percent=False, y_is_percent=False,
        width_is_percent=False, height_is_percent=False,
        z_index=0, x_align: str = 'left', y_align: str = 'top', 
        background_color: Optional[pygame.Color] = None
    ):
        self.parent = parent
        self.children = []

        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.background_color = background_color

        self.x_is_percent = x_is_percent
        self.y_is_percent = y_is_percent
        self.width_is_percent = width_is_percent
        self.height_is_percent = height_is_percent

        self.z_index = z_index
        self.x_align = x_align
        self.y_align = y_align

        self.is_hovered = False
        self.is_pressed = False

        # Automatically add parent if its passed as an argument
        if parent is not None:
            parent.add_child(self)

    @property
    def debug_outline_color(self) -> pygame.Color:
        return pygame.Color(255, 0, 0)  # Default: red
    
    @property
    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)

    def update_position(self, x_offset: int, y_offset: int) -> None:
        """Move this frame and all children by the given pixel offset, regardless of percent/absolute mode."""
        self._apply_offset(x_offset, y_offset)

        for child in self.children:
            # Get parent's size in pixels for percent-based calculations
            _, _, parent_width, parent_height = self.get_absolute_geometry()
            percent_dx = x_offset / parent_width if parent_width else 0
            percent_dy = y_offset / parent_height if parent_height else 0
            child._apply_offset(x_offset, y_offset, percent_dx, percent_dy)

    def _apply_offset(self, dx: float, dy: float, dx_percent: float = 0.0, dy_percent: float = 0.0) -> None:
        """Shift this frame by the given amounts, using percent or absolute logic as needed."""
        if self.x_is_percent:
            self.x += dx_percent
        else:
            self.x += dx

        if self.y_is_percent:
            self.y += dy_percent
        else:
            self.y += dy

    def update_size(self, width_offset: int, height_offset: int) -> None:
        """Resize this frame and all children by pixel delta, adjusting percent-based children proportionally."""
        self._apply_size_change(width_offset, height_offset)

        for child in self.children:
            _, _, parent_width, parent_height = self.get_absolute_geometry()
            percent_dw = width_offset / parent_width if parent_width else 0
            percent_dh = height_offset / parent_height if parent_height else 0
            child._apply_size_change(width_offset, height_offset, percent_dw, percent_dh)

    def _apply_size_change(self, dw: float, dh: float, dw_percent: float = 0.0, dh_percent: float = 0.0) -> None:
        """Apply size delta to this frame, supporting both pixel and percent-based width/height."""
        if self.width_is_percent:
            self.width += dw_percent
        # else: don't modify absolute width

        if self.height_is_percent:
            self.height += dh_percent
        # else: don't modify absolute height


    @property
    def absolute_position(self) -> Tuple[int, int]:
        abs_width = self._width * parent_width if self.width_is_percent else self._width
        abs_height = self._height * parent_height if self.height_is_percent else self._height
        return (abs_width, abs_height)

    @property
    def size(self) -> Tuple[int, int]:
        return (self.width, self.height)
    
    def get_absolute_geometry(self):
        """Returns absolute screen coordinates"""
        if self.parent:
            parent_x, parent_y, parent_width, parent_height = self.parent.get_absolute_geometry()
        else:
            parent_x, parent_y = 0, 0
            parent_width, parent_height = pygame.display.get_surface().get_size()

        # Get absolute position given alignment
        raw_x = self.x * parent_width if self.x_is_percent else self.x
        raw_y = self.y * parent_height if self.y_is_percent else self.y

        abs_width = self.width * parent_width if self.width_is_percent else self.width
        abs_height = self.height * parent_height if self.height_is_percent else self.height

        # Apply horizontal alignment
        if self.x_align == 'left':
            abs_x = parent_x + raw_x
        elif self.x_align == 'center':
            abs_x = parent_x + (parent_width // 2) + raw_x - (abs_width // 2)
        elif self.x_align == 'right':
            abs_x = parent_x + parent_width - raw_x - abs_width
        else:
            raise ValueError(f"Invalid x_align: {self.x_align}")

        # Apply vertical alignment
        if self.y_align == 'top':
            abs_y = parent_y + raw_y
        elif self.y_align == 'center':
            abs_y = parent_y + (parent_height // 2) + raw_y - (abs_height // 2)
        elif self.y_align == 'bottom':
            abs_y = parent_y + parent_height - raw_y - abs_height
        else:
            raise ValueError(f"Invalid y_align: {self.y_align}")

        abs_width = self.width * parent_width if self.width_is_percent else self.width
        abs_height = self.height * parent_height if self.height_is_percent else self.height

        return abs_x, abs_y, abs_width, abs_height

    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        self.children.sort(key=lambda c: c.z_index, reverse=True)  # front-to-back order
    
    def contains_point(self, px, py):
        abs_x, abs_y, abs_width, abs_height = self.get_absolute_geometry()
        return abs_x <= px <= abs_x + abs_width and abs_y <= py <= abs_y + abs_height

    def handle_click(self, px, py):
        # First check children, then self
        for child in (self.children):
            if child.contains_point(px, py):
                child.handle_click(px, py)
                return  # Only send to first child that contains it (or remove this if you want overlapping elements to handle too)

        self.on_click()
    
    def handle_hover(self, px, py):
        for child in (self.children):
            if child.contains_point(px, py):
                child.handle_hover(px, py)
                return
        if self.contains_point(px, py):
            self.on_hover()

    def _clear_hover_recursive(self):
        if self.is_hovered:
            self.is_hovered = False
            self.on_hover_leave()
        for ch in self.children:
            ch._clear_hover_recursive()

    def process_mouse_move(self, px, py):
        """Hover handling with z occlusion"""
        
        # First propagate to children front-to-back
        top_hit = None
        for child in (self.children):
            if child.contains_point(px, py):
                top_hit = child
                break
        
        for child in self.children:
            if child is top_hit:
                child.process_mouse_move(px, py)
            else:
                child._clear_hover_recursive()

        # Now check self hover state
        inside = self.contains_point(px, py)
        if inside and not self.is_hovered:
            self.is_hovered = True
            self.on_hover_enter()
        elif not inside and self.is_hovered:
            self.is_hovered = False
            self.on_hover_leave()

    def process_mouse_press(self, px, py, button):
        for child in self.children:
            if child.contains_point(px, py):
                child.process_mouse_press(px, py, button)
                return

        if self.contains_point(px, py):
            self.is_pressed = True
            self.on_mouse_press(button)

    def process_mouse_release(self, px, py, button):
        for child in self.children:
            if child.contains_point(px, py):
                child.process_mouse_release(px, py, button)
                return

        if self.is_pressed:
            self.is_pressed = False
            self.on_mouse_release(button)
            if self.contains_point(px, py):
                self.on_click(button)
    
    def draw(self, surface: pygame.Surface) -> None:
        abs_x, abs_y, abs_w, abs_h = self.get_absolute_geometry()

        if self.background_color:
            pygame.draw.rect(surface, self.background_color, (abs_x, abs_y, abs_w, abs_h))

        for child in reversed(self.children):
            child.draw(surface)

    # --- Override these ---
    def on_click(self, button=None):
        pass

    def on_mouse_press(self, button):
        pass

    def on_mouse_release(self, button):
        pass

    def on_hover(self):
        pass

    def on_hover_enter(self):
        pass

    def on_hover_leave(self):
        pass

    def broadcast_mouse_press(self, px, py, button):
        """Give every widget a chance to react to a global mouse press (e.g., focus/unfocus)."""
        for child in self.children:
            child.broadcast_mouse_press(px, py, button)
        self.on_global_mouse_press(px, py, button)

    def on_global_mouse_press(self, px, py, button):
        """Override in widgets that need to react even if the click was outside them."""
        pass

    def broadcast_key_event(self, event):
        """Bubble key events to all widgets; inactive widgets can ignore them."""

        print(f"BROADCAST -> {self.__class__.__name__} id={id(self)} children={len(self.children)}")

        for child in self.children:
            child.broadcast_key_event(event)

        self.on_key_event(event)

    def on_key_event(self, event):
        """Override in widgets that want keyboard input."""
        print(f"ON_KEY     -> {self.__class__.__name__} id={id(self)}")
        pass
