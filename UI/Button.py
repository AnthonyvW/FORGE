import pygame
from typing import Callable, Optional, Tuple
from dataclasses import dataclass, field
from UI.text import Text, TextStyle

def default_background() -> pygame.Color:
    return pygame.Color(32, 128, 32)

def default_foreground() -> pygame.Color:
    return pygame.Color(64, 255, 64)

def default_hover_background() -> pygame.Color:
    return pygame.Color(128, 128, 255)

def default_disabled_background() -> pygame.Color:
    return pygame.Color(128, 128, 128)

def default_disabled_foreground() -> pygame.Color:
    return pygame.Color(192, 192, 192)

@dataclass
class ButtonColors:
    """Color configuration for button states"""
    background: pygame.Color = field(default_factory=default_background)
    foreground: pygame.Color = field(default_factory=default_foreground)
    hover_background: pygame.Color = field(default_factory=default_hover_background)
    disabled_background: pygame.Color = field(default_factory=default_disabled_background)
    disabled_foreground: pygame.Color = field(default_factory=default_disabled_foreground)

class Button:
    def __init__(
        self,
        function_to_call: Callable,
        x: int,
        y: int,
        width: int,
        height: int,
        text: str = "",
        colors: Optional[ButtonColors] = None,
        text_style: Optional[TextStyle] = None
    ):
        """
        Initialize a button with position, size, callback function and optional styling.
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
        self.function_to_call = function_to_call
        self.is_hover = False
        self.is_enabled = True
        
        self.colors = colors or ButtonColors()
        
        # Create text component if text is provided
        if text:
            if not text_style:
                text_style = TextStyle(
                    color=self.colors.foreground,
                    font_size=min(height - 4, 32)  # Adjust font size based on height
                )
            self.text = Text(
                text,
                x + width // 2,
                y + height // 2,
                text_style
            )
        else:
            self.text = None
    
    @property
    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)
    
    @position.setter
    def position(self, pos: Tuple[int, int]) -> None:
        old_x, old_y = self.x, self.y
        self.x, self.y = pos
        if self.text:
            self.text.update_position(
                self.x - old_x,
                self.y - old_y
            )
    
    @property
    def size(self) -> Tuple[int, int]:
        return (self.width, self.height)
    
    def update_position(self, x_offset: int, y_offset: int) -> None:
        """Update button position by the given offset"""
        self.x += x_offset
        self.y += y_offset
        if self.text:
            self.text.update_position(x_offset, y_offset)
    
    def _is_mouse_over(self, mouse_x: int, mouse_y: int) -> bool:
        """Check if mouse is over the button"""
        return (self.x <= mouse_x <= self.x + self.width and 
                self.y <= mouse_y <= self.y + self.height)
    
    def _handle_click(self, args: tuple) -> None:
        """Handle button click with proper argument passing"""
        if not self.is_enabled:
            return
            
        if args:
            self.function_to_call(*args)
        else:
            self.function_to_call()
    
    def check_button(self, mouse_x: int, mouse_y: int, mouse_click: bool, *args) -> bool:
        """
        Check button state and handle interactions.
        
        Returns:
            bool: True if mouse is over button, False otherwise
        """
        if self._is_mouse_over(mouse_x, mouse_y):
            if mouse_click and self.is_enabled:
                self._handle_click(args)
            self.is_hover = True
            return True
        self.is_hover = False
        return False
    
    def set_text(self, text: str) -> None:
        """Update button text"""
        if self.text:
            self.text.set_text(text)
        elif text:
            self.text = Text(
                text,
                self.x + self.width // 2,
                self.y + self.height // 2,
                TextStyle(color=self.colors.foreground)
            )
    
    def draw(self, surface: pygame.Surface) -> None:
        """Draw the button on the given surface"""
        if not self.is_enabled:
            bg_color = self.colors.disabled_background
            fg_color = self.colors.disabled_foreground
        elif self.is_hover:
            bg_color = self.colors.hover_background
            fg_color = self.colors.foreground
        else:
            bg_color = self.colors.background
            fg_color = self.colors.foreground
            
        # Draw button background and border
        pygame.draw.rect(surface, bg_color, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(surface, fg_color, (self.x, self.y, self.width, self.height), 2)
        
        # Update text color based on button state and draw text
        if self.text:
            if not self.is_enabled:
                self.text.style.color = self.colors.disabled_foreground
            else:
                self.text.style.color = self.colors.foreground
            self.text.draw(surface)