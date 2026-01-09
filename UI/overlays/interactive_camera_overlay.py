"""
Interactive camera overlay UI component.

This overlay provides a crosshair for visual reference and handles user
interactions (click-to-move, wheel-to-zoom) by delegating to the controller's
calibration and movement system.
"""

import pygame
from typing import Optional

from UI.frame import Frame
from UI.camera_view import CameraView
from UI.styles import CROSSHAIR_COLOR

class InteractiveCameraOverlay(Frame):
    """
    UI overlay that renders a crosshair in the center of the camera view.
    Supports click-to-move and mousewheel Z-axis control.
    Delegates calibration and movement logic to the controller.
    """
    def __init__(
        self,
        camera_view: CameraView,
        controller,  # AutomatedPrinter instance with CameraCalibrationMixin
        visible: bool = True,
        
        # Crosshair visual properties
        crosshair_color: Optional[pygame.Color] = None,
        crosshair_length: int = 20,
        crosshair_thickness: int = 2,
        crosshair_gap: int = 5,
    ):
        super().__init__(
            parent=camera_view,
            x=0, y=0,
            width=1, height=1,
            x_is_percent=True, y_is_percent=True,
            width_is_percent=True, height_is_percent=True,
            z_index=camera_view.z_index + 1,
            background_color=None
        )

        self.camera_view = camera_view
        self.controller = controller
        self.visible = visible
        
        # Enable click handling
        self.mouse_passthrough = False

        # Crosshair properties
        self.crosshair_color = crosshair_color or CROSSHAIR_COLOR
        self.crosshair_length = crosshair_length
        self.crosshair_thickness = crosshair_thickness
        self.crosshair_gap = crosshair_gap

        # Cache overlay surface
        self._overlay = None
        self._overlay_size = None

    # ==================== Visibility Control ====================
    
    def toggle_overlay(self) -> None:
        """Toggle visibility of the crosshair overlay."""
        self.visible = not self.visible

    def set_visible(self, value: bool) -> None:
        """Set visibility of the crosshair overlay."""
        self.visible = bool(value)

    def set_crosshair_color(self, color: pygame.Color) -> None:
        """Update the crosshair color."""
        self.crosshair_color = color

    def set_crosshair_properties(
        self,
        length: Optional[int] = None,
        thickness: Optional[int] = None,
        gap: Optional[int] = None
    ) -> None:
        """Update crosshair geometry properties."""
        if length is not None:
            self.crosshair_length = length
        if thickness is not None:
            self.crosshair_thickness = thickness
        if gap is not None:
            self.crosshair_gap = gap

    # ==================== Calibration Helpers ====================
    
    def run_calibration(self) -> None:
        """
        Trigger camera calibration through the controller.
        This will move the printer and determine the pixel-to-world mapping.
        """
        if not hasattr(self.controller, 'start_camera_calibration'):
            self.controller.status(
                "Controller does not support camera calibration",
                True
            )
            return
        
        self.controller.start_camera_calibration()
    
    def is_calibrated(self) -> bool:
        """Check if camera calibration is available."""
        if not hasattr(self.controller, 'M_inv'):
            return False
        return self.controller.M_inv is not None

    # ==================== Mouse Event Handlers ====================
    
    def on_click(self, button=None) -> None:
        """Handle click events to move printer to clicked position."""
        if not self.camera_view.camera.initialized:
            return
        
        if not self.is_calibrated():
            self.controller.status(
                "Cannot move: run calibration first (call run_calibration())",
                True
            )
            return
        
        # Get mouse position
        mouse_x, mouse_y = pygame.mouse.get_pos()
        
        # Get the camera frame rectangle
        fr = self.camera_view.get_frame_rect()
        if not fr:
            return
        
        fx, fy, fw, fh = fr
        
        # Check if click is within camera frame
        if not (fx <= mouse_x <= fx + fw and fy <= mouse_y <= fy + fh):
            return
        
        # Convert display coordinates to image coordinates
        # The camera_view may scale/letterbox the image, so we need to account for that
        rel_x = mouse_x - fx
        rel_y = mouse_y - fy
        
        # Get calibration image dimensions from controller
        cal_status = self.controller.get_calibration_status()
        img_w = cal_status.get('image_width')
        img_h = cal_status.get('image_height')
        
        if img_w is None or img_h is None:
            self.controller.status("Calibration image dimensions not available", True)
            return
        
        # Calculate scaling factor (camera_view letterboxes to fit)
        # The displayed image maintains aspect ratio within the frame
        img_aspect = img_w / img_h if img_h > 0 else 1.0
        frame_aspect = fw / fh if fh > 0 else 1.0
        
        if img_aspect > frame_aspect:
            # Image is wider - letterbox top/bottom
            display_w = fw
            display_h = fw / img_aspect
            offset_x = 0
            offset_y = (fh - display_h) / 2
        else:
            # Image is taller - letterbox left/right
            display_w = fh * img_aspect
            display_h = fh
            offset_x = (fw - display_w) / 2
            offset_y = 0
        
        # Adjust for letterboxing
        adj_x = rel_x - offset_x
        adj_y = rel_y - offset_y
        
        # Check if click is in the actual image area
        if not (0 <= adj_x <= display_w and 0 <= adj_y <= display_h):
            self.controller.status("Click outside image area", False)
            return
        
        # Scale to image coordinates
        pixel_x = (adj_x / display_w) * img_w
        pixel_y = (adj_y / display_h) * img_h
        
        # Use controller's vision movement (relative to current position)
        self.controller.move_to_vision_point(pixel_x, pixel_y, relative=True)

    def on_wheel(self, dx: int, dy: int, px: int, py: int) -> bool:
        """
        Handle mousewheel events to adjust Z-axis position when hovering over camera view.
        
        Args:
            dx: Horizontal wheel movement (unused)
            dy: Vertical wheel movement (positive = wheel up = Z up)
            px: Mouse X position
            py: Mouse Y position
            
        Returns:
            True if the event was handled, False otherwise
        """
        if not self.camera_view.camera.initialized:
            return False
        
        # Check if mouse is over the camera frame
        fr = self.camera_view.get_frame_rect()
        if not fr:
            return False
        
        fx, fy, fw, fh = fr
        if not (fx <= px <= fx + fw and fy <= py <= fy + fh):
            return False
        
        # Get current position
        current_pos = self.controller.get_position()
        
        # Use minimum step size: 4 ticks = 0.04mm (printer's minimum step)
        MIN_STEP_TICKS = 4
        
        # Calculate Z change (positive dy = wheel up = move Z up)
        dz_ticks = dy * MIN_STEP_TICKS
        
        # Calculate new Z position
        new_z_ticks = current_pos.z + int(round(dz_ticks))
        new_z_mm = new_z_ticks / 100.0
        
        # Bounds check
        max_z = self.controller.get_max_z()
        if not (0 <= new_z_mm <= max_z):
            self.controller.status(
                f"Z position out of bounds: {new_z_mm:.2f}mm (max: {max_z}mm)",
                False
            )
            return True
        
        # Send move command
        self.controller.enqueue_printer(
            f"G0 Z{new_z_mm:.2f}",
            message=f"Z: {new_z_mm:.2f}mm",
            log=False
        )
        
        return True

    # ==================== Drawing ====================

    def _get_overlay(self, surface_size: tuple[int, int]) -> pygame.Surface:
        """Return an RGBA overlay the size of the target surface (recreate on resize)."""
        if self._overlay is None or self._overlay_size != surface_size:
            self._overlay_size = surface_size
            self._overlay = pygame.Surface(surface_size, flags=pygame.SRCALPHA)
        else:
            self._overlay.fill((0, 0, 0, 0))
        return self._overlay

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the crosshair overlay if visible and camera is initialized."""
        if not self.visible:
            return

        if not self.camera_view.camera.initialized:
            return

        fr = self.camera_view.get_frame_rect()
        if not fr:
            return

        fx, fy, fw, fh = fr

        # Build/resize overlay and clear it
        overlay = self._get_overlay(surface.get_size())
        overlay.fill((0, 0, 0, 0))

        # Calculate center point of the camera frame
        center_x = fx + fw // 2
        center_y = fy + fh // 2

        # Draw crosshair lines with gap in the middle
        
        # Horizontal line (left and right segments)
        pygame.draw.line(
            overlay,
            self.crosshair_color,
            (center_x - self.crosshair_gap - self.crosshair_length, center_y),
            (center_x - self.crosshair_gap, center_y),
            self.crosshair_thickness
        )
        pygame.draw.line(
            overlay,
            self.crosshair_color,
            (center_x + self.crosshair_gap, center_y),
            (center_x + self.crosshair_gap + self.crosshair_length, center_y),
            self.crosshair_thickness
        )

        # Vertical line (top and bottom segments)
        pygame.draw.line(
            overlay,
            self.crosshair_color,
            (center_x, center_y - self.crosshair_gap - self.crosshair_length),
            (center_x, center_y - self.crosshair_gap),
            self.crosshair_thickness
        )
        pygame.draw.line(
            overlay,
            self.crosshair_color,
            (center_x, center_y + self.crosshair_gap),
            (center_x, center_y + self.crosshair_gap + self.crosshair_length),
            self.crosshair_thickness
        )

        # Composite overlay onto the screen surface
        surface.blit(overlay, (0, 0))