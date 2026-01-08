import pygame
import numpy as np
import cv2
import time
from typing import Optional, Tuple

from UI.frame import Frame
from UI.camera_view import CameraView
from UI.styles import (
    CROSSHAIR_COLOR,
    CROSSHAIR_LENGTH,
    CROSSHAIR_THICKNESS,
    CROSSHAIR_GAP,
)


class InteractiveCameraOverlay(Frame):
    """
    UI overlay that renders a crosshair in the center of the camera view.
    Supports click-to-move and camera calibration using phase correlation.
    Only displays when the camera is initialized.
    """
    def __init__(
        self,
        camera_view: CameraView,
        controller,  # AutomatedPrinter instance
        visible: bool = True,
        
        # Crosshair visual properties (defaults from styles.py)
        crosshair_color: Optional[pygame.Color] = None,
        crosshair_length: Optional[int] = None,
        crosshair_thickness: Optional[int] = None,
        crosshair_gap: Optional[int] = None,
        
        # Calibration parameters (world units in 0.01mm)
        cal_move_x_ticks: int = 100,  # 1.00mm in 0.01mm units
        cal_move_y_ticks: int = 100,  # 1.00mm in 0.01mm units (increased from 75 for better correlation)
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
        
        # Enable click handling - CRITICAL for receiving mouse events
        self.mouse_passthrough = False

        # Crosshair properties (use styles.py defaults if not provided)
        self.crosshair_color = crosshair_color if crosshair_color is not None else CROSSHAIR_COLOR
        self.crosshair_length = crosshair_length if crosshair_length is not None else CROSSHAIR_LENGTH
        self.crosshair_thickness = crosshair_thickness if crosshair_thickness is not None else CROSSHAIR_THICKNESS
        self.crosshair_gap = crosshair_gap if crosshair_gap is not None else CROSSHAIR_GAP

        # Cache overlay surface
        self._overlay = None
        self._overlay_size = None
        
        # Calibration state
        self.M_est = None  # 2x2 estimated mapping matrix (pixels = M * world_delta)
        self.M_inv = None  # Inverse mapping (world_delta = M_inv * pixel_delta)
        self._cal_move_x = cal_move_x_ticks
        self._cal_move_y = cal_move_y_ticks
        self._calibrating = False
        self._cal_ref_pos = None  # Position where calibration was performed (camera center reference)
        self._cal_start_x_mm = None  # Starting X position for absolute moves
        self._cal_start_y_mm = None  # Starting Y position for absolute moves
        
        # Store calibration data during process
        self._cal_base_pos = None
        self._cal_edges_base = None
        self._dp1 = None
        self._dp2 = None

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

    def _get_overlay(self, surface_size: tuple[int, int]) -> pygame.Surface:
        """Return an RGBA overlay the size of the target surface (recreate on resize)."""
        if self._overlay is None or self._overlay_size != surface_size:
            self._overlay_size = surface_size
            self._overlay = pygame.Surface(surface_size, flags=pygame.SRCALPHA)
        else:
            # Clear with fully transparent color
            self._overlay.fill((0, 0, 0, 0))
        return self._overlay

    # ==================== Calibration Methods ====================
    
    def _surface_to_gray_cv(self, arr: np.ndarray) -> np.ndarray:
        """Convert RGB numpy array to grayscale for OpenCV."""
        if arr.ndim == 2:
            return arr
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return gray

    def _edges_canny(self, gray_u8: np.ndarray) -> np.ndarray:
        """Compute normalized Canny edges."""
        g = cv2.GaussianBlur(gray_u8, (5, 5), 0)
        e = cv2.Canny(g, 60, 180)
        ef = e.astype(np.float32)
        ef -= ef.mean()
        ef /= (ef.std() + 1e-6)
        return ef

    def _phase_corr_shift(self, img_a_f32: np.ndarray, img_b_f32: np.ndarray) -> Tuple[float, float, float]:
        """Compute phase correlation shift between two images."""
        win = cv2.createHanningWindow((img_a_f32.shape[1], img_a_f32.shape[0]), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(img_a_f32, img_b_f32, win)
        return float(dx), float(dy), float(response)

    def _capture_and_process_edges(self) -> Optional[np.ndarray]:
        """Capture a still image and return its edge map."""
        try:
            # Capture still
            self.camera_view.camera.capture_image()
            while self.camera_view.camera.is_taking_image:
                time.sleep(0.01)
            
            # Get frame as numpy array
            arr = self.camera_view.camera.get_last_frame(prefer="still", wait_for_still=False)
            if arr is None:
                return None
            
            # Store the calibration image resolution (the actual image we're correlating)
            if not hasattr(self, '_cal_image_height'):
                self._cal_image_height = arr.shape[0]
                self._cal_image_width = arr.shape[1]
                
                # Also get the current display frame size
                fr = self.camera_view.get_frame_rect()
                if fr:
                    _, _, display_w, display_h = fr
                    self._cal_display_width = display_w
                    self._cal_display_height = display_h
                    
                    self.controller.status(
                        f"Calibration image: {self._cal_image_width}x{self._cal_image_height}, "
                        f"Display frame: {display_w:.0f}x{display_h:.0f}",
                        True
                    )
                else:
                    self.controller.status(
                        f"Calibration using image resolution: {self._cal_image_width}x{self._cal_image_height}",
                        True
                    )
            
            # Convert to grayscale and compute edges
            gray = self._surface_to_gray_cv(arr)
            edges = self._edges_canny(gray)
            return edges
        except Exception as e:
            self.controller.status(f"Edge capture failed: {e}", True)
            return None

    def run_calibration(self) -> None:
        """
        Run the calibration routine to determine the mapping between screen pixels and world coordinates.
        This will move the printer to two positions and use phase correlation to determine the transformation.
        Similar to start_autofocus in automated_controller.py.
        """
        if self._calibrating:
            self.controller.status("Calibration already in progress.", True)
            return
        
        self._calibrating = True
        self.controller.status("Starting camera calibration...", True)
        
        # Reset calibration state
        self.M_est = None
        self.M_inv = None
        self._dp1 = None
        self._dp2 = None
        
        # Store the starting position BEFORE building the macro
        # This ensures we know where we started from
        start_pos = self.controller.get_position()
        self._cal_start_x_mm = start_pos.x / 100.0
        self._cal_start_y_mm = start_pos.y / 100.0
        
        # Build calibration macro using ABSOLUTE positioning
        steps = []
        
        # Step 1: Capture base image
        steps.append(self.controller.create_cmd(
            kind="CALIBRATION_BASE",
            value="",
            message="Capturing base calibration image...",
            log=True
        ))
        
        # Step 2: Move to X+ offset (absolute positioning)
        move_x_mm = self._cal_move_x / 100.0
        target_x_mm = self._cal_start_x_mm + move_x_mm
        steps.append(self.controller.printer_cmd(
            f"G0 X{target_x_mm:.2f}",
            message=f"Moving +X {move_x_mm:.2f}mm for calibration...",
            log=True
        ))
        
        # Step 3: Capture moved image 1
        steps.append(self.controller.create_cmd(
            kind="CALIBRATION_MOVE1",
            value="",
            message="Capturing X-moved calibration image...",
            log=True
        ))
        
        # Step 4: Return to base X
        steps.append(self.controller.printer_cmd(
            f"G0 X{self._cal_start_x_mm:.2f}",
            message="Returning to base X position...",
            log=True
        ))
        
        # Step 5: Move to Y+ offset (absolute positioning)
        move_y_mm = self._cal_move_y / 100.0
        target_y_mm = self._cal_start_y_mm + move_y_mm
        steps.append(self.controller.printer_cmd(
            f"G0 Y{target_y_mm:.2f}",
            message=f"Moving +Y {move_y_mm:.2f}mm for calibration...",
            log=True
        ))
        
        # Step 6: Capture moved image 2
        steps.append(self.controller.create_cmd(
            kind="CALIBRATION_MOVE2",
            value="",
            message="Capturing Y-moved calibration image...",
            log=True
        ))
        
        # Step 7: Return to base Y
        steps.append(self.controller.printer_cmd(
            f"G0 Y{self._cal_start_y_mm:.2f}",
            message="Returning to base position...",
            log=True
        ))
        
        # Step 8: Finish calibration
        steps.append(self.controller.create_cmd(
            kind="CALIBRATION_FINISH",
            value="",
            message="Computing calibration matrix...",
            log=True
        ))
        
        # Register handlers for calibration commands
        self.controller.register_handler("CALIBRATION_BASE", self._handle_calibration_base)
        self.controller.register_handler("CALIBRATION_MOVE1", self._handle_calibration_move1)
        self.controller.register_handler("CALIBRATION_MOVE2", self._handle_calibration_move2)
        self.controller.register_handler("CALIBRATION_FINISH", self._handle_calibration_finish)
        
        # Create and enqueue macro
        macro = self.controller.macro_cmd(
            steps,
            wait_printer=True,
            message="Camera calibration routine",
            log=True
        )
        self.controller.enqueue_cmd(macro)

    def _handle_calibration_base(self, cmd) -> None:
        """Handler: Capture base image."""
        time.sleep(0.5)  # Allow printer to fully settle (increased for Y-axis stability)
        self._cal_base_pos = self.controller.get_position()
        self._cal_edges_base = self._capture_and_process_edges()
        if self._cal_edges_base is None:
            self.controller.status("Failed to capture base calibration image.", True)
            self._calibrating = False

    def _handle_calibration_move1(self, cmd) -> None:
        """Handler: Capture first moved image and compute shift."""
        time.sleep(0.5)  # Allow printer to fully settle (increased for Y-axis stability)
        edges = self._capture_and_process_edges()
        if edges is None or self._cal_edges_base is None:
            self.controller.status("Failed to capture first calibration image.", True)
            self._calibrating = False
            return
        
        dx, dy, response = self._phase_corr_shift(self._cal_edges_base, edges)
        self._dp1 = np.array([dx, dy], dtype=np.float64)
        self._response1 = response
        self.controller.status(f"X-move shift: dx={dx:.2f}, dy={dy:.2f}, response={response:.3f}", True)
        
        # Warn if correlation confidence is low
        if response < 0.3:
            self.controller.status("WARNING: Low phase correlation confidence for X-move. Calibration may be inaccurate.", True)

    def _handle_calibration_move2(self, cmd) -> None:
        """Handler: Capture second moved image and compute shift."""
        time.sleep(0.5)  # Allow printer to fully settle (increased for Y-axis stability)
        edges = self._capture_and_process_edges()
        if edges is None or self._cal_edges_base is None:
            self.controller.status("Failed to capture second calibration image.", True)
            self._calibrating = False
            return
        
        dx, dy, response = self._phase_corr_shift(self._cal_edges_base, edges)
        self._dp2 = np.array([dx, dy], dtype=np.float64)
        self._response2 = response
        self.controller.status(f"Y-move shift: dx={dx:.2f}, dy={dy:.2f}, response={response:.3f}", True)
        
        # Warn if correlation confidence is low
        if response < 0.3:
            self.controller.status("WARNING: Low phase correlation confidence for Y-move. Calibration may be inaccurate.", True)

    def _handle_calibration_finish(self, cmd) -> None:
        """Handler: Compute final calibration matrix."""
        if self._dp1 is None or self._dp2 is None:
            self.controller.status("Calibration failed: missing measurements.", True)
            self._calibrating = False
            return
        
        # World deltas (in 0.01mm units) - these are what we commanded
        DW1 = np.array([self._cal_move_x, 0.0], dtype=np.float64)
        DW2 = np.array([0.0, self._cal_move_y], dtype=np.float64)
        
        # Pixel deltas - these are what we measured
        # Phase correlation: positive dx/dy means second image shifted right/down
        # 
        # For a standard camera (not inverted):
        #   Stage +X → Image shifts LEFT (negative dx)
        #   Stage +Y → Image shifts DOWN (positive dy)
        #
        # But camera orientation varies! We need to check the SIGN of the correlation:
        #   - If stage +X causes image +dx → camera X follows stage (don't negate)
        #   - If stage +X causes image -dx → camera X opposes stage (negate)
        #   - Same logic for Y
        #
        # From your calibration:
        #   X-move: dx=+452.95 → camera X follows stage → use as-is
        #   Y-move: dy=-341.63 → camera Y opposes stage → negate
        
        # Detect orientation by checking correlation signs
        x_inverted = (self._dp1[0] < 0)  # True if camera X opposes stage X
        y_inverted = (self._dp2[1] < 0)  # True if camera Y opposes stage Y
        
        dp1_corrected = self._dp1.copy()
        dp2_corrected = self._dp2.copy()
        
        if x_inverted:
            dp1_corrected[0] = -dp1_corrected[0]
        if y_inverted:
            dp2_corrected[1] = -dp2_corrected[1]
        
        # Debug: show raw and corrected deltas
        self.controller.status(
            f"Raw pixel deltas: X-move=[{self._dp1[0]:.2f}, {self._dp1[1]:.2f}], "
            f"Y-move=[{self._dp2[0]:.2f}, {self._dp2[1]:.2f}]",
            True
        )
        self.controller.status(
            f"Corrected deltas: X-move=[{dp1_corrected[0]:.2f}, {dp1_corrected[1]:.2f}], "
            f"Y-move=[{dp2_corrected[0]:.2f}, {dp2_corrected[1]:.2f}]",
            True
        )
        
        # Report detected orientation
        orient_msg = f"Camera orientation: X={'inverted' if x_inverted else 'normal'}, Y={'inverted' if y_inverted else 'normal'}"
        self.controller.status(orient_msg, True)
        
        # Build matrices: columns are the basis vectors
        # DP: pixel space basis (each column is pixel response to a world move)
        # DW: world space basis (each column is a world delta)
        DP = np.column_stack([dp1_corrected, dp2_corrected])  # 2x2
        DW = np.column_stack([DW1, DW2])                       # 2x2
        
        # Check if DW is invertible
        det_dw = np.linalg.det(DW)
        if abs(det_dw) < 1e-9:
            self.controller.status("Calibration failed: DW matrix singular.", True)
            self._calibrating = False
            return
        
        # The relationship is: dp = M @ dw
        # Therefore: M = DP @ inv(DW)
        # This maps world deltas (in 0.01mm units) to pixel deltas
        self.M_est = DP @ np.linalg.inv(DW)
        
        # Check if M is invertible
        detM = np.linalg.det(self.M_est)
        if abs(detM) < 1e-9:
            self.controller.status("Calibration failed: M not invertible.", True)
            self.M_inv = None
            self._calibrating = False
            return
        
        # Invert to get world = M_inv @ pixel
        self.M_inv = np.linalg.inv(self.M_est)
        
        # Debug: show M_inv for understanding click-to-world mapping
        self.controller.status(
            f"M_inv (pixel→world) = [[{self.M_inv[0,0]:.6f}, {self.M_inv[0,1]:.6f}], "
            f"[{self.M_inv[1,0]:.6f}, {self.M_inv[1,1]:.6f}]]",
            True
        )
        
        # Store the calibration reference position (where camera center corresponds to)
        # This is the position at the END of calibration (after returning to base)
        self._cal_ref_pos = self.controller.get_position()
        
        # Calculate DPI for full resolution (2592x1944)
        # Extract pixels per 0.01mm unit from M_est
        # M_est[0,0] is dx_pixels per 1.00 unit of world X (which is 0.01mm)
        # M_est[1,1] is dy_pixels per 1.00 unit of world Y (which is 0.01mm)
        px_per_0p01mm_x = abs(self.M_est[0, 0])
        px_per_0p01mm_y = abs(self.M_est[1, 1])
        
        # Convert to pixels per mm
        px_per_mm_x = px_per_0p01mm_x * 100.0
        px_per_mm_y = px_per_0p01mm_y * 100.0
        px_per_mm_avg = (px_per_mm_x + px_per_mm_y) / 2
        
        # DPI = pixels per mm * mm per inch (25.4)
        dpi = px_per_mm_avg * 25.4
        
        # Log results
        self.controller.status(
            f"Calibration complete! M = [[{self.M_est[0,0]:.3f}, {self.M_est[0,1]:.3f}], "
            f"[{self.M_est[1,0]:.3f}, {self.M_est[1,1]:.3f}]]",
            True
        )
        self.controller.status(
            f"Estimated full-res DPI: {dpi:.1f} (avg {px_per_mm_avg:.2f} px/mm, "
            f"X: {px_per_mm_x:.2f} px/mm, Y: {px_per_mm_y:.2f} px/mm)",
            True
        )
        
        self._calibrating = False

    # ==================== Click-to-Move ====================
    
    def go_to_calibration_pattern(self) -> None:
        """
        Move the printer to the overlay calibration pattern position.
        First moves Z up to 33.12mm, then moves to X=226.24mm, Y=187.08mm.
        """
        self.controller.reset_after_stop()
        
        # Move Z up first (safe height)
        self.controller.enqueue_printer(
            "G0 Z33.12",
            message="Moving to calibration height Z=33.12mm",
            log=True
        )
        
        # Then move to XY position
        self.controller.enqueue_printer(
            "G0 X226.08 Y186.90",
            message="Moving to calibration pattern at X=226.24mm, Y=187.08mm",
            log=True
        )
    
    def _click_to_world_delta(self, screen_x: int, screen_y: int) -> Optional[Tuple[float, float]]:
        """
        Convert a screen click position to world delta (in 0.01mm units).
        Returns None if calibration hasn't been run yet.
        """
        if self.M_inv is None:
            return None
        
        # Get frame rectangle
        fr = self.camera_view.get_frame_rect()
        if not fr:
            return None
        
        fx, fy, fw, fh = fr
        
        # Calculate center of camera frame
        center_x = fx + fw / 2
        center_y = fy + fh / 2
        
        # Pixel delta from center (in DISPLAY coordinates)
        pixel_delta = np.array([screen_x - center_x, screen_y - center_y], dtype=np.float64)
        
        # CRITICAL: Scale pixel delta to match calibration image coordinates
        # During calibration, we measured pixel shifts in the calibration image resolution
        # But clicks are measured in the displayed frame resolution
        # We need to scale clicks by (calibration_image_size / displayed_frame_size)
        if hasattr(self, '_cal_image_width') and hasattr(self, '_cal_image_height'):
            # Scale factor = calibration image pixels / display frame pixels
            scale_x = self._cal_image_width / fw
            scale_y = self._cal_image_height / fh
            
            pixel_delta_scaled = np.array([
                pixel_delta[0] * scale_x,
                pixel_delta[1] * scale_y
            ], dtype=np.float64)
            
            # Debug: show scaling (only once)
            if not hasattr(self, '_scaling_reported'):
                self._scaling_reported = True
                self.controller.status(
                    f"Click scaling: Display {fw:.0f}x{fh:.0f} → Calibration {self._cal_image_width}x{self._cal_image_height} "
                    f"(scale X={scale_x:.3f}, Y={scale_y:.3f})",
                    True
                )
        else:
            pixel_delta_scaled = pixel_delta
        
        # Convert to world delta using inverse mapping
        dw = self.M_inv @ pixel_delta_scaled
        
        # Store for debug output (will be printed in on_click)
        self._last_pixel_delta = pixel_delta
        self._last_pixel_delta_scaled = pixel_delta_scaled
        self._last_world_delta = dw
        
        return float(dw[0]), float(dw[1])

    def on_click(self, button=None) -> None:
        """Handle click events to move printer to clicked position."""
        if not self.camera_view.camera.initialized:
            return
        
        if self.M_inv is None:
            self.controller.status("Cannot move: run calibration first (call run_calibration())", True)
            return
        
        # Get mouse position
        mouse_x, mouse_y = pygame.mouse.get_pos()
        
        # Convert to world delta from the CURRENT printer position
        # The calibration matrix M_inv tells us how pixel deltas map to world deltas
        result = self._click_to_world_delta(mouse_x, mouse_y)
        if result is None:
            return
        
        dx_ticks, dy_ticks = result
        
        # CRITICAL FIX: Negate X-axis because screen X and stage X are opposite
        # When you click right (positive screen X), stage should move right (positive stage X)
        # But empirically, clicking right moves left, so we need to flip it
        dx_ticks = -dx_ticks
        
        # Get CURRENT position (where the camera actually is now)
        current_pos = self.controller.get_position()
        
        # Debug: show pixel delta, world delta (before X flip), and world delta (after X flip)
        if hasattr(self, '_last_pixel_delta') and hasattr(self, '_last_world_delta'):
            if hasattr(self, '_last_pixel_delta_scaled'):
                self.controller.status(
                    f"Pixel Δ (display): [{self._last_pixel_delta[0]:.1f}, {self._last_pixel_delta[1]:.1f}] → "
                    f"Pixel Δ (scaled): [{self._last_pixel_delta_scaled[0]:.1f}, {self._last_pixel_delta_scaled[1]:.1f}] → "
                    f"World (raw): [{self._last_world_delta[0]:.1f}, {self._last_world_delta[1]:.1f}] → "
                    f"World: [{dx_ticks:.1f}, {dy_ticks:.1f}] ticks",
                    True
                )
            else:
                self.controller.status(
                    f"Pixel delta: [{self._last_pixel_delta[0]:.1f}, {self._last_pixel_delta[1]:.1f}] → "
                    f"World (raw): [{self._last_world_delta[0]:.1f}, {self._last_world_delta[1]:.1f}] → "
                    f"World: [{dx_ticks:.1f}, {dy_ticks:.1f}] ticks",
                    True
                )
        
        # Calculate new position relative to CURRENT position
        # The delta tells us how far from center we clicked
        new_x_ticks = current_pos.x + int(round(dx_ticks))
        new_y_ticks = current_pos.y + int(round(dy_ticks))
        
        # Convert to mm for G-code
        new_x_mm = new_x_ticks / 100.0
        new_y_mm = new_y_ticks / 100.0
        
        # Bounds check
        max_x = self.controller.get_max_x()
        max_y = self.controller.get_max_y()
        
        if not (0 <= new_x_mm <= max_x and 0 <= new_y_mm <= max_y):
            self.controller.status(
                f"Click position out of bounds: ({new_x_mm:.2f}, {new_y_mm:.2f})",
                True
            )
            return
        
        # Send move command
        self.controller.enqueue_printer(
            f"G0 X{new_x_mm:.2f} Y{new_y_mm:.2f}",
            message=f"Moving to clicked position: X={new_x_mm:.2f}, Y={new_y_mm:.2f}",
            log=True
        )

    def on_wheel(self, dx: int, dy: int, px: int, py: int) -> bool:
        """
        Handle mousewheel events to adjust Z-axis position when hovering over camera view.
        Uses the printer's minimum step size (0.04mm) for precise control.
        
        Args:
            dx: Horizontal wheel movement (unused for Z-axis)
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
        # Each wheel tick moves by the minimum step size
        dz_ticks = dy * MIN_STEP_TICKS
        
        # Calculate new Z position
        new_z_ticks = current_pos.z + int(round(dz_ticks))
        new_z_mm = new_z_ticks / 100.0
        
        # Bounds check
        max_z = self.controller.get_max_z()
        if not (0 <= new_z_mm <= max_z):
            self.controller.status(
                f"Z position out of bounds: {new_z_mm:.2f}mm (max: {max_z}mm)",
                False  # Don't log to console, just status
            )
            return True  # Still handled, just rejected
        
        # Send move command
        self.controller.enqueue_printer(
            f"G0 Z{new_z_mm:.2f}",
            message=f"Z: {new_z_mm:.2f}mm",
            log=False  # Don't clutter the log with every wheel movement
        )
        
        return True  # Event was handled

    # ==================== Drawing ====================

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the crosshair overlay if visible and camera is initialized."""
        if not self.visible:
            return

        # Only draw if camera is initialized
        if not self.camera_view.camera.initialized:
            return

        # Get the frame rectangle from camera view
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

        # Draw crosshair lines
        # Horizontal line (left and right segments with gap in middle)
        # Left segment
        pygame.draw.line(
            overlay,
            self.crosshair_color,
            (center_x - self.crosshair_gap - self.crosshair_length, center_y),
            (center_x - self.crosshair_gap, center_y),
            self.crosshair_thickness
        )
        # Right segment
        pygame.draw.line(
            overlay,
            self.crosshair_color,
            (center_x + self.crosshair_gap, center_y),
            (center_x + self.crosshair_gap + self.crosshair_length, center_y),
            self.crosshair_thickness
        )

        # Vertical line (top and bottom segments with gap in middle)
        # Top segment
        pygame.draw.line(
            overlay,
            self.crosshair_color,
            (center_x, center_y - self.crosshair_gap - self.crosshair_length),
            (center_x, center_y - self.crosshair_gap),
            self.crosshair_thickness
        )
        # Bottom segment
        pygame.draw.line(
            overlay,
            self.crosshair_color,
            (center_x, center_y + self.crosshair_gap),
            (center_x, center_y + self.crosshair_gap + self.crosshair_length),
            self.crosshair_thickness
        )

        # Composite overlay onto the actual screen surface
        surface.blit(overlay, (0, 0))