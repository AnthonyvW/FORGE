"""
Red mark detection overlay for camera view.

This overlay detects red registration marks in real-time from the camera feed
and displays the detection results with center line and distance from center.
"""

import pygame
import numpy as np
import cv2
from typing import Optional, List, Tuple

from UI.frame import Frame
from UI.camera_view import CameraView
from UI.text import Text, TextStyle


class RedMarkDetectionOverlay(Frame):
    """
    Real-time red mark detection overlay that shows detected registration marks
    and displays distance from image center.
    """
    def __init__(
        self,
        camera_view: CameraView,
        visible: bool = False,
        
        # Detection parameters
        min_area: int = 50,
        max_area: int = 10000,
        max_aspect_ratio: float = 3.0,
        red_threshold_percentile: int = 75,

        # Stabilization / smoothing parameters
        smoothing_alpha: float = 0.25,
        deadband_px: float = 2.0,
        max_step_px_per_frame: float = 30.0,

        # Orientation switching: if marks are clustered on one side, draw a horizontal line
        side_cluster_fraction: float = 0.85,
        side_cluster_margin: float = 0.18,
        
        # Visual parameters
        valid_mark_color: pygame.Color = pygame.Color(0, 255, 0),  # Green
        filtered_mark_color: pygame.Color = pygame.Color(255, 0, 0),  # Red
        center_line_color: pygame.Color = pygame.Color(255, 255, 0),  # Yellow
        text_color: pygame.Color = pygame.Color(255, 255, 255),  # White
        text_bg_color: pygame.Color = pygame.Color(0, 0, 0, 180),  # Semi-transparent black
    ):
        super().__init__(
            parent=camera_view,
            x=0, y=0,
            width=1, height=1,
            x_is_percent=True, y_is_percent=True,
            width_is_percent=True, height_is_percent=True,
            z_index=camera_view.z_index + 2,
            background_color=None
        )

        self.camera_view = camera_view
        self.visible = visible
        self.mouse_passthrough = True
        
        # Detection parameters
        self.min_area = min_area
        self.max_area = max_area
        self.max_aspect_ratio = max_aspect_ratio
        self.red_threshold_percentile = red_threshold_percentile

        # Stabilization parameters
        self.smoothing_alpha = float(smoothing_alpha)
        self.deadband_px = float(deadband_px)
        self.max_step_px_per_frame = float(max_step_px_per_frame)

        # Orientation switching parameters
        self.side_cluster_fraction = float(side_cluster_fraction)
        self.side_cluster_margin = float(side_cluster_margin)
        
        # Visual parameters
        self.valid_mark_color = valid_mark_color
        self.filtered_mark_color = filtered_mark_color
        self.center_line_color = center_line_color
        self.text_color = text_color
        self.text_bg_color = text_bg_color
        
        # Detection results (updated each frame)
        self.valid_centers: List[Tuple[float, float]] = []
        self.filtered_centers: List[Tuple[float, float]] = []
        # Raw per-frame measurements
        self.mean_x_raw: Optional[float] = None
        self.mean_y_raw: Optional[float] = None

        # Stabilized (displayed) measurements
        self.mean_x: Optional[float] = None
        self.mean_y: Optional[float] = None
        self.distance_from_center: Optional[float] = None
        self.image_width: Optional[int] = None
        self.image_height: Optional[int] = None
        
        # Jump threshold: if raw value is this many pixels away, jump instantly
        self.jump_threshold_px: float = 50.0
        
        # Cached overlay surface
        self._overlay = None
        self._overlay_size = None

        # Filter state
        self._smoothed_mean_x: Optional[float] = None
        self._smoothed_mean_y: Optional[float] = None

        # Line orientation: 'vertical' (default) or 'horizontal'
        self._line_orientation: str = 'vertical'

        # Hysteresis counter to prevent rapid flipping
        self._cluster_frames: int = 0

    # ==================== Public API ====================
    
    def toggle(self) -> None:
        """Toggle visibility of the detection overlay."""
        self.visible = not self.visible
        
    def set_visible(self, value: bool) -> None:
        """Set visibility of the detection overlay."""
        self.visible = bool(value)
    
    def is_visible(self) -> bool:
        """Check if overlay is currently visible."""
        return self.visible
    
    def set_jump_threshold(self, threshold_px: float) -> None:
        """
        Set the jump threshold in pixels.
        When the raw measurement differs from the smoothed value by more than this
        threshold, the smoothed value will jump directly to the raw value instead
        of gradually moving towards it.
        
        Args:
            threshold_px: Jump threshold in pixels (default: 50.0)
        """
        self.jump_threshold_px = float(threshold_px)

    # ==================== Detection ====================
    
    def _detect_red_marks(self, img_rgb: np.ndarray) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Detect red registration marks in an RGB image.
        
        Args:
            img_rgb: RGB image as numpy array (H, W, 3)
            
        Returns:
            Tuple of (valid_centers, filtered_centers)
        """
        # Extract channels
        r = img_rgb[:, :, 0].astype(float)
        g = img_rgb[:, :, 1].astype(float)
        b = img_rgb[:, :, 2].astype(float)
        
        # Enhanced red isolation: R - max(G, B) to handle reflections better
        red_isolated = r - np.maximum(g, b)
        red_isolated = np.clip(red_isolated, 0, 255)
        
        # Normalize to 0-255 range
        if red_isolated.max() > 0:
            red_isolated = (red_isolated / red_isolated.max() * 255).astype(np.uint8)
        else:
            red_isolated = red_isolated.astype(np.uint8)
        
        # Apply morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        red_isolated = cv2.morphologyEx(red_isolated, cv2.MORPH_CLOSE, kernel)
        
        # Adaptive thresholding to handle varying lighting
        threshold = (
            np.percentile(red_isolated[red_isolated > 0], self.red_threshold_percentile) 
            if np.any(red_isolated > 0) 
            else 50
        )
        binary = (red_isolated > threshold).astype(np.uint8) * 255
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Get image dimensions
        img_height = img_rgb.shape[0]
        lower_half_y = img_height / 2
        
        # Filter components - restrict to lower half of image
        centers = []
        filtered_centers = []
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if self.min_area < area < self.max_area:
                x, y, w, h = (
                    stats[i, cv2.CC_STAT_LEFT],
                    stats[i, cv2.CC_STAT_TOP],
                    stats[i, cv2.CC_STAT_WIDTH],
                    stats[i, cv2.CC_STAT_HEIGHT]
                )
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
                
                if aspect_ratio < self.max_aspect_ratio:
                    center = (float(centroids[i][0]), float(centroids[i][1]))
                    # Check if in lower half
                    if center[1] >= lower_half_y:
                        centers.append(center)
                    else:
                        filtered_centers.append(center)
        
        # Remove outliers using IQR method on X coordinates
        valid_centers = centers.copy()
        outlier_centers = []
        
        if len(centers) > 3:  # Need at least 4 points for meaningful outlier detection
            x_coords = np.array([x for x, y in centers])
            q1 = np.percentile(x_coords, 25)
            q3 = np.percentile(x_coords, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            valid_centers = []
            outlier_centers = []
            for center in centers:
                if lower_bound <= center[0] <= upper_bound:
                    valid_centers.append(center)
                else:
                    outlier_centers.append(center)
        
        # Combine all filtered centers
        all_filtered = filtered_centers + outlier_centers
        
        return valid_centers, all_filtered

    # ==================== Stabilization ====================

    def _update_smoothed_value(self, prev: Optional[float], raw: Optional[float]) -> Optional[float]:
        """Update a stabilized value from a noisy per-frame measurement.

        - If there is no raw measurement this frame, hold the previous value.
        - Jump threshold: if raw is too far from prev, jump directly to it.
        - Deadband: ignore tiny changes.
        - Slew-rate limit: clamp maximum change per frame.
        - EMA: ease toward the (clamped) target.
        """
        if raw is None:
            return prev
        if prev is None:
            return float(raw)

        delta = float(raw) - float(prev)

        # If the raw value is too far away, jump directly to it
        if abs(delta) > self.jump_threshold_px:
            return float(raw)

        if abs(delta) <= self.deadband_px:
            return prev

        if self.max_step_px_per_frame > 0:
            delta = float(np.clip(delta, -self.max_step_px_per_frame, self.max_step_px_per_frame))

        alpha = float(np.clip(self.smoothing_alpha, 0.0, 1.0))
        return float(prev + alpha * delta)

    def _clustered_on_one_side(self, centers: List[Tuple[float, float]], img_w: int) -> bool:
        """Return True if most detected marks are clustered on the left or right side."""
        if not centers or img_w <= 0:
            self._cluster_frames = 0
            return False

        xs = np.array([c[0] for c in centers], dtype=np.float32)
        margin = float(np.clip(self.side_cluster_margin, 0.0, 0.45))
        left_edge = img_w * margin
        right_edge = img_w * (1.0 - margin)

        left_frac = float(np.mean(xs <= left_edge))
        right_frac = float(np.mean(xs >= right_edge))
        frac = max(left_frac, right_frac)

        # Simple hysteresis: require a couple consecutive frames before switching to horizontal.
        if frac >= self.side_cluster_fraction:
            self._cluster_frames += 1
        else:
            # decay rather than hard reset, to avoid flicker
            self._cluster_frames = max(0, self._cluster_frames - 1)

        return self._cluster_frames >= 2

    # ==================== Drawing ====================
    
    def _get_overlay(self, surface_size: Tuple[int, int]) -> pygame.Surface:
        """Return an RGBA overlay the size of the target surface (recreate on resize)."""
        if self._overlay is None or self._overlay_size != surface_size:
            self._overlay_size = surface_size
            self._overlay = pygame.Surface(surface_size, flags=pygame.SRCALPHA)
        else:
            self._overlay.fill((0, 0, 0, 0))
        return self._overlay
    
    def _draw_info_text(self, overlay: pygame.Surface, fr: Tuple[int, int, int, int]) -> None:
        """Draw information text at the top of the overlay."""
        if not self.valid_centers and not self.filtered_centers:
            return
        
        fx, fy, fw, fh = fr
        
        # Prepare text lines
        lines = []
        lines.append(f"Valid marks: {len(self.valid_centers)}")
        if self.filtered_centers:
            lines.append(f"Filtered: {len(self.filtered_centers)}")
        
        if self.image_width is not None and self.image_height is not None:
            lines.append(f"Line: {self._line_orientation}")

        if self._line_orientation == 'horizontal':
            if self.mean_y is not None and self.image_height is not None and self.distance_from_center is not None:
                lines.append(f"Center Y (stable): {self.mean_y:.1f} px")
                if self.mean_y_raw is not None:
                    lines.append(f"Center Y (raw): {self.mean_y_raw:.1f} px")

                center_y = self.image_height / 2
                if center_y > 0:
                    percent = (self.distance_from_center / center_y) * 100
                    direction = "down" if self.mean_y > center_y else "up"
                    lines.append(f"Distance from center: {self.distance_from_center:.1f} px ({percent:.1f}% {direction})")
        else:
            if self.mean_x is not None and self.image_width is not None and self.distance_from_center is not None:
                lines.append(f"Center X (stable): {self.mean_x:.1f} px")
                if self.mean_x_raw is not None:
                    lines.append(f"Center X (raw): {self.mean_x_raw:.1f} px")

                center_x = self.image_width / 2
                if center_x > 0:
                    percent = (self.distance_from_center / center_x) * 100
                    direction = "right" if self.mean_x > center_x else "left"
                    lines.append(f"Distance from center: {self.distance_from_center:.1f} px ({percent:.1f}% {direction})")
        
        # Render text with background
        font = pygame.font.Font(None, 24)
        y_offset = fy + 10
        
        for line in lines:
            text_surface = font.render(line, True, self.text_color)
            text_rect = text_surface.get_rect()
            text_rect.topleft = (fx + 10, y_offset)
            
            # Draw semi-transparent background
            bg_rect = text_rect.inflate(10, 4)
            bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            bg_surface.fill(self.text_bg_color)
            overlay.blit(bg_surface, bg_rect.topleft)
            
            # Draw text
            overlay.blit(text_surface, text_rect)
            y_offset += text_rect.height + 2

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the detection overlay if visible and camera is initialized."""
        if not self.visible:
            return

        if not self.camera_view.camera.initialized:
            return

        fr = self.camera_view.get_frame_rect()
        if not fr:
            return

        fx, fy, fw, fh = fr

        # Get the current camera frame
        arr = self.camera_view.camera.get_last_frame(prefer="latest", wait_for_still=False)
        if arr is None:
            return
        
        # Ensure RGB format
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        
        img_h, img_w = arr.shape[:2]
        self.image_width = img_w
        self.image_height = img_h
        
        # Detect red marks
        self.valid_centers, self.filtered_centers = self._detect_red_marks(arr)
        
        # Raw mean measurements
        if self.valid_centers:
            self.mean_x_raw = float(np.mean([x for x, y in self.valid_centers]))
            self.mean_y_raw = float(np.mean([y for x, y in self.valid_centers]))
        else:
            self.mean_x_raw = None
            self.mean_y_raw = None

        # Update stabilized values
        self._smoothed_mean_x = self._update_smoothed_value(self._smoothed_mean_x, self.mean_x_raw)
        self._smoothed_mean_y = self._update_smoothed_value(self._smoothed_mean_y, self.mean_y_raw)
        self.mean_x = self._smoothed_mean_x
        self.mean_y = self._smoothed_mean_y

        # Decide whether we should draw a horizontal or vertical line
        clustered = self._clustered_on_one_side(self.valid_centers, img_w)
        self._line_orientation = 'horizontal' if clustered else 'vertical'

        # Distance from center depends on which line we're drawing
        if self._line_orientation == 'horizontal':
            if self.mean_y is not None:
                center_y = img_h / 2.0
                self.distance_from_center = float(abs(self.mean_y - center_y))
            else:
                self.distance_from_center = None
        else:
            if self.mean_x is not None:
                center_x = img_w / 2.0
                self.distance_from_center = float(abs(self.mean_x - center_x))
            else:
                self.distance_from_center = None
        
        # Build/resize overlay and clear it
        overlay = self._get_overlay(surface.get_size())
        overlay.fill((0, 0, 0, 0))
        
        # Calculate scaling factor (image coordinates to display coordinates)
        scale_x = fw / img_w
        scale_y = fh / img_h
        
        # Draw filtered out centers in red
        for x, y in self.filtered_centers:
            display_x = fx + x * scale_x
            display_y = fy + y * scale_y
            pygame.draw.circle(overlay, self.filtered_mark_color, (int(display_x), int(display_y)), 5)
        
        # Draw valid center dots in green
        for x, y in self.valid_centers:
            display_x = fx + x * scale_x
            display_y = fy + y * scale_y
            pygame.draw.circle(overlay, self.valid_mark_color, (int(display_x), int(display_y)), 5)
        
        # Draw the stabilized line
        if self._line_orientation == 'horizontal':
            if self.mean_y is not None:
                display_mean_y = fy + self.mean_y * scale_y
                pygame.draw.line(
                    overlay,
                    self.center_line_color,
                    (fx, int(display_mean_y)),
                    (fx + fw, int(display_mean_y)),
                    2
                )

                # Also draw the image center line for reference (dimmer)
                center_y = img_h / 2
                display_center_y = fy + center_y * scale_y
                pygame.draw.line(
                    overlay,
                    (*self.center_line_color[:3], 100),
                    (fx, int(display_center_y)),
                    (fx + fw, int(display_center_y)),
                    1
                )
        else:
            if self.mean_x is not None:
                display_mean_x = fx + self.mean_x * scale_x
                pygame.draw.line(
                    overlay,
                    self.center_line_color,
                    (int(display_mean_x), fy),
                    (int(display_mean_x), fy + fh),
                    2
                )

                # Also draw the image center line for reference (dimmer)
                center_x = img_w / 2
                display_center_x = fx + center_x * scale_x
                pygame.draw.line(
                    overlay,
                    (*self.center_line_color[:3], 100),
                    (int(display_center_x), fy),
                    (int(display_center_x), fy + fh),
                    1
                )
        
        # Draw info text
        self._draw_info_text(overlay, fr)
        
        # Composite overlay onto the screen surface
        surface.blit(overlay, (0, 0))