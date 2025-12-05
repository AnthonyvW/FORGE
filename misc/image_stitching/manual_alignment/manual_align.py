"""
Manual Image Alignment Tool with Real-Time Scoring

A Pygame-based interactive tool for manually aligning two images and
seeing the alignment score update in real-time.

Controls:
    Arrow Keys: Move Image 2 (X and Y position)
    Shift + Arrow Keys: Move in smaller increments (1px)
    W/S: Adjust overlap amount (move Image 2 left/right)
    Shift + W/S: Adjust overlap in smaller increments (1px)
    Space: Toggle semi-transparent overlay mode
    R: Reset to initial position
    C: Toggle crosshair at overlap boundary
    Z/X: Adjust transparency (in overlay mode)
    Mouse Wheel: Zoom in/out (centered on mouse position)
    Middle Mouse: Pan view (click and drag)
    Tab: Reset zoom to fit entire view
    Q/Escape: Quit

Usage:
    python manual_alignment_tool.py image1.jpg image2.jpg [--rotate-90]
"""

import sys
import argparse
from pathlib import Path
import pygame
import cv2 as cv
import numpy as np


class ManualAlignmentTool:
    """Interactive tool for manual image alignment with scoring"""
    
    def __init__(self, img1_path, img2_path, rotate_90=False):
        """Initialize the alignment tool"""
        
        # Load images
        print(f"Loading images...")
        self.img1_bgr = cv.imread(str(img1_path))
        self.img2_bgr = cv.imread(str(img2_path))
        
        if self.img1_bgr is None or self.img2_bgr is None:
            raise ValueError("Could not load one or both images")
        
        # Rotate if requested
        if rotate_90:
            self.img1_bgr = cv.rotate(self.img1_bgr, cv.ROTATE_90_COUNTERCLOCKWISE)
            self.img2_bgr = cv.rotate(self.img2_bgr, cv.ROTATE_90_COUNTERCLOCKWISE)
        
        # Convert BGR to RGB for pygame
        self.img1_rgb = cv.cvtColor(self.img1_bgr, cv.COLOR_BGR2RGB)
        self.img2_rgb = cv.cvtColor(self.img2_bgr, cv.COLOR_BGR2RGB)
        
        # Store grayscale for scoring
        self.gray1 = cv.cvtColor(self.img1_bgr, cv.COLOR_BGR2GRAY)
        self.gray2 = cv.cvtColor(self.img2_bgr, cv.COLOR_BGR2GRAY)
        
        self.h1, self.w1 = self.img1_rgb.shape[:2]
        self.h2, self.w2 = self.img2_rgb.shape[:2]
        
        print(f"Image 1: {self.w1}x{self.h1}")
        print(f"Image 2: {self.w2}x{self.h2}")
        
        # Initialize pygame
        pygame.init()
        
        # Calculate window size
        display_info = pygame.display.Info()
        screen_w = display_info.current_w - 100
        screen_h = display_info.current_h - 150
        
        # Fixed window size
        self.window_width = min(1600, screen_w)
        self.window_height = min(900, screen_h)
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        
        pygame.display.set_caption("Manual Image Alignment Tool")
        
        # Calculate canvas area (left side for images)
        self.canvas_width = self.window_width - 350
        self.canvas_height = self.window_height - 50
        self.canvas_x = 25
        self.canvas_y = 25
        
        # Alignment parameters (must be set before calculate_fit_zoom)
        self.img2_x = self.w1 - int(self.w1 * 0.5)  # 50% overlap initially
        self.img2_y = 0
        
        # Zoom settings
        self.zoom = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.zoom_speed = 0.1
        
        # Calculate initial zoom to fit (now that img2_x/img2_y exist)
        self.calculate_fit_zoom()
        
        # Display settings
        self.overlay_mode = False
        self.show_crosshair = True
        self.transparency = 128  # 0-255
        
        # UI settings
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)
        self.score_history = []
        self.max_history = 50
        
        # Colors
        self.COLOR_BG = (30, 30, 30)
        self.COLOR_PANEL = (50, 50, 50)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_SCORE_HIGH = (0, 255, 0)
        self.COLOR_SCORE_MED = (255, 165, 0)
        self.COLOR_SCORE_LOW = (255, 0, 0)
        self.COLOR_CROSSHAIR = (0, 255, 255)
        self.COLOR_IMG1 = (0, 255, 0)
        self.COLOR_IMG2 = (255, 0, 0)
        
        # Scroll offset for panning
        self.scroll_x = 0
        self.scroll_y = 0
        
        # Mouse state for panning
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.pan_start_scroll_x = 0
        self.pan_start_scroll_y = 0
        
        print("\nControls:")
        print("  Arrow Keys: Move Image 2")
        print("  Shift+Arrows: Fine movement (1px)")
        print("  W/S: Adjust overlap")
        print("  Space: Toggle overlay mode")
        print("  C: Toggle crosshair")
        print("  Z/X: Adjust transparency")
        print("  Mouse Wheel: Zoom in/out (at mouse position)")
        print("  Middle Mouse: Pan view (drag)")
        print("  Tab: Reset zoom to fit")
        print("  R: Reset position")
        print("  Q/Esc: Quit")
    
    def calculate_fit_zoom(self):
        """Calculate zoom level to fit entire canvas in view"""
        # Calculate total canvas size at zoom 1.0
        canvas_w = max(self.w1, self.img2_x + self.w2)
        canvas_h = max(self.h1, abs(self.img2_y) + self.h2)
        
        # Calculate zoom to fit
        zoom_x = self.canvas_width / canvas_w if canvas_w > 0 else 1.0
        zoom_y = self.canvas_height / canvas_h if canvas_h > 0 else 1.0
        
        # Use the smaller zoom to ensure both dimensions fit
        self.zoom = min(zoom_x, zoom_y, 1.0)  # Don't zoom in beyond 1.0 for fit
        
        # Center the view
        self.scroll_x = 0
        self.scroll_y = 0
    
    def update_window_size(self):
        """Update window size - no longer needed but kept for compatibility"""
        pass
    
    def calculate_score(self):
        """Calculate alignment score for current position"""
        try:
            # Calculate overlap region
            overlap_x = max(0, self.w1 - self.img2_x)
            
            if overlap_x <= 0 or overlap_x > min(self.w1, self.w2):
                return None, "No overlap"
            
            # Handle Y offset
            y_offset = self.img2_y
            
            if y_offset >= 0:
                available_h1 = self.h1 - y_offset
                available_h2 = self.h2
                compare_height = min(available_h1, available_h2)
                
                if compare_height <= 10:
                    return None, "Insufficient height"
                
                region1 = self.gray1[y_offset:y_offset + compare_height, -overlap_x:]
                region2 = self.gray2[:compare_height, :overlap_x]
            else:
                available_h1 = self.h1
                available_h2 = self.h2 + y_offset
                compare_height = min(available_h1, available_h2)
                
                if compare_height <= 10:
                    return None, "Insufficient height"
                
                region1 = self.gray1[:compare_height, -overlap_x:]
                region2 = self.gray2[-y_offset:-y_offset + compare_height, :overlap_x]
            
            if region1.shape != region2.shape:
                return None, f"Shape mismatch: {region1.shape} vs {region2.shape}"
            
            # Calculate normalized cross-correlation
            result = cv.matchTemplate(region1, region2, cv.TM_CCOEFF_NORMED)
            score = float(result[0, 0])
            
            return score, "OK"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def get_score_color(self, score):
        """Get color based on score value"""
        if score is None:
            return self.COLOR_TEXT
        elif score >= 0.97:
            return self.COLOR_SCORE_HIGH
        elif score >= 0.85:
            return self.COLOR_SCORE_MED
        else:
            return self.COLOR_SCORE_LOW
    
    def get_confidence(self, score):
        """Get confidence level from score"""
        if score is None:
            return "N/A"
        elif score >= 0.99:
            return "VERY HIGH"
        elif score >= 0.97:
            return "HIGH"
        elif score >= 0.85:
            return "MEDIUM"
        elif score >= 0.70:
            return "LOW"
        else:
            return "VERY LOW"
    
    def render_canvas(self):
        """Render the aligned images on canvas"""
        # Create canvas surface
        canvas_w = int(max(self.w1, self.img2_x + self.w2) * self.zoom)
        canvas_h = int(max(self.h1, abs(self.img2_y) + self.h2) * self.zoom)
        canvas = pygame.Surface((canvas_w, canvas_h))
        canvas.fill(self.COLOR_BG)
        
        # Calculate positions with zoom
        img1_x = 0
        img1_y = max(0, -self.img2_y) if self.img2_y < 0 else 0
        
        img2_x = self.img2_x
        img2_y = max(0, self.img2_y) if self.img2_y >= 0 else 0
        
        if self.overlay_mode:
            # Overlay mode: blend the images
            # First draw image 1
            img1_surf = pygame.surfarray.make_surface(
                np.transpose(self.img1_rgb, (1, 0, 2))
            )
            if self.zoom != 1.0:
                img1_surf = pygame.transform.scale(
                    img1_surf,
                    (int(self.w1 * self.zoom), int(self.h1 * self.zoom))
                )
            canvas.blit(img1_surf, (int(img1_x * self.zoom), int(img1_y * self.zoom)))
            
            # Draw image 2 with transparency
            img2_surf = pygame.surfarray.make_surface(
                np.transpose(self.img2_rgb, (1, 0, 2))
            )
            if self.zoom != 1.0:
                img2_surf = pygame.transform.scale(
                    img2_surf,
                    (int(self.w2 * self.zoom), int(self.h2 * self.zoom))
                )
            img2_surf.set_alpha(self.transparency)
            canvas.blit(img2_surf, (int(img2_x * self.zoom), int(img2_y * self.zoom)))
            
        else:
            # Side-by-side mode
            # Draw image 1
            img1_surf = pygame.surfarray.make_surface(
                np.transpose(self.img1_rgb, (1, 0, 2))
            )
            if self.zoom != 1.0:
                img1_surf = pygame.transform.scale(
                    img1_surf,
                    (int(self.w1 * self.zoom), int(self.h1 * self.zoom))
                )
            canvas.blit(img1_surf, (int(img1_x * self.zoom), int(img1_y * self.zoom)))
            
            # Draw image 2
            img2_surf = pygame.surfarray.make_surface(
                np.transpose(self.img2_rgb, (1, 0, 2))
            )
            if self.zoom != 1.0:
                img2_surf = pygame.transform.scale(
                    img2_surf,
                    (int(self.w2 * self.zoom), int(self.h2 * self.zoom))
                )
            canvas.blit(img2_surf, (int(img2_x * self.zoom), int(img2_y * self.zoom)))
        
        # Draw crosshair at overlap boundary
        if self.show_crosshair:
            crosshair_x = int(self.img2_x * self.zoom)
            pygame.draw.line(canvas, self.COLOR_CROSSHAIR,
                           (crosshair_x, 0), (crosshair_x, canvas_h), 2)
            
            # Draw labels
            label1 = self.small_font.render("IMG1", True, self.COLOR_IMG1)
            label2 = self.small_font.render("IMG2", True, self.COLOR_IMG2)
            canvas.blit(label1, (max(5, crosshair_x - 60), 5))
            canvas.blit(label2, (min(canvas_w - 60, crosshair_x + 10), 5))
        
        # Calculate visible region
        visible_w = self.canvas_width
        visible_h = self.canvas_height
        
        # Clamp scroll to valid range
        max_scroll_x = max(0, canvas_w - visible_w)
        max_scroll_y = max(0, canvas_h - visible_h)
        self.scroll_x = max(0, min(max_scroll_x, self.scroll_x))
        self.scroll_y = max(0, min(max_scroll_y, self.scroll_y))
        
        # Blit visible portion to screen
        self.screen.blit(canvas, (self.canvas_x, self.canvas_y),
                        (self.scroll_x, self.scroll_y, visible_w, visible_h))
        
        # Draw border around canvas
        pygame.draw.rect(self.screen, (100, 100, 100),
                        (self.canvas_x - 2, self.canvas_y - 2,
                         visible_w + 4, visible_h + 4), 2)
    
    def render_info_panel(self, score, score_msg):
        """Render the information panel on the right"""
        panel_x = self.canvas_x + self.canvas_width + 20
        panel_y = 20
        panel_w = self.window_width - panel_x - 20
        
        y_pos = panel_y
        line_height = 35
        
        # Title
        title = self.font.render("ALIGNMENT INFO", True, self.COLOR_TEXT)
        self.screen.blit(title, (panel_x, y_pos))
        y_pos += line_height + 10
        
        # Score
        if score is not None:
            score_color = self.get_score_color(score)
            score_text = f"Score: {score:.6f}"
            score_surf = self.font.render(score_text, True, score_color)
            self.screen.blit(score_surf, (panel_x, y_pos))
            y_pos += line_height
            
            # Confidence
            confidence = self.get_confidence(score)
            conf_text = f"Conf: {confidence}"
            conf_surf = self.small_font.render(conf_text, True, score_color)
            self.screen.blit(conf_surf, (panel_x, y_pos))
            y_pos += line_height - 5
            
            # Add to history
            self.score_history.append(score)
            if len(self.score_history) > self.max_history:
                self.score_history.pop(0)
            
        else:
            error_text = f"Score: {score_msg}"
            error_surf = self.small_font.render(error_text, True, self.COLOR_SCORE_LOW)
            self.screen.blit(error_surf, (panel_x, y_pos))
            y_pos += line_height
        
        y_pos += 10
        
        # Position info
        pos_title = self.small_font.render("Position:", True, self.COLOR_TEXT)
        self.screen.blit(pos_title, (panel_x, y_pos))
        y_pos += 25
        
        x_text = f"  X: {self.img2_x} px"
        y_text = f"  Y: {self.img2_y} px"
        
        x_surf = self.small_font.render(x_text, True, (200, 200, 200))
        y_surf = self.small_font.render(y_text, True, (200, 200, 200))
        
        self.screen.blit(x_surf, (panel_x, y_pos))
        y_pos += 22
        self.screen.blit(y_surf, (panel_x, y_pos))
        y_pos += 32
        
        # Overlap info
        overlap = max(0, self.w1 - self.img2_x)
        overlap_pct = (overlap / self.w1 * 100) if self.w1 > 0 else 0
        
        overlap_text = f"Overlap: {overlap}px"
        pct_text = f"({overlap_pct:.1f}% of img1)"
        
        overlap_surf = self.small_font.render(overlap_text, True, (200, 200, 200))
        pct_surf = self.small_font.render(pct_text, True, (150, 150, 150))
        
        self.screen.blit(overlap_surf, (panel_x, y_pos))
        y_pos += 22
        self.screen.blit(pct_surf, (panel_x, y_pos))
        y_pos += 32
        
        # Zoom info
        zoom_text = f"Zoom: {self.zoom:.1f}x"
        zoom_surf = self.small_font.render(zoom_text, True, (200, 200, 200))
        self.screen.blit(zoom_surf, (panel_x, y_pos))
        y_pos += 32
        
        # Mode
        mode_text = f"Mode: {'Overlay' if self.overlay_mode else 'Side-by-side'}"
        mode_surf = self.small_font.render(mode_text, True, (200, 200, 200))
        self.screen.blit(mode_surf, (panel_x, y_pos))
        y_pos += 22
        
        if self.overlay_mode:
            trans_text = f"Alpha: {self.transparency}/255"
            trans_surf = self.small_font.render(trans_text, True, (150, 150, 150))
            self.screen.blit(trans_surf, (panel_x, y_pos))
            y_pos += 22
        
        y_pos += 20
        
        # Score graph
        if self.score_history and len(self.score_history) > 1:
            graph_title = self.small_font.render("Score History:", True, self.COLOR_TEXT)
            self.screen.blit(graph_title, (panel_x, y_pos))
            y_pos += 25
            
            graph_w = panel_w - 10
            graph_h = 80
            graph_x = panel_x
            graph_y = y_pos
            
            # Draw graph background
            pygame.draw.rect(self.screen, (40, 40, 40),
                           (graph_x, graph_y, graph_w, graph_h))
            
            # Draw grid lines
            for i in range(5):
                y = graph_y + i * (graph_h / 4)
                pygame.draw.line(self.screen, (60, 60, 60),
                               (graph_x, y), (graph_x + graph_w, y), 1)
            
            # Draw score line
            if len(self.score_history) >= 2:
                points = []
                for i, s in enumerate(self.score_history):
                    x = graph_x + (i / (self.max_history - 1)) * graph_w
                    # Map score (0.5 to 1.0) to graph height
                    normalized = max(0, min(1, (s - 0.5) / 0.5))
                    y = graph_y + graph_h - (normalized * graph_h)
                    points.append((x, y))
                
                if len(points) >= 2:
                    pygame.draw.lines(self.screen, self.COLOR_SCORE_HIGH, False, points, 2)
            
            # Draw reference lines
            # 0.97 line (HIGH threshold)
            high_y = graph_y + graph_h - ((0.97 - 0.5) / 0.5) * graph_h
            pygame.draw.line(self.screen, self.COLOR_SCORE_HIGH,
                           (graph_x, high_y), (graph_x + graph_w, high_y), 1)
            
            # 0.85 line (MEDIUM threshold)
            med_y = graph_y + graph_h - ((0.85 - 0.5) / 0.5) * graph_h
            pygame.draw.line(self.screen, self.COLOR_SCORE_MED,
                           (graph_x, med_y), (graph_x + graph_w, med_y), 1)
            
            y_pos += graph_h + 10
        
        # Controls at bottom
        y_pos = self.window_height - 220
        
        controls_title = self.small_font.render("Controls:", True, self.COLOR_TEXT)
        self.screen.blit(controls_title, (panel_x, y_pos))
        y_pos += 22
        
        controls = [
            "Arrows: Move",
            "W/S: Overlap",
            "Space: Overlay",
            "C: Crosshair",
            "Z/X: Alpha",
            "Wheel: Zoom",
            "MidBtn: Pan",
            "Tab: Fit",
            "R: Reset",
        ]
        
        for ctrl in controls:
            ctrl_surf = self.small_font.render(ctrl, True, (180, 180, 180))
            self.screen.blit(ctrl_surf, (panel_x, y_pos))
            y_pos += 20
    
    def handle_input(self):
        """Handle keyboard input"""
        keys = pygame.key.get_pressed()
        shift = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        step = 1 if shift else 5
        
        # Movement
        if keys[pygame.K_LEFT]:
            self.img2_x -= step
            pygame.time.wait(30)
        if keys[pygame.K_RIGHT]:
            self.img2_x += step
            pygame.time.wait(30)
        if keys[pygame.K_UP]:
            self.img2_y -= step
            pygame.time.wait(30)
        if keys[pygame.K_DOWN]:
            self.img2_y += step
            pygame.time.wait(30)
        
        # Overlap adjustment
        if keys[pygame.K_w]:
            self.img2_x += step  # Move right = more overlap
            pygame.time.wait(30)
        if keys[pygame.K_s]:
            self.img2_x -= step  # Move left = less overlap
            pygame.time.wait(30)
        
        # Transparency
        if keys[pygame.K_z]:
            self.transparency = max(0, self.transparency - 5)
            pygame.time.wait(30)
        if keys[pygame.K_x]:
            self.transparency = min(255, self.transparency + 5)
            pygame.time.wait(30)
    
    def run(self):
        """Main loop"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                        running = False
                    
                    elif event.key == pygame.K_SPACE:
                        self.overlay_mode = not self.overlay_mode
                    
                    elif event.key == pygame.K_c:
                        self.show_crosshair = not self.show_crosshair
                    
                    elif event.key == pygame.K_r:
                        # Reset position
                        self.img2_x = self.w1 - int(self.w1 * 0.5)
                        self.img2_y = 0
                        self.score_history.clear()
                    
                    elif event.key == pygame.K_TAB:
                        # Reset zoom to fit
                        self.calculate_fit_zoom()
                
                elif event.type == pygame.MOUSEWHEEL:
                    # Get mouse position relative to canvas
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    
                    # Check if mouse is over canvas
                    if (self.canvas_x <= mouse_x < self.canvas_x + self.canvas_width and
                        self.canvas_y <= mouse_y < self.canvas_y + self.canvas_height):
                        
                        # Calculate position in canvas coordinates (before zoom)
                        canvas_mouse_x = self.scroll_x + (mouse_x - self.canvas_x)
                        canvas_mouse_y = self.scroll_y + (mouse_y - self.canvas_y)
                        
                        # Calculate world position (at current zoom)
                        if self.zoom > 0:
                            world_x = canvas_mouse_x / self.zoom
                            world_y = canvas_mouse_y / self.zoom
                        else:
                            world_x = 0
                            world_y = 0
                        
                        # Update zoom
                        old_zoom = self.zoom
                        if event.y > 0:  # Scroll up - zoom in
                            self.zoom = min(self.max_zoom, self.zoom * (1 + self.zoom_speed))
                        else:  # Scroll down - zoom out
                            self.zoom = max(self.min_zoom, self.zoom * (1 - self.zoom_speed))
                        
                        # Calculate new canvas position of the same world point
                        new_canvas_mouse_x = world_x * self.zoom
                        new_canvas_mouse_y = world_y * self.zoom
                        
                        # Adjust scroll to keep the mouse point stationary
                        self.scroll_x = new_canvas_mouse_x - (mouse_x - self.canvas_x)
                        self.scroll_y = new_canvas_mouse_y - (mouse_y - self.canvas_y)
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 2:  # Middle mouse button
                        self.panning = True
                        self.pan_start_x = event.pos[0]
                        self.pan_start_y = event.pos[1]
                        self.pan_start_scroll_x = self.scroll_x
                        self.pan_start_scroll_y = self.scroll_y
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 2:  # Middle mouse button
                        self.panning = False
                
                elif event.type == pygame.MOUSEMOTION:
                    if self.panning:
                        # Pan the view
                        dx = self.pan_start_x - event.pos[0]
                        dy = self.pan_start_y - event.pos[1]
                        self.scroll_x = self.pan_start_scroll_x + dx
                        self.scroll_y = self.pan_start_scroll_y + dy
            
            # Handle continuous input
            self.handle_input()
            
            # Calculate score
            score, score_msg = self.calculate_score()
            
            # Render
            self.screen.fill(self.COLOR_BG)
            self.render_canvas()
            self.render_info_panel(score, score_msg)
            
            pygame.display.flip()
            clock.tick(60)  # 60 FPS
        
        pygame.quit()
        
        # Print final result
        if score is not None:
            print(f"\nFinal alignment:")
            print(f"  X position: {self.img2_x}")
            print(f"  Y offset: {self.img2_y}")
            print(f"  Overlap: {max(0, self.w1 - self.img2_x)} px")
            print(f"  Score: {score:.6f}")
            print(f"  Confidence: {self.get_confidence(score)}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Manual Image Alignment Tool with Real-Time Scoring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  Arrow Keys        Move Image 2 (X and Y position)
  Shift + Arrows    Fine movement (1px steps)
  W/S               Adjust overlap (move Image 2 left/right)
  Shift + W/S       Fine overlap adjustment (1px)
  Space             Toggle overlay/side-by-side mode
  C                 Toggle crosshair at overlap boundary
  Z/X               Adjust transparency (overlay mode)
  Mouse Wheel       Zoom in/out (centered on mouse position)
  Middle Mouse      Pan view (click and drag)
  Tab               Reset zoom to fit entire view
  R                 Reset to initial position
  Q/Escape          Quit
        """
    )
    
    parser.add_argument('image1', type=str, help='Path to first image (left)')
    parser.add_argument('image2', type=str, help='Path to second image (right)')
    parser.add_argument('--rotate-90', action='store_true',
                       help='Rotate images 90° counter-clockwise')
    
    args = parser.parse_args()
    
    img1_path = Path(args.image1)
    img2_path = Path(args.image2)
    
    if not img1_path.exists():
        print(f"Error: Image 1 not found: {img1_path}")
        return 1
    
    if not img2_path.exists():
        print(f"Error: Image 2 not found: {img2_path}")
        return 1
    
    try:
        tool = ManualAlignmentTool(img1_path, img2_path, args.rotate_90)
        tool.run()
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())