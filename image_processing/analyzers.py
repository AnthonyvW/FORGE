import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class FocusAnalysisResult:
    """Results from focus analysis."""
    focus_score: float
    quadrant_scores: Dict[str, float]
    best_quadrant: Tuple[str, float]

class ImageAnalyzer:
    """Class for analyzing images."""
    
    @staticmethod
    def is_black(image: np.ndarray, threshold: float = 5.0) -> bool:
        """
        Determine if an image is effectively black.
        
        Args:
            image: numpy array of the image
            threshold: standard deviation threshold for considering a channel black
            
        Returns:
            bool: True if image is considered black
        """
        return any(np.std(image[:,:,i]) < threshold for i in range(3))

    @staticmethod
    def analyze_focus(
        image: np.ndarray,
        kernel_size: int = 7,
        threshold: float = 100
    ) -> FocusAnalysisResult:
        """
        Analyze focus quality across image quadrants.
        
        Args:
            image: numpy array of the image
            kernel_size: Size of the Laplacian kernel
            threshold: Threshold for determining if quadrant is in focus
            
        Returns:
            FocusAnalysisResult containing analysis details
        """
        height, width = image.shape[:2]
        mid_h, mid_w = height // 2, width // 2
        
        # Define quadrants
        quadrants = {
            'Top Left': image[0:mid_h, 0:mid_w],
            'Top Right': image[0:mid_h, mid_w:],
            'Bottom Left': image[mid_h:, 0:mid_w],
            'Bottom Right': image[mid_h:, mid_w:]
        }
        
        quadrant_scores = {}
        
        for name, quad in quadrants.items():
            # Convert to grayscale if needed
            if len(quad.shape) == 3:
                quad = cv2.cvtColor(quad, cv2.COLOR_BGR2GRAY)
                
            blurred = cv2.GaussianBlur(quad, (3, 3), 0)
            laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)
            abs_laplacian = np.absolute(laplacian)
            
            # Calculate focus metrics
            variance = np.var(abs_laplacian)
            percentile_90 = np.percentile(abs_laplacian, 90)
            focus_score = (variance + percentile_90) / 2
            
            quadrant_scores[name] = focus_score
        
        best_quadrant = max(quadrant_scores.items(), key=lambda x: x[1])
        overall_score = sum(quadrant_scores.values()) / len(quadrant_scores)
        
        return FocusAnalysisResult(
            focus_score=overall_score,
            quadrant_scores=quadrant_scores,
            best_quadrant=best_quadrant
        )
    

@dataclass
class FocusTile:
    x: int
    y: int
    w: int
    h: int
    score: float
    band: str = "hard"  # "hard" or "soft"

def find_focused_areas(
    image: np.ndarray,
    tile_size: int = 48,
    stride: int = 48,
    laplacian_ksize: int = 3,
    blur_ksize: int = 3,
    top_percent: float = 0.15,
    min_score: Optional[float] = None,       # absolute threshold (hard)
    soft_min_score: Optional[float] = None,  # soft band lower bound
) -> List[FocusTile]:
    """
    Return rectangles where the image is relatively 'in focus' using a Laplacian-based score.

    Selection priority:
      1) If min_score (and optionally soft_min_score) provided:
         - 'hard' band: score >= min_score
         - 'soft' band: soft_min_score <= score < min_score (if soft_min_score is not None)
      2) Else: keep top `top_percent` by score (band='hard').

    Notes:
      - If soft_min_score > min_score, it will be clamped down to min_score.
    """
    if image is None or image.size == 0:
        return []

    # grayscale + denoise
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if blur_ksize and blur_ksize > 0:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    H, W = gray.shape[:2]
    tiles: List[FocusTile] = []
    for y in range(0, max(1, H - tile_size + 1), max(1, stride)):
        for x in range(0, max(1, W - tile_size + 1), max(1, stride)):
            roi = gray[y:y + tile_size, x:x + tile_size]
            lap = cv2.Laplacian(roi, cv2.CV_64F, ksize=laplacian_ksize)
            abs_lap = np.abs(lap)
            variance = float(np.var(abs_lap))
            pct90 = float(np.percentile(abs_lap, 90))
            score = 0.5 * (variance + pct90)
            tiles.append(FocusTile(x=x, y=y, w=tile_size, h=tile_size, score=score))

    if not tiles:
        return []

    # --- Selection logic ---
    if min_score is not None:
        # clamp soft_min_score if provided
        if soft_min_score is not None and soft_min_score > min_score:
            soft_min_score = min_score

        selected: List[FocusTile] = []
        for t in tiles:
            if t.score >= min_score:
                t.band = "hard"
                selected.append(t)
            elif soft_min_score is not None and t.score >= soft_min_score:
                t.band = "soft"
                selected.append(t)

        # Sort so highest scores draw last (on top)
        selected.sort(key=lambda t: t.score, reverse=True)
        return selected

    # Fallback: top-percent mode
    tiles.sort(key=lambda t: t.score, reverse=True)
    keep = max(1, int(round(len(tiles) * top_percent)))
    for i in range(keep):
        tiles[i].band = "hard"
    return tiles[:keep]