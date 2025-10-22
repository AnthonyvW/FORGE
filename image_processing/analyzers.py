import numpy as np
import math
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
        edge_left_pct: float = 0.0,
        edge_right_pct: float = 0.0,
        edge_top_pct: float = 0.0,
        edge_bottom_pct: float = 0.0,
        scale_factor: float = 1.0,
    ) -> FocusAnalysisResult:
        """
        Analyze focus quality across image quadrants, ignoring edges by percentage.

        Args:
            image: numpy array of the image
            kernel_size: Size of the Laplacian kernel
            edge_left_pct: Fraction (0–1) of width to ignore from left edge
            edge_right_pct: Fraction (0–1) of width to ignore from right edge
            edge_top_pct: Fraction (0–1) of height to ignore from top edge
            edge_bottom_pct: Fraction (0–1) of height to ignore from bottom edge
            scale_factor: optional down/up scale before scoring (<=1.0 recommended)

        Returns:
            FocusAnalysisResult containing analysis details
        """
        height, width = image.shape[:2]

        # --- Crop edges based on percentages ---
        x_start = int(width * edge_left_pct)
        x_end = int(width * (1.0 - edge_right_pct))
        y_start = int(height * edge_top_pct)
        y_end = int(height * (1.0 - edge_bottom_pct))

        # Ensure valid crop
        if x_end <= x_start or y_end <= y_start:
            raise ValueError("Invalid edge crop percentages — resulting region is empty.")

        cropped = image[y_start:y_end, x_start:x_end]

        sf = float(scale_factor) if scale_factor is not None else 1.0
        if sf <= 0:
            sf = 1.0
        if not math.isclose(sf, 1.0):
            interp = cv2.INTER_AREA if sf < 1.0 else cv2.INTER_LINEAR
            cropped = cv2.resize(cropped, dsize=None, fx=sf, fy=sf, interpolation=interp)

        height, width = cropped.shape[:2]
        mid_h, mid_w = height // 2, width // 2

        # Define quadrants within the cropped region
        quadrants = {
            'Top Left': cropped[0:mid_h, 0:mid_w],
            'Top Right': cropped[0:mid_h, mid_w:],
            'Bottom Left': cropped[mid_h:, 0:mid_w],
            'Bottom Right': cropped[mid_h:, mid_w:]
        }

        quadrant_scores = {}
        for name, quad in quadrants.items():
            if len(quad.shape) == 3:
                quad = cv2.cvtColor(quad, cv2.COLOR_BGR2GRAY)

            blurred = cv2.GaussianBlur(quad, (3, 3), 0)
            laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)
            abs_laplacian = np.absolute(laplacian)

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
    min_score: Optional[float] = None,
    soft_min_score: Optional[float] = None,
    *,
    scale_factor: float = 1.0,
) -> List[FocusTile]:
    """
    Return rectangles where the image is relatively 'in focus' using a Laplacian-based score.
    If scale_factor != 1.0, the image is analyzed at that scale, but returned tile rects
    are mapped back to ORIGINAL image coordinates.
    """
    if image is None or image.size == 0:
        return []

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if blur_ksize and blur_ksize > 0:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    sf = float(scale_factor) if scale_factor is not None else 1.0
    if sf <= 0:
        sf = 1.0

    # Work image and local tile geometry (in the scaled space)
    if not math.isclose(sf, 1.0):
        interp = cv2.INTER_AREA if sf < 1.0 else cv2.INTER_LINEAR
        gray_work = cv2.resize(gray, dsize=None, fx=sf, fy=sf, interpolation=interp)
        local_tile = max(1, int(round(tile_size * sf)))
        local_stride = max(1, int(round(stride * sf)))
        inv_sf = 1.0 / sf
    else:
        gray_work = gray
        local_tile = int(tile_size)
        local_stride = int(stride)
        inv_sf = 1.0

    H, W = gray_work.shape[:2]
    tiles: List[FocusTile] = []
    for y in range(0, max(1, H - local_tile + 1), max(1, local_stride)):
        for x in range(0, max(1, W - local_tile + 1), max(1, local_stride)):
            roi = gray_work[y:y + local_tile, x:x + local_tile]
            lap = cv2.Laplacian(roi, cv2.CV_64F, ksize=laplacian_ksize)
            abs_lap = np.abs(lap)
            variance = float(np.var(abs_lap))
            pct90 = float(np.percentile(abs_lap, 90))
            score = 0.5 * (variance + pct90)

            # Map rect back to ORIGINAL coords
            tiles.append(FocusTile(
                x=int(round(x * inv_sf)),
                y=int(round(y * inv_sf)),
                w=int(round(local_tile * inv_sf)),
                h=int(round(local_tile * inv_sf)),
                score=score
            ))

    if not tiles:
        return []

    if min_score is not None:
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
        selected.sort(key=lambda t: t.score, reverse=True)
        return selected

    tiles.sort(key=lambda t: t.score, reverse=True)
    keep = max(1, int(round(len(tiles) * top_percent)))
    for i in range(keep):
        tiles[i].band = "hard"
    return tiles[:keep]