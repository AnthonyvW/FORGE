import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Dict

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
        kernel_size: int = 5,
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