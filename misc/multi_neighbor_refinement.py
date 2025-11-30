"""
Multi-Neighbor Refinement System

This module implements a second-pass refinement for poorly aligned image pairs
by using constraints from well-aligned neighbors.

Key insight: With 75% overlap, we can use multiple neighboring alignments
to constrain and improve problematic pairs.
"""

import numpy as np
from typing import List, Tuple, Optional
import cv2 as cv
from pathlib import Path


class AlignmentConstraints:
    """Manages alignment constraints from neighboring pairs"""
    
    def __init__(self, image_paths: List[Path], initial_offsets: List[Tuple]):
        """
        Args:
            image_paths: List of image file paths in order
            initial_offsets: List of (x_overlap, y_offset, score, confidence, flags) tuples
        """
        self.image_paths = image_paths
        self.initial_offsets = initial_offsets
        self.num_images = len(image_paths)
        
    def get_neighbor_constraints(self, pair_idx: int, window: int = 2) -> dict:
        """
        Get alignment constraints from neighboring pairs
        
        Args:
            pair_idx: Index of the pair to refine (0-based)
            window: Number of neighbors on each side to consider
            
        Returns:
            Dictionary with predicted overlap and Y-offset ranges
        """
        # Collect constraints from HIGH confidence neighbors
        neighbor_overlaps = []
        neighbor_y_offsets = []
        
        # Look at neighbors before this pair
        for i in range(max(0, pair_idx - window), pair_idx):
            offset_data = self.initial_offsets[i]
            confidence = offset_data[3] if len(offset_data) > 3 else "UNKNOWN"
            
            if confidence == "HIGH":
                neighbor_overlaps.append(offset_data[0])
                neighbor_y_offsets.append(offset_data[1])
        
        # Look at neighbors after this pair
        for i in range(pair_idx + 1, min(len(self.initial_offsets), pair_idx + window + 1)):
            offset_data = self.initial_offsets[i]
            confidence = offset_data[3] if len(offset_data) > 3 else "UNKNOWN"
            
            if confidence == "HIGH":
                neighbor_overlaps.append(offset_data[0])
                neighbor_y_offsets.append(offset_data[1])
        
        if not neighbor_overlaps:
            return None
        
        # Calculate tight constraints from neighbors
        mean_overlap = np.mean(neighbor_overlaps)
        std_overlap = np.std(neighbor_overlaps) if len(neighbor_overlaps) > 1 else 10
        
        mean_y = np.mean(neighbor_y_offsets)
        std_y = np.std(neighbor_y_offsets) if len(neighbor_y_offsets) > 1 else 2
        
        constraints = {
            'overlap_min': int(mean_overlap - 2 * std_overlap - 20),
            'overlap_max': int(mean_overlap + 2 * std_overlap + 20),
            'overlap_predicted': int(mean_overlap),
            'y_predicted': int(round(mean_y)),
            'y_range': max(4, int(2 * std_y + 3)),
            'num_neighbors': len(neighbor_overlaps),
            'neighbor_confidence': 'HIGH'
        }
        
        return constraints


def refine_pair_with_constraints(img1_path: Path, img2_path: Path,
                                 initial_alignment: Tuple,
                                 constraints: dict,
                                 rotate_images: bool = False) -> Optional[Tuple]:
    """
    Refine a poorly aligned pair using neighbor constraints
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        initial_alignment: (x_overlap, y_offset, score, confidence, flags)
        constraints: Constraints from neighbors
        rotate_images: Whether to rotate images
        
    Returns:
        Refined (x_overlap, y_offset, score, confidence, flags) or None
    """
    try:
        # Load images
        img1 = cv.imread(str(img1_path))
        img2 = cv.imread(str(img2_path))
        
        if img1 is None or img2 is None:
            return None
        
        if rotate_images:
            img1 = cv.rotate(img1, cv.ROTATE_90_COUNTERCLOCKWISE)
            img2 = cv.rotate(img2, cv.ROTATE_90_COUNTERCLOCKWISE)
        
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Use tighter search bounds from constraints
        overlap_min = constraints['overlap_min']
        overlap_max = constraints['overlap_max']
        overlap_predicted = constraints['overlap_predicted']
        y_predicted = constraints['y_predicted']
        y_range = constraints['y_range']
        
        # Extract evaluation regions
        max_expected_overlap = int(w1 * 0.95)
        img1_eval_width = min(max_expected_overlap, w1)
        img1_eval_start = w1 - img1_eval_width
        img1_eval_region = img1[:, img1_eval_start:]
        
        img2_eval_width = min(max_expected_overlap, w2)
        img2_eval_region = img2[:, :img2_eval_width]
        
        gray1_eval = cv.cvtColor(img1_eval_region, cv.COLOR_BGR2GRAY)
        gray2_eval = cv.cvtColor(img2_eval_region, cv.COLOR_BGR2GRAY)
        
        # CONSTRAINED SEARCH: Focus tightly around neighbor predictions
        best_score = -1
        best_overlap = overlap_predicted
        best_y = y_predicted
        
        # Stage 1: Tight coarse search around predicted values
        coarse_x_radius = 30  # ±30px from predicted overlap
        coarse_x_step = 5
        coarse_y_step = 2
        
        for x_overlap in range(overlap_predicted - coarse_x_radius, 
                              overlap_predicted + coarse_x_radius + 1, 
                              coarse_x_step):
            if x_overlap < overlap_min or x_overlap > overlap_max:
                continue
            if x_overlap > min(img1_eval_width, img2_eval_width):
                continue
                
            for y_offset in range(y_predicted - y_range, 
                                 y_predicted + y_range + 1, 
                                 coarse_y_step):
                # Calculate comparison regions
                if y_offset >= 0:
                    available_height1 = gray1_eval.shape[0] - y_offset
                    available_height2 = gray2_eval.shape[0]
                    compare_height = min(available_height1, available_height2)
                    
                    if compare_height <= 50:
                        continue
                    
                    region1 = gray1_eval[y_offset:y_offset + compare_height, -x_overlap:]
                    region2 = gray2_eval[:compare_height, :x_overlap]
                else:
                    available_height1 = gray1_eval.shape[0]
                    available_height2 = gray2_eval.shape[0] + y_offset
                    compare_height = min(available_height1, available_height2)
                    
                    if compare_height <= 50:
                        continue
                    
                    region1 = gray1_eval[:compare_height, -x_overlap:]
                    region2 = gray2_eval[-y_offset:-y_offset + compare_height, :x_overlap]
                
                if region1.shape != region2.shape:
                    continue
                
                score = cv.matchTemplate(region1, region2, cv.TM_CCOEFF_NORMED)[0, 0]
                
                if score > best_score:
                    best_score = score
                    best_overlap = x_overlap
                    best_y = y_offset
        
        # Stage 2: Fine refinement (smaller steps)
        fine_x_radius = 6
        fine_y_radius = 4
        fine_x_step = 2
        fine_y_step = 1
        
        final_score = best_score
        final_overlap = best_overlap
        final_y = best_y
        
        for x_overlap in range(best_overlap - fine_x_radius,
                              best_overlap + fine_x_radius + 1,
                              fine_x_step):
            if x_overlap < overlap_min or x_overlap > overlap_max:
                continue
            if x_overlap > min(img1_eval_width, img2_eval_width):
                continue
                
            for y_offset in range(best_y - fine_y_radius,
                                 best_y + fine_y_radius + 1,
                                 fine_y_step):
                if abs(y_offset - y_predicted) > y_range:
                    continue
                
                # Calculate comparison regions
                if y_offset >= 0:
                    available_height1 = gray1_eval.shape[0] - y_offset
                    available_height2 = gray2_eval.shape[0]
                    compare_height = min(available_height1, available_height2)
                    
                    if compare_height <= 50:
                        continue
                    
                    region1 = gray1_eval[y_offset:y_offset + compare_height, -x_overlap:]
                    region2 = gray2_eval[:compare_height, :x_overlap]
                else:
                    available_height1 = gray1_eval.shape[0]
                    available_height2 = gray2_eval.shape[0] + y_offset
                    compare_height = min(available_height1, available_height2)
                    
                    if compare_height <= 50:
                        continue
                    
                    region1 = gray1_eval[:compare_height, -x_overlap:]
                    region2 = gray2_eval[-y_offset:-y_offset + compare_height, :x_overlap]
                
                if region1.shape != region2.shape:
                    continue
                
                score = cv.matchTemplate(region1, region2, cv.TM_CCOEFF_NORMED)[0, 0]
                
                if score > final_score:
                    final_score = score
                    final_overlap = x_overlap
                    final_y = y_offset
        
        # Determine if refinement improved alignment
        initial_score = initial_alignment[2]
        score_improvement = final_score - initial_score
        
        # Upgrade confidence if score improved significantly
        if final_score > 0.99 and score_improvement > 0.005:
            new_confidence = "HIGH"
        elif final_score > 0.97:
            new_confidence = "MEDIUM"
        else:
            new_confidence = "LOW"
        
        return (final_overlap, final_y, final_score, new_confidence, 
                [f"REFINED(+{score_improvement:.4f})"])
        
    except Exception as e:
        print(f"      Refinement error: {e}")
        return None


def multi_neighbor_refinement_pass(image_paths: List[Path], 
                                   initial_offsets: List[Tuple],
                                   axis: str = 'y',
                                   confidence_threshold: str = "MEDIUM") -> List[Tuple]:
    """
    Second pass: Refine poorly aligned pairs using neighbor constraints
    
    Args:
        image_paths: List of image paths in order
        initial_offsets: List of (x_overlap, y_offset, score, confidence, flags) tuples
        axis: 'x' or 'y'
        confidence_threshold: Refine pairs with this confidence or lower
        
    Returns:
        Refined offsets list
    """
    print("\n" + "=" * 80)
    print("REFINEMENT PASS: Multi-Neighbor Constrained Search")
    print("=" * 80)
    
    constraints_manager = AlignmentConstraints(image_paths, initial_offsets)
    refined_offsets = list(initial_offsets)  # Copy
    
    # Identify pairs needing refinement
    pairs_to_refine = []
    for i, offset_data in enumerate(initial_offsets):
        confidence = offset_data[3] if len(offset_data) > 3 else "UNKNOWN"
        
        # Refine MEDIUM and LOW confidence pairs
        if confidence in [confidence_threshold, "LOW"]:
            pairs_to_refine.append(i)
    
    if not pairs_to_refine:
        print(f"No pairs with {confidence_threshold} or lower confidence found. Skipping refinement.")
        return refined_offsets
    
    print(f"Found {len(pairs_to_refine)} pairs to refine: {pairs_to_refine}")
    
    # Refine each problematic pair
    refinement_stats = {'improved': 0, 'degraded': 0, 'unchanged': 0}
    
    for pair_idx in pairs_to_refine:
        img1_path = image_paths[pair_idx]
        img2_path = image_paths[pair_idx + 1]
        initial_alignment = initial_offsets[pair_idx]
        
        print(f"\nRefining pair {pair_idx+1}: {img1_path.name} + {img2_path.name}")
        print(f"  Initial: overlap={initial_alignment[0]}, y={initial_alignment[1]}, score={initial_alignment[2]:.4f}, conf={initial_alignment[3]}")
        
        # Get constraints from neighbors
        constraints = constraints_manager.get_neighbor_constraints(pair_idx, window=3)
        
        if constraints is None:
            print(f"  ⚠ No HIGH confidence neighbors found, skipping")
            continue
        
        print(f"  Constraints from {constraints['num_neighbors']} neighbors:")
        print(f"    Predicted overlap: {constraints['overlap_predicted']} (±{constraints['overlap_max'] - constraints['overlap_predicted']})")
        print(f"    Predicted Y: {constraints['y_predicted']} (±{constraints['y_range']})")
        
        # Refine the alignment
        refined = refine_pair_with_constraints(
            img1_path, img2_path,
            initial_alignment,
            constraints,
            rotate_images=(axis == 'y')
        )
        
        if refined:
            score_change = refined[2] - initial_alignment[2]
            overlap_change = refined[0] - initial_alignment[0]
            y_change = refined[1] - initial_alignment[1]
            
            print(f"  Refined: overlap={refined[0]}, y={refined[1]}, score={refined[2]:.4f}, conf={refined[3]}")
            print(f"  Changes: Δoverlap={overlap_change:+d}px, Δy={y_change:+d}px, Δscore={score_change:+.4f}")
            
            # Update if improved
            if score_change > 0.001:  # At least 0.1% improvement
                refined_offsets[pair_idx] = refined
                refinement_stats['improved'] += 1
                print(f"  ✓ Refinement accepted (score improved)")
            elif score_change < -0.001:
                refinement_stats['degraded'] += 1
                print(f"  ✗ Refinement rejected (score degraded)")
            else:
                refinement_stats['unchanged'] += 1
                print(f"  ≈ No significant change")
        else:
            print(f"  ✗ Refinement failed")
    
    # Summary
    print("\n" + "=" * 80)
    print("REFINEMENT SUMMARY")
    print("=" * 80)
    print(f"Pairs refined: {len(pairs_to_refine)}")
    print(f"  Improved: {refinement_stats['improved']}")
    print(f"  Unchanged: {refinement_stats['unchanged']}")
    print(f"  Degraded (rejected): {refinement_stats['degraded']}")
    
    # Count confidence levels after refinement
    high_count = sum(1 for o in refined_offsets if o[3] == "HIGH")
    med_count = sum(1 for o in refined_offsets if o[3] == "MEDIUM")
    low_count = sum(1 for o in refined_offsets if o[3] == "LOW")
    
    print(f"\nFinal confidence distribution:")
    print(f"  HIGH: {high_count}/{len(refined_offsets)}")
    print(f"  MEDIUM: {med_count}/{len(refined_offsets)}")
    print(f"  LOW: {low_count}/{len(refined_offsets)}")
    
    return refined_offsets
