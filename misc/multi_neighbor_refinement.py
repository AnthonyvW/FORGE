"""
Multi-Neighbor Refinement System with Gap Awareness

Drop-in replacement that detects position gaps (e.g., Y187→Y189)
and adapts search strategy accordingly.

This module exports multi_neighbor_refinement_pass() to maintain
compatibility with existing stitching scripts.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import cv2 as cv
from pathlib import Path
import re


def extract_coordinate(filename: str, axis: str) -> Optional[int]:
    """Extract coordinate value from filename"""
    coord_letter = axis.upper()
    match = re.search(rf'{coord_letter}(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


class GapAwareConstraints:
    """Manages alignment constraints with gap detection"""
    
    def __init__(self, image_paths: List[Path], initial_offsets: List[Tuple], axis: str = 'y'):
        self.image_paths = image_paths
        self.initial_offsets = initial_offsets
        self.num_images = len(image_paths)
        self.axis = axis
        
        # Extract coordinates and detect gaps
        self.coordinates = []
        self.gaps = []
        
        for i, path in enumerate(image_paths):
            coord = extract_coordinate(path.stem, axis)
            self.coordinates.append(coord)
        
        # Calculate gaps: gaps[i] = coordinate[i+1] - coordinate[i]
        for i in range(len(self.coordinates) - 1):
            if self.coordinates[i] is not None and self.coordinates[i+1] is not None:
                gap = self.coordinates[i+1] - self.coordinates[i]
                self.gaps.append(gap)
            else:
                self.gaps.append(1)  # Default gap
        
        # Calculate typical gap
        valid_gaps = [g for g in self.gaps if g is not None and g > 0]
        if valid_gaps:
            self.typical_gap = int(np.median(valid_gaps))
        else:
            self.typical_gap = 1
            
        print(f"\n  Gap Analysis: Typical gap = {self.typical_gap}")
        
    def detect_position_gap(self, pair_idx: int) -> Dict:
        """Detect if there's a position gap at this pair"""
        if pair_idx >= len(self.gaps):
            return {'has_gap': False, 'gap_size': 1, 'gap_type': 'normal', 'gap_multiplier': 1.0}
        
        gap = self.gaps[pair_idx]
        
        if gap is None:
            return {'has_gap': False, 'gap_size': 1, 'gap_type': 'normal', 'gap_multiplier': 1.0}
        
        # Compare to typical gap
        gap_multiplier = gap / self.typical_gap if self.typical_gap > 0 else 1.0
        
        gap_info = {
            'has_gap': abs(gap - self.typical_gap) > 0,
            'gap_size': gap,
            'typical_gap': self.typical_gap,
            'gap_multiplier': gap_multiplier,
            'gap_type': 'skip' if gap > self.typical_gap else ('overlap' if gap < self.typical_gap else 'normal')
        }
        
        return gap_info
    
    def get_neighbor_constraints(self, pair_idx: int, window: int = 3) -> Optional[Dict]:
        """Get alignment constraints from neighboring pairs, accounting for gaps"""
        
        # Detect gap at this position
        gap_info = self.detect_position_gap(pair_idx)
        
        # Collect constraints from HIGH confidence neighbors
        neighbor_overlaps = []
        neighbor_y_offsets = []
        neighbor_gap_multipliers = []
        
        # Look at neighbors
        for i in range(max(0, pair_idx - window), min(len(self.initial_offsets), pair_idx + window + 1)):
            if i == pair_idx:
                continue
                
            offset_data = self.initial_offsets[i]
            confidence = offset_data[3] if len(offset_data) > 3 else "UNKNOWN"
            
            # Only use HIGH confidence pairs
            if confidence == "HIGH":
                neighbor_gap_info = self.detect_position_gap(i)
                neighbor_overlaps.append(offset_data[0])
                neighbor_y_offsets.append(offset_data[1])
                neighbor_gap_multipliers.append(neighbor_gap_info['gap_multiplier'])
        
        if not neighbor_overlaps:
            return None
        
        # Calculate base statistics from neighbors
        mean_overlap = np.mean(neighbor_overlaps)
        std_overlap = np.std(neighbor_overlaps) if len(neighbor_overlaps) > 1 else 25
        
        mean_y = np.mean(neighbor_y_offsets)
        std_y = np.std(neighbor_y_offsets) if len(neighbor_y_offsets) > 1 else 3
        
        # CRITICAL: Adjust for gap size using EMPIRICAL scaling
        if gap_info['has_gap'] and gap_info['gap_multiplier'] != 1.0:
            gap_mult = gap_info['gap_multiplier']
            
            # EMPIRICAL OBSERVATION from actual data:
            # Gap=1: overlap ≈ 1455px (100%)
            # Gap=2: overlap ≈ 862px  (59.2%)
            # 
            # The scanning system doesn't move exactly 2x distance for gap=2
            # Use empirically-derived scaling factors:
            if gap_mult >= 2.0:
                # For gap=2+: use ~59% of normal overlap (matches observed 862px)
                overlap_scale_factor = 0.59
            elif gap_mult >= 1.5:
                # For gap=1.5: interpolate to ~75%
                overlap_scale_factor = 0.75
            elif gap_mult > 1.0:
                # For smaller gaps: linear interpolation
                # gap=1.1 → 95%, gap=1.3 → 85%, etc.
                overlap_scale_factor = 1.0 - (gap_mult - 1.0) * 0.5
            else:
                # Unexpected case - images closer than normal
                overlap_scale_factor = 1.0 + (1.0 - gap_mult)
            
            predicted_overlap = mean_overlap * overlap_scale_factor
            # Wider uncertainty due to the gap
            overlap_uncertainty = max(std_overlap * 1.5, 50)
        else:
            # Normal gap - use statistics directly
            predicted_overlap = mean_overlap
            overlap_uncertainty = std_overlap
        
        # Build constraints
        constraints = {
            'overlap_min': max(100, int(predicted_overlap - 2.5 * overlap_uncertainty - 40)),
            'overlap_max': int(predicted_overlap + 2.5 * overlap_uncertainty + 40),
            'overlap_predicted': int(predicted_overlap),
            'y_predicted': int(round(mean_y)),
            'y_range': max(8, int(2.5 * std_y + 6)),  # Wider for gap cases
            'num_neighbors': len(neighbor_overlaps),
            'neighbor_confidence': 'HIGH',
            'gap_info': gap_info,
            'gap_adjusted': gap_info['has_gap']
        }
        
        return constraints


def refine_pair_with_gap_awareness(img1_path: Path, img2_path: Path,
                                   initial_alignment: Tuple,
                                   constraints: Optional[Dict],
                                   rotate_images: bool = False) -> Optional[Tuple]:
    """Refine a poorly aligned pair using gap-aware neighbor constraints"""
    
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
        
        # Determine search bounds
        if constraints:
            overlap_min = constraints['overlap_min']
            overlap_max = constraints['overlap_max']
            overlap_predicted = constraints['overlap_predicted']
            has_gap = constraints['gap_info']['has_gap']
            gap_mult = constraints['gap_info']['gap_multiplier']
            
            # CRITICAL FIX: For gap pairs with large initial Y-offset,
            # use the initial Y as the center (it's likely correct!)
            # instead of forcing y=0 from neighbors
            initial_y = initial_alignment[1]
            neighbor_y = constraints['y_predicted']
            
            if has_gap and abs(initial_y) > 5:
                # Large Y-offset detected - trust the initial alignment more than neighbors
                y_predicted = initial_y
                y_range = max(12, constraints['y_range'] + 5)  # Wider range for gap cases
                print(f"      Using initial Y={initial_y} as center (gap pair with large offset)")
            else:
                # Normal case - use neighbor prediction
                y_predicted = neighbor_y
                y_range = constraints['y_range']
        else:
            # No constraints - wide search around initial result
            initial_overlap = initial_alignment[0]
            overlap_predicted = initial_overlap
            overlap_min = int(initial_overlap * 0.5)
            overlap_max = int(initial_overlap * 1.5)
            y_predicted = initial_alignment[1]
            y_range = 20
            has_gap = True
            gap_mult = 1.0
        
        # Extract evaluation regions
        max_expected_overlap = int(w1 * 0.95)
        img1_eval_width = min(max_expected_overlap, w1)
        img1_eval_start = w1 - img1_eval_width
        img1_eval_region = img1[:, img1_eval_start:]
        
        img2_eval_width = min(max_expected_overlap, w2)
        img2_eval_region = img2[:, :img2_eval_width]
        
        gray1_eval = cv.cvtColor(img1_eval_region, cv.COLOR_BGR2GRAY)
        gray2_eval = cv.cvtColor(img2_eval_region, cv.COLOR_BGR2GRAY)
        
        # ADAPTIVE SEARCH based on gap
        if has_gap and gap_mult > 1.2:
            # Significant gap - very wide search
            coarse_x_radius = 120
            coarse_x_step = 10
            coarse_y_step = 4
            fine_x_radius = 15
            fine_y_radius = 8
        elif has_gap:
            # Minor gap - moderately wide search  
            coarse_x_radius = 70
            coarse_x_step = 7
            coarse_y_step = 3
            fine_x_radius = 10
            fine_y_radius = 6
        else:
            # Normal - tight search
            coarse_x_radius = 40
            coarse_x_step = 5
            coarse_y_step = 2
            fine_x_radius = 8
            fine_y_radius = 4
        
        # Stage 1: Coarse search
        best_score = -1
        best_overlap = overlap_predicted
        best_y = y_predicted
        
        search_count = 0
        
        for x_overlap in range(overlap_predicted - coarse_x_radius, 
                              overlap_predicted + coarse_x_radius + 1, 
                              coarse_x_step):
            if x_overlap < overlap_min or x_overlap > overlap_max:
                continue
            if x_overlap > min(img1_eval_width, img2_eval_width) or x_overlap < 50:
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
                search_count += 1
                
                if score > best_score:
                    best_score = score
                    best_overlap = x_overlap
                    best_y = y_offset
        
        # Stage 2: Fine refinement
        final_score = best_score
        final_overlap = best_overlap
        final_y = best_y
        
        for x_overlap in range(best_overlap - fine_x_radius,
                              best_overlap + fine_x_radius + 1,
                              2):
            if x_overlap < overlap_min or x_overlap > overlap_max:
                continue
            if x_overlap > min(img1_eval_width, img2_eval_width) or x_overlap < 50:
                continue
                
            for y_offset in range(best_y - fine_y_radius,
                                 best_y + fine_y_radius + 1,
                                 1):
                if abs(y_offset - y_predicted) > y_range + 2:
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
                search_count += 1
                
                if score > final_score:
                    final_score = score
                    final_overlap = x_overlap
                    final_y = y_offset
        
        print(f"      Search iterations: {search_count}")
        
        # Determine confidence
        initial_score = initial_alignment[2]
        score_improvement = final_score - initial_score
        
        if final_score > 0.99:
            new_confidence = "HIGH"
        elif final_score > 0.97:
            new_confidence = "MEDIUM"
        else:
            new_confidence = "LOW"
        
        # Build flags
        flags = []
        if score_improvement > 0.001:
            flags.append(f"REFINED(+{score_improvement:.4f})")
        elif score_improvement < -0.001:
            flags.append(f"REFINED({score_improvement:.4f})")
        
        if constraints and constraints['gap_adjusted']:
            gap_info = constraints['gap_info']
            if gap_info['gap_type'] == 'skip':
                flags.append(f"GAP_SKIP(×{gap_info['gap_multiplier']:.1f})")
        
        return (final_overlap, final_y, final_score, new_confidence, flags)
        
    except Exception as e:
        print(f"      Refinement error: {e}")
        import traceback
        traceback.print_exc()
        return None


def multi_neighbor_refinement_pass(image_paths: List[Path], 
                                   initial_offsets: List[Tuple],
                                   axis: str = 'y',
                                   confidence_threshold: str = "MEDIUM") -> List[Tuple]:
    """
    Gap-aware refinement pass for poorly aligned pairs
    
    This is the main exported function that maintains compatibility
    with existing stitching scripts.
    """
    print("\n" + "=" * 80)
    print("GAP-AWARE REFINEMENT PASS")
    print("=" * 80)
    
    constraints_manager = GapAwareConstraints(image_paths, initial_offsets, axis)
    refined_offsets = list(initial_offsets)
    
    # Print gap summary
    print("\nPosition Gap Summary:")
    gap_count = 0
    for i, gap in enumerate(constraints_manager.gaps):
        if i < len(image_paths) - 1 and gap != constraints_manager.typical_gap:
            coord1 = constraints_manager.coordinates[i]
            coord2 = constraints_manager.coordinates[i + 1]
            print(f"  ⚠️  Pair {i+1}: {axis.upper()}{coord1} → {axis.upper()}{coord2} (gap: {gap}, typical: {constraints_manager.typical_gap})")
            gap_count += 1
    
    if gap_count == 0:
        print(f"  ✓ No position gaps detected (all gaps = {constraints_manager.typical_gap})")
    else:
        print(f"  Found {gap_count} position gap(s)")
    
    # Identify pairs needing refinement
    pairs_to_refine = []
    for i, offset_data in enumerate(initial_offsets):
        confidence = offset_data[3] if len(offset_data) > 3 else "UNKNOWN"
        
        if confidence in [confidence_threshold, "LOW"]:
            pairs_to_refine.append(i)
    
    if not pairs_to_refine:
        print(f"\nNo pairs with {confidence_threshold} or lower confidence. Skipping refinement.")
        return refined_offsets
    
    print(f"\nRefining {len(pairs_to_refine)} pair(s) with {confidence_threshold}/LOW confidence: {[i+1 for i in pairs_to_refine]}")
    
    # Refine each problematic pair
    stats = {'improved': 0, 'degraded': 0, 'unchanged': 0, 'gap_cases': 0}
    
    for pair_idx in pairs_to_refine:
        img1_path = image_paths[pair_idx]
        img2_path = image_paths[pair_idx + 1]
        initial_alignment = initial_offsets[pair_idx]
        
        print(f"\nRefining pair {pair_idx+1}: {img1_path.name} + {img2_path.name}")
        print(f"  Initial: overlap={initial_alignment[0]}, y={initial_alignment[1]}, score={initial_alignment[2]:.4f}, conf={initial_alignment[3]}")
        
        # Get gap-aware constraints
        constraints = constraints_manager.get_neighbor_constraints(pair_idx, window=4)
        
        if constraints:
            gap_info = constraints['gap_info']
            
            if gap_info['has_gap']:
                stats['gap_cases'] += 1
                print(f"  Gap detected: {gap_info['gap_type']} (×{gap_info['gap_multiplier']:.2f})")
            
            print(f"  Constraints from {constraints['num_neighbors']} HIGH-conf neighbors:")
            print(f"    Overlap: {constraints['overlap_predicted']} ({constraints['overlap_min']} to {constraints['overlap_max']})")
            print(f"    Y-offset: {constraints['y_predicted']} ± {constraints['y_range']}")
        else:
            print(f"  ⚠️  No HIGH-confidence neighbors - using wide search")
        
        # Refine
        refined = refine_pair_with_gap_awareness(
            img1_path, img2_path,
            initial_alignment,
            constraints,
            rotate_images=(axis == 'y')
        )
        
        if refined:
            score_change = refined[2] - initial_alignment[2]
            overlap_change = refined[0] - initial_alignment[0]
            y_change = refined[1] - initial_alignment[1]
            
            print(f"  Result: overlap={refined[0]}, y={refined[1]}, score={refined[2]:.4f}, conf={refined[3]}")
            print(f"  Changes: Δoverlap={overlap_change:+d}px, Δy={y_change:+d}px, Δscore={score_change:+.4f}")
            
            # Acceptance criteria
            accept = False
            reason = ""
            
            if score_change > 0.002:
                accept = True
                reason = "score improved"
            elif constraints and constraints['gap_info']['has_gap']:
                # More lenient criteria for gap cases
                # Especially if there's a large Y-offset (physical misalignment)
                has_large_y = abs(refined[1]) > 5 or abs(initial_alignment[1]) > 5
                
                if refined[2] > 0.85 and score_change > -0.05:
                    accept = True
                    if has_large_y:
                        reason = "gap case with physical misalignment (y-shift)"
                    else:
                        reason = "gap case with acceptable score"
                elif refined[2] > 0.80 and score_change > -0.03 and has_large_y:
                    # Very lenient for gap cases with y-shift
                    accept = True
                    reason = "gap case with large y-shift (best possible)"
                else:
                    reason = "gap case but score too low"
            elif score_change < -0.005:
                reason = "score degraded significantly"
            else:
                reason = "no significant change"
            
            if accept:
                refined_offsets[pair_idx] = refined
                stats['improved'] += 1
                print(f"  ✓ Accepted ({reason})")
            else:
                if score_change < -0.001:
                    stats['degraded'] += 1
                else:
                    stats['unchanged'] += 1
                print(f"  ✗ Rejected ({reason})")
        else:
            print(f"  ✗ Refinement failed")
    
    # Summary
    print("\n" + "=" * 80)
    print("REFINEMENT SUMMARY")
    print("=" * 80)
    print(f"Pairs processed: {len(pairs_to_refine)}")
    print(f"  Position gaps: {stats['gap_cases']}")
    print(f"  Improved: {stats['improved']}")
    print(f"  Unchanged: {stats['unchanged']}")
    print(f"  Degraded (rejected): {stats['degraded']}")
    
    # Final confidence distribution
    high = sum(1 for o in refined_offsets if o[3] == "HIGH")
    med = sum(1 for o in refined_offsets if o[3] == "MEDIUM")
    low = sum(1 for o in refined_offsets if o[3] == "LOW")
    
    print(f"\nFinal confidence distribution:")
    print(f"  HIGH: {high}/{len(refined_offsets)}")
    print(f"  MEDIUM: {med}/{len(refined_offsets)}")
    print(f"  LOW: {low}/{len(refined_offsets)}")
    
    return refined_offsets