import time
from collections import defaultdict
from typing import Iterable, Set, Tuple, Dict, Optional, Sequence

import numpy as np
import pygame

from image_processing.analyzers import find_focused_areas


class MachineVision:
    """
    Camera capture + focus analysis + hot-pixel (invalid tile) management.

    Responsibilities pulled out of FocusOverlay:
      - frame capture -> NumPy BGR
      - focused tile computation (hard/soft bands, thresholds)
      - invalid tile map build/clear/query (hot-pixel map)
      - tile index/rect helpers & filtering by invalid map
    """

    def __init__(
        self,
        camera,
        *,
        tile_size: int = 48,
        stride: int = 48,
        top_percent: float = 0.15,
        min_score: float | None = None,
        soft_min_score: float | None = None,
    ):
        self.camera = camera

        # Analysis parameters
        self.tile_size = int(tile_size)
        self.stride = int(stride)
        self.top_percent = float(top_percent)
        self.min_score = min_score
        self.soft_min_score = soft_min_score

        # Hot-pixel / invalid tiles as grid indices
        self._invalid_tiles: Set[Tuple[int, int]] = set()

    # --------------- public properties ---------------
    @property
    def invalid_tiles(self) -> Set[Tuple[int, int]]:
        return self._invalid_tiles

    # --------------- frame I/O ---------------
    def capture_current_frame_bgr(self) -> np.ndarray | None:
        """
        Grab the current camera frame and return it as a NumPy BGR array (H,W,3).
        Returns None if no frame is available.
        """
        frame_surface = self.camera.get_frame()
        if not frame_surface:
            return None

        pg_pixels = pygame.surfarray.array3d(frame_surface)  # (W,H,3) RGB
        img_rgb = np.transpose(pg_pixels, (1, 0, 2))         # (H,W,3)
        img_bgr = img_rgb[:, :, ::-1]                        # RGB->BGR
        return img_bgr

    # --------------- tile helpers ---------------
    def tile_index_from_xy(self, x: int, y: int) -> Tuple[int, int]:
        """Top-left pixel (overlay space) → (col,row) index using stride."""
        col = max(0, int(x) // self.stride)
        row = max(0, int(y) // self.stride)
        return (col, row)

    def tile_rect_from_index(self, col: int, row: int, w: int | None = None, h: int | None = None) -> pygame.Rect:
        """(col,row) grid index → pygame.Rect in overlay space."""
        rw = int(w if w is not None else self.tile_size)
        rh = int(h if h is not None else self.tile_size)
        rx = int(col * self.stride)
        ry = int(row * self.stride)
        return pygame.Rect(rx, ry, rw, rh)

    # --------------- invalid map ---------------
    def clear_hot_pixel_map(self) -> None:
        self._invalid_tiles.clear()

    def is_tile_invalid(self, col: int, row: int) -> bool:
        return (col, row) in self._invalid_tiles

    def filter_tiles(self, tiles: Iterable) -> list:
        """
        Drop any tiles whose top-left falls on an invalid (col,row).
        Assumes tiles have x,y members in overlay coordinates.
        """
        out = []
        for t in tiles:
            col, row = self.tile_index_from_xy(int(t.x), int(t.y))
            if (col, row) not in self._invalid_tiles:
                out.append(t)
        return out

    # --------------- focused tile computation ---------------
    def compute_focused_tiles(
        self,
        *,
        include_soft: bool = True,
        filter_invalid: bool = True,
    ) -> dict:
        """
        Returns a dict with lists of tiles:
        {
          "all":  [...],
          "hard": [...],  # >= min_score
          "soft": [...],  # soft_min_score <= score < min_score (if enabled)
        }
        """
        img_bgr = self.capture_current_frame_bgr()
        if img_bgr is None:
            return {"all": [], "hard": [], "soft": []}

        tiles_all = find_focused_areas(
            img_bgr,
            tile_size=self.tile_size,
            stride=self.stride,
            top_percent=self.top_percent,
            min_score=self.min_score,
            soft_min_score=(self.soft_min_score if include_soft else None),
        ) or []

        # Split soft/hard by score (if soft band enabled)
        soft_tiles, hard_tiles = [], []
        if include_soft and self.soft_min_score is not None:
            hard_cut = self.min_score if self.min_score is not None else float("inf")
            for t in tiles_all:
                s = getattr(t, "score", None)
                if s is not None and self.soft_min_score <= s < hard_cut:
                    soft_tiles.append(t)
                else:
                    hard_tiles.append(t)
        else:
            hard_tiles = tiles_all

        if filter_invalid:
            soft_tiles = self.filter_tiles(soft_tiles)
            hard_tiles = self.filter_tiles(hard_tiles)

        return {
            "all":  (hard_tiles + soft_tiles) if include_soft else hard_tiles,
            "hard": hard_tiles,
            "soft": soft_tiles,
        }

    def get_in_focus_tiles(
        self,
        *,
        band: str = "all",       # "all" | "hard" | "soft"
        as_rects: bool = False   # pygame.Rects in overlay coords
    ):
        result = self.compute_focused_tiles(include_soft=(band in ("all", "soft")), filter_invalid=True)
        tiles = result.get(band, result["all"]) if band in ("all", "hard", "soft") else result["all"]

        if not as_rects:
            return tiles

        rects = []
        for t in tiles:
            rects.append(pygame.Rect(int(t.x), int(t.y), int(t.w), int(t.h)))
        return rects

    # --------------- hot-pixel sampler ---------------
    def build_hot_pixel_map(
        self,
        duration_sec: float = 1.0,
        *,
        dilate: int = 0,
        min_hits: int = 1,
        max_fps: int = 30,
        include_soft: bool = True,
    ) -> Dict[str, int]:
        """
        Sample the camera with the lens cap on. Any tile that appears 'in focus'
        ≥ min_hits times during duration_sec is marked INVALID (hot).
        """
        t_end = time.monotonic() + max(0.05, float(duration_sec))
        min_hits = max(1, int(min_hits))
        stride = int(self.stride)

        hits: Dict[Tuple[int, int], int] = defaultdict(int)

        # Simple FPS throttle
        min_dt = 1.0 / float(max(1, int(max_fps)))
        t_next = 0.0

        frames = 0
        while time.monotonic() < t_end:
            now = time.monotonic()
            if now < t_next:
                time.sleep(max(0.0, t_next - now))
            t_next = time.monotonic() + min_dt

            res = self.compute_focused_tiles(include_soft=include_soft, filter_invalid=False)
            for t in res["all"]:
                col = int(t.x) // stride
                row = int(t.y) // stride
                hits[(col, row)] += 1

            frames += 1

        candidates: Set[Tuple[int, int]] = {ij for ij, n in hits.items() if n >= min_hits}

        if dilate > 0 and candidates:
            expanded: Set[Tuple[int, int]] = set()
            for (c, r) in candidates:
                for dc in range(-dilate, dilate + 1):
                    for dr in range(-dilate, dilate + 1):
                        expanded.add((c + dc, r + dr))
            candidates = expanded

        before = len(self._invalid_tiles)
        self._invalid_tiles |= candidates
        after = len(self._invalid_tiles)

        return {
            "frames": frames,
            "candidates": len(candidates),
            "newly_marked": after - before,
            "total_invalid": after,
        }

    # --------------- Color Sampler -----------------
    def get_average_color(
        self,
        *,
        space: str = "BGR",               # "BGR", "RGB", or "HSV"
        rect: Optional[pygame.Rect | Sequence[int]] = None,
        as_int: bool = True,
        y_method: str = "luma601"         # "luma601", "luma709", "relY_linear", "LabLstar"
    ) -> Optional[Tuple[int | float, int | float, int | float, int | float]]:
        """
        Compute the average color of the current camera frame and append Y (luminance-like scalar).

        Args:
            space:
                Output color space for the first 3 channels: "BGR", "RGB", or "HSV".
                Return is a 4-tuple: (B,G,R,Y), (R,G,B,Y), or (H,S,V,Y).
            rect:
                Optional ROI to average, accepts pygame.Rect or (x, y, w, h).
            as_int:
                If True, quantizes to ints (BGR/RGB/SV in [0,255], H in [0,179]).
                Y scaling depends on y_method (see below).
            y_method:
                - "luma601": Y′ = 0.299R′ + 0.587G′ + 0.114B′ on gamma-encoded 8-bit channels.
                  as_int=True → [0,255]; as_int=False → float in [0,255].
                - "luma709": Y′ = 0.2126R′ + 0.7152G′ + 0.0722B′ on gamma-encoded channels.
                  as_int=True → [0,255]; as_int=False → float in [0,255].
                - "relY_linear": Undo sRGB gamma, then Y = 0.2126R + 0.7152G + 0.0722B (linear).
                  as_int=True → scaled to [0,255]; as_int=False → float in [0,1].
                - "LabLstar": Convert to linear Y then CIELAB L* from Y/Yn (D65, Yn=1).
                  as_int=True → scaled to [0,255] from [0,100]; as_int=False → float in [0,100].

        Returns:
            4-tuple in the chosen space with Y appended, or None if no frame / empty ROI.
        """
        img_bgr = self.capture_current_frame_bgr()
        if img_bgr is None:
            return None

        h, w, _ = img_bgr.shape

        # ----- ROI clamp -----
        if rect is not None:
            r = rect if isinstance(rect, pygame.Rect) else pygame.Rect(*rect)
            x0 = max(0, min(r.x, w))
            y0 = max(0, min(r.y, h))
            x1 = max(0, min(r.x + r.w, w))
            y1 = max(0, min(r.y + r.h, h))
            if x1 <= x0 or y1 <= y0:
                return None
            roi = img_bgr[y0:y1, x0:x1, :]
        else:
            roi = img_bgr

        # ---- helpers ----
        def srgb_to_linear_01(u8: np.ndarray) -> np.ndarray:
            """u8: uint8/float in 0..255 -> linear 0..1"""
            u = u8.astype(np.float64) / 255.0
            return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)

        def lab_Lstar_from_Y(Y_lin_01: np.ndarray) -> np.ndarray:
            """Y in 0..1 (relative to Yn=1, D65) -> L* per pixel"""
            # CIE f(t)
            eps = (6.0 / 29.0) ** 3  # ~0.008856
            kappa = (29.0 / 3.0) ** 3 / 116.0  # used in linear segment form
            t = Y_lin_01
            # Standard piecewise:
            f = np.where(t > eps, np.cbrt(t), (t * (1.0 / (3.0 * (6.0 / 29.0) ** 2))) + (4.0 / 29.0))
            return 116.0 * f - 16.0  # 0..100

        # ---- compute out3 (first three channels) ----
        space_u = space.upper()
        if space_u == "RGB":
            mean_bgr = roi.mean(axis=(0, 1)).astype(np.float64)  # [B,G,R] in 0..255
            out3 = (mean_bgr[2], mean_bgr[1], mean_bgr[0])      # (R,G,B)
        elif space_u == "BGR":
            mean_bgr = roi.mean(axis=(0, 1)).astype(np.float64)  # [B,G,R] in 0..255
            out3 = (mean_bgr[0], mean_bgr[1], mean_bgr[2])      # (B,G,R)
        elif space_u == "HSV":
            try:
                import cv2
            except Exception as e:
                raise RuntimeError(
                    "HSV output requires OpenCV (cv2). Install opencv-python to use space='HSV'."
                ) from e
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)      # H:0..179, S,V:0..255
            H = roi_hsv[:, :, 0].astype(np.float64)
            S = roi_hsv[:, :, 1].astype(np.float64)
            V = roi_hsv[:, :, 2].astype(np.float64)
            # Circular mean for Hue
            ang = H * (2.0 * np.pi / 180.0)
            sin_mean = np.sin(ang).mean()
            cos_mean = np.cos(ang).mean()
            mean_ang = np.arctan2(sin_mean, cos_mean)
            if mean_ang < 0:
                mean_ang += 2.0 * np.pi
            H_mean = (mean_ang * 180.0) / (2.0 * np.pi)
            out3 = (H_mean, float(S.mean()), float(V.mean()))
        else:
            raise ValueError("space must be one of: 'BGR', 'RGB', 'HSV'")

        # ---- compute Y according to y_method ----
        ym = y_method.lower()
        if ym not in ("luma601", "luma709", "rely_linear", "lablstar"):
            raise ValueError("y_method must be one of: 'luma601', 'luma709', 'relY_linear', 'LabLstar'")

        if ym in ("luma601", "luma709"):
            # Use channel means on gamma-encoded values (0..255)
            # mean_bgr is available if space was BGR/RGB; recompute if space='HSV'
            if space_u == "HSV":
                mean_bgr_full = roi.mean(axis=(0, 1)).astype(np.float64)  # [B,G,R]
            else:
                mean_bgr_full = mean_bgr  # from above
            Bm, Gm, Rm = mean_bgr_full[0], mean_bgr_full[1], mean_bgr_full[2]
            if ym == "luma601":
                Y_val = 0.114 * Bm + 0.587 * Gm + 0.299 * Rm
            else:
                Y_val = 0.0722 * Bm + 0.7152 * Gm + 0.2126 * Rm  # BT.709 luma on gamma-encoded
            # Scale rules
            Y_out = int(round(Y_val)) if as_int else float(Y_val)

        elif ym == "rely_linear":
            # Undo gamma per channel, then weighted sum (BT.709/sRGB primaries)
            B_lin = srgb_to_linear_01(roi[:, :, 0])
            G_lin = srgb_to_linear_01(roi[:, :, 1])
            R_lin = srgb_to_linear_01(roi[:, :, 2])
            Y_lin_mean = float((0.0722 * B_lin + 0.7152 * G_lin + 0.2126 * R_lin).mean())
            if as_int:
                Y_out = int(round(Y_lin_mean * 255.0))  # present in 8-bit scale
            else:
                Y_out = Y_lin_mean  # 0..1

        else:  # "LabLstar"
            # Linear relative Y then L* per pixel, average L*
            B_lin = srgb_to_linear_01(roi[:, :, 0])
            G_lin = srgb_to_linear_01(roi[:, :, 1])
            R_lin = srgb_to_linear_01(roi[:, :, 2])
            Y_lin = 0.0722 * B_lin + 0.7152 * G_lin + 0.2126 * R_lin  # 0..1
            Lstar = lab_Lstar_from_Y(Y_lin)
            Lstar_mean = float(Lstar.mean())  # 0..100
            if as_int:
                Y_out = int(round(Lstar_mean * 255.0 / 100.0))  # map 0..100 -> 0..255
            else:
                Y_out = Lstar_mean

        # ---- quantize out3 if requested ----
        if as_int:
            if space_u == "HSV":
                out3_q = (int(round(out3[0])), int(round(out3[1])), int(round(out3[2])))
            else:
                out3_q = (int(round(out3[0])), int(round(out3[1])), int(round(out3[2])))
            return (*out3_q, Y_out)
        else:
            return (float(out3[0]), float(out3[1]), float(out3[2]), float(Y_out))