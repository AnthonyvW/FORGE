import time
from collections import defaultdict
from typing import Iterable, Set, Tuple, Dict, Optional, Sequence

import numpy as np
import pygame

from image_processing.analyzers import find_focused_areas, ImageAnalyzer, FocusAnalysisResult


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
        inset_left_pct: float = 0.10,
        inset_top_pct: float = 0.0,
        inset_right_pct: float = 0.10,
        inset_bottom_pct: float = 0.0,
        scale_factor: float = 1.0,
    ):
        self.camera = camera

        # Analysis parameters
        self.tile_size = int(tile_size)
        self.stride = int(stride)
        self.top_percent = float(top_percent)
        self.min_score = min_score
        self.soft_min_score = soft_min_score

        self._scale_factor: float = float(scale_factor) if scale_factor is not None else 1.0
        if self._scale_factor <= 0:
            self._scale_factor = 1.0

        # Hot-pixel / invalid tiles as grid indices
        self._invalid_maps: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}

        # Edge distances as percentages of the frame (0.0..1.0)
        self._edge_left_pct: float = inset_left_pct
        self._edge_right_pct: float = inset_right_pct
        self._edge_top_pct: float = inset_top_pct
        self._edge_bottom_pct: float = inset_bottom_pct


    # ---------------- internal helpers ----------------
    def _key_for_shape(self, arr: np.ndarray | None) -> Optional[Tuple[int, int]]:
        if arr is None or arr.ndim < 2:
            return None
        h, w = int(arr.shape[0]), int(arr.shape[1])
        return (w, h)

    def _get_invalid_set(self, shape_key: Tuple[int, int], *, create: bool) -> Optional[Set[Tuple[int, int]]]:
        if shape_key in self._invalid_maps:
            return self._invalid_maps[shape_key]
        if create:
            s: Set[Tuple[int, int]] = set()
            self._invalid_maps[shape_key] = s
            return s
        return None

    # --------------- public properties ---------------
    @property
    def scale_factor(self) -> float:
        return self._scale_factor

    @scale_factor.setter
    def scale_factor(self, v: float) -> None:
        try:
            f = float(v)
        except Exception:
            f = 1.0
        # Recommend <= 1.0 (downscale). Allow >1.0 but clamp to a sane range.
        if f <= 0:
            f = 1.0
        self._scale_factor = f

    @staticmethod
    def _clamp01(v: float) -> float:
        try:
            vf = float(v)
        except Exception:
            return 0.0
        if vf < 0.0: return 0.0
        if vf > 1.0: return 1.0
        return vf

    # Left
    @property
    def edge_left_pct(self) -> float:
        return self._edge_left_pct
    @edge_left_pct.setter
    def edge_left_pct(self, v: float) -> None:
        self._edge_left_pct = self._clamp01(v)

    # Right
    @property
    def edge_right_pct(self) -> float:
        return self._edge_right_pct
    @edge_right_pct.setter
    def edge_right_pct(self, v: float) -> None:
        self._edge_right_pct = self._clamp01(v)

    # Top
    @property
    def edge_top_pct(self) -> float:
        return self._edge_top_pct
    @edge_top_pct.setter
    def edge_top_pct(self, v: float) -> None:
        self._edge_top_pct = self._clamp01(v)

    # Bottom
    @property
    def edge_bottom_pct(self) -> float:
        return self._edge_bottom_pct
    @edge_bottom_pct.setter
    def edge_bottom_pct(self, v: float) -> None:
        self._edge_bottom_pct = self._clamp01(v)

    def set_edge_margins(
        self,
        *,
        left: float | None = None,
        right: float | None = None,
        top: float | None = None,
        bottom: float | None = None,
    ) -> None:
        """Set any subset of edge percentages (0..1)."""
        if left   is not None: self.edge_left_pct   = left
        if right  is not None: self.edge_right_pct  = right
        if top    is not None: self.edge_top_pct    = top
        if bottom is not None: self.edge_bottom_pct = bottom

    def get_edge_margins(self) -> tuple[float, float, float, float]:
        """Return (left, right, top, bottom) as 0..1 floats."""
        return (self._edge_left_pct, self._edge_right_pct, self._edge_top_pct, self._edge_bottom_pct)
    
    
    def get_interior_rect_pixels(self, img_w: int, img_h: int) -> pygame.Rect:
        """Return the interior (non-edge) rect in RAW pixel coords."""
        l_pct, r_pct, t_pct, b_pct = self.get_edge_margins()
        left   = int(round(img_w * max(0.0, min(1.0, l_pct))))
        right  = int(round(img_w * max(0.0, min(1.0, r_pct))))
        top    = int(round(img_h * max(0.0, min(1.0, t_pct))))
        bottom = int(round(img_h * max(0.0, min(1.0, b_pct))))
        x = left
        y = top
        w = max(0, img_w - left - right)
        h = max(0, img_h - top - bottom)
        return pygame.Rect(x, y, w, h)
    
    @property
    def invalid_tiles(self) -> Set[Tuple[int, int]]:
        """
        Backwards-compatible view: returns the union of all invalid tiles across maps.
        Prefer invalid_tiles_for_current_frame() when applying to a specific frame.
        """
        all_set: Set[Tuple[int, int]] = set()
        for s in self._invalid_maps.values():
            all_set |= s
        return all_set

    def invalid_tiles_for_current_frame(self) -> Set[Tuple[int, int]]:
        """Invalid tiles set for the most recent available frame (auto-picks source)."""
        img = self.capture_current_frame(color="rgb")  # auto
        k = self._key_for_shape(img)
        if k is None:
            return set()
        s = self._invalid_maps.get(k)
        return s if s is not None else set()
    
    # --------------- frame I/O ---------------
    def capture_current_frame(self, color: str = "rgb", source: str = "latest") -> np.ndarray | None:
        """
        Return the latest RAW frame as a NumPy array in the requested color/order.

        Args:
            color (str): "rgb" (default), "bgr", or "gray".
            source (str):
                - "latest"   -> prefer full-res STILL if available, else STREAM (default)
                - "still"  -> full-resolution still only
                - "stream" -> live stream frame only (lower res)

        Returns:
            np.ndarray | None: (H, W, C) uint8 for rgb/bgr, (H, W) uint8 for gray.
        """
        color = color.lower()
        source = source.lower()

        # 1) Choose source
        if source == "latest":
            # Prefer still; fall back to stream. Wait if a still is currently being taken.
            img = self.camera.get_last_frame(prefer="latest", wait_for_still=True)
        elif source == "still":
            # Strict: only return a finished still (no fallback)
            img = self.camera.get_last_image()
        elif source == "stream":
            # Strict: only return the latest stream frame (no fallback)
            img = self.camera.get_last_stream_array()
        else:
            raise ValueError("source must be 'latest', 'still', or 'stream'")

        if img is None:
            return None

        # 2) Convert color request
        if color == "rgb":
            arr = img
        elif color == "bgr":
            arr = img[..., ::-1]
        elif color == "gray":
            # ITU-R BT.601 luma, uint8
            arr = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            raise ValueError("Unsupported color. Use 'rgb', 'bgr', or 'gray'.")

        # 3) Ensure contiguous
        return arr if arr.flags.c_contiguous else np.ascontiguousarray(arr)

    def get_interior_cropped_frame(
        self,
        *,
        color: str = "bgr",
        source: str = "latest",
        return_rect: bool = False
    ) -> np.ndarray | tuple[np.ndarray, pygame.Rect] | None:
        """
        Return the current frame cropped to the interior (respects edge_*_pct).
        If return_rect=True, also return the pygame.Rect used for cropping.
        """
        img = self.capture_current_frame(color=color, source=source)
        if img is None:
            return None

        h_raw, w_raw = img.shape[:2]
        interior = self.get_interior_rect_pixels(w_raw, h_raw)  # uses edge margins
        if interior.w <= 0 or interior.h <= 0:
            return None

        y0, y1 = int(interior.top), int(interior.bottom)
        x0, x1 = int(interior.left), int(interior.right)
        cropped = img[y0:y1, x0:x1]  # raw pixels

        return (cropped, interior) if return_rect else cropped

    # --------------- tile helpers ---------------
    def tile_index_from_xy(self, x: int, y: int) -> Tuple[int, int]:
        col = max(0, int(x) // self.stride)
        row = max(0, int(y) // self.stride)
        return (col, row)

    def tile_rect_from_index(self, col: int, row: int, w: int | None = None, h: int | None = None) -> pygame.Rect:
        rw = int(w if w is not None else self.tile_size)
        rh = int(h if h is not None else self.tile_size)
        rx = int(col * self.stride)
        ry = int(row * self.stride)
        return pygame.Rect(rx, ry, rw, rh)
    
    # --------------- invalid map ---------------
    def clear_hot_pixel_map(self, *, source: Optional[str] = None, shape: Optional[Tuple[int, int]] = None) -> None:
        """
        Clear invalid maps.
        - If shape is provided, clears only that (w,h) grid.
        - Else if source provided, clears the current frame map for that source.
        - Else clears all maps.
        """
        if shape is not None:
            self._invalid_maps.pop(tuple(shape), None)
            return

        if source is not None:
            img = self.capture_current_frame(color="rgb", source=source)
            k = self._key_for_shape(img)
            if k is not None and k in self._invalid_maps:
                self._invalid_maps.pop(k, None)
            return

        # Clear all
        self._invalid_maps.clear()

    def is_tile_invalid(self, col: int, row: int, *, shape: Optional[Tuple[int, int]] = None) -> bool:
        """
        Check invalid tile against the specified shape (w,h) or the latest frame's map.
        """
        if shape is None:
            img = self.capture_current_frame(color="rgb")
            k = self._key_for_shape(img)
        else:
            k = tuple(shape)
        if k is None:
            return False
        s = self._invalid_maps.get(k)
        return (col, row) in s if s is not None else False

    def _filter_tiles_for_shape(self, tiles: Iterable, shape_key: Tuple[int, int]) -> list:
        inv = self._invalid_maps.get(shape_key, set())
        out = []
        stride = self.stride
        for t in tiles:
            col = int(t.x) // stride
            row = int(t.y) // stride
            if (col, row) not in inv:
                out.append(t)
        return out

    # --------------- focused tile computation ---------------
    def compute_focused_tiles(
        self,
        *,
        include_soft: bool = True,
        filter_invalid: bool = True,
        source: str = "latest",
        restrict_to_interior: bool = True,
    ) -> dict:
        """
        Returns a dict with lists of tiles for the requested source/frame grid.
        """
        img_bgr = self.capture_current_frame(color="bgr", source=source)
        if img_bgr is None:
            return {"all": [], "hard": [], "soft": []}

        shape_key = self._key_for_shape(img_bgr)

        tiles_all = find_focused_areas(
            img_bgr,
            tile_size=self.tile_size,
            stride=self.stride,
            top_percent=self.top_percent,
            min_score=self.min_score,
            soft_min_score=(self.soft_min_score if include_soft else None),
            scale_factor=self._scale_factor,
        ) or []

        if restrict_to_interior and img_bgr is not None:
            h_raw, w_raw = img_bgr.shape[:2]
            interior = self.get_interior_rect_pixels(w_raw, h_raw)

            def _tile_fully_inside(t) -> bool:
                x0 = int(t.x); y0 = int(t.y)
                x1 = int(t.x + t.w); y1 = int(t.y + t.h)
                # Use 'contains' semantics (no edge bleed into red bands)
                return (x0 >= interior.left and y0 >= interior.top and
                        x1 <= interior.right and y1 <= interior.bottom)

            tiles_all = [t for t in tiles_all if _tile_fully_inside(t)]

        # Split soft/hard
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

        if filter_invalid and shape_key is not None:
            hard_tiles = self._filter_tiles_for_shape(hard_tiles, shape_key)
            soft_tiles = self._filter_tiles_for_shape(soft_tiles, shape_key)

        return {
            "all":  (hard_tiles + soft_tiles) if include_soft else hard_tiles,
            "hard": hard_tiles,
            "soft": soft_tiles,
        }

    def get_in_focus_tiles(
        self,
        *,
        band: str = "all",       # "all" | "hard" | "soft"
        as_rects: bool = False,  # pygame.Rects in raw image coords
        source: str = "latest",
    ):
        result = self.compute_focused_tiles(include_soft=(band in ("all", "soft")),
                                            filter_invalid=True,
                                            source=source)
        tiles = result.get(band, result["all"]) if band in ("all", "hard", "soft") else result["all"]

        if not as_rects:
            return tiles

        rects = [pygame.Rect(int(t.x), int(t.y), int(t.w), int(t.h)) for t in tiles]
        return rects

    def analyze_focus(
        self,
        *,
        kernel_size: int = 7,
        source: str = "latest",
    ) -> Optional[FocusAnalysisResult]:
        """
        Quadrant focus analysis using ImageAnalyzer.analyze_focus on the interior crop.
        Edge percents come from this MachineVision instance.
        Returns a FocusAnalysisResult or None if no frame available.
        """
        img_bgr = self.capture_current_frame(color="bgr", source=source)
        if img_bgr is None:
            return None

        # Hand margins straight through so ImageAnalyzer does the crop internally
        return ImageAnalyzer.analyze_focus(
            img_bgr,
            kernel_size=kernel_size,
            edge_left_pct=self._edge_left_pct,
            edge_right_pct=self._edge_right_pct,
            edge_top_pct=self._edge_top_pct,
            edge_bottom_pct=self._edge_bottom_pct,
            scale_factor=self._scale_factor,
        )

    # --------------- hot-pixel sampler ---------------
    def build_hot_pixel_map(
        self,
        duration_sec: float = 1.0,
        *,
        dilate: int = 0,
        min_hits: int = 1,
        max_fps: int = 30,
        include_soft: bool = True,
        sources: Sequence[str] = ("stream", "still"),   # <— default: learn both grids
        still_every_n_frames: int = 1,                  # capture a still each time we sample "still"
    ) -> Dict[str, int]:
        """
        Sample with the lens cap on. Any tile that appears 'in focus' ≥ min_hits times
        during duration_sec is marked INVALID for that frame's (w,h) grid.

        Builds/updates a separate map per grid. If 'still' is included in sources,
        we actively capture stills so that map is populated too.
        """
        t_end = time.monotonic() + max(0.05, float(duration_sec))
        min_hits = max(1, int(min_hits))
        min_dt = 1.0 / float(max(1, int(max_fps)))
        stride = self.stride

        # Per-grid hit counters
        pergrid_hits: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))

        frames = 0
        while time.monotonic() < t_end:
            t0 = time.monotonic()

            for src in sources:
                # If sampling 'still', capture a fresh still periodically (default: every pass)
                if src == "still" and (frames % max(1, still_every_n_frames) == 0):
                    # Assumes your camera exposes capture_image(). (You mentioned using it.)
                    self.camera.capture_image()
                    while self.camera.is_taking_image:
                        time.sleep(0.01)

                res = self.compute_focused_tiles(include_soft=include_soft, filter_invalid=False, source=src)
                img = self.capture_current_frame(color="rgb", source=src)
                k = self._key_for_shape(img)
                if k is None:
                    continue

                hits = pergrid_hits[k]
                for t in res["all"]:
                    col = int(t.x) // stride
                    row = int(t.y) // stride
                    hits[(col, row)] = hits.get((col, row), 0) + 1

            frames += 1

            # FPS throttle
            dt = time.monotonic() - t0
            if dt < min_dt:
                time.sleep(min_dt - dt)

        # Commit candidates per grid
        total_new = 0
        total_candidates = 0

        for grid_key, hits in pergrid_hits.items():
            candidates: Set[Tuple[int, int]] = {ij for ij, n in hits.items() if n >= min_hits}

            if dilate > 0 and candidates:
                expanded: Set[Tuple[int, int]] = set()
                for (c, r) in candidates:
                    for dc in range(-dilate, dilate + 1):
                        for dr in range(-dilate, dilate + 1):
                            expanded.add((c + dc, r + dr))
                candidates = expanded

            inv = self._get_invalid_set(grid_key, create=True)
            before = len(inv)
            inv |= candidates
            after = len(inv)

            total_new += (after - before)
            total_candidates += len(candidates)
        print("Finished Creating Hot Pixel Maps")
        return {
            "frames": frames,
            "candidates": total_candidates,
            "newly_marked": total_new,
            "total_invalid_maps": {f"{w}x{h}": len(s) for (w, h), s in self._invalid_maps.items()},
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
        img_bgr = self.capture_current_frame(color="bgr")
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
        
    def is_black(self, *, threshold: float = 5.0, source: str = "latest") -> bool:
        """
        Black-frame check on the *interior-cropped* region.
        """
        frame = self.get_interior_cropped_frame(color="bgr", source=source)
        if frame is None:
            return True  # treat as black/empty if nothing to analyze
        return ImageAnalyzer.is_black(frame, threshold=threshold)