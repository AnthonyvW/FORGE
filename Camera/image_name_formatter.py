import math
import re
from typing import Optional, Dict, Any

class ImageNameFormatter:
    """
    Safe formatter for image names.

    Recognized placeholders: {x}, {y}, {z}, {f}
      - x/y/z: current position in millimeters (int), optionally zero-padded
      - f: focus score (int; -1 if invalid)

    Unknown placeholders are left as-is (e.g., {sample} stays {sample}).

    If pad_positions=True, X/Y/Z are left-padded with zeros to match the
    digit width of controller.get_max_x()/get_max_y()/get_max_z() (in mm).
    """

    _TOKEN_OPEN  = "\uE000"  # placeholder for '{{'
    _TOKEN_CLOSE = "\uE001"  # placeholder for '}}'
    _FIELD_RE = re.compile(r"\{([^{}]+)\}")  # matches {name} but not doubled braces

    def __init__(self, controller, *, pad_positions: bool = False):
        self.controller = controller
        self.pad_positions = pad_positions

        # Cached/last render info
        self._last_template: Optional[str] = None
        self._last_values: Optional[Dict[str, Any]] = None  # {"x":..., "y":..., "z":..., "f":...}
        self._last_result: Optional[str] = None

    # ---------- internal helpers ----------

    def _axis_widths(self) -> Dict[str, int]:
        """
        Compute digit widths from controller max values for x, y, z (already in mm).
        Falls back to width=1 if not available.
        """
        widths = {"x": 1, "y": 1, "z": 1}
        try:
            mx = int(round(float(self.controller.get_max_x())))
            widths["x"] = max(1, len(str(abs(mx))))
        except Exception:
            pass
        try:
            my = int(round(float(self.controller.get_max_y())))
            widths["y"] = max(1, len(str(abs(my))))
        except Exception:
            pass
        try:
            mz = int(round(float(self.controller.get_max_z())))
            widths["z"] = max(1, len(str(abs(mz))))
        except Exception:
            pass
        return widths

    def _current_values(self, focus_score: Optional[float]) -> Dict[str, int]:
        """
        Pull current X/Y/Z (ticks → mm ints) and a focus score (int).
        """
        pos = self.controller.get_position()  # .x/.y/.z are in 0.01 mm ticks
        x_mm = int(round(pos.x / 100.0))
        y_mm = int(round(pos.y / 100.0))
        z_mm = int(round(pos.z / 100.0))

        if focus_score is None:
            focus_score = self.controller._af_score_still()

        f_int = int(round(focus_score)) if math.isfinite(focus_score) else -1
        return {"x": x_mm, "y": y_mm, "z": z_mm, "f": f_int}

    def _render(self, template: str, values: Dict[str, Any]) -> str:
        """
        Core render (handles escaping, selective replacement, optional padding).
        """
        # Protect literal doubled braces
        s = template.replace("{{", self._TOKEN_OPEN).replace("}}", self._TOKEN_CLOSE)

        recognized = set(("x", "y", "z", "f"))
        widths = self._axis_widths() if self.pad_positions else {"x": 0, "y": 0, "z": 0}

        def _sub(m: re.Match) -> str:
            key = m.group(1).strip()
            if key in recognized:
                val = values[key]
                if key in ("x", "y", "z") and self.pad_positions:
                    w = max(1, widths.get(key, 1))
                    sign = "-" if int(val) < 0 else ""
                    return f"{sign}{abs(int(val)):0{w}d}"
                return str(val)
            # Unknown → keep as-is
            return m.group(0)

        s = self._FIELD_RE.sub(_sub, s)
        # Restore literal braces
        return s.replace(self._TOKEN_OPEN, "{").replace(self._TOKEN_CLOSE, "}")

    # ---------- public API ----------

    def format(self, template: str, focus_score: Optional[float] = None, *, store: bool = True) -> str:
        """
        Format using live positions; compute focus score unless provided.
        Stores the result by default for quick reuse.
        """
        values = self._current_values(focus_score)
        result = self._render(template, values)
        if store:
            self._last_template = template
            self._last_values = values
            self._last_result = result
        return result

    def format_with_focus(self, template: str, focus_score: float, *, store: bool = True) -> str:
        """
        Same as format(), but ALWAYS uses the provided focus_score (no recompute).
        """
        return self.format(template, focus_score=focus_score, store=store)

    def last(self) -> Optional[str]:
        """
        Return the last stored formatted string (or None if none yet).
        """
        return self._last_result

    def rerender_last_with_focus(self, focus_score: float, *, store: bool = True) -> Optional[str]:
        """
        Re-render the last-used template with the CURRENT positions and a provided focus score.
        Returns None if no prior template was stored.
        """
        if not self._last_template:
            return None
        # Refresh positions, but inject provided focus score
        values = self._current_values(focus_score)
        result = self._render(self._last_template, values)
        if store:
            self._last_values = values
            self._last_result = result
        return result
