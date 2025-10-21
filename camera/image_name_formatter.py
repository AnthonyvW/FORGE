import math
import re
from typing import Optional, Dict, Any, List
from datetime import datetime

class ImageNameFormatter:
    """
    Safe image filename formatter that can work with or without a controller.

    Recognized placeholders:
      {x} {y} {z}  -> positions in millimeters (float, formatted with decimals)
      {f}          -> focus score (int; -1 if invalid/unavailable)
      {i}          -> index (controller.current_sample_index if controller given,
                       else internal counter self._index)
      {d[:<strftime>]} -> date from system clock; custom via {d:%Y%m%d_%H%M%S}
                          (default is %Y%m%d if no format is provided)

    Unknown placeholders are left intact (e.g., {sample}).

    Optional zero-padding for X/Y/Z integer part to match axis maxima:
      - With controller: uses controller.get_max_x/y/z() (already in mm).
      - Without controller: use set_axis_maxes(x=?, y=?, z=?), else width=1.

    Decimal handling:
      - position_decimals controls number of digits after decimal point.
      - Leading zero is always included for fractional values (e.g., 0.04, not .04).
    """

    _TOKEN_OPEN  = "\uE000"  # for '{{'
    _TOKEN_CLOSE = "\uE001"  # for '}}'
    _FIELD_RE = re.compile(r"\{([^{}]+)\}")

    def __init__(
        self,
        controller: Optional[object] = None,
        *,
        pad_positions: bool = False,
        position_decimals: int = 2,
        delimiter: str = ".",
        template: Optional[str] = None,
        default_date_format: str = "%Y%m%d",
        start_index: int = 1,
    ):
        self.controller = controller
        self.pad_positions = pad_positions
        self.position_decimals = max(0, int(position_decimals))
        self.axis_delimiter = str(delimiter)
        self._template: Optional[str] = template
        self._default_date_format = default_date_format

        # Internal index used when no controller index is available
        self._index = int(start_index)

        # Axis maxima (mm) for padding when no controller is available
        self._axis_max_mm = {"x": None, "y": None, "z": None}

    # ---------------- Template / Config ----------------

    def set_template(self, template: str) -> None:
        self._template = template

    def get_template(self) -> Optional[str]:
        return self._template

    def set_index(self, value: int) -> None:
        self._index = int(value)

    def set_axis_maxes(self, *, x: Optional[int] = None, y: Optional[int] = None, z: Optional[int] = None) -> None:
        if x is not None: self._axis_max_mm["x"] = int(x)
        if y is not None: self._axis_max_mm["y"] = int(y)
        if z is not None: self._axis_max_mm["z"] = int(z)

    # ---------------- Internals ----------------

    @staticmethod
    def _needed_fields(template: str) -> set:
        needed = set()
        for m in ImageNameFormatter._FIELD_RE.finditer(template):
            raw = m.group(1).strip()
            key = raw.split(":", 1)[0].strip()
            needed.add(key)
        return needed

    def _axis_widths(self, needed: set) -> Dict[str, int]:
        """Compute integer digit widths for x/y/z."""
        widths = {"x": 1, "y": 1, "z": 1}

        def digits(n: Optional[int]) -> int:
            if n is None: return 1
            return max(1, len(str(abs(int(n)))))

        def want(axis: str) -> bool:
            return axis in needed

        if self.controller is not None:
            if want("x"):
                try:    widths["x"] = digits(int(round(float(self.controller.get_max_x()))))
                except: widths["x"] = 1
            if want("y"):
                try:    widths["y"] = digits(int(round(float(self.controller.get_max_y()))))
                except: widths["y"] = 1
            if want("z"):
                try:    widths["z"] = digits(int(round(float(self.controller.get_max_z()))))
                except: widths["z"] = 1
        else:
            if want("x"): widths["x"] = digits(self._axis_max_mm["x"])
            if want("y"): widths["y"] = digits(self._axis_max_mm["y"])
            if want("z"): widths["z"] = digits(self._axis_max_mm["z"])

        return widths

    def _collect_values(
        self,
        needed: set,
        *,
        focus_score: Optional[float],
        index: Optional[int],
        auto_increment_index: bool,
    ) -> Dict[str, Any]:
        """
        Pull values for fields that appear in the template. Avoids unnecessary calls.
        - Positions use controller if available; else default to 0.
        - Focus uses provided focus_score if given; else controller if available; else -1.
        - Index prefers explicit index > controller.current_sample_index > internal _index.
          If using internal index and auto_increment_index=True, increments after use.
        - Date is handled during rendering to allow per-token custom formats.
        """
        values: Dict[str, Any] = {}

        # Positions
        if any(k in needed for k in ("x","y","z")):
            if self.controller is not None:
                pos = self.controller.get_position()
                values["x"] = float(pos.x) / 100.0
                values["y"] = float(pos.y) / 100.0
                values["z"] = float(pos.z) / 100.0
            else:
                values.setdefault("x", 0.0)
                values.setdefault("y", 0.0)
                values.setdefault("z", 0.0)

        # Focus
        if "f" in needed:
            if focus_score is not None:
                fs = float(focus_score)
            elif self.controller is not None:
                fs = float(self.controller._af_score_still())
            else:
                fs = float("-inf")
            values["f"] = int(round(fs)) if math.isfinite(fs) else -1

        # Index
        if "i" in needed:
            if index is not None:
                i_val = int(index)
            elif self.controller is not None and hasattr(self.controller, "current_sample_index"):
                i_val = int(getattr(self.controller, "current_sample_index"))
            else:
                i_val = int(self._index)
                if auto_increment_index:
                    self._index += 1
            values["i"] = i_val

        return values

    def _format_axis_value(self, axis: str, v: float, widths: Dict[str, int]) -> str:
        """Format axis value with padding, decimals, and custom delimiter instead of '.'."""
        v_round = round(float(v), self.position_decimals)
        sign = "-" if v_round < 0 else ""
        abs_v = abs(v_round)

        # Generate fixed-precision string, then split integer & fraction
        if self.position_decimals > 0:
            formatted = f"{abs_v:.{self.position_decimals}f}"
            int_part, frac_part = formatted.split(".")
        else:
            int_part, frac_part = f"{int(abs_v)}", ""

        # Pad integer part if requested
        if self.pad_positions:
            w = max(1, widths.get(axis, 1))
            int_part = f"{int(int_part):0{w}d}"

        # Replace '.' with delimiter for filesystem-safe filenames
        if self.position_decimals > 0:
            return f"{sign}{int_part}{self.axis_delimiter}{frac_part}"
        else:
            return f"{sign}{int_part}"

    def _render(self, template: str, values: Dict[str, Any], widths: Dict[str, int]) -> str:
        """Replace placeholders with formatted values."""
        s = template.replace("{{", self._TOKEN_OPEN).replace("}}", self._TOKEN_CLOSE)
        recognized = {"x","y","z","f","i","d"}

        def _sub(m: re.Match) -> str:
            raw = m.group(1).strip()
            parts = raw.split(":", 1)
            key = parts[0].strip()

            if key not in recognized:
                return m.group(0)  # keep unknown placeholder

            if key in ("x","y","z"):
                if key not in values:
                    return m.group(0)
                return self._format_axis_value(key, float(values[key]), widths)

            if key in ("f","i"):
                if key not in values:
                    return m.group(0)
                return str(values[key])

            if key == "d":
                fmt = parts[1] if len(parts) == 2 and parts[1] else self._default_date_format
                try:
                    return datetime.now().strftime(fmt)
                except Exception:
                    # If bad format, leave the token as-is
                    return m.group(0)

            return m.group(0)

        s = self._FIELD_RE.sub(_sub, s)
        return s.replace(self._TOKEN_OPEN, "{").replace(self._TOKEN_CLOSE, "}")

    # ---------------- Public API ----------------

    def get_formatted_string(
        self,
        *,
        template: Optional[str] = None,
        focus_score: Optional[float] = None,
        index: Optional[int] = None,
        auto_increment_index: bool = False,
    ) -> str:
        """
        Build the formatted string.

        Args (named-only, all optional):
          template: override format for this call; or use saved template.
          focus_score: use precomputed focus score (avoid recompute).
          index: explicit index; defaults to controller.current_sample_index or internal _index.
          auto_increment_index: if using the internal index, increment it after this call.

        Notes:
          - Include date directly in the template using {d:%Y%m%d} (or leave as {d} for default).
          - If controller is None and the template includes {x}/{y}/{z}, those fields default to 0.
            You can still get padded zeros by calling set_axis_maxes(...) and enabling pad_positions.
        """
        tpl = template if template is not None else self._template
        if not tpl:
            raise ValueError("No template provided or saved. Call set_template(...) or pass template=...")

        needed = self._needed_fields(tpl)
        widths = self._axis_widths(needed) if self.pad_positions else {"x": 0, "y": 0, "z": 0}

        values = self._collect_values(
            needed,
            focus_score=focus_score,
            index=index,
            auto_increment_index=auto_increment_index,
        )
        return self._render(tpl, values, widths)

    def validate_template(self, template: str, *, strict: bool = False) -> Dict[str, Any]:
        """Return structured validation report."""
        issues: List[str] = []

        s = template.replace("{{", self._TOKEN_OPEN).replace("}}", self._TOKEN_CLOSE)
        depth = 0
        for i, ch in enumerate(s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    issues.append(f"Unmatched '}}' at position {i}.")
                    break
        if depth != 0:
            issues.append("Unbalanced braces detected.")

        recognized_keys = {"x","y","z","f","i","d"}
        seen_recognized: List[str] = []
        seen_unknown: List[str] = []
        date_tokens: List[Dict[str, Any]] = []
        uses_default_date = False
        needs_positions = needs_focus = needs_index = needs_date = False

        for m in self._FIELD_RE.finditer(template):
            raw = m.group(1).strip()
            parts = raw.split(":", 1)
            key = parts[0].strip()
            has_format = len(parts) == 2

            if key not in recognized_keys:
                if strict:
                    issues.append(f"Unknown placeholder {{{raw}}} at pos {m.start()}.")
                if key not in seen_unknown:
                    seen_unknown.append(key)
                continue

            if key not in seen_recognized:
                seen_recognized.append(key)

            if key in ("x","y","z"):
                needs_positions = True
            elif key == "f":
                needs_focus = True
            elif key == "i":
                needs_index = True
            elif key == "d":
                needs_date = True

            if key == "d":
                if has_format:
                    fmt = parts[1]
                    ok = True
                    try:
                        _ = datetime.now().strftime(fmt)
                    except Exception as e:
                        ok = False
                        issues.append(f"Invalid date format in {{{raw}}}: {e}")
                    date_tokens.append({"raw": raw, "format": fmt, "valid": ok})
                else:
                    uses_default_date = True
                    date_tokens.append({"raw": raw, "format": self._default_date_format, "valid": True})
            else:
                if has_format:
                    issues.append(f"Formatting is only supported for {{d}}. Found on {{{raw}}}.")

        needed_keys = set(self._needed_fields(template))
        pad_enabled = bool(self.pad_positions)
        widths = self._axis_widths(needed_keys) if pad_enabled else {"x": 0, "y": 0, "z": 0}

        report: Dict[str, Any] = {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "placeholders": {"recognized": seen_recognized, "unknown": seen_unknown},
            "requires": {
                "positions": needs_positions,
                "focus": needs_focus,
                "index": needs_index,
                "date": needs_date,
            },
            "date": {"tokens": date_tokens, "uses_default_format": uses_default_date},
            "padding": {"enabled": pad_enabled, "widths": widths},
        }

        return report

    def is_template_valid(self, template: str, *, strict: bool = False) -> bool:
        return self.validate_template(template, strict=strict)["is_valid"]
