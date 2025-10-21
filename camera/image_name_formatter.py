import math
import re
from typing import Optional, Dict, Any, List
from datetime import datetime

class ImageNameFormatter:
    """
    Safe image filename formatter that can work with or without a controller.

    Recognized placeholders:
      {x} {y} {z}  -> positions in millimeters (int)
      {f}          -> focus score (int; -1 if invalid/unavailable)
      {i}          -> index (controller.current_sample_index if controller given,
                       else internal counter self._index)
      {d[:<strftime>]} -> date from system clock; custom format via {d:%Y%m%d_%H%M%S}
                          (default is %Y%m%d if no format is provided)

    Unknown placeholders are left intact (e.g., {sample}).
    Optional zero-padding for X/Y/Z to match axis maxima:
      - With controller: uses controller.get_max_x/y/z() (already in mm).
      - Without controller: use set_axis_maxes(x=?, y=?, z=?), else width=1.
    """

    _TOKEN_OPEN  = "\uE000"  # for '{{'
    _TOKEN_CLOSE = "\uE001"  # for '}}'
    _FIELD_RE = re.compile(r"\{([^{}]+)\}")  # captures the content inside {}

    def __init__(
        self,
        controller: Optional[object] = None,
        *,
        pad_positions: bool = False,
        template: Optional[str] = None,
        default_date_format: str = "%Y%m%d",
        start_index: int = 1,
    ):
        self.controller = controller
        self.pad_positions = pad_positions
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
        """Set the internal index used when no controller index is available."""
        self._index = int(value)

    def set_axis_maxes(self, *, x: Optional[int] = None, y: Optional[int] = None, z: Optional[int] = None) -> None:
        """
        Set axis maxima (integers in mm) for padding when no controller is present.
        Any arg left as None will keep its previous value.
        """
        if x is not None: self._axis_max_mm["x"] = int(x)
        if y is not None: self._axis_max_mm["y"] = int(y)
        if z is not None: self._axis_max_mm["z"] = int(z)

    # ---------------- Internals ----------------

    @staticmethod
    def _needed_fields(template: str) -> set:
        """Return the set of placeholder *keys* that appear in the template (without formats)."""
        needed = set()
        for m in ImageNameFormatter._FIELD_RE.finditer(template):
            raw = m.group(1).strip()
            key = raw.split(":", 1)[0].strip()  # 'd:%Y%m%d' -> 'd'
            needed.add(key)
        return needed

    def _axis_widths(self, needed: set) -> Dict[str, int]:
        """
        Compute digit widths for x/y/z. If controller exists, pull from it.
        Otherwise, use self._axis_max_mm (if set), else width=1.
        Only compute for axes that are actually needed by the template.
        """
        widths = {"x": 1, "y": 1, "z": 1}

        def digits(n: Optional[int]) -> int:
            if n is None: return 1
            return max(1, len(str(abs(int(n)))))

        if self.controller is not None:
            if "x" in needed:
                try:    widths["x"] = digits(int(round(float(self.controller.get_max_x()))))
                except: widths["x"] = 1
            if "y" in needed:
                try:    widths["y"] = digits(int(round(float(self.controller.get_max_y()))))
                except: widths["y"] = 1
            if "z" in needed:
                try:    widths["z"] = digits(int(round(float(self.controller.get_max_z()))))
                except: widths["z"] = 1
        else:
            if "x" in needed: widths["x"] = digits(self._axis_max_mm["x"])
            if "y" in needed: widths["y"] = digits(self._axis_max_mm["y"])
            if "z" in needed: widths["z"] = digits(self._axis_max_mm["z"])

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
        if any(k in needed for k in ("x", "y", "z")):
            if self.controller is not None:
                pos = self.controller.get_position()  # ticks (0.01 mm)
                if "x" in needed: values["x"] = int(round(pos.x / 100.0))
                if "y" in needed: values["y"] = int(round(pos.y / 100.0))
                if "z" in needed: values["z"] = int(round(pos.z / 100.0))
            else:
                if "x" in needed: values["x"] = 0
                if "y" in needed: values["y"] = 0
                if "z" in needed: values["z"] = 0

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

        # NOTE: Date ({d:...}) is rendered per-occurence to allow different formats in one template.
        return values

    def _render(self, template: str, values: Dict[str, Any], widths: Dict[str, int]) -> str:
        """
        Selective replacement + optional zero padding; unknowns preserved.
        Supports:
          {d}             -> default self._default_date_format
          {d:<strftime>}  -> custom format, e.g., {d:%Y%m%d_%H%M%S}
        """
        s = template.replace("{{", self._TOKEN_OPEN).replace("}}", self._TOKEN_CLOSE)
        recognized = {"x", "y", "z", "f", "i", "d"}

        def _sub(m: re.Match) -> str:
            raw = m.group(1).strip()             # e.g., 'd:%Y%m%d_%H%M%S' or 'x'
            parts = raw.split(":", 1)
            key = parts[0].strip()

            if key not in recognized:
                return m.group(0)  # keep unknown placeholder

            if key in ("x", "y", "z"):
                if key not in values:
                    return m.group(0)
                v = int(values[key])
                if self.pad_positions:
                    w = max(1, widths.get(key, 1))
                    sign = "-" if v < 0 else ""
                    return f"{sign}{abs(v):0{w}d}"
                return str(v)

            if key in ("f", "i"):
                if key not in values:
                    return m.group(0)
                return str(values[key])

            if key == "d":
                # Per-token format: {d:%Y%m%d_%H%M%S}, else default
                fmt = parts[1] if len(parts) == 2 and parts[1] else self._default_date_format
                try:
                    return datetime.now().strftime(fmt)
                except Exception:
                    # If bad format, leave the token as-is to avoid surprises
                    return m.group(0)

            return m.group(0)

        s = self._FIELD_RE.sub(_sub, s)
        return s.replace(self._TOKEN_OPEN, "{").replace(self._TOKEN_CLOSE, "}")

    # ---------------- Single public API ----------------

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

    def validate_template(
        self,
        template: str,
        *,
        strict: bool = False
    ) -> Dict[str, Any]:
        """
        Validate a filename template and return a structured report.

        Args:
        template: The template string to validate.
        strict:   If True, unknown placeholders are treated as errors.

        Returns:
        {
            "is_valid": bool,            # overall validity flag
            "issues": [str],             # human-readable problems found (if any)
            "brace_balance": {           # basic brace sanity
            "balanced": bool
            },
            "placeholders": {            # what the template uses
            "recognized": ["x","y",...],
            "unknown": ["sample", ...]
            },
            "requires": {                # what data is needed at render time
            "needs_positions": bool,   # any of x/y/z appears
            "needs_focus": bool,       # f appears
            "needs_index": bool,       # i appears
            "needs_date": bool         # d appears
            },
            "date": {                    # per-token date statuses
            "tokens": [                # one item per {d} occurrence
                {"raw": "d:%Y%m%d", "format": "%Y%m%d", "valid": True},
                ...
            ],
            "uses_default_format": bool  # True if any {d} had no explicit format
            },
            "padding": {                 # whether XYZ padding can be applied
            "pad_positions_enabled": bool,
            "axis_widths": {"x": int, "y": int, "z": int},   # computed widths
            "will_pad": {"x": bool, "y": bool, "z": bool}    # only true if field is used in template
            }
        }
        """
        issues: List[str] = []

        # 1) Brace balance check (respect literal {{ and }})
        sentinel_open  = self._TOKEN_OPEN
        sentinel_close = self._TOKEN_CLOSE
        s = template.replace("{{", sentinel_open).replace("}}", sentinel_close)

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
            issues.append("Unbalanced braces: number of '{' and '}' does not match.")

        # 2) Scan placeholders
        recognized_keys = {"x", "y", "z", "f", "i", "d"}
        seen_recognized: List[str] = []
        seen_unknown: List[str] = []
        date_tokens: List[Dict[str, Any]] = []
        uses_default_date = False

        # Track what data will be required to render
        needs_positions = False
        needs_focus = False
        needs_index = False
        needs_date = False

        for m in self._FIELD_RE.finditer(template):
            raw = m.group(1).strip()        # e.g., 'd:%Y%m%d' or 'x' or 'sample'
            parts = raw.split(":", 1)
            key = parts[0].strip()
            has_format = (len(parts) == 2)

            if key not in recognized_keys:
                if strict:
                    issues.append(f"Unknown placeholder {{{raw}}} at pos {m.start()}.")
                # record unknown either way
                if key not in seen_unknown:
                    seen_unknown.append(key)
                continue

            # collect presence of recognized keys
            if key not in seen_recognized:
                seen_recognized.append(key)

            # mark requirements
            if key in ("x", "y", "z"):
                needs_positions = True
            elif key == "f":
                needs_focus = True
            elif key == "i":
                needs_index = True
            elif key == "d":
                needs_date = True

            # formatting rules
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
                    # store with default marker
                    date_tokens.append({"raw": raw, "format": self._default_date_format, "valid": True})
            else:
                # no formatting allowed on non-date fields
                if has_format:
                    issues.append(
                        f"Formatting is only supported for {{d}}. Found format on {{{raw}}}."
                    )

        # 3) Padding readiness report (only meaningful for x/y/z)
        needed_keys = set(self._needed_fields(template))  # keys only (no formats)
        pad_enabled = bool(self.pad_positions)
        if pad_enabled:
            widths = self._axis_widths(needed_keys)
        else:
            widths = {"x": 0, "y": 0, "z": 0}

        will_pad = {
            "x": pad_enabled and ("x" in needed_keys) and (widths.get("x", 0) > 1),
            "y": pad_enabled and ("y" in needed_keys) and (widths.get("y", 0) > 1),
            "z": pad_enabled and ("z" in needed_keys) and (widths.get("z", 0) > 1),
        }

        # 4) Compile report
        report: Dict[str, Any] = {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "brace_balance": {"balanced": depth == 0},
            "placeholders": {
                "recognized": seen_recognized,
                "unknown": seen_unknown,
            },
            "requires": {
                "needs_positions": needs_positions,
                "needs_focus": needs_focus,
                "needs_index": needs_index,
                "needs_date": needs_date,
            },
            "date": {
                "tokens": date_tokens,
                "uses_default_format": uses_default_date,
            },
            "padding": {
                "pad_positions_enabled": pad_enabled,
                "axis_widths": {"x": widths.get("x", 0), "y": widths.get("y", 0), "z": widths.get("z", 0)},
                "will_pad": will_pad,
            },
        }

        return report

    # Convenience: a tiny wrapper if you prefer a boolean call
    def is_template_valid(self, template: str, *, strict: bool = False) -> bool:
        return self.validate_template(template, strict=strict)["is_valid"]