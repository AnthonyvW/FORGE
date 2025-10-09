from __future__ import annotations
import pygame
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

from UI.text import Text
from UI.input.text_field import TextField
from UI.input.slider import Slider
from UI.input.radio import RadioButton, RadioGroup
from UI.input.button import Button, ButtonColors
from UI.input.scroll_frame import ScrollFrame

from UI.styles import (
    make_settings_text_style,
    BASE_BUTTON_COLORS,
    SELECTED_RADIO_COLORS,
    RADIO_TEXT_STYLE,
)

from camera.camera_settings import CameraSettingsManager  # adjust import if needed

# ---------------------------------------------------------------------------
# Constants / patterns
# ---------------------------------------------------------------------------
NUMERIC_PATTERN = r"^-?\d*\.?\d*$"   # existing for slider text fields
DIGITS_SIGNED   = r"^-?\d{0,5}$"      # allow up to 5 digits while typing; clamp on commit

# ---- Sync helpers ----
def _ensure_sync_registry(modal):
    if not hasattr(modal, "_settings_syncers"):
        modal._settings_syncers = []

def _register_syncer(modal, fn):
    _ensure_sync_registry(modal)
    modal._settings_syncers.append(fn)

def sync_modal_from_camera(modal, camera):
    # Run all registered syncers to update the UI from camera.settings
    for fn in getattr(modal, "_settings_syncers", []):
        try:
            fn()
        except Exception as e:
            print(f"[Modal Sync] syncer failed: {e}")

# ---------------------------------------------------------------------------
# Layout helper
# ---------------------------------------------------------------------------
class _Layout:
    """Simple vertical slot layout using a fixed section offset."""
    def __init__(self, offset: int = 60):
        self.offset = offset
        self._i = 0

    def next_y(self) -> int:
        y = self.offset * self._i
        self._i += 1
        return y


# ---------------------------------------------------------------------------
# Low-level builders (reusable widgets)
# ---------------------------------------------------------------------------

def _fmt_value(v, value_type: type, decimals: int | None) -> str:
    if value_type is int:
        try:
            return str(int(float(v)))
        except Exception:
            return "0"
    if decimals is not None:
        try:
            return f"{float(v):.{decimals}f}"
        except Exception:
            return f"{0.0:.{decimals}f}"
    # Default float-ish formatting but hide trailing .0
    try:
        fv = float(v)
        return str(int(fv)) if fv.is_integer() else str(fv)
    except Exception:
        return "0"


def create_numeric_setting(
    *,
    modal,
    camera,
    settings,
    title: str,
    y: int,
    attr: str,                  # e.g. "exposure"
    min_value: float,
    max_value: float,
    value_type: type = int,
    tick_count: int = 8,
    decimals: int | None = None,
    x: int = 8,
) -> None:
    """Build a labeled slider + numeric text field bound to a single settings attr."""
    Text(title, parent=modal, x=x, y=y + 8, style=make_settings_text_style())

    cur = getattr(settings, attr)
    slider = Slider(
        parent=modal, x=x, y=y + 28, width=200, height=32,
        min_value=min_value, max_value=max_value, initial_value=cur,
        tick_count=tick_count, with_buttons=True,
    )

    text_field = TextField(
        parent=modal, x=x + 208, y=y + 28, width=80, height=32,
        placeholder=str(cur), allowed_pattern=NUMERIC_PATTERN,
        border_color=pygame.Color("#b3b4b6"), text_color=pygame.Color("#5a5a5a"),
    )

    last_applied = [None]

    def apply_value(v):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = float(getattr(settings, attr))
        clamped = max(min_value, min(max_value, fv))
        if last_applied[0] is not None and clamped == last_applied[0]:
            return
        camera.update_settings(
            persist=False,
            **{attr: int(clamped) if value_type is int else float(clamped)},
        )
        text_field.set_text(_fmt_value(clamped, value_type, decimals), emit=False)
        last_applied[0] = clamped

    def on_slider(val: float):
        apply_value(val)

    slider.on_change = on_slider

    def on_text_change(txt: str):
        # Update the slider visual while typing, but do not apply until commit
        if txt in ("", "-", ".", "-."):
            return
        try:
            v = float(txt)
        except ValueError:
            return
        v = max(min_value, min(max_value, v))
        slider.value = v

    text_field.on_text_change = on_text_change

    def on_commit(txt: str):
        if txt in ("", "-", ".", "-."):
            text_field.set_text(_fmt_value(slider.value, value_type, decimals), emit=False)
            return
        apply_value(txt)
        if last_applied[0] is not None:
            slider.value = last_applied[0]

    text_field.on_commit = on_commit
    text_field.set_text(_fmt_value(cur, value_type, decimals), emit=False)

    def _sync_from_camera():
        cur_val = getattr(camera.settings, attr)
        slider.value = float(cur_val)
        text_field.set_text(_fmt_value(cur_val, value_type, decimals), emit=False)
    
    _register_syncer(modal, _sync_from_camera)



def create_rgb_triplet_setting(
    *,
    modal,
    camera,
    title: str,
    y: int,
    get_vals,              # () -> tuple[int, int, int]
    set_field_name: str,   # name of settings field to update via update_settings(...)
    per_channel_bounds: list[tuple[int, int]] | None = None,
    x: int = 8,
) -> None:
    """Build three numeric fields (R/G/B) for an RGB-like tuple setting."""
    Text(title, parent=modal, x=x, y=y + 8, style=make_settings_text_style())

    Text("R", parent=modal, x=x,        y=y + 34, style=make_settings_text_style())
    Text("G", parent=modal, x=x + 86,   y=y + 34, style=make_settings_text_style())
    Text("B", parent=modal, x=x + 172,  y=y + 34, style=make_settings_text_style())

    current = list(get_vals())
    bounds = per_channel_bounds or [(0, 255), (0, 255), (0, 255)]

    def make_commit(idx: int, tf: TextField):
        lo, hi = bounds[idx]
        def _commit(txt: str):
            try:
                v = int(txt)
            except (TypeError, ValueError):
                v = lo
            v = max(lo, min(hi, v))
            current[idx] = v
            camera.update_settings(persist=False, **{set_field_name: tuple(current)})
            tf.set_text(str(v), emit=False)
        return _commit

    r_field = TextField(
        parent=modal, x=x + 16, y=y + 28, width=64, height=32,
        placeholder=str(current[0]), allowed_pattern=DIGITS_SIGNED,
        border_color=pygame.Color("#b3b4b6"), text_color=pygame.Color("#5a5a5a"),
    )
    r_field.on_commit = make_commit(0, r_field)
    r_field.set_text(str(current[0]), emit=False)

    g_field = TextField(
        parent=modal, x=x + 102, y=y + 28, width=64, height=32,
        placeholder=str(current[1]), allowed_pattern=DIGITS_SIGNED,
        border_color=pygame.Color("#b3b4b6"), text_color=pygame.Color("#5a5a5a"),
    )
    g_field.on_commit = make_commit(1, g_field)
    g_field.set_text(str(current[1]), emit=False)

    b_field = TextField(
        parent=modal, x=x + 188, y=y + 28, width=64, height=32,
        placeholder=str(current[2]), allowed_pattern=DIGITS_SIGNED,
        border_color=pygame.Color("#b3b4b6"), text_color=pygame.Color("#5a5a5a"),
    )
    b_field.on_commit = make_commit(2, b_field)
    b_field.set_text(str(current[2]), emit=False)

    def _sync_from_camera():
        vals = list(get_vals())  # this already reads from settings
        r_field.set_text(str(vals[0]), emit=False)
        g_field.set_text(str(vals[1]), emit=False)
        b_field.set_text(str(vals[2]), emit=False)

    _register_syncer(modal, _sync_from_camera)

# ---------------------------------------------------------------------------
# Individual setting sections
# ---------------------------------------------------------------------------

def add_file_format_section(modal, camera, settings, *, y: int, x: int = 8) -> None:
    Text("File Format", parent=modal, x=x, y=y + 8, style=make_settings_text_style())

    def on_image_format_change(selected_btn):
        camera.update_settings(persist=False, fformat=(None if selected_btn is None else selected_btn.value))

    base_y = y + 28
    image_format = RadioGroup(allow_deselect=False, on_change=on_image_format_change)
    RadioButton(lambda: None, x=x,      y=base_y, width=48, height=32, text="png",
                value="png", group=image_format, parent=modal,
                colors=BASE_BUTTON_COLORS, selected_colors=SELECTED_RADIO_COLORS, text_style=RADIO_TEXT_STYLE)
    RadioButton(lambda: None, x=x + 56, y=base_y, width=56, height=32, text="jpeg",
                value="jpeg", group=image_format, parent=modal,
                colors=BASE_BUTTON_COLORS, selected_colors=SELECTED_RADIO_COLORS, text_style=RADIO_TEXT_STYLE)
    RadioButton(lambda: None, x=x + 120, y=base_y, width=56, height=32, text="tiff",
                value="tiff", group=image_format, parent=modal,
                colors=BASE_BUTTON_COLORS, selected_colors=SELECTED_RADIO_COLORS, text_style=RADIO_TEXT_STYLE)

    # Initialize from current settings
    image_format.set_value(settings.fformat)

    _register_syncer(modal, lambda: image_format.set_value(camera.settings.fformat))


def add_camera_temperature_setting(modal, camera, settings, *, y: int, x: int = 8) -> None:
    create_numeric_setting(
        modal=modal, camera=camera, settings=settings,
        title="Camera Temperature", y=y,
        min_value=settings.temp_min, max_value=settings.temp_max,
        attr="temp", value_type=int, x=x,
    )


def add_auto_exposure_section(modal, camera, settings, *, y: int, x: int = 8) -> None:
    Text("Use Auto Exposure", parent=modal, x=x, y=y + 8, style=make_settings_text_style())

    def on_auto_expo_change(selected_val):
        value = True if (selected_val == "true" or getattr(selected_val, "value", None) == "true") else False
        camera.update_settings(persist=False, auto_expo=value)

    base_y = y + 28
    group = RadioGroup(allow_deselect=False, on_change=on_auto_expo_change)
    RadioButton(lambda: None, x=x,      y=base_y, width=48, height=32, text="True",
                value="true", group=group, parent=modal,
                colors=BASE_BUTTON_COLORS, selected_colors=SELECTED_RADIO_COLORS, text_style=RADIO_TEXT_STYLE)
    RadioButton(lambda: None, x=x + 56, y=base_y, width=56, height=32, text="False",
                value="false", group=group, parent=modal,
                colors=BASE_BUTTON_COLORS, selected_colors=SELECTED_RADIO_COLORS, text_style=RADIO_TEXT_STYLE)

    group.set_value("true" if settings.auto_expo else "false")

    _register_syncer(
        modal,
        lambda g=group: g.set_value("true" if camera.settings.auto_expo else "false")
    )



# Scalar slider wrappers (one function per setting)

def add_exposure_setting(modal, camera, settings, *, y: int, x: int = 8) -> None:
    create_numeric_setting(modal=modal, camera=camera, settings=settings,
                           title="Exposure", y=y, attr="exposure",
                           min_value=settings.exposure_min, max_value=settings.exposure_max,
                           value_type=int, x=x)



def add_tint_setting(modal, camera, settings, *, y: int, x: int = 8) -> None:
    create_numeric_setting(modal=modal, camera=camera, settings=settings,
                           title="Tint", y=y, attr="tint",
                           min_value=settings.tint_min, max_value=settings.tint_max,
                           value_type=int, x=x)


def add_contrast_setting(modal, camera, settings, *, y: int, x: int = 8) -> None:
    create_numeric_setting(modal=modal, camera=camera, settings=settings,
                           title="Contrast", y=y, attr="contrast",
                           min_value=settings.contrast_min, max_value=settings.contrast_max,
                           value_type=int, x=x)


def add_hue_setting(modal, camera, settings, *, y: int, x: int = 8) -> None:
    create_numeric_setting(modal=modal, camera=camera, settings=settings,
                           title="Hue", y=y, attr="hue",
                           min_value=settings.hue_min, max_value=settings.hue_max,
                           value_type=int, x=x)


def add_saturation_setting(modal, camera, settings, *, y: int, x: int = 8) -> None:
    create_numeric_setting(modal=modal, camera=camera, settings=settings,
                           title="Saturation", y=y, attr="saturation",
                           min_value=settings.saturation_min, max_value=settings.saturation_max,
                           value_type=int, x=x)


def add_brightness_setting(modal, camera, settings, *, y: int, x: int = 8) -> None:
    create_numeric_setting(modal=modal, camera=camera, settings=settings,
                           title="Brightness", y=y, attr="brightness",
                           min_value=settings.brightness_min, max_value=settings.brightness_max,
                           value_type=int, x=x)


def add_gamma_setting(modal, camera, settings, *, y: int, x: int = 8) -> None:
    create_numeric_setting(modal=modal, camera=camera, settings=settings,
                           title="Gamma", y=y, attr="gamma",
                           min_value=settings.gamma_min, max_value=settings.gamma_max,
                           value_type=int, x=x)


def add_sharpening_setting(modal, camera, settings, *, y: int, x: int = 8) -> None:
    create_numeric_setting(modal=modal, camera=camera, settings=settings,
                           title="Sharpening", y=y, attr="sharpening",
                           min_value=settings.sharpening_min, max_value=settings.sharpening_max,
                           value_type=int, x=x)


def add_linear_tone_mapping_section(modal, camera, settings, *, y: int, x: int = 8) -> None:
    Text("Use Linear Tone Mapping", parent=modal, x=x, y=y + 8, style=make_settings_text_style())

    def on_linear_change(selected_val):
        value = 1 if (selected_val == "true" or getattr(selected_val, "value", None) == "true") else 0
        camera.update_settings(persist=False, linear=value)

    base_y = y + 28
    group = RadioGroup(allow_deselect=False, on_change=on_linear_change)
    RadioButton(lambda: None, x=x,      y=base_y, width=48, height=32, text="True",
                value="true", group=group, parent=modal,
                colors=BASE_BUTTON_COLORS, selected_colors=SELECTED_RADIO_COLORS, text_style=RADIO_TEXT_STYLE)
    RadioButton(lambda: None, x=x + 56, y=base_y, width=56, height=32, text="False",
                value="false", group=group, parent=modal,
                colors=BASE_BUTTON_COLORS, selected_colors=SELECTED_RADIO_COLORS, text_style=RADIO_TEXT_STYLE)

    group.set_value("true" if settings.linear == 1 else "false")

    _register_syncer(
        modal,
        lambda g=group: g.set_value("true" if camera.settings.linear == 1 else "false")
    )



def add_curved_tone_mapping_section(modal, camera, settings, *, y: int, x: int = 8) -> None:
    Text("Curved Tone Mapping", parent=modal, x=x, y=y + 8, style=make_settings_text_style())

    def on_curved_change(selected_btn):
        camera.update_settings(persist=False, curve=(None if selected_btn is None else selected_btn.value))

    base_y = y + 28
    group = RadioGroup(allow_deselect=False, on_change=on_curved_change)
    RadioButton(lambda: None, x=x,        y=base_y, width=104, height=32, text="Logarithmic",
                value="Logarithmic", group=group, parent=modal,
                colors=BASE_BUTTON_COLORS, selected_colors=SELECTED_RADIO_COLORS, text_style=RADIO_TEXT_STYLE)
    RadioButton(lambda: None, x=x + 112,  y=base_y, width=104, height=32, text="Polynomial",
                value="Polynomial", group=group, parent=modal,
                colors=BASE_BUTTON_COLORS, selected_colors=SELECTED_RADIO_COLORS, text_style=RADIO_TEXT_STYLE)
    RadioButton(lambda: None, x=x + 224,  y=base_y, width=48, height=32, text="Off",
                value="Off", group=group, parent=modal,
                colors=BASE_BUTTON_COLORS, selected_colors=SELECTED_RADIO_COLORS, text_style=RADIO_TEXT_STYLE)

    group.set_value(settings.curve)

    _register_syncer(
        modal,
        lambda g=group: g.set_value(camera.settings.curve)
    )





def add_level_range_low_setting(modal, camera, settings, *, y: int, x: int = 8) -> None:
    lr_min = settings.levelrange_min
    lr_max = settings.levelrange_max
    lr_bounds = [(lr_min, lr_max), (lr_min, lr_max), (lr_min, lr_max)]

    def get_level_low_rgb():
        lr = settings.levelrange_low
        return (lr[0], lr[1], lr[2])

    create_rgb_triplet_setting(
        modal=modal, camera=camera, title="Level Range Low", y=y,
        get_vals=get_level_low_rgb, set_field_name="levelrange_low",
        per_channel_bounds=lr_bounds, x=x,
    )


def add_level_range_high_setting(modal, camera, settings, *, y: int, x: int = 8) -> None:
    lr_min = settings.levelrange_min
    lr_max = settings.levelrange_max
    lr_bounds = [(lr_min, lr_max), (lr_min, lr_max), (lr_min, lr_max)]

    def get_level_high_rgb():
        lr = settings.levelrange_high
        return (lr[0], lr[1], lr[2])

    create_rgb_triplet_setting(
        modal=modal, camera=camera, title="Level Range High", y=y,
        get_vals=get_level_high_rgb, set_field_name="levelrange_high",
        per_channel_bounds=lr_bounds, x=x,
    )


def add_white_balance_gain_setting(modal, camera, settings, *, y: int, x: int = 8) -> None:
    wb_min = settings.wbgain_min
    wb_max = settings.wbgain_max
    wb_bounds = [(wb_min, wb_max), (wb_min, wb_max), (wb_min, wb_max)]

    def get_wbgain():
        return settings.wbgain

    create_rgb_triplet_setting(
        modal=modal, camera=camera, title="White Balance Gain", y=y,
        get_vals=get_wbgain, set_field_name="wbgain",
        per_channel_bounds=wb_bounds, x=x,
    )


def add_save_load_reset_section(modal, camera, *, y: int, x: int = 8) -> None:
    btn_w, btn_h = 88, 28
    spacing = 12
    y += 8

    # Save
    Button(
        lambda: camera.save_settings(),
        x=x, y=y, width=btn_w, height=btn_h,
        text="Save", parent=modal,
        colors=BASE_BUTTON_COLORS, text_style=RADIO_TEXT_STYLE,
    )

    # Load
    def on_load():
        root = tk.Tk(); root.withdraw()
        cfg_dir = camera.get_config_dir()
        filepath = filedialog.askopenfilename(
            initialdir=str(cfg_dir),
            title="Select Camera Config File",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
        )
        root.destroy()
        if not filepath:
            return
        try:
            loaded = CameraSettingsManager.load_from_file(filepath)
            camera.set_settings(loaded, persist=False)  # applies immediately
            sync_modal_from_camera(modal, camera)       # <-- refresh widgets in-place
        except Exception as e:
            print(f"[Load Settings] Failed to load/apply '{filepath}': {e}")

    Button(
        on_load,
        x=x + (btn_w + spacing), y=y, width=btn_w, height=btn_h,
        text="Load", parent=modal,
        colors=BASE_BUTTON_COLORS, text_style=RADIO_TEXT_STYLE,
    )

    # Reset
    def on_reset():
        try:
            camera.restore_default_settings(persist=True)  # applies + saves active
            sync_modal_from_camera(modal, camera)          # <-- refresh widgets in-place
        except Exception as e:
            print(f"[Reset Settings] Failed to restore defaults: {e}")

    Button(
        on_reset,
        x=x + 2*(btn_w + spacing), y=y, width=btn_w, height=btn_h,
        text="Reset", parent=modal,
        colors=BASE_BUTTON_COLORS, text_style=RADIO_TEXT_STYLE,
    )

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_camera_settings_modal(modal, camera) -> None:
    """Populate the provided Modal with camera setting controls.
    This applies values live via camera.update_settings(persist=False).

    Layout is organized into vertically stacked sections. Each section is
    built by a dedicated function to keep this module modular and easy to
    extend or rearrange.
    """
    _ensure_sync_registry(modal)
    modal._settings_syncers.clear()
    settings = camera.settings

    scroll_area = ScrollFrame(parent=modal, x=0, y= 0, width=modal.width, height=600)
    layout = _Layout(offset=60)
    

    # File format
    add_file_format_section(scroll_area, camera, settings, y=layout.next_y())

    # Camera temperature
    add_camera_temperature_setting(scroll_area, camera, settings, y=layout.next_y())

    # Auto exposure toggle
    add_auto_exposure_section(scroll_area, camera, settings, y=layout.next_y())

    # Core scalar sliders
    add_exposure_setting(scroll_area, camera, settings, y=layout.next_y())
    add_tint_setting(scroll_area, camera, settings, y=layout.next_y())
    add_contrast_setting(scroll_area, camera, settings, y=layout.next_y())
    add_hue_setting(scroll_area, camera, settings, y=layout.next_y())
    add_saturation_setting(scroll_area, camera, settings, y=layout.next_y())
    add_brightness_setting(scroll_area, camera, settings, y=layout.next_y())
    add_gamma_setting(scroll_area, camera, settings, y=layout.next_y())
    add_sharpening_setting(scroll_area, camera, settings, y=layout.next_y())

    # Tone mapping toggles
    add_linear_tone_mapping_section(scroll_area, camera, settings, y=layout.next_y())
    add_curved_tone_mapping_section(scroll_area, camera, settings, y=layout.next_y())

    # Level ranges + WB
    add_level_range_low_setting(scroll_area, camera, settings, y=layout.next_y())
    add_level_range_high_setting(scroll_area, camera, settings, y=layout.next_y())
    add_white_balance_gain_setting(scroll_area, camera, settings, y=layout.next_y())

    add_save_load_reset_section(
        modal, camera,
        y=modal.height-80
    )
