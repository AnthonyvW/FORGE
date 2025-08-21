from dataclasses import dataclass
from typing import List, Tuple

import pygame
from UI.text import Text, TextStyle
from UI.frame import Frame
from UI.section_frame import Section
from UI.modal import Modal
from UI.camera_view import CameraView

from UI.input.text_field import TextField
from UI.input.button import Button, ButtonShape, ButtonColors
from UI.input.slider import Slider
from UI.input.radio import RadioButton, RadioGroup, SelectedColors

RIGHT_PANEL_WIDTH = 400

@dataclass
class ControlPanel:
    frame: Frame
    sample_label: Text
    inc_button: Button
    dec_button: Button
    go_button: Button
    speed_display: Text
    position_display: Text

def make_button_text_style()->TextStyle:
    return TextStyle(color=pygame.Color("#5a5a5a"), font_size=20)

def make_display_text_style()->TextStyle:
    return TextStyle(color=pygame.Color(32, 32, 32), font_size=18, font_name="assets/fonts/SofiaSans-Regular.ttf")

def make_settings_text_style()->TextStyle:
    return TextStyle(color=pygame.Color(32, 32, 32), font_size=20, font_name="assets/fonts/SofiaSans-Regular.ttf")

def make_button(fn, x, y, w, h, text, shape=ButtonShape.RECTANGLE, z_index = 0, args_provider=None):
    btn = Button(
        function_to_call=fn,
        x=x, y=y,
        width=w, height=h,
        text=text,
        text_style=make_button_text_style(),
        args_provider=args_provider,
        shape=shape,
        z_index=z_index
    )
    return btn

def create_control_panel(
    root_frame: Frame,
    movementSystem,
    camera,
    current_sample_index: int
) -> Tuple[Frame, Text, Button, Button, Button, Text, Text]:
    """
    Builds the right-side control panel and returns:
      control_frame, sample_label, increment_button, decrement_button, go_to_sample_button,
      speed_display, position_display
    """

    control_frame = _build_right_control_panel(root_frame)
    box_spacing = 10

    # --- Camera View
    camera_view = CameraView(
        camera=camera,
        parent=root_frame,
        x=0, y=0,
        width=1.0, height=1.0,
        x_is_percent=True, y_is_percent=True,
        width_is_percent=True, height_is_percent=True,
        z_index=0,  # keep it behind panels/modals that use higher z
        background_color=pygame.Color("black"),
        right_margin_px=RIGHT_PANEL_WIDTH # reserve space for the control panel
    )

    # --- Control Box ---
    control_box = Section(parent=control_frame, title="Control", collapsible=False,
        x=10, y=60, width=RIGHT_PANEL_WIDTH - 20, height=250)
    speed_display, position_display = _build_movement_controls(control_box, movementSystem)

    # --- Automation Box ---
    automation_box = Section(parent=control_frame, title= "Automation", collapsible=False, 
        x=10, y=control_box.y + control_box.height + box_spacing, width = RIGHT_PANEL_WIDTH - 20, height = 90)
    _build_automation_control(automation_box, movementSystem)

    # --- Camera Settings Modal ---
    camera_settings_modal = Modal(parent=root_frame, title="Camera Settings", overlay=False, width=308, height=1010)
    _build_camera_settings_modal(camera_settings_modal, camera)
    camera_settings_modal.open()

    # --- Camera Settings ---
    camera_control = Section(parent=control_frame, title="Camera Control", collapsible=False, 
        x=10,y=automation_box.y + automation_box.height + box_spacing, width = RIGHT_PANEL_WIDTH - 20, height = 123)
    _build_camera_control(camera_control, movementSystem, camera, camera_settings_modal)

    # --- Sample Box ---
    sample_box = Section(parent=control_frame, title="Sample Management", 
        x=10, y=camera_control.y + camera_control.height + box_spacing, width = RIGHT_PANEL_WIDTH - 20, height = 233)
    go_to_sample_button, decrement_button, increment_button, sample_label, pos1_display, pos2_display = _build_sample_box(sample_box, movementSystem, camera, current_sample_index)
  
    # --- Modal ---

    return (
        sample_label,
        increment_button,
        decrement_button,
        go_to_sample_button,
        speed_display,
        position_display,
        pos1_display,
        pos2_display
    )


def _build_camera_settings_modal(modal, camera):
    NUMERIC_PATTERN = r"^-?\d*\.?\d*$"   # existing for slider text fields
    DIGITS_NUM = r"^\d{0,5}$"            # allow up to 5 digits while typing; clamp on commit
    DIGITS_SIGNED = r"^-?\d{0,5}$"

    def clamp_text_to_slider(text_value: str, slider: Slider, tf: TextField, decimals: int | None = None):
        # Treat empty, "-", ".", or "-." as “in-progress” edits; don’t clamp yet.
        if text_value in ("", "-", ".", "-."):
            return
        try:
            val = float(text_value)
        except ValueError:
            return
        clamped = max(slider.min_value, min(slider.max_value, val))
        if clamped != slider.value:
            slider.value = clamped
            if slider.on_change:
                slider.on_change(clamped)
        if decimals is not None:
            tf.set_text(f"{clamped:.{decimals}f}", emit=False)
        else:
            tf.set_text(str(int(clamped)) if float(clamped).is_integer() else str(clamped), emit=False)

    def create_setting(
        parent=modal,
        title="None",
        x=8,
        y=0,
        *,
        min_value=0,
        max_value=100,
        tick_count=8,
        attr: str,                 # e.g., "exposure"
        value_type=int,            # int or float
        decimals: int | None = None
    ):
        Text(title, parent=modal, x=x, y=y + 8, style=make_settings_text_style())

        def get_val():
            return getattr(settings, attr)

        cur = get_val()
        slider = Slider(
            parent=modal, x=x, y=y + 28, width=200, height=32,
            min_value=min_value, max_value=max_value, initial_value=cur,
            tick_count=tick_count, with_buttons=True
        )

        text_field = TextField(
            parent=modal, x=x + 208, y=y + 28, width=80, height=32,
            placeholder=str(cur), allowed_pattern=NUMERIC_PATTERN,
            border_color=pygame.Color("#b3b4b6"), text_color=pygame.Color("#5a5a5a")
        )

        def fmt(v):
            if value_type is int:
                return str(int(v))
            if decimals is not None:
                return f"{float(v):.{decimals}f}"
            return str(v if not (isinstance(v, float) and v.is_integer()) else int(v))

        last_applied = [None]  # closure box to avoid re-applying identical values

        def apply_value(v):
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = float(get_val())
            clamped = max(min_value, min(max_value, v))
            if last_applied[0] is not None and clamped == last_applied[0]:
                return
            # Live-apply to the active camera without persisting
            if value_type is int:
                camera.update_settings(persist=False, **{attr: int(clamped)})
            else:
                camera.update_settings(persist=False, **{attr: float(clamped)})
            text_field.set_text(fmt(clamped), emit=False)
            last_applied[0] = clamped

        # Slider -> live apply
        def on_slider(val: float):
            apply_value(val)

        slider.on_change = on_slider

        # Text while typing: keep it simple; commit applies
        def on_text_change(txt: str):
            if txt in ("", "-", ".", "-."):
                return
            try:
                v = float(txt)
            except ValueError:
                return
            v = max(min_value, min(max_value, v))
            slider.value = v  # keep visuals in sync; apply on commit/drag

        text_field.on_text_change = on_text_change

        # Commit (Enter / blur): apply + snap slider
        def on_commit(txt: str):
            if txt in ("", "-", ".", "-."):
                text_field.set_text(fmt(slider.value), emit=False)
                return
            apply_value(txt)
            slider.value = last_applied[0] if last_applied[0] is not None else slider.value

        text_field.on_commit = on_commit
        text_field.set_text(fmt(cur), emit=False)

    def post_inc(x: list):
        val = x[0]
        x[0] += 1
        return val

    def create_rgb_triplet(
        title: str,
        y: int,
        get_vals,            # () -> tuple[int, int, int]
        set_field_name: str, # name of settings field to update via update_settings(...)
        *,
        per_channel_bounds=None,   # list[ (min,max), (min,max), (min,max) ]
        x: int = 8
    ):
        Text(title, parent=modal, x=x, y=y + 8, style=make_settings_text_style())

        # Small R/G/B labels
        Text("R", parent=modal, x=x,         y=y + 34, style=make_settings_text_style())
        Text("G", parent=modal, x=x + 86,    y=y + 34, style=make_settings_text_style())
        Text("B", parent=modal, x=x + 172,   y=y + 34, style=make_settings_text_style())

        current = list(get_vals())
        bounds = per_channel_bounds or [(0, 255), (0, 255), (0, 255)]

        def make_commit(idx: int, tf: "TextField"):
            lo, hi = bounds[idx]
            def _commit(txt: str):
                try:
                    v = int(txt)
                except (TypeError, ValueError):
                    v = lo
                v = max(lo, min(hi, v))
                current[idx] = v
                # Live-apply tuple via update_settings
                triplet = tuple(current)
                camera.update_settings(persist=False, **{set_field_name: triplet})
                tf.set_text(str(v), emit=False)
            return _commit

        # R
        r_field = TextField(
            parent=modal, x=x + 16, y=y + 28, width=64, height=32,
            placeholder=str(current[0]), allowed_pattern=DIGITS_SIGNED,
            border_color=pygame.Color("#b3b4b6"), text_color=pygame.Color("#5a5a5a"),
            on_commit=None
        )
        r_field.on_commit = make_commit(0, r_field)
        r_field.set_text(str(current[0]), emit=False)

        # G
        g_field = TextField(
            parent=modal, x=x + 102, y=y + 28, width=64, height=32,
            placeholder=str(current[1]), allowed_pattern=DIGITS_SIGNED,
            border_color=pygame.Color("#b3b4b6"), text_color=pygame.Color("#5a5a5a"),
            on_commit=None
        )
        g_field.on_commit = make_commit(1, g_field)
        g_field.set_text(str(current[1]), emit=False)

        # B
        b_field = TextField(
            parent=modal, x=x + 188, y=y + 28, width=64, height=32,
            placeholder=str(current[2]), allowed_pattern=DIGITS_SIGNED,
            border_color=pygame.Color("#b3b4b6"), text_color=pygame.Color("#5a5a5a"),
            on_commit=None
        )
        b_field.on_commit = make_commit(2, b_field)
        b_field.set_text(str(current[2]), emit=False)

    base_colors = ButtonColors(hover_foreground=pygame.Color("#5a5a5a"))
    sel_colors = SelectedColors(
        background=pygame.Color("#b3b4b6"),
        hover_background=pygame.Color("#b3b4b6"),
        foreground=pygame.Color("#b3b4b6"),
        hover_foreground=pygame.Color("#5a5a5a")
    )
    radio_text_style = TextStyle(
        font_size=16, color=pygame.Color("#5a5a5a"),
        hover_color=pygame.Color("#5a5a5a"),
        disabled_color=pygame.Color("#5a5a5a")
    )

    settings = camera.settings
    offset = 60
    offset_index = [0]

    # File Format
    Text("File Format", parent=modal, x=8, y=offset * offset_index[0] + 8, style=make_settings_text_style())

    def on_image_format_change(selected_btn):
        camera.update_settings(persist=False, fformat=(None if selected_btn is None else selected_btn.value))

    image_format = RadioGroup(allow_deselect=False, on_change=on_image_format_change)
    RadioButton(lambda: None, x=8,   y=offset * offset_index[0] + 28, width=48, height=32, text="png",
                value="png", group=image_format, selected=True, parent=modal,
                colors=base_colors, selected_colors=sel_colors, text_style=radio_text_style)
    RadioButton(lambda: None, x=64,  y=offset * offset_index[0] + 28, width=48, height=32, text="jpeg",
                value="jpeg", group=image_format, selected=True, parent=modal,
                colors=base_colors, selected_colors=sel_colors, text_style=radio_text_style)
    RadioButton(lambda: None, x=120, y=offset * offset_index[0] + 28, width=48, height=32, text="tiff",
                value="tiff", group=image_format, selected=True, parent=modal,
                colors=base_colors, selected_colors=sel_colors, text_style=radio_text_style)
    image_format.set_value(settings.fformat)
    offset_index[0] += 1

    # Sliders
    create_setting(title="Camera Temperature", y=offset * post_inc(offset_index),
                   min_value=settings.temp_min, max_value=settings.temp_max,
                   attr="temp", value_type=int)

    # Auto Exposure
    Text("Use Auto Exposure", parent=modal, x=8, y=offset * offset_index[0] + 8, style=make_settings_text_style())
    def on_auto_expo_change(selected_val):
        value = True if selected_val == "true" else False
        camera.update_settings(persist=False, auto_expo=value)
    auto_expo = RadioGroup(allow_deselect=False, on_change=on_auto_expo_change)
    RadioButton(lambda: None, x=8,  y=offset * offset_index[0] + 28, width=48, height=32, text="True",
                value="true", group=auto_expo, selected=True, parent=modal,
                colors=base_colors, selected_colors=sel_colors, text_style=radio_text_style)
    RadioButton(lambda: None, x=64, y=offset * offset_index[0] + 28, width=52, height=32, text="False",
                value="false", group=auto_expo, selected=True, parent=modal,
                colors=base_colors, selected_colors=sel_colors, text_style=radio_text_style)
    auto_expo.set_value("true" if settings.auto_expo else "false")
    offset_index[0] += 1

    create_setting(title="Exposure", y=offset * post_inc(offset_index),
                   min_value=settings.exposure_min, max_value=settings.exposure_max,
                   attr="exposure", value_type=int)

    create_setting(title="Tint", y=offset * post_inc(offset_index),
                   min_value=settings.tint_min, max_value=settings.tint_max,
                   attr="tint", value_type=int)

    create_setting(title="Contrast", y=offset * post_inc(offset_index),
                   min_value=settings.contrast_min, max_value=settings.contrast_max,
                   attr="contrast", value_type=int)

    create_setting(title="Hue", y=offset * post_inc(offset_index),
                   min_value=settings.hue_min, max_value=settings.hue_max,
                   attr="hue", value_type=int)

    create_setting(title="Saturation", y=offset * post_inc(offset_index),
                   min_value=settings.saturation_min, max_value=settings.saturation_max,
                   attr="saturation", value_type=int)

    create_setting(title="Brightness", y=offset * post_inc(offset_index),
                   min_value=settings.brightness_min, max_value=settings.brightness_max,
                   attr="brightness", value_type=int)

    create_setting(title="Gamma", y=offset * post_inc(offset_index),
                   min_value=settings.gamma_min, max_value=settings.gamma_max,
                   attr="gamma", value_type=int)

    create_setting(title="Sharpening", y=offset * post_inc(offset_index),
                   min_value=settings.sharpening_min, max_value=settings.sharpening_max,
                   attr="sharpening", value_type=int)

    # Linear Tone Mapping
    Text("Use Linear Tone Mapping", parent=modal, x=8, y=offset * offset_index[0] + 8, style=make_settings_text_style())
    def on_linear_change(selected_val):
        value = 1 if (selected_val == "true" or (hasattr(selected_val, "value") and selected_val.value == "true")) else 0
        camera.update_settings(persist=False, linear=value)
    linear_tone = RadioGroup(allow_deselect=False, on_change=on_linear_change)
    RadioButton(lambda: None, x=8,  y=offset * offset_index[0] + 28, width=48, height=32, text="True",
                value="true", group=linear_tone, selected=True, parent=modal,
                colors=base_colors, selected_colors=sel_colors, text_style=radio_text_style)
    RadioButton(lambda: None, x=64, y=offset * offset_index[0] + 28, width=52, height=32, text="False",
                value="false", group=linear_tone, selected=True, parent=modal,
                colors=base_colors, selected_colors=sel_colors, text_style=radio_text_style)
    linear_tone.set_value("true" if settings.linear == 1 else "false")
    offset_index[0] += 1

    # Curved Tone Mapping
    Text("Curved Tone Mapping", parent=modal, x=8, y=offset * offset_index[0] + 8, style=make_settings_text_style())
    def on_curved_change(selected_btn):
        camera.update_settings(persist=False, curve=(None if selected_btn is None else selected_btn.value))
    curved_tone = RadioGroup(allow_deselect=False, on_change=on_curved_change)
    RadioButton(lambda: None, x=8,   y=offset * offset_index[0] + 28, width=104, height=32, text="Logarithmic",
                value="Logarithmic", group=curved_tone, selected=True, parent=modal,
                colors=base_colors, selected_colors=sel_colors, text_style=radio_text_style)
    RadioButton(lambda: None, x=120, y=offset * offset_index[0] + 28, width=104, height=32, text="Polynomial",
                value="Polynomial", group=curved_tone, selected=True, parent=modal,
                colors=base_colors, selected_colors=sel_colors, text_style=radio_text_style)
    RadioButton(lambda: None, x=232, y=offset * offset_index[0] + 28, width=48, height=32, text="Off",
                value="Off", group=curved_tone, selected=True, parent=modal,
                colors=base_colors, selected_colors=sel_colors, text_style=radio_text_style)
    curved_tone.set_value(settings.curve)
    offset_index[0] += 1

    # Level Range Low — preserve 4th channel
    def get_level_low_rgb():
        lr = settings.levelrange_low
        return (lr[0], lr[1], lr[2], 0)
    def get_level_high_rgb():
        lr = settings.levelrange_high
        return (lr[0], lr[1], lr[2], 255)

    # Bounds for level ranges (use first 3 channels)
    lr_min = settings.levelrange_min
    lr_max = settings.levelrange_max
    lr_bounds = [(lr_min, lr_max), (lr_min, lr_max), (lr_min, lr_max)]

    create_rgb_triplet(
        title="Level Range Low",
        y=offset * post_inc(offset_index),
        get_vals=get_level_low_rgb,
        set_field_name="levelrange_low",   # update via update_settings
        per_channel_bounds=lr_bounds
    )

    create_rgb_triplet(
        title="Level Range High",
        y=offset * post_inc(offset_index),
        get_vals=get_level_high_rgb,
        set_field_name="levelrange_high",  # update via update_settings
        per_channel_bounds=lr_bounds
    )

    # White Balance Gain
    def get_wbgain():
        return settings.wbgain  # (R, G, B)

    wb_min = settings.wbgain_min
    wb_max = settings.wbgain_max
    wb_bounds = [(wb_min, wb_max), (wb_min, wb_max), (wb_min, wb_max)]

    create_rgb_triplet(
        title="White Balance Gain",
        y=offset * post_inc(offset_index),
        get_vals=get_wbgain,
        set_field_name="wbgain",
        per_channel_bounds=wb_bounds
    )



def _build_right_control_panel(root_frame)-> Frame:
    # --- Control Panel Container ---
    control_frame = Frame(
        parent=root_frame,
        x=0, y=0,
        width=RIGHT_PANEL_WIDTH,
        height=1.0,  # percent height to fill root
        height_is_percent=True,
        x_align='right',
        y_align='top',
        background_color=pygame.Color("#b3b4b6")
    )

    # --- Title Bar ---
    title_bar = Frame(
        parent=control_frame,
        x=0, y=0,
        width=1.0,
        height=50,
        width_is_percent=True,
        background_color=pygame.Color("#909398")
    )

    # --- Title Text ---
    title_text = Text(
        parent=title_bar,
        text="FORGE",
        x=10, y=10,
        x_align="left",
        y_align="top",
        style=TextStyle(
            color=pygame.Color("white"),
            font_size=40,
            bold=True,
            font_name="assets/fonts/SofiaSans-Light.ttf"
        )
    )

    return control_frame


def _build_movement_controls(control_box, movementSystem)-> Frame:
    
    # Movement buttons
    control_box.add_child(make_button(movementSystem.move_x_right,        10,  55,  80, 80, "<", ButtonShape.DIAMOND))
    control_box.add_child(make_button(movementSystem.move_x_left,         100, 55,  80, 80, ">", ButtonShape.DIAMOND))
    control_box.add_child(make_button(movementSystem.move_y_backward,     55,  10,  80, 80, "^", ButtonShape.DIAMOND))
    control_box.add_child(make_button(movementSystem.move_y_forward,      55,  100, 80, 80, "v", ButtonShape.DIAMOND))

    control_box.add_child(make_button(movementSystem.move_z_up,           200, 53,  40, 40, "+"))
    control_box.add_child(make_button(movementSystem.move_z_down,         200, 103, 40, 40, "-"))

    # Speed Buttons
    control_box.add_child(make_button(movementSystem.increase_speed,      250, 53,  40, 40, "S+"))
    control_box.add_child(make_button(movementSystem.decrease_speed,      250, 103, 40, 40, "S-"))
    control_box.add_child(make_button(movementSystem.increase_speed_fast, 300, 53,  40, 40, "F+"))
    control_box.add_child(make_button(movementSystem.decrease_speed_fast, 300, 103, 40, 40, "F-"))

    # Homing Button
    control_box.add_child(make_button(movementSystem.home,                70,  70,  50, 50, "H", ButtonShape.DIAMOND, z_index=1))

    # --- Live readouts ---
    speed_display = Text(
        text=f"Speed: {movementSystem.speed / 100:.2f}",
        parent=control_box,
        x=200, y=155,
        x_align="left",
        y_align="top",
        style=make_display_text_style()
    )

    position_display = Text(
        text=f"X: {movementSystem.position.x/100:.2f} Y: {movementSystem.position.y/100:.2f} Z: {movementSystem.position.z/100:.2f}",
        parent=control_box,
        x=343, y=175,
        x_align="right",
        y_align="top",
        style=make_display_text_style()
    )

    return speed_display, position_display


def _build_sample_box(sample_box, movementSystem, camera, current_sample_index):
    # --- Sample navigation (callbacks assigned later in main.py) ---
    button_height = 40

    # 1st Row
    go_to_sample_button = Button(None, parent=sample_box, 
        x=10, y=10, width=150, height=button_height, text="Go to Sample", text_style=make_button_text_style())

    decrement_button = Button(None, parent=sample_box, 
        x=170, y=10, width=40, height=button_height, text="-", text_style=make_button_text_style())

    sample_label = Text(f"Sample {current_sample_index}", parent=sample_box, 
        x=220, y=20, x_align="left", y_align="top", style=make_button_text_style())

    increment_button = Button(None, parent=sample_box, 
        x=330, y=10, width=40, height=button_height, text="+", text_style=make_button_text_style())

    # 2nd Row
    Button(movementSystem.setPosition1, 10 , 60, 150, button_height, "Set Position 1", parent=sample_box, text_style=make_button_text_style())

    pos1_display = Text(
        text=f"X: {movementSystem.automation_config.x_start/100:.2f} Y: {movementSystem.automation_config.y_start/100:.2f} Z: {movementSystem.automation_config.z_start/100:.2f}",
        parent=sample_box,
        x=170, y=75,
        style=make_display_text_style()
    )

    # 3rd Row
    Button(movementSystem.setPosition2, 10, 110, 150, button_height, "Set Position 2", parent=sample_box, text_style=make_button_text_style())

    pos2_display = Text(
        text=f"X: {movementSystem.automation_config.x_end/100:.2f} Y: {movementSystem.automation_config.y_end/100:.2f} Z: {movementSystem.automation_config.z_end/100:.2f}",
        parent=sample_box,
        x=170, y=125,
        style=make_display_text_style()
    )

    # 4th Row
    Text(
        text=f"Sample Name :",
        parent=sample_box,
        x=10, y=165,
        style=make_button_text_style()
    )
    TextField(parent=sample_box, x=170, y=160, width=200, height=30, placeholder="sample", border_color=pygame.Color("#b3b4b6"), text_color=pygame.Color("#5a5a5a"), on_text_change=camera.set_capture_name)

    return go_to_sample_button, decrement_button, increment_button, sample_label, pos1_display, pos2_display


def _build_camera_control(camera_control, movementSystem, camera, camera_settings_modal):
    camera_control.add_child(make_button(
        lambda pos: camera.capture_image() or camera.save_image(False, filename=pos.to_gcode()),
        10, 10, 117, 40, "Take Photo",
        args_provider=lambda: (movementSystem.get_position(),)
    ))

    path_label = Text(f"Save Path: {camera.capture_path}", parent=camera_control, 
        x=10, y=60, x_align="left", y_align="top", style=make_display_text_style(), truncate_mode="middle", max_width=RIGHT_PANEL_WIDTH - 20 - 20)
    
    def on_set_path():
        path_label.set_text(f"Save Path: {camera.select_capture_path()}")

    Button(on_set_path, 132,  10, 117, 40, "Set Path", parent=camera_control, text_style=make_button_text_style())

    
    Button(lambda: camera_settings_modal.open(), 254,  10, 117, 40, "Settings", parent=camera_control, text_style=make_button_text_style())
    

def _build_automation_control(automation_box, movementSystem):
    
    Button(movementSystem.start_automation, 10,  10, 115, 40, "Start", parent=automation_box, text_style=make_button_text_style())
    Button(movementSystem.halt,             133, 10, 115, 40, "Stop" , parent=automation_box, text_style=make_button_text_style())
    pause = Button(movementSystem.toggle_pause,     255, 10, 115, 40, "Pause", parent=automation_box, text_style=make_button_text_style())
    pause.add_hidden_reason("SYSTEM")


