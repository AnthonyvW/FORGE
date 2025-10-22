import pygame

from UI.modal import Modal
from printer.automated_controller import AutomatedPrinter
from printer.automation_config import AutomationSettingsManager

from UI.focus_overlay import FocusOverlay

from UI.input.text_field import TextField
from UI.input.button import Button, ButtonShape, ButtonColors
from UI.input.scroll_frame import ScrollFrame
from UI.input.slider import Slider
from UI.input.radio import RadioButton, RadioGroup

from UI.tooltip import Tooltip
from UI.text import Text, TextStyle

from UI.styles import (
    make_button_text_style,
    make_display_text_style,
    make_settings_text_style,
    BASE_BUTTON_COLORS,
    SELECTED_RADIO_COLORS,
    RADIO_TEXT_STYLE,
)


class _Layout:
    """Simple vertical slot layout using a fixed section offset."""
    def __init__(self, offset: int = 60):
        self.offset = offset
        self._i = 0

    def next_y(self) -> int:
        y = self.offset * self._i
        self._i += 1
        return y


def _add_pct_slider(
    *,
    parent,
    x: int,
    y: int,
    title: str,
    initial_pct_0to1: float,
    on_change_pct_0to1
):
    """
    Render a labeled 0..100% slider with 0.1% resolution.
    The underlying callback receives a float in [0.0, 1.0].
    Returns a setter: set_pct_0to1(float) -> None for external syncing.
    """
    Text(title, parent=parent, x=x, y=y + 6, style=make_button_text_style())

    ui_value = max(0.0, min(100.0, float(initial_pct_0to1) * 100.0))

    slider = Slider(
        parent=parent, x=x + 125, y=y, width=230, height=32,
        min_value=0.0, max_value=100.0, initial_value=ui_value,
        step=0.1, tick_count=0, with_buttons=True,
    )

    value_text = Text(f"{ui_value:.1f}%", parent=parent, x=x + 360, y=y + 8, style=make_button_text_style())

    def _on_slider(val: float):
        val = max(0.0, min(100.0, float(val)))
        value_text.set_text(f"{val:.1f}%")
        try:
            on_change_pct_0to1(val / 100.0)
        except Exception as e:
            print(f"[Automation Settings] Failed to apply '{title}' = {val}% → {e}")

    slider.on_change = _on_slider

    def set_pct_0to1(p: float):
        p = max(0.0, min(1.0, float(p)))
        slider.set_value(p * 100.0, notify=False)
        value_text.set_text(f"{p * 100.0:.1f}%")

    return set_pct_0to1


def add_save_load_reset_section(modal, automated_controller: AutomatedPrinter, sync_modal_from_automation, y: int, x: int = 8) -> None:
    import tkinter as tk
    from tkinter import filedialog
    btn_w, btn_h = 88, 28
    spacing = 12
    y += 8

    # Save
    def on_save():
        try:
            automated_controller.save_automation_settings()  # persists to active file (with backups)
        except Exception as e:
            print(f"[Automation Save] Failed: {e}")
        print("Saved Settings")

    Button(
        on_save,
        x=x, y=y, width=btn_w, height=btn_h,
        text="Save", parent=modal,
        colors=BASE_BUTTON_COLORS, text_style=RADIO_TEXT_STYLE,
    )

    # Load (from arbitrary YAML)
    def on_load():
        root = tk.Tk(); root.withdraw()
        cfg_dir = automated_controller.get_automation_config_dir()
        filepath = filedialog.askopenfilename(
            initialdir=str(cfg_dir),
            title="Select Automation Config File",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
        )
        root.destroy()
        if not filepath:
            return
        try:
            # Parity with camera settings loader
            loaded = AutomationSettingsManager.load_from_file(filepath)
            automated_controller.set_automation_settings(loaded, persist=False)  # apply immediately
            sync_modal_from_automation(modal, automated_controller)             # refresh widgets
        except Exception as e:
            print(f"[Automation Load] Failed to load/apply '{filepath}': {e}")
        print("Loaded Settings")

    Button(
        on_load,
        x=x + (btn_w + spacing), y=y, width=btn_w, height=btn_h,
        text="Load", parent=modal,
        colors=BASE_BUTTON_COLORS, text_style=RADIO_TEXT_STYLE,
    )

    # Reset → restore defaults into active (with backup), apply, and persist
    def on_reset():
        try:
            automated_controller.restore_default_automation_settings(persist=True)
            sync_modal_from_automation(modal, automated_controller)
        except Exception as e:
            print(f"[Automation Reset] Failed to restore defaults: {e}")
        print("Reset Settings")

    Button(
        on_reset,
        x=x + 2*(btn_w + spacing), y=y, width=btn_w, height=btn_h,
        text="Reset", parent=modal,
        colors=BASE_BUTTON_COLORS, text_style=RADIO_TEXT_STYLE,
    )


def build_automation_settings_modal(modal: Modal, automated_controller: AutomatedPrinter):

    scroll_area = ScrollFrame(parent=modal, x=0, y=0, width=modal.width, height=365)
    layout = _Layout(offset=40)
    s = automated_controller.automation_settings

    # Image name format
    image_name_format_height = 8 + layout.next_y()
    Text("Image Name Format: ", parent=scroll_area, x=8, y=image_name_format_height + 5, style=make_button_text_style())
    format_field = TextField(
        parent=scroll_area,
        x=220, y=image_name_format_height, width=250,
        allowed_pattern=r'^[^\\/:*?"<>|\x00-\x1F]+$',
        border_color=pygame.Color("#b3b4b6"), text_color=pygame.Color("#5a5a5a")
    )
    format_field.set_text(s.image_name_template)

    def _on_template_change(text: str):
        # empty -> fallback to existing template so we don't persist a blank
        tpl = text.strip() or automated_controller.automation_settings.image_name_template
        automated_controller.update_automation_settings(persist=False, image_name_template=tpl)

    format_field.on_change = _on_template_change
    Tooltip.attach(format_field, 
                   "Format Options\n\
{x}, {y}, {z} - Position coordinates (supports zero-padding and custom decimal delimiter)\n\
{i} - Image index\n\
{f} - Focus score\n\
{d:%Y%m%d} - Date/time (customizable with standard strftime format codes)\n\
\n\
Example:\n\
{d:%Y%m%d}_X={x}_Y={y}_Image_{i} -> 20251021_X=010.40_Y=000.04_Image_1"
                   )


    row_y = layout.next_y()
    Text("Zero-Pad Coordinates", parent=scroll_area, x=8, y=row_y + 5, style=make_button_text_style())

    def on_zero_pad_change(selected_val):
        # Accept "true"/"false" (string) or button.value
        value = True if (selected_val == "true" or getattr(selected_val, "value", None) == "true") else False
        automated_controller.update_automation_settings(persist=False, zero_pad=value)

    zero_group = RadioGroup(allow_deselect=False, on_change=on_zero_pad_change)
    RadioButton(lambda: None, x=220, y=row_y, width=56, height=32, text="True",
                value="true", group=zero_group, parent=scroll_area,
                colors=BASE_BUTTON_COLORS, selected_colors=SELECTED_RADIO_COLORS, text_style=RADIO_TEXT_STYLE)
    RadioButton(lambda: None, x=280, y=row_y, width=64, height=32, text="False",
                value="false", group=zero_group, parent=scroll_area,
                colors=BASE_BUTTON_COLORS, selected_colors=SELECTED_RADIO_COLORS, text_style=RADIO_TEXT_STYLE)

    # Initialize from current settings (default True if missing)
    zero_group.set_value("true" if automated_controller.automation_settings.zero_pad else "false")

    # Decimal delimiter (mutually exclusive)
    row_y = layout.next_y()
    Text("Decimal Delimiter", parent=scroll_area, x=8, y=row_y + 5, style=make_button_text_style())

    def on_delim_change(selected_btn):
        val = None if selected_btn is None else selected_btn.value
        # Only accept the four allowed delimiters
        if val in {"_", "-", "=", "."}:
            automated_controller.update_automation_settings(persist=False, delimiter=val)

    delim_group = RadioGroup(allow_deselect=False, on_change=on_delim_change)

    # Buttons: "_", "-", "=", "."
    RadioButton(lambda: None, x=220, y=row_y, width=36, height=32, text="_",
                value="_", group=delim_group, parent=scroll_area,
                colors=BASE_BUTTON_COLORS, selected_colors=SELECTED_RADIO_COLORS, text_style=RADIO_TEXT_STYLE)
    RadioButton(lambda: None, x=260, y=row_y, width=36, height=32, text="-",
                value="-", group=delim_group, parent=scroll_area,
                colors=BASE_BUTTON_COLORS, selected_colors=SELECTED_RADIO_COLORS, text_style=RADIO_TEXT_STYLE)
    RadioButton(lambda: None, x=300, y=row_y, width=36, height=32, text="=",
                value="=", group=delim_group, parent=scroll_area,
                colors=BASE_BUTTON_COLORS, selected_colors=SELECTED_RADIO_COLORS, text_style=RADIO_TEXT_STYLE)
    RadioButton(lambda: None, x=340, y=row_y, width=36, height=32, text=".",
                value=".", group=delim_group, parent=scroll_area,
                colors=BASE_BUTTON_COLORS, selected_colors=SELECTED_RADIO_COLORS, text_style=RADIO_TEXT_STYLE)

    # Initialize from current settings (default ".")
    delim_group.set_value(automated_controller.automation_settings.delimiter)


    # Focus Scale
    row_y = layout.next_y()
    Text("Focus Scale", parent=scroll_area, x=8, y=row_y + 5, style=make_button_text_style())

    focus_scale_slider = Slider(
        parent=scroll_area, x=153, y=row_y, width=230, height=32,
        min_value=0.0, max_value=1.0, initial_value=getattr(automated_controller.machine_vision, "scale_factor", 1.0),
        step=0.001, tick_count=0, with_buttons=True,
    )

    focus_scale_value = Text(
        f"{getattr(automated_controller.machine_vision, 'scale_factor', 1.0):.3f}",
        parent=scroll_area, x=390, y=row_y + 8, style=make_button_text_style()
    )

    def on_focus_scale_change(val: float):
        try:
            val = max(0.0, min(1.0, float(val)))
            automated_controller.machine_vision.scale_factor = val
            focus_scale_value.set_text(f"{val:.3f}")
        except Exception as e:
            print(f"[Automation Settings] Failed to set focus scale: {e}")

    focus_scale_slider.on_change = on_focus_scale_change


    # Machine Vision Exclusion Zones
    Text("Machine Vision Exclusion Zones", parent=scroll_area, x=8, y=8 + layout.next_y(), style=make_button_text_style())

    slider_setters = {}

    row_y = layout.next_y()
    slider_setters["top"] = _add_pct_slider(
        parent=scroll_area, x=28, y=row_y,
        title="Top Side:",
        initial_pct_0to1=getattr(s, "inset_top_pct", 0.0),
        on_change_pct_0to1=lambda v: automated_controller.update_automation_settings(
            persist=False, inset_top_pct=float(v))
    )

    row_y = layout.next_y()
    slider_setters["left"] = _add_pct_slider(
        parent=scroll_area, x=28, y=row_y,
        title="Left Side:",
        initial_pct_0to1=getattr(s, "inset_left_pct", 0.0),
        on_change_pct_0to1=lambda v: automated_controller.update_automation_settings(
            persist=False, inset_left_pct=float(v))
    )

    row_y = layout.next_y()
    slider_setters["bottom"] = _add_pct_slider(
        parent=scroll_area, x=28, y=row_y,
        title="Bottom Side:",
        initial_pct_0to1=getattr(s, "inset_bottom_pct", 0.0),
        on_change_pct_0to1=lambda v: automated_controller.update_automation_settings(
            persist=False, inset_bottom_pct=float(v))
    )

    row_y = layout.next_y()
    slider_setters["right"] = _add_pct_slider(
        parent=scroll_area, x=28, y=row_y,
        title="Right Side:",
        initial_pct_0to1=getattr(s, "inset_right_pct", 0.0),
        on_change_pct_0to1=lambda v: automated_controller.update_automation_settings(
            persist=False, inset_right_pct=float(v))
    )

    def sync_modal_from_automation(modal_obj, controller):
        st = controller.automation_settings

        # template
        try:
            format_field.set_text(st.image_name_template or "")
        except Exception:
            pass

        # zero-pad
        try:
            zero_group.set_value("true" if st.zero_pad else "false")
        except Exception:
            pass

        # delimiter
        try:
            delim_group.set_value(st.delimiter if st.delimiter in {"_", "-", "=", "."} else ".")
        except Exception:
            pass

        # sliders
        try:
            slider_setters["top"](float(getattr(st, "inset_top_pct", 0.0)))
            slider_setters["left"](float(getattr(st, "inset_left_pct", 0.0)))
            slider_setters["bottom"](float(getattr(st, "inset_bottom_pct", 0.0)))
            slider_setters["right"](float(getattr(st, "inset_right_pct", 0.0)))
        except Exception:
            pass

    add_save_load_reset_section(
        modal,
        automated_controller,
        sync_modal_from_automation,
        y=modal.height - 80
    )