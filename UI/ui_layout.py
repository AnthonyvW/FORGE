from dataclasses import dataclass
from typing import List, Tuple
import os
import sys
import subprocess

import pygame

from printer.automated_controller import AutomatedPrinter

from UI.text import Text, TextStyle
from UI.frame import Frame
from UI.section_frame import Section
from UI.modal import Modal
from UI.camera_view import CameraView
from UI.focus_overlay import FocusOverlay
from UI.list_frame import ListFrame
from UI.flex_frame import FlexFrame

from UI.input.text_field import TextField
from UI.input.button import Button, ButtonShape, ButtonColors
from UI.input.button_icon import ButtonIcon
from UI.input.toggle_button import ToggleButton, ToggledColors
from UI.input.scroll_frame import ScrollFrame
from UI.styles import (
    make_button_text_style,
    make_display_text_style,
    make_settings_text_style,
)
from UI.modals.camera_settings_modal import build_camera_settings_modal
from UI.modals.automation_settings_modal import build_automation_settings_modal

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
    movementSystem: AutomatedPrinter,
    camera,
    current_sample_index: int
) -> Tuple[Frame, Text, Button, Button, Button, Text, Text]:
    """
    Builds the right-side control panel and returns:
      control_frame, sample_label, increment_button, decrement_button, go_to_sample_button,
      speed_display, position_display
    """

    control_frame = _build_right_control_panel(root_frame)

    # --- Camera View
    camera_view = CameraView(
        camera=camera,
        parent=root_frame,
        x=0, y=0,
        width=1.0, height=1.0,
        x_is_percent=True, y_is_percent=True,
        width_is_percent=True, height_is_percent=True,
        z_index=0,
        background_color=pygame.Color("black"),
        right_margin_px=RIGHT_PANEL_WIDTH # reserve space for the control panel
    )
    machine_vision_overlay = FocusOverlay(camera_view, movementSystem.machine_vision)


    # --- Control Box ---
    control_box = Section(
        parent=control_frame,
        title="Control",
        collapsible=True,
        x=0, y=0, width=1.0, height=250,
        width_is_percent=True
    )
    speed_display, position_display = _build_movement_controls(control_box, movementSystem)

    # --- Automation Box ---
    automation_box = Section(
        parent=control_frame,
        title="Automation",
        collapsible=True,
        x=0, y=0, width=1.0, height=140,
        width_is_percent=True
    )
    automation_settings_modal = Modal(parent=root_frame, title="Automation Settings", overlay=False, width=500, height=445)
    build_automation_settings_modal(automation_settings_modal, movementSystem)
    automation_settings_modal.open()
    _build_automation_control(automation_box, movementSystem, machine_vision_overlay, automation_settings_modal)

    # --- Camera Settings Modal ---
    camera_settings_modal = Modal(parent=root_frame, title="Camera Settings", overlay=False, width=308, height=660)
    build_camera_settings_modal(camera_settings_modal, camera)

    # --- Camera Settings ---
    camera_control = Section(
        parent=control_frame,
        title="Camera Control",
        collapsible=True,
        x=0, y=0, width=1.0, height=163,
        width_is_percent=True
    )
    _build_camera_control(camera_control, movementSystem, camera, camera_settings_modal)

    # --- Sample Box ---
    sample_box = Section(
        parent=control_frame,
        title="Sample Management",
        collapsible=True,
        x=0, y=0, width=1.0, fill_remaining_height=True,
        width_is_percent=True, padding=(0,0,10,0)
    )
    go_to_sample_button, decrement_button, increment_button, sample_label = _build_sample_box(
        sample_box, movementSystem, camera, current_sample_index
    )

    return (
        sample_label,
        increment_button,
        decrement_button,
        go_to_sample_button,
        speed_display,
        position_display
    )

def _build_right_control_panel(root_frame) -> Frame:
    # --- Control Panel Container (plain Frame) ---
    control_frame = Frame(
        parent=root_frame,
        x=0, y=0,
        width=RIGHT_PANEL_WIDTH,
        height=1.0,              # fill vertical space of root
        height_is_percent=True,
        x_align='right',
        y_align='top',
        background_color=pygame.Color("#b3b4b6")
    )

    # --- Title Bar (not part of flex) ---
    title_bar = Frame(
        parent=control_frame,
        x=0, y=0,
        width=1.0,
        height=50,
        width_is_percent=True,
        background_color=pygame.Color("#909398")
    )

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

    # --- Content Column (this is the flex container) ---
    content_column = FlexFrame(
        parent=control_frame,
        x=0,
        y=50,                        # start 50px down
        width=RIGHT_PANEL_WIDTH,
        height=0,                    # ignored when fill_remaining_height=True
        height_is_percent=False,
        padding=(10, 10, 10, 10),
        gap=10,
        fill_child_width=True,
        align_horizontal="left",

        # key bits:
        fill_remaining_height=True,  # <-- stretch to parent's bottom
        auto_height_to_content=False # <-- avoid fighting with fill-to-bottom
    )

    # Return both so caller can attach sections to content_column
    return content_column

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
    """
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
    """
    # 4th Row
    def build_row(i: int, parent: Frame) -> None:
        on_overrides = ToggledColors(
            background=pygame.Color("#7ed957"),
            hover_background=pygame.Color("#6bc24b"),
            foreground=pygame.Color("#2f6f2a"),
            hover_foreground=pygame.Color("#2f6f2a"),
        )

        def on_state_changed(state: bool, btn: ToggleButton):
            # Fires only when the ON/OFF value changes.
            btn.set_text("X" if state else "")

        ToggleButton(
            parent=parent,
            x=0, y=0, width=30, height=30,
            text="",             # label is independent of state; change it in on_change if you want
            toggled=False,
            on_change=on_state_changed,
            toggled_colors=on_overrides,
            text_style=make_button_text_style()
        )

        Text(
            text=f"Sample {i+1}:",
            parent=parent,
            x=40, y=5,
            style=make_button_text_style()
        )

        TextField(parent=parent, x=150, y=0, width=180, height=30, placeholder=f"Sample {i+1} Name", border_color=pygame.Color("#b3b4b6"), text_color=pygame.Color("#5a5a5a"))
        
    scroll_area = ScrollFrame(parent=sample_box, x=10, y= 60, width=RIGHT_PANEL_WIDTH - 40, height=295, fill_remaining_height=True)

    lst = ListFrame(parent=scroll_area, x=10, y=10, width=1.0, height=700,
                width_is_percent=True,
                row_height=35, count=movementSystem.get_num_slots(), row_builder=build_row)
    
    movementSystem.sample_list = lst

    return go_to_sample_button, decrement_button, increment_button, sample_label#, pos1_display, pos2_display


def _build_camera_control(camera_control, movementSystem: AutomatedPrinter, camera, camera_settings_modal):

    # Header Settings Button
    settings = Button(lambda: camera_settings_modal.open(), x=0, y=0, 
        width=camera_control.header.height, 
        height=camera_control.header.height, 
        parent=camera_control.header,
        colors=ButtonColors(
            background=pygame.Color("#dbdbdb"),
            foreground=pygame.Color("#dbdbdb"),
            hover_background=pygame.Color("#b3b4b6"),
            disabled_background=pygame.Color("#dbdbdb"),
            disabled_foreground=pygame.Color("#dbdbdb")
        )
    )
    camera_control.add_header_button(settings)
    ButtonIcon(
        parent_button=settings,
        image="assets/gear.png",
        normal_replace=(122, 122, 122, 255),
        hover_replace=(122, 122, 122, 255),
        size=(camera_control.header.height - 8, camera_control.header.height - 8),         # explicit size in pixels
        inset_px=0
    )

    # Body of Camera Control
    camera_control.add_child(make_button(
        camera.capture_and_save,
        10, 10, 117, 40, "Take Photo"
    ))

    path_label = Text(f"Save Path: {camera.capture_path}", parent=camera_control, 
        x=10, y=60, x_align="left", y_align="top", style=make_display_text_style(), truncate_mode="middle", max_width=RIGHT_PANEL_WIDTH - 20 - 20)
    
    def on_set_path():
        path_label.set_text(f"Save Path: {camera.select_capture_path()}")

    Button(on_set_path, 132,  10, 117, 40, "Set Path", parent=camera_control, text_style=make_button_text_style())

    
    Button(lambda: movementSystem.start_autofocus(), 10, 85, 117, 40, "Autofocus", parent=camera_control, text_style=make_button_text_style())
    Button(lambda: movementSystem.start_fine_autofocus(), 132, 85, 167, 40, "Fine Autofocus", parent=camera_control, text_style=make_button_text_style())
    
    def open_capture_folder():
        """Open the capture folder in the system's default file explorer."""
        # Convert relative paths to absolute
        folder = os.path.abspath(camera.capture_path)

        if not os.path.isdir(folder):
            print(f"Path does not exist or is not a folder: {folder}")
            return

        if sys.platform.startswith("win"):
            os.startfile(folder)  # type: ignore[attr-defined]
        elif sys.platform.startswith("darwin"):  # macOS
            subprocess.run(["open", folder])
        else:  # Linux and other Unix
            subprocess.run(["xdg-open", folder])

        print("Opened Image Output Folder")

    Button(open_capture_folder,x=254, y=10, width=117, height=40, text="Open Path", parent=camera_control, text_style=make_button_text_style())
    

def _build_automation_control(automation_box, movementSystem, machine_vision_overlay, automation_settings_modal):

    settings = Button(lambda: automation_settings_modal.open(), x=0, y=0, 
        width=automation_box.header.height, 
        height=automation_box.header.height, 
        parent=automation_box.header,
        colors=ButtonColors(
            background=pygame.Color("#dbdbdb"),
            foreground=pygame.Color("#dbdbdb"),
            hover_background=pygame.Color("#b3b4b6"),
            disabled_background=pygame.Color("#dbdbdb"),
            disabled_foreground=pygame.Color("#dbdbdb")
        )
    )
    automation_box.add_header_button(settings)
    ButtonIcon(
        parent_button=settings,
        image="assets/gear.png",
        normal_replace=(122, 122, 122, 255),
        hover_replace=(122, 122, 122, 255),
        size=(automation_box.header.height - 8, automation_box.header.height - 8),         # explicit size in pixels
        inset_px=0
    )

    
    Button(movementSystem.start_automation, 10,  10, 115, 40, "Start", parent=automation_box, text_style=make_button_text_style())
    Button(movementSystem.stop,             133, 10, 115, 40, "Stop" , parent=automation_box, text_style=make_button_text_style())
    Button(movementSystem.toggle_pause,     255, 10, 115, 40, "Pause", parent=automation_box, text_style=make_button_text_style())    

    def toggle_overlay():
        print("Toggling Overlay")
        machine_vision_overlay.toggle_overlay()

    Button(toggle_overlay,x=10, y=60, width=117, height=40,text="Toggle MV", parent=automation_box, text_style=make_button_text_style())

    def toggle_overlay():
        print("Setting Hot Pixel Map")
        machine_vision_overlay.clear_hot_pixel_map()
        count = machine_vision_overlay.build_hot_pixel_map(include_soft=True)  
        print(f"Marked {count} hot tiles invalid")

    Button(toggle_overlay,x=132, y=60, width=212, height=40, text="MV Hot Pixel Filter", parent=automation_box, text_style=make_button_text_style())






