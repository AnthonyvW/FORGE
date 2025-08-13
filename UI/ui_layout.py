from dataclasses import dataclass
from typing import List, Tuple

import pygame
from UI.button import Button, ButtonShape
from UI.text import Text, TextStyle
from UI.frame import Frame
from UI.section_frame import Section
from UI.text_field import TextField

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


    # --- Control Box ---
    control_box = Section(parent=control_frame, title="Control", 
        x=10, y=60, width=RIGHT_PANEL_WIDTH - 20, height=250)
    speed_display, position_display = _build_movement_controls(control_box, movementSystem)

    # --- Automation Box ---
    automation_box = Section(parent=control_frame, title= "Automation", 
        x=10, y=control_box.y + control_box.height + box_spacing, width = RIGHT_PANEL_WIDTH - 20, height = 90)
    _build_automation_control(automation_box, movementSystem)

    # --- Camera Settings ---
    camera_control = Section(parent=control_frame, title="Camera Control",
        x=10,y=automation_box.y + automation_box.height + box_spacing, width = RIGHT_PANEL_WIDTH - 20, height = 163)
    _build_camera_control(camera_control, camera)

    # --- Sample Box ---
    sample_box = Section(parent=control_frame, title="Sample Management", 
        x=10, y=camera_control.y + camera_control.height + box_spacing, width = RIGHT_PANEL_WIDTH - 20, height = 270)
    go_to_sample_button, decrement_button, increment_button, sample_label, pos1_display, pos2_display, sample_name_field = _build_sample_box(sample_box, movementSystem, current_sample_index)
  
    return (
        control_frame,
        sample_label,
        increment_button,
        decrement_button,
        go_to_sample_button,
        speed_display,
        position_display,
        pos1_display,
        pos2_display,
        sample_name_field
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


def _build_sample_box(sample_box, movementSystem, current_sample_index):
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
    sample_name_field = TextField(parent=sample_box, x=170, y=160, width=200, height=30, placeholder="sample", border_color=pygame.Color("#b3b4b6"), text_color=pygame.Color("#5a5a5a"))

    return go_to_sample_button, decrement_button, increment_button, sample_label, pos1_display, pos2_display, sample_name_field


def _build_camera_control(camera_control, camera):
    camera_control.add_child(make_button(
        lambda pos: camera.capture_image() or camera.save_image(filename=pos.to_gcode()),
        10, 10, 120, 40, "Take Photo",
        args_provider=lambda: (movementSystem.get_position(),)
    ))

    path_label = Text(f"Save Path: {camera.capture_path}", parent=camera_control, 
        x=10, y=60, x_align="left", y_align="top", style=make_display_text_style(), truncate_mode="middle", max_width=RIGHT_PANEL_WIDTH - 20 - 20)
    
    def on_set_path():
        path_label.set_text(f"Save Path: {camera.select_capture_path()}")

    Button(on_set_path, 135,  10, 120, 40, "Set Path", parent=camera_control, text_style=make_button_text_style())
    

def _build_automation_control(automation_box, movementSystem):
    
    Button(movementSystem.start_automation, 10,  10, 115, 40, "Start", parent=automation_box, text_style=make_button_text_style())
    Button(movementSystem.halt,             133, 10, 115, 40, "Stop" , parent=automation_box, text_style=make_button_text_style())
    pause = Button(movementSystem.toggle_pause,     255, 10, 115, 40, "Pause", parent=automation_box, text_style=make_button_text_style())
    pause.is_hidden = True


