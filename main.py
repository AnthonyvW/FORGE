import pygame
import time
from typing import List

from UI.button import Button, ButtonShape
from UI.text import Text, TextStyle
from UI.frame import Frame
from camera.amscope import AmscopeCamera
from printer.automated_controller import AutomatedPrinter, Position
from printer.config import PrinterConfig, AutomationConfig

pygame.init()
pygame.display.set_caption("Tree Ring Imaging Machine v2")
width, height = (1920, 1080)
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)

# Frame in which everything is based on
root_frame = Frame(x=0, y=0, width=width, height=height)

# A clock to limit the frame rate.
clock = pygame.time.Clock()

right_panel_width = 400
# Initialize camera with the refactored class
camera = AmscopeCamera(width-right_panel_width, height)

# Initialize printer configurations
printer_config = PrinterConfig()  # Using default values
automation_config = AutomationConfig()  # Using default values

# Initialize the automated printer with configurations
movementSystem = AutomatedPrinter(printer_config, automation_config, camera)

time.sleep(1.5)
camera.resize(width - right_panel_width, height)

# Define button styles
button_style = TextStyle(
    color=pygame.Color(64, 255, 64),  # Matching original foreground color
    font_size=20
)

# Create the control frame (right-aligned, full-height, fixed right_panel_width width)
control_frame = Frame(
    parent=root_frame,
    x=0, y=0,
    width=right_panel_width,
    height=1.0,  # percent-based height to fill root
    height_is_percent=True,
    x_align='right',
    y_align='top',
    background_color=pygame.Color("#b3b4b6")
)
root_frame.add_child(control_frame)

title_bar = Frame(
    parent=control_frame,
    x=0, y=0,
    width=1.0,
    height=50,
    width_is_percent=True,
    background_color=pygame.Color("#909398")
)
control_frame.add_child(title_bar)

title_text = Text(
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
title_bar.add_child(title_text)

# Main box below the title bar
control_box = Frame(
    parent=control_frame,
    x=10, y=60,
    width=right_panel_width-20, height=300,
    background_color=pygame.Color("#ffffff")
)
control_frame.add_child(control_box)

# Header bar inside the control box
header_bar = Frame(
    parent=control_box,
    x=0, y=0,
    width=1.0,
    height=32,
    width_is_percent=True,
    background_color=pygame.Color("#dbdbdb")
)
control_box.add_child(header_bar)

# Panel under header bar to hold movement buttons
movement_button_panel = Frame(
    parent=control_box,
    x=0, y=header_bar.height,
    width=1.0,
    height=control_box.height - header_bar.height,
    width_is_percent=True,
    background_color=pygame.Color("#ffffff")
)
control_box.add_child(movement_button_panel)

speed_display = Text(
    text=f"Speed: {movementSystem.speed / 100:.2f}",
    x=200, y=155,
    x_align="left",
    y_align="top",
    style=TextStyle(
        color=pygame.Color(32, 32, 32),
        font_size=18,
        font_name="assets/fonts/SofiaSans-Regular.ttf"
    )
)
movement_button_panel.add_child(speed_display)

# Text inside the header bar
control_label = Text(
    text="Control",
    x=header_bar.width // 2 + 5, y=header_bar.height // 2,
    x_align="left",
    y_align="center",
    style=TextStyle(
        color=pygame.Color("#7a7a7a"),
        font_size=24,
        font_name="assets/fonts/SofiaSans-Regular.ttf"
    )
)
header_bar.add_child(control_label)

header_bar.add_child(control_label)

# Button positions are now relative to control_frame's local (0, 0)
button_specs = [
    { "fn": movementSystem.move_x_right       , "x": 10,  "y": 55,  "w": 80, "h": 80, "text": "<", "shape": ButtonShape.DIAMOND},
    { "fn": movementSystem.move_x_left        , "x": 100, "y": 55,  "w": 80, "h": 80, "text": ">", "shape": ButtonShape.DIAMOND},
    { "fn": movementSystem.move_y_backward    , "x": 55, "y": 10,  "w": 80, "h": 80, "text": "^" , "shape": ButtonShape.DIAMOND},
    { "fn": movementSystem.move_y_forward     , "x": 55, "y": 100, "w": 80, "h": 80, "text": "v" , "shape": ButtonShape.DIAMOND},
    { "fn": movementSystem.move_z_up          , "x": 200, "y": 55, "w": 40, "h": 40, "text": "+"},
    { "fn": movementSystem.move_z_down        , "x": 200, "y": 105, "w": 40, "h": 40, "text": "-"},
    { "fn": movementSystem.increase_speed     , "x": 250, "y": 55, "w": 40, "h": 40, "text": "S+" },
    { "fn": movementSystem.decrease_speed     , "x": 250, "y": 105, "w": 40, "h": 40, "text": "S-" },
    { "fn": movementSystem.increase_speed_fast, "x": 300, "y": 55, "w": 40, "h": 40, "text": "F+" },
    { "fn": movementSystem.decrease_speed_fast, "x": 300, "y": 105, "w": 40, "h": 40, "text": "F-" },

    { "fn": movementSystem.home               , "x": 10, "y": 10, "w": 80, "h": 40, "text": "Home" },
    { "fn": movementSystem.toggle_pause       , "x": 100, "y": 10, "w": 80, "h": 40, "text": "Pause" },
    { "fn": movementSystem.halt               , "x": 190, "y": 10, "w": 80, "h": 40, "text": "Stop" },

    { "fn": movementSystem.start_automation   , "x": 110, "y": 250, "w": 80, "h": 40, "text": "Start" },
    { "fn": movementSystem.setPosition1       , "x": 110, "y": 150, "w": 120, "h": 40, "text": "Set Position 1" },
    { "fn": movementSystem.setPosition2       , "x": 250, "y": 150, "w": 120, "h": 40, "text": "Set Position 2" },
    {
        "fn": lambda pos: camera.capture_image() or camera.save_image(filename=pos.to_gcode()),
        "x": 110, "y": 350, "w": 80, "h": 40,
        "text": "Take Photo",
        "args_provider": lambda: (movementSystem.get_position(),)
    }
]

# Create and add buttons to the control_frame
# Categorize and add buttons accordingly
buttons: List[Button] = []
for spec in button_specs:
    btn = Button(
        function_to_call=spec["fn"],
        x=spec["x"], y=spec["y"],
        width=spec["w"], height=spec["h"],
        text=spec["text"],
        text_style=button_style,
        args_provider=spec.get("args_provider"),
        shape=spec.get("shape", ButtonShape.RECTANGLE)
    )

    # Movement-related functions go inside control_box
    if spec["fn"] in {
        movementSystem.move_x_right,
        movementSystem.move_x_left,
        movementSystem.move_y_forward,
        movementSystem.move_y_backward,
        movementSystem.move_z_up,
        movementSystem.move_z_down,
        movementSystem.increase_speed,
        movementSystem.decrease_speed,
        movementSystem.increase_speed_fast,
        movementSystem.decrease_speed_fast
    }:
        movement_button_panel.add_child(btn)
    elif spec["fn"] in {
        movementSystem.home,
        movementSystem.toggle_pause,
        movementSystem.halt
    }:
        # Align to bottom of movement_button_panel, 10px from bottom
        btn.y_align = 'bottom'
        movement_button_panel.add_child(btn)
    else:
        # Default non-movement buttons go below the control box
        btn.y += control_box.height + 20
        control_frame.add_child(btn)

    buttons.append(btn)


running = True
while running:
    clock.tick(60)
    # Mouse Position
    pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.VIDEORESIZE:
            new_width, new_height = event.w, event.h

            width, height = new_width, new_height

            root_frame.width = new_width
            root_frame.height = new_height
            camera.resize(width - right_panel_width, height)

            print(width, height)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            root_frame.handle_click(*pos)
            root_frame.process_mouse_press(*pos, button="left")
        elif event.type == pygame.KEYDOWN:
            root_frame.process_mouse_release(*pos, button="left")
            if event.key == pygame.K_ESCAPE:
                running = False
        elif event.type == pygame.KEYUP:
            background_color = pygame.Color(255, 0, 0)
            screen.fill(background_color)

    root_frame.process_mouse_move(*pos)

    # Rendering
    screen.fill([60, 60, 60])

    def draw_debug_outline(surface, frame):
        x, y, w, h = frame.get_absolute_geometry()
        color = frame.debug_outline_color
        pygame.draw.rect(surface, color, pygame.Rect(x, y, w, h), 2)

        for child in frame.children:
            draw_debug_outline(surface, child)

    # Draw Buttons
    control_frame.draw(screen)

    #draw_debug_outline(screen, root_frame)
    # Draw Camera
    try:
        frame = camera.get_frame()
        if frame is not None:
            screen.blit(frame, (0, 0))
    except Exception as e:
        print(f"Error displaying camera frame: {e}")

    speed_display.set_text(f"Step Size: {movementSystem.speed / 100:.2f}mm")
    pygame.display.flip()

# Ensure camera is properly closed
camera.close()
pygame.quit()