import pygame
import time
from typing import List

from UI.button import Button
from UI.text import TextStyle
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

# Initialize camera with the refactored class
camera = AmscopeCamera(width-500, height)

# Initialize printer configurations
printer_config = PrinterConfig()  # Using default values
automation_config = AutomationConfig()  # Using default values

# Initialize the automated printer with configurations
movementSystem = AutomatedPrinter(printer_config, automation_config, camera)

time.sleep(1.5)
camera.resize(width - 500, height)

# Define button styles
button_style = TextStyle(
    color=pygame.Color(64, 255, 64),  # Matching original foreground color
    font_size=20
)

# Create the control frame (right-aligned, full-height, fixed 500px width)
control_frame = Frame(
    parent=root_frame,
    x=0, y=0,
    width=500,
    height=1.0,  # percent-based height to fill root
    height_is_percent=True,
    x_align='right',
    y_align='top'
)
root_frame.add_child(control_frame)

# Button positions are now relative to control_frame's local (0, 0)
button_specs = [
    { "fn": movementSystem.move_x_right       , "x": 100, "y": 500, "w": 40, "h": 40, "text": "<"  },
    { "fn": movementSystem.move_x_left        , "x": 200, "y": 500, "w": 40, "h": 40, "text": ">"  },
    { "fn": movementSystem.move_y_backward    , "x": 150, "y": 450, "w": 40, "h": 40, "text": "^"  },
    { "fn": movementSystem.move_y_forward     , "x": 150, "y": 550, "w": 40, "h": 40, "text": "v"  },
    { "fn": movementSystem.move_z_up          , "x": 300, "y": 475, "w": 40, "h": 40, "text": "+"  },
    { "fn": movementSystem.move_z_down        , "x": 300, "y": 525, "w": 40, "h": 40, "text": "-"  },
    { "fn": movementSystem.increase_speed     , "x": 350, "y": 475, "w": 40, "h": 40, "text": "S+" },
    { "fn": movementSystem.decrease_speed     , "x": 350, "y": 525, "w": 40, "h": 40, "text": "S-" },
    { "fn": movementSystem.increase_speed_fast, "x": 400, "y": 475, "w": 40, "h": 40, "text": "F+" },
    { "fn": movementSystem.decrease_speed_fast, "x": 400, "y": 525, "w": 40, "h": 40, "text": "F-" },
    { "fn": movementSystem.toggle_pause       , "x": 210, "y": 350, "w": 80, "h": 40, "text": "Pause" },
    { "fn": movementSystem.start_automation   , "x": 110, "y": 250, "w": 80, "h": 40, "text": "Start" },
    { "fn": movementSystem.home               , "x": 210, "y": 250, "w": 80, "h": 40, "text": "Home" },
    { "fn": movementSystem.halt               , "x": 310, "y": 350, "w": 80, "h": 40, "text": "Stop" },
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
buttons: List[Button] = []
for spec in button_specs:
    btn = Button(
        function_to_call=spec["fn"],
        x=spec["x"], y=spec["y"],
        width=spec["w"], height=spec["h"],
        text=spec["text"],
        text_style=button_style,
        args_provider=spec.get("args_provider")
    )
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
            camera.resize(width - 500, height)

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
        pygame.draw.rect(surface, (255, 0, 0), pygame.Rect(x, y, w, h), 2)
        for child in frame.children:
            draw_debug_outline(surface, child)

    # Draw Buttons
    for button in buttons:
        button.draw(screen)

    #draw_debug_outline(screen, root_frame)
    # Draw Camera
    try:
        frame = camera.get_frame()
        if frame is not None:
            screen.blit(frame, (0, 0))
    except Exception as e:
        print(f"Error displaying camera frame: {e}")

    pygame.display.flip()

# Ensure camera is properly closed
camera.close()
pygame.quit()