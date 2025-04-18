import pygame
import time
from typing import List

from UI.Button import Button
from UI.text import TextStyle
from camera.amscope import AmscopeCamera
from printer.automated_controller import AutomatedPrinter, Position
from printer.config import PrinterConfig, AutomationConfig

pygame.init()
pygame.display.set_caption("Tree Ring Imaging Machine v2")
width, height = (1280, 720)
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)

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

# Create buttons with labels
buttons: List[Button] = [
    Button(movementSystem.move_x_right, width - 400, 500, 40, 40, text="→", text_style=button_style),
    Button(movementSystem.move_x_left, width - 300, 500, 40, 40, text="←", text_style=button_style),
    Button(movementSystem.move_y_backward, width - 350, 450, 40, 40, text="↑", text_style=button_style),
    Button(movementSystem.move_y_forward, width - 350, 550, 40, 40, text="↓", text_style=button_style),
    Button(movementSystem.move_z_up, width - 250, 475, 40, 40, text="+", text_style=button_style),
    Button(movementSystem.move_z_down, width - 250, 525, 40, 40, text="-", text_style=button_style),
    Button(movementSystem.increase_speed, width - 200, 475, 40, 40, text="S+", text_style=button_style),
    Button(movementSystem.decrease_speed, width - 200, 525, 40, 40, text="S-", text_style=button_style),
    Button(movementSystem.increase_speed_fast, width - 150, 475, 40, 40, text="F+", text_style=button_style),
    Button(movementSystem.decrease_speed_fast, width - 150, 525, 40, 40, text="F-", text_style=button_style),
    Button(movementSystem.toggle_pause, width - 250, 350, 40, 40, text="⏸", text_style=button_style),
    Button(movementSystem.start_automation, width - 350, 250, 40, 40, text="▶", text_style=button_style),
    Button(movementSystem.halt, width - 150, 350, 40, 40, text="⏹", text_style=button_style),
    Button(lambda pos: camera.capture_image() or camera.save_image(filename=movementSystem.get_position().to_gcode()), width - 350, 350, 40, 40, text="📷", text_style=button_style),
]

running = True
while running:
    clock.tick(60)
    # Mouse Position
    pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.VIDEORESIZE:
            for button in buttons:
                button.update_position(screen.get_size()[0] - width, screen.get_size()[1] - height)

            width, height = screen.get_size()
            camera.resize(width-500, height)

            print(width, height)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            for i, button in enumerate(buttons):
                if i < len(buttons) - 1:
                    button.check_button(pos[0], pos[1], True)
                else:
                    # Last button (camera capture) needs position argument
                    button.check_button(pos[0], pos[1], True, movementSystem.get_position())
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
        elif event.type == pygame.KEYUP:
            background_color = pygame.Color(255, 0, 0)
            screen.fill(background_color)

    # Update
    try:
        camera.update()
    except Exception as e:
        print(f"Error updating camera: {e}")

    for button in buttons:
        button.check_button(pos[0], pos[1], False)

    # Rendering
    screen.fill([60, 60, 60])

    # Draw Buttons
    for button in buttons:
        button.draw(screen)

    # Draw Camera
    try:
        frame = camera.get_frame()
        if frame is not None:
            screen.blit(frame, (0, 0))
    except Exception as e:
        print(f"Error displaying camera frame: {e}")

    pygame.display.flip()
    pygame.display.update()

# Ensure camera is properly closed
camera.close()
pygame.quit()