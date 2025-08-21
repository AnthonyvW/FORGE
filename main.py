import pygame
import time
from typing import List

from camera.amscope import AmscopeCamera
from printer.automated_controller import AutomatedPrinter, Position
from printer.config import PrinterConfig, AutomationConfig

from UI.text import Text, TextStyle
from UI.frame import Frame
from UI.ui_layout import create_control_panel, RIGHT_PANEL_WIDTH

from UI.input.text_field import TextField
from UI.input.button import Button, ButtonShape

pygame.init()
pygame.display.set_caption("FORGE")
width, height = (1920, 1080)
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)

# Frame in which everything is based on
root_frame = Frame(x=0, y=0, width=width, height=height)

# A clock to limit the frame rate.
clock = pygame.time.Clock()

right_panel_width = RIGHT_PANEL_WIDTH
# Initialize camera with the refactored class
camera = AmscopeCamera(width - right_panel_width, height)

# Initialize printer configurations
printer_config = PrinterConfig()  # Using default values
automation_config = AutomationConfig()  # Using default values

# Initialize the automated printer with configurations
movementSystem = AutomatedPrinter(printer_config, automation_config, camera)

time.sleep(1.5)
camera.resize(width - right_panel_width, height)

current_sample_index = 1

def get_sample_position(index: int) -> Position:
    return Position(
        x=int((20 + 11 * (index - 1)) * 100),
        y=int(210 * 100),
        z=int(9.4 * 100)
    )

(
    sample_label,
    inc_btn,
    dec_btn,
    go_btn,
    speed_display,
    position_display,
    position1_display,
    position2_display
) = create_control_panel(root_frame, movementSystem, camera, current_sample_index)

# Verify no duplicate nodes are present
def audit_tree(node):
    seen = {}
    for ch in node.children:
        seen.setdefault(id(ch), []).append(ch)
    for ids, lst in seen.items():
        if len(lst) > 1:
            print(f"[DUP] {node.__class__.__name__} id={id(node)} has child repeated x{len(lst)} -> {lst[0].__class__.__name__} id={id(lst[0])}")
    for ch in node.children:
        audit_tree(ch)

audit_tree(root_frame)


def go_to_sample():
    pos = get_sample_position(current_sample_index)
    movementSystem.move_to_position(pos)

def increment_sample():
    global current_sample_index
    if current_sample_index < 20:
        current_sample_index += 1
        sample_label.set_text(f"Sample {current_sample_index}")

def decrement_sample():
    global current_sample_index
    if current_sample_index > 1:
        current_sample_index -= 1
        sample_label.set_text(f"Sample {current_sample_index}")


inc_btn.function_to_call = increment_sample
dec_btn.function_to_call = decrement_sample
go_btn.function_to_call  = go_to_sample





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
        elif event.type == pygame.MOUSEBUTTONUP:
            root_frame.process_mouse_release(*pos, button="left")
        elif event.type == pygame.MOUSEBUTTONDOWN:
            root_frame.broadcast_mouse_press(*pos, button="left")
            root_frame.process_mouse_press(*pos, button="left")
        elif event.type == pygame.KEYDOWN:
            root_frame.broadcast_key_event(event)
            if event.key == pygame.K_ESCAPE:
                running = False
        elif event.type == pygame.KEYUP:
            root_frame.broadcast_key_event(event)
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

    #draw_debug_outline(screen, root_frame)
    
    # Draw GUI
    root_frame.draw(screen)

    speed_display.set_text(f"Step Size: {movementSystem.speed / 100:.2f}mm")
    position_display.set_text( f"X: {movementSystem.position.x/100:.2f} Y: {movementSystem.position.y/100:.2f} Z: {movementSystem.position.z/100:.2f}")
    position1_display.set_text(f"X: {movementSystem.automation_config.x_start/100:.2f} Y: {movementSystem.automation_config.y_start/100:.2f} Z: {movementSystem.automation_config.z_start/100:.2f}")
    position2_display.set_text(f"X: {movementSystem.automation_config.x_end/100:.2f} Y: {movementSystem.automation_config.y_end/100:.2f} Z: {movementSystem.automation_config.z_end/100:.2f}")
    pygame.display.flip()
        
# Ensure camera is properly closed
camera.close()
pygame.quit()