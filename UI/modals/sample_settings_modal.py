"""
Sample Settings Modal - Manual configuration of sample positions for the printer.
"""

import pygame
from typing import List, Tuple, Callable

from UI.frame import Frame
from UI.modal import Modal
from UI.text import Text, TextStyle
from UI.input.button import Button, ButtonColors
from UI.input.text_field import TextField
from UI.input.scroll_frame import ScrollFrame
from UI.list_frame import ListFrame
from UI.styles import make_button_text_style, make_display_text_style

from printer.automated_controller import AutomatedPrinter


def build_sample_settings_modal(modal: Modal, controller: AutomatedPrinter) -> None:
    """
    Build the sample settings modal UI for configuring sample positions.
    
    Args:
        modal: The Modal frame to populate
        controller: The printer controller instance
    """
    
    # Container for all settings
    content = modal.body
    
    y_offset = 10
    
    # ========== Camera Height Section ==========
    Text(
        text="Camera Height (mm):",
        parent=content,
        x=10, y=y_offset,
        style=make_display_text_style(16)
    )
    
    camera_height_field = TextField(
        parent=content,
        x=10, y=y_offset + 25,
        width=200, height=30,
        placeholder="0.00",
        border_color=pygame.Color("#b3b4b6"),
        text_color=pygame.Color("#5a5a5a")
    )
    
    def set_camera_height_from_position():
        """Set camera height field to current Z position"""
        z_mm = controller.position.z / 100.0
        camera_height_field.set_text(f"{z_mm:.2f}")
    
    Button(
        set_camera_height_from_position,
        x=220, y=y_offset + 25,
        width=175, height=30,
        text="Set from Current",
        parent=content,
        text_style=make_button_text_style()
    )
    
    # Note about camera height
    Text(
        text="Must be above and out of focus of all samples",
        parent=content,
        x=10, y=y_offset + 60,
        style=TextStyle(
            font_size=12,
            color=pygame.Color("#7a7a7a"),
            font_name="assets/fonts/SofiaSans-Regular.ttf"
        )
    )
    
    y_offset += 90
    
    # ========== Y Start Offset Section ==========
    Text(
        text="Y Start Offset (mm):",
        parent=content,
        x=10, y=y_offset,
        style=make_display_text_style(16)
    )
    
    y_start_field = TextField(
        parent=content,
        x=10, y=y_offset + 25,
        width=200, height=30,
        placeholder="0.00",
        border_color=pygame.Color("#b3b4b6"),
        text_color=pygame.Color("#5a5a5a")
    )
    
    def set_y_start_from_position():
        """Set Y start field to current Y position"""
        y_mm = controller.position.y / 100.0
        y_start_field.set_text(f"{y_mm:.2f}")
    
    Button(
        set_y_start_from_position,
        x=220, y=y_offset + 25,
        width=175, height=30,
        text="Set from Current",
        parent=content,
        text_style=make_button_text_style()
    )
    
    y_offset += 70
    
    # ========== Calibration Y Position Section ==========
    Text(
        text="Calibration Y Position (mm):",
        parent=content,
        x=10, y=y_offset,
        style=make_display_text_style(16)
    )
    
    calibration_y_field = TextField(
        parent=content,
        x=10, y=y_offset + 25,
        width=200, height=30,
        placeholder="220.00",
        border_color=pygame.Color("#b3b4b6"),
        text_color=pygame.Color("#5a5a5a")
    )
    
    def set_calibration_y_from_position():
        """Set calibration Y field to current Y position"""
        y_mm = controller.position.y / 100.0
        calibration_y_field.set_text(f"{y_mm:.2f}")
    
    Button(
        set_calibration_y_from_position,
        x=220, y=y_offset + 25,
        width=175, height=30,
        text="Set from Current",
        parent=content,
        text_style=make_button_text_style()
    )
    
    y_offset += 70
    
    # ========== Calibration Z Position Section ==========
    Text(
        text="Calibration Z Position (mm):",
        parent=content,
        x=10, y=y_offset,
        style=make_display_text_style(16)
    )
    
    calibration_z_field = TextField(
        parent=content,
        x=10, y=y_offset + 25,
        width=200, height=30,
        placeholder="26.00",
        border_color=pygame.Color("#b3b4b6"),
        text_color=pygame.Color("#5a5a5a")
    )
    
    def set_calibration_z_from_position():
        """Set calibration Z field to current Z position"""
        z_mm = controller.position.z / 100.0
        calibration_z_field.set_text(f"{z_mm:.2f}")
    
    Button(
        set_calibration_z_from_position,
        x=220, y=y_offset + 25,
        width=175, height=30,
        text="Set from Current",
        parent=content,
        text_style=make_button_text_style()
    )
    
    y_offset += 70
    
    # ========== Sample X Offsets Section ==========
    Text(
        text="Sample X Offsets:",
        parent=content,
        x=10, y=y_offset,
        style=make_display_text_style(16)
    )
    
    y_offset += 30
    
    # Scroll frame for sample list
    scroll_height = 340
    scroll_area = ScrollFrame(
        parent=content,
        x=10, y=y_offset,
        width=445, height=scroll_height,
        background_color=pygame.Color("#f5f5f5"),
        scrollbar_width=12
    )
    
    # Store references to sample fields (index, field) tuples
    sample_fields: List[Tuple[int, TextField]] = []
    
    def build_sample_row(i: int, parent: Frame) -> None:
        """Build a row for each sample's X offset"""
        sample_num = i + 1  # Display number (1-based)
        sample_index = i + 1  # Config key (also 1-based)
        
        # Sample label
        Text(
            text=f"Sample {sample_num}:",
            parent=parent,
            x=5, y=15,
            style=make_display_text_style(14)
        )
        
        # Go To button (C)
        def go_to_sample():
            """Move to this sample's X position using calibration Y and Z"""
            try:
                x_mm = float(x_field.text or "0.0")
                cal_y_mm = float(calibration_y_field.text or "220.0")
                cal_z_mm = float(calibration_z_field.text or "26.0")
                
                print(f"Moving to Sample {sample_num} calibration position: X={x_mm:.2f}, Y={cal_y_mm:.2f}, Z={cal_z_mm:.2f}")
                
                # Convert to ticks (0.01 mm units)
                x_ticks = int(x_mm * 100)
                y_ticks = int(cal_y_mm * 100)
                z_ticks = int(cal_z_mm * 100)
                
                from printer.models import Position
                target_pos = Position(x=x_ticks, y=y_ticks, z=z_ticks)
                controller.move_to_position(target_pos)
                
            except ValueError as e:
                print(f"Error parsing position values: {e}")
        
        Button(
            go_to_sample,
            x=100, y=5,
            width=30, height=30,
            text="C",
            parent=parent,
            text_style=TextStyle(font_size=14, color=pygame.Color("#5a5a5a"))
        )
        
        # X offset field
        x_field = TextField(
            parent=parent,
            x=135, y=5,
            width=115, height=30,
            placeholder="0.00",
            border_color=pygame.Color("#b3b4b6"),
            text_color=pygame.Color("#5a5a5a")
        )
        sample_fields.append((sample_index, x_field))  # Store with config index
        
        # Set from current button (only sets X)
        def set_x_from_current():
            x_mm = controller.position.x / 100.0
            x_field.set_text(f"{x_mm:.2f}")
            print(f"Sample {sample_num}: Set X offset to {x_mm:.2f} mm from current position")
        
        Button(
            set_x_from_current,
            x=260, y=5,
            width=160, height=30,
            text="Set X from Current",
            parent=parent,
            text_style=TextStyle(font_size=14, color=pygame.Color("#5a5a5a"))
        )
    
    # Create list of sample rows
    num_samples = controller.get_num_slots()
    sample_list = ListFrame(
        parent=scroll_area,
        x=0, y=0,
        width=1.0, height=num_samples * 35,
        width_is_percent=True,
        row_height=35,
        count=num_samples,
        row_builder=build_sample_row
    )
    
    y_offset += scroll_height + 10
    
    # ========== Load current values ==========
    def load_values_from_config():
        """Load current values from printer config"""
        try:
            # Load calibration positions from proper fields
            calibration_y = getattr(controller.config, 'calibration_y', 220.0)
            calibration_z = getattr(controller.config, 'calibration_z', 26.0)
            
            calibration_y_field.set_text(f"{calibration_y:.2f}")
            calibration_z_field.set_text(f"{calibration_z:.2f}")
            
            # Check if sample_positions exists and is a dict
            if not hasattr(controller.config, 'sample_positions'):
                print("No sample_positions found in config")
                return
            
            if not isinstance(controller.config.sample_positions, dict):
                print(f"sample_positions is not a dict: {type(controller.config.sample_positions)}")
                return
            
            if len(controller.config.sample_positions) == 0:
                print("sample_positions dict is empty")
                return
            
            # Load camera height (from first sample's Z, assuming all share this)
            # sample_positions is a dict with 1-based integer keys (1, 2, 3, ...)
            if 1 in controller.config.sample_positions:
                first_sample = controller.config.sample_positions[1]
                if isinstance(first_sample, dict):
                    camera_z = first_sample.get("z", 0.0)
                    camera_height_field.set_text(f"{camera_z:.2f}")
                    
                    # Load Y start (from first sample's Y)
                    y_start = first_sample.get("y", 0.0)
                    y_start_field.set_text(f"{y_start:.2f}")
                else:
                    print(f"First sample is not a dict: {type(first_sample)}")
                    return
            
            # Load each sample's X offset
            for sample_index, field in sample_fields:
                if sample_index in controller.config.sample_positions:
                    sample_pos = controller.config.sample_positions[sample_index]
                    if isinstance(sample_pos, dict):
                        x_offset = sample_pos.get("x", 0.0)
                        field.set_text(f"{x_offset:.2f}")
                    else:
                        print(f"Sample {sample_index} is not a dict: {type(sample_pos)}")
        except Exception as e:
            import traceback
            print(f"Error loading sample settings: {e}")
            print(traceback.format_exc())
    
    # ========== Bottom Buttons ==========
    button_y = y_offset + 5
    
    def save_settings():
        """Save settings to printer config"""
        try:
            # Parse camera height, Y start, and calibration positions
            camera_z = float(camera_height_field.text or "0.0")
            y_start = float(y_start_field.text or "0.0")
            calibration_y = float(calibration_y_field.text or "220.0")
            calibration_z = float(calibration_z_field.text or "26.0")
            
            # Save calibration positions to proper fields
            controller.config.calibration_y = calibration_y
            controller.config.calibration_z = calibration_z
            
            # Create a fresh sample_positions dict to prevent accumulation of extra entries
            new_sample_positions = {}
            
            # Update each sample position from the UI fields
            for sample_index, field in sample_fields:
                x_offset = float(field.text or "0.0")
                
                # Create the sample position entry
                new_sample_positions[sample_index] = {
                    "x": x_offset,
                    "y": y_start,
                    "z": camera_z
                }
            
            # Replace the entire sample_positions dict with our clean version
            controller.config.sample_positions = new_sample_positions
            
            # Save the config
            from printer.printerConfig import PrinterSettingsManager
            PrinterSettingsManager.save(controller.CONFIG_SUBDIR, controller.config)
            
            print(f"Sample settings saved successfully ({len(new_sample_positions)} sample positions)")
            modal.close()
            
        except ValueError as e:
            print(f"Error parsing values: {e}")
        except Exception as e:
            import traceback
            print(f"Error saving sample settings: {e}")
            print(traceback.format_exc())
    
    def reset_settings():
        """Reload settings from printer config"""
        try:
            # Reload the config from disk
            from printer.printerConfig import PrinterSettingsManager
            controller.config = PrinterSettingsManager.load(controller.CONFIG_SUBDIR)
            
            # Update UI fields
            load_values_from_config()
            
            print("Sample settings reloaded from config")
            
        except Exception as e:
            print(f"Error reloading sample settings: {e}")
    
    Button(
        save_settings,
        x=10, y=button_y,
        width=150, height=40,
        text="Save",
        parent=content,
        text_style=make_button_text_style()
    )
    
    Button(
        reset_settings,
        x=170, y=button_y,
        width=150, height=40,
        text="Reset",
        parent=content,
        text_style=make_button_text_style()
    )
    
    # Load initial values when modal is built
    load_values_from_config()