from typing import Tuple
import serial
import time
import queue
import threading
import re
from .models import Position
from .config import PrinterConfig

class BasePrinterController:
    """Base class for 3D printer control"""
    def __init__(self, config: PrinterConfig):
        self.config = config
        self.command_queue = queue.Queue()
        self.position = Position(0, 0, 0)
        self.speed = 4  # Default speed
        self.paused = False
        
        # Initialize serial connection
        self._initialize_printer()
        
        # Start command processing thread
        self._processing_thread = threading.Thread(target=self._process_commands, daemon=True)
        self._processing_thread.start()

    def _initialize_printer(self):
        """Initialize printer serial connection"""
        try:
            self.printer_serial = serial.Serial(
                self.config.serial_port, 
                self.config.baud_rate
            )
            # Home all axes
            # self.command_queue.put("G28")
            # Move to safe Z height
            # self.command_queue.put("G0 Z30")
        except serial.SerialException as e:
            raise RuntimeError(f"Failed to initialize printer: {e}")

    def _process_commands(self):
        """Main thread for processing commands from the queue"""
        time.sleep(1)  # Allow time for serial connection to stabilize
        while True:
            if not self.paused:
                try:
                    command = self.command_queue.get()
                    if command.startswith("G"):
                        self._update_position(command)
                        self._send_and_wait(command)
                except Exception as e:
                    print(f"Error in command processing: {e}")
                    self.halt()

    def _send_and_wait(self, command: str) -> None:
        """Send command and wait for OK response"""
        self._send_command(command)
        self._wait_for_ok()

    def _wait_for_ok(self) -> None:
        """Wait for 'ok' response from printer"""
        while True:
            response = self.printer_serial.readline().decode().strip()
            if response.lower() == 'ok':
                break
            elif response.startswith('error'):
                raise RuntimeError(f"Printer reported error: {response}")

    def _update_position(self, command: str) -> None:
        """Update internal position tracking based on G-code command"""
        updates = {}
        for axis, pattern in [('x', r'X([\d\.]+)'), ('y', r'Y([\d\.]+)'), ('z', r'Z([\d\.]+)')]:
            match = re.search(pattern, command)
            if match:
                updates[axis] = int(float(match.group(1)) * 100)
        
        if updates:
            self.position = Position(
                x=updates.get('x', self.position.x),
                y=updates.get('y', self.position.y),
                z=updates.get('z', self.position.z)
            )

    def _send_command(self, command: str) -> None:
        """Send command to printer"""
        if not command.startswith("G0") and command != "M400":
            print(command)
        self.printer_serial.write(f"{command}\n".encode())

    def get_position(self) -> Position:
        """Get current position as tuple"""
        return self.position

    def move_to_position(self, position: Position) -> None:
        """Move to specified position"""
        self.command_queue.put(f"G0 {position.to_gcode()}")

    def move_axis(self, axis: str, direction: int) -> None:
        """Move specified axis by current speed * direction"""
        current_value = getattr(self.position, axis)
        new_value = current_value + (self.speed * direction)
        
        # Check bounds
        max_value = getattr(self.config, f"max_{axis}")
        if 0 <= new_value <= max_value:
            # Create and send the G-code command
            command = f"G1 {axis.upper()}{new_value / 100}"
            self.command_queue.put(command)
            # Position will be updated by _update_position when command is processed

    def halt(self) -> None:
        """Halt all operations and clear queue"""
        self.paused = True
        while not self.command_queue.empty():
            self.command_queue.get(False)
        self.paused = False
        print("Cleared Queue")

    def toggle_pause(self) -> None:
        """Toggle pause state"""
        self.paused = not self.paused
        print("Unpaused" if not self.paused else "Paused")

    def adjust_speed(self, amount: int) -> None:
        """Adjust movement speed"""
        self.speed = max(1, self.speed + amount)  # Prevent negative speed
        print("Current Speed", self.speed / 100)

    # Convenience methods for movement
    def move_z_up(self): self.move_axis('z', 1)
    def move_z_down(self): self.move_axis('z', -1)
    def move_x_left(self): self.move_axis('x', 1)
    def move_x_right(self): self.move_axis('x', -1)
    def move_y_backward(self): self.move_axis('y', 1)
    def move_y_forward(self): self.move_axis('y', -1)

    # Convenience methods for speed
    def increase_speed(self): self.adjust_speed(1)
    def decrease_speed(self): self.adjust_speed(-1)
    def increase_speed_fast(self): self.adjust_speed(100)
    def decrease_speed_fast(self): self.adjust_speed(-100)