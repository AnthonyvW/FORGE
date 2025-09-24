from typing import Tuple
import serial
import serial.tools.list_ports
import time
import queue
import threading
import re
from .models import Position
from .printerConfig import (
    PrinterSettings,
    PrinterSettingsManager
)
from forgeConfig import (
    ForgeSettings,
)

class BasePrinterController:
    CONFIG_SUBDIR = "Ender3"
    """Base class for 3D printer control"""
    def __init__(self, forgeConfig: ForgeSettings):
        self.config = PrinterSettings()
        PrinterSettingsManager.scope_dir(self.CONFIG_SUBDIR)

        self.command_queue = queue.Queue()
        self.position = Position(0, 0, 0)
        self.speed = self.config.step_size  # Default speed
        self.paused = False
        
        # Initialize serial connection
        self._initialize_printer(forgeConfig)
        
        # Start command processing thread
        self._processing_thread = threading.Thread(target=self._process_commands, daemon=True)
        self._processing_thread.start()

    def _initialize_printer(self, forgeConfig):
        """Initialize printer serial connection"""
        # First try the configured serial port if it exists
        if forgeConfig.serial_port:
            try:
                print(f"Trying configured port: {forgeConfig.serial_port}")
                test_connection = serial.Serial(
                    forgeConfig.serial_port,
                    self.config.baud_rate,
                    timeout=5  # Longer timeout
                )
                
                # Clear any startup messages first
                time.sleep(2)  # Give the printer time to send initial messages
                test_connection.reset_input_buffer()
                
                # Send firmware info request
                test_connection.write(b"M115\n")
                
                # Read multiple lines with a timeout
                printer_found = False
                valid_responses = ["FIRMWARE_NAME", "Marlin", "Ender", "TF init", "echo:"]
                start_time = time.time()
                response_lines = []
                
                # Try reading for up to 10 seconds (increased from 5)
                while time.time() - start_time < 10:
                    if test_connection.in_waiting > 0:
                        line = test_connection.readline().decode('utf-8', errors='ignore').strip()
                        response_lines.append(line)
                        print(f"Response from {forgeConfig.serial_port}: {line}")
                        
                        # Check for any of our valid printer response indicators
                        for indicator in valid_responses:
                            if indicator in line:
                                printer_found = True
                        
                        # If we got at least some response and enough time has passed, consider it a success
                        if len(response_lines) >= 3 and time.time() - start_time > 3:
                            printer_found = True
                    
                    time.sleep(0.1)  # Small delay between reads
                
                # If we found any printer-like responses, this is probably our printer
                if printer_found:
                    self.printer_serial = test_connection
                    print(f"Printer found on configured port: {forgeConfig.serial_port}")
                    print(f"Responses: {response_lines}")
                    return
                
                # Not the right device, close and continue to port scanning
                test_connection.close()
                print(f"Configured port {forgeConfig.serial_port} did not respond as expected. Scanning all ports...")
                
            except (serial.SerialException, UnicodeDecodeError, Exception) as e:
                print(f"Configured port {forgeConfig.serial_port} failed: {e}")
                print("Falling back to scanning all available ports...")
        
        # Fall back to scanning all available ports
        available_ports = list(serial.tools.list_ports.comports())

        if not available_ports:
            raise RuntimeError("No serial ports found. Is the printer connected?")
        
        print(f"Available ports: {[port.device for port in available_ports]}")
        
        # Try to find the printer by testing each available port
        for port in available_ports:
            try:
                print(f"Trying port: {port.device}")
                test_connection = serial.Serial(
                    port.device,
                    self.config.baud_rate,
                    timeout=5  # Longer timeout
                )
                
                # Clear any startup messages first
                time.sleep(2)  # Give the printer time to send initial messages
                test_connection.reset_input_buffer()
                
                # Send firmware info request
                test_connection.write(b"M115\n")
                
                # Read multiple lines with a timeout
                printer_found = False
                valid_responses = ["FIRMWARE_NAME", "Marlin", "Ender", "TF init", "echo:"]
                start_time = time.time()
                response_lines = []
                
                # Try reading for up to 10 seconds (increased from 5)
                while time.time() - start_time < 10:
                    if test_connection.in_waiting > 0:
                        line = test_connection.readline().decode('utf-8', errors='ignore').strip()
                        response_lines.append(line)
                        print(f"Response from {port.device}: {line}")
                        
                        # Check for any of our valid printer response indicators
                        for indicator in valid_responses:
                            if indicator in line:
                                printer_found = True
                        
                        # If we got at least some response and enough time has passed, consider it a success
                        if len(response_lines) >= 3 and time.time() - start_time > 3:
                            printer_found = True
                    
                    time.sleep(0.1)  # Small delay between reads
                
                # If we found any printer-like responses, this is probably our printer
                if printer_found:
                    self.printer_serial = test_connection
                    print(f"Printer found on port: {port.device}")
                    print(f"Responses: {response_lines}")
                    return
                
                # Not the right device, close and try next
                test_connection.close()
                
            except (serial.SerialException, UnicodeDecodeError, Exception) as e:
                print(f"Port {port.device} failed: {e}")
                continue
        
        # If we get here, we couldn't find the printer
        ports_tried = [p.device for p in available_ports]
        if hasattr(self.config, 'serial_port') and self.serial_port:
            ports_tried.append(self.serial_port + " (from config)")
                
        raise RuntimeError(f"Printer not found on any available serial port. Tried: {ports_tried}")

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
            if response.lower() == 'ok' or response.lower() == 'processing':
                break
            elif response.startswith('error'):
                raise RuntimeError(f"Printer reported error: {response}")
            else:
                print(response)

    def _update_position(self, command: str) -> None:
        """Update internal position tracking based on G-code command"""
        # Handle full homing
        if command.strip().upper() == "G28":
            self.position = Position(x=0, y=0, z=0)
            return

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

        # Wait 2/10 of a second to allow the printer thread to see that its paused
        time.sleep(0.2)
        self.paused = False
        print("Cleared Queue")

    def toggle_pause(self) -> None:
        """Toggle pause state"""
        self.paused = not self.paused
        print("Unpaused" if not self.paused else "Paused")

    def adjust_speed(self, amount: int) -> None:
        """Adjust movement speed"""
        self.speed = max(self.config.step_size, self.speed + amount)  # Prevent negative speed
        print("Current Speed", self.speed / 100)

    def home(self) -> None:
        print("Homing")
        # Home the printer
        self.command_queue.put("G28")

    # Convenience methods for movement
    def move_z_up(self): self.move_axis('z', 1)
    def move_z_down(self): self.move_axis('z', -1)
    def move_x_left(self): self.move_axis('x', 1)
    def move_x_right(self): self.move_axis('x', -1)
    def move_y_backward(self): self.move_axis('y', 1)
    def move_y_forward(self): self.move_axis('y', -1)

    # Convenience methods for speed
    def increase_speed(self): self.adjust_speed(self.config.step_size)
    def decrease_speed(self): self.adjust_speed(-self.config.step_size)
    def increase_speed_fast(self): self.adjust_speed(self.config.step_size * 25)
    def decrease_speed_fast(self): self.adjust_speed(-self.config.step_size * 25)