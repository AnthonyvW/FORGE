import serial
import time

# Connect to the 3D printer
port = 'COM9'
baudrate = 115200  # common Marlin baudrate
timeout = 2

with serial.Serial(port, baudrate, timeout=timeout) as ser:
    # Wait for the printer to initialize
    time.sleep(2)
    ser.reset_input_buffer()

    def send(cmd):
        """Send a G-code command and wait for 'ok' response."""
        ser.write((cmd + '\n').encode())
        ser.flush()
        while True:
            line = ser.readline().decode(errors='ignore').strip()
            if line:
                print(">>", line)
                if line.lower().startswith('ok'):
                    break

    # Set steps/mm for Z axis to 800
    send("M92 Z800")

    # Save settings to EEPROM
    send("M500")

    # Optional: verify settings
    send("M503")

print("Z steps/mm set to 800 and saved to EEPROM.")
