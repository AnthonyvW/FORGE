import serial
import time

printer_serial = serial.Serial('COM6', 115200)
time.sleep(1)
while True:
    input_user = input('Enter Command : ')
    command = input_user + "\n"
    printer_serial.write(command.encode())
    response = printer_serial.readline().decode().strip()
    print(response)

    while response.lower() != 'ok':
        response = printer_serial.readline().decode().strip()
        print(response)
