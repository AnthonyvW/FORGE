import serial
import time
import queue
import threading

class printer():
    commandQueue = queue.Queue()

    X = 0
    Y = 0
    Z = 0
    printerSerial = None

    def __init__(self):
        self.commandQueue.put("G1 Z0")
        self.printerSerial = serial.Serial('COM3', 115200) #250000
        threading.Thread(target=self.startThread, daemon=True).start()

    def startThread(self):
        time.sleep(1)
        while(True):
            self.sendCommand(self.commandQueue.get())
            response = self.printerSerial.readline().decode().strip()
            print(response)
            while response.lower() != 'ok':
                response = self.printerSerial.readline().decode().strip()
                print(response)
        
    def sendCommand(self, command):
        print((command + "\n"))
        self.printerSerial.write((command + "\n").encode())

    def moveZUp(self):
        self.Z += 0.08
        self.commandQueue.put(f"G1 Z{self.Z + 0.08}")

    def moveZDown(self):
        self.Z -= 0.04
        self.commandQueue.put(f"G1 Z{self.Z - 0.04}")

    def moveXLeft(self):
        self.X += 0.04
        self.commandQueue.put(f"G1 X{self.X + 0.04}")
        
    def moveXRight(self):
        self.X -= 0.04
        self.commandQueue.put(f"G1 X{self.X - 0.04}")

    def moveYBackward(self):
        self.Y += 0.04
        self.commandQueue.put(f"G1 Y{self.Y + 0.04}")
        
    def moveYForward(self):
        self.Y -= 0.04
        self.commandQueue.put(f"G1 Y{self.Y - 0.04}")