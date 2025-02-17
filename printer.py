import serial
import time
import queue
import threading
import re  # Import regex to extract X, Y, Z values

class printer():

    def __init__(self, camera):
        self.commandQueue = queue.Queue()
        self.X = 0
        self.Y = 0
        self.Z = 3000
        self.speed = 4
        self.printerSerial = None
        self.commandQueue.put("G28")
        self.commandQueue.put("G0 Z30")
        self.printerSerial = serial.Serial('COM6', 115200) #250000
        self.paused = False
        self.camera = camera
        threading.Thread(target=self.startThread, daemon=True).start()
        self.isAutomated = False

    def startThread(self):

        def processCommand(command):
            self.sendCommand(command)
            response = self.printerSerial.readline().decode().strip()
            while response.lower() != 'ok':
                response = self.printerSerial.readline().decode().strip()

        time.sleep(1)
        last_image_black = False
        z_start = 4380
        z_end = 4780
        z_step = 4
        z_dir = 1

        while True:
            if not self.paused:
                command = self.commandQueue.get()

                if command.startswith("G"):

                    # Extract X, Y, Z positions from the command
                    match = re.search(r'X([\d\.]+)', command)
                    if match:
                        self.X = int(float(match.group(1)) * 100)

                    match = re.search(r'Y([\d\.]+)', command)
                    if match:
                        self.Y = int(float(match.group(1)) * 100)

                    match = re.search(r'Z([\d\.]+)', command)
                    if match:
                        self.Z = int(float(match.group(1)) * 100)

                    processCommand(command)

                    if(self.isAutomated):
                        # Handle Z moves dynamically in a zig-zag pattern
                        z_range = range(z_start, z_end + z_step, z_step) if z_dir > 0 else range(z_end, z_start - z_step, -z_step)

                        noFocusCount = 0
                        focusCount = 0
                        for z in z_range:
                            self.Z = z

                            processCommand(f"G0 Z{z / 100}") # Move Z

                            # Take picture after Z move
                            processCommand("M400") # Wait for all moves to finish
                            time.sleep(0.1)  # Let camera stabilize

                            # Capture the image, but do not save it
                            self.camera.captureImage([self.getPosition()])

                            while self.camera.isTakingImage:
                                time.sleep(0.01)  # Avoid excessive CPU usage

                            # If image is black, stop further Z moves
                            if self.camera.isBlack():
                                print(f"Skipping remaining Z moves at X/Y position")
                                last_image_black = True
                                break
                            else:
                                last_image_black = False  # Reset since we got a non-black image
                            
                            if self.camera.isInFocus():
                                # Save the camera's image
                                self.camera.saveImage("X" + str(self.X) + " Y" + str(self.Y))
                                focusCount += 1
                            elif(focusCount > 0):
                                print("Lost Focus")
                                break
                            elif(noFocusCount > 10):
                                print("Failed to Find Focus")
                                break
                            else:
                                noFocusCount += 1

                        # Only flip Z direction if the last image wasn't black
                        if not last_image_black:
                            z_dir *= -1


        
    def getPosition(self):
        #print((self.X, self.Y, self.Z))
        return (self.X, self.Y, self.Z)

    def sendCommand(self, command):
        if(not command.startswith("G0") and command != "M400"):
            print((command))
        self.printerSerial.write((command + "\n").encode())

    def togglePause(self):
        if(self.paused):
            print("Unpaused")
            self.paused = False
        else:
            print("Paused")
            self.paused = True

    def halt(self):
        while(not self.commandQueue.empty()):
            self.commandQueue.get(False)
        print("Cleared Queue")

    def startAutomation(self):
        x_start = 10200
        y_start = 10500
        z_start = 4380

        x_end = 16500
        y_end = 16700

        x_step = 200
        y_step = 200

        y_dir = 1  # Zig-zag direction for Y-axis

        self.commandQueue.put(f"G0 X{x_start / 100} Y{y_start / 100} Z{z_start / 100}")
        x_range = range(x_start, x_end + x_step, x_step) if x_start < x_end else range(x_start, x_end - x_step, -x_step)
        y_range = range(y_start, y_end + y_step, y_step)

        for x in x_range:
            if x != x_start:
                y_dir *= -1  # Flip y-direction
            for y in (y_range if y_dir > 0 else reversed(y_range)):
                self.commandQueue.put(f"G0 X{x/100} Y{y/100}")  # Queue only X/Y moves

        print("Automation Queued")
        self.isAutomated = True

    def moveZUp(self):
        self.Z += self.speed * 2
        self.commandQueue.put(f"G1 Z{(self.Z) / 100}")

    def moveZDown(self):
        self.Z -= self.speed
        self.commandQueue.put(f"G1 Z{(self.Z) / 100}")

    def moveXLeft(self):
        self.X += self.speed
        self.commandQueue.put(f"G1 X{(self.X) / 100}")
        
    def moveXRight(self):
        self.X -= self.speed
        self.commandQueue.put(f"G1 X{(self.X) / 100}")

    def moveYBackward(self):
        self.Y += self.speed
        self.commandQueue.put(f"G1 Y{(self.Y) / 100}")
        
    def moveYForward(self):
        self.Y -= self.speed
        self.commandQueue.put(f"G1 Y{(self.Y) / 100}")

    def increaseSpeed(self):
        self.speed += 4
        print("Current Speed", self.speed / 100)

    def decreaseSpeed(self):
        self.speed -= 4
        print("Current Speed", self.speed / 100)

    def increaseSpeedFast(self):
        self.speed += 100
        print("Current Speed", self.speed / 100)

    def decreaseSpeedFast(self):
        self.speed -= 100
        print("Current Speed", self.speed / 100)