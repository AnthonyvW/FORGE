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
        z_start = 4250#4380
        z_end = 4430#4620
        normal_z_step = 2
        fast_z_step = normal_z_step * 10
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
                        current_z_step = fast_z_step
                        z_position = z_start if z_dir > 0 else z_end
                        noFocusCount = 0
                        focusCount = 0
                        print("ATTENTION : Starting X/Y Step")
                        while (z_dir > 0 and z_position <= z_end) or (z_dir < 0 and z_position >= z_start):
                            self.Z = z_position
                            processCommand(f"G0 Z{z_position / 100}")
                            processCommand("M400")
                            time.sleep(0.1)

                            self.camera.captureImage([self.getPosition()])
                            while self.camera.isTakingImage:
                                time.sleep(0.01)

                            if self.camera.isBlack():
                                print(f"Skipping remaining Z moves at X/Y position : Black Image")
                                last_image_black = True
                                break
                            else:
                                last_image_black = False

                            focus_score = self.camera.isInFocus()
                            if focus_score > 680: # 680
                                self.camera.saveImage("X" + str(self.X) + " Y" + str(self.Y))
                                focusCount += 1
                                noFocusCount = 0
                                steps_remaining = 7
                            elif focus_score > 610 and current_z_step == fast_z_step:
                                # Back up 3 steps
                                z_position -= 3 * normal_z_step * z_dir
                                # Switch to normal step size for next 7 steps
                                current_z_step = normal_z_step
                                steps_remaining = 7
                            elif focusCount > 0 and noFocusCount > 2:
                                print("Skipping remaining Z moves at X/Y position : Lost Focus")
                                break
                            else:
                                if(current_z_step == fast_z_step):
                                    noFocusCount += 1
                                if(noFocusCount >= 10):
                                    break
                                if current_z_step == normal_z_step:
                                    steps_remaining -= 1
                                    if steps_remaining == 0:
                                        current_z_step = fast_z_step

                            z_position += current_z_step * z_dir

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
        self.paused = True
        while(not self.commandQueue.empty()):
            self.commandQueue.get(False)
        self.isAutomated = False
        self.paused = False
        print("Cleared Queue")

    def startAutomation(self):
        x_start = 7098#10200 + 400
        y_start = 8556#10500
        z_start = 4250#4380

        x_end = 7388#16500
        y_end = 17236#16700

        x_step = 100
        y_step = 100

        dir = 1  # Zig-zag direction for Y-axis

        self.commandQueue.put(f"G0 X{x_start / 100} Y{y_start / 100} Z{z_start / 100}")
        x_range = range(x_start, x_end + x_step, x_step) if x_start < x_end else range(x_start, x_end - x_step, -x_step)
        y_range = range(y_start, y_end + y_step, y_step)

        for y in y_range:
            if y != y_start:
                dir *= -1  # Flip x-direction
            for x in (x_range if dir > 0 else reversed(x_range)):
                self.commandQueue.put(f"G0 X{x/100} Y{y/100}")  # Queue only X/Y moves

        print("Automation Queued")
        self.isAutomated = True

    def moveZUp(self):
        self.Z += self.speed
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
        self.speed += 2
        print("Current Speed", self.speed / 100)

    def decreaseSpeed(self):
        self.speed -= 2
        print("Current Speed", self.speed / 100)

    def increaseSpeedFast(self):
        self.speed += 100
        print("Current Speed", self.speed / 100)

    def decreaseSpeedFast(self):
        self.speed -= 100
        print("Current Speed", self.speed / 100)