import serial
import time
import queue
import threading

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
        self.move_times = deque(maxlen=10)  # Store last 10 move times
        threading.Thread(target=self.startThread, daemon=True).start()

    def startThread(self):
        time.sleep(1)
        while(True):
            if(not self.paused):
                command = self.commandQueue.get()
                if command.startswith("G"):  # Only send move commands to the printer
                    start_time = time.time()
                    self.sendCommand(command)

                    response = self.printerSerial.readline().decode().strip()
                    while response.lower() != 'ok':
                        response = self.printerSerial.readline().decode().strip()
                        if(response != "echo:busy: processing"):
                            print("printer response:", response)
                    
                    move_time = time.time() - start_time  # Calculate actual move time
                    self.move_times.append(move_time)  # Store in queue

                # Execute CameraStill after the move
                if command.startswith("CAMERASTILL"):
                    # Send M400 to wait for all moves to finish
                    self.sendCommand("M400")
                    response = self.printerSerial.readline().decode().strip()
                    while response.lower() != 'ok':
                        response = self.printerSerial.readline().decode().strip()

                    start_time = time.time()
                    time.sleep(0.1) # Let camera stabilize
                    self.camera.takeStillImage([self.getPosition()])
                    while self.camera.isTakingImage:
                        time.sleep(0.01)  # Short sleep to avoid excessive CPU usage

                    capture_time = time.time() - start_time  # Time the image took to capture

                    # Estimate remaining time
                    remaining_commands = self.commandQueue.qsize()
                    avg_move_time = sum(self.move_times) / len(self.move_times) if self.move_times else 0.1
                    estimated_seconds = remaining_commands * (avg_move_time + capture_time)
                    estimated_minutes = estimated_seconds / 60
                    estimated_hours = estimated_minutes / 60


        
    def getPosition(self):
        print((self.X, self.Y, self.Z))
        return (self.X, self.Y, self.Z)

    def sendCommand(self, command):
        if(not command.startswith("G0")):
            print((command + "\n"))
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
        #self.camera.takeStillImage([self.getPosition()])
        x_start = 10000
        y_start = 10500
        z_start  = 4380

        x_end = 16500
        y_end = 16900
        z_end  = 4780

        x_step = 20
        y_step = 20
        z_step = 4
        
        y_dir = 1  # Zig-zag direction for Y-axis
        z_dir = 1  # Zig-zag direction for Z-axis

        self.commandQueue.put(f"G0 X{x_start / 100} Y{y_start / 100} Z{z_start / 100}")
        x_range = range(x_start, x_end + x_step, x_step) if x_start < x_end else range(x_start, x_end - x_step, -x_step)
        y_range = range(y_start, y_end + y_step, y_step)
        z_range = range(z_start, z_end + z_step, z_step)

        for x in x_range:
            if x != x_start:
                y_dir *= -1  # Flip y-direction
            for y in (y_range if y_dir > 0 else reversed(y_range)):
                if y != y_start:
                    z_dir *= -1  # Flip z-direction

                for z in (z_range if z_dir > 0 else reversed(z_range)):
                    self.commandQueue.put(f"G0 X{x/ 100} Y{y/ 100} Z{z/ 100}")
                    self.commandQueue.put("CAMERASTILL")  # Send G-code command

        print("Automation Queued")
        #self.commandQueue.put(f"G0 X{destX / 100} Y{destY / 100} Z{destZ / 100}")

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