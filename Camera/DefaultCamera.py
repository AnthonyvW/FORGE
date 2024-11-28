import cv2
import numpy as np

from Camera.Camera import Camera

class DefaultCamera(Camera):

    def initialize(self):
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.fps = self.camera.get(cv2.CAP_PROP_FPS)
        self.name = "Default Camera"
        
        print("fps:", self.fps)

    def resize(self, frameWidth, frameHeight):
        width  = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.scale = min(frameWidth/width, frameHeight/height)

    def update(self):
        success, self.frame = self.camera.read()
        if not success:
            return False
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.frame = np.rot90(self.frame)
        return True

    def setFPS(self, fps):
        self.fps = fps
        print("fps:", fps)

    def saveStillImage(self):
        pass
