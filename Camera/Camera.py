import pygame
import abc

class Camera():
    __metaclass__ = abc.ABCMeta

    camera = None
    fps = None
    frame = None
    scale = 1
    width = 0
    height = 0

    def __init__(self, frameWidth, frameHeight):
        self.initialize()
        self.resize(frameWidth, frameHeight)

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def resize(self, frameWidth, frameHeight):
        pass

    @abc.abstractmethod
    def update(self):
        pass

    @abc.abstractmethod
    def setFPS(self, fps):
        pass

    def getFrame(self):
        return pygame.transform.scale_by(pygame.surfarray.make_surface(self.frame), self.scale)

