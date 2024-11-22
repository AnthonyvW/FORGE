import cv2
import pygame
import numpy as np

pygame.init()
pygame.display.set_caption("OpenCV camera stream on Pygame")
surface = pygame.display.set_mode([1280,720])
#0 Is the built in camera
cap = cv2.VideoCapture(0)
#Gets fps of your camera
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps:", fps)
#If your camera can achieve 60 fps
#Else just have this be 1-30 fps
cap.set(cv2.CAP_PROP_FPS, 60)

while True:
    surface.fill([0,0,0])

    success, frame = cap.read()
    if not success:
        break

    #for some reasons the frames appeared inverted
    frame = np.fliplr(frame)
    frame = np.rot90(frame)

    # The video uses BGR colors and PyGame needs RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    surf = pygame.surfarray.make_surface(frame)

    for event in pygame.event.get():
        if event.type == pygame.KEYUP:
            background_color = red
            surface.fill(background_color)
            pygame.display.update
            end_time = self.time()

    # Show the PyGame surface!
    surface.blit(surf, (0,0))
    pygame.display.flip()