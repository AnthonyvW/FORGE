import cv2
import numpy as np
import pygame

from UI.Button import Button

pygame.init()
pygame.display.set_caption("Tree Ring Imaging Machine v2")
screen = pygame.display.set_mode([1280,720])
#0 Is the built in camera
cap = cv2.VideoCapture(0)
#Gets fps of your camera
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps:", fps)
#If your camera can achieve 60 fps
#Else just have this be 1-30 fps
cap.set(cv2.CAP_PROP_FPS, 60)
# A clock to limit the frame rate.
clock = pygame.time.Clock()  
# Font
defaultFont = pygame.font.SysFont("Arial", 30)

# Draws Text on screen at x, y
def draw_text(text, _x, _y, color=(255, 255, 255), font=defaultFont):
    img = font.render(text, True, color)
    screen.blit(img, (_x, _y))


def say():
    print("Hello")

button = Button(say, 1000, 500, 40, 40)

running = True
while running:
    clock.tick(60)
    # Mouse Position
    pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            button.CheckButton(pos[0], pos[1], True)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
        if event.type == pygame.KEYUP:
            background_color = pygame.Color(255,0,0)
            screen.fill(background_color)

    # Update

    button.CheckButton(pos[0], pos[1], False)

    success, frame = cap.read()
    if not success:
        break
    # The video uses BGR colors and PyGame needs RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #for some reasons the frames appeared inverted
    frame = np.fliplr(frame)
    frame = np.rot90(frame)
    surf = pygame.surfarray.make_surface(frame)


    # Rendering

    screen.fill([0,0,0])

    button.draw(screen)
    
    screen.blit(surf, (0,0))
    pygame.display.flip()
    pygame.display.update()

pygame.quit()