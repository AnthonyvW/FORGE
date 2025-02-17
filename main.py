import pygame
import time

from UI.Button import Button
from UI.TextInput import TextInput
from Camera.DefaultCamera import DefaultCamera
from Camera.AmScopeCamera import AmscopeCamera
from printer import printer

pygame.init()
pygame.display.set_caption("Tree Ring Imaging Machine v2")
width, height = (1280,720)
#scale = min(width/src.width, height/src.height)
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)

# A clock to limit the frame rate.
clock = pygame.time.Clock()  

camera = AmscopeCamera(width-500,height)
movementSystem = printer(camera)
time.sleep(1.5)
camera.resize(width - 500, height)

def func1():
    pass

textFields = [
    #TextInput(func1, width - 350, 250, 160, 40)
]

buttons = [
    Button(movementSystem.moveXRight       , width - 400, 500, 40, 40),
    Button(movementSystem.moveXLeft        , width - 300, 500, 40, 40),
    Button(movementSystem.moveYBackward    , width - 350, 450, 40, 40),
    Button(movementSystem.moveYForward     , width - 350, 550, 40, 40),
    Button(movementSystem.moveZUp          , width - 250, 475, 40, 40),
    Button(movementSystem.moveZDown        , width - 250, 525, 40, 40),
    Button(movementSystem.increaseSpeed    , width - 200, 475, 40, 40),
    Button(movementSystem.decreaseSpeed    , width - 200, 525, 40, 40),
    Button(movementSystem.increaseSpeedFast, width - 150, 475, 40, 40),
    Button(movementSystem.decreaseSpeedFast, width - 150, 525, 40, 40),
    Button(movementSystem.togglePause      , width - 250, 350, 40, 40),
    Button(movementSystem.startAutomation  , width - 350, 250, 40, 40),
    Button(movementSystem.halt             , width - 150, 350, 40, 40),
    Button(camera.captureAndSaveImage      , width - 350, 350, 40, 40),
    ]

running = True
while running:
    clock.tick(60)
    # Mouse Position
    pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.VIDEORESIZE:

            for button in buttons:
                button.updatePosition(screen.get_size()[0] - width, screen.get_size()[1] - height)

            for field in textFields:
                field.updatePosition(screen.get_size()[0] - width, screen.get_size()[1] - height)

            width, height = screen.get_size()
            camera.resize(width-500, height)

            print(width, height)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            for i in range(len(buttons)):
                if(i < len(buttons) - 3):
                    buttons[i].CheckButton(pos[0], pos[1], True)
                elif(i == len(buttons) - 3):
                    buttons[i].CheckButton(pos[0], pos[1], True)
                elif(i == len(buttons) - 2):
                    buttons[i].CheckButton(pos[0], pos[1], True)
                elif(i == len(buttons) - 1):
                    buttons[i].CheckButton(pos[0], pos[1], True, movementSystem.getPosition())
            for field in textFields:
                field.CheckButton(pos[0], pos[1], True)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
        elif event.type == pygame.KEYUP:
            background_color = pygame.Color(255,0,0)
            screen.fill(background_color)

    # Update
    camera.update()
    for button in buttons:
        button.CheckButton(pos[0], pos[1], False)

    # Rendering
    screen.fill([60,60,60])

    # Draw Buttons
    for button in buttons:
        button.draw(screen)

    for field in textFields:
        field.draw(screen)

    # Draw Camera

    screen.blit(camera.getFrame(), (0,0))

    pygame.display.flip()
    pygame.display.update()

pygame.quit()