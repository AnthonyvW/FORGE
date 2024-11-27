import pygame

from UI.Button import Button
from Camera.DefaultCamera import DefaultCamera


pygame.init()
pygame.display.set_caption("Tree Ring Imaging Machine v2")
width, height = (1280,720)
#scale = min(width/src.width, height/src.height)
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)

# A clock to limit the frame rate.
clock = pygame.time.Clock()  

def say():
    print("Hello")

button = Button(say, 1000, 500, 40, 40)
camera = DefaultCamera(width-500,height)

running = True
while running:
    clock.tick(60)
    # Mouse Position
    pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.VIDEORESIZE:
            width, height = screen.get_size()
            camera.resize(width-500, height)
            print(width, height)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            button.CheckButton(pos[0], pos[1], True)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
        elif event.type == pygame.KEYUP:
            background_color = pygame.Color(255,0,0)
            screen.fill(background_color)

    # Update
    camera.update()
    button.CheckButton(pos[0], pos[1], False)

    # Rendering

    screen.fill([0,0,0])

    button.draw(screen)
    
    screen.blit(camera.getFrame(), (0,0))
    pygame.display.flip()
    pygame.display.update()

pygame.quit()