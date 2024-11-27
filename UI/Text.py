import pygame

# Font
defaultFont = pygame.font.SysFont("Arial", 30)

# Draws Text on screen at x, y
def draw_text(text, _x, _y, color=(255, 255, 255), font=defaultFont):
    img = font.render(text, True, color)
    screen.blit(img, (_x, _y))