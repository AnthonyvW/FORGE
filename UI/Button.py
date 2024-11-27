import pygame

class Button():
    def __init__(self, functionToCall, x:int, y:int, width:int, height:int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
        self.functionToCall = functionToCall
        self.isHover = False

        self.foreground = pygame.Color(64, 255, 64)
        self.hoverBackground = pygame.Color(128, 128, 255)

        self.background = pygame.Color(32, 128, 32)

    def CheckButton(self, mouseX:int, mouseY:int, mouseClick: bool):
        if(mouseX > self.x and mouseX < self.x + self.width and mouseY < self.y + self.height and mouseY > self.y):
            if(mouseClick):
                self.functionToCall()
            
            self.isHover = True
            return False
        else:
            self.isHover = False
            return False

    def draw(self, surface):
        if(self.isHover):
            pygame.draw.rect(surface, self.hoverBackground, (self.x, self.y, self.width, self.height))
            pygame.draw.rect(surface, self.foreground, (self.x, self.y, self.width, self.height), 2)
        else:
            pygame.draw.rect(surface, self.background, (self.x, self.y, self.width, self.height))
            pygame.draw.rect(surface, self.foreground, (self.x, self.y, self.width, self.height), 2)
    


