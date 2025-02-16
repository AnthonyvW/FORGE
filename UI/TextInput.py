import pygame

class TextInput():
    def __init__(self, functionToCall, x:int, y:int, width:int, height:int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
        self.rect = pygame.Rect(x, y, width, height)
        
        self.functionToCall = functionToCall
        self.isSelected = False

        self.foreground = pygame.Color(64, 255, 64)
        self.SelectedBackground = pygame.Color(128, 128, 255)

        self.background = pygame.Color(32, 128, 32)

    def setPosition(self, x, y):
        self.x = x
        self.y = y
    
    def updatePosition(self, xOffset, yOffset):
        self.x += xOffset
        self.y += yOffset

    def CheckButton(self, mouseX:int, mouseY:int, mouseClick: bool):
        if(self.rect.collidepoint(mouseX, mouseY)):
            if(mouseClick):
                self.functionToCall()
            
            self.isSelected = True
            return False
        else:
            self.isSelected = False
            return False

    def draw(self, surface):
        if(self.isSelected):
            pygame.draw.rect(surface, self.SelectedBackground, (self.x, self.y, self.width, self.height))
            pygame.draw.rect(surface, self.foreground, (self.x, self.y, self.width, self.height), 2)
        else:
            pygame.draw.rect(surface, self.background, (self.x, self.y, self.width, self.height))
            pygame.draw.rect(surface, self.foreground, (self.x, self.y, self.width, self.height), 2)
    


