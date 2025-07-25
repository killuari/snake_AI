import pygame
from enum import Enum

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
dt = 0

class Direction(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

class Snake:
    pass

class SnakePart:
    def __init__(self, screen, pos):
        self.screen = screen
        self.pos = pos

    def draw(self):
        pygame.draw.rect(self.screen, pygame.Color(0, 200, 0), pygame.Rect(self.pos.x, self.pos.y, 20, 20))

    def move(self, dir: Direction, dt):
        if dir == Direction.LEFT:
            self.pos.x -= 300 * dt
        elif dir == Direction.RIGHT:
            self.pos.x += 300 * dt
        elif dir == Direction.UP:
            self.pos.y -= 300 * dt
        elif dir == Direction.DOWN:
            self.pos.y += 300 * dt

player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
dir = Direction.RIGHT

p1 = SnakePart(screen, player_pos)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill("black")

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        dir = Direction.UP
    if keys[pygame.K_s]:
        dir = Direction.DOWN
    if keys[pygame.K_a]:
        dir = Direction.LEFT
    if keys[pygame.K_d]:
        dir = Direction.RIGHT

    p1.move(dir, dt)
    p1.draw()


    pygame.display.flip()

    # limits FPS to 120
    dt = clock.tick(120) / 1000

pygame.quit()