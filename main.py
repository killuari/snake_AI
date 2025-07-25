import pygame
from enum import Enum

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1700, 956))
clock = pygame.time.Clock()
running = True
dt = 0

class Direction(Enum):
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    UP = (0, -1)
    DOWN = (0, 1)

class Snake:
    def __init__(self, screen, pos):
        self.screen = screen
        self.snake_list = [SnakePart(screen, pos)]

    def add_part(self):
        self.snake_list.append(SnakePart(self.screen, self.snake_list[-1].pos))

    def move(self, dir: Direction):
        pos = self.snake_list[0].pos.copy()
        print(pos)
        self.snake_list[0].move(dir)
    
        for part in self.snake_list[1:]:
            print(part.pos)
            new_pos = part.pos.copy()
            part.pos = pos.copy()
            print(part.pos)
            pos = new_pos

    def draw(self):
        for part in self.snake_list:
            part.draw()

class SnakePart:
    def __init__(self, screen, pos):
        self.screen = screen
        self.pos = pos

    def draw(self):
        pygame.draw.rect(self.screen, pygame.Color(0, 200, 0), pygame.Rect(self.pos.x, self.pos.y, 25, 25))

    def move(self, dir: Direction):
        if dir == Direction.LEFT:
            self.pos.x -= 25
        elif dir == Direction.RIGHT:
            self.pos.x += 25
        elif dir == Direction.UP:
            self.pos.y -= 25
        elif dir == Direction.DOWN:
            self.pos.y += 25

player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
dir = Direction.RIGHT

snake = Snake(screen, player_pos)

while running:
    dt = clock.tick(10)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_x:
                snake.add_part()

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

    snake.move(dir)
    snake.draw()
    pygame.display.flip()

pygame.quit()