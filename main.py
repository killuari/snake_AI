import pygame, random
from enum import Enum

GRID_SIZE = 30
GRID_WIDTH = 50
GRID_HEIGHT = 30

window_width = GRID_WIDTH * GRID_SIZE
window_height = GRID_HEIGHT * GRID_SIZE

# pygame setup
pygame.init()
screen = pygame.display.set_mode((window_width, window_height))
clock = pygame.time.Clock()
running = True


class Direction(Enum):
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    UP = (0, -1)
    DOWN = (0, 1)

class Snake:
    def __init__(self, screen, pos):
        self.screen = screen
        self.head = SnakePart(screen, pos)
        self.snake_list = [self.head, SnakePart(screen, pos), SnakePart(screen, pos)]

    def add_part(self):
        self.snake_list.append(SnakePart(self.screen, self.snake_list[-1].pos))

    def move(self, dir: Direction):
        pos = self.snake_list[0].pos.copy()
        self.snake_list[0].move(dir)
    
        for part in self.snake_list[1:]:
            new_pos = part.pos.copy()
            part.pos = pos.copy()
            pos = new_pos

    def draw(self):
        for part in self.snake_list:
            part.draw()

class SnakePart:
    def __init__(self, screen, pos):
        self.screen = screen
        self.pos = pos

    def draw(self):
        pygame.draw.rect(self.screen, pygame.Color(0, 200, 0), pygame.Rect(self.pos.x, self.pos.y, GRID_SIZE, GRID_SIZE))

    def move(self, dir: Direction):
        if dir == Direction.LEFT:
            self.pos.x -= GRID_SIZE
        elif dir == Direction.RIGHT:
            self.pos.x += GRID_SIZE
        elif dir == Direction.UP:
            self.pos.y -= GRID_SIZE
        elif dir == Direction.DOWN:
            self.pos.y += GRID_SIZE

class Apple:
    def __init__(self, screen):
        self.screen = screen
        self.place()

    def place(self):
        x, y = random.randrange(int(GRID_SIZE/2), screen.get_width(), GRID_SIZE), random.randrange(int(GRID_SIZE/2), screen.get_height(), GRID_SIZE)
        self.pos = pygame.Vector2(x, y)

    def draw(self):
        pygame.draw.circle(self.screen, pygame.Color(200, 0, 100), self.pos, GRID_SIZE/2)

player_pos = pygame.Vector2(window_width / 2, window_height / 2)
dir = Direction.RIGHT

snake = Snake(screen, player_pos)
apple = Apple(screen)

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

    if snake.head.pos.x + 15 == apple.pos.x and snake.head.pos.y + 15 == apple.pos.y:
        apple.place()
        snake.add_part()

    apple.draw()

    pygame.display.flip()

pygame.quit()