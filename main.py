import pygame, random
import gymnasium as gym
import numpy as np
from enum import Enum
from typing import Optional

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

font = pygame.font.Font(None, 50)

class Direction(Enum):
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    UP = (0, -1)
    DOWN = (0, 1)

    def opposite(self):
        opposites = {
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP
        }
        return opposites[self]

class Snake:
    def __init__(self, screen, pos):
        self.screen = screen
        self.head = SnakePart(screen, pos)
        self.snake_list = [self.head, SnakePart(self.screen, pos - pygame.Vector2(GRID_SIZE, 0))]

        [self.add_part() for i in range(2)]

    def add_part(self):
        self.snake_list.append(SnakePart(self.screen, self.snake_list[-1].pos))

    def move(self, dir: Direction):
        pos = self.snake_list[0].pos.copy()
        self.snake_list[0].move(dir)
    
        for part in self.snake_list[1:]:
            new_pos = part.pos.copy()
            part.pos = pos.copy()
            pos = new_pos

    def detect_collision(self):
        for part in self.snake_list[1:]:
            if part.pos == self.head.pos:
                return True

    def draw(self):
        for part in self.snake_list:
            part.draw()

class SnakePart:
    def __init__(self, screen, pos):
        self.screen = screen
        self.pos = pos

    def draw(self):
        pygame.draw.rect(self.screen, pygame.Color(80, 220, 80), pygame.Rect(self.pos.x, self.pos.y, GRID_SIZE, GRID_SIZE))

    def move(self, dir: Direction):
        if dir == Direction.LEFT:
            self.pos.x -= GRID_SIZE
        elif dir == Direction.RIGHT:
            self.pos.x += GRID_SIZE
        elif dir == Direction.UP:
            self.pos.y -= GRID_SIZE
        elif dir == Direction.DOWN:
            self.pos.y += GRID_SIZE

        if self.pos.x > window_width:
            self.pos.x = 0
            dir = dir.opposite()
        elif self.pos.x < 0:
            self.pos.x = window_width
            dir = dir.opposite()
        elif self.pos.y > window_height:
            self.pos.y = 0
            dir = dir.opposite()
        elif self.pos.y < 0:
            self.pos.y = window_height
            dir = dir.opposite()

class Apple:
    def __init__(self, screen):
        self.screen = screen
        self.place()

    def place(self):
        x, y = random.randrange(int(GRID_SIZE/2), screen.get_width(), GRID_SIZE), random.randrange(int(GRID_SIZE/2), screen.get_height(), GRID_SIZE)
        self.pos = pygame.Vector2(x, y)

    def draw(self):
        pygame.draw.circle(self.screen, pygame.Color(230, 100, 100), self.pos, GRID_SIZE/2)

class SnakeGameEnvironment(gym.Env):
    def __init__(self, grid_size = 30, grid_width = 50, grid_height = 30):
        self.grid_size = grid_size
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.max_length = grid_width * grid_height

        self.head_location = np.array(-1, -1)
        self.apple_location = np.array(-1, -1)
        self.tail_locations = []

        coord_space = gym.spaces.MultiDiscrete([self.grid_width, self.grid_size])

        self.observation_space = gym.spaces.Dict(
            {
                "head": coord_space,
                "apple": coord_space,
                "tail": gym.spaces.Tuple(coord_space for _ in range(self.max_length - 1))
            }
        )

        self.action_space = gym.spaces.Discrete(4)

        self.action_to_direction = {
            0: Direction.RIGHT,
            1: Direction.LEFT,
            2: Direction.UP,
            3: Direction.DOWN
        }

    def _get_obs(self):
        tail_locations = (np.array(-1, -1) for _ in self.max_length - 1)
        for idx, loc in enumerate(self.tail_locations):
            tail_locations[idx] = loc

        return {"head": self.head_location, "apple": self.apple_location, "tail": tail_locations}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        

player_pos = pygame.Vector2(window_width / 2, window_height / 2)
dir = Direction.RIGHT

snake = Snake(screen, player_pos)
apple = Apple(screen)
score = 0

while running:
    dt = clock.tick(2)

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
    if keys[pygame.K_w] and dir != Direction.DOWN:
        dir = Direction.UP
    elif keys[pygame.K_s] and dir != Direction.UP:
        dir = Direction.DOWN
    elif keys[pygame.K_a] and dir != Direction.RIGHT:
        dir = Direction.LEFT
    elif keys[pygame.K_d] and dir != Direction.LEFT:
        dir = Direction.RIGHT

    snake.move(dir)
    snake.draw()
    if snake.detect_collision():
        running = False

    if snake.head.pos.x + 15 == apple.pos.x and snake.head.pos.y + 15 == apple.pos.y:
        apple.place()
        snake.add_part()
        score += 1

    apple.draw()

    score_display = font.render(f"Score: {score}", True, pygame.Color(255, 255, 255))
    screen.blit(score_display, (15, 15))

    pygame.display.flip()

pygame.quit()