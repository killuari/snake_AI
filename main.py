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
    def __init__(self, grid_size, grid_width, grid_height):
        self.grid_size = grid_size
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        pos = pygame.Vector2(grid_size*grid_width // 2, grid_size*grid_height // 2)
        self.head = SnakePart(grid_size, grid_width, grid_height, pos)
        self.snake_list = [self.head, SnakePart(grid_size, grid_width, grid_height, pos - pygame.Vector2(self.grid_size, 0))]

        [self.add_part() for i in range(2)]

    def get_tail_locations(self):
        return [part.grid_pos for part in self.snake_list[1:]]

    def add_part(self):
        self.snake_list.append(SnakePart(self.grid_size, self.grid_width, self.grid_height, self.snake_list[-1].pos))

    def move(self, dir: Direction):
        pos = self.snake_list[0].pos.copy()
        self.snake_list[0].move(dir)
    
        for part in self.snake_list[1:]:
            new_pos = part.pos.copy()
            part.pos = pos.copy()
            part.grid_pos = np.array([part.pos.x // self.grid_size, part.pos.y // self.grid_size])
            pos = new_pos

    def detect_collision(self):
        for part in self.snake_list[1:]:
            if part.pos == self.head.pos:
                return True

    def draw(self, screen):
        for part in self.snake_list:
            part.draw(screen)

class SnakePart:
    def __init__(self, grid_size, grid_width, grid_height, pos: pygame.Vector2):
        self.grid_size = grid_size
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.window_width = grid_size * grid_width
        self.window_height = grid_size * grid_height

        self.pos = pos
        self.grid_pos = np.array([self.pos.x // self.grid_size, self.pos.y // self.grid_size])

    def draw(self, screen):
        pygame.draw.rect(screen, pygame.Color(80, 220, 80), pygame.Rect(self.pos.x, self.pos.y, self.grid_size, self.grid_size))

    def move(self, dir: Direction):
        if dir == Direction.LEFT:
            self.pos.x -= self.grid_size
        elif dir == Direction.RIGHT:
            self.pos.x += self.grid_size
        elif dir == Direction.UP:
            self.pos.y -= self.grid_size
        elif dir == Direction.DOWN:
            self.pos.y += self.grid_size

        if self.pos.x > self.window_width:
            self.pos.x = 0
            dir = dir.opposite()
        elif self.pos.x < 0:
            self.pos.x = self.window_width
            dir = dir.opposite()
        elif self.pos.y > self.window_height:
            self.pos.y = 0
            dir = dir.opposite()
        elif self.pos.y < 0:
            self.pos.y = self.window_height
            dir = dir.opposite()

        self.grid_pos = np.array([self.pos.x // self.grid_size, self.pos.y // self.grid_size])        

class Apple:
    def __init__(self, grid_size, grid_width, grid_height):
        self.grid_size = grid_size
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.place()

    def place(self):
        x, y = random.randrange(self.grid_size // 2, self.grid_width * self.grid_size, self.grid_size), random.randrange(self.grid_size // 2, self.grid_height * self.grid_size, self.grid_size)
        self.pos = pygame.Vector2(x, y)
        self.grid_pos = np.array([x // self.grid_size, y // self.grid_size])

    def draw(self, screen):
        pygame.draw.circle(screen, pygame.Color(230, 100, 100), self.pos, self.grid_size/2)

class GameEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, grid_size, grid_width, grid_height, render_mode = None):
        self.grid_size = grid_size
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.max_length = grid_width * grid_height

        self.snake = None
        self.apple = None
        
        self.screen = None
        self.clock = None

        self.head_location = np.array([-1, -1])
        self.apple_location = np.array([-1, -1])
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

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        tail_locations = (np.array([-1, -1]) for _ in self.max_length - 1)
        for idx, loc in enumerate(self.tail_locations):
            tail_locations[idx] = loc

        return {"head": self.head_location, "apple": self.apple_location, "tail": tail_locations}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.snake = Snake(self.grid_size, self.grid_width, self.grid_height)
        self.apple = Apple(self.grid_size, self.grid_width, self.grid_height)

        self.head_location = self.snake.head.grid_pos
        self.apple_location = self.apple.grid_pos
        self.tail_locations = self.snake.get_tail_locations()

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation
        
dir = Direction.RIGHT

snake = Snake(GRID_SIZE, GRID_WIDTH, GRID_HEIGHT)
apple = Apple(GRID_SIZE, GRID_WIDTH, GRID_HEIGHT)
score = 0

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
    if keys[pygame.K_w] and dir != Direction.DOWN:
        dir = Direction.UP
    elif keys[pygame.K_s] and dir != Direction.UP:
        dir = Direction.DOWN
    elif keys[pygame.K_a] and dir != Direction.RIGHT:
        dir = Direction.LEFT
    elif keys[pygame.K_d] and dir != Direction.LEFT:
        dir = Direction.RIGHT

    snake.move(dir)
    snake.draw(screen)
    if snake.detect_collision():
        running = False

    if snake.head.pos.x + 15 == apple.pos.x and snake.head.pos.y + 15 == apple.pos.y:
        apple.place()
        snake.add_part()
        score += 1

    apple.draw(screen)

    score_display = font.render(f"Score: {score}", True, pygame.Color(255, 255, 255))
    screen.blit(score_display, (15, 15))

    pygame.display.flip()

pygame.quit()