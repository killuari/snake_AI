import pygame, random
import numpy as np
from enum import Enum

GRID_SIZE = 30
GRID_WIDTH = 30
GRID_HEIGHT = 20

class Direction(Enum):
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    UP = (0, -1)
    DOWN = (0, 1)

    @property
    def array(self):
        return np.array(self.value)

    def opposite(self):
        opposites = {
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP
        }
        return opposites[self]
    
    def left(self):
        lefts = {
            Direction.LEFT: Direction.DOWN,
            Direction.RIGHT: Direction.UP,
            Direction.UP: Direction.LEFT,
            Direction.DOWN: Direction.RIGHT
        }
        return lefts[self]
    
    def right(self):
        rights = {
            Direction.LEFT: Direction.UP,
            Direction.RIGHT: Direction.DOWN,
            Direction.UP: Direction.RIGHT,
            Direction.DOWN: Direction.LEFT
        }
        return rights[self]

class SnakeGame:
    def __init__(self, grid_size, grid_width, grid_height):
        self.grid_size = grid_size
        self.grid_width = grid_width
        self.grid_height = grid_height

        self.apple = Apple(grid_size, grid_width, grid_height)
        self.score = 0
        
        pos = pygame.Vector2(grid_size*grid_width // 2, grid_size*grid_height // 2)
        self.head = SnakePart(grid_size, grid_width, grid_height, pos)
        self.snake_list = [self.head, SnakePart(grid_size, grid_width, grid_height, pos - pygame.Vector2(self.grid_size, 0))]

        [self.add_part() for i in range(2)]

    def get_tail_locations(self):
        return [part.grid_pos for part in self.snake_list[1:]]

    def add_part(self):
        self.snake_list.append(SnakePart(self.grid_size, self.grid_width, self.grid_height, self.snake_list[-1].pos))

    def move_snake(self, dir: Direction):
        pos = self.head.pos.copy()
        alive = self.head.move(dir)
    
        for part in self.snake_list[1:]:
            new_pos = part.pos.copy()
            part.pos = pos.copy()
            part.grid_pos = np.array([part.pos.x // self.grid_size, part.pos.y // self.grid_size])
            pos = new_pos

        return alive

    def detect_collision(self):
        for part in self.snake_list[1:]:
            if np.array_equal(part.grid_pos, self.head.grid_pos):
                return True
        return False
            
    def eat_apple(self):
        apple_eaten = False

        while any(np.array_equal(self.apple.grid_pos, part.grid_pos) for part in self.snake_list):
            self.apple.place()
            self.add_part()
            self.score += 1
            apple_eaten = True

        return apple_eaten

    def draw(self, screen):
        for part in self.snake_list:
            part.draw(screen)
        self.apple.draw(screen)

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

        self.grid_pos = np.array([self.pos.x // self.grid_size, self.pos.y // self.grid_size])    

        if self.pos.x >= self.window_width or self.pos.x < 0 or self.pos.y >= self.window_height or self.pos.y < 0:
            return False
        return True

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

if __name__ == "__main__":
    pygame.init()
    font = pygame.font.Font(None, 45)
    screen = pygame.display.set_mode((GRID_SIZE * GRID_WIDTH, GRID_SIZE * GRID_HEIGHT))
    clock = pygame.time.Clock()

    dir = Direction.RIGHT
    snakeGame = SnakeGame(GRID_SIZE, GRID_WIDTH, GRID_HEIGHT)
    running = True

    while running:
        dt = clock.tick(10)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

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

        alive = snakeGame.move_snake(dir)

        snakeGame.eat_apple()

        if snakeGame.detect_collision() or not alive:
            running = False
        
        snakeGame.draw(screen)

        score_display = font.render(f"Score: {snakeGame.score}", True, pygame.Color(255, 255, 255))
        screen.blit(score_display, (15, 15))

        pygame.display.flip()

    pygame.quit()