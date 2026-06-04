"""
snake_game.py — Core Snake Game Engine (Pygame)

This module contains the pure game logic for a classic Snake game.
It is independent of any AI/RL framework and can be played standalone
by running this file directly (WASD controls).

Classes:
    Direction — Enum for movement directions with helper methods.
    SnakeGame — Main game controller managing the snake, apple, and score.
    SnakePart  — A single segment of the snake body.
    Apple      — The food item that the snake tries to eat.
"""

import pygame, random
import numpy as np
from enum import Enum

# Default grid configuration used when running this file standalone.
# These are NOT used by the RL environment — it passes its own values.
GRID_SIZE = 30       # Size of each grid cell in pixels
GRID_WIDTH = 50      # Number of cells horizontally
GRID_HEIGHT = 30     # Number of cells vertically


class Direction(Enum):
    """
    Represents the four cardinal movement directions.

    Each direction stores its (dx, dy) vector as the enum value.
    Positive x = right, positive y = down (Pygame coordinate system).
    """
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    UP = (0, -1)
    DOWN = (0, 1)

    @property
    def array(self):
        """Return the direction as a numpy array, e.g. np.array([-1, 0]) for LEFT."""
        return np.array(self.value)

    def opposite(self):
        """Return the opposite direction (e.g. LEFT → RIGHT).
        Used to prevent the snake from reversing into itself."""
        opposites = {
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP
        }
        return opposites[self]
    
    def left(self):
        """Return the direction 90° counter-clockwise (e.g. UP → LEFT).
        Useful for relative-direction observations."""
        lefts = {
            Direction.LEFT: Direction.DOWN,
            Direction.RIGHT: Direction.UP,
            Direction.UP: Direction.LEFT,
            Direction.DOWN: Direction.RIGHT
        }
        return lefts[self]
    
    def right(self):
        """Return the direction 90° clockwise (e.g. UP → RIGHT).
        Useful for relative-direction observations."""
        rights = {
            Direction.LEFT: Direction.UP,
            Direction.RIGHT: Direction.DOWN,
            Direction.UP: Direction.RIGHT,
            Direction.DOWN: Direction.LEFT
        }
        return rights[self]


class SnakeGame:
    """
    Main game controller that manages the snake, apple, scoring, and game state.

    The game operates on a discrete grid. Positions are stored in pixel coordinates
    (multiples of grid_size) and also tracked as grid coordinates for collision
    detection and observation building.

    Args:
        grid_size:   Pixel size of one grid cell.
        grid_width:  Number of cells in the horizontal axis.
        grid_height: Number of cells in the vertical axis.
    """

    def __init__(self, grid_size, grid_width, grid_height):
        self.grid_size = grid_size
        self.grid_width = grid_width
        self.grid_height = grid_height

        # Create the apple (randomly placed on the grid)
        self.apple = Apple(grid_size, grid_width, grid_height)
        self.score = 0
        
        # Initialize the snake at the center of the grid, facing right,
        # with 3 segments: head + 2 body parts extending to the left.
        pos = pygame.Vector2(grid_size*grid_width // 2, grid_size*grid_height // 2)
        self.head = SnakePart(grid_size, grid_width, grid_height, pos)
        self.snake_list = [self.head, SnakePart(grid_size, grid_width, grid_height, pos - pygame.Vector2(self.grid_size, 0)), SnakePart(grid_size, grid_width, grid_height, pos - 2*pygame.Vector2(self.grid_size, 0))]

    def get_tail_locations(self):
        """Return a list of grid positions for all body parts (excluding the head).
        Used by the RL environment to build the observation."""
        return [part.grid_pos for part in self.snake_list[1:]]

    def add_part(self):
        """Append a new segment at the tail's current position.
        The new part will separate from the tail on the next move."""
        self.snake_list.append(SnakePart(self.grid_size, self.grid_width, self.grid_height, self.snake_list[-1].pos))

    def move_snake(self, dir: Direction):
        """
        Move the entire snake one step in the given direction.

        Movement logic:
        1. Move the head in the specified direction.
        2. Check for self-collision (head overlaps any body segment).
        3. Cascade body positions: each segment takes the previous position
           of the segment in front of it (classic snake movement).

        Args:
            dir: The Direction to move the head.

        Returns:
            True if the snake is still alive, False if it hit a wall or itself.
        """
        pos = self.head.pos.copy()
        alive = self.head.move(dir)

        # Check self-collision (head position vs. all body parts)
        if self.detect_collision():
            alive = False
    
        # Cascade positions: each body part takes the previous position
        # of the part in front of it
        for part in self.snake_list[1:]:
            new_pos = part.pos.copy()
            part.pos = pos.copy()
            part.grid_pos = np.array([part.pos.x // self.grid_size, part.pos.y // self.grid_size])
            pos = new_pos

        return alive

    def detect_collision(self):
        """Check if the snake's head overlaps with any body segment.
        Returns True if a self-collision is detected."""
        for part in self.snake_list[1:]:
            if np.array_equal(part.grid_pos, self.head.grid_pos):
                return True
        return False
            
    def eat_apple(self):
        """
        Check if the apple overlaps with any snake segment.
        If so, re-place the apple, grow the snake, and increment the score.

        NOTE: This uses a while-loop to keep re-placing until the apple
        lands on an empty cell. Each iteration also grows the snake,
        which means multiple growth events can occur if the apple
        keeps landing on the snake body. (See code review for details.)

        Returns:
            True if the apple was eaten, False otherwise.
        """
        apple_eaten = False

        while any(np.array_equal(self.apple.grid_pos, part.grid_pos) for part in self.snake_list):
            self.apple.place()
            self.add_part()
            self.score += 1
            apple_eaten = True

        return apple_eaten

    def draw(self, screen):
        """Render all snake segments and the apple onto the given Pygame surface."""
        for part in self.snake_list:
            part.draw(screen)
        self.apple.draw(screen)


class SnakePart:
    """
    Represents a single segment of the snake (head or body).

    Each part tracks both its pixel position (for rendering) and its
    grid position (for collision detection and observations).

    Args:
        grid_size:   Pixel size of one grid cell.
        grid_width:  Number of cells horizontally.
        grid_height: Number of cells vertically.
        pos:         Initial pixel position as a pygame.Vector2.
    """

    def __init__(self, grid_size, grid_width, grid_height, pos: pygame.Vector2):
        self.grid_size = grid_size
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.window_width = grid_size * grid_width       # Total window width in pixels
        self.window_height = grid_size * grid_height     # Total window height in pixels

        self.pos = pos
        # Convert pixel position to grid coordinates (integer division)
        self.grid_pos = np.array([self.pos.x // self.grid_size, self.pos.y // self.grid_size])

    def draw(self, screen):
        """Draw this segment as a green rectangle on the given surface."""
        pygame.draw.rect(screen, pygame.Color(80, 220, 80), pygame.Rect(self.pos.x, self.pos.y, self.grid_size, self.grid_size))

    def move(self, dir: Direction):
        """
        Move this segment one grid cell in the given direction.
        Only used for the head segment — body parts are moved via position cascading.

        Args:
            dir: The Direction to move.

        Returns:
            True if the new position is within bounds, False if it hit a wall.
        """
        if dir == Direction.LEFT:
            self.pos.x -= self.grid_size
        elif dir == Direction.RIGHT:
            self.pos.x += self.grid_size
        elif dir == Direction.UP:
            self.pos.y -= self.grid_size
        elif dir == Direction.DOWN:
            self.pos.y += self.grid_size

        # Update grid coordinates after moving
        self.grid_pos = np.array([self.pos.x // self.grid_size, self.pos.y // self.grid_size])    

        # Check wall collision: return False if out of bounds
        if self.pos.x >= self.window_width or self.pos.x < 0 or self.pos.y >= self.window_height or self.pos.y < 0:
            return False
        return True


class Apple:
    """
    Represents the food item on the grid.

    The apple is drawn as a red circle centered in its grid cell.

    Args:
        grid_size:   Pixel size of one grid cell.
        grid_width:  Number of cells horizontally.
        grid_height: Number of cells vertically.
    """

    def __init__(self, grid_size, grid_width, grid_height):
        self.grid_size = grid_size
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.place()  # Randomly place the apple on initialization

    def place(self):
        """
        Randomly place the apple on a grid cell.

        NOTE: The pixel position uses an offset of grid_size // 2 so that the
        circle's center aligns with the center of the grid cell. The grid_pos
        is calculated via integer division, which maps back to the correct cell.
        """
        x, y = random.randrange(self.grid_size // 2, self.grid_width * self.grid_size, self.grid_size), random.randrange(self.grid_size // 2, self.grid_height * self.grid_size, self.grid_size)
        self.pos = pygame.Vector2(x, y)
        self.grid_pos = np.array([x // self.grid_size, y // self.grid_size])

    def draw(self, screen):
        """Draw the apple as a red circle centered at its pixel position."""
        pygame.draw.circle(screen, pygame.Color(230, 100, 100), self.pos, self.grid_size/2)


# ─── Standalone Game Loop ────────────────────────────────────────────────────
# Run this file directly to play the game manually with WASD keys.
if __name__ == "__main__":
    pygame.init()
    font = pygame.font.Font(None, 45)
    screen = pygame.display.set_mode((GRID_SIZE * GRID_WIDTH, GRID_SIZE * GRID_HEIGHT))
    clock = pygame.time.Clock()

    dir = Direction.RIGHT       # Initial movement direction
    snakeGame = SnakeGame(GRID_SIZE, GRID_WIDTH, GRID_HEIGHT)
    running = True

    while running:
        dt = clock.tick(10)     # Limit to 10 FPS (controls game speed)

        # Handle quit events (window close or ESC key)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        screen.fill("black")

        # Read keyboard input for direction changes.
        # Prevent reversing direction (e.g., can't go DOWN while moving UP).
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] and dir != Direction.DOWN:
            dir = Direction.UP
        elif keys[pygame.K_s] and dir != Direction.UP:
            dir = Direction.DOWN
        elif keys[pygame.K_a] and dir != Direction.RIGHT:
            dir = Direction.LEFT
        elif keys[pygame.K_d] and dir != Direction.LEFT:
            dir = Direction.RIGHT

        # Move the snake; if it dies, stop the game loop
        running = snakeGame.move_snake(dir)

        # Check and handle apple eating
        snakeGame.eat_apple()
        
        # Draw everything
        snakeGame.draw(screen)

        # Render score text overlay
        score_display = font.render(f"Score: {snakeGame.score}", True, pygame.Color(255, 255, 255))
        screen.blit(score_display, (15, 15))

        pygame.display.flip()

    pygame.quit()