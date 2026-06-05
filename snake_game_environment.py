"""
snake_game_environment.py — Gymnasium Environment Wrapper for the Snake Game

Wraps SnakeGame into a gym.Env for RL training with Stable Baselines3.
Uses a local FOV observation so models generalize across grid sizes.

Observation encoding: 0=empty, 1=body, 2=apple, 3=wall.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

import pygame
import gymnasium as gym
import numpy as np
from typing import Optional
from stable_baselines3.common.monitor import Monitor

from snake_game import SnakeGame, Direction


class SnakeGameEnvironment(gym.Env):
    """
    Gymnasium environment wrapping the Snake game for RL training.

    Observation: MultiDiscrete array of the local FOV around the head.
    Action: Discrete(4) — 0=RIGHT, 1=DOWN, 2=LEFT, 3=UP.

    Args:
        grid_size: Pixel size of each cell.
        grid_width/height: Grid dimensions in cells.
        snake_fov_radius: FOV radius around head ((2r+1)^2 - 1 cells).
        render_mode: None, "human", or "rgb_array".
        training: If True, apply reward shaping (penalties). False for eval.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, grid_size, grid_width, grid_height, snake_fov_radius = 1, render_mode = None, training = True):
        self.grid_size = grid_size
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.snake_fov_radius = snake_fov_radius
        self.training = training

        self.snakeGame = None          # Created on reset()

        # Pygame objects (lazy-initialized on first render)
        self.screen = None
        self.clock = None

        # Cached positions for building observations
        self.head_location = np.array([-1, -1])
        self.apple_location = np.array([-1, -1])
        self.tail_locations = []
        self.dir = None

        # Observation: one int per FOV cell (excl. head). Values: 0-3
        n_cells = (2*self.snake_fov_radius + 1)**2 - 1
        self.observation_space = gym.spaces.MultiDiscrete([4] * n_cells)

        # 4 discrete actions
        self.action_space = gym.spaces.Discrete(4)

        # Map action ints to Direction enums
        self.action_to_direction = {
            0: Direction.RIGHT,
            1: Direction.DOWN,
            2: Direction.LEFT,
            3: Direction.UP
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        """Build observation array from current game state.
        Scans FOV around head. Each cell: 0=empty, 1=body, 2=apple, 3=wall."""
        locations = []
        hx, hy = self.head_location

        for dy in range(-self.snake_fov_radius, self.snake_fov_radius + 1):
            for dx in range(-self.snake_fov_radius, self.snake_fov_radius + 1):
                if dx == 0 and dy == 0:
                    continue  # Skip head position
                loc = np.array([hx + dx, hy + dy])
                locations.append(loc)

        # Classify each cell in the FOV
        for loc in range(len(locations)):
            if locations[loc][0] < 0 or locations[loc][0] >= self.grid_width or locations[loc][1] < 0 or locations[loc][1] >= self.grid_height:
                locations[loc] = 3      # Wall / out of bounds
            elif np.array_equal(locations[loc], self.apple_location):
                locations[loc] = 2      # Apple
            elif any(np.array_equal(locations[loc], tail_part) for tail_part in self.tail_locations):
                locations[loc] = 1      # Snake body
            else:
                locations[loc] = 0      # Empty

        return np.array(locations)
    
    def _get_info(self):
        """Return auxiliary info dict with current snake length."""
        return {
            "snake_length": len(self.snakeGame.snake_list)
        }
    
    def update_locations(self):
        """Cache head, apple, and tail positions from game state."""
        self.head_location = self.snakeGame.head.grid_pos
        self.apple_location = self.snakeGame.apple.grid_pos
        self.tail_locations = self.snakeGame.get_tail_locations()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset to a fresh game. Returns (observation, info)."""
        super().reset(seed=seed)

        self.snakeGame = SnakeGame(self.grid_size, self.grid_width, self.grid_height)
        self.dir = Direction.RIGHT     # Always start facing right
        self.steps = 0                 # Step counter for max-step detection

        self.update_locations()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        """
        Execute one game step.

        Reward (training mode only):
            +1.0 for eating apple (resets step counter).
            -0.5 for exceeding max_steps (anti-loop, terminates).
            -min(20, max(1, (length-3)*0.5)) for death (scales with length).

        Returns: (observation, reward, terminated, truncated=False, info)
        """
        action = action.item()
        # Prevent 180° reversal (would cause instant self-collision)
        if self.action_to_direction[action] != self.dir.opposite():
            self.dir = self.action_to_direction[action]

        reward = 0
        terminated = False
        self.steps += 1
        snake_length = len(self.snakeGame.snake_list)

        alive = self.snakeGame.move_snake(self.dir)
        apple_eaten = self.snakeGame.eat_apple()
        
        # Max steps before timeout = 2/3 of total grid cells
        max_steps = 2/3 * (self.grid_width * self.grid_height)

        if apple_eaten:
            reward = 1
            self.steps = 0             # Reset after eating
        elif self.steps >= max_steps and self.training:
            reward = -0.5              # Anti-loop penalty
            terminated = True

        if not alive:
            terminated = True
            if self.training:
                # Death penalty scaled by snake length (min -1, max -20)
                reward = -min(20, max(1, (snake_length-3) * 0.5))

        self.update_locations()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def render(self):
        """Public render (Gymnasium API). Only returns data for rgb_array mode."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self):
        """Internal render. Human mode: Pygame window. rgb_array: returns np array."""
        # Lazy-init Pygame display for human mode
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_size * self.grid_width, self.grid_size * self.grid_height))
            self.font = pygame.font.Font(None, 45)
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Draw on off-screen canvas
        canvas = pygame.Surface(((self.grid_size * self.grid_width, self.grid_size * self.grid_height)))
        canvas.fill("black")

        self.snakeGame.draw(canvas)
        
        # Render score overlay (font only exists in human mode)
        if self.font is not None:
            score_display = self.font.render(f"Score: {self.snakeGame.score}", True, pygame.Color(255, 255, 255))
            canvas.blit(score_display, (15, 15))

        if self.render_mode == "human":
            self.screen.blit(canvas, (0, 0))
            pygame.event.pump()        # Keep window responsive
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        """Clean up Pygame resources."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


def make_snake_env(grid_size, grid_width, grid_height, snake_fov_radius = 1, render_mode = None, training = True):
    """Factory returning a callable that creates a Monitor-wrapped environment.
    Required by SubprocVecEnv (one callable per subprocess)."""
    def _init():
        env = SnakeGameEnvironment(grid_size, grid_width, grid_height, snake_fov_radius, render_mode, training)
        env = Monitor(env, filename=None)
        return env
    return _init