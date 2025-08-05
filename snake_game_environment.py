import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

import pygame
import gymnasium as gym
import numpy as np
from typing import Optional
from stable_baselines3.common.monitor import Monitor

from snake_game import SnakeGame, Direction


class SnakeGameEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, grid_size, grid_width, grid_height, snake_fov_radius = 1, render_mode = None, training = True):
        self.grid_size = grid_size
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.snake_fov_radius = snake_fov_radius
        self.training = training

        self.snakeGame = None

        self.screen = None
        self.clock = None

        self.head_location = np.array([-1, -1])
        self.apple_location = np.array([-1, -1])
        self.tail_locations = []
        self.dir = None

        n_cells = (2*self.snake_fov_radius + 1)**2 - 1
        self.observation_space = gym.spaces.MultiDiscrete([4] * n_cells)

        self.action_space = gym.spaces.Discrete(4)

        self.action_to_direction = {
            0: Direction.RIGHT,
            1: Direction.DOWN,
            2: Direction.LEFT,
            3: Direction.UP
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        locations = []
        hx, hy = self.head_location

        for dy in range(-self.snake_fov_radius, self.snake_fov_radius + 1):
            for dx in range(-self.snake_fov_radius, self.snake_fov_radius + 1):
                if dx == 0 and dy == 0:
                    continue  # Skip head position
                loc = np.array([hx + dx, hy + dy])
                locations.append(loc)

        for loc in range(len(locations)):
            if locations[loc][0] < 0 or locations[loc][0] >= self.grid_width or locations[loc][1] < 0 or locations[loc][1] >= self.grid_height:
                locations[loc] = 3
            elif np.array_equal(locations[loc], self.apple_location):
                locations[loc] = 2
            elif any(np.array_equal(locations[loc], tail_part) for tail_part in self.tail_locations):
                locations[loc] = 1
            else:
                locations[loc] = 0

        return np.array(locations)
    
    def _get_info(self):
        return {
            "snake_length": len(self.snakeGame.snake_list)
        }
    
    def update_locations(self):
        self.head_location = self.snakeGame.head.grid_pos
        self.apple_location = self.snakeGame.apple.grid_pos
        self.tail_locations = self.snakeGame.get_tail_locations()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.snakeGame = SnakeGame(self.grid_size, self.grid_width, self.grid_height)
        self.dir = Direction.RIGHT
        self.steps = 0

        self.update_locations()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        action = action.item()
        if self.action_to_direction[action] != self.dir.opposite():
            self.dir = self.action_to_direction[action]

        reward = 0
        terminated = False
        self.steps += 1
        snake_length = len(self.snakeGame.snake_list)

        alive = self.snakeGame.move_snake(self.dir)
        apple_eaten = self.snakeGame.eat_apple()
        
        # set max_steps to 2/3 of grid area
        max_steps = 2/3 * (self.grid_width * self.grid_height)

        if apple_eaten:
            reward = 1
            self.steps = 0
        elif self.steps >= max_steps and self.training:
            reward = -0.5
            terminated = True

        if not alive:
            terminated = True

            if self.training:
            # Bestrafung abhängig von aktueller Snake-Länge -(min: 1  max: 20)
                reward = -min(20, max(1, (snake_length-3) * 0.5))

        self.update_locations()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_size * self.grid_width, self.grid_size * self.grid_height))
            self.font = pygame.font.Font(None, 45)
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(((self.grid_size * self.grid_width, self.grid_size * self.grid_height)))
        canvas.fill("black")

        self.snakeGame.draw(canvas)
        
        if self.font is not None:
            score_display = self.font.render(f"Score: {self.snakeGame.score}", True, pygame.Color(255, 255, 255))
        canvas.blit(score_display, (15, 15))

        if self.render_mode == "human":
            self.screen.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.flip()

            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

def make_snake_env(grid_size, grid_width, grid_height, snake_fov_radius = 1, render_mode = None, training = True):
    def _init():
        env = SnakeGameEnvironment(grid_size, grid_width, grid_height, snake_fov_radius, render_mode, training)
        env = Monitor(env, filename=None)
        return env
    return _init