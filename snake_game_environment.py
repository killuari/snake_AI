import pygame
import gymnasium as gym
import numpy as np
from typing import Optional

from snake_game import SnakeGame, Direction

class SnakeGameEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, grid_size, grid_width, grid_height, render_mode = None):
        self.grid_size = grid_size
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.max_length = grid_width * grid_height

        self.snakeGame = None

        self.screen = None
        self.clock = None

        self.head_location = np.array([-1, -1])
        self.apple_location = np.array([-1, -1])
        self.tail_locations = []

        coord_space = gym.spaces.MultiDiscrete([self.grid_width, self.grid_height])

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
        tail_locations = np.full((self.max_length - 1, 2), -1, dtype=np.int32)

        for idx, loc in enumerate(self.tail_locations):
            if idx < len(tail_locations):
                tail_locations[idx] = loc

        return {
            "head": self.head_location, 
            "apple": self.apple_location, 
            "tail": tail_locations
        }
    
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

        self.update_locations()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        dir = self.action_to_direction[action]

        reward = 0
        terminated = False

        self.snakeGame.move_snake(dir)
        if self.snakeGame.detect_collision():
            terminated = True
            reward = -1

        apple_eaten = self.snakeGame.eat_apple()
        
        if apple_eaten:
            reward = 1

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
            self.screen.blit(canvas)
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