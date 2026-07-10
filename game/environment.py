"""
snake_game_environment.py - Gymnasium Environment Wrapper for the Snake Game

Wraps SnakeGame into a gym.Env for RL training with Stable Baselines3.
Uses a local FOV observation (plus the apple's direction) so models
generalize across grid sizes.

FOV cell encoding: 0=empty, 1=body, 2=apple, 3=wall.
"""

# Imported first (before `import pygame` below): game.snake_game's module-level
# pygame startup hygiene (env vars + warnings filters, see its docstring) must
# run before pygame is imported anywhere in this process. A SubprocVecEnv
# worker reconstructing a pickled env only imports what it needs to resolve
# SnakeGameEnvironment -- i.e. this module -- so if `import pygame` ran first
# right here, that worker's AVX2/pkg_resources warnings would already have
# fired before game.snake_game's protective code got a chance to run.
from game.snake_game import SnakeGame, Direction, COLOR_BACKGROUND, draw_hud

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

import pygame
import gymnasium as gym
import numpy as np
import random
from typing import Optional
from stable_baselines3.common.monitor import Monitor


class SnakeGameEnvironment(gym.Env):
    """
    Gymnasium environment wrapping the Snake game for RL training.

    Observation: local FOV around the head plus the apple's direction, in one
    of two layouts (see `obs_mode`).
    Action: Discrete(4) - 0=RIGHT, 1=DOWN, 2=LEFT, 3=UP.

    Args:
        grid_size: Pixel size of each cell.
        grid_width/height: Grid dimensions in cells.
        snake_fov_radius: FOV radius around head ((2r+1)^2 - 1 cells).
        render_mode: None, "human", or "rgb_array".
        training: If True, apply reward shaping (penalties). False for eval.
        obs_mode: "flat" - MultiDiscrete vector for an MlpPolicy (default).
                  "grid" - Dict({"grid": Box(4,2r+1,2r+1), "apple_dir": MultiDiscrete([3,3])})
                           for a CNN-based policy that keeps the FOV's 2D structure.
        render_fps: Frames (= model decisions) per second in human render mode.
                    Defaults to metadata["render_fps"] if None.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, grid_size, grid_width, grid_height, snake_fov_radius = 1, render_mode = None, training = True, obs_mode = "flat", render_fps = None):
        self.grid_size = grid_size
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.snake_fov_radius = snake_fov_radius
        self.training = training
        assert obs_mode in ("flat", "grid")
        self.obs_mode = obs_mode
        self.render_fps = render_fps or self.metadata["render_fps"]

        self.snakeGame = None          # Created on reset()

        # Pygame objects (lazy-initialized on first render)
        self.screen = None
        self.clock = None

        # Debug visualization (human render mode only), toggled with the 'f' key:
        # highlights the FOV the model currently observes and shows an arrow
        # for the apple-direction observation feature.
        self.show_debug_overlay = False

        # Cached positions for building observations
        self.head_location = np.array([-1, -1])
        self.apple_location = np.array([-1, -1])
        self.tail_locations = []
        self.dir = None

        # Observation: FOV cells (excl. head, values 0-3) + apple direction (2 values, 0-2).
        n_cells = (2*self.snake_fov_radius + 1)**2 - 1
        if self.obs_mode == "flat":
            self.observation_space = gym.spaces.MultiDiscrete([4] * n_cells + [3, 3])
        else:
            fov_side = 2*self.snake_fov_radius + 1
            self.observation_space = gym.spaces.Dict({
                "grid": gym.spaces.Box(low=0, high=1, shape=(4, fov_side, fov_side), dtype=np.uint8),
                "apple_dir": gym.spaces.MultiDiscrete([3, 3]),
            })

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

    def classify_cell(self, loc: np.ndarray) -> int:
        """Classify a single cell in the FOV."""
        if loc[0] < 0 or loc[0] >= self.grid_width or loc[1] < 0 or loc[1] >= self.grid_height:
            return 3      # Wall / out of bounds
        elif np.array_equal(loc, self.apple_location):
            return 2      # Apple
        elif any(np.array_equal(loc, tail_part) for tail_part in self.tail_locations):
            return 1      # Snake body
        else:
            return 0      # Empty

    def _apple_direction(self):
        """Sign of the apple's offset from the head, mapped from {-1,0,1} to {0,1,2}
        per axis. Scale-invariant (direction only, no distance), so it doesn't
        depend on grid size, and gives the agent a signal towards the apple even
        when it is outside the FOV."""
        hx, hy = self.head_location
        ax, ay = self.apple_location
        return int(np.sign(ax - hx)) + 1, int(np.sign(ay - hy)) + 1

    def _get_obs(self):
        """Build the observation from the current game state, in whichever
        layout `obs_mode` selects."""
        if self.obs_mode == "flat":
            return self._get_obs_flat()
        return self._get_obs_grid()

    def _get_obs_flat(self):
        """Flat vector: FOV cells (0=empty, 1=body, 2=apple, 3=wall) followed by
        the apple direction (dx_sign, dy_sign)."""
        locations = []
        hx, hy = self.head_location

        for dy in range(-self.snake_fov_radius, self.snake_fov_radius + 1):
            for dx in range(-self.snake_fov_radius, self.snake_fov_radius + 1):
                if dx == 0 and dy == 0:
                    continue  # Skip head position

                loc = np.array([hx + dx, hy + dy])
                classification = self.classify_cell(loc)

                locations.append(classification)

        dx_sign, dy_sign = self._apple_direction()
        locations.append(dx_sign)
        locations.append(dy_sign)

        return np.array(locations)

    def _get_obs_grid(self):
        """2D one-hot FOV grid (4 channels: empty/body/apple/wall) plus the
        apple direction as a separate small vector, for a CNN-based policy.
        The head's own cell is left all-zero across every channel (it is
        always exactly the center of the grid)."""
        fov_side = 2*self.snake_fov_radius + 1
        grid = np.zeros((4, fov_side, fov_side), dtype=np.uint8)
        hx, hy = self.head_location

        for dy in range(-self.snake_fov_radius, self.snake_fov_radius + 1):
            for dx in range(-self.snake_fov_radius, self.snake_fov_radius + 1):
                if dx == 0 and dy == 0:
                    continue  # Head cell stays all-zero

                loc = np.array([hx + dx, hy + dy])
                classification = self.classify_cell(loc)
                grid[classification, dy + self.snake_fov_radius, dx + self.snake_fov_radius] = 1

        dx_sign, dy_sign = self._apple_direction()
        return {"grid": grid, "apple_dir": np.array([dx_sign, dy_sign])}
    
    def _get_info(self) -> dict:
        """Return auxiliary info dict with current snake length."""
        if self.snakeGame is None:
            return {"snake_length": 0}

        return {"snake_length": len(self.snakeGame.snake_list)}
    
    def update_locations(self):
        """Cache head, apple, and tail positions from game state."""
        if self.snakeGame is None:
            self.head_location = np.array([-1, -1])
            self.apple_location = np.array([-1, -1])
            self.tail_locations = []
            return

        self.head_location = self.snakeGame.head.grid_pos
        self.apple_location = self.snakeGame.apple.grid_pos
        self.tail_locations = self.snakeGame.get_tail_locations()
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset to a fresh game. Returns (observation, info)."""
        super().reset(seed=seed, options=options)
        if seed is not None:
            random.seed(seed)  # SnakeGame/Apple use the global `random` module

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
            -0.5 for exceeding max_steps (anti-loop, truncates).
            -min(20, max(1, (length-3)*0.5)) for death (scales with length, terminates).

        The anti-loop timeout is a `truncated` event (an artificial cutoff, not
        a real game-over), while a wall/self collision is a `terminated` event
        (a genuine terminal state). Keeping this distinction lets SB3 bootstrap
        the value of the cutoff state correctly instead of treating it as an
        absorbing terminal state with zero future value.

        Returns: (observation, reward, terminated, truncated, info)
        """
        if self.snakeGame is None:
            raise RuntimeError("Call reset() before step().")

        if self.dir is None:
            raise RuntimeError("Direction not initialized. Call reset() first.")

        action = action.item()
        # Prevent 180° reversal (would cause instant self-collision)
        if self.action_to_direction[action] != self.dir.opposite():
            self.dir = self.action_to_direction[action]

        reward = 0
        terminated = False
        truncated = False
        death_cause = None
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
            truncated = True           # Anti-loop cutoff, not a real game-over
            death_cause = "timeout"
            reward = -0.5              # Anti-loop penalty

        if not alive:
            terminated = True
            death_cause = "collision"
            if self.training:
                # Death penalty scaled by snake length (min -1, max -20)
                reward = -min(20, max(1, (snake_length-3) * 0.5))

        self.update_locations()

        observation = self._get_obs()
        info = self._get_info()
        info["death_cause"] = death_cause

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info
    
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

        if self.snakeGame is None:
            raise RuntimeError("Call reset() before rendering.")

        # Draw on off-screen canvas
        canvas = pygame.Surface((self.grid_size * self.grid_width, self.grid_size * self.grid_height))
        canvas.fill(COLOR_BACKGROUND)

        self.snakeGame.draw(canvas)

        if self.show_debug_overlay:
            self._draw_fov_overlay(canvas)

        # Render score overlay (font only exists in human mode)
        if self.font is not None:
            draw_hud(canvas, self.font, self.snakeGame.score)

        if self.show_debug_overlay:
            self._draw_apple_direction_arrow(canvas)

        if self.render_mode == "human" and self.screen is not None and self.clock is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.close()
                    raise SystemExit
                if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                    self.show_debug_overlay = not self.show_debug_overlay

            self.screen.blit(canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.render_fps)
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _draw_fov_overlay(self, canvas):
        """Highlight the (2r+1)x(2r+1) FOV window the model currently observes."""
        hx, hy = self.head_location
        r = self.snake_fov_radius
        size = self.grid_size * (2*r + 1)

        overlay = pygame.Surface((size, size), pygame.SRCALPHA)
        overlay.fill(pygame.Color(255, 220, 90, 55))
        pygame.draw.rect(overlay, pygame.Color(255, 220, 90, 170), overlay.get_rect(), width=3)

        # Pygame clips blits at the canvas edge automatically, so no special
        # handling is needed when the head is close to a wall.
        top_left = ((hx - r) * self.grid_size, (hy - r) * self.grid_size)
        canvas.blit(overlay, top_left)

    def _draw_apple_direction_arrow(self, canvas):
        """Draw a small fixed arrow icon (top-right corner) pointing in the
        direction of the apple_dir observation feature."""
        dx_sign, dy_sign = self._apple_direction()
        dx, dy = dx_sign - 1, dy_sign - 1      # back from {0,1,2} to {-1,0,1}
        if dx == 0 and dy == 0:
            return

        direction = pygame.Vector2(dx, dy)
        if direction.length() > 0:
            direction = direction.normalize()

        box_size = 56
        margin = 16
        icon = pygame.Surface((box_size, box_size), pygame.SRCALPHA)
        center = pygame.Vector2(box_size / 2, box_size / 2)
        pygame.draw.circle(icon, pygame.Color(15, 16, 24, 180), center, box_size / 2)

        arrow_len = box_size * 0.32
        tip = center + direction * arrow_len
        back = center - direction * arrow_len * 0.6
        perp = pygame.Vector2(-direction.y, direction.x)
        left = back + perp * arrow_len * 0.45
        right = back - perp * arrow_len * 0.45
        pygame.draw.polygon(icon, pygame.Color(255, 210, 90), [tip, left, right])

        canvas.blit(icon, (canvas.get_width() - margin - box_size, margin))

    def close(self):
        """Clean up Pygame resources."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


def make_snake_env(grid_size, grid_width, grid_height, snake_fov_radius = 1, render_mode = None, training = True, obs_mode = "flat", render_fps = None):
    """Factory returning a callable that creates a Monitor-wrapped environment.
    Required by SubprocVecEnv (one callable per subprocess)."""
    def _init():
        env = SnakeGameEnvironment(grid_size, grid_width, grid_height, snake_fov_radius, render_mode, training, obs_mode, render_fps)
        env = Monitor(env, filename=None)
        return env
    return _init