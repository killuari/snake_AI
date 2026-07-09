"""
rl/playback.py - Playing/watching the Snake game (human play and trained-model playback).

Functions:
    play_game():          Play Snake yourself with WASD controls.
    test_model():          Load a saved model and watch it play visually.
    test_environment():    Manually play the environment via terminal input (debugging tool).
"""

import os
import numpy as np

from game.snake_game import SnakeGame, Direction, draw_hud, COLOR_BACKGROUND
from game.environment import make_snake_env
from rl.paths import GRID_SIZE, PPO_PATH, DQN_PATH, _find_checkpoint

import pygame
from stable_baselines3 import PPO, DQN


def test_model(model_name="DQN", grid_width=30, grid_height=20, snake_fov_radius=1, use_cnn=False, fps=None, deterministic=True):
    """
    Load a trained model and watch it play in a Pygame window.

    Args:
        model_name:       "DQN" or "PPO".
        grid_width:       Grid width (must match the model's training config).
        grid_height:      Grid height (must match the model's training config).
        snake_fov_radius: FOV radius (must match the model's training config).
        use_cnn:          Must match the obs_mode/policy the model was trained with.
        fps:              Playback speed (frames = model decisions per second).
                          Defaults to 50 if None. Lower it (e.g. 5-10) to follow
                          along move by move, raise it to skim through episodes.
        deterministic:    If True (default), always pick the greedy action. If False,
                          sample from the policy's action distribution instead (matches
                          the "stochastic" evaluation mode in evaluate_model_performance()).

    Controls while watching: 'f' toggles a debug overlay showing the FOV the
    model observes and an arrow for the apple-direction observation; ESC or
    closing the window exits.
    """
    obs_mode = "grid" if use_cnn else "flat"
    obs_mode_dir = "GRID" if use_cnn else "FLAT"

    # Create environment with human rendering (opens Pygame window)
    env = make_snake_env(GRID_SIZE, grid_width, grid_height, snake_fov_radius, "human", training=False, obs_mode=obs_mode, render_fps=fps)()

    if model_name == "DQN":
        checkpoint_dir = os.path.join(DQN_PATH, obs_mode_dir, f"GRID_{grid_width}_{grid_height}", f"FOV_RADIUS_{snake_fov_radius}")
        model = DQN.load(_find_checkpoint(checkpoint_dir, "best_model"), env, device="cpu")
    else:
        checkpoint_dir = os.path.join(PPO_PATH, obs_mode_dir, f"GRID_{grid_width}_{grid_height}", f"FOV_RADIUS_{snake_fov_radius}")
        load_path = _find_checkpoint(checkpoint_dir, "best_model")
        model = PPO.load(load_path, env, device="cpu")
        print(f"Successfully loaded PPO Model ({model._total_timesteps} total_timesteps)\n[from Path: {load_path}]")

    obs, info = env.reset()
    done = False
    score = 0.0

    try:
        # Run the agent until the episode ends
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += float(reward)
    finally:
        # Without this, the Pygame window is left open with a dead event loop once
        # the episode ends (nothing calls pygame.event.get() again), so it can't
        # even process its own close button anymore.
        env.close()

    print(f"Ended with Score: {score}")


def test_environment(grid_width=30, grid_height=20, snake_fov_radius=1):
    """
    Debug tool: manually play the environment via terminal input.

    Controls: 'w'=UP, 'a'=LEFT, 's'=DOWN, 'd'=RIGHT.
    Prints the observation array after each step.
    """
    env = make_snake_env(GRID_SIZE, grid_width, grid_height, snake_fov_radius, "human", training=False)()

    obs, info = env.reset()
    done = False
    score = 0.0
    key_map = {"d": 0, "s": 1, "a": 2, "w": 3}  # RIGHT, DOWN, LEFT, UP

    try:
        while not done:
            raw = input("action: ")
            if raw not in key_map:
                print("Invalid input, please use w/a/s/d.")
                continue
            action = np.array(key_map[raw])
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += float(reward)
            print(obs)
    finally:
        env.close()


def play_game(grid_width=30, grid_height=20, fps=10):
    """
    Play Snake yourself in a Pygame window.

    Controls: 'w'=UP, 'a'=LEFT, 's'=DOWN, 'd'=RIGHT. ESC or closing the window quits.
    """
    pygame.init()
    font = pygame.font.Font(None, 45)
    screen = pygame.display.set_mode((GRID_SIZE * grid_width, GRID_SIZE * grid_height))
    clock = pygame.time.Clock()

    direction = Direction.RIGHT
    game = SnakeGame(GRID_SIZE, grid_width, grid_height)
    running = True

    while running:
        clock.tick(fps)

        # Handle quit events (window close or ESC key)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        if not running:
            break

        screen.fill(COLOR_BACKGROUND)

        # Read keyboard input for direction changes.
        # Prevent reversing direction (e.g., can't go DOWN while moving UP).
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] and direction != Direction.DOWN:
            direction = Direction.UP
        elif keys[pygame.K_s] and direction != Direction.UP:
            direction = Direction.DOWN
        elif keys[pygame.K_a] and direction != Direction.RIGHT:
            direction = Direction.LEFT
        elif keys[pygame.K_d] and direction != Direction.LEFT:
            direction = Direction.RIGHT

        # Move the snake; if it dies, stop the game loop
        alive = game.move_snake(direction)
        if not alive:
            running = False

        # Check and handle apple eating
        game.eat_apple()

        # Draw everything
        game.draw(screen)
        draw_hud(screen, font, game.score)

        pygame.display.flip()

    pygame.quit()
    print(f"Ended with Score: {game.score}")
