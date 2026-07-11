"""
rl/playback.py - Playing/watching the Snake game (human play and trained-model playback).

Functions:
    play_game():          Play Snake yourself with WASD controls.
    test_model():          Load a saved model and watch it play visually.
    test_environment():    Manually play the environment via terminal input (debugging tool).
"""

import os
import numpy as np

from game.snake_game import SnakeGame, Direction, draw_hud, COLOR_BACKGROUND, COLOR_SCORE_TEXT, COLOR_SCORE_PANEL
from game.environment import make_snake_env
from game.game_over import run_game_over
from rl.paths import GRID_SIZE, PPO_PATH, DQN_PATH, _find_checkpoint
# Not referenced by name below -- importing it registers the "feature_extractors"
# sys.modules alias (see rl/feature_extractors.py) needed to unpickle GRID-mode
# checkpoints saved before the package refactor, before any PPO/DQN.load() runs.
import rl.feature_extractors  # noqa: F401

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
    model observes and an arrow for the apple-direction observation. When the
    snake dies, a game-over overlay offers Restart (a fresh episode in the
    same window) or Quit; ESC or closing the window quits directly.
    """
    obs_mode = "grid" if use_cnn else "flat"
    obs_mode_dir = "GRID" if use_cnn else "FLAT"

    # Create environment with human rendering (opens Pygame window)
    env = make_snake_env(GRID_SIZE, grid_width, grid_height, snake_fov_radius, "human", training=False, obs_mode=obs_mode, render_fps=fps)()

    model_class = DQN if model_name == "DQN" else PPO
    base_path = DQN_PATH if model_name == "DQN" else PPO_PATH
    checkpoint_dir = os.path.join(base_path, obs_mode_dir, f"GRID_{grid_width}_{grid_height}", f"FOV_RADIUS_{snake_fov_radius}")
    load_path = _find_checkpoint(checkpoint_dir, "best_model")
    try:
        model = model_class.load(load_path, env, device="cpu")
    except Exception as exc:
        # Bare exceptions from .load() (e.g. a ModuleNotFoundError from an old
        # checkpoint's pickled class references -- see rl/feature_extractors.py)
        # give no clue *which* checkpoint failed once this reaches the UI's log
        # box (SubScreen._start_background just prints str(exc)) -- add the path.
        raise RuntimeError(f"Failed to load checkpoint {load_path}: {exc}") from exc
    if model_name == "PPO":
        print(f"Successfully loaded PPO Model ({model._total_timesteps} total_timesteps)\n[from Path: {load_path}]")

    # Monitor -> SnakeGameEnvironment: the base env owns screen/clock/snakeGame,
    # which the game-over overlay needs (Monitor doesn't expose them itself).
    base = env.unwrapped
    final_score = 0

    try:
        while True:  # restart loop -- exits via break, either branch below
            obs, info = env.reset()
            done = False
            try:
                while not done:
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
            except SystemExit:
                # Window closed or ESC: _render_frame() already tore down the
                # display, so there's no surface left to show an overlay on.
                if base.snakeGame is not None:
                    final_score = base.snakeGame.score
                break

            final_score = base.snakeGame.score
            if run_game_over(base.screen, base.clock, final_score) == "quit":
                break
            # else "restart": loop continues; env.reset() rebuilds the game in
            # the same, still-open window.
    finally:
        # Without this, the Pygame window is left open with a dead event loop once
        # the episode ends (nothing calls pygame.event.get() again), so it can't
        # even process its own close button anymore. Safe to call again even if
        # the SystemExit path above already closed it (close() is idempotent).
        env.close()

    print(f"Ended with Score: {final_score}")


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


def _draw_press_to_start(screen, font):
    """
    Overlay shown until the first WASD press. Without this, the snake started
    moving RIGHT the instant the window opened -- a player who hadn't even
    located the window yet could lose a life to the wall before ever pressing
    a key. Styled like draw_hud()'s score panel for visual consistency.
    """
    text = font.render("Press W / A / S / D to start", True, COLOR_SCORE_TEXT)
    padding = 16
    panel_rect = pygame.Rect(0, 0, text.get_width() + 2 * padding, text.get_height() + 2 * padding)
    panel_rect.center = screen.get_rect().center

    panel = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
    r, g, b, _ = COLOR_SCORE_PANEL
    pygame.draw.rect(panel, pygame.Color(r, g, b, 210), panel.get_rect(), border_radius=12)

    screen.blit(panel, panel_rect.topleft)
    screen.blit(text, (panel_rect.x + padding, panel_rect.y + padding))


def play_game(grid_width=30, grid_height=20, fps=10):
    """
    Play Snake yourself in a Pygame window.

    Controls: 'w'=UP, 'a'=LEFT, 's'=DOWN, 'd'=RIGHT. The snake stands still
    (with a "press to start" prompt) until the first directional key is
    pressed. When it dies, a game-over overlay offers Restart (a fresh game
    in the same window) or Quit; ESC or closing the window quits directly.
    """
    pygame.init()
    font = pygame.font.Font(None, 45)
    screen = pygame.display.set_mode((GRID_SIZE * grid_width, GRID_SIZE * grid_height))
    clock = pygame.time.Clock()

    def _poll_quit():
        """Handle quit events (window close or ESC key). Returns True if the
        player wants to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False

    def _draw_frame(game, started):
        screen.fill(COLOR_BACKGROUND)
        game.draw(screen)
        draw_hud(screen, font, game.score)
        if not started:
            _draw_press_to_start(screen, font)
        pygame.display.flip()

    final_score = 0
    while True:  # restart loop -- exits via break, either branch below
        direction = Direction.RIGHT
        game = SnakeGame(GRID_SIZE, grid_width, grid_height)
        started = False
        died = False
        user_quit = False

        while not died and not user_quit:
            if _poll_quit():
                user_quit = True
                break

            # Read keyboard input for direction changes.
            # Prevent reversing direction (e.g., can't go DOWN while moving UP).
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w] and direction != Direction.DOWN:
                direction = Direction.UP
                started = True
            elif keys[pygame.K_s] and direction != Direction.UP:
                direction = Direction.DOWN
                started = True
            elif keys[pygame.K_a] and direction != Direction.RIGHT:
                direction = Direction.LEFT
                started = True
            elif keys[pygame.K_d] and direction != Direction.LEFT:
                direction = Direction.RIGHT
                started = True

            if started:
                # Move the snake; if it dies, this is the death frame -- still
                # drawn and flipped below before the outer while-check exits,
                # so run_game_over() below has the actual moment of death to
                # freeze, not a blank/stale frame.
                alive = game.move_snake(direction)
                if not alive:
                    died = True

                # Check and handle apple eating
                game.eat_apple()

            _draw_frame(game, started)

            # Hold this frame for 1000/fps ms, redrawing at a fixed ~60 FPS during
            # the wait instead of a single blit -- decouples the apple's
            # idle-shimmer/eat-ring animation smoothness (both real-time-based,
            # see game/snake_game.py) from the chosen game speed, without
            # changing how fast the snake actually moves. Skipped on the death
            # frame so the game-over handoff below still freezes the exact
            # moment of death, not a later redraw. Uses wall-clock time
            # (get_ticks()), not accumulated clock.tick(60) deltas, which
            # quantize in ~16.7ms steps and would overshoot noticeably at
            # speeds close to or above 60fps.
            if died or fps >= 60:
                clock.tick(fps)
            else:
                target_ms = 1000.0 / fps
                start = pygame.time.get_ticks()
                while True:
                    if _poll_quit():
                        user_quit = True
                        break
                    _draw_frame(game, started)
                    clock.tick(60)
                    if pygame.time.get_ticks() - start >= target_ms:
                        break

        final_score = game.score
        if user_quit:
            break
        if run_game_over(screen, clock, final_score) == "quit":
            break
        # else "restart": loop continues, a fresh game starts in the same window

    pygame.quit()
    print(f"Ended with Score: {final_score}")
