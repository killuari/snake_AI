"""
main.py - Training & Testing Pipeline for the Snake RL Agent

This is the central entry point for the project. It provides functions to:
    - play_game():          Play Snake yourself with WASD controls.
    - train_model():        Train a DQN or PPO agent with parallel environments.
    - test_model():         Load a saved model and watch it play visually.
    - test_environment():   Manually play the game via terminal input (debugging tool).

Usage:
    Run this file directly (`python main.py`) to launch the graphical launcher UI
    (see ui.py), which lets you choose Play / Test Model / Train Model and
    configure parameters. The functions above can also be imported and called
    directly, e.g. for scripting or debugging.
"""

from snake_game import SnakeGame, Direction, draw_hud, COLOR_BACKGROUND
from snake_game_environment import SnakeGameEnvironment, make_snake_env
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
from DQN_hyper_tuning import run_hyperparameter_optimization, load_best_params
from custom_callback import DeathLogger
from feature_extractors import SnakeCombinedExtractor
import os
import json
import glob
import numpy as np
import pygame

# Default grid cell size in pixels (used for all environments created here)
GRID_SIZE = 30

# Paths for saving/loading trained models and training logs
LOG_PATH = os.path.join("Training", "Logs")
PPO_PATH = os.path.join("Training", "Saved Models", "PPO")
DQN_PATH = os.path.join("Training", "Saved Models", "DQN")


def _finalize_checkpoint(path, prefix, total_timesteps):
    """
    Rename path/{prefix}.zip (just written by model.save()/EvalCallback) to
    path/{prefix}_{total_timesteps}.zip, so the filename itself records how many
    timesteps the checkpoint was trained for. Replaces any stale checkpoint left
    over from a previous training run under the same prefix.
    """
    plain_path = os.path.join(path, f"{prefix}.zip")
    if not os.path.exists(plain_path):
        return
    for stale in glob.glob(os.path.join(path, f"{prefix}_*.zip")):
        os.remove(stale)
    os.rename(plain_path, os.path.join(path, f"{prefix}_{total_timesteps}.zip"))


def _find_checkpoint(path, prefix):
    """Find the (single) path/{prefix}_{timesteps}.zip checkpoint written by _finalize_checkpoint()."""
    matches = glob.glob(os.path.join(path, f"{prefix}_*.zip"))
    if not matches:
        raise FileNotFoundError(f"No '{prefix}' checkpoint found in {path}")
    return max(matches, key=os.path.getmtime)


def evaluate_model_performance(model, grid_size, grid_width, grid_height, snake_fov_radius, obs_mode="flat", n_episodes=10, model_label=""):
    """
    Run n_episodes deterministic and n_episodes stochastic episodes (no rendering,
    no reward shaping) and return the mean "clean" score (apples eaten) for each,
    the same score shown by test_model().
    """
    if model_label:
        print(f"Evaluating {model_label}...")

    # training=False disables the environment's own anti-loop timeout, so wrap
    # with a hard step limit -- otherwise an undertrained/looping policy (quite
    # possible here, especially in the stochastic episodes) can hang this
    # function indefinitely instead of just producing a low score.
    env = TimeLimit(
        SnakeGameEnvironment(grid_size, grid_width, grid_height, snake_fov_radius, render_mode=None, training=False, obs_mode=obs_mode),
        max_episode_steps=10000,
    )
    results = {}
    for label, deterministic in [("deterministic", True), ("stochastic", False)]:
        scores = []
        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            score = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                score += float(reward)
            scores.append(score)
        results[label] = {"mean_score": sum(scores) / len(scores), "episodes": n_episodes, "scores": scores}
        print(f"  {label}: mean score = {results[label]['mean_score']:.2f} ({n_episodes} episodes)")
    env.close()
    return results


def train_model(model_name="DQN", grid_width=30, grid_height=20, snake_fov_radius=3, timesteps=3_000_000, num_envs=4, new=True, params=None, best=True, use_tuned_params=False, use_cnn=False):
    """
    Train a DQN or PPO agent on the Snake environment.

    Creates parallel training environments via SubprocVecEnv and a separate
    evaluation environment. Periodically evaluates the agent, saves the best
    model, and optionally stops training early if a reward threshold is reached.

    Args:
        model_name:       "DQN" or "PPO".
        grid_width:       Grid width in cells.
        grid_height:      Grid height in cells.
        snake_fov_radius: Agent's field-of-view radius.
        timesteps:        Total training timesteps.
        num_envs:         Number of parallel environments (subprocesses).
        new:              If True, create a fresh model. If False, load from disk.
        params:           Optional dict of hyperparameters (DQN only). Uses defaults if None.
        best:             If loading (new=False), load the "best_model" (True) or "last_model" (False)
                          checkpoint (filenames carry the timesteps they were trained for, e.g.
                          "best_model_3000000.zip" -- see _find_checkpoint()).
        use_tuned_params: If True and params is None (DQN only), load hyperparameters
                          previously found by DQN_hyper_tuning.run_hyperparameter_optimization()
                          instead of using the hardcoded defaults.
        use_cnn:          If True, use the "grid" observation mode (2D FOV + apple direction)
                          with a custom CNN feature extractor (SnakeCombinedExtractor) instead
                          of the flat MultiDiscrete + MlpPolicy setup. Saved under a separate
                          "GRID" folder, incompatible with "FLAT" (MLP) models.
    """
    obs_mode = "grid" if use_cnn else "flat"
    policy = "MultiInputPolicy" if use_cnn else "MlpPolicy"
    policy_kwargs = {"features_extractor_class": SnakeCombinedExtractor} if use_cnn else None

    # Create parallel training environments (one per subprocess)
    train_env = SubprocVecEnv([make_snake_env(GRID_SIZE, grid_width, grid_height, snake_fov_radius, obs_mode=obs_mode) for _ in range(num_envs)])

    # Create a single evaluation environment with a step limit to prevent infinite episodes.
    # training=False disables reward shaping/penalties, so EvalCallback's mean_reward
    # reflects the clean score (apples eaten) instead of being mixed with training-only
    # shaping terms.
    raw_eval_env = SnakeGameEnvironment(GRID_SIZE, grid_width, grid_height, snake_fov_radius, training=False, obs_mode=obs_mode)
    raw_eval_env = TimeLimit(raw_eval_env, max_episode_steps=10000)
    eval_env = Monitor(raw_eval_env, filename=None)

    if model_name == "DQN":
        # Build the save/load path based on obs_mode, grid size, and FOV
        path = os.path.join(DQN_PATH, "GRID" if use_cnn else "FLAT", f"GRID_{grid_width}_{grid_height}", f"FOV_RADIUS_{snake_fov_radius}")
        if not new:
            # Resume training from a saved model (policy/feature-extractor are restored from the checkpoint)
            model = DQN.load(_find_checkpoint(path, "best_model" if best else "last_model"), train_env, device='cpu', verbose=1)
        else:
            # Default DQN hyperparameters, or previously tuned ones from Optuna
            if params is None:
                if use_tuned_params:
                    params = load_best_params()
                else:
                    params = {
                        "learning_rate": 6e-5,
                        "buffer_size": 2_000_000,
                        "learning_starts": 50_000,      # Steps before training starts (fill replay buffer)
                        "batch_size": 256,
                        "tau": 0.4,                      # Soft update coefficient for target network
                        "gamma": 0.988,                  # Discount factor
                        "train_freq": 4,                 # Train every N steps
                        "gradient_steps": 4,             # Gradient updates per training step
                        "target_update_interval": 2000,  # Steps between target network hard updates
                        "exploration_fraction": 0.2,     # Fraction of training for epsilon decay
                        "exploration_final_eps": 0.05,   # Final exploration rate
                    }

            model = DQN(
                policy, train_env,
                policy_kwargs = policy_kwargs,
                learning_rate = params["learning_rate"],
                buffer_size = params["buffer_size"],
                learning_starts = params["learning_starts"],
                batch_size = params["batch_size"],
                tau = params["tau"],
                gamma = params["gamma"],
                train_freq = params["train_freq"],
                gradient_steps = params["gradient_steps"],
                target_update_interval = params["target_update_interval"],
                exploration_fraction = params["exploration_fraction"],
                exploration_final_eps = params["exploration_final_eps"],
                device='cpu', verbose=1
            )
    else:
        # PPO branch
        path = os.path.join(PPO_PATH, "GRID" if use_cnn else "FLAT", f"GRID_{grid_width}_{grid_height}", f"FOV_RADIUS_{snake_fov_radius}")
        if not new:
            model = PPO.load(path=_find_checkpoint(path, "best_model" if best else "last_model"), env=train_env, device='cpu', verbose=1, force_reset=True)

            print(f"Successfully loaded PPO Model ({model._total_timesteps} total_timesteps) [from Path: {path}]")
        else:
            model = PPO(
                policy, train_env,
                policy_kwargs = policy_kwargs,
                n_steps=5000,          # Steps per rollout before policy update
                batch_size=1000,       # Minibatch size for PPO updates
                n_epochs=4,            # Passes over each rollout (reduced from SB3's default 10,
                                       # since batch_size=250/n_epochs=10 meant hundreds of gradient
                                       # steps per rollout -- too many updates on the same on-policy
                                       # data, risking overfitting/instability)
                ent_coef=0.01,         # Small entropy bonus: SB3's PPO default is 0.0 (no exploration
                                       # incentive at all), risky with Snake's sparse reward
                device='cpu', verbose=1
            )

    # Stop training early once mean reward is clearly very good. Scaled to board
    # size (not a fixed constant) since a given score means very different things
    # on a small vs. a large grid -- ~30% of all cells eaten as apples.
    reward_threshold = max(10, int(0.3 * grid_width * grid_height))
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)

    # Evaluate periodically, save best model, and optionally stop early
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        best_model_save_path=path,
        eval_freq=max((timesteps // 25) // num_envs, 1),    # ~25 evaluations per training run
        n_eval_episodes=10,                                  # Episodes per evaluation
        verbose=1,
        deterministic=True
    )

    # Custom callback to log death causes (max-step vs collision)
    death_logger = DeathLogger()

    # Start training
    model.learn(total_timesteps=timesteps, callback=[eval_callback, death_logger], reset_num_timesteps=False)

    # EvalCallback only evaluates every eval_freq calls, so the final stretch of
    # training (up to eval_freq-1 calls) may never have been checked -- evaluate
    # once more now so the just-finished model is guaranteed to be compared
    # against the best one seen so far, and saved as the new best if it wins.
    final_mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=eval_callback.n_eval_episodes, deterministic=True)

    def _scalar_mean_reward(value) -> float:
        if isinstance(value, (list, tuple, np.ndarray)):
            return float(np.mean(value))
        return float(value)

    final_mean_reward = _scalar_mean_reward(final_mean_reward)
    best_mean_reward = _scalar_mean_reward(eval_callback.best_mean_reward)
    if final_mean_reward > best_mean_reward:
        print(f"Final evaluation found a new best model: {final_mean_reward:.2f} (previous best: {best_mean_reward:.2f})")
        model.save(os.path.join(path, "best_model"))

    # Always save the final model (in addition to the best model saved by EvalCallback)
    model.save(os.path.join(path, "last_model"))

    # Bake the cumulative trained timesteps into the checkpoint filenames (e.g.
    # "best_model_3000000.zip"). model.num_timesteps already reflects the right
    # value here: it started at 0 for a fresh model (new=True), or continued
    # counting up from the loaded checkpoint's timesteps (new=False, since
    # model.learn() above was called with reset_num_timesteps=False) -- so
    # continuing training accumulates instead of overwriting the count.
    total_timesteps_trained = model.num_timesteps
    _finalize_checkpoint(path, "best_model", total_timesteps_trained)
    _finalize_checkpoint(path, "last_model", total_timesteps_trained)

    # Clean up environments
    train_env.close()
    eval_env.close()

    # Evaluate both checkpoints in this folder (deterministic + stochastic, no
    # rendering, clean score) and save the results so performance can be read
    # at a glance without manually testing the model.
    print("\n=== Starting post-training evaluation ===")
    evaluation = {
        "timesteps": total_timesteps_trained,
        "last_model": evaluate_model_performance(model, GRID_SIZE, grid_width, grid_height, snake_fov_radius, obs_mode=obs_mode, model_label="last model"),
    }

    if glob.glob(os.path.join(path, "best_model_*.zip")):
        ModelClass = DQN if model_name == "DQN" else PPO
        best = ModelClass.load(_find_checkpoint(path, "best_model"), device="cpu")
        evaluation["best_model"] = evaluate_model_performance(best, GRID_SIZE, grid_width, grid_height, snake_fov_radius, obs_mode=obs_mode, model_label="best model")

    evaluation_path = os.path.join(path, "evaluation.json")
    print(f"\nEvaluation complete. Writing results to {evaluation_path}")
    with open(evaluation_path, "w") as file:
        json.dump(evaluation, file, indent=4)


def test_model(model_name="DQN", grid_width=30, grid_height=20, snake_fov_radius=1, use_cnn=False, fps=None):
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

    # Run the agent until the episode ends
    while not done:
        action, _ = model.predict(obs, deterministic=True)   # Use greedy policy (no exploration)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += float(reward)

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


if __name__ == "__main__":
    from ui import App
    App().mainloop()