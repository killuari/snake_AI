"""
main.py — Training & Testing Pipeline for the Snake RL Agent

This is the central entry point for the project. It provides functions to:
    - train_model():        Train a DQN or PPO agent with parallel environments.
    - test_model():         Load a saved model and watch it play visually.
    - test_environment():   Manually play the game via terminal input (debugging tool).

Usage:
    Modify the __main__ block at the bottom to call the desired function.
"""

from snake_game_environment import SnakeGameEnvironment, make_snake_env
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
from DQN_hyper_tuning import run_hyperparameter_optimization
from custom_callback import MaxStepPunishLogger, DeathLogger
import os
import numpy as np

# Default grid cell size in pixels (used for all environments created here)
GRID_SIZE = 30

# Paths for saving/loading trained models and training logs
LOG_PATH = os.path.join("Training", "Logs")
PPO_PATH = os.path.join("Training", "Saved Models", "PPO")
DQN_PATH = os.path.join("Training", "Saved Models", "DQN")


def train_model(model_name="DQN", grid_width=30, grid_height=20, snake_fov_radius=3, timesteps=500_000, num_envs=4, new=True, params=None, best=True):
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
        best:             If loading (new=False), load "best_model" (True) or "last_model" (False).
    """
    # Create parallel training environments (one per subprocess)
    train_env = SubprocVecEnv([make_snake_env(GRID_SIZE, grid_width, grid_height, snake_fov_radius) for _ in range(num_envs)])

    # Create a single evaluation environment with a step limit to prevent infinite episodes
    raw_eval_env = SnakeGameEnvironment(GRID_SIZE, grid_width, grid_height, snake_fov_radius)
    raw_eval_env = TimeLimit(raw_eval_env, max_episode_steps=10000)
    eval_env = Monitor(raw_eval_env, filename=None)

    if model_name == "DQN":
        # Build the save/load path based on grid size and FOV
        path = os.path.join(DQN_PATH, f"GRID_{grid_width}_{grid_height}", f"FOV_RADIUS_{snake_fov_radius}")
        if not new:
            # Resume training from a saved model
            model = DQN.load(os.path.join(path, "best_model" if best else "last_model"), train_env, device='cpu', verbose=1)
        else:
            # Default DQN hyperparameters (tuned via Optuna)
            if params is None:
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
                "MlpPolicy", train_env,
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
        path = os.path.join(PPO_PATH, f"GRID_{grid_width}_{grid_height}", f"FOV_RADIUS_{snake_fov_radius}")
        if not new:
            model = PPO.load(path=os.path.join(path, "best_model" if best else "last_model"), env=train_env, device='cpu', verbose=1, force_reset=True)

            print(f"Successfully loaded PPO Model ({model._total_timesteps} total_timesteps) [from Path: {path}]")
        else:
            model = PPO(
                "MlpPolicy", train_env,
                n_steps=5000,          # Steps per rollout before policy update
                batch_size=250,        # Minibatch size for PPO updates
                device='cpu', verbose=1
            )

    # Stop training early if mean reward exceeds this threshold
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=100, verbose=1)

    # Evaluate periodically, save best model, and optionally stop early
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        best_model_save_path=path,
        eval_freq=max((timesteps // 10) // num_envs, 1),    # ~10 evaluations per training run
        n_eval_episodes=10,                                  # Episodes per evaluation
        verbose=1,
        deterministic=True
    )

    # Custom callback to log death causes (max-step vs collision)
    max_step_logger = DeathLogger()

    # Start training
    model.learn(total_timesteps=timesteps, callback=[eval_callback, max_step_logger], reset_num_timesteps=False)

    # Always save the final model (in addition to the best model saved by EvalCallback)
    model.save(os.path.join(path, "last_model"))

    # Clean up environments
    train_env.close()
    eval_env.close()


def test_model(model_name="DQN", grid_width=30, grid_height=20, snake_fov_radius=1):
    """
    Load a trained model and watch it play in a Pygame window.

    Args:
        model_name:       "DQN" or "PPO".
        grid_width:       Grid width (must match the model's training config).
        grid_height:      Grid height (must match the model's training config).
        snake_fov_radius: FOV radius (must match the model's training config).
    """
    # Create environment with human rendering (opens Pygame window)
    env = make_snake_env(GRID_SIZE, grid_width, grid_height, snake_fov_radius, "human", training=False)()

    if model_name == "DQN":
        model = DQN.load(os.path.join(DQN_PATH, f"GRID_{grid_width}_{grid_height}", f"FOV_RADIUS_{snake_fov_radius}", "best_model"), env)
    else:
        load_path = os.path.join(PPO_PATH, f"GRID_{grid_width}_{grid_height}", f"FOV_RADIUS_{snake_fov_radius}", "best_model")
        model = PPO.load(load_path, env, device="cpu")
        print(f"Successfully loaded PPO Model ({model._total_timesteps} total_timesteps)\n[from Path: {load_path}]")

    obs, info = env.reset()
    done = False
    score = 0

    # Run the agent until the episode ends
    while not done:
        action, _ = model.predict(obs, deterministic=True)   # Use greedy policy (no exploration)
        obs, reward, done, _, info = env.step(action)
        score += reward

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
    score = 0

    while not done:
        action = input("action: ")
        if action == "d":
            action = np.array(0)       # RIGHT
        elif action == "s":
            action = np.array(1)       # DOWN
        elif action == "a":
            action = np.array(2)       # LEFT
        elif action == "w":
            action = np.array(3)       # UP
        obs, reward, done, _, info = env.step(action)
        score += reward
        print(obs)


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Switch between these calls to train, test, or debug:
    test_model(model_name="PPO", grid_width=50, grid_height=30, snake_fov_radius=5)
    #train_model(model_name="PPO", grid_width=30, grid_height=20, snake_fov_radius=1, timesteps=3_000_000, num_envs=24, new=False, best=False)