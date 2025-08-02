from snake_game_environment import SnakeGameEnvironment, make_snake_env
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
from DQN_hyper_tuning import run_hyperparameter_optimization
import os
import numpy as np

GRID_SIZE = 30

# Reward_1_(500)-0.5_-1
# Reward_1_(500)-0.5_-5
# Reward_1_(500)-0.5_-(1, 0.5(len-3), 10)
# Reward_1_(500)-0.5_-(1, 0.5(len-3), 20)

LOG_PATH = os.path.join("Training", "Logs")
PPO_PATH = os.path.join("Training", "Saved Models", "PPO", "Reward_1_(500)-0.5_-(1, 0.5(len-3), 20)")
DQN_PATH = os.path.join("Training", "Saved Models", "DQN", "Reward_1_(500)-0.5_-(1, 0.5(len-3), 20)")

def train_model(model_name="DQN", grid_width=30, grid_height=20, snake_fov_radius=3, timesteps=500_000, num_envs=4, new=True, params=None):
    train_env = SubprocVecEnv([make_snake_env(GRID_SIZE, grid_width, grid_height, snake_fov_radius) for _ in range(num_envs)])
    raw_eval_env = SnakeGameEnvironment(GRID_SIZE, grid_width, grid_height, snake_fov_radius)
    raw_eval_env = TimeLimit(raw_eval_env, max_episode_steps=10000)
    eval_env = Monitor(raw_eval_env, filename=None)

    if model_name == "DQN":
        path = os.path.join(DQN_PATH, f"GRID_{grid_width}_{grid_height}", f"FOV_RADIUS_{snake_fov_radius}")
        if not new:
            model = DQN.load(os.path.join(path, "last_model"), train_env, device='cpu', verbose=1)
        else:
            if params is None:
                params = {
                    "learning_rate": 5e-4,
                    "buffer_size": 200_000,
                    "learning_starts": 50_000,
                    "batch_size": 64,
                    "tau": 0.005,
                    "gamma": 0.99,
                    "train_freq": 4,
                    "gradient_steps": 1,
                    "target_update_interval": 2000,
                    "exploration_fraction": 0.2,
                    "exploration_final_eps": 0.02,
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
        path = os.path.join(PPO_PATH, f"GRID_{grid_width}_{grid_height}", f"FOV_RADIUS_{snake_fov_radius}")
        if not new:
            model = PPO.load(os.path.join(path, "last_model"), train_env, device='cpu', verbose=1)
        else:
            model = PPO(
                "MlpPolicy", train_env,
                n_steps=5000,
                batch_size=250,
                device='cpu', verbose=1
            )

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=100, verbose=1)

    # EvalCallback mit Stop- und Save-Best-Funktion
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        best_model_save_path=path,
        eval_freq=20_000,                # alle 20k Steps evaluieren
        n_eval_episodes=10,
        verbose=1,
        deterministic=True
    )

    model.learn(total_timesteps=timesteps, callback=eval_callback)
    model.save(os.path.join(path, "last_model"))

    train_env.close()
    eval_env.close()

def test_model(model_name="DQN", grid_width=30, grid_height=20, snake_fov_radius=1):
    env = make_snake_env(GRID_SIZE, grid_width, grid_height, snake_fov_radius, "human", training=False)()

    if model_name == "DQN":
        model = DQN.load(os.path.join(DQN_PATH, f"GRID_{grid_width}_{grid_height}", f"FOV_RADIUS_{snake_fov_radius}", "best_model"), env)
    else:
        model = PPO.load(os.path.join(PPO_PATH, f"GRID_{grid_width}_{grid_height}", f"FOV_RADIUS_{snake_fov_radius}", "best_model"), env, device="cpu")

    obs, info = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

def test_environment(grid_width=30, grid_height=20, snake_fov_radius=1):
    env = make_snake_env(GRID_SIZE, grid_width, grid_height, snake_fov_radius, "human", training=False)()

    obs, info = env.reset()
    done = False
    score = 0

    while not done:
        action = input("action: ")
        if action == "d":
            action = np.array(0)
        elif action == "s":
            action = np.array(1)
        elif action == "a":
            action = np.array(2)
        elif action == "w":
            action = np.array(3)
        obs, reward, done, _, info = env.step(action)
        score += reward
        print(obs)

if __name__ == "__main__":
    #test_model(model_name="PPO", grid_width=30, grid_height=20, snake_fov_radius=5)

    best_params = run_hyperparameter_optimization(GRID_SIZE, grid_width=30, grid_height=20, snake_fov_radius=5)
    print(best_params)

    train_model(model_name="DQN", grid_width=30, grid_height=20, snake_fov_radius=5, timesteps=3_000_000, num_envs=12, params=best_params)