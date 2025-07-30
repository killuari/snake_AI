from snake_game_environment import SnakeGameEnvironment, make_snake_env
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
import os
import numpy as np

GRID_SIZE = 30
GRID_WIDTH = 30
GRID_HEIGHT = 20

LOG_PATH = os.path.join("Training", "Logs")
PPO_PATH = os.path.join("Training", "Saved Models", "PPO")
DQN_PATH = os.path.join("Training", "Saved Models", "DQN")

def train_model(model_name="DQN", timesteps=20000, num_envs=4, new=True):
    train_env = SubprocVecEnv([make_snake_env(GRID_SIZE, GRID_WIDTH, GRID_HEIGHT) for _ in range(num_envs)])
    raw_eval_env = SnakeGameEnvironment(GRID_SIZE, GRID_WIDTH, GRID_HEIGHT)
    raw_eval_env = TimeLimit(raw_eval_env, max_episode_steps=5000)
    eval_env = Monitor(raw_eval_env, filename=None)

    if model_name == "DQN":
        path = DQN_PATH
        if not new:
            model = DQN.load(os.path.join(path, "best_model"), train_env, device='cpu', verbose=1)
        else:
            model = DQN(
                "MultiInputPolicy",
                train_env,
                learning_rate=1e-4,
                buffer_size=2_000_000,
                exploration_fraction=0.5,
                device='cpu',
                verbose=1)
    else:
        path = PPO_PATH
        if not new:
            model = PPO.load(os.path.join(path, "best_model"), train_env, device='cpu', verbose=1)
        else:
            model = PPO("MultiInputPolicy", train_env, device='cpu', verbose=1)

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=100, verbose=1)

    # EvalCallback mit Stop- und Save-Best-Funktion
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        best_model_save_path=path,
        eval_freq=20_000,                # alle 20k Steps evaluieren
        n_eval_episodes=5,
        verbose=1,
        deterministic=True
    )

    model.learn(total_timesteps=timesteps, callback=eval_callback)

    train_env.close()
    eval_env.close()

def test_model(model_name="DQN"):
    env = SnakeGameEnvironment(GRID_SIZE, GRID_WIDTH, GRID_HEIGHT, "human")

    if model_name == "DQN":
        model = DQN.load(os.path.join(DQN_PATH, "best_model"), env)
    else:
        model = PPO.load(os.path.join(PPO_PATH, "best_model"), env)

    obs, info = env.reset()
    done = False
    score = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        score += reward

def test_environment():
    env = SnakeGameEnvironment(GRID_SIZE, GRID_WIDTH, GRID_HEIGHT, "human")

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
    #test_model("DQN")
    train_model(model_name="PPO", timesteps=1_000_000, num_envs=10)