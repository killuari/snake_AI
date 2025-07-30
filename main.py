from snake_game_environment import SnakeGameEnvironment
from stable_baselines3 import PPO, DQN
import os
import numpy as np

GRID_SIZE = 30
GRID_WIDTH = 30
GRID_HEIGHT = 20

PPO_PATH = os.path.join("Saved Models", "PPO")
DQN_PATH = os.path.join("Saved Models", "DQN")

def train_model(model_name="DQN", timesteps=20000, new=True):
    env = SnakeGameEnvironment(GRID_SIZE, GRID_WIDTH, GRID_HEIGHT, "rgb_array")

    if model_name == "DQN":
        path = DQN_PATH
        if not new:
            model = DQN.load(path, env, device='cpu', verbose=1)
        else:
            model = DQN(
                "MultiInputPolicy",
                env,
                buffer_size=200_000,
                exploration_fraction=0.5,
                target_update_interval=10_000,
                device='cpu',
                verbose=1)
    else:
        path = PPO_PATH
        if not new:
            model = PPO.load(path, env, device='cpu', verbose=1)
        else:
            model = PPO("MultiInputPolicy", env, device='cpu', verbose=1)

    model.learn(total_timesteps=timesteps)
    model.save(path)

    env.close()
    return model

def test_model(model_name="DQN"):
    env = SnakeGameEnvironment(GRID_SIZE, GRID_WIDTH, GRID_HEIGHT, "human")

    if model_name == "DQN":
        model = DQN.load(DQN_PATH, env)
    else:
        model = PPO.load(PPO_PATH, env)

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
    test_model("PPO")
    #train_model("PPO", 5000000, new=False)
    #train_model("DQN", 5000000, new=False)