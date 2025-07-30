from snake_game_environment import SnakeGameEnvironment, FlattenDictObservationWrapper
from stable_baselines3 import PPO, DQN
import os

GRID_SIZE = 30
GRID_WIDTH = 30
GRID_HEIGHT = 20

PPO_PATH = os.path.join("Saved Models", "PPO")
DQN_PATH = os.path.join("Saved Models", "DQN")

def train_model(model_name="DQN", timesteps=20000, new=True):
    base_env = SnakeGameEnvironment(GRID_SIZE, GRID_WIDTH, GRID_HEIGHT, "rgb_array")
    env = FlattenDictObservationWrapper(base_env)

    if model_name == "DQN":
        path = DQN_PATH
        if not new:
            model = DQN.load(path, env, device='cpu', verbose=1)
        else:
            model = DQN("MultiInputPolicy", env, device='cpu', verbose=1)
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
    base_env = SnakeGameEnvironment(GRID_SIZE, GRID_WIDTH, GRID_HEIGHT, "human")
    env = FlattenDictObservationWrapper(base_env)

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


if __name__ == "__main__":
    train_model(model_name="DQN", timesteps=500000)