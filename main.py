from snake_game_environment import SnakeGameEnvironment, FlattenDictObservationWrapper
from stable_baselines3 import PPO
import os

GRID_SIZE = 30
GRID_WIDTH = 50
GRID_HEIGHT = 30

PPO_PATH = os.path.join("Saved Models", "PPO")

def train_model():
    base_env = SnakeGameEnvironment(GRID_SIZE, GRID_WIDTH, GRID_HEIGHT, "rgb_array")
    env = FlattenDictObservationWrapper(base_env)

    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=20000)

    model.save(PPO_PATH)

    env.close()
    return model

def test_model():
    base_env = SnakeGameEnvironment(GRID_SIZE, GRID_WIDTH, GRID_HEIGHT, "human")
    env = FlattenDictObservationWrapper(base_env)

    model = PPO.load(PPO_PATH, env)

    obs, info = env.reset()
    done = False
    score = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        score += reward


if __name__ == "__main__":
    test_model()