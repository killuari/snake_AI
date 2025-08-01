from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np

class BestModelComparisonCallback(BaseCallback):
    def __init__(self, eval_env, best_model_save_path, eval_freq=20000, n_eval_episodes=5, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.episode_infos = []
        
        # Versuche bestehendes best_model zu laden
        best_model_path = os.path.join(best_model_save_path, "best_model.zip")
        if os.path.exists(best_model_path):
            try:
                self.baseline_model = DQN.load(best_model_path)
                self.best_mean_reward = self._evaluate_model(self.baseline_model)
                print(f"Loaded existing best model with reward: {self.best_mean_reward}")
            except:
                self.best_mean_reward = -np.inf
                print("Could not load existing model, starting fresh")
        else:
            self.best_mean_reward = -np.inf
            print("No existing best model found, starting fresh")
    
    def _evaluate_model(self, model):
        episode_rewards = []
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = self.eval_env.step(action)
                episode_reward += reward
            episode_rewards.append(episode_reward)
        return np.mean(episode_rewards)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward = self._evaluate_model(self.model)
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                if self.verbose > 0:
                    print(f"New best model saved with reward: {mean_reward}")
        
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if info.get("episode"):  # Episode beendet
                    # Info der letzten Episode
                    if "snake_length" in info:
                        self.episode_infos.append(info["snake_length"])
                        print(f"Episode ended. Final Snake Length: {info['snake_length']}")
                        
                        # Statistiken über letzte 100 Episodes
                        if len(self.episode_infos) >= 100:
                            avg_length = np.mean(self.episode_infos[-100:])
                            print(f"Average Snake Length (last 100): {avg_length:.2f}")
        
        return True