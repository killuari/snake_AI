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

class MaxStepPunishLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.death_counts = 0
        self.total_steps = 0

    def _on_step(self) -> bool:
        rewards = np.array(self.locals["rewards"])  # array of shape (num_envs,)
        self.death_counts += np.sum(rewards == -0.5)
        self.total_steps += rewards.size
        if self.num_timesteps % 100_000 == 0:
            rate = 100 * self.death_counts / max(1, self.total_steps)
            print(f"[Step {self.num_timesteps}] MaxStepPunish-Rate: {rate:.2f}%")
        return True
    
class DeathLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.maxstep_deaths = 0
        self.collision_deaths = 0
        self.total_episodes = 0
        self.last_reported_timestep = 0
    
    def _on_step(self) -> bool:
        # Check if any episodes ended this step
        dones = np.array(self.locals.get("dones", []))
        rewards = np.array(self.locals.get("rewards", []))
        
        # Count completed episodes
        episodes_completed = np.sum(dones)
        self.total_episodes += episodes_completed
        
        # Count different death types when episodes end
        maxstep_deaths = np.sum((dones == True) & (rewards == -0.5))
        collision_deaths = np.sum((dones == True) & (rewards <= -1))  # Collision deaths have more negative rewards
        
        self.maxstep_deaths += maxstep_deaths
        self.collision_deaths += collision_deaths
        
        # Report every 100k timesteps
        if self.num_timesteps - self.last_reported_timestep >= 100_000:
            if self.total_episodes > 0:
                maxstep_rate = 100 * self.maxstep_deaths / self.total_episodes
                collision_rate = 100 * self.collision_deaths / self.total_episodes
                print(f"[Step {self.num_timesteps}] MaxStep: {maxstep_rate:.1f}% | Collision: {collision_rate:.1f}% | Total Episodes: {self.total_episodes}")
            else:
                print(f"[Step {self.num_timesteps}] No episodes completed yet")
            
            self.last_reported_timestep = self.num_timesteps
        
        return True