"""
custom_callback.py — Custom Stable Baselines3 Callbacks for Training Monitoring

Provides callbacks that hook into the SB3 training loop to log additional
metrics beyond the standard reward/episode-length tracking.

Classes:
    BestModelComparisonCallback — (UNUSED) Evaluates and saves best model by comparing
                                  against a previously saved baseline.
    MaxStepPunishLogger         — Tracks how often the max-step penalty (-0.5) is triggered.
    DeathLogger                 — Tracks death causes (max-step timeout vs. collision) and
                                  reports percentages every 100k timesteps.
"""

from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
from stable_baselines3 import DQN


class BestModelComparisonCallback(BaseCallback):
    """
    Callback that periodically evaluates the current model and saves it
    if it outperforms the previously saved best model.

    NOTE: This class is currently UNUSED 

    Args:
        eval_env:             Environment used for evaluation episodes.
        best_model_save_path: Directory to save/load the best model.
        eval_freq:            Evaluate every N training steps.
        n_eval_episodes:      Number of episodes per evaluation.
        verbose:              Verbosity level.
    """

    def __init__(self, eval_env, best_model_save_path, eval_freq=20000, n_eval_episodes=5, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.episode_infos = []
        
        # Try to load an existing best model to use as the baseline for comparison
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
        """Run the model for n episodes and return the mean reward."""
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
        """Called at every training step. Evaluates and saves if improved."""
        # Periodic evaluation
        if self.n_calls % self.eval_freq == 0:
            mean_reward = self._evaluate_model(self.model)
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                if self.verbose > 0:
                    print(f"New best model saved with reward: {mean_reward}")
        
        # Log snake length from completed episodes (if available in info dict)
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if info.get("episode"):  # Episode just ended
                    if "snake_length" in info:
                        self.episode_infos.append(info["snake_length"])
                        print(f"Episode ended. Final Snake Length: {info['snake_length']}")
                        
                        # Print rolling average every 100 episodes
                        if len(self.episode_infos) >= 100:
                            avg_length = np.mean(self.episode_infos[-100:])
                            print(f"Average Snake Length (last 100): {avg_length:.2f}")
        
        return True  # Return True to continue training


class MaxStepPunishLogger(BaseCallback):
    """
    Tracks how often the max-step penalty (reward == -0.5) is triggered.

    Reports the percentage of steps that resulted in a max-step punishment
    every 100k timesteps. Useful for monitoring if the agent is learning
    to find food efficiently or just looping.

    NOTE: Uses float equality (reward == -0.5) which can be fragile.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.death_counts = 0      # Number of max-step punishments observed
        self.total_steps = 0       # Total step observations across all envs

    def _on_step(self) -> bool:
        """Count max-step punishments and report every 100k steps."""
        rewards = np.array(self.locals["rewards"])  # Shape: (num_envs,)
        self.death_counts += np.sum(rewards == -0.5)
        self.total_steps += rewards.size

        if self.num_timesteps % 100_000 == 0:
            rate = 100 * self.death_counts / max(1, self.total_steps)
            print(f"[Step {self.num_timesteps}] MaxStepPunish-Rate: {rate:.2f}%")

        return True

    
class DeathLogger(BaseCallback):
    """
    Tracks and reports death causes during training: max-step timeout vs. collision.

    Distinguishes death types by checking the reward value when an episode ends:
        - reward == -0.5     → max-step timeout (looping without progress)
        - reward <= -1       → collision death (wall or self-collision)

    Reports percentages every 100k timesteps.

    NOTE: Relies on float equality for reward detection, which can be fragile.
    Consider using info dict flags for more robust death-cause tracking.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.maxstep_deaths = 0        # Count of max-step timeout deaths
        self.collision_deaths = 0      # Count of collision deaths
        self.total_episodes = 0        # Total completed episodes
        self.last_reported_timestep = 0

    def _on_training_start(self) -> None:
        """Initialize the reporting baseline at training start."""
        self.last_reported_timestep = self.num_timesteps
    
    def _on_step(self) -> bool:
        """Check for completed episodes and classify death causes."""
        # Get done flags and rewards for all parallel environments
        dones = np.array(self.locals.get("dones", []))
        rewards = np.array(self.locals.get("rewards", []))
        
        # Count how many episodes completed this step
        episodes_completed = np.sum(dones)
        self.total_episodes += episodes_completed
        
        # Classify deaths by reward value (only when episode actually ended)
        maxstep_deaths = np.sum((dones == True) & (rewards == -0.5))
        collision_deaths = np.sum((dones == True) & (rewards <= -1))  # Collision = more negative
        
        self.maxstep_deaths += maxstep_deaths
        self.collision_deaths += collision_deaths
        
        # Report death statistics every 100k timesteps
        if self.num_timesteps - self.last_reported_timestep >= 100_000:
            if self.total_episodes > 0:
                maxstep_rate = 100 * self.maxstep_deaths / self.total_episodes
                collision_rate = 100 * self.collision_deaths / self.total_episodes
                print(f"[Step {self.num_timesteps:,}] MaxStep: {maxstep_rate:.1f}% | Collision: {collision_rate:.1f}% | Total Episodes: {self.total_episodes}")
            else:
                print(f"[Step {self.num_timesteps:,}] No episodes completed yet")
            
            self.last_reported_timestep = self.num_timesteps
        
        return True  # Continue training