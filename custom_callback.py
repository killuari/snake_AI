"""
custom_callback.py — Custom Stable Baselines3 Callbacks for Training Monitoring

Provides callbacks that hook into the SB3 training loop to log additional
metrics beyond the standard reward/episode-length tracking.

Classes:
    DeathLogger — Tracks death causes (max-step timeout vs. collision) and
                  reports percentages every 100k timesteps.
"""

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class DeathLogger(BaseCallback):
    """
    Tracks and reports death causes during training: max-step timeout vs. collision.

    Distinguishes death types via the environment's explicit `info["death_cause"]`
    flag ("timeout" | "collision" | None) instead of inferring it from the reward
    value, which is robust to future changes of the reward constants.

    Reports percentages every 100k timesteps.
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
        dones = np.array(self.locals.get("dones", []))
        infos = self.locals.get("infos", [])

        self.total_episodes += np.sum(dones)

        for done, info in zip(dones, infos):
            if not done:
                continue
            death_cause = info.get("death_cause")
            if death_cause == "timeout":
                self.maxstep_deaths += 1
            elif death_cause == "collision":
                self.collision_deaths += 1

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
