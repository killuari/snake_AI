"""
rl/paths.py - Filesystem layout for training artifacts, and shared checkpoint helpers.

Checkpoint directory convention:
    Training/SAVED_MODELS/{PPO,DQN}/{FLAT,GRID}/GRID_{w}_{h}/FOV_RADIUS_{r}/
        best_model_{timesteps}.zip
        last_model_{timesteps}.zip
        evaluation.json
"""

import os
import glob

# Default grid cell size in pixels (used for all environments created here)
GRID_SIZE = 30

# Paths for saving/loading trained models and training logs
LOG_PATH = os.path.join("Training", "Logs")
PPO_PATH = os.path.join("Training", "SAVED_MODELS", "PPO")
DQN_PATH = os.path.join("Training", "SAVED_MODELS", "DQN")


def _finalize_checkpoint(path, prefix, total_timesteps):
    """
    Rename path/{prefix}.zip (just written by model.save()/EvalCallback) to
    path/{prefix}_{total_timesteps}.zip, so the filename itself records how many
    timesteps the checkpoint was trained for. Replaces any stale checkpoint left
    over from a previous training run under the same prefix.
    """
    plain_path = os.path.join(path, f"{prefix}.zip")
    if not os.path.exists(plain_path):
        return
    for stale in glob.glob(os.path.join(path, f"{prefix}_*.zip")):
        os.remove(stale)
    os.rename(plain_path, os.path.join(path, f"{prefix}_{total_timesteps}.zip"))


def _find_checkpoint(path, prefix):
    """Find the (single) path/{prefix}_{timesteps}.zip checkpoint written by _finalize_checkpoint()."""
    matches = glob.glob(os.path.join(path, f"{prefix}_*.zip"))
    if not matches:
        raise FileNotFoundError(f"No '{prefix}' checkpoint found in {path}")
    return max(matches, key=os.path.getmtime)
