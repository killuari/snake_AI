"""
ui/models.py - Discovering trained models on disk for the Test/Continue-Training screens.
"""

import os
import re
import glob
import json

from rl.paths import PPO_PATH, DQN_PATH

# Explicit display order for the two non-numeric grouping levels -- PPO before
# DQN and FLAT before GRID, matching _discover_models()'s walk order below (so
# sorting doesn't visually reorder single-algorithm/obs-mode lists), used as a
# sort key so the *combined* list (multiple algos/obs modes) is grouped the
# same way instead of falling back to alphabetical.
_ALGO_ORDER = {"PPO": 0, "DQN": 1}
_OBS_ORDER = {"FLAT": 0, "GRID": 1}


def _discover_models():
    """
    Scan Training/SAVED_MODELS/{PPO,DQN}/{FLAT,GRID}/GRID_*_*/FOV_RADIUS_*/ for
    checkpoints, skipping configs with neither a best_model_*.zip nor a
    last_model_*.zip (see rl.paths._finalize_checkpoint/_find_checkpoint).
    """
    models = []
    for algo, base in (("PPO", PPO_PATH), ("DQN", DQN_PATH)):
        for obs_label in ("FLAT", "GRID"):
            root = os.path.join(base, obs_label)
            if not os.path.isdir(root):
                continue
            for grid_dir in sorted(os.listdir(root)):
                grid_match = re.match(r"GRID_(\d+)_(\d+)$", grid_dir)
                if not grid_match:
                    continue
                grid_path = os.path.join(root, grid_dir)
                for fov_dir in sorted(os.listdir(grid_path)):
                    fov_match = re.match(r"FOV_RADIUS_(\d+)$", fov_dir)
                    if not fov_match:
                        continue
                    model_path = os.path.join(grid_path, fov_dir)

                    def _timesteps(prefix):
                        matches = glob.glob(os.path.join(model_path, f"{prefix}_*.zip"))
                        if not matches:
                            return None
                        m = re.match(rf"{prefix}_(\d+)\.zip", os.path.basename(matches[0]))
                        return int(m.group(1)) if m else None

                    best_timesteps = _timesteps("best_model")
                    last_timesteps = _timesteps("last_model")
                    if best_timesteps is None and last_timesteps is None:
                        continue

                    evaluation = None
                    eval_path = os.path.join(model_path, "evaluation.json")
                    if os.path.exists(eval_path):
                        with open(eval_path) as file:
                            evaluation = json.load(file)

                    models.append({
                        "algo": algo,
                        "obs_mode": obs_label,
                        "use_cnn": obs_label == "GRID",
                        "grid_width": int(grid_match.group(1)),
                        "grid_height": int(grid_match.group(2)),
                        "fov": int(fov_match.group(1)),
                        "path": model_path,
                        "best_timesteps": best_timesteps,
                        "last_timesteps": last_timesteps,
                        "evaluation": evaluation,
                    })

    # Directory listings above are sorted lexicographically (os.listdir), which
    # misorders numeric grid/FOV values (e.g. "GRID_100_..." < "GRID_30_...")
    # -- sort explicitly by algorithm, then obs mode, then grid size, then FOV,
    # all as actual numbers/declared order rather than strings.
    models.sort(key=lambda m: (
        _ALGO_ORDER[m["algo"]], _OBS_ORDER[m["obs_mode"]],
        m["grid_width"], m["grid_height"], m["fov"],
    ))
    return models
