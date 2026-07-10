"""
rl/check_models.py - Manual sanity check: verify every discovered checkpoint
actually loads.

Catches problems like a refactor leaving a stale module path baked into an
old checkpoint's pickle (see rl/feature_extractors.py's sys.modules alias for
a real example this caught) before a user hits it while testing in the UI,
instead of only finding out when someone happens to click "Start Test".

Usage: python -m rl.check_models
"""

import sys

from stable_baselines3 import DQN, PPO

from ui.models import _discover_models
from rl.paths import _find_checkpoint
# Not referenced by name below -- importing it registers the "feature_extractors"
# sys.modules alias (see rl/feature_extractors.py) needed to unpickle GRID-mode
# checkpoints saved before the package refactor, before any .load() runs.
import rl.feature_extractors  # noqa: F401


def check_all_models_loadable():
    """
    Try to load every checkpoint _discover_models() finds (its best
    checkpoint if one exists, else its last). Returns a list of
    (info, checkpoint_path, exception) for the ones that failed to load --
    an empty list means every discovered model loads cleanly.
    """
    models = _discover_models()
    print(f"Checking {len(models)} discovered model configuration(s)...")

    failures = []
    for info in models:
        model_class = DQN if info["algo"] == "DQN" else PPO
        prefix = "best_model" if info["best_timesteps"] is not None else "last_model"
        timesteps = info["best_timesteps"] if info["best_timesteps"] is not None else info["last_timesteps"]
        label = f"{info['algo']} {info['obs_mode']} {info['grid_width']}x{info['grid_height']} FOV{info['fov']} ({prefix}, {timesteps:,} steps)"

        checkpoint_path = _find_checkpoint(info["path"], prefix)
        try:
            model_class.load(checkpoint_path, device="cpu")
            print(f"  OK    {label}")
        except Exception as exc:
            print(f"  FAIL  {label}\n        [{checkpoint_path}]\n        {type(exc).__name__}: {exc}")
            failures.append((info, checkpoint_path, exc))

    return failures


if __name__ == "__main__":
    failures = check_all_models_loadable()
    if failures:
        print(f"\n{len(failures)} of the discovered model(s) failed to load.")
        sys.exit(1)
    print("\nAll discovered models load successfully.")
