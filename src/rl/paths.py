"""
rl/paths.py - Filesystem layout for training artifacts, and shared checkpoint helpers.

Checkpoint directory convention:
    Training/SAVED_MODELS/{PPO,DQN}/{FLAT,GRID}/GRID_{w}_{h}/FOV_RADIUS_{r}/
        best_model_{timesteps}.zip
        last_model_{timesteps}.zip
        evaluation.json
        continue_markers.json
        best_score.json
        logs/tb_0/events.out.tfevents...
        replay_buffer.pkl (DQN only)
"""

import os
import glob
import json

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Default grid cell size in pixels (used for all environments created here)
GRID_SIZE = 30

# The only grid sizes the app offers, for both training and human play -- a
# curated set (not a free width/height slider) since a model's grid size is
# baked into its checkpoint folder name (GRID_{w}_{h}) and, for human play, an
# unconstrained size previously let the grid land off-cell (see the spawn-
# position fix in game/snake_game.py). All three keep a 3:2 aspect ratio.
GRID_PRESETS = [("Small (30x20)", 30, 20), ("Medium (45x30)", 45, 30), ("Large (60x40)", 60, 40)]

# Paths for saving/loading trained models
PPO_PATH = os.path.join("Training", "SAVED_MODELS", "PPO")
DQN_PATH = os.path.join("Training", "SAVED_MODELS", "DQN")

# tb_log_name passed to model.learn(). Doesn't need to encode model_name/grid/
# fov itself (unlike the checkpoint path) -- tensorboard_log_dir() below
# already nests this under each model's own, already-unique checkpoint
# folder. With reset_num_timesteps=False (always used by rl.training), SB3's
# configure_logger()/get_latest_run_id() deterministically resolve this to
# "{tensorboard_log_dir(path)}/tb_0" on every call, new or continue -- never
# incrementing (verified against stable_baselines3/common/utils.py).
TB_RUN_NAME = "tb"


def tensorboard_log_dir(path):
    """Nested tensorboard log directory for a model's checkpoint folder --
    pass as tensorboard_log= to DQN()/DQN.load()/PPO()/PPO.load()."""
    return os.path.join(path, "logs")


def best_score_path(path):
    """Path to the persisted true best mean_reward score for a model --
    written whenever best_model.zip changes, and read back to seed
    EvalCallback.best_mean_reward on "Continue Existing" (see
    rl.training.train_model), since EvalCallback itself always starts a
    fresh instance at -inf, with no memory of a model's actual historical
    best across continuations."""
    return os.path.join(path, "best_score.json")


def _read_best_score(path):
    p = best_score_path(path)
    if not os.path.exists(p):
        return None
    with open(p) as file:
        return json.load(file).get("mean_reward")


def _write_best_score(path, mean_reward):
    with open(best_score_path(path), "w") as file:
        json.dump({"mean_reward": mean_reward}, file)


def replay_buffer_path(path):
    """Path to a DQN model's persisted replay buffer -- saved once at the end
    of each successful run (rl.training.train_model()) and reloaded on
    "Continue Existing" so continuing behaves like one uninterrupted run
    instead of restarting with an empty buffer and a freshly-reset
    exploration schedule."""
    return os.path.join(path, "replay_buffer.pkl")


def tb_run_dir(path):
    """The exact, deterministic tensorboard run directory model.learn()
    writes to for this model (see TB_RUN_NAME's docstring: always
    logs/tb_0)."""
    return os.path.join(tensorboard_log_dir(path), f"{TB_RUN_NAME}_0")


def _snapshot_run_dir(path):
    """Filenames already in path's tensorboard run dir before model.learn()
    starts -- used by _discard_run_artifacts() to work out exactly what a
    discarded run wrote, so only that gets deleted."""
    run_dir = tb_run_dir(path)
    return set(os.listdir(run_dir)) if os.path.isdir(run_dir) else set()


def _discard_run_artifacts(path, pre_existing_files):
    """
    Delete only the tensorboard event file(s) this run just wrote (anything
    in tb_run_dir(path) not present in pre_existing_files), and clean up the
    run/logs directories if that leaves them empty -- so a discarded
    "Continue Existing" or "New Model" run leaves the folder exactly as it
    was before it started, instead of permanently polluting future plots
    with the abandoned attempt's data (SB3 always reuses the same tb_0
    folder, see TB_RUN_NAME's docstring, so without this the discarded
    run's event file would sit right next to -- and get merged with -- the
    legitimate ones).
    """
    run_dir = tb_run_dir(path)
    if not os.path.isdir(run_dir):
        return
    for name in set(os.listdir(run_dir)) - pre_existing_files:
        os.remove(os.path.join(run_dir, name))
    if not os.listdir(run_dir):
        os.rmdir(run_dir)
        logs_dir = tensorboard_log_dir(path)
        if os.path.isdir(logs_dir) and not os.listdir(logs_dir):
            os.rmdir(logs_dir)


def _existing_max_step(path):
    """Highest step already logged in path's tensorboard run dir across all
    plotted tags, or None if nothing's logged yet. Used to detect whether a
    run about to start would "rewind" relative to what's already plotted
    (see _clear_stale_tb_history)."""
    run_dir = tb_run_dir(path)
    if not os.path.isdir(run_dir):
        return None
    ea = EventAccumulator(run_dir)
    ea.Reload()
    max_step = None
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        if events:
            tag_max = max(e.step for e in events)
            max_step = tag_max if max_step is None else max(max_step, tag_max)
    return max_step


def _clear_stale_tb_history(path, pre_existing_files):
    """
    Delete all pre-existing tensorboard event file(s) for this model -- used
    when a run's resume point would otherwise overlap with data already
    plotted for later steps (a "New Model" overwrite starting back at step 0,
    or a "Continue Existing" resume from an earlier checkpoint -- e.g. "Best"
    -- than what's already been logged). Without this, EventAccumulator
    merges the old (further-along) file with the new (earlier-starting) one,
    producing two overlapping curves for the same step range. Trades away
    the old, now-superseded visual history for a clean plot going forward --
    there is no supported way to surgically truncate a .tfevents file to
    keep only the still-valid prefix.
    """
    run_dir = tb_run_dir(path)
    if not os.path.isdir(run_dir):
        return
    for name in pre_existing_files:
        stale = os.path.join(run_dir, name)
        if os.path.exists(stale):
            os.remove(stale)


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


def _record_continue_marker(path, step):
    """
    Append `step` (the cumulative timestep count a "Continue Existing" run
    resumed from) to path/continue_markers.json, so the tensorboard plot can
    draw a vertical line at every past continuation point, not just the
    current one -- persisted so it survives app restarts and repeated
    continuations. Skipped if identical to the last recorded marker (e.g. a
    discarded continuation re-run from the same checkpoint).
    """
    marker_path = os.path.join(path, "continue_markers.json")
    markers = []
    if os.path.exists(marker_path):
        with open(marker_path) as file:
            markers = json.load(file)
    if not markers or markers[-1] != step:
        markers.append(step)
        with open(marker_path, "w") as file:
            json.dump(markers, file)
