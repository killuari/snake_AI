# Deep Reinforcement Learning Snake Game

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Environment-Gymnasium-green.svg)](https://gymnasium.farama.org/)
[![Stable Baselines3](https://img.shields.io/badge/RL--Library-SB3-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optimization-Optuna-blueviolet.svg)](https://optuna.org/)
[![CustomTkinter](https://img.shields.io/badge/UI-CustomTkinter-9cf.svg)](https://github.com/TomSchimansky/CustomTkinter)

An advanced Reinforcement Learning pipeline featuring a custom-built Game Engine, a Gymnasium-standardized environment, automated hyperparameter optimization, and a graphical desktop launcher to play, watch, and train from.

---

## Launcher

Everything is driven from a single dark-mode desktop app (`python src/main.py`) — no editing scripts or calling functions by hand. Five modes, each a full screen: **Play** yourself, **Test Model** to watch a trained agent, **Train Model** to start or continue a training run, **Models** to browse/filter/delete every checkpoint on disk, and **Exit**.

<p align="center">
  <img src="docs/screenshots/home.png" alt="Launcher home screen" width="420">
</p>

**Train Model** is a single scrollable form — algorithm, observation mode, grid size, FOV radius, timesteps, parallel environments, and the live training log all in one screen, so you never lose sight of the log while configuring a run. Continuing an existing run shows the same log, plus an independently-scrollable list of saved checkpoints with a "Resume from" choice (Best or Last, each tracked with its own timestep and score) to its right.

Below the log, a live TensorBoard-style plot (mean episode reward + loss) redraws every ~1.5s while training runs, with a dashed marker at every past "Continue Existing" resume point. An opt-in "Render training live" checkbox adds a second live panel showing one of the parallel training environments actually playing, throttled to stay cheap (~5% training-throughput cost, see [Technical Highlights](#technical-highlights)).

<p align="center">
  <img src="docs/screenshots/train_model.png" alt="Train Model screen, fully scrollable form" width="480">
</p>

**Test Model** lists every checkpoint found under `Training/SAVED_MODELS/`, color-coded by algorithm (PPO/DQN) and observation mode (FLAT/GRID) so a long list stays easy to scan, sorted by algorithm → observation mode → grid size → FOV radius. Each card shows its Best/Last timesteps and deterministic/stochastic evaluation scores at a glance.

<p align="center">
  <img src="docs/screenshots/test_model.png" alt="Test Model screen with color-coded model list" width="480">
</p>

**Models** is the same color-coded, sortable list, with algorithm/observation-mode/grid-size filters and a free-text search box on top, plus a Delete button and a shortcut into Train Model's "Continue Existing" flow for a given checkpoint.

---

## Project Architecture & Engineering

The project is a set of small, focused packages:

### `game/` — Core Engine & Environment
* **`snake_game.py`** — The Snake game itself, built from scratch on Pygame. Object-oriented (`SnakeGame`, `SnakePart`, `Apple`, `Direction`), fully decoupled from the AI so it runs identically for human play, headless training, and visual playback.
* **`environment.py`** — A custom Gymnasium (`gym.Env`) wrapper around the game.
* **`game_over.py`** — The shared death-animation and game-over overlay used by both human play and model playback.

### `rl/` — Training & Playback Pipeline
* **`training.py`** — `train_model()`: trains DQN or PPO with `SubprocVecEnv`-parallelized environments, periodic evaluation, independent Best/Last checkpoint tracking (each with its own true timestep and score, correct across any number of "Continue Existing" runs), DQN replay-buffer persistence for seamless continuation, optional live-frame streaming for the UI's render-during-training view, and graceful cancel/discard support.
* **`playback.py`** — `play_game()` (human play) and `test_model()` (watch a trained agent), both pygame windows.
* **`hyperparameter_tuning.py`** — Optuna-driven DQN hyperparameter search.
* **`feature_extractors.py`**, **`callbacks.py`**, **`paths.py`**, **`check_models.py`** — the CNN feature extractor for GRID mode, training callbacks, the checkpoint directory layout (including the two-track TensorBoard/best-score bookkeeping described below), and a manual "does every saved model still load?" sanity check.

### `ui/` — Desktop Launcher
A CustomTkinter app (`ui/app.py`) with one screen per mode under `ui/screens/` (`home`, `play`, `test_model`, `train_model`, `models`, `base`), sharing a small theme/widget toolkit (`ui/theme.py`, `ui/widgets.py`), the checkpoint-discovery logic (`ui/models.py`), and two embeddable live-refreshing widgets used on the Train Model screen: `ui/plot_window.py` (the TensorBoard-style reward/loss plot) and `ui/game_view.py` (the optional live-rendered training view).

### `main.py`
The entry point: `python main.py` launches the UI. Nothing else lives here — kept intentionally thin to avoid an import cycle with `ui`.

---

## Technical Highlights

### The Observation Logic
The agent doesn't see pixels; it perceives a local **Field of View (FOV)** around its head — a $(2n+1) \times (2n+1)$ window — so a trained model generalizes to grid sizes it never trained on. Two observation layouts are supported, selectable per training run:
* **FLAT** — a `MultiDiscrete` vector of the FOV's cell contents (`0`=empty, `1`=body, `2`=apple, `3`=wall) plus the apple's direction, for a standard `MlpPolicy`.
* **GRID** — the same FOV as a one-hot `(4, H, W)` tensor plus the apple's direction, fed through a small purpose-built CNN (`SnakeCombinedExtractor`) via a `MultiInputPolicy` — SB3's default CNN assumes much larger image-like inputs, so a small custom extractor was needed for a 7×7–17×17 FOV window.

Press **`f`** while watching a model play to toggle a debug overlay: the FOV window it's currently looking at, and an arrow for the apple-direction feature.

### Complex Reward Shaping
* **Positive:** reaching the apple.
* **Negative:** dying, scaled by snake length (a late-game mistake costs more than an early one).
* **Loop prevention:** a small penalty for excessive steps without progress, to discourage infinite circling.

### Automated Logging & Callbacks
`rl/callbacks.py`'s `DeathLogger` tracks more than reward during training:
* **Death Analysis:** collision vs. `MaxSteps` timeout, reported periodically.
* **Model Comparison:** the "best" checkpoint is only overwritten if a new evaluation genuinely beats the model's true historical best — including across "Continue Existing" runs. SB3's `EvalCallback` normally resets its own best-tracking to `-∞` on every fresh instantiation (i.e. on every continuation), so without correcting for that, the very first evaluation of any continuation would always count as "improved," even if it's actually worse than what came before. `train_model()` persists the real score (`best_score.json`) and seeds `EvalCallback` with it on every continuation instead.

### Live Training Monitoring
* **Reward/loss plot** (`ui/plot_window.py`), read straight from the run's TensorBoard event files and redrawn every ~1.5s — no need to open TensorBoard separately. Best and Last are tracked as two independent, always-continuous histories (`logs/tb_0` and `logs/tb_best` under each checkpoint folder): since Best's own checkpoint can be evaluated at an earlier timestep than the model's current Last, "Continue from Best" reseeds the live plot from Best's own history before training starts, so the graph never mixes two different trajectories into one overlapping mess, and no history is ever lost.
* **Live game view** (`ui/game_view.py`), opt-in via a checkbox on the Train Model screen: pulls a rendered frame from one of the parallel training environments every ~0.12s (over the existing `SubprocVecEnv` pipe, no extra window), converts it to a `CTkImage`, and shows actual gameplay updating live next to the plot. Measured overhead: ~5% training throughput.

### Error Checking
`rl/check_models.py` (`python -m rl.check_models`) loads every checkpoint the launcher would list, and reports which ones actually load — catching a broken/incompatible checkpoint (e.g. a stale module reference from a refactor) before it surfaces as a cryptic error mid-session in the UI.

---

## Installation & Usage

### 1. Prerequisites
Python 3.13+. The project uses [`uv`](https://docs.astral.sh/uv/) for dependency management (a `uv.lock` is checked in); plain `pip` works too.

### 2. Install Dependencies

```bash
uv sync
```

or, without `uv`:

```bash
pip install pygame numpy gymnasium stable-baselines3 optuna customtkinter tensorboard matplotlib pillow
```

### 3. Run It

```bash
python src/main.py
```

Run this from the repo root (not from inside `src/`) — `Training/` (and `Training/SAVED_MODELS/`) is resolved relative to the current working directory and is created automatically as needed; nothing has to exist beforehand.

This opens the launcher — pick **Play**, **Test Model**, **Train Model**, or **Models** from there.

Everything is also directly importable for scripting, e.g.:

```python
from rl.training import train_model
from rl.playback import play_game, test_model

train_model(model_name="DQN", grid_width=30, grid_height=20, timesteps=3_000_000)
test_model(model_name="DQN", grid_width=30, grid_height=20, snake_fov_radius=3)
```

---

## Project Structure

```text
├── src/
│   ├── main.py                   # Entry point: launches the UI
│   ├── game/
│   │   ├── snake_game.py         # Pygame-based game engine
│   │   ├── environment.py        # Gymnasium environment wrapper
│   │   └── game_over.py          # Shared death animation + game-over overlay
│   ├── rl/
│   │   ├── training.py           # train_model()
│   │   ├── playback.py           # play_game(), test_model()
│   │   ├── hyperparameter_tuning.py  # Optuna DQN search
│   │   ├── feature_extractors.py # CNN extractor for GRID observation mode
│   │   ├── callbacks.py          # DeathLogger, PeriodicCheckpoint training callbacks
│   │   ├── paths.py              # Checkpoint directory layout + grid presets
│   │   └── check_models.py       # Manual "do all saved models load?" check
│   └── ui/
│       ├── app.py                # App root window + navigation
│       ├── theme.py, widgets.py  # Shared color palette + widget factories
│       ├── models.py             # Checkpoint discovery for the UI
│       ├── plot_window.py        # Live reward/loss plot (Train Model screen)
│       ├── game_view.py          # Live rendered training view (Train Model screen)
│       └── screens/              # home, play, test_model, train_model, models, base
└── Training/
    └── SAVED_MODELS/
        └── {PPO,DQN}/{FLAT,GRID}/GRID_{w}_{h}/FOV_RADIUS_{r}/
            ├── best_model_{steps}.zip     # steps = the timestep Best was actually found at
            ├── last_model_{steps}.zip
            ├── evaluation.json            # deterministic/stochastic scores for both checkpoints
            ├── continue_markers.json      # timesteps of past "Continue Existing" resume points
            ├── best_score.json            # Best's true mean_reward, kept across continuations
            ├── replay_buffer.pkl          # DQN only -- lets a continuation resume seamlessly
            └── logs/
                ├── tb_0/                  # "Last" track's TensorBoard history (live, continuous)
                └── tb_best/               # "Best" track's own history, truncated at its timestep
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

---

**Developed by Luis Kahles**

*Focus: Reinforcement Learning, Modular Software Architecture, and AI-Driven Automation.*
