# Deep Reinforcement Learning Snake Game

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Environment-Gymnasium-green.svg)](https://gymnasium.farama.org/)
[![Stable Baselines3](https://img.shields.io/badge/RL--Library-SB3-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optimization-Optuna-blueviolet.svg)](https://optuna.org/)

An advanced Reinforcement Learning pipeline featuring a custom-built Game Engine, a Gymnasium-standardized environment, and automated hyperparameter optimization.

---

## Project Architecture & Engineering

This project is built on three main pillars, emphasizing modularity:

### 1. Custom Game (`snake_game.py`)
I developed the **Snake Game** from scratch using Pygame.
* **Object-Oriented Design:** Dedicated classes for `SnakeGame`, `Direction`, and Game Entities.
* **Decoupled Logic:** The game logic is entirely independent of the AI, allowing for both human play and high-speed headless simulation.
* **Collision System:** Optimized coordinate-based collision detection for the snake body, walls, and apples.

### 2. Specialized Gym Environment (`snake_game_environment.py`)
To make the game "trainable", I implemented a custom `gym.Env` (Gymnasium) wrapper.
* **Local FOV (Field of View) Observation:** Instead of feeding the agent the entire grid, it perceives a $(2n+1) \times (2n+1)$ local area around its head. This allows the model to **generalize** to any grid size without retraining.
* **Complex Reward Shaping:** 
    * `Positive`: Reaching food.
    * `Negative`: Deaths (scaled by length to punish late-game mistakes harder).
    * `Loop Prevention`: Small penalties for excessive steps without progress to avoid infinite circling.

### 3. AI Training Pipeline (`main.py` & `DQN_hyper_tuning.py`)
* **Algorithms:** Implementation of **Deep Q-Networks (DQN)** and **Proximal Policy Optimization (PPO)**.
* **Bayesian Optimization:** Integrated **Optuna** to automate the search for optimal hyperparameters.
* **Parallelization:** Utilizes `SubprocVecEnv` to run multiple environment instances in parallel, maximizing CPU utilization and speeding up convergence.

---

## Technical Highlights

### The Observation Logic
The agent doesn't just see pixels; it understands the state. The observation space is a `MultiDiscrete` array representing the contents of the local FOV:
- `0`: Empty Space
- `1`: Wall
- `2`: Snake Body
- `3`: Apple



### Automated Logging & Callbacks
Using `custom_callback.py`, the training process monitors more than just the reward:
* **Death Analysis:** Tracks whether the agent died due to a collision or reaching the `MaxSteps` limit.
* **Model Comparison:** Automatically saves the "Best Model" only if it outperforms previous iterations during evaluation.

---

## Installation & Usage

## 1. Prerequisites
Ensure you have Python 3.9 or higher installed. It is highly recommended to use a virtual environment to manage dependencies.

### 2. Install Dependencies

Install the required libraries using pip:

```bash
pip install gymnasium pygame stable-baselines3 optuna numpy
```

*(Note: simply run `pip install -r requirements.txt`)*

### 3. Execution Guide

The `main.py` script is the central entry point of the project. You can switch between different modes by modifying the function calls in the `if __name__ == "__main__":` block at the bottom of the script.

* **Training a Model:**
Call `train_model(model_name="DQN", ...)` to start a new training session. The script will automatically use `SubprocVecEnv` for parallel processing.

* **Hyperparameter Optimization:**
To find the best settings, uncomment `run_hyperparameter_optimization(...)`. This triggers an **Optuna study** that explores the best learning rates and architectures.

* **Testing & Visualization:**
To watch the trained agent play, use `test_model(model_name="DQN")`. This will open a Pygame window and render the snake's decision-making process in real-time.

---

## Project Structure

```text
├── DQN_hyper_tuning.py        # Hyperparameter optimization script
├── custom_callback.py         # Custom logging and evaluation callbacks
├── main.py                    # Main entry point and training pipeline
├── snake_game.py              # Pygame-based game engine
└── snake_game_environment.py  # Gymnasium environment wrapper
```
---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Developed by Luis Kahles**

*Focus: Reinforcement Learning, Modular Software Architecture, and AI-Driven Automation.*
