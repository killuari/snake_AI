import optuna, os, json
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

PATH = os.path.join("Training", "DQN_Hyperparameter_Tuning")

def optimize_dqn(trial, grid_size = 30, grid_width = 30, grid_height = 20, snake_fov_radius = 1, timesteps = 200_000, num_envs = 1):
    """
    Objective function für Optuna Hyperparameter Optimization
    """
    # Hyperparameter die optimiert werden sollen
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    buffer_size = trial.suggest_categorical("buffer_size", [50_000, 100_000, 500_000, 1_000_000, 2_000_000])
    learning_starts = trial.suggest_int("learning_starts", 1000, 50_000)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    tau = trial.suggest_float("tau", 0.01, 1, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16])
    gradient_steps = trial.suggest_categorical("gradient_steps", [1, 2, 4])
    target_update_interval = trial.suggest_int("target_update_interval", 100, 10000)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.5)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.1)
    
    
    # Environment erstellen (nur eine für schnelleres Tuning)
    from snake_game_environment import make_snake_env
    env = SubprocVecEnv([make_snake_env(grid_size, grid_width, grid_height, snake_fov_radius) for _ in range(num_envs)])
    
    try:
        # DQN Modell mit den vorgeschlagenen Parametern
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            device='cpu',
            verbose=0
        )
        
        # Kurzes Training für schnelles Feedback
        model.learn(total_timesteps=timesteps)
        
        # Evaluation
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
        
        env.close()
        
        return mean_reward
        
    except Exception as e:
        print(f"Trial failed: {e}")
        env.close()
        return -1000

def run_hyperparameter_optimization(grid_size = 30, grid_width = 30, grid_height = 20, snake_fov_radius = 1, timesteps = 200_000, num_envs = 4, n_trials = 50, filename = os.path.join(PATH, "best_dqn_params.json")):
    """
    Führt die Hyperparameter-Optimierung durch
    """
    # Optuna Study erstellen
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30),
        sampler=optuna.samplers.TPESampler()
    )
    
    # Optimization durchführen
    print("Starting hyperparameter optimization...")
    study.optimize(lambda trial: optimize_dqn(trial, grid_size, grid_width, grid_height, snake_fov_radius, timesteps, num_envs), n_trials=n_trials)  # Anzahl Trials anpassen
    
    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    with open(filename, "w") as file:
        json.dump(study.best_params, file, indent=4)
        
        print(f"Parameters saved to {filename}")
    
    return study.best_params