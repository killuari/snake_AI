"""
rl/training.py - Training pipeline for the Snake RL agent (DQN/PPO).

Functions:
    train_model():                Train a DQN or PPO agent with parallel environments.
    evaluate_model_performance(): Headless deterministic/stochastic scoring of a trained model.
"""

import os
import json
import glob
import time
import numpy as np

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

from game.environment import SnakeGameEnvironment, make_snake_env
from rl.paths import (
    GRID_SIZE, PPO_PATH, DQN_PATH, TB_RUN_NAME, tensorboard_log_dir, replay_buffer_path,
    _find_checkpoint, _finalize_checkpoint, _record_continue_marker,
    _snapshot_run_dir, _discard_run_artifacts, _existing_max_step,
    _backup_run_dir, _restore_run_dir_backup, _discard_run_dir_backup,
    _seed_run_dir_from_best, _rebuild_tb_best, tb_best_dir,
    _read_best_score, _write_best_score,
)
from rl.callbacks import DeathLogger, PeriodicCheckpoint
from rl.feature_extractors import SnakeCombinedExtractor
from rl.hyperparameter_tuning import load_best_params


def linear_schedule(initial_value):
    """Linearly decays from `initial_value` at training start to 0 by training
    end (SB3's own documented recipe for a decaying learning rate), as a
    function of `progress_remaining` (1.0 at the start, 0.0 at the end)."""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


def _rebase_schedule_for_continuation(schedule_fn, model, new_timesteps):
    """
    SB3 computes `progress_remaining` against `new_timesteps + model.num_timesteps`
    combined when `reset_num_timesteps=False` (which train_model() always uses --
    see stable_baselines3/common/base_class.py's `_setup_learn()`), so on a
    "Continue Existing" run `progress_remaining` starts well below 1.0 instead of
    at 1.0 (e.g. continuing a 3M-step model with 500k more requested starts at
    ~0.14, not 1.0). Any progress-based schedule built on top of that is
    therefore already partway/fully decayed from the very first step of the
    continuation instead of getting a fresh cycle.

    Used for DQN's exploration_schedule only, deliberately NOT for the learning
    rate: model.save()/.load() here never persist the replay buffer, so every
    continuation starts with an empty one regardless -- re-exploring to refill
    it with diverse data is genuinely useful, not just cosmetic. A fresh
    learning-rate cycle, in contrast, would jump a well-converged model's
    optimizer (whose Adam momentum *is* restored from the checkpoint) back up
    to full step size on every single continuation -- risky with this app's
    workflow of frequent, small continuations, and a low/decayed rate on
    continuation isn't really a bug the way the epsilon floor is: it's usually
    the conservative, desired behavior for fine-tuning an already-good model.

    Wraps `schedule_fn` so it instead gets a fresh 1.0->0.0 application over
    just the newly-requested timesteps, as if this continuation were its own
    short run.
    """
    p_start = new_timesteps / (new_timesteps + model.num_timesteps)

    def rebased(progress_remaining):
        local = min(1.0, progress_remaining / p_start) if p_start > 0 else 0.0
        return schedule_fn(local)
    return rebased


def evaluate_model_performance(model, grid_size, grid_width, grid_height, snake_fov_radius, obs_mode="flat", n_episodes=10, model_label=""):
    """
    Run n_episodes deterministic and n_episodes stochastic episodes (no rendering,
    no reward shaping) and return the mean "clean" score (apples eaten) for each,
    the same score shown by test_model().
    """
    if model_label:
        print(f"Evaluating {model_label}...")

    # training=False disables the environment's own anti-loop timeout, so wrap
    # with a hard step limit -- otherwise an undertrained/looping policy (quite
    # possible here, especially in the stochastic episodes) can hang this
    # function indefinitely instead of just producing a low score.
    env = TimeLimit(
        SnakeGameEnvironment(grid_size, grid_width, grid_height, snake_fov_radius, render_mode=None, training=False, obs_mode=obs_mode),
        max_episode_steps=10000,
    )
    results = {}
    for label, deterministic in [("deterministic", True), ("stochastic", False)]:
        scores = []
        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            score = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                score += float(reward)
            scores.append(score)
        results[label] = {"mean_score": sum(scores) / len(scores), "episodes": n_episodes, "scores": scores}
        print(f"  {label}: mean score = {results[label]['mean_score']:.2f} ({n_episodes} episodes)")
    env.close()
    return results


def train_model(model_name="DQN", grid_width=30, grid_height=20, snake_fov_radius=3, timesteps=3_000_000, num_envs=4, new=True, params=None, best=True, use_tuned_params=False, use_cnn=False, cancel_event=None, discard_event=None, on_log_dir=None, on_frame=None):
    """
    Train a DQN or PPO agent on the Snake environment.

    Creates parallel training environments via SubprocVecEnv and a separate
    evaluation environment. Periodically evaluates the agent, saves the best
    model, and optionally stops training early if a reward threshold is reached.

    Args:
        model_name:       "DQN" or "PPO".
        grid_width:       Grid width in cells.
        grid_height:      Grid height in cells.
        snake_fov_radius: Agent's field-of-view radius.
        timesteps:        Total training timesteps.
        num_envs:         Number of parallel environments (subprocesses).
        new:              If True, create a fresh model. If False, load from disk.
        params:           Optional dict of hyperparameters (DQN only). Uses defaults if None.
        best:             If loading (new=False), load the "best_model" (True) or "last_model" (False)
                          checkpoint (filenames carry the timesteps they were trained for, e.g.
                          "best_model_3000000.zip" -- see _find_checkpoint()).
        use_tuned_params: If True and params is None (DQN only), load hyperparameters
                          previously found by rl.hyperparameter_tuning.run_hyperparameter_optimization()
                          instead of using the hardcoded defaults.
        use_cnn:          If True, use the "grid" observation mode (2D FOV + apple direction)
                          with a custom CNN feature extractor (SnakeCombinedExtractor) instead
                          of the flat MultiDiscrete + MlpPolicy setup. Saved under a separate
                          "GRID" folder, incompatible with "FLAT" (MLP) models.
        cancel_event:     Optional threading.Event. When set, training stops early (as if
                          `timesteps` had been reached) -- the normal save/finalize/evaluate
                          steps below still run against whatever was trained so far.
        discard_event:    Optional threading.Event. When set (only meaningful together with
                          cancel_event), the run is abandoned after stopping: nothing is saved
                          or evaluated, unlike a plain cancel_event which keeps the partial result.
        on_log_dir:       Optional callable(str), invoked once with the tensorboard log
                          directory as soon as it's known -- right at the start of
                          training, not just at the end (unlike this function's own
                          return value) -- so callers (e.g. the UI) can live-poll the
                          same event file while training is still running.
        on_frame:         Optional callable(np.ndarray), invoked periodically (throttled
                          to roughly every 0.12s of wall-clock time, not every step) with
                          an (H, W, 3) uint8 RGB frame rendered by training-env worker 0,
                          so a UI can show a live view of training. render_mode is only
                          switched on for train_env when this is provided -- zero extra
                          overhead otherwise.
    """
    obs_mode = "grid" if use_cnn else "flat"
    policy = "MultiInputPolicy" if use_cnn else "MlpPolicy"
    policy_kwargs = {"features_extractor_class": SnakeCombinedExtractor} if use_cnn else None

    # Create parallel training environments (one per subprocess). render_mode
    # is only turned on when on_frame is requested -- render() is only ever
    # actually called on worker 0 (see _FrameCallback below), so enabling it
    # uniformly across all workers costs nothing on the ones it's never
    # called on.
    train_env_render_mode = "rgb_array" if on_frame is not None else None
    train_env = SubprocVecEnv([
        make_snake_env(GRID_SIZE, grid_width, grid_height, snake_fov_radius, obs_mode=obs_mode, render_mode=train_env_render_mode)
        for _ in range(num_envs)
    ])

    # Create a single evaluation environment with a step limit to prevent infinite episodes.
    # training=False disables reward shaping/penalties, so EvalCallback's mean_reward
    # reflects the clean score (apples eaten) instead of being mixed with training-only
    # shaping terms.
    raw_eval_env = SnakeGameEnvironment(GRID_SIZE, grid_width, grid_height, snake_fov_radius, training=False, obs_mode=obs_mode)
    raw_eval_env = TimeLimit(raw_eval_env, max_episode_steps=10000)
    eval_env = Monitor(raw_eval_env, filename=None)

    # Set below (to the loaded checkpoint's cumulative timestep count) when
    # continuing an existing model -- but the actual continue_markers.json
    # write is deferred until we know the run wasn't discarded (see the
    # discard_event handling below), so a discarded continuation never
    # permanently pollutes it.
    continue_marker_step = None

    if model_name == "DQN":
        # Build the save/load path based on obs_mode, grid size, and FOV
        path = os.path.join(DQN_PATH, "GRID" if use_cnn else "FLAT", f"GRID_{grid_width}_{grid_height}", f"FOV_RADIUS_{snake_fov_radius}")
        if not new:
            # Resume training from a saved model (policy/feature-extractor are restored from the checkpoint)
            model = DQN.load(_find_checkpoint(path, "best_model" if best else "last_model"), train_env, device='cpu', verbose=1, tensorboard_log=tensorboard_log_dir(path))
            # Restore the replay buffer saved at the end of the previous run
            # (see the final-save block below) so this continuation genuinely
            # resumes training -- same buffer contents, same exploration
            # schedule state -- instead of restarting with an empty buffer,
            # matching "train 3M then 3M more" to "train 6M in one go".
            buffer_path = replay_buffer_path(path)
            if os.path.exists(buffer_path):
                model.load_replay_buffer(buffer_path)
            else:
                # No saved buffer (e.g. a model continued for the first time
                # since before this feature existed) -- fall back to the old
                # behavior: give exploration a fresh cycle over just this
                # continuation's timesteps instead of starting already at its
                # epsilon floor with an empty buffer to learn from (see
                # _rebase_schedule_for_continuation's docstring; deliberately
                # not applied to the learning rate, see the same docstring).
                model.exploration_schedule = _rebase_schedule_for_continuation(model.exploration_schedule, model, timesteps)
            # Where this continuation resumed from, so the UI's live plot can
            # draw a vertical line there (see ui/plot_window.py) --
            # model.num_timesteps here is the loaded checkpoint's cumulative
            # count, i.e. exactly the x-position this continuation starts at.
            # Captured now (before .learn() mutates num_timesteps); actually
            # recorded further down, once we know the run wasn't discarded.
            continue_marker_step = model.num_timesteps
        else:
            # Default DQN hyperparameters, or previously tuned ones from Optuna
            if params is None:
                if use_tuned_params:
                    params = load_best_params()
                else:
                    params = {
                        "learning_rate": linear_schedule(6e-5),
                        "buffer_size": 2_000_000,
                        "learning_starts": 50_000,      # Steps before training starts (fill replay buffer)
                        "batch_size": 256,
                        "tau": 0.4,                      # Soft update coefficient for target network
                        "gamma": 0.988,                  # Discount factor
                        "train_freq": 4,                 # Train every N steps
                        # Gradient updates per training step, scaled with num_envs
                        # (calibrated so the existing default, num_envs=4, still
                        # gives 4): all num_envs write into the same replay
                        # buffer, so without this, more parallel envs collect
                        # data faster without training on proportionally more of
                        # it (a shrinking "replay ratio"), risking under-training
                        # at high num_envs.
                        "gradient_steps": max(1, round(4 * num_envs / 4)),
                        "target_update_interval": 2000,  # Steps between target network hard updates
                        "exploration_fraction": 0.2,     # Fraction of training for epsilon decay
                        "exploration_final_eps": 0.05,   # Final exploration rate
                    }

            model = DQN(
                policy, train_env,
                policy_kwargs = policy_kwargs,
                learning_rate = params["learning_rate"],
                buffer_size = params["buffer_size"],
                learning_starts = params["learning_starts"],
                batch_size = params["batch_size"],
                tau = params["tau"],
                gamma = params["gamma"],
                train_freq = params["train_freq"],
                gradient_steps = params["gradient_steps"],
                target_update_interval = params["target_update_interval"],
                exploration_fraction = params["exploration_fraction"],
                exploration_final_eps = params["exploration_final_eps"],
                device='cpu', verbose=1, tensorboard_log=tensorboard_log_dir(path)
            )
    else:
        # PPO branch
        path = os.path.join(PPO_PATH, "GRID" if use_cnn else "FLAT", f"GRID_{grid_width}_{grid_height}", f"FOV_RADIUS_{snake_fov_radius}")
        if not new:
            model = PPO.load(path=_find_checkpoint(path, "best_model" if best else "last_model"), env=train_env, device='cpu', verbose=1, force_reset=True, tensorboard_log=tensorboard_log_dir(path))
            # No schedule rebase here (unlike DQN's exploration_schedule above) --
            # PPO has no exploration_schedule, and the learning rate deliberately
            # keeps SB3's natural continued-decay behavior (see
            # _rebase_schedule_for_continuation's docstring for why).
            # Same continuation marker as the DQN branch above.
            continue_marker_step = model.num_timesteps

            print(f"Successfully loaded PPO Model ({model._total_timesteps} total_timesteps) [from Path: {path}]")
        else:
            model = PPO(
                policy, train_env,
                policy_kwargs = policy_kwargs,
                n_steps=5000,          # Steps per rollout before policy update
                batch_size=1000,       # Minibatch size for PPO updates
                n_epochs=4,            # Passes over each rollout (reduced from SB3's default 10,
                                       # since batch_size=250/n_epochs=10 meant hundreds of gradient
                                       # steps per rollout -- too many updates on the same on-policy
                                       # data, risking overfitting/instability)
                ent_coef=0.01,         # Small entropy bonus: SB3's PPO default is 0.0 (no exploration
                                       # incentive at all), risky with Snake's sparse reward
                learning_rate=linear_schedule(3e-4),  # Makes SB3's previously-implicit default explicit and lets it decay
                device='cpu', verbose=1, tensorboard_log=tensorboard_log_dir(path)
            )

    # Stop training early once mean reward is clearly very good. Scaled to board
    # size (not a fixed constant) since a given score means very different things
    # on a small vs. a large grid -- ~30% of all cells eaten as apples.
    reward_threshold = max(10, int(0.3 * grid_width * grid_height))
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)

    class _RecordBestTimestep(BaseCallback):
        """Captures self.num_timesteps whenever EvalCallback finds a new best
        (fires exactly when EvalCallback saves best_model.zip, before renaming
        it with a timestep suffix below) -- _finalize_checkpoint() must use
        THIS value for "best_model"'s filename suffix, not the run's final
        total, otherwise the filename (and thus ui.models._discover_models()'s
        best_timesteps, shown throughout the UI, and used as the "Continue
        Existing" resume point) silently claims best_model.zip is from later
        in training than it actually is."""
        def __init__(self):
            super().__init__()
            self.value = None

        def _on_step(self) -> bool:
            self.value = self.num_timesteps
            return True

    record_best_timestep = _RecordBestTimestep()

    # Evaluate periodically, save best model, and optionally stop early
    eval_freq = max((timesteps // 25) // num_envs, 1)    # ~25 evaluations per training run
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=CallbackList([stop_callback, record_best_timestep]),
        best_model_save_path=path,
        eval_freq=eval_freq,
        n_eval_episodes=10,                                  # Episodes per evaluation
        verbose=1,
        deterministic=True
    )
    if not new:
        # EvalCallback.__init__ always starts best_mean_reward at -inf, with
        # no memory of a model's actual historical best across continuations
        # (a fresh instance is constructed on every train_model() call) --
        # without seeding it here, the very first eval of ANY continuation
        # would always count as a "new best" and overwrite best_model.zip,
        # even if this continuation is actually performing worse than the
        # model's true historical best. Not done for new=True: a fresh model
        # (even one overwriting an existing config) should start with no
        # inherited "best" baseline at all.
        previous_best_score = _read_best_score(path)
        if previous_best_score is not None:
            eval_callback.best_mean_reward = previous_best_score

    # Custom callback to log death causes (max-step vs collision)
    death_logger = DeathLogger()

    # Periodically saves last_model too (EvalCallback above already gives
    # best_model this same safety net via its own eval-triggered saves), so a
    # crash mid-run loses at most eval_freq calls of progress instead of the
    # whole run. Reuses the same "~25 checkpoints per run" cadence instead of
    # inventing a new constant.
    periodic_checkpoint = PeriodicCheckpoint(save_freq=eval_freq, save_path=path)

    callbacks = [eval_callback, death_logger, periodic_checkpoint]
    if cancel_event is not None:
        class _CancelCallback(BaseCallback):
            def _on_step(self) -> bool:
                return not cancel_event.is_set()
        callbacks.append(_CancelCallback())

    if on_log_dir is not None:
        class _LogDirCallback(BaseCallback):
            # SB3 only sets self.model.logger once _setup_learn() runs, right at
            # the start of .learn() -- before this hook fires, but after
            # construction/.load(), so this is the earliest point it's available.
            def _on_training_start(self) -> None:
                on_log_dir(self.model.logger.dir)

            def _on_step(self) -> bool:
                return True
        callbacks.append(_LogDirCallback())

    if on_frame is not None:
        class _FrameCallback(BaseCallback):
            """Periodically pulls one rendered frame from training-env worker 0
            and hands it to on_frame(), throttled by wall-clock time (not step
            count) -- _on_step() fires once per vectorized step across all
            num_envs workers combined, which can be hundreds/thousands of times
            a second, while the UI can only usefully redraw every ~100-150ms
            anyway (see ui/game_view.py)."""
            def __init__(self, on_frame, min_interval=0.12):
                super().__init__()
                self._on_frame_cb = on_frame
                self._min_interval = min_interval
                self._last_frame_time = 0.0

            def _on_step(self) -> bool:
                now = time.time()
                if now - self._last_frame_time >= self._min_interval:
                    self._last_frame_time = now
                    try:
                        frame = self.training_env.env_method("render", indices=[0])[0]
                    except Exception:
                        frame = None  # Transient IPC hiccup -- skip this tick, don't kill the run
                    if frame is not None:
                        self._on_frame_cb(frame)
                return True
        callbacks.append(_FrameCallback(on_frame))

    # Snapshot the tensorboard run dir's contents before .learn() starts
    # writing to it, so a discarded (and NOT reseeded, see needs_reseed below)
    # run can be told apart from whatever was legitimately there already
    # (see the discard_event handling below).
    pre_existing_tb_files = _snapshot_run_dir(path)
    # Highest step already plotted for this model, if any -- compared against
    # this run's resume point (0 for "New Model", the loaded checkpoint's
    # timestep for "Continue Existing") to detect a "rewind": a "New Model"
    # overwrite starting back at step 0, or a "Continue Existing" resume from
    # an earlier checkpoint (e.g. "Best") than what's already logged. Without
    # reseeding, the old, now-superseded history would overlap with this
    # run's freshly-written data at the same steps, and -- since the
    # overlap-safe reseed below has to happen BEFORE .learn() starts writing,
    # not after it finishes -- the live plot would show that overlap for the
    # run's entire duration, not just in the final saved result.
    existing_max_step = _existing_max_step(path)
    resume_point = continue_marker_step if continue_marker_step is not None else 0
    needs_reseed = existing_max_step is not None and resume_point < existing_max_step
    if needs_reseed:
        if resume_point > 0 and not os.path.isdir(tb_best_dir(path)):
            # Backward compatibility: a model whose best/last already
            # diverged before this two-track mechanism existed (or that
            # simply never had a best_model_updated run since) has no
            # tb_best snapshot yet to seed from -- build one now, from the
            # CURRENT "last" track's own history (still intact at this
            # point, before the backup below), truncated at this model's
            # best timestep. Without this, seeding below finds nothing to
            # copy and the live plot starts completely blank instead of
            # showing this checkpoint's real history.
            _rebuild_tb_best(path, resume_point)
        # Move the current ("last") history aside rather than deleting it
        # outright, so a discarded run can restore it exactly (see the
        # discard_event handling below) -- then seed the live run dir with
        # the resumed checkpoint's OWN history (tb_best, if resuming from a
        # real earlier point) or leave it empty (a "New Model" overwrite,
        # resume_point == 0, has no prior history of its own to seed from).
        _backup_run_dir(path)
        if resume_point > 0:
            _seed_run_dir_from_best(path)

    # train_env/eval_env are real OS subprocesses (SubprocVecEnv) -- guarantee
    # they're always closed exactly once, whether training finishes normally,
    # is cancelled (discarded or not), or raises.
    try:
        # Start training. tb_log_name doesn't need to be descriptive -- unlike
        # the old shared Training/Logs/ tree, tensorboard_log= above already
        # nests the run under this model's own, already-unique checkpoint
        # folder (see TB_RUN_NAME's docstring). SB3's configure_logger()
        # deliberately reuses the same run folder (doesn't bump the run number)
        # when reset_num_timesteps=False, so a "Continue Existing" run's reward
        # curve keeps appending to the original run's, not starting a new one.
        model.learn(total_timesteps=timesteps, callback=callbacks, reset_num_timesteps=False, tb_log_name=TB_RUN_NAME)

        if discard_event is not None and discard_event.is_set():
            print("\nTraining cancelled -- discarding this run (nothing saved).")
            # Delete the live, unfinalized checkpoints PeriodicCheckpoint/
            # EvalCallback wrote during this run, so a discarded run leaves the
            # folder exactly as it was before it started -- _finalize_checkpoint()
            # (which would otherwise clean these up) never runs on this path.
            for prefix in ("best_model", "last_model"):
                stray = os.path.join(path, f"{prefix}.zip")
                if os.path.exists(stray):
                    os.remove(stray)
            # Same for the tensorboard event file(s) this attempt wrote --
            # without this they'd sit in the same tb_0 folder as legitimate
            # runs and get merged into future plots. Two cases: if we backed
            # up+reseeded before training (needs_reseed), undo that exactly;
            # otherwise (the normal, non-rewinding case) fall back to the
            # snapshot-diff cleanup (see _discard_run_artifacts's docstring).
            if needs_reseed:
                _restore_run_dir_backup(path)
            else:
                _discard_run_artifacts(path, pre_existing_tb_files)
            return

        # This run wasn't discarded (finished normally, or was cancelled but
        # kept via "Cancel & Save Last Model") -- now safe to actually record
        # the continuation marker captured above, if this was a "Continue
        # Existing" run.
        if continue_marker_step is not None:
            _record_continue_marker(path, continue_marker_step)

        if needs_reseed:
            # The run succeeded (or was cancelled-and-kept) -- the pre-run
            # backup is no longer needed.
            _discard_run_dir_backup(path)

        # EvalCallback only evaluates every eval_freq calls, so the final stretch of
        # training (up to eval_freq-1 calls) may never have been checked -- evaluate
        # once more now so the just-finished model is guaranteed to be compared
        # against the best one seen so far, and saved as the new best if it wins.
        final_mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=eval_callback.n_eval_episodes, deterministic=True)

        def _scalar_mean_reward(value) -> float:
            if isinstance(value, (list, tuple, np.ndarray)):
                return float(np.mean(value))
            return float(value)

        final_mean_reward = _scalar_mean_reward(final_mean_reward)
        best_mean_reward = _scalar_mean_reward(eval_callback.best_mean_reward)

        # Bake the cumulative trained timesteps into the checkpoint filenames (e.g.
        # "best_model_3000000.zip"). model.num_timesteps already reflects the right
        # value here: it started at 0 for a fresh model (new=True), or continued
        # counting up from the loaded checkpoint's timesteps (new=False, since
        # model.learn() above was called with reset_num_timesteps=False) -- so
        # continuing training accumulates instead of overwriting the count.
        total_timesteps_trained = model.num_timesteps
        # "best_model"'s filename suffix must match whatever timestep its
        # actual weights were saved at, not the run's final total -- default
        # to record_best_timestep.value (set by EvalCallback's callback_on_new_best,
        # see above), which is exactly that. best_model_score mirrors it in
        # parallel (see best_score_path's docstring) -- eval_callback.best_mean_reward
        # already reflects the same moment, since it's only ever updated
        # (line "self.best_mean_reward = float(mean_reward)" inside SB3's
        # EvalCallback._on_step()) immediately before callback_on_new_best
        # fires. Now that eval_callback.best_mean_reward is seeded from this
        # model's true historical best on a continuation (see above), this
        # run may genuinely find no improvement at all -- record_best_timestep.value
        # then stays None and the block below never fires, correctly leaving
        # the existing best_model_*.zip untouched.
        best_model_timestep = record_best_timestep.value if record_best_timestep.value is not None else total_timesteps_trained
        best_model_score = eval_callback.best_mean_reward

        if final_mean_reward > best_mean_reward:
            print(f"Final evaluation found a new best model: {final_mean_reward:.2f} (previous best: {best_mean_reward:.2f})")
            model.save(os.path.join(path, "best_model"))
            # This save supersedes EvalCallback's (if any) -- it's the just-
            # finished model's own final state, so its filename must reflect
            # the final total, not whatever EvalCallback last recorded.
            best_model_timestep = total_timesteps_trained
            best_model_score = final_mean_reward

        # Always save the final model (in addition to the best model saved by EvalCallback)
        model.save(os.path.join(path, "last_model"))

        if model_name == "DQN":
            # Snapshot the replay buffer so the next "Continue Existing" run
            # can resume from it instead of starting empty (see the load
            # branch above) -- only at this natural end point, not from
            # PeriodicCheckpoint's crash-safety saves, since buffer files are
            # large (~1.5-2GB); a mid-run crash just falls back to the
            # graceful empty-buffer path on the next continuation, same as
            # before this feature existed.
            model.save_replay_buffer(replay_buffer_path(path))

        # Whether a genuinely new best_model.zip exists to finalize -- False
        # is now a real, expected outcome for a continuation that never beat
        # this model's true historical best (see the seeding above), in
        # which case the existing best_model_*.zip/best_score.json are
        # correctly left completely untouched.
        best_model_updated = os.path.exists(os.path.join(path, "best_model.zip"))

        _finalize_checkpoint(path, "best_model", best_model_timestep)
        _finalize_checkpoint(path, "last_model", total_timesteps_trained)

        if best_model_updated:
            _write_best_score(path, best_model_score)
            # Keep tb_best in sync with the checkpoint it describes -- a
            # truncated-at-best_model_timestep snapshot of this run's own
            # (now current) "last" history, so the NEXT "Continue from Best"
            # (if any) can seed its live plot from it (see
            # _seed_run_dir_from_best).
            _rebuild_tb_best(path, best_model_timestep)
    finally:
        train_env.close()
        eval_env.close()

    # Evaluate both checkpoints in this folder (deterministic + stochastic, no
    # rendering, clean score) and save the results so performance can be read
    # at a glance without manually testing the model.
    print("\n=== Starting post-training evaluation ===")
    evaluation = {
        "timesteps": total_timesteps_trained,
        "last_model": evaluate_model_performance(model, GRID_SIZE, grid_width, grid_height, snake_fov_radius, obs_mode=obs_mode, model_label="last model"),
    }

    if glob.glob(os.path.join(path, "best_model_*.zip")):
        ModelClass = DQN if model_name == "DQN" else PPO
        best = ModelClass.load(_find_checkpoint(path, "best_model"), device="cpu")
        evaluation["best_model"] = evaluate_model_performance(best, GRID_SIZE, grid_width, grid_height, snake_fov_radius, obs_mode=obs_mode, model_label="best model")

    evaluation_path = os.path.join(path, "evaluation.json")
    print(f"\nEvaluation complete. Writing results to {evaluation_path}")
    os.makedirs(path, exist_ok=True)
    with open(evaluation_path, "w") as file:
        json.dump(evaluation, file, indent=4)

    # The exact tensorboard log directory for this run, so the UI can plot it
    # once training finishes (see ui/plot_window.py).
    return model.logger.dir
