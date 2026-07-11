"""
ui/screens/train_model.py - "Train Model" screen: train a new model or continue an existing one.
"""

import os
import threading
import tkinter as tk
import customtkinter as ctk

from rl.training import train_model
from rl.hyperparameter_tuning import PATH as DQN_TUNING_PATH
from rl.paths import GRID_PRESETS
from ui.theme import PANEL, BORDER, TEXT, TEXT_MUTED, AMBER, RED, GREEN, BLUE, RADIUS, _mix
from ui.widgets import (
    _make_content_column, _make_choice_row, _make_slider_row, _make_entry_row,
    _make_outline_button, _bind_recursive, _enable_mousewheel, show_confirm_dialog, _make_model_badge,
)
from ui.models import _discover_models, _read_continue_markers
from ui.plot_window import LiveTrainingPlot
from ui.screens.base import SubScreen


class TrainModelScreen(SubScreen):
    MIN_TIMESTEPS = 1000

    def __init__(self, parent, app):
        super().__init__(parent, app, "TRAIN MODEL")
        # Grid size is restricted to 3 curated presets (rather than free
        # width/height) since it determines the save-folder structure
        # (GRID_{w}_{h}); shared with PlayScreen (rl.paths.GRID_PRESETS) so
        # both screens offer the same sizes.
        self._grid_by_label = {label: (w, h) for label, w, h in GRID_PRESETS}
        tuned_params_path = os.path.join(DQN_TUNING_PATH, "best_dqn_params.json")
        self._tuned_params_available = os.path.exists(tuned_params_path)

        self._continue_selected = None
        self._continue_selected_card = None
        self._collision_match = None
        self._is_training = False
        self._cancel_event = None
        self._discard_event = None
        self._current_log_dir = None
        self._current_model_path = None

        # The whole form -- mode toggle, algo/obs/grid/fov (or the continue-
        # model list), timesteps/parallel-envs/tuned-checkbox, start button,
        # log -- lives inside one outer scrollable frame instead of packing
        # the log/button/shared controls as fixed, always-visible chrome
        # around a small inner scroll region: scrolling down now reveals more
        # of the (taller) log, and no field is pinned to a fixed spot. The
        # model list and the log each get their own nested, height-bounded
        # scroll region inside that (see _rebind_mousewheel below).
        self.outer_scroll = ctk.CTkScrollableFrame(self.body, fg_color="transparent", corner_radius=RADIUS)
        self.outer_scroll.pack(fill="both", expand=True)

        self.train_mode_seg = _make_choice_row(
            self.outer_scroll, "Train", ["New Model", "Continue Existing"], "New Model", app.font_body,
            command=self._on_train_mode_change,
        )

        # Full width of the scroll region (not content-column-capped) so the
        # continue panel's 2-column [card list | resume-from] layout has room;
        # new_panel re-caps its own width below, matching every other form.
        self.mode_container = ctk.CTkFrame(self.outer_scroll, fg_color="transparent")
        self.mode_container.pack(fill="x", expand=True, pady=(8, 0))

        self.new_panel = _make_content_column(self.mode_container)
        self.continue_panel = ctk.CTkFrame(self.mode_container, fg_color="transparent")
        self._build_new_panel(self.new_panel)
        self._build_continue_panel(self.continue_panel)
        self.new_panel.pack(fill="y", expand=True)

        shared = _make_content_column(self.outer_scroll)
        self.timesteps_entry = _make_entry_row(shared, "Timesteps", 3_000_000, app.font_body)
        cpu_count = os.cpu_count() or 4
        self.num_envs_var = _make_slider_row(shared, "Parallel environments", 1, cpu_count, 1, min(4, cpu_count), app.font_body)
        self.tuned_var = tk.BooleanVar(value=False)
        self.tuned_checkbox = ctk.CTkCheckBox(
            shared, text="Use tuned DQN hyperparameters (from rl/hyperparameter_tuning.py)", variable=self.tuned_var,
            font=app.font_small, text_color=TEXT, checkbox_width=18, checkbox_height=18,
            fg_color=AMBER, hover_color=_mix(AMBER, "#000000", 0.2), border_color=BORDER,
        )
        self.tuned_checkbox.pack(anchor="w", pady=(8, 4))
        shared.pack(fill="y", pady=(16, 0))

        self.start_btn = _make_outline_button(self.outer_scroll, "Start Training", BLUE, self._start, app.font_body, width=220, height=44)
        self.start_btn.pack(pady=(16, 16))
        # CTkButton renders text_color_disabled (not text_color) while disabled
        # (see ctk_button.py: _draw()) -- save the theme's real default instead
        # of guessing a value to restore once cancelling's red state is over.
        self._start_btn_default_text_color_disabled = self.start_btn.cget("text_color_disabled")

        # Taller than the other screens' log box (140px) -- this is the field
        # the user specifically wanted more of visible at once; identical
        # styling to SubScreen._make_log_box's default, just more height.
        log_box = self._make_log_box(self.outer_scroll, height=400)
        log_box.pack(fill="both", expand=False, pady=(0, 8))

        # Live reward/loss graph below the text log -- polled while training
        # runs (see _poll_plot()), not just shown once at the end, since
        # train_model()'s on_log_dir callback reports the tensorboard log
        # directory right at the start of training.
        self.plot_widget = LiveTrainingPlot(self.outer_scroll, app)
        self.plot_widget.pack(fill="x", pady=(0, 8))

        self._on_algo_change("DQN")
        self._check_collision()
        self._update_start_enabled()
        self._rebind_mousewheel()

    # --- "New Model" panel ---------------------------------------------------

    def _build_new_panel(self, parent):
        self.algo_seg = _make_choice_row(parent, "Algorithm", ["DQN", "PPO"], "DQN", self.app.font_body, command=self._on_new_config_change)
        self.obs_seg = _make_choice_row(parent, "Observation mode", ["FLAT", "GRID"], "FLAT", self.app.font_body, command=self._on_new_config_change)
        self.grid_seg = _make_choice_row(parent, "Grid size", [p[0] for p in GRID_PRESETS], GRID_PRESETS[0][0], self.app.font_body, command=self._on_new_config_change)
        self.fov_var = _make_slider_row(parent, "FOV radius", 1, 8, 1, 3, self.app.font_body, on_change=self._on_new_config_change)

        self.warning_frame = ctk.CTkFrame(parent, fg_color=_mix(PANEL, RED, 0.15), corner_radius=RADIUS, border_width=2, border_color=RED)
        self.warning_label = ctk.CTkLabel(
            self.warning_frame, text="", font=self.app.font_small, text_color=TEXT, wraplength=500, justify="left",
        )
        self.warning_label.pack(anchor="w", padx=12, pady=(10, 4))
        _make_outline_button(
            self.warning_frame, "Continue this model instead", AMBER, self._switch_to_continue_for_collision,
            self.app.font_small, width=240, height=36,
        ).pack(anchor="w", padx=12, pady=(0, 10))

    def _on_new_config_change(self, _value=None):
        if self.train_mode_seg.get() != "New Model":
            return
        self._on_algo_change(self.algo_seg.get())
        self._check_collision()

    def _check_collision(self):
        grid_width, grid_height = self._grid_by_label[self.grid_seg.get()]
        algo = self.algo_seg.get()
        use_cnn = self.obs_seg.get() == "GRID"
        fov = int(self.fov_var.get())

        match = next(
            (info for info in _discover_models()
             if info["algo"] == algo and info["use_cnn"] == use_cnn
             and info["grid_width"] == grid_width and info["grid_height"] == grid_height and info["fov"] == fov),
            None,
        )
        self._collision_match = match

        if match is None:
            self.warning_frame.pack_forget()
            return

        steps = match["best_timesteps"] if match["best_timesteps"] is not None else match["last_timesteps"]
        lines = [f"A model already exists for this configuration ({steps:,} timesteps trained)."]
        evaluation = match["evaluation"]
        if evaluation:
            for key, label in (("best_model", "best"), ("last_model", "last")):
                if key in evaluation:
                    det = evaluation[key]["deterministic"]["mean_score"]
                    sto = evaluation[key]["stochastic"]["mean_score"]
                    lines.append(f"{label}: deterministic {det:.1f} | stochastic {sto:.1f}")
        lines.append("Starting a new run here will overwrite it once training finishes.")
        self.warning_label.configure(text="\n".join(lines))
        self.warning_frame.pack(fill="x", pady=(8, 0))

    def _switch_to_continue_for_collision(self):
        self.select_model_to_continue(self._collision_match)

    def select_model_to_continue(self, info):
        """Entry point for other screens (e.g. ModelsScreen's "Continue Training"
        button) to land here with a specific model already selected in the
        Continue Existing flow."""
        self.train_mode_seg.set("Continue Existing")
        self._on_train_mode_change("Continue Existing")
        if info is not None:
            card = self._continue_cards_by_path.get(info["path"])
            if card is not None:
                self._select_continue_card(info, card)

    # --- "Continue Existing" panel -------------------------------------------

    def _build_continue_panel(self, parent):
        # 2-column layout: card list on the left, "Resume from" to its right
        # (only meaningful -- and only shown enabled -- once a card is
        # selected). grid() here is scoped to `parent`'s own children; `parent`
        # itself is still pack()ed/pack_forget()'d by _on_train_mode_change,
        # so mixing geometry managers is safe (different parent each level).
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=0)
        parent.grid_rowconfigure(0, weight=1)

        # Height-bounded so this is a real, independently-scrolling nested
        # region instead of growing to fit every card and defeating the point
        # (see _enable_mousewheel's `nested` docstring in ui/widgets.py).
        self.continue_cards_frame = ctk.CTkScrollableFrame(parent, fg_color="transparent", corner_radius=RADIUS, height=280)
        self.continue_cards_frame.grid(row=0, column=0, sticky="nsew")
        self._continue_cards_by_path = {}
        self._refresh_continue_models()

        resume_col = ctk.CTkFrame(parent, fg_color="transparent")
        resume_col.grid(row=0, column=1, sticky="n", padx=(16, 0))
        self.continue_checkpoint_seg = _make_choice_row(resume_col, "Resume from", ["Best", "Last"], "Best", self.app.font_body)
        self.continue_checkpoint_seg.configure(state="disabled")

    def _refresh_continue_models(self):
        for child in self.continue_cards_frame.winfo_children():
            child.destroy()
        self._continue_cards_by_path.clear()
        self._continue_selected = None
        self._continue_selected_card = None

        models = _discover_models()
        if not models:
            ctk.CTkLabel(
                self.continue_cards_frame, text='No existing models yet -- switch to "New Model" to train your first one.',
                font=self.app.font_body, text_color=TEXT_MUTED, wraplength=500, justify="left",
            ).pack(pady=20)
        else:
            for info in models:
                card = self._make_continue_card(self.continue_cards_frame, info)
                card.pack(fill="x", pady=6)
                self._continue_cards_by_path[info["path"]] = card
        if hasattr(self, "continue_checkpoint_seg"):
            self.continue_checkpoint_seg.configure(state="disabled")
        self._rebind_mousewheel()
        self._update_start_enabled()

    def _rebind_mousewheel(self):
        """
        (Re)bind wheel handling on the outer scroll, with the log box and (if
        built yet) the continue-model card list as independent nested scroll
        targets -- called once at the end of __init__ (after the log box
        exists) and again whenever _refresh_continue_models() rebuilds the
        card list (new card widgets need fresh binds).
        """
        nested = []
        if self.log_box is not None:
            nested.append((self.log_box, lambda d: self.log_box._textbox.yview_scroll(d, "units")))
        if hasattr(self, "continue_cards_frame"):
            cards_canvas = self.continue_cards_frame._parent_canvas
            nested.append((self.continue_cards_frame, lambda d: cards_canvas.yview_scroll(d, "units")))
        _enable_mousewheel(self.outer_scroll, nested=nested)

    def _make_continue_card(self, parent, info):
        frame = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=RADIUS, border_width=2, border_color=BORDER)

        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 2))
        _make_model_badge(header, info, self.app.font_body, TEXT)

        steps = info["best_timesteps"] if info["best_timesteps"] is not None else info["last_timesteps"]
        if steps is not None:
            ctk.CTkLabel(header, text=f"{steps:,} timesteps", font=self.app.font_small, text_color=TEXT_MUTED).pack(side="right")

        evaluation = info["evaluation"]
        if evaluation:
            for key, label in (("best_model", "best"), ("last_model", "last")):
                if key in evaluation:
                    det = evaluation[key]["deterministic"]["mean_score"]
                    sto = evaluation[key]["stochastic"]["mean_score"]
                    ctk.CTkLabel(
                        frame, text=f"{label}: deterministic {det:.1f}  |  stochastic {sto:.1f}",
                        font=self.app.font_small, text_color=TEXT_MUTED,
                    ).pack(anchor="w", padx=12, pady=(0, 2))
        else:
            ctk.CTkLabel(
                frame, text="No evaluation data yet.", font=self.app.font_small, text_color=TEXT_MUTED,
            ).pack(anchor="w", padx=12, pady=(0, 2))

        ctk.CTkFrame(frame, fg_color="transparent", height=6).pack()

        def select(_event=None):
            self._select_continue_card(info, frame)

        _bind_recursive(frame, "<Button-1>", select)
        return frame

    def _select_continue_card(self, info, frame):
        if self._continue_selected_card is not None:
            self._continue_selected_card.configure(border_color=BORDER)
        frame.configure(border_color=AMBER)
        self._continue_selected_card = frame
        self._continue_selected = info

        has_best = info["best_timesteps"] is not None
        has_last = info["last_timesteps"] is not None
        self.continue_checkpoint_seg.configure(state="normal")
        if has_best and not has_last:
            self.continue_checkpoint_seg.set("Best")
            self.continue_checkpoint_seg.configure(state="disabled")
        elif has_last and not has_best:
            self.continue_checkpoint_seg.set("Last")
            self.continue_checkpoint_seg.configure(state="disabled")

        self._on_algo_change(info["algo"])
        self._update_start_enabled()

    # --- shared ---------------------------------------------------------------

    def _on_train_mode_change(self, mode):
        if mode == "New Model":
            self.continue_panel.pack_forget()
            self.new_panel.pack(fill="y", expand=True)
            self._on_algo_change(self.algo_seg.get())
            self._check_collision()
        else:
            self.new_panel.pack_forget()
            self.continue_panel.pack(fill="both", expand=True)
            self.warning_frame.pack_forget()
            self._on_algo_change(self._continue_selected["algo"] if self._continue_selected else None)
        self._update_start_enabled()

    def _on_algo_change(self, algo):
        if algo == "DQN" and self._tuned_params_available:
            self.tuned_checkbox.configure(state="normal")
        else:
            self.tuned_var.set(False)
            self.tuned_checkbox.configure(state="disabled")

    def _update_start_enabled(self):
        if self._is_training or not hasattr(self, "start_btn"):
            return
        if self.train_mode_seg.get() == "Continue Existing" and self._continue_selected is None:
            self.start_btn.configure(state="disabled")
        else:
            self.start_btn.configure(state="normal")

    def _show_error(self, message):
        if self.log_box is not None:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", f"Error: {message}\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")

    def _start(self):
        if self._is_training:
            return

        try:
            timesteps = int(self.timesteps_entry.get())
            if timesteps < self.MIN_TIMESTEPS:
                raise ValueError
        except ValueError:
            self._show_error(f"Timesteps must be an integer >= {self.MIN_TIMESTEPS:,}.")
            return

        if self.train_mode_seg.get() == "New Model":
            grid_width, grid_height = self._grid_by_label[self.grid_seg.get()]
            kwargs = dict(
                model_name=self.algo_seg.get(),
                grid_width=grid_width, grid_height=grid_height,
                snake_fov_radius=int(self.fov_var.get()),
                use_cnn=self.obs_seg.get() == "GRID",
                new=True,
            )
            # No continuation history yet for a fresh model.
            self._current_model_path = None
        else:
            info = self._continue_selected
            if info is None:
                self._show_error("Select a model to continue first.")
                return
            kwargs = dict(
                model_name=info["algo"],
                grid_width=info["grid_width"], grid_height=info["grid_height"],
                snake_fov_radius=info["fov"],
                use_cnn=info["use_cnn"],
                new=False,
                best=self.continue_checkpoint_seg.get() == "Best",
            )
            # Known upfront (unlike the tensorboard log dir) -- lets the live
            # plot show past "Continue Existing" markers from the very first
            # poll, not just once training.py writes a new one.
            self._current_model_path = info["path"]

        kwargs.update(
            timesteps=timesteps,
            num_envs=int(self.num_envs_var.get()),
            use_tuned_params=self.tuned_var.get(),
        )

        self._cancel_event = threading.Event()
        self._discard_event = threading.Event()
        kwargs["cancel_event"] = self._cancel_event
        kwargs["discard_event"] = self._discard_event
        self._current_log_dir = None
        kwargs["on_log_dir"] = self._on_log_dir_known

        self._is_training = True
        self.start_btn.configure(text="Cancel Training", command=self._request_cancel)
        self.plot_widget.reset()
        self._start_background(train_model, kwargs, self.start_btn, on_finish=self._on_training_finished)
        self._poll_plot()

    def _on_log_dir_known(self, log_dir):
        """Called from the training thread (train_model()'s on_log_dir
        callback) as soon as the tensorboard log directory is known -- right
        at the start of training, not just at the end."""
        self._current_log_dir = log_dir

    def _poll_plot(self):
        """Redraws the live plot every ~1.5s while training is running --
        much less often than the text log's 100ms poll, since tensorboard
        itself only gets new data every eval/log interval, not every step."""
        if not self._is_training:
            return
        self.plot_widget.update(self._current_log_dir, self._read_markers())
        self.after(1500, self._poll_plot)

    def _read_markers(self):
        """Continuation-start markers for the model currently training (see
        rl.paths._record_continue_marker), or [] for a fresh "New Model" run
        (self._current_model_path is None) which has no continuation history."""
        if self._current_model_path is None:
            return []
        return _read_continue_markers(self._current_model_path)

    def _on_training_finished(self):
        self._is_training = False
        self.start_btn.configure(
            text="Start Training", command=self._start, text_color=BLUE,
            text_color_disabled=self._start_btn_default_text_color_disabled,
        )
        self._refresh_continue_models()
        if self.train_mode_seg.get() == "New Model":
            self._check_collision()
        self._update_start_enabled()
        # One last redraw to catch anything logged after _poll_plot()'s last
        # tick (e.g. the final evaluation) -- _last_result is train_model()'s
        # return value (the log dir, or None on discard/exception), set by
        # _start_background()'s worker(); _current_log_dir covers the discard
        # case too (train_model() still reports it before discarding).
        self.plot_widget.update(self._last_result or self._current_log_dir, self._read_markers())

    def _request_cancel(self):
        show_confirm_dialog(
            self.app, "Cancel Training?", "Training is still running.",
            [
                ("Keep Training", AMBER, None),
                ("Cancel & Save Last Model", GREEN, self._do_cancel_and_save),
                ("Cancel & Discard", RED, self._do_cancel_and_discard),
            ],
        )

    def _do_cancel_and_save(self):
        if self._cancel_event is not None:
            self._cancel_event.set()
            self._show_cancelling()

    def _do_cancel_and_discard(self):
        if self._cancel_event is not None:
            self._discard_event.set()
            self._cancel_event.set()
            self._show_cancelling()

    def _show_cancelling(self):
        """
        Cancelling isn't instant -- rl.training.train_model() still has to stop
        model.learn() at its next step check and (unless discarding) run a final
        evaluation/save pass before it's actually done -- so leave a visible
        "Cancelling..." trail instead of letting the button keep inviting a
        confusing second click on an already-in-flight cancel.
        """
        self.start_btn.configure(text="Cancelling...", state="disabled", text_color=RED, text_color_disabled=RED)
        if self.log_box is not None:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", "Cancelling...\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")

    def _handle_back(self):
        if not self._is_training:
            super()._handle_back()
            return

        def _cancel_and_leave(discard):
            if discard:
                self._do_cancel_and_discard()
            else:
                self._do_cancel_and_save()
            self.app.show("HomeScreen")

        show_confirm_dialog(
            self.app, "Cancel Training?", "Training is still running. Do you want to cancel it before leaving?",
            [
                ("Keep Training", AMBER, None),
                ("Cancel & Save Last Model", GREEN, lambda: _cancel_and_leave(False)),
                ("Cancel & Discard", RED, lambda: _cancel_and_leave(True)),
            ],
        )
