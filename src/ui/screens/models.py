"""
ui/screens/models.py - "Models" screen: browse, filter, search, continue-train, and
delete trained model checkpoints. The only screen with a Delete button -- Test Model's
and Train Model's own model lists intentionally don't get one.
"""

import tkinter as tk
import customtkinter as ctk

from rl.paths import GRID_PRESETS
from ui.theme import PANEL, BORDER, TEXT, TEXT_MUTED, AMBER, RED, BLUE, RADIUS
from ui.widgets import (
    _make_content_column, _make_choice_row, _make_outline_button,
    _enable_mousewheel, _make_model_badge, show_confirm_dialog,
)
from ui.models import _discover_models, _delete_model
from ui.screens.base import SubScreen


class ModelsScreen(SubScreen):
    def __init__(self, parent, app):
        super().__init__(parent, app, "MODELS")

        filters = _make_content_column(self.body)
        self.algo_seg = _make_choice_row(filters, "Algorithm", ["All", "PPO", "DQN"], "All", app.font_body, command=self._apply_filters)
        self.obs_seg = _make_choice_row(filters, "Observation mode", ["All", "FLAT", "GRID"], "All", app.font_body, command=self._apply_filters)
        self.grid_seg = _make_choice_row(filters, "Grid size", ["All"] + [p[0] for p in GRID_PRESETS], "All", app.font_body, command=self._apply_filters)

        search_row = ctk.CTkFrame(filters, fg_color="transparent")
        search_row.pack(fill="x", pady=10)
        ctk.CTkLabel(search_row, text="Search", font=app.font_body, text_color=TEXT).pack(anchor="w")
        self.search_var = tk.StringVar()
        ctk.CTkEntry(
            search_row, textvariable=self.search_var, corner_radius=RADIUS,
            fg_color=PANEL, border_color=BORDER, text_color=TEXT, font=app.font_body,
            placeholder_text="e.g. PPO 30x20 FOV5",
        ).pack(fill="x", pady=(4, 0))
        # trace_add (not a <KeyRelease> bind) so paste/clear/any edit re-filters
        # too, not just literal keystrokes.
        self.search_var.trace_add("write", lambda *_args: self._apply_filters())
        filters.pack(fill="x", pady=(0, 8))

        self.list_frame = ctk.CTkScrollableFrame(self.body, fg_color="transparent", corner_radius=RADIUS)
        self.list_frame.pack(fill="both", expand=True)

        self._refresh_models()

    # --- filtering -------------------------------------------------------------

    def _apply_filters(self, _value=None):
        self._refresh_models()

    def _matches_filters(self, info):
        if self.algo_seg.get() != "All" and info["algo"] != self.algo_seg.get():
            return False
        if self.obs_seg.get() != "All" and info["obs_mode"] != self.obs_seg.get():
            return False
        grid_label = self.grid_seg.get()
        if grid_label != "All":
            size = next(((w, h) for label, w, h in GRID_PRESETS if label == grid_label), None)
            if size != (info["grid_width"], info["grid_height"]):
                return False
        query = self.search_var.get().strip().lower()
        if query:
            haystack = f"{info['algo']} {info['obs_mode']} {info['grid_width']}x{info['grid_height']} FOV{info['fov']}".lower()
            if query not in haystack:
                return False
        return True

    # --- list --------------------------------------------------------------

    def _refresh_models(self):
        """Rebuilds the (filtered) card list from disk -- called on init, on
        every filter/search change, and after a delete."""
        for child in self.list_frame.winfo_children():
            child.destroy()

        models = [info for info in _discover_models() if self._matches_filters(info)]
        if not models:
            ctk.CTkLabel(
                self.list_frame, text="No models match these filters.",
                font=self.app.font_body, text_color=TEXT_MUTED,
            ).pack(pady=20)
        for info in models:
            self._make_model_card(self.list_frame, info).pack(fill="x", pady=6)
        _enable_mousewheel(self.list_frame)

        # Filtering/searching to a much shorter list doesn't itself move the
        # scroll position -- without resetting it, a rebuild while scrolled
        # down can leave the viewport looking blank (scrolled past the now-
        # shorter content) even though the (correct, filtered) result is
        # right there at the top.
        self.list_frame._parent_canvas.yview_moveto(0)

    def _make_model_card(self, parent, info):
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

        btn_row = ctk.CTkFrame(frame, fg_color="transparent")
        btn_row.pack(anchor="e", padx=12, pady=(6, 10))
        _make_outline_button(
            btn_row, "Continue Training", BLUE, lambda: self._continue_training(info),
            self.app.font_small, width=160, height=36,
        ).pack(side="left", padx=(0, 8))
        _make_outline_button(
            btn_row, "Delete", RED, lambda: self._request_delete(info),
            self.app.font_small, width=100, height=36,
        ).pack(side="left")

        return frame

    # --- actions -------------------------------------------------------------

    def _continue_training(self, info):
        self.app.show("TrainModelScreen")
        self.app.screens["TrainModelScreen"].select_model_to_continue(info)

    def _request_delete(self, info):
        label = f"{info['algo']}/{info['obs_mode']} Grid {info['grid_width']}x{info['grid_height']} FOV {info['fov']}"
        show_confirm_dialog(
            self.app, "Delete Model?", f"Permanently delete the {label} model? This cannot be undone.",
            [("Cancel", AMBER, None), ("Delete", RED, lambda: self._do_delete(info))],
        )

    def _do_delete(self, info):
        _delete_model(info["path"])
        self._refresh_models()
        # Keep the two other model-list screens from pointing at what was just
        # deleted (see Train Model's own continue-list and Test Model's list).
        train_screen = self.app.screens.get("TrainModelScreen")
        if train_screen is not None:
            train_screen._refresh_continue_models()
        test_screen = self.app.screens.get("TestModelScreen")
        if test_screen is not None:
            test_screen._refresh_models()
