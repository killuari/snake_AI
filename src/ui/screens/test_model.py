"""
ui/screens/test_model.py - "Test Model" screen: pick a trained model and watch it play.
"""

import customtkinter as ctk

from rl.playback import test_model
from ui.theme import PANEL, BORDER, TEXT, TEXT_MUTED, AMBER, RED, RADIUS
from ui.widgets import _make_content_column, _make_choice_row, _make_slider_row, _make_outline_button, _bind_recursive, _enable_mousewheel, _speed_label, _make_model_badge
from ui.models import _discover_models
from ui.screens.base import SubScreen


class TestModelScreen(SubScreen):
    def __init__(self, parent, app):
        super().__init__(parent, app, "TEST MODEL")
        self.selected = None
        self.selected_card = None

        models = _discover_models()

        # Packed with side="bottom" (bottom-most to topmost: log box, start
        # button, controls) so they always claim their own space first; only
        # list_frame (which has its own internal scrollbar) flexes/shrinks.
        log_box = self._make_log_box(self.body)
        log_box.pack(side="bottom", fill="both", expand=False)

        self.start_btn = _make_outline_button(self.body, "Start Test", AMBER, self._start, app.font_body, width=200, height=44)
        self.start_btn.configure(state="disabled")
        self.start_btn.pack(side="bottom", pady=(10, 16))

        controls = _make_content_column(self.body)
        self.mode_seg = _make_choice_row(controls, "Playback", ["Deterministic", "Stochastic"], "Deterministic", app.font_body)
        self.speed_var = _make_slider_row(controls, "Speed", 1, 10, 1, 3, app.font_body, value_fmt=_speed_label)
        controls.pack(side="bottom", pady=(12, 0))

        list_frame = ctk.CTkScrollableFrame(self.body, fg_color="transparent", corner_radius=RADIUS)
        list_frame.pack(side="top", fill="both", expand=True)

        if not models:
            ctk.CTkLabel(
                list_frame, text="No trained models found yet. Train one first.",
                font=app.font_body, text_color=TEXT_MUTED,
            ).pack(pady=20)
        for info in models:
            self._make_model_card(list_frame, info).pack(fill="x", pady=6)
        _enable_mousewheel(list_frame)

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

        ctk.CTkFrame(frame, fg_color="transparent", height=6).pack()

        # test_model() always loads the best checkpoint (see rl/playback.py) -- a
        # config that only has a last_model checkpoint can't be tested yet, so
        # don't make its card selectable (and say why, instead of it silently
        # doing nothing).
        if info["best_timesteps"] is None:
            ctk.CTkLabel(
                frame, text="(no best checkpoint yet -- cannot be tested)",
                font=self.app.font_small, text_color=RED,
            ).pack(anchor="w", padx=12, pady=(0, 8))
            return frame

        def select(_event=None):
            if self.selected_card is not None:
                self.selected_card.configure(border_color=BORDER)
            frame.configure(border_color=AMBER)
            self.selected_card = frame
            self.selected = info
            self.start_btn.configure(state="normal")

        _bind_recursive(frame, "<Button-1>", select)
        return frame

    def _start(self):
        if not self.selected:
            return
        info = self.selected
        self._start_background(
            test_model,
            dict(
                model_name=info["algo"], grid_width=info["grid_width"], grid_height=info["grid_height"],
                snake_fov_radius=info["fov"], use_cnn=info["use_cnn"], fps=int(self.speed_var.get()) * 10,
                deterministic=self.mode_seg.get() == "Deterministic",
            ),
            self.start_btn,
        )
