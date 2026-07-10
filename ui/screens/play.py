"""
ui/screens/play.py - "Play yourself" screen.
"""

import customtkinter as ctk

from rl.playback import play_game
from rl.paths import GRID_PRESETS
from ui.theme import TEXT_MUTED, GREEN
from ui.widgets import _make_content_column, _make_choice_row, _make_slider_row, _make_outline_button, _speed_label
from ui.screens.base import SubScreen


class PlayScreen(SubScreen):
    def __init__(self, parent, app):
        super().__init__(parent, app, "PLAY YOURSELF")
        self._grid_by_label = {label: (w, h) for label, w, h in GRID_PRESETS}

        form = _make_content_column(self.body)
        form.pack(pady=(4, 16))

        # Same 3 curated presets as Train Model (rl.paths.GRID_PRESETS), not a
        # free width/height slider -- these are the only sizes any trained
        # model exists for, so a free size here would just be pointless choice.
        self.grid_seg = _make_choice_row(form, "Grid size", [p[0] for p in GRID_PRESETS], GRID_PRESETS[0][0], app.font_body)
        self.speed_var = _make_slider_row(form, "Speed", 1, 5, 1, 1, app.font_body, value_fmt=_speed_label)

        ctk.CTkLabel(
            self.body, text="Controls: WASD to move, ESC or close the window to quit",
            font=app.font_small, text_color=TEXT_MUTED,
        ).pack(pady=(0, 14))

        # Log box packed (bottom-most) before the button (which then stacks above
        # it) so the log always claims its own space from the window's bottom
        # edge first, instead of being the first thing clipped on a small window.
        log_box = self._make_log_box(self.body)
        log_box.pack(side="bottom", fill="both", expand=True)

        self.start_btn = _make_outline_button(self.body, "Start Game", GREEN, self._start, app.font_body, width=200, height=44)
        self.start_btn.pack(side="bottom", pady=(0, 16))

    def _start(self):
        grid_width, grid_height = self._grid_by_label[self.grid_seg.get()]
        self._start_background(
            play_game,
            dict(
                grid_width=grid_width,
                grid_height=grid_height,
                fps=int(self.speed_var.get()) * 10,
            ),
            self.start_btn,
        )
