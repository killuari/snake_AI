"""
ui/screens/home.py - Landing screen with the three main navigation tiles.
"""

import customtkinter as ctk

from ui.theme import BG, TEXT, TEXT_MUTED, AMBER, GREEN, RED, BLUE
from ui.widgets import _make_nav_item


class HomeScreen(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=BG)
        self.app = app

        ctk.CTkLabel(self, text="SNAKE REINFORCEMENT LEARNING LAUNCHER", font=app.font_h1, text_color=TEXT).pack(pady=(60, 14))
        ctk.CTkFrame(self, fg_color=AMBER, width=120, height=2, corner_radius=0).pack(pady=(0, 14))
        ctk.CTkLabel(self, text="Choose a mode", font=app.font_body, text_color=TEXT_MUTED).pack(pady=(0, 10))

        nav = ctk.CTkFrame(self, fg_color="transparent")
        nav.pack(expand=True)

        # EXIT reuses RED (now free -- Train Model moved to BLUE below) since
        # quitting is the one nav item that isn't a "mode": going through
        # App._on_close_request() (not a bare self.app.destroy()) keeps the
        # existing busy/training confirmation guard intact.
        specs = [
            ("PLAY", "Play yourself (WASD controls)", GREEN, lambda: app.show("PlayScreen")),
            ("TEST MODEL", "Watch a trained model play", AMBER, lambda: app.show("TestModelScreen")),
            ("TRAIN MODEL", "Train a new model", BLUE, lambda: app.show("TrainModelScreen")),
            ("EXIT", "Quit the launcher", RED, app._on_close_request),
        ]
        for title, subtitle, accent, command in specs:
            item = _make_nav_item(
                nav, title, subtitle, accent,
                command=command,
                font_title=app.font_card_title, font_small=app.font_small,
            )
            item.pack(pady=12)
