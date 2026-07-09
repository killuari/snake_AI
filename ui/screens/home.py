"""
ui/screens/home.py - Landing screen with the three main navigation tiles.
"""

import customtkinter as ctk

from ui.theme import BG, TEXT, TEXT_MUTED, AMBER, GREEN, RED
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

        specs = [
            ("PLAY", "Play yourself (WASD controls)", GREEN, "PlayScreen"),
            ("TEST MODEL", "Watch a trained model play", AMBER, "TestModelScreen"),
            ("TRAIN MODEL", "Train a new model", RED, "TrainModelScreen"),
        ]
        for title, subtitle, accent, screen_name in specs:
            item = _make_nav_item(
                nav, title, subtitle, accent,
                command=lambda name=screen_name: app.show(name),
                font_title=app.font_card_title, font_small=app.font_small,
            )
            item.pack(pady=12)
