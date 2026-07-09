"""
ui.py - Dark-mode desktop launcher for the Snake RL project.

Lets you choose between playing yourself, watching a trained model play,
or training a new model, all through a graphical interface instead of
editing main.py's __main__ block by hand.

Run directly with `python main.py` (or `python ui.py`).
"""

import customtkinter as ctk

from snake_game import (
    COLOR_BACKGROUND,
    COLOR_SCORE_PANEL,
    COLOR_GRID_LINE,
    COLOR_SCORE_TEXT,
    COLOR_SNAKE_HEAD,
    COLOR_SNAKE_TAIL,
    COLOR_APPLE,
)


def _hex(color) -> str:
    return f"#{color.r:02x}{color.g:02x}{color.b:02x}"


def _darken(hex_color: str, factor: float = 0.85) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"#{int(r * factor):02x}{int(g * factor):02x}{int(b * factor):02x}"


# Color theme - reuses the same palette as the game itself (snake_game.py)
# so the launcher and the game look like one cohesive app.
BG = _hex(COLOR_BACKGROUND)
PANEL = _hex(COLOR_SCORE_PANEL)
BORDER = _hex(COLOR_GRID_LINE)
TEXT = _hex(COLOR_SCORE_TEXT)
TEXT_MUTED = "#9195a8"

GREEN = _hex(COLOR_SNAKE_HEAD)          # success / positive (e.g. "Play")
GREEN_HOVER = _darken(GREEN)
RED = _hex(COLOR_APPLE)                 # danger / stop / errors
RED_HOVER = _darken(RED)
# Not a named constant in snake_game.py, but matches the amber accent used
# for the FOV debug overlay (snake_game_environment.py), kept consistent here.
AMBER = "#ffd25a"
AMBER_HOVER = _darken(AMBER)


def _make_nav_card(parent, title, subtitle, accent, accent_hover, command, font_title, font_body):
    card = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=16, border_width=2, border_color=BORDER, width=220, height=220)
    card.pack_propagate(False)

    title_lbl = ctk.CTkLabel(card, text=title, font=font_title, text_color=accent)
    title_lbl.pack(pady=(46, 10))
    subtitle_lbl = ctk.CTkLabel(card, text=subtitle, font=font_body, text_color=TEXT_MUTED, wraplength=170, justify="center")
    subtitle_lbl.pack(pady=(0, 10))

    def on_enter(_event):
        card.configure(border_color=accent)

    def on_leave(_event):
        card.configure(border_color=BORDER)

    def on_click(_event):
        command()

    for widget in (card, title_lbl, subtitle_lbl):
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)
        widget.bind("<Button-1>", on_click)
        widget.configure(cursor="hand2")

    return card


class HomeScreen(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=BG)
        self.app = app

        ctk.CTkLabel(self, text="SNAKE RL — LAUNCHER", font=app.font_h1, text_color=TEXT).pack(pady=(60, 8))
        ctk.CTkLabel(self, text="Wähle einen Modus", font=app.font_body, text_color=TEXT_MUTED).pack(pady=(0, 10))

        cards = ctk.CTkFrame(self, fg_color="transparent")
        cards.pack(expand=True)

        specs = [
            ("PLAY", "Selbst spielen\n(WASD-Steuerung)", GREEN, GREEN_HOVER, "PlayScreen"),
            ("TEST MODEL", "Trainiertem Modell\nzuschauen", AMBER, AMBER_HOVER, "TestModelScreen"),
            ("TRAIN MODEL", "Neues Modell\ntrainieren", RED, RED_HOVER, "TrainModelScreen"),
        ]
        for col, (title, subtitle, accent, accent_hover, screen_name) in enumerate(specs):
            card = _make_nav_card(
                cards, title, subtitle, accent, accent_hover,
                command=lambda name=screen_name: app.show(name),
                font_title=app.font_card_title, font_body=app.font_small,
            )
            card.grid(row=0, column=col, padx=18)


class SubScreen(ctk.CTkFrame):
    """Base for every non-home screen: a back button + title header, plus a body frame for content."""

    def __init__(self, parent, app, title):
        super().__init__(parent, fg_color=BG)
        self.app = app

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=(20, 10))

        ctk.CTkButton(
            header, text="← Zurück", width=90, fg_color="transparent", hover_color=PANEL,
            text_color=TEXT_MUTED, border_width=1, border_color=BORDER,
            command=lambda: app.show("HomeScreen"),
        ).pack(side="left")

        ctk.CTkLabel(header, text=title, font=app.font_h2, text_color=TEXT).pack(side="left", padx=20)

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="both", expand=True, padx=24, pady=(0, 24))


class PlayScreen(SubScreen):
    def __init__(self, parent, app):
        super().__init__(parent, app, "PLAY YOURSELF")
        ctk.CTkLabel(self.body, text="Formular folgt im nächsten Schritt...", text_color=TEXT_MUTED, font=app.font_body).pack(expand=True)


class TestModelScreen(SubScreen):
    def __init__(self, parent, app):
        super().__init__(parent, app, "TEST MODEL")
        ctk.CTkLabel(self.body, text="Modell-Auswahl folgt im nächsten Schritt...", text_color=TEXT_MUTED, font=app.font_body).pack(expand=True)


class TrainModelScreen(SubScreen):
    def __init__(self, parent, app):
        super().__init__(parent, app, "TRAIN MODEL")
        ctk.CTkLabel(self.body, text="Trainings-Formular folgt im nächsten Schritt...", text_color=TEXT_MUTED, font=app.font_body).pack(expand=True)


class App(ctk.CTk):
    def __init__(self):
        ctk.set_appearance_mode("dark")
        super().__init__()

        self.title("Snake RL — Launcher")
        self.geometry("960x680")
        self.minsize(780, 560)
        self.configure(fg_color=BG)

        self.font_h1 = ctk.CTkFont(family="DejaVu Sans", size=28, weight="bold")
        self.font_h2 = ctk.CTkFont(family="DejaVu Sans", size=20, weight="bold")
        self.font_card_title = ctk.CTkFont(family="DejaVu Sans", size=22, weight="bold")
        self.font_body = ctk.CTkFont(family="DejaVu Sans", size=14)
        self.font_small = ctk.CTkFont(family="DejaVu Sans", size=12)
        self.font_mono = ctk.CTkFont(family="DejaVu Sans Mono", size=12)

        container = ctk.CTkFrame(self, fg_color=BG)
        container.pack(fill="both", expand=True)

        self.screens = {}
        for screen_cls in (HomeScreen, PlayScreen, TestModelScreen, TrainModelScreen):
            screen = screen_cls(container, self)
            screen.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.screens[screen_cls.__name__] = screen

        self.show("HomeScreen")

    def show(self, name):
        self.screens[name].tkraise()


if __name__ == "__main__":
    App().mainloop()
