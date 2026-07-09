"""
ui.py - Dark-mode desktop launcher for the Snake RL project.

Lets you choose between playing yourself, watching a trained model play,
or training a new model, all through a graphical interface instead of
editing main.py's __main__ block by hand.

Run directly with `python main.py` (or `python ui.py`).
"""

import os
import re
import sys
import json
import glob
import queue
import threading
import tkinter as tk
import customtkinter as ctk

from main import PPO_PATH, DQN_PATH, play_game, test_model, train_model
from DQN_hyper_tuning import PATH as DQN_TUNING_PATH
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


def _mix(hex_a: str, hex_b: str, t: float) -> str:
    """Blend two hex colors; t=0 -> hex_a, t=1 -> hex_b."""
    ar, ag, ab = (int(hex_a[i:i + 2], 16) for i in (1, 3, 5))
    br, bg, bb = (int(hex_b[i:i + 2], 16) for i in (1, 3, 5))
    return f"#{int(ar + (br - ar) * t):02x}{int(ag + (bg - ag) * t):02x}{int(ab + (bb - ab) * t):02x}"


def _spaced(text: str) -> str:
    """'TEST MODEL' -> 'T E S T   M O D E L' - a cheap letterspacing trick for button titles."""
    return " ".join(text)


# Color theme - reuses the same palette as the game itself (snake_game.py)
# so the launcher and the game look like one cohesive app.
BG = _hex(COLOR_BACKGROUND)
PANEL = _hex(COLOR_SCORE_PANEL)
BORDER = _hex(COLOR_GRID_LINE)
TEXT = _hex(COLOR_SCORE_TEXT)
TEXT_MUTED = "#9195a8"

GREEN = _hex(COLOR_SNAKE_HEAD)          # success / positive (e.g. "Play")
RED = _hex(COLOR_APPLE)                 # danger / stop / errors
# Not a named constant in snake_game.py, but matches the amber accent used
# for the FOV debug overlay (snake_game_environment.py), kept consistent here.
AMBER = "#ffd25a"


def _rounded_rect_points(x1, y1, x2, y2, r):
    return [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
        x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]


def _make_nav_button(parent, title, subtitle, accent, command, font_title, font_small):
    # customtkinter's own rounded-corner rendering is broken on this Tk build
    # (verified: every CTkButton/CTkFrame corner_radius > 0 draws a notched,
    # glitchy corner regardless of customtkinter version) -- so rounded corners
    # are instead drawn by hand on a plain tk.Canvas using create_polygon(),
    # which renders cleanly. This also sidesteps the hover/cursor quirks that
    # came with the earlier hand-rolled CTkFrame card.
    width, height, radius = 380, 58, 16
    wrapper = ctk.CTkFrame(parent, fg_color="transparent")

    canvas = tk.Canvas(wrapper, width=width, height=height, bg=BG, highlightthickness=0)
    canvas.pack()
    rect = canvas.create_polygon(
        _rounded_rect_points(1, 1, width - 1, height - 1, radius),
        smooth=True, fill=PANEL, outline=BORDER, width=2,
    )
    canvas.create_text(width / 2, height / 2, text=_spaced(title), fill=accent, font=font_title)

    hover_fill = _mix(PANEL, accent, 0.22)
    canvas.bind("<Enter>", lambda _e: canvas.itemconfig(rect, fill=hover_fill))
    canvas.bind("<Leave>", lambda _e: canvas.itemconfig(rect, fill=PANEL))
    canvas.bind("<Button-1>", lambda _e: command())

    ctk.CTkLabel(wrapper, text=subtitle, font=font_small, text_color=TEXT_MUTED).pack(pady=(8, 0))
    return wrapper


class _QueueWriter:
    """A file-like object that pushes written text onto a queue instead of a stream."""

    def __init__(self, q):
        self.q = q

    def write(self, text):
        if text:
            self.q.put(text)

    def flush(self):
        pass


def _bind_recursive(widget, event, handler):
    widget.bind(event, handler)
    for child in widget.winfo_children():
        _bind_recursive(child, event, handler)


def _make_slider_row(parent, label, from_, to, step, initial, font_body, value_fmt=None):
    """A labeled CTkSlider with a live value readout. Returns a tk.DoubleVar tracking the (step-snapped) value."""
    value_fmt = value_fmt or (lambda v: str(int(v)))
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", pady=8)

    top = ctk.CTkFrame(row, fg_color="transparent")
    top.pack(fill="x")
    ctk.CTkLabel(top, text=label, font=font_body, text_color=TEXT).pack(side="left")
    value_lbl = ctk.CTkLabel(top, text=value_fmt(initial), font=font_body, text_color=AMBER)
    value_lbl.pack(side="right")

    var = tk.DoubleVar(value=initial)
    steps = max(1, int(round((to - from_) / step)))

    def on_move(v):
        snapped = round((float(v) - from_) / step) * step + from_
        var.set(snapped)
        value_lbl.configure(text=value_fmt(snapped))

    slider = ctk.CTkSlider(
        row, from_=from_, to=to, number_of_steps=steps, variable=var, command=on_move,
        fg_color=PANEL, progress_color=AMBER, button_color=AMBER,
        button_hover_color=_mix(AMBER, "#ffffff", 0.2), corner_radius=0, button_corner_radius=0,
    )
    slider.pack(fill="x", pady=(6, 0))
    return var


def _make_segmented_row(parent, label, values, initial, font_body, command=None):
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", pady=8)
    ctk.CTkLabel(row, text=label, font=font_body, text_color=TEXT).pack(anchor="w")

    seg = ctk.CTkSegmentedButton(
        row, values=values, corner_radius=0, font=font_body,
        fg_color=PANEL, selected_color=AMBER, selected_hover_color=_mix(AMBER, "#000000", 0.2),
        unselected_color=PANEL, unselected_hover_color=BORDER,
        text_color=TEXT, text_color_disabled=TEXT_MUTED, command=command,
    )
    seg.set(initial)
    seg.pack(fill="x", pady=(4, 0))
    return seg


def _make_entry_row(parent, label, initial, font_body):
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", pady=8)
    ctk.CTkLabel(row, text=label, font=font_body, text_color=TEXT).pack(anchor="w")

    entry = ctk.CTkEntry(
        row, corner_radius=0, fg_color=PANEL, border_color=BORDER,
        text_color=TEXT, font=font_body,
    )
    entry.insert(0, str(initial))
    entry.pack(fill="x", pady=(4, 0))
    return entry


def _discover_models():
    """
    Scan Training/Saved Models/{PPO,DQN}/{FLAT,GRID}/GRID_*_*/FOV_RADIUS_*/ for
    checkpoints, skipping configs with neither a best_model_*.zip nor a
    last_model_*.zip (see main.py's _finalize_checkpoint/_find_checkpoint).
    """
    models = []
    for algo, base in (("PPO", PPO_PATH), ("DQN", DQN_PATH)):
        for obs_label in ("FLAT", "GRID"):
            root = os.path.join(base, obs_label)
            if not os.path.isdir(root):
                continue
            for grid_dir in sorted(os.listdir(root)):
                grid_match = re.match(r"GRID_(\d+)_(\d+)$", grid_dir)
                if not grid_match:
                    continue
                grid_path = os.path.join(root, grid_dir)
                for fov_dir in sorted(os.listdir(grid_path)):
                    fov_match = re.match(r"FOV_RADIUS_(\d+)$", fov_dir)
                    if not fov_match:
                        continue
                    model_path = os.path.join(grid_path, fov_dir)

                    def _timesteps(prefix):
                        matches = glob.glob(os.path.join(model_path, f"{prefix}_*.zip"))
                        if not matches:
                            return None
                        m = re.match(rf"{prefix}_(\d+)\.zip", os.path.basename(matches[0]))
                        return int(m.group(1)) if m else None

                    best_timesteps = _timesteps("best_model")
                    last_timesteps = _timesteps("last_model")
                    if best_timesteps is None and last_timesteps is None:
                        continue

                    evaluation = None
                    eval_path = os.path.join(model_path, "evaluation.json")
                    if os.path.exists(eval_path):
                        with open(eval_path) as file:
                            evaluation = json.load(file)

                    models.append({
                        "algo": algo,
                        "obs_mode": obs_label,
                        "use_cnn": obs_label == "GRID",
                        "grid_width": int(grid_match.group(1)),
                        "grid_height": int(grid_match.group(2)),
                        "fov": int(fov_match.group(1)),
                        "path": model_path,
                        "best_timesteps": best_timesteps,
                        "last_timesteps": last_timesteps,
                        "evaluation": evaluation,
                    })
    return models


class HomeScreen(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=BG)
        self.app = app

        ctk.CTkLabel(self, text="SNAKE RL — LAUNCHER", font=app.font_h1, text_color=TEXT).pack(pady=(60, 14))
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
            item = _make_nav_button(
                nav, title, subtitle, accent,
                command=lambda name=screen_name: app.show(name),
                font_title=app.font_card_title, font_small=app.font_small,
            )
            item.pack(pady=12)


class SubScreen(ctk.CTkFrame):
    """Base for every non-home screen: a back button + title header, plus a body frame for content."""

    def __init__(self, parent, app, title):
        super().__init__(parent, fg_color=BG)
        self.app = app
        self.log_box = None
        self._log_queue = queue.Queue()

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=(20, 10))

        ctk.CTkButton(
            header, text="← Back", width=90, fg_color="transparent", hover_color=PANEL,
            text_color=TEXT_MUTED, border_width=1, border_color=BORDER, corner_radius=0,
            command=lambda: app.show("HomeScreen"),
        ).pack(side="left")

        ctk.CTkLabel(header, text=title, font=app.font_h2, text_color=TEXT).pack(side="left", padx=20)

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="both", expand=True, padx=24, pady=(0, 24))

    def _make_log_box(self, parent):
        self.log_box = ctk.CTkTextbox(
            parent, font=self.app.font_mono, fg_color=PANEL, text_color=TEXT_MUTED,
            corner_radius=0, height=140, state="disabled", wrap="word",
        )
        return self.log_box

    def _start_background(self, func, kwargs, start_button):
        """Run func(**kwargs) in a daemon thread, streaming its stdout into self.log_box (if any)."""
        if self.app.busy:
            return
        self.app.busy = True
        start_button.configure(state="disabled")
        if self.log_box is not None:
            self.log_box.configure(state="normal")
            self.log_box.delete("1.0", "end")
            self.log_box.configure(state="disabled")

        def worker():
            old_stdout = sys.stdout
            sys.stdout = _QueueWriter(self._log_queue)
            try:
                func(**kwargs)
            except Exception as exc:
                self._log_queue.put(f"Error: {exc}\n")
            finally:
                sys.stdout = old_stdout
                self._log_queue.put(None)

        threading.Thread(target=worker, daemon=True).start()
        self._poll_log(start_button)

    def _poll_log(self, start_button):
        finished = False
        try:
            while True:
                item = self._log_queue.get_nowait()
                if item is None:
                    finished = True
                    break
                if self.log_box is not None:
                    self.log_box.configure(state="normal")
                    self.log_box.insert("end", item)
                    self.log_box.see("end")
                    self.log_box.configure(state="disabled")
        except queue.Empty:
            pass

        if finished:
            self.app.busy = False
            start_button.configure(state="normal")
        else:
            self.after(100, lambda: self._poll_log(start_button))


def _speed_label(v):
    return f"{int(v)}  ({int(v) * 10} FPS)"


class PlayScreen(SubScreen):
    def __init__(self, parent, app):
        super().__init__(parent, app, "PLAY YOURSELF")

        form = ctk.CTkFrame(self.body, fg_color="transparent")
        form.pack(fill="x", pady=(4, 16))

        self.width_var = _make_slider_row(form, "Grid width", 10, 80, 5, 30, app.font_body)
        self.height_var = _make_slider_row(form, "Grid height", 10, 60, 5, 20, app.font_body)
        self.speed_var = _make_slider_row(form, "Speed", 1, 5, 1, 1, app.font_body, value_fmt=_speed_label)

        ctk.CTkLabel(
            self.body, text="Controls: WASD to move, ESC or close the window to quit",
            font=app.font_small, text_color=TEXT_MUTED,
        ).pack(pady=(0, 14))

        self.start_btn = ctk.CTkButton(
            self.body, text="Start Game", font=app.font_body,
            fg_color=GREEN, hover_color=_mix(GREEN, "#000000", 0.25), text_color=PANEL,
            corner_radius=0, height=44, command=self._start,
        )
        self.start_btn.pack(pady=(0, 16))

        self._make_log_box(self.body).pack(fill="both", expand=True)

    def _start(self):
        self._start_background(
            play_game,
            dict(
                grid_width=int(self.width_var.get()),
                grid_height=int(self.height_var.get()),
                fps=int(self.speed_var.get()) * 10,
            ),
            self.start_btn,
        )


class TestModelScreen(SubScreen):
    def __init__(self, parent, app):
        super().__init__(parent, app, "TEST MODEL")
        self.selected = None
        self.selected_card = None

        models = _discover_models()

        list_frame = ctk.CTkScrollableFrame(self.body, fg_color="transparent", corner_radius=0)
        list_frame.pack(fill="both", expand=True)

        if not models:
            ctk.CTkLabel(
                list_frame, text="No trained models found yet. Train one first.",
                font=app.font_body, text_color=TEXT_MUTED,
            ).pack(pady=20)
        for info in models:
            self._make_model_card(list_frame, info).pack(fill="x", pady=6)

        controls = ctk.CTkFrame(self.body, fg_color="transparent")
        controls.pack(fill="x", pady=(12, 0))
        self.speed_var = _make_slider_row(controls, "Speed", 1, 5, 1, 3, app.font_body, value_fmt=_speed_label)

        self.start_btn = ctk.CTkButton(
            self.body, text="Start Test", font=app.font_body,
            fg_color=AMBER, hover_color=_mix(AMBER, "#000000", 0.25), text_color=PANEL,
            corner_radius=0, height=44, state="disabled", command=self._start,
        )
        self.start_btn.pack(pady=(10, 16))

        self._make_log_box(self.body).pack(fill="both", expand=False)

    def _make_model_card(self, parent, info):
        frame = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=0, border_width=2, border_color=BORDER)

        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 2))
        badge = f"[{info['algo']}] [{info['obs_mode']}]  Grid {info['grid_width']}x{info['grid_height']}  FOV {info['fov']}"
        ctk.CTkLabel(header, text=badge, font=self.app.font_body, text_color=TEXT).pack(side="left")

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

        # test_model() always loads the best checkpoint (see main.py) -- a config
        # that only has a last_model checkpoint can't be tested yet, so don't make
        # its card selectable (and say why, instead of it silently doing nothing).
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
            ),
            self.start_btn,
        )


class TrainModelScreen(SubScreen):
    # Grid size is restricted to 3 curated presets (rather than free width/height)
    # since it determines the save-folder structure (GRID_{w}_{h}); all three keep
    # the same 3:2 aspect ratio as the existing default, just scaled 1x/1.5x/2x.
    GRID_PRESETS = [("Small (30x20)", 30, 20), ("Medium (45x30)", 45, 30), ("Large (60x40)", 60, 40)]

    def __init__(self, parent, app):
        super().__init__(parent, app, "TRAIN MODEL")
        self._grid_by_label = {label: (w, h) for label, w, h in self.GRID_PRESETS}
        tuned_params_path = os.path.join(DQN_TUNING_PATH, "best_dqn_params.json")
        self._tuned_params_available = os.path.exists(tuned_params_path)

        form = ctk.CTkScrollableFrame(self.body, fg_color="transparent", corner_radius=0)
        form.pack(fill="both", expand=True)

        self.algo_seg = _make_segmented_row(form, "Algorithm", ["DQN", "PPO"], "DQN", app.font_body, command=self._on_algo_change)
        self.obs_seg = _make_segmented_row(form, "Observation mode", ["FLAT", "GRID"], "FLAT", app.font_body)
        self.grid_seg = _make_segmented_row(form, "Grid size", [p[0] for p in self.GRID_PRESETS], self.GRID_PRESETS[0][0], app.font_body)
        self.fov_var = _make_slider_row(form, "FOV radius", 1, 8, 1, 3, app.font_body)
        self.timesteps_entry = _make_entry_row(form, "Timesteps", 3_000_000, app.font_body)

        cpu_count = os.cpu_count() or 4
        self.num_envs_var = _make_slider_row(form, "Parallel environments", 1, cpu_count, 1, min(4, cpu_count), app.font_body)

        self.mode_seg = _make_segmented_row(form, "Mode", ["Start Fresh", "Continue"], "Start Fresh", app.font_body, command=self._on_mode_change)
        self.checkpoint_seg = _make_segmented_row(form, "Resume from", ["Best", "Last"], "Best", app.font_body)
        self.checkpoint_seg.configure(state="disabled")

        self.tuned_var = tk.BooleanVar(value=False)
        self.tuned_checkbox = ctk.CTkCheckBox(
            form, text="Use tuned DQN hyperparameters (from DQN_hyper_tuning.py)", variable=self.tuned_var,
            font=app.font_small, text_color=TEXT, corner_radius=0, checkbox_width=18, checkbox_height=18,
            fg_color=AMBER, hover_color=_mix(AMBER, "#000000", 0.2), border_color=BORDER,
        )
        self.tuned_checkbox.pack(anchor="w", pady=(8, 4))
        self._on_algo_change("DQN")

        self.start_btn = ctk.CTkButton(
            self.body, text="Start Training", font=app.font_body,
            fg_color=RED, hover_color=_mix(RED, "#000000", 0.25), text_color=PANEL,
            corner_radius=0, height=44, command=self._start,
        )
        self.start_btn.pack(pady=(12, 16))

        self._make_log_box(self.body).pack(fill="both", expand=True)

    def _on_algo_change(self, algo):
        if algo == "DQN" and self._tuned_params_available:
            self.tuned_checkbox.configure(state="normal")
        else:
            self.tuned_var.set(False)
            self.tuned_checkbox.configure(state="disabled")

    def _on_mode_change(self, mode):
        self.checkpoint_seg.configure(state="normal" if mode == "Continue" else "disabled")

    def _show_error(self, message):
        if self.log_box is not None:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", f"Error: {message}\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")

    def _start(self):
        try:
            timesteps = int(self.timesteps_entry.get())
            if timesteps <= 0:
                raise ValueError
        except ValueError:
            self._show_error("Timesteps must be a positive integer.")
            return

        grid_width, grid_height = self._grid_by_label[self.grid_seg.get()]
        algo = self.algo_seg.get()

        self._start_background(
            train_model,
            dict(
                model_name=algo,
                grid_width=grid_width, grid_height=grid_height,
                snake_fov_radius=int(self.fov_var.get()),
                timesteps=timesteps,
                num_envs=int(self.num_envs_var.get()),
                new=self.mode_seg.get() == "Start Fresh",
                best=self.checkpoint_seg.get() == "Best",
                use_tuned_params=self.tuned_var.get(),
                use_cnn=self.obs_seg.get() == "GRID",
            ),
            self.start_btn,
        )


class App(ctk.CTk):
    def __init__(self):
        ctk.set_appearance_mode("dark")
        super().__init__()

        self.title("Snake RL — Launcher")
        self.geometry("960x680")
        self.minsize(780, 560)
        self.configure(fg_color=BG)

        # Only one background action (play/test/train) may run at a time.
        self.busy = False

        # Monospace throughout: this Python's Tk build has no FreeType/Xft linkage
        # at all (verified via ldd), so it can't antialias *any* TTF outline font --
        # every family looks equally blocky. A monospace "terminal" font at least
        # reads as an intentional retro-arcade look instead of a broken sans-serif.
        self.font_h1 = ctk.CTkFont(family="DejaVu Sans Mono", size=26, weight="bold")
        self.font_h2 = ctk.CTkFont(family="DejaVu Sans Mono", size=18, weight="bold")
        self.font_card_title = ctk.CTkFont(family="DejaVu Sans Mono", size=18, weight="bold")
        self.font_body = ctk.CTkFont(family="DejaVu Sans Mono", size=13)
        self.font_small = ctk.CTkFont(family="DejaVu Sans Mono", size=12)
        self.font_mono = self.font_body

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
