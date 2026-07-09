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

# customtkinter's default rounded-corner renderer ("font_shapes", the non-macOS
# default) draws corners using a special embedded shapes-font -- this Python's Tk
# build has no FreeType/Xft linkage at all, so that font can't be rasterized and
# every corner_radius > 0 renders as a glitchy notch. "polygon_shapes" (macOS's
# default) draws corners with plain canvas polygons instead, which works fine here.
from customtkinter.windows.widgets.core_rendering.draw_engine import DrawEngine
DrawEngine.preferred_drawing_method = "polygon_shapes"

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

# Corner radius shared by every rounded widget (buttons, sliders, cards, entries, ...).
RADIUS = 10


def _make_outline_button(parent, text, accent, command, font, width=220, height=48, corner_radius=None):
    """
    A button that stays dark (PANEL) at rest and highlights its border in the
    given accent color on hover, instead of filling solid -- used everywhere
    (home screen nav, and every screen's start/cancel action button) so the
    whole app shares one consistent button language.
    """
    btn = ctk.CTkButton(
        parent, text=text, font=font, command=command,
        fg_color=PANEL, hover_color=PANEL, text_color=accent,
        border_width=2, border_color=BORDER, corner_radius=RADIUS if corner_radius is None else corner_radius,
        width=width, height=height,
    )
    btn.bind("<Enter>", lambda _e: btn.configure(border_color=accent), add="+")
    btn.bind("<Leave>", lambda _e: btn.configure(border_color=BORDER), add="+")
    return btn


def _make_nav_item(parent, title, subtitle, accent, command, font_title, font_small):
    wrapper = ctk.CTkFrame(parent, fg_color="transparent")
    btn = _make_outline_button(wrapper, _spaced(title), accent, command, font_title, width=380, height=58, corner_radius=16)
    btn.pack()
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
    # Some composite customtkinter widgets (e.g. CTkSegmentedButton) manage their
    # own internal sub-buttons and raise NotImplementedError from .bind() itself --
    # skip binding on those specifically, but still recurse into their children.
    try:
        widget.bind(event, handler)
    except NotImplementedError:
        pass
    for child in widget.winfo_children():
        _bind_recursive(child, event, handler)


def _enable_mousewheel(scrollable_frame):
    """
    customtkinter's own CTkScrollableFrame wheel handling refuses to scroll
    whenever the pointer is over a CTkSlider/CTkTextbox/CTkScrollbar (see
    CTkScrollableFrame._check_if_valid_scroll) -- given how much of these
    screens *is* sliders/a log textbox, that reads as "the wheel doesn't
    scroll" in practice. Bind directly on every current descendant instead,
    scrolling the frame's own canvas regardless of what's under the cursor.

    Binds both the legacy X11 wheel protocol (Button-4/5, discrete "clicks")
    and <MouseWheel> (event.delta, positive/negative) -- some libinput/XInput2
    setups deliver one but not the other, so covering both is what actually
    makes this reliable across different mice/touchpads/compositors.
    """
    canvas = scrollable_frame._parent_canvas

    def _on_wheel(event):
        if event.num == 4:
            canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            canvas.yview_scroll(1, "units")
        elif getattr(event, "delta", 0):
            canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    _bind_recursive(scrollable_frame, "<Button-4>", _on_wheel)
    _bind_recursive(scrollable_frame, "<Button-5>", _on_wheel)
    _bind_recursive(scrollable_frame, "<MouseWheel>", _on_wheel)


def _make_slider_row(parent, label, from_, to, step, initial, font_body, value_fmt=None, on_change=None):
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
        if on_change:
            on_change(snapped)

    slider = ctk.CTkSlider(
        row, from_=from_, to=to, number_of_steps=steps, variable=var, command=on_move,
        fg_color=PANEL, progress_color=AMBER, button_color=AMBER,
        button_hover_color=_mix(AMBER, "#ffffff", 0.2),
        height=20, corner_radius=10, button_corner_radius=10, button_length=18,
    )
    slider.pack(fill="x", pady=(8, 0))
    return var


def _make_segmented_row(parent, label, values, initial, font_body, command=None):
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", pady=8)
    ctk.CTkLabel(row, text=label, font=font_body, text_color=TEXT).pack(anchor="w")

    seg = ctk.CTkSegmentedButton(
        row, values=values, corner_radius=RADIUS, font=font_body,
        fg_color=PANEL, selected_color=AMBER, selected_hover_color=_mix(AMBER, "#000000", 0.2),
        unselected_color=PANEL, unselected_hover_color=BORDER,
        text_color=TEXT, text_color_disabled=TEXT_MUTED, command=command,
    )
    seg.set(initial)
    seg.pack(fill="x", pady=(4, 0))
    return seg


class _Choice:
    """A .get()/.set()/.configure(state=...) handle for _make_choice_row(), mirroring
    just enough of CTkSegmentedButton's interface to be a drop-in replacement."""

    def __init__(self, select, buttons):
        self._value = None
        self._select = select
        self._buttons = buttons

    def get(self):
        return self._value

    def set(self, value):
        self._select(value)

    def configure(self, **kwargs):
        if "state" in kwargs:
            for btn in self._buttons.values():
                btn.configure(state=kwargs["state"])


def _make_choice_row(parent, label, values, initial, font_body, command=None):
    """
    Like _make_segmented_row(), but built from individual outline buttons instead
    of CTkSegmentedButton -- worked around a customtkinter rendering bug where a
    segmented button with 3+ long-text options went completely blank (no visible
    text on any segment) on this Tk build, even though the widget's internal state
    was otherwise correct; 2-option segmented buttons elsewhere are unaffected.
    """
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", pady=8)
    ctk.CTkLabel(row, text=label, font=font_body, text_color=TEXT).pack(anchor="w")

    btn_row = ctk.CTkFrame(row, fg_color="transparent")
    btn_row.pack(fill="x", pady=(4, 0))

    buttons = {}

    def apply_visual(value):
        choice._value = value
        for v, btn in buttons.items():
            if v == value:
                btn.configure(fg_color=AMBER, text_color=PANEL, border_color=AMBER)
            else:
                btn.configure(fg_color=PANEL, text_color=TEXT, border_color=BORDER)

    def select(value):
        # Only user clicks notify `command` -- matches CTkSegmentedButton, whose
        # .set() (used for the initial value and programmatic changes) does not.
        apply_visual(value)
        if command:
            command(value)

    for i, value in enumerate(values):
        btn = _make_outline_button(btn_row, value, AMBER, lambda v=value: select(v), font_body, width=200, height=36)
        btn.pack(side="left", expand=True, fill="x", padx=(0 if i == 0 else 4, 0))
        buttons[value] = btn

    choice = _Choice(apply_visual, buttons)
    apply_visual(initial)
    return choice


def _make_entry_row(parent, label, initial, font_body):
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", pady=8)
    ctk.CTkLabel(row, text=label, font=font_body, text_color=TEXT).pack(anchor="w")

    entry = ctk.CTkEntry(
        row, corner_radius=RADIUS, fg_color=PANEL, border_color=BORDER,
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
            item = _make_nav_item(
                nav, title, subtitle, accent,
                command=lambda name=screen_name: app.show(name),
                font_title=app.font_card_title, font_small=app.font_small,
            )
            item.pack(pady=12)


def show_confirm_dialog(app, title, message, options):
    """
    A small modal dialog: `options` is a list of (label, accent_color, callback)
    tuples rendered as one outline button each, left to right. Clicking a button
    closes the dialog, then calls its callback (if any).
    """
    dialog = ctk.CTkToplevel(app)
    dialog.title(title)
    dialog.configure(fg_color=BG)
    dialog.resizable(False, False)
    dialog.transient(app)
    dialog.grab_set()

    ctk.CTkLabel(
        dialog, text=message, font=app.font_body, text_color=TEXT,
        wraplength=360, justify="center",
    ).pack(padx=24, pady=(24, 16))

    btn_row = ctk.CTkFrame(dialog, fg_color="transparent")
    btn_row.pack(padx=24, pady=(0, 24))

    for label, accent, callback in options:
        def on_click(cb=callback):
            dialog.destroy()
            if cb:
                cb()
        _make_outline_button(btn_row, label, accent, on_click, app.font_small, width=170, height=40).pack(side="left", padx=6)

    dialog.update_idletasks()
    x = app.winfo_rootx() + (app.winfo_width() - dialog.winfo_width()) // 2
    y = app.winfo_rooty() + (app.winfo_height() - dialog.winfo_height()) // 2
    dialog.geometry(f"+{x}+{y}")
    return dialog


class SubScreen(ctk.CTkFrame):
    """Base for every non-home screen: a back button + title header, plus a body frame for content."""

    def __init__(self, parent, app, title):
        super().__init__(parent, fg_color=BG)
        self.app = app
        self.log_box = None
        self._log_queue = queue.Queue()

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=(20, 10))

        _make_outline_button(
            header, "← Back", TEXT_MUTED, lambda: self._handle_back(), app.font_body, width=90, height=36,
        ).pack(side="left")

        ctk.CTkLabel(header, text=title, font=app.font_h2, text_color=TEXT).pack(side="left", padx=20)

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="both", expand=True, padx=24, pady=(0, 24))

    def _handle_back(self):
        """Overridden by screens that need to guard navigating away (e.g. Train Model mid-run)."""
        self.app.show("HomeScreen")

    def _make_log_box(self, parent):
        self.log_box = ctk.CTkTextbox(
            parent, font=self.app.font_mono, fg_color=PANEL, text_color=TEXT_MUTED,
            corner_radius=RADIUS, height=140, state="disabled", wrap="word",
        )
        return self.log_box

    def _start_background(self, func, kwargs, start_button, on_finish=None):
        """Run func(**kwargs) in a daemon thread, streaming its stdout into self.log_box (if any)."""
        if self.app.busy:
            return
        self.app.busy = True
        if on_finish is None:
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
        self._poll_log(start_button, on_finish)

    def _poll_log(self, start_button, on_finish=None):
        finished = False
        log_box = self.log_box
        # Only auto-follow new lines if the user was already at the bottom --
        # DQN/PPO log heavily during training, so unconditionally forcing the
        # view to "end" on every line made it impossible to scroll up and read
        # anything (it kept getting yanked back down before you could).
        follow = log_box is None or log_box.yview()[1] >= 0.999
        try:
            while True:
                item = self._log_queue.get_nowait()
                if item is None:
                    finished = True
                    break
                if log_box is not None:
                    log_box.configure(state="normal")
                    log_box.insert("end", item)
                    log_box.configure(state="disabled")
        except queue.Empty:
            pass

        if follow and log_box is not None:
            log_box.see("end")

        if finished:
            self.app.busy = False
            if on_finish:
                on_finish()
            else:
                start_button.configure(state="normal")
        else:
            self.after(100, lambda: self._poll_log(start_button, on_finish))


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

        self.start_btn = _make_outline_button(self.body, "Start Game", GREEN, self._start, app.font_body, width=200, height=44)
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

        list_frame = ctk.CTkScrollableFrame(self.body, fg_color="transparent", corner_radius=RADIUS)
        list_frame.pack(fill="both", expand=True)

        if not models:
            ctk.CTkLabel(
                list_frame, text="No trained models found yet. Train one first.",
                font=app.font_body, text_color=TEXT_MUTED,
            ).pack(pady=20)
        for info in models:
            self._make_model_card(list_frame, info).pack(fill="x", pady=6)
        _enable_mousewheel(list_frame)

        controls = ctk.CTkFrame(self.body, fg_color="transparent")
        controls.pack(fill="x", pady=(12, 0))
        self.mode_seg = _make_segmented_row(controls, "Playback", ["Deterministic", "Stochastic"], "Deterministic", app.font_body)
        self.speed_var = _make_slider_row(controls, "Speed", 1, 5, 1, 3, app.font_body, value_fmt=_speed_label)

        self.start_btn = _make_outline_button(self.body, "Start Test", AMBER, self._start, app.font_body, width=200, height=44)
        self.start_btn.configure(state="disabled")
        self.start_btn.pack(pady=(10, 16))

        self._make_log_box(self.body).pack(fill="both", expand=False)

    def _make_model_card(self, parent, info):
        frame = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=RADIUS, border_width=2, border_color=BORDER)

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
                deterministic=self.mode_seg.get() == "Deterministic",
            ),
            self.start_btn,
        )


class TrainModelScreen(SubScreen):
    # Grid size is restricted to 3 curated presets (rather than free width/height)
    # since it determines the save-folder structure (GRID_{w}_{h}); all three keep
    # the same 3:2 aspect ratio as the existing default, just scaled 1x/1.5x/2x.
    GRID_PRESETS = [("Small (30x20)", 30, 20), ("Medium (45x30)", 45, 30), ("Large (60x40)", 60, 40)]
    MIN_TIMESTEPS = 1000

    def __init__(self, parent, app):
        super().__init__(parent, app, "TRAIN MODEL")
        self._grid_by_label = {label: (w, h) for label, w, h in self.GRID_PRESETS}
        tuned_params_path = os.path.join(DQN_TUNING_PATH, "best_dqn_params.json")
        self._tuned_params_available = os.path.exists(tuned_params_path)

        self._continue_selected = None
        self._continue_selected_card = None
        self._collision_match = None
        self._is_training = False
        self._cancel_event = None
        self._discard_event = None

        self.train_mode_seg = _make_segmented_row(
            self.body, "Train", ["New Model", "Continue Existing"], "New Model", app.font_body,
            command=self._on_train_mode_change,
        )

        self.scroll = ctk.CTkScrollableFrame(self.body, fg_color="transparent", corner_radius=RADIUS, height=170)
        self.scroll.pack(fill="both", expand=True, pady=(8, 0))

        self.new_panel = ctk.CTkFrame(self.scroll, fg_color="transparent")
        self.continue_panel = ctk.CTkFrame(self.scroll, fg_color="transparent")
        self._build_new_panel(self.new_panel)
        self._build_continue_panel(self.continue_panel)
        self.new_panel.pack(fill="both", expand=True)

        shared = ctk.CTkFrame(self.body, fg_color="transparent")
        shared.pack(fill="x", pady=(8, 0))
        self.timesteps_entry = _make_entry_row(shared, "Timesteps", 3_000_000, app.font_body)
        cpu_count = os.cpu_count() or 4
        self.num_envs_var = _make_slider_row(shared, "Parallel environments", 1, cpu_count, 1, min(4, cpu_count), app.font_body)
        self.tuned_var = tk.BooleanVar(value=False)
        self.tuned_checkbox = ctk.CTkCheckBox(
            shared, text="Use tuned DQN hyperparameters (from DQN_hyper_tuning.py)", variable=self.tuned_var,
            font=app.font_small, text_color=TEXT, checkbox_width=18, checkbox_height=18,
            fg_color=AMBER, hover_color=_mix(AMBER, "#000000", 0.2), border_color=BORDER,
        )
        self.tuned_checkbox.pack(anchor="w", pady=(8, 4))

        self.start_btn = _make_outline_button(self.body, "Start Training", RED, self._start, app.font_body, width=220, height=44)
        self.start_btn.pack(pady=(12, 16))

        # expand=False (fixed height): only self.scroll above should flex with the
        # window -- it already has its own scrollbar, so shrinking it just means
        # scrolling the form, whereas the log has nowhere else to go and used to
        # get squeezed to near-nothing unless the window was maximized.
        self._make_log_box(self.body).pack(fill="both", expand=False)

        self._on_algo_change("DQN")
        self._check_collision()
        self._update_start_enabled()
        _enable_mousewheel(self.scroll)

    # --- "New Model" panel ---------------------------------------------------

    def _build_new_panel(self, parent):
        self.algo_seg = _make_segmented_row(parent, "Algorithm", ["DQN", "PPO"], "DQN", self.app.font_body, command=self._on_new_config_change)
        self.obs_seg = _make_segmented_row(parent, "Observation mode", ["FLAT", "GRID"], "FLAT", self.app.font_body, command=self._on_new_config_change)
        self.grid_seg = _make_choice_row(parent, "Grid size", [p[0] for p in self.GRID_PRESETS], self.GRID_PRESETS[0][0], self.app.font_body, command=self._on_new_config_change)
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
        match = self._collision_match
        self.train_mode_seg.set("Continue Existing")
        self._on_train_mode_change("Continue Existing")
        if match is not None:
            card = self._continue_cards_by_path.get(match["path"])
            if card is not None:
                self._select_continue_card(match, card)

    # --- "Continue Existing" panel -------------------------------------------

    def _build_continue_panel(self, parent):
        self.continue_cards_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.continue_cards_frame.pack(fill="both", expand=True)
        self._continue_cards_by_path = {}
        self._refresh_continue_models()

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(8, 0))
        self.continue_checkpoint_seg = _make_segmented_row(row, "Resume from", ["Best", "Last"], "Best", self.app.font_body)
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
        _enable_mousewheel(self.scroll)
        self._update_start_enabled()

    def _make_continue_card(self, parent, info):
        frame = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=RADIUS, border_width=2, border_color=BORDER)

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
            self.new_panel.pack(fill="both", expand=True)
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

        kwargs.update(
            timesteps=timesteps,
            num_envs=int(self.num_envs_var.get()),
            use_tuned_params=self.tuned_var.get(),
        )

        self._cancel_event = threading.Event()
        self._discard_event = threading.Event()
        kwargs["cancel_event"] = self._cancel_event
        kwargs["discard_event"] = self._discard_event

        self._is_training = True
        self.start_btn.configure(text="Cancel Training", command=self._request_cancel)
        self._start_background(train_model, kwargs, self.start_btn, on_finish=self._on_training_finished)

    def _on_training_finished(self):
        self._is_training = False
        self.start_btn.configure(text="Start Training", command=self._start)
        self._refresh_continue_models()
        if self.train_mode_seg.get() == "New Model":
            self._check_collision()
        self._update_start_enabled()

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

    def _do_cancel_and_discard(self):
        if self._cancel_event is not None:
            self._discard_event.set()
            self._cancel_event.set()

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


class App(ctk.CTk):
    def __init__(self):
        ctk.set_appearance_mode("dark")
        super().__init__()

        self.title("Snake RL — Launcher")
        # Tall enough that Train Model's log panel is visible by default without
        # maximizing: CTkScrollableFrame pins its own canvas to a fixed height
        # (its `height=` constructor arg) regardless of pack(expand=True), so the
        # form area doesn't actually shrink to make room -- the window has to be
        # tall enough up front instead.
        self.geometry("960x800")
        self.minsize(780, 600)
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
        self.protocol("WM_DELETE_WINDOW", self._on_close_request)

    def show(self, name):
        self.screens[name].tkraise()

    def _on_close_request(self):
        # Closing the window kills every background daemon thread outright (no
        # cleanup) -- warn first instead of silently losing a training run,
        # same risk as the Train Model Back button, just one level up.
        train_screen = self.screens["TrainModelScreen"]
        if train_screen._is_training:
            def _cancel_and_quit(discard):
                if discard:
                    train_screen._do_cancel_and_discard()
                else:
                    train_screen._do_cancel_and_save()
                self.destroy()

            show_confirm_dialog(
                self, "Quit Snake RL Launcher?", "Training is still running. Do you want to cancel it before quitting?",
                [
                    ("Keep Training", AMBER, None),
                    ("Save & Quit", GREEN, lambda: _cancel_and_quit(False)),
                    ("Discard & Quit", RED, lambda: _cancel_and_quit(True)),
                ],
            )
        elif self.busy:
            show_confirm_dialog(
                self, "Quit Snake RL Launcher?", "A play/test session is still running. Quitting now will close it immediately.",
                [
                    ("Stay", GREEN, None),
                    ("Quit Anyway", RED, self.destroy),
                ],
            )
        else:
            self.destroy()


if __name__ == "__main__":
    App().mainloop()
