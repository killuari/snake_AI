"""
ui/app.py - App: the root window, screen registry, and navigation.
"""

import tkinter.font as tkfont
import customtkinter as ctk

from ui.theme import BG, MIN_WIDTH, AMBER, GREEN, RED
from ui.widgets import show_confirm_dialog
from ui.screens.base import SubScreen
from ui.screens.home import HomeScreen
from ui.screens.play import PlayScreen
from ui.screens.test_model import TestModelScreen
from ui.screens.train_model import TrainModelScreen


def _pick_monospace_family():
    """
    Pick the first available real monospace family instead of hardcoding one --
    font availability differs across Linux/Windows/macOS. "Courier" is one of
    Tk's built-in core families and is guaranteed to exist even if none of the
    preferred candidates are installed.
    """
    available = set(tkfont.families())
    for candidate in ("Consolas", "DejaVu Sans Mono", "Noto Sans Mono", "Liberation Mono", "Menlo", "Courier New"):
        if candidate in available:
            return candidate
    return "Courier"


def _force_redraw(widget):
    """
    Recursively re-invoke every CTk widget's own internal canvas-paint hook.
    CustomTkinter widgets draw themselves onto an internal canvas via _draw()
    on creation/state-change, not via Tk's native repaint -- an external
    geometry change (e.g. maximizing the window) doesn't by itself make CTk
    re-run _draw() on widgets whose own allocated size didn't change, so their
    canvas can be left showing nothing until some later per-widget event
    (hover/click) happens to trigger it. Forcing every widget's _draw() once
    after a resize settles fixes that without needing user interaction first.
    """
    draw = getattr(widget, "_draw", None)
    if callable(draw):
        try:
            draw()
        except Exception:
            pass
    for child in widget.winfo_children():
        _force_redraw(child)


class App(ctk.CTk):
    def __init__(self):
        ctk.set_appearance_mode("dark")
        super().__init__()
        # Hidden until HomeScreen is showing (see deiconify() at the end of
        # __init__) -- otherwise the window maps and paints whatever screen
        # happens to be topmost mid-construction (place() stacks each screen
        # as it's built, last-built on top) before show("HomeScreen") below
        # raises the right one, which reads as a brief flicker of the wrong
        # screen on every launch.
        self.withdraw()

        self.title("Snake RL - Launcher")

        # Start at the same ~2/3 fraction of the screen that an HD (1280x720)
        # window occupies on a Full HD (1920x1080) display, in a 16:9 ratio --
        # scales proportionally on any monitor instead of a fixed pixel size
        # (tiny on 4K, cramped on a small laptop) and doesn't fill the screen.
        # Derived from screen *height* only (not width): side-by-side multi-monitor
        # setups sum widths into one wide virtual desktop but keep a real height,
        # so sizing off height avoids computing a giant window spanning monitors.
        # No explicit +x+y either, for the same reason -- the window manager
        # places/centers new windows on the actual active monitor far more
        # reliably than we could from Tk's combined virtual-desktop coordinates.
        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
        target_h = max(int(screen_h * (720 / 1080)), 600)
        target_w = min(int(target_h * 16 / 9), screen_w)
        self.geometry(f"{target_w}x{target_h}")
        self.minsize(MIN_WIDTH, 600)
        self.configure(fg_color=BG)

        # Only one background action (play/test/train) may run at a time.
        self.busy = False

        self.font_h1 = ctk.CTkFont(family="Noto Sans", size=30, weight="bold")
        self.font_h2 = ctk.CTkFont(family="Noto Sans", size=22, weight="bold")
        self.font_card_title = ctk.CTkFont(family="Noto Sans", size=20, weight="bold")
        self.font_body = ctk.CTkFont(family="Noto Sans", size=15)
        self.font_small = ctk.CTkFont(family="Noto Sans", size=13)
        self.font_mono = ctk.CTkFont(family=_pick_monospace_family(), size=13)

        container = ctk.CTkFrame(self, fg_color=BG)
        container.pack(fill="both", expand=True)

        self.screens = {}
        for screen_cls in (HomeScreen, PlayScreen, TestModelScreen, TrainModelScreen):
            screen = screen_cls(container, self)
            screen.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.screens[screen_cls.__name__] = screen

        self.show("HomeScreen")
        self.protocol("WM_DELETE_WINDOW", self._on_close_request)

        # Redraw every widget once a resize (e.g. maximizing) settles -- see
        # _force_redraw()'s docstring. Debounced via after()/after_cancel() so
        # a drag-resize (many rapid <Configure> events) triggers one redraw
        # pass shortly after it stops, not one per intermediate event.
        self._redraw_after_id = None
        self.bind("<Configure>", self._on_configure)

        self.deiconify()

    def _on_configure(self, event):
        if event.widget is not self:
            return
        if self._redraw_after_id is not None:
            self.after_cancel(self._redraw_after_id)
        self._redraw_after_id = self.after(150, self._redraw_all)

    def _redraw_all(self):
        self._redraw_after_id = None
        _force_redraw(self)

    def show(self, name):
        screen = self.screens[name]
        screen.tkraise()
        if isinstance(screen, SubScreen):
            # Measured, not guessed: keeps whatever's expandable in this screen
            # (a scrollable list, or a log box) from ever collapsing to 0px, and
            # grows the window immediately if it's currently smaller than that --
            # switching to a more demanding screen never leaves a clipped layout.
            req_h = screen.min_required_height()
            self.minsize(MIN_WIDTH, req_h)
            cur_w, cur_h = self.winfo_width(), self.winfo_height()
            if cur_w < MIN_WIDTH or cur_h < req_h:
                self.geometry(f"{max(cur_w, MIN_WIDTH)}x{max(cur_h, req_h)}")

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
