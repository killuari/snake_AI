"""
ui/screens/base.py - SubScreen: shared header/body/log/background-worker scaffolding
for every non-home screen.
"""

import sys
import queue
import threading
import customtkinter as ctk

from ui.theme import BG, PANEL, TEXT, TEXT_MUTED, RADIUS, MIN_FLEX_HEIGHT
from ui.widgets import _make_outline_button, _QueueWriter


class SubScreen(ctk.CTkFrame):
    """Base for every non-home screen: a back button + title header, plus a body frame for content."""

    def __init__(self, parent, app, title):
        super().__init__(parent, fg_color=BG)
        self.app = app
        self.log_box = None
        self._log_queue = queue.Queue()

        self.header = ctk.CTkFrame(self, fg_color="transparent")
        self.header.pack(fill="x", padx=24, pady=(20, 10))

        _make_outline_button(
            self.header, "← Back", TEXT_MUTED, lambda: self._handle_back(), app.font_body, width=90, height=36,
        ).pack(side="left")

        ctk.CTkLabel(self.header, text=title, font=app.font_h2, text_color=TEXT).pack(side="left", padx=20)

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="both", expand=True, padx=24, pady=(0, 24))

    def _handle_back(self):
        """Overridden by screens that need to guard navigating away (e.g. Train Model mid-run)."""
        self.app.show("HomeScreen")

    def min_required_height(self):
        """
        The real minimum window height this screen needs: every always-visible
        child of self.body measured via winfo_reqheight() (not guessed), plus
        MIN_FLEX_HEIGHT for whichever single child is the expandable one (a
        CTkScrollableFrame's list, or a log box) -- so the expandable region
        can shrink down to a still-functional sliver but never to 0px, and this
        number automatically grows if a screen ever gains more always-visible
        controls instead of silently going stale like a hand-picked constant.
        """
        self.update_idletasks()

        def pad_total(value):
            if isinstance(value, (tuple, list)):
                return sum(int(v) for v in value)
            return int(value) * 2

        total = self.header.winfo_reqheight() + pad_total(self.header.pack_info().get("pady", 0))
        total += pad_total(self.body.pack_info().get("pady", 0))
        for child in self.body.winfo_children():
            info = child.pack_info()
            if info.get("expand"):
                continue
            total += child.winfo_reqheight() + pad_total(info.get("pady", 0))
        return total + MIN_FLEX_HEIGHT + 20

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
        follow = log_box is None or log_box.yview()[1] >= 0.999 # type: ignore
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
