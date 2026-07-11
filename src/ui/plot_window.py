"""
ui/plot_window.py - Embeddable, live-refreshable training-progress graph, read
straight from the tensorboard event file train_model() is writing to -- can be
polled repeatedly while training is still running, not just after it finishes.
"""

import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from ui.theme import PANEL, TEXT, TEXT_MUTED, BORDER, RADIUS

# (tag, subplot label) -- only plotted once actually present in the run's event
# file (e.g. DQN doesn't log train/loss until learning_starts steps are
# collected, so an early poll may only have rollout/ep_rew_mean).
_PLOTTED_TAGS = [
    ("rollout/ep_rew_mean", "Mean episode reward"),
    ("train/loss", "Loss"),
]


class LiveTrainingPlot(ctk.CTkFrame):
    """A small reward/loss plot that redraws whenever `.update(log_dir)` is
    called -- cheap to call repeatedly (e.g. from a polling timer) since it
    only rebuilds its subplots when the *set* of available tags changes,
    otherwise just updates the existing lines' data."""

    def __init__(self, parent, app, height=220):
        super().__init__(parent, fg_color=PANEL, corner_radius=RADIUS, height=height)
        self.app = app
        self._axes = {}
        self._lines = {}
        self._marker_artists = []

        self._fig = Figure(dpi=100)
        self._fig.patch.set_facecolor(PANEL)
        # Named _mpl_canvas, not _canvas -- CTkFrame itself already uses
        # self._canvas internally (its own rounded-corner rendering canvas);
        # reusing that name here silently shadows it and breaks CTkFrame.bind().
        self._mpl_canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._mpl_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

        self._placeholder = ctk.CTkLabel(
            self, text="No training data yet.", font=app.font_body, text_color=TEXT_MUTED,
        )
        self._placeholder.place(relx=0.5, rely=0.5, anchor="center")

    def reset(self):
        """Called when a new run starts, so stale data from a previous run
        (or a previous config) doesn't linger on screen until the new run's
        first data point arrives."""
        self._fig.clear()
        self._axes = {}
        self._lines = {}
        self._marker_artists = []
        self._mpl_canvas.draw_idle()
        self._placeholder.place(relx=0.5, rely=0.5, anchor="center")

    def update(self, log_dir, marker_steps=None):
        """Re-read `log_dir`'s tensorboard event file and redraw. No-op if
        `log_dir` is None (not known yet) or nothing plottable has been
        logged yet. `marker_steps` (see rl.paths._record_continue_marker) draws
        a dashed vertical line at each past "Continue Existing" resume point,
        so multiple continuations are each visible on the merged curve."""
        if log_dir is None:
            return

        ea = EventAccumulator(log_dir)
        ea.Reload()
        available = ea.Tags().get("scalars", [])
        plots = [(tag, label) for tag, label in _PLOTTED_TAGS if tag in available]
        if not plots:
            return

        self._placeholder.place_forget()

        if set(tag for tag, _ in plots) != set(self._axes.keys()):
            # The set of plottable tags grew (e.g. train/loss just started
            # appearing) -- rebuild the subplot layout to fit all of them.
            self._fig.clear()
            self._axes = {}
            self._lines = {}
            for i, (tag, label) in enumerate(plots):
                ax = self._fig.add_subplot(len(plots), 1, i + 1)
                ax.set_ylabel(label, color=TEXT)
                ax.set_facecolor(PANEL)
                ax.tick_params(colors=TEXT_MUTED, labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color(BORDER)
                if i == len(plots) - 1:
                    ax.set_xlabel("Timesteps", color=TEXT)
                line, = ax.plot([], [], color="#4d96ff")
                self._axes[tag] = ax
                self._lines[tag] = line
            self._fig.tight_layout()

        for tag, _ in plots:
            events = ea.Scalars(tag)
            self._lines[tag].set_data([e.step for e in events], [e.value for e in events])
            self._axes[tag].relim()
            self._axes[tag].autoscale_view()

        # Redrawn from scratch each call (cheap -- marker_steps is a handful of
        # ints at most) rather than diffed, since axvline artists don't support
        # updating their x-position in place.
        for artist in self._marker_artists:
            artist.remove()
        self._marker_artists = []
        if marker_steps:
            top_axis = next(iter(self._axes.values()))
            for i, step in enumerate(marker_steps):
                for tag, ax in self._axes.items():
                    label = "Continue" if (ax is top_axis and i == 0) else None
                    self._marker_artists.append(
                        ax.axvline(x=step, color=TEXT_MUTED, linestyle="--", linewidth=1, alpha=0.7, label=label)
                    )
            top_axis.legend(fontsize=7, loc="upper left", facecolor=PANEL, labelcolor=TEXT_MUTED, framealpha=0.5)

        self._mpl_canvas.draw_idle()
