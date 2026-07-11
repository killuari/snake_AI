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

# (tag, subplot label) -- all subplots are built up front so their (shared)
# x-axis is visible from the start, even for tags with no data yet (e.g. DQN
# doesn't log train/loss until learning_starts steps are collected) -- only
# each tag's own line waits for its data to actually appear.
_PLOTTED_TAGS = [
    ("rollout/ep_rew_mean", "Mean episode reward"),
    ("train/loss", "Loss"),
]

_DPI = 100

# Default is 10,000 points per tag, after which older points get randomly
# evicted (reservoir sampling) -- raised so long production runs (millions of
# timesteps) don't lose historical resolution.
_SIZE_GUIDANCE = {"scalars": 100_000}


class LiveTrainingPlot(ctk.CTkFrame):
    """A small reward/loss plot that redraws whenever `.update(log_dir)` is
    called -- cheap to call repeatedly (e.g. from a polling timer) since it
    only builds its subplots once (the first call with any data at all),
    otherwise just updates the existing lines' data."""

    def __init__(self, parent, app, height=672):
        super().__init__(parent, fg_color=PANEL, corner_radius=RADIUS, height=height)
        self.app = app
        self._axes = {}
        self._lines = {}
        self._marker_artists = []

        self._fig = Figure(figsize=(6.4, height / _DPI), dpi=_DPI)
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

        ea = EventAccumulator(log_dir, size_guidance=_SIZE_GUIDANCE)
        ea.Reload()
        available = ea.Tags().get("scalars", [])
        if not any(tag in available for tag, _ in _PLOTTED_TAGS):
            return

        self._placeholder.place_forget()

        if not self._axes:
            # First data of any kind for this run -- build every subplot in
            # _PLOTTED_TAGS up front (not just the ones with data yet) so the
            # loss subplot's axis is visible immediately instead of popping
            # in once train/loss starts logging; otherwise the reward
            # subplot would balloon to the whole figure's height until then.
            self._fig.clear()
            self._axes = {}
            self._lines = {}
            shared_ax = None
            for i, (tag, label) in enumerate(_PLOTTED_TAGS):
                # sharex so the reward and loss subplots always cover the
                # same timestep range -- otherwise each autoscales to its own
                # data (loss starts logging late, at learning_starts steps)
                # and the same continue marker lands at different x-pixels
                # on each subplot.
                ax = self._fig.add_subplot(len(_PLOTTED_TAGS), 1, i + 1, sharex=shared_ax)
                shared_ax = shared_ax or ax
                ax.set_ylabel(label, color=TEXT)
                ax.set_facecolor(PANEL)
                ax.tick_params(colors=TEXT_MUTED, labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color(BORDER)
                if i == len(_PLOTTED_TAGS) - 1:
                    ax.set_xlabel("Timesteps", color=TEXT)
                else:
                    ax.tick_params(labelbottom=False)
                line, = ax.plot([], [], color="#4d96ff")
                self._axes[tag] = ax
                self._lines[tag] = line
            self._fig.tight_layout()

        for tag, _ in _PLOTTED_TAGS:
            if tag not in available:
                continue  # no data logged for this tag yet -- leave its axis empty but visible
            # Sorted by step -- EventAccumulator merges multiple event files
            # (e.g. across "Continue Existing" runs) in file-processing order,
            # not step order, so an unsorted plot can connect distant points
            # with a long diagonal line whenever two files' step ranges
            # aren't already contiguous (see rl.paths._clear_stale_tb_history
            # for the data-level fix; this is a defensive backstop for any
            # other multi-file merge situation).
            events = sorted(ea.Scalars(tag), key=lambda e: e.step)
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
