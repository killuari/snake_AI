"""
ui/widgets.py - Reusable CustomTkinter widget factories shared across screens.
"""

import tkinter as tk
import customtkinter as ctk

from ui.theme import BG, PANEL, BORDER, TEXT, TEXT_MUTED, AMBER, RADIUS, CONTENT_WIDTH, ALGO_COLORS, OBS_COLORS, _mix, _spaced


def _make_content_column(parent):
    """
    A width-capped container for a screen's control rows, packed without
    fill="x" so pack's default centering places it in the middle of the
    (wider) panel instead of stretching every row edge-to-edge. Deliberately
    doesn't disable pack-propagate: the width is a hint the rows inside
    (which use fill="x" relative to *this* frame, not the outer panel) settle
    around, not a hard clamp that could collapse the frame's height to zero.
    """
    return ctk.CTkFrame(parent, fg_color="transparent", width=CONTENT_WIDTH)


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


def _enable_mousewheel(scrollable_frame, nested=()):
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

    `nested`: an iterable of (widget, scroll_fn) pairs for regions that must
    scroll independently of `scrollable_frame` (e.g. a taller log box or a
    card list nested inside an overall-scrollable screen) -- wheel events
    anywhere inside one of these widgets' subtrees call its own `scroll_fn(d)`
    (d: -1 up / +1 down) instead of the outer canvas, and consume the event
    ("break") so it doesn't *also* scroll the outer frame at the same time.
    """
    outer_canvas = scrollable_frame._parent_canvas
    nested_map = dict(nested)

    def _make_handler(scroll_fn):
        def _on_wheel(event):
            if event.num == 4:
                scroll_fn(-1)
            elif event.num == 5:
                scroll_fn(1)
            elif getattr(event, "delta", 0):
                scroll_fn(-1 if event.delta > 0 else 1)
            return "break"
        return _on_wheel

    outer_handler = _make_handler(lambda d: outer_canvas.yview_scroll(d, "units"))

    def _walk(widget, handler):
        if widget in nested_map:
            handler = _make_handler(nested_map[widget])
        for event_name in ("<Button-4>", "<Button-5>", "<MouseWheel>"):
            try:
                widget.bind(event_name, handler)
            except NotImplementedError:
                pass
        for child in widget.winfo_children():
            _walk(child, handler)

    _walk(scrollable_frame, outer_handler)


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


class _Choice:
    """A .get()/.set()/.configure(state=...) handle for _make_choice_row()."""

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
    A labeled row of mutually-exclusive outline buttons (used everywhere a
    single choice needs to be made -- algorithm, grid size, playback mode, ...).
    Built from individual buttons instead of CTkSegmentedButton for two reasons:
    CTkSegmentedButton only exposes one shared text_color for every segment (so
    the selected segment's bright fill and the unselected ones can never both
    have good text contrast), and a 3+ option segmented button was also observed
    to render completely blank on this Tk build even though its internal state
    was correct.
    """
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", pady=10)
    ctk.CTkLabel(row, text=label, font=font_body, text_color=TEXT).pack(anchor="w")

    # anchor="w" (not fill="x"): the button group sits at its own natural width
    # instead of being stretched across the whole panel -- a 2-3 word choice
    # spanning ~900px read as a thin, oddly-proportioned bar.
    btn_row = ctk.CTkFrame(row, fg_color="transparent")
    btn_row.pack(anchor="w", pady=(6, 0))

    buttons = {}

    def restyle(value, btn):
        # Selected/unselected is pushed through fg_color *and* hover_color
        # together (not layered as a separate <Enter>/<Leave> border bind on
        # top, like the plain outline buttons use) -- CTkButton applies its
        # configured hover_color on <Enter> and restores fg_color on <Leave>
        # by itself, regardless of what apply_visual() last set. A bind that
        # unconditionally paints the hover/rest border independently of
        # selection used to fight that: selecting a button while the pointer
        # stayed over it flashed amber then reverted to looking unselected
        # until the pointer moved away. Locking hover_color to the same fill
        # as the selected state removes the conflict instead of racing it.
        if value == choice._value:
            fill = _mix(PANEL, AMBER, 0.28)
            btn.configure(fg_color=fill, hover_color=fill, text_color=AMBER, border_color=AMBER)
        else:
            btn.configure(fg_color=PANEL, hover_color=PANEL, text_color=TEXT_MUTED, border_color=BORDER)

    def apply_visual(value):
        choice._value = value
        for v, btn in buttons.items():
            restyle(v, btn)

    def select(value):
        # Only user clicks notify `command` -- matches CTkSegmentedButton, whose
        # .set() (used for the initial value and programmatic changes) does not.
        apply_visual(value)
        if command:
            command(value)

    def on_enter(value, btn, _event=None):
        # Border-only hover highlight for the *unselected* buttons, matching
        # the plain outline-button language -- skipped for the selected
        # button, whose border is pinned to AMBER by restyle() above and must
        # never be touched by hover (that was the other half of the bug: it
        # otherwise got reset to BORDER on <Leave>).
        if value != choice._value:
            btn.configure(border_color=AMBER)

    def on_leave(value, btn, _event=None):
        if value != choice._value:
            btn.configure(border_color=BORDER)

    for i, value in enumerate(values):
        # Width follows each option's own text instead of one fixed width for
        # every row -- "DQN" and "Continue Existing" need very different sizes.
        width = max(110, 40 + len(value) * 10)
        btn = ctk.CTkButton(
            btn_row, text=value, font=font_body, command=lambda v=value: select(v),
            fg_color=PANEL, hover_color=PANEL, text_color=TEXT_MUTED,
            border_width=2, border_color=BORDER, corner_radius=RADIUS,
            width=width, height=44,
        )
        btn.bind("<Enter>", lambda _e, v=value, b=btn: on_enter(v, b), add="+")
        btn.bind("<Leave>", lambda _e, v=value, b=btn: on_leave(v, b), add="+")
        btn.pack(side="left", padx=(0 if i == 0 else 8, 0))
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


def _speed_label(v):
    return f"{int(v)}  ({int(v) * 10} FPS)"


def _make_model_badge(parent, info, font, text_color):
    """
    The "[PPO] [FLAT]  Grid 30x20  FOV 5" header line on a model card, with the
    algorithm and observation-mode brackets colored per ALGO_COLORS/OBS_COLORS
    so scanning a long list for a specific combination doesn't require reading
    every badge's text -- three separate labels (CTkLabel only takes one
    text_color for its whole string) packed together to reproduce the exact
    spacing of the original single-string badge.
    """
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(side="left")
    ctk.CTkLabel(row, text=f"[{info['algo']}]", font=font, text_color=ALGO_COLORS[info["algo"]]).pack(side="left")
    ctk.CTkLabel(row, text=f" [{info['obs_mode']}]", font=font, text_color=OBS_COLORS[info["obs_mode"]]).pack(side="left")
    ctk.CTkLabel(
        row, text=f"  Grid {info['grid_width']}x{info['grid_height']}  FOV {info['fov']}",
        font=font, text_color=text_color,
    ).pack(side="left")


def _format_timesteps(info):
    """Both best/last timestep counts, formatted for display -- e.g.
    "Timesteps: best 1,440  |  last 6,000" (the "Timesteps:" prefix makes
    clear what the numbers mean, since they're otherwise just bare counts).
    Shows both since they can genuinely differ (best_model.zip is saved at
    whichever timestep EvalCallback last found an improvement, not
    necessarily the run's final one -- see rl.training.train_model's
    _RecordBestTimestep). Omits either half if that checkpoint doesn't exist
    yet (e.g. right after a discard left only "last")."""
    parts = []
    if info["best_timesteps"] is not None:
        parts.append(f"best {info['best_timesteps']:,}")
    if info["last_timesteps"] is not None:
        parts.append(f"last {info['last_timesteps']:,}")
    if not parts:
        return ""
    return "Timesteps: " + "  |  ".join(parts)
