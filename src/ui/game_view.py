"""
ui/game_view.py - Embeddable, live-refreshable view of the training environment,
fed frames straight from a SubprocVecEnv worker's SnakeGameEnvironment.render()
call (see rl.training.train_model's on_frame callback). Unlike
ui/plot_window.py's LiveTrainingPlot (which re-reads a tensorboard event file
from disk on every poll), the frame here is already an in-memory numpy array --
update() just needs to be called with the latest one.
"""

import customtkinter as ctk
from PIL import Image

from ui.theme import PANEL, TEXT_MUTED, RADIUS

_DISPLAY_WIDTH = 440   # Fixed display width; height follows the frame's own aspect ratio.


class LiveGameView(ctk.CTkFrame):
    """Shows the latest rendered training frame, scaled down to a fixed display
    width. Cheap to call repeatedly (e.g. from a ~120ms polling timer): each
    update() wraps the frame in a PIL Image, resizes it, and reconfigures a
    single reused CTkImage/CTkLabel. Main-thread only (like
    LiveTrainingPlot.update())."""

    def __init__(self, parent, app, display_width=_DISPLAY_WIDTH):
        super().__init__(parent, fg_color=PANEL, corner_radius=RADIUS)
        self.app = app
        self._display_width = display_width
        self._ctk_image = None   # Created lazily on first frame

        self._image_label = ctk.CTkLabel(self, text="")
        self._image_label.pack(pady=8)

        self._placeholder = ctk.CTkLabel(
            self, text="No live frame yet.", font=app.font_body, text_color=TEXT_MUTED,
        )
        self._placeholder.place(relx=0.5, rely=0.5, anchor="center")

    def reset(self):
        """Called when a new run starts, clearing any frame left over from a
        previous run/config so it doesn't linger until the new run's first
        frame arrives."""
        self._image_label.configure(image=None)
        self._ctk_image = None
        self._placeholder.place(relx=0.5, rely=0.5, anchor="center")

    def update(self, frame):
        """Redraw with `frame`, an (H, W, 3) uint8 RGB numpy array. No-op if
        `frame` is None (nothing rendered yet this run)."""
        if frame is None:
            return

        self._placeholder.place_forget()

        pil_image = Image.fromarray(frame)
        aspect = pil_image.height / pil_image.width
        size = (self._display_width, round(self._display_width * aspect))

        if self._ctk_image is None:
            self._ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=size)
            self._image_label.configure(image=self._ctk_image)
        else:
            self._ctk_image.configure(light_image=pil_image, dark_image=pil_image, size=size)
