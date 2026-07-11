"""
main.py - Entry point for the Snake RL project.

Run this file directly (`python main.py`) to launch the graphical launcher UI
(see ui/), which lets you choose Play / Test Model / Train Model and
configure parameters.

The actual functionality lives in:
    game/    - Core Snake game engine and the Gymnasium environment.
    rl/      - Training pipeline, playback, hyperparameter tuning.
    ui/      - The CustomTkinter desktop launcher.
"""

if __name__ == "__main__":
    from ui import App
    App().mainloop()
