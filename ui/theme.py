"""
ui/theme.py - Color palette, small color/text helpers, and shared layout constants.
"""

from game.snake_game import (
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


# Color theme - reuses the same palette as the game itself (game/snake_game.py)
# so the launcher and the game look like one cohesive app.
BG = _hex(COLOR_BACKGROUND)
PANEL = _hex(COLOR_SCORE_PANEL)
BORDER = _hex(COLOR_GRID_LINE)
TEXT = _hex(COLOR_SCORE_TEXT)
TEXT_MUTED = "#9195a8"

GREEN = _hex(COLOR_SNAKE_HEAD)          # success / positive (e.g. "Play")
RED = _hex(COLOR_APPLE)                 # danger / stop / errors
# Not a named constant in snake_game.py, but matches the amber accent used
# for the FOV debug overlay (game/environment.py), kept consistent here.
AMBER = "#ffd25a"

# Per-category accent colors for model-list badges (algorithm / observation mode) --
# distinct from GREEN/RED/AMBER above, which already carry a semantic meaning
# (success/danger/selection) elsewhere in the app, so reusing them here would
# make a PPO or GRID badge look like a status indicator instead of a label.
ALGO_COLORS = {"PPO": "#5ac8fa", "DQN": "#bf8cff"}   # sky blue / violet
OBS_COLORS = {"FLAT": "#ff8a65", "GRID": "#4fd1c5"}  # coral / teal

# Corner radius shared by every rounded widget (buttons, sliders, cards, entries, ...).
RADIUS = 10

# Max width for a screen's control column (sliders, choice rows, entries) -- kept
# narrower than the full panel and centered, rather than stretching every control
# edge-to-edge across the whole (much wider) window. Model-card lists intentionally
# don't use this -- they read fine at full width, like any list of items.
CONTENT_WIDTH = 560

# Minimum usable height for whichever child of a SubScreen's body is the single
# expandable one (a CTkScrollableFrame's list, or a log box) -- enough to show a
# real, functioning scroll region (roughly one row plus a hint of the next),
# never a collapsed 0px sliver. Used by SubScreen.min_required_height().
MIN_FLEX_HEIGHT = 240

# Minimum window width -- comfortably fits CONTENT_WIDTH plus side padding and a
# scrollbar; not computed dynamically since no screen has reported width issues.
MIN_WIDTH = 760
