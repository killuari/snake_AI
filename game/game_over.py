"""
game/game_over.py - Shared death-animation + game-over overlay.

run_game_over() is called by both rl.playback.play_game() (human play) and
.test_model() (watching a trained model) right after the snake dies, so dying
looks and behaves the same on both paths: a brief death flash over the frame
the snake died on, then a blurred/darkened freeze of that frame with the score
and Restart/Quit buttons, blocking until the player picks one.
"""

import pygame

from game.snake_game import COLOR_BACKGROUND, COLOR_SCORE_TEXT, COLOR_SCORE_PANEL, COLOR_APPLE, COLOR_SNAKE_HEAD


def _blur(surface, downscale=0.12):
    """
    Cheap gaussian-ish blur: shrink the surface then smoothscale it back up.
    This pygame build (2.6.1, the regular PyPI wheel -- not pygame-ce) has no
    gaussian_blur()/box_blur(), so the downscale/upscale trick is the portable
    option; smoothscale's bilinear interpolation on both passes is what
    produces the blur.
    """
    w, h = surface.get_size()
    small_size = (max(1, int(w * downscale)), max(1, int(h * downscale)))
    small = pygame.transform.smoothscale(surface, small_size)
    return pygame.transform.smoothscale(small, (w, h))


def _draw_button(surface, rect, label, font, mouse_pos, accent):
    hovered = rect.collidepoint(mouse_pos)
    pygame.draw.rect(surface, pygame.Color(COLOR_SCORE_PANEL.r, COLOR_SCORE_PANEL.g, COLOR_SCORE_PANEL.b), rect, border_radius=10)
    pygame.draw.rect(surface, accent if hovered else pygame.Color(60, 64, 84), rect, width=2, border_radius=10)
    text = font.render(label, True, accent if hovered else COLOR_SCORE_TEXT)
    surface.blit(text, text.get_rect(center=rect.center))


def run_game_over(screen, clock, final_score, fps=60):
    """
    Blocking game-over sequence rendered onto `screen` (a live Pygame display
    surface -- its current content is taken as the death frame). Returns
    "restart" or "quit" once the player picks a button, presses
    Enter/R/Space (restart) or Escape/Q (quit), or closes the window (quit).
    """
    w, h = screen.get_size()
    final_frame = screen.copy()

    # --- 1) brief flash over the frame the snake died on ---
    flash = pygame.Surface((w, h))
    flash.fill(COLOR_SCORE_TEXT)
    for i in range(6):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
        screen.blit(final_frame, (0, 0))
        if i % 2 == 0:
            flash.set_alpha(130)
            screen.blit(flash, (0, 0))
        pygame.display.flip()
        clock.tick(12)

    # --- 2) blurred, darkened backdrop ---
    backdrop = _blur(final_frame)
    dark = pygame.Surface((w, h), pygame.SRCALPHA)
    dark.fill(pygame.Color(COLOR_BACKGROUND.r, COLOR_BACKGROUND.g, COLOR_BACKGROUND.b, 165))

    title_font = pygame.font.Font(None, max(48, w // 12))
    score_font = pygame.font.Font(None, max(30, w // 20))
    btn_font = pygame.font.Font(None, 36)

    bw, bh, gap = 180, 52, 24
    cx, cy = w // 2, h // 2
    restart_rect = pygame.Rect(cx - bw - gap // 2, cy + 40, bw, bh)
    quit_rect = pygame.Rect(cx + gap // 2, cy + 40, bw, bh)

    # --- 3) interactive overlay: "GAME OVER" + score + Restart/Quit ---
    while True:
        mouse = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return "quit"
                if event.key in (pygame.K_RETURN, pygame.K_r, pygame.K_SPACE):
                    return "restart"
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if restart_rect.collidepoint(event.pos):
                    return "restart"
                if quit_rect.collidepoint(event.pos):
                    return "quit"

        screen.blit(backdrop, (0, 0))
        screen.blit(dark, (0, 0))

        title = title_font.render("GAME OVER", True, COLOR_APPLE)
        score_text = score_font.render(f"Score: {final_score}", True, COLOR_SCORE_TEXT)
        screen.blit(title, title.get_rect(center=(cx, cy - 60)))
        screen.blit(score_text, score_text.get_rect(center=(cx, cy - 15)))

        _draw_button(screen, restart_rect, "Restart", btn_font, mouse, COLOR_SNAKE_HEAD)
        _draw_button(screen, quit_rect, "Quit", btn_font, mouse, COLOR_APPLE)

        pygame.display.flip()
        clock.tick(fps)
