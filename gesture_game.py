"""
gesture_game.py
---------------
Gesture-Controlled Dodge Game
==============================
Controls
--------
  ☝  Index finger UP   → move paddle UP
  ✊  Fist (DOWN)       → move paddle DOWN
  🖐  Open hand (OPEN)  → PAUSE / RESUME
  ✌  Peace sign        → restart after Game-Over

The webcam feed is shown in a small overlay inside the Pygame window.
Press  Q  or close the window to quit.
"""

import sys
import random
import time

import cv2
import numpy as np
import pygame

from hand_detector import HandDetector

# ── Constants ──────────────────────────────────────────────────────────────────

WIN_W, WIN_H   = 900, 600          # Pygame window size
CAM_W, CAM_H   = 240, 180          # Webcam overlay size (bottom-right corner)
FPS            = 60

PADDLE_W, PADDLE_H = 18, 90
BALL_SIZE           = 18
BALL_SPEED_INIT     = 5
BALL_SPEED_MAX      = 14
SPEED_INCREMENT     = 0.4          # speed increase per scored point
PADDLE_SPEED        = 7

# Colour palette
C_BG        = (10,  10,  30)
C_GRID      = (20,  20,  50)
C_PADDLE    = (0,   220, 180)
C_BALL      = (255, 80,  80)
C_TEXT      = (230, 230, 255)
C_ACCENT    = (255, 200, 0)
C_OVERLAY   = (0,   0,   0,  160)  # semi-transparent (RGBA)
C_GESTURE   = {
    "UP":    (0,   255, 150),
    "DOWN":  (255, 100, 100),
    "OPEN":  (255, 220, 0),
    "PEACE": (100, 180, 255),
    "OTHER": (180, 180, 180),
    "NONE":  (80,  80,  80),
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def draw_grid(surface: pygame.Surface) -> None:
    """Draw a subtle dot-grid background."""
    for x in range(0, WIN_W, 40):
        for y in range(0, WIN_H, 40):
            pygame.draw.circle(surface, C_GRID, (x, y), 1)


def draw_rounded_rect(surface: pygame.Surface, color: tuple,
                      rect: pygame.Rect, radius: int = 10) -> None:
    pygame.draw.rect(surface, color, rect, border_radius=radius)


def cv_frame_to_pygame(frame: np.ndarray, size: tuple[int, int]) -> pygame.Surface:
    """Convert an OpenCV BGR frame to a scaled Pygame surface."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, size)
    # Mirror so it feels like a mirror
    frame_rgb = cv2.flip(frame_rgb, 1)
    return pygame.surfarray.make_surface(np.rot90(frame_rgb))


# ── Game Objects ───────────────────────────────────────────────────────────────

class Paddle:
    def __init__(self, x: int):
        self.rect = pygame.Rect(x, WIN_H // 2 - PADDLE_H // 2, PADDLE_W, PADDLE_H)
        self.trail: list[pygame.Rect] = []

    def move(self, direction: int) -> None:
        """direction: -1 = up, +1 = down, 0 = still."""
        self.trail.append(self.rect.copy())
        if len(self.trail) > 6:
            self.trail.pop(0)
        self.rect.y += direction * PADDLE_SPEED
        self.rect.clamp_ip(pygame.Rect(0, 0, WIN_W, WIN_H))

    def draw(self, surface: pygame.Surface) -> None:
        # Motion trail
        for i, r in enumerate(self.trail):
            alpha = int(40 + 30 * (i / max(len(self.trail), 1)))
            s = pygame.Surface((r.width, r.height), pygame.SRCALPHA)
            s.fill((*C_PADDLE, alpha))
            surface.blit(s, r.topleft)
        draw_rounded_rect(surface, C_PADDLE, self.rect, radius=6)


class Ball:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.rect  = pygame.Rect(WIN_W // 2, WIN_H // 2, BALL_SIZE, BALL_SIZE)
        self.speed = BALL_SPEED_INIT
        self.vx    = self.speed * random.choice([-1, 1])
        self.vy    = self.speed * random.choice([-1, 1])
        self.trail: list[tuple[int, int]] = []

    def update(self) -> None:
        self.trail.append(self.rect.center)
        if len(self.trail) > 10:
            self.trail.pop(0)
        self.rect.x += int(self.vx)
        self.rect.y += int(self.vy)
        # Bounce off top / bottom
        if self.rect.top <= 0 or self.rect.bottom >= WIN_H:
            self.vy *= -1

    def draw(self, surface: pygame.Surface) -> None:
        for i, pos in enumerate(self.trail):
            r = max(2, int(BALL_SIZE * 0.4 * (i / max(len(self.trail), 1))))
            alpha = int(20 + 20 * (i / max(len(self.trail), 1)))
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*C_BALL, alpha), (r, r), r)
            surface.blit(s, (pos[0] - r, pos[1] - r))
        pygame.draw.rect(surface, C_BALL, self.rect, border_radius=4)


# ── HUD drawing ────────────────────────────────────────────────────────────────

def draw_hud(surface: pygame.Surface, font_lg: pygame.font.Font,
             font_sm: pygame.font.Font, score: int, high_score: int,
             gesture: str, paused: bool) -> None:
    # Score
    score_surf = font_lg.render(f"{score:03d}", True, C_ACCENT)
    surface.blit(score_surf, (WIN_W // 2 - score_surf.get_width() // 2, 12))

    # High score
    hs_surf = font_sm.render(f"BEST  {high_score:03d}", True, C_TEXT)
    surface.blit(hs_surf, (WIN_W // 2 - hs_surf.get_width() // 2, 52))

    # Gesture badge
    g_color = C_GESTURE.get(gesture, C_GESTURE["OTHER"])
    badge_rect = pygame.Rect(14, 14, 160, 36)
    draw_rounded_rect(surface, (30, 30, 60), badge_rect, radius=8)
    pygame.draw.rect(surface, g_color, badge_rect, width=2, border_radius=8)
    g_surf = font_sm.render(f"✋  {gesture}", True, g_color)
    surface.blit(g_surf, (badge_rect.x + 10, badge_rect.y + 8))

    # Controls legend
    legend = ["☝ UP", "✊ DOWN", "🖐 PAUSE", "✌ RESTART"]
    for i, txt in enumerate(legend):
        s = font_sm.render(txt, True, (120, 120, 160))
        surface.blit(s, (14, WIN_H - 28 - i * 22))

    # Pause banner
    if paused:
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        surface.blit(overlay, (0, 0))
        p_surf = font_lg.render("⏸  PAUSED", True, C_ACCENT)
        surface.blit(p_surf, (WIN_W // 2 - p_surf.get_width() // 2,
                               WIN_H // 2 - p_surf.get_height() // 2))


def draw_game_over(surface: pygame.Surface, font_xl: pygame.font.Font,
                   font_sm: pygame.font.Font, score: int) -> None:
    overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surface.blit(overlay, (0, 0))

    go_surf = font_xl.render("GAME OVER", True, C_BALL)
    surface.blit(go_surf, (WIN_W // 2 - go_surf.get_width() // 2, WIN_H // 2 - 70))

    sc_surf = font_sm.render(f"Score: {score}", True, C_TEXT)
    surface.blit(sc_surf, (WIN_W // 2 - sc_surf.get_width() // 2, WIN_H // 2))

    hint = font_sm.render("✌  Peace sign to Restart   |   Q to Quit", True, C_ACCENT)
    surface.blit(hint, (WIN_W // 2 - hint.get_width() // 2, WIN_H // 2 + 50))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Pygame init ──
    pygame.init()
    screen  = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("✋ Gesture-Controlled Dodge Game")
    clock   = pygame.time.Clock()

    # Fonts (fall back gracefully if emoji font unavailable)
    try:
        font_xl = pygame.font.SysFont("segoeuiemoji", 64, bold=True)
        font_lg = pygame.font.SysFont("segoeuiemoji", 42, bold=True)
        font_sm = pygame.font.SysFont("segoeuiemoji", 20)
    except Exception:
        font_xl = pygame.font.SysFont(None, 64, bold=True)
        font_lg = pygame.font.SysFont(None, 42, bold=True)
        font_sm = pygame.font.SysFont(None, 20)

    # ── OpenCV + MediaPipe init ──
    # Auto-detect camera index (works for USB, built-in, or external cameras)
    cap = None
    for cam_index in range(4):
        test = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)  # CAP_DSHOW = faster on Windows
        if test.isOpened() and test.read()[0]:
            cap = test
            print(f"Camera found at index {cam_index}")
            break
        test.release()
    if cap is None:
        print("ERROR: No camera found. Connect your camera and try again.")
        pygame.quit()
        sys.exit()

    detector = HandDetector(max_hands=1, detection_conf=0.7)

    # ── Game state ──
    paddle     = Paddle(x=40)
    ball       = Ball()
    score      = 0
    high_score = 0
    paused     = False
    game_over  = False
    gesture    = "NONE"

    # Debounce open-hand pause toggle
    last_open_time = 0.0

    while True:
        dt = clock.tick(FPS)

        # ── Events ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                cap.release()
                pygame.quit()
                sys.exit()

        # ── Webcam + gesture ──
        ret, frame = cap.read()
        if ret:
            frame    = cv2.flip(frame, 1)          # mirror
            frame    = detector.find_hands(frame, draw=True)
            landmarks = detector.get_landmarks(frame)
            gesture  = detector.get_gesture(landmarks)

        # ── Gesture → game action ──
        if not game_over:
            now = time.time()

            if gesture == "OPEN" and (now - last_open_time) > 1.0:
                paused = not paused
                last_open_time = now

            if not paused:
                if gesture == "UP":
                    paddle.move(-1)
                elif gesture == "DOWN":
                    paddle.move(1)

                # Ball physics
                ball.update()

                # Ball hits right wall → score
                if ball.rect.right >= WIN_W:
                    score += 1
                    ball.speed = min(BALL_SPEED_INIT + score * SPEED_INCREMENT,
                                     BALL_SPEED_MAX)
                    ball.vx = -abs(ball.vx) * (ball.speed / BALL_SPEED_INIT)
                    ball.rect.right = WIN_W - 1

                # Ball exits left wall → game over
                if ball.rect.left <= 0:
                    high_score = max(high_score, score)
                    game_over  = True

                # Paddle deflects ball
                if ball.rect.colliderect(paddle.rect):
                    ball.vx = abs(ball.vx)          # always bounce right
                    # Add slight vertical influence based on hit position
                    rel = (ball.rect.centery - paddle.rect.centery) / (PADDLE_H / 2)
                    ball.vy = rel * ball.speed

        else:
            # Waiting for PEACE gesture to restart
            if gesture == "PEACE":
                paddle    = Paddle(x=40)
                ball      = Ball()
                score     = 0
                paused    = False
                game_over = False

        # ── Draw ──
        screen.fill(C_BG)
        draw_grid(screen)

        # Centre divider
        for y in range(0, WIN_H, 20):
            pygame.draw.rect(screen, (40, 40, 80), (WIN_W // 2 - 2, y, 4, 10))

        paddle.draw(screen)
        ball.draw(screen)

        draw_hud(screen, font_lg, font_sm, score, high_score, gesture, paused)

        if game_over:
            draw_game_over(screen, font_xl, font_sm, score)

        # ── Webcam overlay (bottom-right) ──
        if ret:
            cam_surf = cv_frame_to_pygame(frame, (CAM_W, CAM_H))
            ox = WIN_W - CAM_W - 10
            oy = WIN_H - CAM_H - 10
            # Border
            pygame.draw.rect(screen, C_PADDLE,
                             pygame.Rect(ox - 2, oy - 2, CAM_W + 4, CAM_H + 4),
                             border_radius=6)
            screen.blit(cam_surf, (ox, oy))
            label = font_sm.render("Live Cam", True, C_TEXT)
            screen.blit(label, (ox + 4, oy + 4))

        pygame.display.flip()

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
