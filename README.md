# Gesture-Controlled Game using OpenCV and MediaPipe

> **Mini Project Report** | Artificial Intelligence & Machine Learning  
> Language: Python 3.10+ | Domain: Computer Vision / Human-Computer Interaction

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Hardware Requirements](#4-hardware-requirements)
5. [Software Requirements](#5-software-requirements)
6. [System Architecture](#6-system-architecture)
7. [Working Principle](#7-working-principle)
8. [Algorithm / Steps](#8-algorithm--steps)
9. [Implementation](#9-implementation)
10. [Simple Game Example](#10-simple-game-example)
11. [Output Description](#11-output-description)
12. [Advantages and Limitations](#12-advantages-and-limitations)
13. [Future Enhancements](#13-future-enhancements)
14. [Conclusion](#14-conclusion)

---

## 1. Introduction

Human-Computer Interaction (HCI) has evolved far beyond keyboards and mice.
Today, computers can understand natural human gestures through a regular webcam —
no special gloves, sensors, or expensive hardware required.

**Gesture recognition** is the process of identifying meaningful human movements
(usually of the hands or face) and translating them into commands that a computer
can act upon. This technology is already used in:

- Sign-language translation apps
- Touchless kiosk interfaces in hospitals
- AR/VR controllers
- Accessibility tools for differently-abled users

This project demonstrates a **real-time, gesture-controlled dodge game** built
entirely in Python using two powerful open-source libraries:

| Library | Role |
|---------|------|
| **OpenCV** | Captures and processes webcam frames |
| **MediaPipe** | Detects and tracks 21 hand landmarks per frame |
| **Pygame** | Renders the interactive game window |

A player moves a paddle up or down purely by showing hand gestures to the webcam —
no keyboard or mouse needed.

---

## 2. Problem Statement

Traditional game controllers require physical contact and can be inaccessible to
people with motor disabilities. Keyboard-based controls also limit the naturalness
of interaction. There is a need for a **low-cost, contactless control mechanism**
that uses a standard webcam to interpret hand gestures and map them to game actions
in real time.

---

## 3. Objectives

- Capture live video from a webcam using OpenCV.
- Detect and track hand landmarks in real time using MediaPipe.
- Classify hand gestures (UP, DOWN, OPEN, FIST, PEACE) from landmark positions.
- Map gestures to game controls (move paddle, pause, restart).
- Build an interactive Pygame-based dodge game controlled entirely by gestures.
- Display the webcam feed as a live overlay inside the game window.

---

## 4. Hardware Requirements

| Component | Minimum Specification |
|-----------|-----------------------|
| Processor | Intel Core i3 / AMD Ryzen 3 (or better) |
| RAM | 4 GB (8 GB recommended) |
| Webcam | 720p USB or built-in laptop camera |
| Display | 1024 × 768 or higher resolution |
| Storage | 500 MB free disk space |
| GPU | Not required (CPU inference is sufficient) |

> **Note:** MediaPipe runs its hand-detection model entirely on the CPU using
> optimised TFLite kernels, so no GPU is needed.

---

## 5. Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10 or 3.11 | Core programming language |
| OpenCV (`opencv-python`) | ≥ 4.8 | Webcam capture & image processing |
| MediaPipe | ≥ 0.10 | Hand landmark detection |
| Pygame | ≥ 2.5 | Game rendering & event loop |
| NumPy | ≥ 1.24 | Array operations for frame conversion |
| pip | Latest | Package installation |

### Installation

```bash
pip install -r requirements.txt
```

---

## 6. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SYSTEM PIPELINE                          │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Webcam  │───▶│   OpenCV     │───▶│  MediaPipe Hands     │  │
│  │ (Camera) │    │ (BGR Frame)  │    │  (21 Landmarks)      │  │
│  └──────────┘    └──────────────┘    └──────────┬───────────┘  │
│                                                 │               │
│                                    ┌────────────▼────────────┐  │
│                                    │   HandDetector Module   │  │
│                                    │  • fingers_up()         │  │
│                                    │  • get_gesture()        │  │
│                                    └────────────┬────────────┘  │
│                                                 │               │
│                                    ┌────────────▼────────────┐  │
│                                    │   Gesture → Action Map  │  │
│                                    │  UP    → paddle moves ↑ │  │
│                                    │  DOWN  → paddle moves ↓ │  │
│                                    │  OPEN  → pause/resume   │  │
│                                    │  PEACE → restart        │  │
│                                    └────────────┬────────────┘  │
│                                                 │               │
│                                    ┌────────────▼────────────┐  │
│                                    │     Pygame Game Loop    │  │
│                                    │  • Ball physics         │  │
│                                    │  • Score tracking       │  │
│                                    │  • HUD + cam overlay    │  │
│                                    └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Explanation

**OpenCV** reads one frame at a time from the webcam (typically at 30 fps).
Each frame is a NumPy array of shape `(height, width, 3)` in BGR colour order.

**MediaPipe Hands** converts the frame to RGB, runs a palm-detection model to
locate hands, then runs a landmark model to place 21 keypoints on each detected
hand. Each keypoint has normalised (x, y, z) coordinates.

**HandDetector** (our custom module) converts normalised coordinates to pixel
positions, checks which fingers are extended by comparing tip vs. PIP joint
y-coordinates, and returns a named gesture string.

**Pygame** runs the main game loop at 60 fps, reads the latest gesture string,
updates game objects (paddle, ball), and renders everything to the screen
including a small live webcam overlay.

---

## 7. Working Principle

### Hand Landmark Model

MediaPipe places **21 landmarks** on each detected hand:

```
                 8   12  16  20
                 |   |   |   |
             7   |   |   |   19
             |  6|  10  14  18
             5   |   |   |   |
              \  9  11  13  17
               \ |   |   |   |
                4|   |   |   |
                 3   |   |   |
                  \  |   |   |
                   2 |   |   |
                    \|   |   |
                     1   |   |
                      \  |   |
                       0─────── (wrist)
```

Key landmark indices used in this project:

| Landmark | Index | Meaning |
|----------|-------|---------|
| Wrist | 0 | Base reference |
| Thumb tip | 4 | Thumb extended check |
| Index tip | 8 | Index finger extended |
| Middle tip | 12 | Middle finger extended |
| Ring tip | 16 | Ring finger extended |
| Pinky tip | 20 | Pinky finger extended |

### Finger Extension Logic

A finger is considered **extended (up)** when its **tip landmark** has a smaller
y-coordinate than its **PIP (middle) joint** — because y increases downward in
image coordinates.

```
  y=100  ← tip (finger up, smaller y)
  y=150  ← PIP joint
  y=200  ← MCP joint
```

So: `tip.y < pip.y` → finger is UP.

### Gesture Table

| Gesture | Thumb | Index | Middle | Ring | Pinky | Action |
|---------|-------|-------|--------|------|-------|--------|
| UP | 0 | 1 | 0 | 0 | 0 | Move paddle up |
| DOWN (Fist) | 0 | 0 | 0 | 0 | 0 | Move paddle down |
| OPEN | 1 | 1 | 1 | 1 | 1 | Pause / Resume |
| PEACE | 0 | 1 | 1 | 0 | 0 | Restart game |

---

## 8. Algorithm / Steps

```
START
│
├─ 1. Initialise OpenCV VideoCapture (webcam index 0)
├─ 2. Initialise MediaPipe Hands model
├─ 3. Initialise Pygame window, fonts, clock
├─ 4. Create Paddle and Ball objects
│
└─ LOOP (while game is running):
    │
    ├─ 5. Read frame from webcam
    ├─ 6. Flip frame horizontally (mirror effect)
    ├─ 7. Pass frame to MediaPipe → get 21 landmarks
    ├─ 8. For each finger: check if tip.y < pip.y → finger UP
    ├─ 9. Map finger states → gesture string (UP/DOWN/OPEN/PEACE)
    │
    ├─ 10. If gesture == OPEN → toggle pause (with 1s debounce)
    ├─ 11. If not paused:
    │       ├─ gesture UP   → paddle.y -= PADDLE_SPEED
    │       ├─ gesture DOWN → paddle.y += PADDLE_SPEED
    │       ├─ Update ball position (vx, vy)
    │       ├─ Bounce ball off top/bottom walls
    │       ├─ If ball hits paddle → reverse vx, adjust vy
    │       ├─ If ball exits right  → score++, increase speed
    │       └─ If ball exits left   → GAME OVER
    │
    ├─ 12. If GAME OVER and gesture == PEACE → reset game
    │
    ├─ 13. Draw background grid, paddle, ball, HUD, cam overlay
    └─ 14. pygame.display.flip() → show frame
│
END (Q key or window close)
```

---

## 9. Implementation

The project is split into two Python files for clarity and reusability.

### File Structure

```
New folder/
├── hand_detector.py   ← Reusable gesture detection module
├── gesture_game.py    ← Main game (Pygame + OpenCV integration)
├── requirements.txt   ← Python dependencies
└── README.md          ← This report
```

### `hand_detector.py` — Key Methods

```python
# Check which fingers are extended
def fingers_up(self, landmarks) -> list[int]:
    fingers = []
    # Thumb: compare x (horizontal)
    fingers.append(1 if landmarks[4][0] < landmarks[3][0] else 0)
    # Four fingers: compare y (vertical)
    for tip, pip in zip([8,12,16,20], [6,10,14,18]):
        fingers.append(1 if landmarks[tip][1] < landmarks[pip][1] else 0)
    return fingers  # e.g. [0, 1, 0, 0, 0] = index finger only

# Map finger states to gesture name
def get_gesture(self, landmarks) -> str:
    f = self.fingers_up(landmarks)
    if f == [0,1,0,0,0]: return "UP"
    if f == [0,0,0,0,0]: return "DOWN"
    if f == [1,1,1,1,1]: return "OPEN"
    if f == [0,1,1,0,0]: return "PEACE"
    return "OTHER"
```

### `gesture_game.py` — Game Loop Core

```python
# Read webcam frame and detect gesture
ret, frame = cap.read()
frame      = cv2.flip(frame, 1)
frame      = detector.find_hands(frame, draw=True)
landmarks  = detector.get_landmarks(frame)
gesture    = detector.get_gesture(landmarks)

# Apply gesture to game
if gesture == "UP":
    paddle.move(-1)          # move up
elif gesture == "DOWN":
    paddle.move(1)           # move down
elif gesture == "OPEN":
    paused = not paused      # toggle pause
elif gesture == "PEACE" and game_over:
    reset_game()             # restart
```

---

## 10. Simple Game Example

### Game: Gesture Dodge

The player controls a **vertical paddle** on the left side of the screen.
A ball bounces around the window. The goal is to **deflect the ball** with the
paddle as many times as possible without letting it exit through the left wall.

```
┌──────────────────────────────────────────────────────┐
│  ☝ UP    ✊ DOWN    🖐 PAUSE    ✌ RESTART            │
│                                                      │
│  ║                    ●                              │
│  ║  ←── Paddle        ↗  Ball                       │
│  ║                                                   │
│                    Score: 007                        │
│                                          ┌─────────┐ │
│                                          │ Live Cam│ │
│                                          └─────────┘ │
└──────────────────────────────────────────────────────┘
```

### Scoring Rules

| Event | Score Change |
|-------|-------------|
| Ball reaches right wall | +1 point |
| Ball exits left wall | Game Over |
| Each point scored | Ball speed increases by 0.4 |

### Gesture Controls Summary

| Show this gesture | Effect |
|-------------------|--------|
| ☝ Index finger up | Paddle moves UP |
| ✊ Closed fist | Paddle moves DOWN |
| 🖐 Open hand (all 5 fingers) | Pause / Resume |
| ✌ Peace sign (index + middle) | Restart after Game Over |

---

## 11. Output Description

When you run `python gesture_game.py`, you will see:

1. **Game Window (900 × 600 px)**
   - Dark grid background with a subtle dot pattern
   - A cyan rounded paddle on the left
   - A red ball with a motion trail
   - Score displayed in gold at the top centre
   - Best score shown below the current score
   - Gesture badge (top-left) showing the currently detected gesture in colour
   - Controls legend (bottom-left)

2. **Live Webcam Overlay (240 × 180 px, bottom-right)**
   - Shows your hand with MediaPipe skeleton drawn on it
   - Cyan border around the overlay
   - Updates in real time at ~30 fps

3. **Pause Screen**
   - Semi-transparent dark overlay
   - "⏸ PAUSED" text in gold

4. **Game Over Screen**
   - "GAME OVER" in red
   - Final score displayed
   - Instruction to show ✌ to restart

### Console Output (none by default)
The game runs silently. If the webcam cannot be opened, Python will raise a
`cv2.error` — ensure no other application is using the camera.

---

## 12. Advantages and Limitations

### Advantages

| # | Advantage |
|---|-----------|
| 1 | **No special hardware** — works with any standard webcam |
| 2 | **Contactless** — hygienic, useful in public kiosks |
| 3 | **Real-time** — MediaPipe runs at 20–30 fps on a mid-range CPU |
| 4 | **Accessible** — can be adapted for users with limited mobility |
| 5 | **Extensible** — easy to add new gestures or map to different games |
| 6 | **Cross-platform** — runs on Windows, macOS, and Linux |

### Limitations

| # | Limitation |
|---|-----------|
| 1 | **Lighting sensitivity** — poor lighting reduces detection accuracy |
| 2 | **Single hand only** — current setup tracks one hand at a time |
| 3 | **Background clutter** — complex backgrounds can confuse the model |
| 4 | **Gesture ambiguity** — some gestures look similar (e.g. "OTHER") |
| 5 | **Latency** — on very slow CPUs, frame processing may lag |
| 6 | **No depth** — 2D gestures only; no 3D spatial control |

---

## 13. Future Enhancements

1. **Two-hand control** — use left hand for one player, right hand for another
   (multiplayer mode).

2. **More gestures** — add swipe detection using wrist velocity for faster
   paddle movement.

3. **3D gesture support** — use MediaPipe's z-coordinate (depth) to detect
   push/pull gestures.

4. **Voice + gesture fusion** — combine speech commands with hand gestures for
   richer control.

5. **Mobile deployment** — port to Android using MediaPipe's mobile SDK and
   Flutter for the UI.

6. **Sign-language game** — create an educational game where players spell words
   using ASL hand signs.

7. **Difficulty levels** — dynamically adjust ball speed and paddle size based
   on player performance.

8. **Online leaderboard** — store high scores in AWS DynamoDB and display a
   global leaderboard.

---

## 14. Conclusion

This project successfully demonstrates how **computer vision and machine learning**
can be combined to create an engaging, contactless gaming experience using only a
standard webcam and free open-source libraries.

Key takeaways:

- **MediaPipe** provides a robust, pre-trained hand landmark model that runs
  efficiently on CPU, making real-time gesture recognition accessible to everyone.
- **OpenCV** handles all low-level camera I/O and image preprocessing with a
  simple, consistent API.
- **Pygame** provides a lightweight game engine that integrates cleanly with
  NumPy arrays produced by OpenCV.
- The modular design (`hand_detector.py` + `gesture_game.py`) makes it easy to
  reuse the gesture detection logic in any other Python project.

Gesture-controlled interfaces represent the future of natural human-computer
interaction — from gaming and accessibility tools to industrial control panels
and medical devices. This mini project is a solid foundation for exploring that
exciting space.

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the game
python gesture_game.py
```

> Make sure your webcam is connected and not in use by another application.  
> Press **Q** or close the window to exit.

---

*Project developed using Python 3.11, OpenCV 4.9, MediaPipe 0.10, Pygame 2.5*
