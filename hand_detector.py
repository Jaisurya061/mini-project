"""
hand_detector.py
----------------
Reusable Hand Detection Module using MediaPipe Tasks API (v0.10+).
Detects hand landmarks and recognises basic gestures (UP / DOWN / OPEN / PEACE).
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarker
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
import urllib.request
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Drawing connections (21 landmarks, standard MediaPipe hand topology)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (0,9),(9,10),(10,11),(11,12),     # middle
    (0,13),(13,14),(14,15),(15,16),   # ring
    (0,17),(17,18),(18,19),(19,20),   # pinky
    (5,9),(9,13),(13,17),             # palm
]

FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]


def _download_model() -> None:
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~5 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


class HandDetector:
    def __init__(self, max_hands: int = 1, detection_conf: float = 0.7,
                 tracking_conf: float = 0.7):
        _download_model()
        options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self._detector = HandLandmarker.create_from_options(options)
        self._last_landmarks: list[tuple[int, int]] = []

    # ------------------------------------------------------------------
    def find_hands(self, frame: np.ndarray, draw: bool = True) -> np.ndarray:
        """Detect hands, cache landmarks, optionally draw skeleton. Returns frame."""
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_img)

        self._last_landmarks = []
        if result.hand_landmarks:
            h, w = frame.shape[:2]
            lm_list = result.hand_landmarks[0]
            self._last_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]

            if draw and self._last_landmarks:
                self._draw_landmarks(frame, self._last_landmarks)

        return frame

    def _draw_landmarks(self, frame: np.ndarray,
                        pts: list[tuple[int, int]]) -> None:
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 220, 180), 2)
        for i, pt in enumerate(pts):
            r = 5 if i in (4, 8, 12, 16, 20) else 3
            cv2.circle(frame, pt, r, (255, 255, 255), -1)
            cv2.circle(frame, pt, r, (0, 180, 140), 1)

    # ------------------------------------------------------------------
    def get_landmarks(self, frame: np.ndarray,
                      hand_index: int = 0) -> list[tuple[int, int]]:
        """Return cached (x, y) pixel landmarks from last find_hands() call."""
        return self._last_landmarks

    # ------------------------------------------------------------------
    def fingers_up(self, landmarks: list[tuple[int, int]]) -> list[int]:
        if not landmarks:
            return [0, 0, 0, 0, 0]
        fingers = []
        # Thumb: x comparison
        fingers.append(1 if landmarks[4][0] < landmarks[3][0] else 0)
        # Four fingers: tip y < pip y  →  finger up
        for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
            fingers.append(1 if landmarks[tip][1] < landmarks[pip][1] else 0)
        return fingers

    def get_gesture(self, landmarks: list[tuple[int, int]]) -> str:
        if not landmarks:
            return "NONE"
        f = self.fingers_up(landmarks)
        if f == [0, 1, 0, 0, 0]: return "UP"
        if f == [0, 0, 0, 0, 0]: return "DOWN"
        if f == [1, 1, 1, 1, 1]: return "OPEN"
        if f == [0, 1, 1, 0, 0]: return "PEACE"
        return "OTHER"
