"""Eye Detection Cursor — hands-free mouse control via webcam facial-landmark tracking.

Tracks the right iris (MediaPipe FaceLandmarker landmarks 474-477) to move the cursor
and detects left-eye blinks (landmarks 145 / 159) to click. Uses exponential
smoothing on cursor motion and a non-blocking click debounce.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import mediapipe as mp
import pyautogui

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# MediaPipe FaceLandmarker landmark indices (same as legacy FaceMesh).
RIGHT_IRIS_LANDMARKS = [474, 475, 476, 477]
IRIS_TRACKING_POINT = 475  # single landmark used to drive the cursor

# Left eye vertical landmarks — lower lid (145) and upper lid (159).
LEFT_EYE_LOWER = 145
LEFT_EYE_UPPER = 159

# pyautogui safety: moving cursor to a screen corner aborts the script.
pyautogui.FAILSAFE = False

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eye Detection Cursor")
    p.add_argument(
        "--smoothing",
        type=float,
        default=0.3,
        help="Cursor smoothing factor 0-1 (lower = smoother, higher = more responsive). Default 0.3.",
    )
    p.add_argument(
        "--blink-threshold",
        type=float,
        default=0.005,
        help="Eye-closure threshold in normalized coords. Lower = stricter. Default 0.005.",
    )
    p.add_argument(
        "--click-cooldown",
        type=float,
        default=1.0,
        help="Minimum seconds between clicks. Default 1.0.",
    )
    p.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index. Default 0.",
    )
    p.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirroring of the camera feed.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Show FPS and landmark overlay in the preview window.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"error: model file not found at {MODEL_PATH}", file=sys.stderr)
        print("Download it with:", file=sys.stderr)
        print(f'  curl -L -o "{MODEL_PATH}" '
              '"https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"',
              file=sys.stderr)
        return 1

    cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        print(f"error: could not open camera index {args.camera}", file=sys.stderr)
        return 1

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    screen_w, screen_h = pyautogui.size()

    smoothed_x: float | None = None
    smoothed_y: float | None = None
    last_click_at = 0.0
    last_frame_at = time.time()
    fps = 0.0
    frame_timestamp_ms = 0

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("warning: dropped frame from camera", file=sys.stderr)
                continue

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms += 33  # ~30fps increment
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            frame_h, frame_w, _ = frame.shape

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                # Iris tracking
                tracking_lm = landmarks[IRIS_TRACKING_POINT]
                target_x = tracking_lm.x * screen_w
                target_y = tracking_lm.y * screen_h
                if smoothed_x is None:
                    smoothed_x, smoothed_y = target_x, target_y
                else:
                    smoothed_x += (target_x - smoothed_x) * args.smoothing
                    smoothed_y += (target_y - smoothed_y) * args.smoothing
                pyautogui.moveTo(smoothed_x, smoothed_y)

                if args.debug:
                    for idx in RIGHT_IRIS_LANDMARKS:
                        lm = landmarks[idx]
                        cv2.circle(
                            frame,
                            (int(lm.x * frame_w), int(lm.y * frame_h)),
                            3,
                            (100, 255, 50),
                            -1,
                        )

                # Blink detection
                lower = landmarks[LEFT_EYE_LOWER]
                upper = landmarks[LEFT_EYE_UPPER]
                if args.debug:
                    for lm in (lower, upper):
                        cv2.circle(
                            frame,
                            (int(lm.x * frame_w), int(lm.y * frame_h)),
                            3,
                            (100, 255, 250),
                            -1,
                        )

                eye_gap = abs(lower.y - upper.y)
                now = time.time()
                if eye_gap < args.blink_threshold and (now - last_click_at) > args.click_cooldown:
                    pyautogui.click()
                    last_click_at = now

            if args.debug:
                now = time.time()
                dt = now - last_frame_at
                last_frame_at = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Eye Detection Cursor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or Esc
                break
    except KeyboardInterrupt:
        print("\nstopping…", file=sys.stderr)
    finally:
        cam.release()
        cv2.destroyAllWindows()
        landmarker.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
