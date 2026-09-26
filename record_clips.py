"""
STEP 2 -- record clips.

An OpenCV-based recorder that saves straight into the folder structure
setup_vocab.py created, with correct auto-incrementing filenames, so
you never have to manually rename or sort files.

Run:
    python record_clips.py

Controls (shown on screen too):
    SPACE   start / stop recording the current word
    N       next word
    P       previous word
    V       toggle TRAIN / VAL target folder
    Q       quit

Recording behaviour:
    - Recording auto-stops after MAX_CLIP_SECONDS, or press SPACE
      again to stop early.
    - There's a short countdown before recording starts, so you have
      time to get into position after pressing SPACE.
"""

import os
import time
import cv2

from setup_vocab import VOCABULARY, TARGET_CLIPS_PER_WORD

MAX_CLIP_SECONDS = 3          # auto-stop after this long
COUNTDOWN_SECONDS = 2         # pause before recording starts
FPS = 24
FRAME_SIZE = (960, 720)

BASE_DIRS = {
    "TRAIN": "data/raw_videos/train",
    "VAL": "data/val/raw_videos",
}


def existing_clip_count(word, mode):
    folder = os.path.join(BASE_DIRS[mode], word)
    os.makedirs(folder, exist_ok=True)
    return len([f for f in os.listdir(folder) if f.lower().endswith(".mp4")])


def next_clip_path(word, mode):
    n = existing_clip_count(word, mode) + 1
    folder = os.path.join(BASE_DIRS[mode], word)
    return os.path.join(folder, f"clip_{n:02d}.mp4")


def draw_overlay(frame, word, mode, count, target, recording, countdown_left=0):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)

    status = f"{mode}  |  Word: {word}  ({count}/{target} recorded)"
    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if countdown_left > 0:
        cv2.putText(frame, f"Starting in {countdown_left}...", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    elif recording:
        cv2.circle(frame, (20, 55), 8, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "SPACE=record  N/P=word  V=train/val  Q=quit",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


def record_clip(cap, word, mode):
    path = next_clip_path(word, mode)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, FPS, FRAME_SIZE)

    start = time.time()
    while time.time() - start < MAX_CLIP_SECONDS:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, FRAME_SIZE)
        writer.write(frame)

        display = frame.copy()
        count = existing_clip_count(word, mode)
        draw_overlay(display, word, mode, count, TARGET_CLIPS_PER_WORD, recording=True)
        cv2.imshow("record_clips", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") or key == ord("q"):
            break

    writer.release()
    print(f"Saved: {path}")


def countdown(cap, word, mode):
    for i in range(COUNTDOWN_SECONDS, 0, -1):
        ok, frame = cap.read()
        if not ok:
            return
        frame = cv2.resize(frame, FRAME_SIZE)
        count = existing_clip_count(word, mode)
        draw_overlay(frame, word, mode, count, TARGET_CLIPS_PER_WORD,
                     recording=False, countdown_left=i)
        cv2.imshow("record_clips", frame)
        cv2.waitKey(1000)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check camera permissions / index.")
        return

    word_idx = 0
    mode = "TRAIN"

    print("Recording", VOCABULARY[word_idx], f"({mode})")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, FRAME_SIZE)

        word = VOCABULARY[word_idx]
        count = existing_clip_count(word, mode)
        draw_overlay(frame, word, mode, count, TARGET_CLIPS_PER_WORD, recording=False)
        cv2.imshow("record_clips", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("n"):
            word_idx = (word_idx + 1) % len(VOCABULARY)
            print("Now recording:", VOCABULARY[word_idx], f"({mode})")
        elif key == ord("p"):
            word_idx = (word_idx - 1) % len(VOCABULARY)
            print("Now recording:", VOCABULARY[word_idx], f"({mode})")
        elif key == ord("v"):
            mode = "VAL" if mode == "TRAIN" else "TRAIN"
            print("Switched to:", mode)
        elif key == ord(" "):
            countdown(cap, VOCABULARY[word_idx], mode)
            record_clip(cap, VOCABULARY[word_idx], mode)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()