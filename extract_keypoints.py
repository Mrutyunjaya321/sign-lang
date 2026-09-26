"""
STEP 1 of the pipeline.

Reads video clips organized as:
    data/raw_videos/<WORD>/clip1.mp4
    data/raw_videos/<WORD>/clip2.mp4

For each video, detects hand positions frame-by-frame using MediaPipe,
and saves the coordinates as a .json file in:
    data/keypoints/<WORD>/clip1.json

Run this first, after you've recorded your videos:
    python extract_keypoints.py --input_dir data/raw_videos/train --output_dir data/keypoints/train

NOTE ON MEDIAPIPE VERSION:
This uses the current MediaPipe "Tasks" HandLandmarker API. Older
MediaPipe versions exposed a simpler `mp.solutions.hands` API, but
recent releases removed it -- if you installed mediapipe fresh via
pip, you have the newer version and need the Tasks API used here.

This requires a model file, hand_landmarker.task, in the project
root. Download it once with:

    curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

(PowerShell users can also just paste that URL into a browser and
save the downloaded file as hand_landmarker.task in this folder.)
"""

import os
import json
import argparse
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MAX_POINTS = 42  # 2 hands x 21 landmarks each -- we always pad/trim to this size
MODEL_PATH = "hand_landmarker.task"


def build_detector():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Download it first with:\n"
            f"  curl -o hand_landmarker.task "
            f"https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def extract_from_video(video_path, detector):
    """Opens one video and returns a list of per-frame hand keypoints."""
    cap = cv2.VideoCapture(video_path)
    all_frames_keypoints = []

    while True:
        success, frame = cap.read()
        if not success:
            break  # end of video

        # MediaPipe expects RGB; OpenCV loads frames as BGR by default
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = detector.detect(mp_image)

        frame_points = []
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                for point in hand_landmarks:
                    frame_points.append([point.x, point.y])

        # Always keep exactly MAX_POINTS entries per frame so every
        # frame has the same shape, even if a hand wasn't detected.
        while len(frame_points) < MAX_POINTS:
            frame_points.append([0.0, 0.0])
        frame_points = frame_points[:MAX_POINTS]

        all_frames_keypoints.append(frame_points)

    cap.release()
    return all_frames_keypoints


def process_folder(input_dir, output_dir):
    detector = build_detector()

    for word_name in sorted(os.listdir(input_dir)):
        word_folder = os.path.join(input_dir, word_name)
        if not os.path.isdir(word_folder):
            continue

        out_word_folder = os.path.join(output_dir, word_name)
        os.makedirs(out_word_folder, exist_ok=True)

        for clip_name in sorted(os.listdir(word_folder)):
            if not clip_name.lower().endswith((".mp4", ".mov", ".avi")):
                continue

            clip_path = os.path.join(word_folder, clip_name)
            print(f"Processing {clip_path} ...")

            keypoints = extract_from_video(clip_path, detector)

            out_file = os.path.join(out_word_folder, clip_name.rsplit(".", 1)[0] + ".json")
            with open(out_file, "w") as f:
                json.dump(keypoints, f)

    detector.close()
    print("Done extracting keypoints.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw_videos/train")
    parser.add_argument("--output_dir", default="data/keypoints/train")
    args = parser.parse_args()

    process_folder(args.input_dir, args.output_dir)
