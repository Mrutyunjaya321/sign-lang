#!/usr/bin/env python3
"""
Move clips that fail the hand-detection threshold into a rejected/
subfolder inside their own class folder, instead of deleting them.

Reuses the same MediaPipe HandLandmarker (Task API) check as
check_all_clips.py, so a clip that fails here is the same clip that
would show up in check_all_clips.py's "Clips to re-record" list.

Requires hand_landmarker.task in the same folder (same as
debug_single_frame.py and check_all_clips.py).

Usage:
    python move_failed_clips.py
    python move_failed_clips.py --root data/raw_videos --threshold 0.7
    python move_failed_clips.py --dry_run   # preview without moving anything
"""

import argparse
import glob
import os
import shutil

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "hand_landmarker.task"


def check_clip(video_path: str, detector) -> dict:
    """Run detection over every frame of one clip and return stats."""
    cap = cv2.VideoCapture(video_path)
    total_frames = 0
    hand_detected_frames = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        total_frames += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect(mp_image)

        n_hands = len(result.hand_landmarks) if result.hand_landmarks else 0
        if n_hands > 0:
            hand_detected_frames += 1

    cap.release()

    detection_rate = hand_detected_frames / total_frames if total_frames > 0 else 0.0

    return {
        "path": video_path,
        "total_frames": total_frames,
        "hand_detected_frames": hand_detected_frames,
        "detection_rate": detection_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="Move failing clips into a rejected/ subfolder")
    parser.add_argument("--root", type=str, default="data/raw_videos",
                        help="Root folder containing split/class/*.mp4 clips")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Minimum acceptable hand-detection rate (default 0.7)")
    parser.add_argument("--min_confidence", type=float, default=0.5,
                        help="MediaPipe min_hand_detection_confidence (default 0.5, "
                             "matching check_all_clips.py -- not the debug tool's 0.3)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be moved, without moving anything")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found in this folder.")
        print("This must be the same model file debug_single_frame.py uses.")
        return

    pattern = os.path.join(args.root, "**", "*.mp4")
    clips = sorted(glob.glob(pattern, recursive=True))

    # Skip anything already sitting inside a rejected/ folder from a
    # previous run, so re-running this script is safe.
    clips = [c for c in clips if "rejected" not in c.replace("\\", "/").split("/")]

    if not clips:
        print(f"No .mp4 files found under {args.root}")
        return

    print(f"Found {len(clips)} clip(s) under {args.root}")
    print(f"Threshold: {args.threshold:.0%} of frames must have a hand detected")
    print(f"min_hand_detection_confidence: {args.min_confidence}")
    if args.dry_run:
        print("DRY RUN -- nothing will actually be moved\n")
    else:
        print()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=args.min_confidence,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    moved = []
    kept = []

    for clip_path in clips:
        stats = check_clip(clip_path, detector)

        if stats["detection_rate"] >= args.threshold:
            kept.append(stats)
            print(f"[KEEP] {clip_path} ({stats['detection_rate']:.0%})")
            continue

        class_folder = os.path.dirname(clip_path)
        rejected_folder = os.path.join(class_folder, "rejected")
        dest_path = os.path.join(rejected_folder, os.path.basename(clip_path))

        if args.dry_run:
            print(f"[WOULD MOVE] {clip_path} ({stats['detection_rate']:.0%}) -> {dest_path}")
        else:
            os.makedirs(rejected_folder, exist_ok=True)
            shutil.move(clip_path, dest_path)
            print(f"[MOVED] {clip_path} ({stats['detection_rate']:.0%}) -> {dest_path}")

        moved.append(stats)

    detector.close()

    print("\n" + "=" * 60)
    verb = "Would move" if args.dry_run else "Moved"
    print(f"{verb} {len(moved)} clip(s) to rejected/, kept {len(kept)} clip(s)")
    print("=" * 60)

    # Per-class summary, so you can see how many usable clips remain
    # toward your 12-per-sign target.
    by_class = {}
    for r in kept:
        class_name = os.path.basename(os.path.dirname(r["path"]))
        by_class.setdefault(class_name, 0)
        by_class[class_name] += 1

    if by_class:
        print("\nUsable clips remaining per sign:")
        for class_name in sorted(by_class):
            print(f"  {class_name}: {by_class[class_name]} kept")


if __name__ == "__main__":
    main()