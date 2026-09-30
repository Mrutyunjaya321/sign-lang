#!/usr/bin/env python3
"""
Batch-check hand detection rate for every recorded clip.

Uses the same MediaPipe HandLandmarker (Task API) approach as
debug_single_frame.py, so results are consistent with that tool.

Requires hand_landmarker.task in the same folder (same as
debug_single_frame.py).

Usage:
    python check_all_clips.py
    python check_all_clips.py --root data/raw_videos --threshold 0.7
    python check_all_clips.py --min_confidence 0.3   # match debug tool exactly
"""

import argparse
import glob
import os

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
    parser = argparse.ArgumentParser(description="Batch-check clip hand-detection rate")
    parser.add_argument("--root", type=str, default="data/raw_videos",
                        help="Root folder containing split/class/*.mp4 clips")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Minimum acceptable hand-detection rate (default 0.7)")
    parser.add_argument("--min_confidence", type=float, default=0.5,
                        help="MediaPipe min_hand_detection_confidence. "
                             "debug_single_frame.py uses 0.3 (deliberately "
                             "lowered for debugging); this defaults to the "
                             "stricter 0.5 since this script is the accept/"
                             "reject gate, not a debug tool.")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found in this folder.")
        print("This must be the same model file debug_single_frame.py uses.")
        return

    pattern = os.path.join(args.root, "**", "*.mp4")
    clips = sorted(glob.glob(pattern, recursive=True))

    if not clips:
        print(f"No .mp4 files found under {args.root}")
        print("Check the --root path, or that record_clips.py has saved clips there.")
        return

    print(f"Found {len(clips)} clip(s) under {args.root}")
    print(f"Threshold: {args.threshold:.0%} of frames must have a hand detected")
    print(f"min_hand_detection_confidence: {args.min_confidence}\n")

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=args.min_confidence,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    results = []
    for clip_path in clips:
        stats = check_clip(clip_path, detector)
        results.append(stats)
        status = "PASS" if stats["detection_rate"] >= args.threshold else "FAIL"
        print(
            f"[{status}] {stats['path']} - "
            f"{stats['hand_detected_frames']}/{stats['total_frames']} frames "
            f"({stats['detection_rate']:.0%})"
        )

    detector.close()

    passed = [r for r in results if r["detection_rate"] >= args.threshold]
    failed = [r for r in results if r["detection_rate"] < args.threshold]

    print("\n" + "=" * 60)
    print(f"Summary: {len(passed)}/{len(results)} clips passed ({args.threshold:.0%} threshold)")
    print("=" * 60)

    if failed:
        print("\nClips to re-record:")
        for r in sorted(failed, key=lambda x: x["detection_rate"]):
            print(f"  {r['path']}  ({r['detection_rate']:.0%})")

    # Per-class summary, so you can see at a glance which signs still
    # need more usable clips toward the 12-per-sign target.
    by_class = {}
    for r in results:
        class_name = os.path.basename(os.path.dirname(r["path"]))
        by_class.setdefault(class_name, []).append(r)

    print("\nPer-sign pass count:")
    for class_name in sorted(by_class):
        clips_for_class = by_class[class_name]
        passed_for_class = sum(1 for r in clips_for_class if r["detection_rate"] >= args.threshold)
        print(f"  {class_name}: {passed_for_class}/{len(clips_for_class)} passed")


if __name__ == "__main__":
    main()