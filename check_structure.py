"""
CHECK TOOL -- verifies your project's file/folder structure is correct.

Run any time you want to confirm nothing is missing or misplaced:
    python check_structure.py
"""

import os
from setup_vocab import VOCABULARY

REQUIRED_FILES = [
    "main.py",
    "gloss_dict.py",
    "extract_keypoints.py",
    "dataset.py",
    "pose_transformer.py",
    "train.py",
    "render.py",
    "gesture_recognizer.py",
    "train_recognizer.py",
    "setup_vocab.py",
    "record_clips.py",
    "requirements.txt",
    "hand_landmarker.task",
]

OPTIONAL_FILES = [
    "debug_single_frame.py",
    "check_all_clips.py",
    "README.md",
]

REQUIRED_CHECKPOINTS = [
    "outputs/checkpoints/pose_transformer.pt",
    "outputs/checkpoints/gesture_recognizer.pt",
]


def check_files():
    print("=== Core project files ===")
    all_ok = True
    for f in REQUIRED_FILES:
        exists = os.path.isfile(f)
        status = "OK" if exists else "MISSING"
        if not exists:
            all_ok = False
        print(f"  [{status:^7}] {f}")

    print("\n=== Optional/debug files ===")
    for f in OPTIONAL_FILES:
        exists = os.path.isfile(f)
        status = "present" if exists else "not found"
        print(f"  [{status:^9}] {f}")

    return all_ok


def check_checkpoints():
    print("\n=== Trained model checkpoints ===")
    for f in REQUIRED_CHECKPOINTS:
        exists = os.path.isfile(f)
        status = "OK" if exists else "not trained yet"
        print(f"  [{status:^15}] {f}")


def check_data_folders():
    print("\n=== Dataset folders (train) ===")
    raw_dir = "data/raw_videos/train"
    kp_dir = "data/keypoints/train"

    if not os.path.isdir(raw_dir):
        print(f"  MISSING: {raw_dir}")
        return

    total_clips = 0
    total_json = 0
    words_with_data = 0

    print(f"  {'Word':<12} {'Raw clips':>10} {'Keypoint JSON':>15}")
    print("  " + "-" * 40)

    for word in VOCABULARY:
        word_raw = os.path.join(raw_dir, word)
        word_kp = os.path.join(kp_dir, word)

        n_raw = 0
        if os.path.isdir(word_raw):
            n_raw = len([f for f in os.listdir(word_raw) if f.lower().endswith(".mp4")])

        n_kp = 0
        if os.path.isdir(word_kp):
            n_kp = len([f for f in os.listdir(word_kp) if f.lower().endswith(".json")])

        total_clips += n_raw
        total_json += n_kp
        if n_raw > 0:
            words_with_data += 1

        flag = ""
        if n_raw > 0 and n_kp == 0:
            flag = "<- raw clips exist but not extracted yet"
        elif n_raw != n_kp and n_raw > 0:
            flag = "<- mismatch, re-run extract_keypoints.py"

        print(f"  {word:<12} {n_raw:>10} {n_kp:>15}  {flag}")

    print("  " + "-" * 40)
    print(f"  Total: {total_clips} raw clips, {total_json} keypoint files, "
          f"{words_with_data}/{len(VOCABULARY)} words started")

    val_dir = "data/val"
    if os.path.isdir(val_dir):
        print(f"\n  data/val/ folder exists (validation set)")
    else:
        print(f"\n  data/val/ folder not created yet (needed later for real accuracy)")


if __name__ == "__main__":
    files_ok = check_files()
    check_checkpoints()
    check_data_folders()

    print("\n=== Summary ===")
    if files_ok:
        print("All core files present.")
    else:
        print("Some core files are MISSING -- check the list above.")