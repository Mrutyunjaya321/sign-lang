"""
DEBUG TOOL -- not part of the main pipeline.

Scans every frame of one clip, reports what fraction of frames had a
hand detected, and saves the best-detected frame as an image so you
can see exactly what MediaPipe is seeing.

Run:
    python debug_single_frame.py data/raw_videos/train/HELLO/clip_01.mp4

Read the printed detection rate:
    - High (most frames): recording is fine, the earlier all-zero
      JSON was likely from a different/bad clip -- re-run
      extract_keypoints.py on the full folder.
    - Low or 0%: your hand is out of frame, too close/far, or
      lighting is poor for a large chunk of the clip -- re-record
      with the hand centered and fully visible throughout.
"""

import sys
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "hand_landmarker.task"


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_single_frame.py <path_to_video>")
        return

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}")

    if total_frames == 0:
        print("ERROR: video has 0 frames -- file may be corrupt or wrong codec.")
        return

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,  # lowered for this debug test
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    frames_with_hand = 0
    best_frame = None
    best_frame_points = 0
    best_landmarks = None
    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect(mp_image)

        n_hands = len(result.hand_landmarks) if result.hand_landmarks else 0
        if n_hands > 0:
            frames_with_hand += 1
            n_points = sum(len(h) for h in result.hand_landmarks)
            if n_points > best_frame_points:
                best_frame_points = n_points
                best_frame = frame.copy()
                best_landmarks = result.hand_landmarks

        frame_idx += 1

    cap.release()
    detector.close()

    rate = 100 * frames_with_hand / total_frames if total_frames else 0
    print(f"Frames with a hand detected: {frames_with_hand}/{total_frames} ({rate:.0f}%)")

    if best_frame is not None:
        h, w = best_frame.shape[:2]
        for hand in best_landmarks:
            for point in hand:
                x_px = int(point.x * w)
                y_px = int(point.y * h)
                cv2.circle(best_frame, (x_px, y_px), 5, (0, 0, 255), -1)
        cv2.imwrite("debug_frame_output.jpg", best_frame)
        print("Saved debug_frame_output.jpg -- this is the best-detected frame in the clip.")
    else:
        # no detection anywhere -- save the middle frame anyway so you can
        # at least see what the camera captured
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        _, frame = cap.read()
        cap.release()
        if frame is not None:
            cv2.imwrite("debug_frame_output.jpg", frame)
        print("No hand detected in ANY frame of this clip.")
        print("Saved debug_frame_output.jpg (middle frame, no overlay) for reference.")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()