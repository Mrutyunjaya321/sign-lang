"""
STEP 5 of the pipeline.

Takes a predicted sequence of keypoints and draws them as a skeleton
video using OpenCV -- dots for joints, saved frame by frame into an mp4.

You don't run this file directly -- main.py imports it.
"""

import cv2
import numpy as np

FRAME_SIZE = (480, 480)
NUM_POINTS = 42


def render_video(keypoints_sequence, output_path, fps=15):
    """
    keypoints_sequence: array of shape (frames, NUM_POINTS*2), values 0-1
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, FRAME_SIZE)

    for frame_points in keypoints_sequence:
        canvas = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)

        points_xy = frame_points.reshape(NUM_POINTS, 2)

        for (x, y) in points_xy:
            px = int(x * FRAME_SIZE[0])
            py = int(y * FRAME_SIZE[1])
            if px == 0 and py == 0:
                continue  # skip empty/padded points
            cv2.circle(canvas, (px, py), 4, (0, 255, 0), -1)

        writer.write(canvas)

    writer.release()
    print(f"Video saved to {output_path}")
