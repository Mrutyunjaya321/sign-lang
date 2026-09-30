import cv2
import numpy as np

FRAME_SIZE = (480, 480)
NUM_POINTS = 42
POINTS_PER_HAND = 21

# Standard MediaPipe hand connections (within a single 21-point hand):
# thumb, index, middle, ring, pinky, plus the palm base.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle (+ palm)
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring (+ palm)
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky (+ palm)
    (0, 17),                                  # wrist to pinky base
]


def is_empty(point):
    """Padded/missing points are stored as (0, 0) by extract_keypoints.py."""
    return point[0] == 0 and point[1] == 0


def render_video(keypoints_sequence, output_path, fps=15):
    """
    keypoints_sequence: array of shape (frames, NUM_POINTS*2), values 0-1
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, FRAME_SIZE)

    for frame_points in keypoints_sequence:
        canvas = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)

        points_xy = frame_points.reshape(NUM_POINTS, 2)

        # Convert normalized (0-1) coordinates to actual pixel positions
        pixel_points = []
        for (x, y) in points_xy:
            px = int(x * FRAME_SIZE[0])
            py = int(y * FRAME_SIZE[1])
            pixel_points.append((px, py))

        # Draw each hand separately: hand 1 = points 0-20, hand 2 = points 21-41
        for hand_idx in range(2):
            offset = hand_idx * POINTS_PER_HAND

            # bone lines first, so joint dots draw on top of them
            for (a, b) in HAND_CONNECTIONS:
                pa = points_xy[offset + a]
                pb = points_xy[offset + b]
                if is_empty(pa) or is_empty(pb):
                    continue  # don't draw a line to/from a missing point
                cv2.line(canvas, pixel_points[offset + a], pixel_points[offset + b],
                          (0, 200, 0), 2)

            # joint dots
            for i in range(POINTS_PER_HAND):
                point = points_xy[offset + i]
                if is_empty(point):
                    continue
                cv2.circle(canvas, pixel_points[offset + i], 4, (0, 255, 0), -1)

        writer.write(canvas)

    writer.release()
    print(f"Video saved to {output_path}")
