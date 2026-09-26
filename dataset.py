"""
STEP 2 of the pipeline.

Loads the .json keypoint files created by extract_keypoints.py and turns
them into (word, keypoint_sequence) training pairs that PyTorch can use.

You don't run this file directly -- train.py imports it.
"""

import os
import json
import torch
from torch.utils.data import Dataset

MAX_FRAMES = 60   # every clip is padded/trimmed to this many frames
NUM_POINTS = 42   # 2 hands x 21 points
COORD_DIM = 2     # x, y


def load_vocab(keypoints_dir):
    """Builds a mapping like {'HELLO': 0, 'WATER': 1, ...} from folder names."""
    words = sorted(os.listdir(keypoints_dir))
    word_to_idx = {word: idx for idx, word in enumerate(words)}
    return word_to_idx


def load_keypoint_file(path):
    """Loads one .json file and pads/trims it to a fixed length."""
    with open(path, "r") as f:
        data = json.load(f)

    if len(data) > MAX_FRAMES:
        data = data[:MAX_FRAMES]
    while len(data) < MAX_FRAMES:
        data.append([[0.0, 0.0]] * NUM_POINTS)

    return torch.tensor(data, dtype=torch.float32)  # shape: (MAX_FRAMES, NUM_POINTS, COORD_DIM)


class SignDataset(Dataset):
    def __init__(self, keypoints_dir):
        self.samples = []  # list of (word, filepath) pairs
        self.word_to_idx = load_vocab(keypoints_dir)

        for word in os.listdir(keypoints_dir):
            word_folder = os.path.join(keypoints_dir, word)
            if not os.path.isdir(word_folder):
                continue
            for fname in os.listdir(word_folder):
                if fname.endswith(".json"):
                    self.samples.append((word, os.path.join(word_folder, fname)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        word, filepath = self.samples[idx]
        gloss_idx = self.word_to_idx[word]

        keypoints = load_keypoint_file(filepath)
        keypoints = keypoints.view(MAX_FRAMES, NUM_POINTS * COORD_DIM)  # flatten each frame

        return torch.tensor(gloss_idx, dtype=torch.long), keypoints
