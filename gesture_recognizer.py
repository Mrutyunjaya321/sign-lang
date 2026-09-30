"""
STEP 5 of the pipeline -- NEW, not in the reference project.

Defines the sign-to-text recognition model: given a sequence of hand
keypoints (from a recorded video clip), predict which word it is.

This is the reverse direction of pose_transformer.py -- that model
goes word -> keypoints (generation); this one goes keypoints -> word
(recognition).

You don't run this file directly -- train_recognizer.py and main.py
import it.
"""

import torch
import torch.nn as nn

MAX_FRAMES = 60
NUM_POINTS = 42
COORD_DIM = 2
INPUT_DIM = NUM_POINTS * COORD_DIM  # flattened keypoints per frame


class GestureRecognizer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, num_layers=2):
        super().__init__()

        # Reads the keypoint sequence frame-by-frame, building up a
        # running summary of the motion as it goes.
        self.lstm = nn.LSTM(
            input_size=INPUT_DIM,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )

        # Turns the LSTM's final summary into a score for each word in
        # the vocabulary.
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, keypoints):
        # keypoints shape: (batch, MAX_FRAMES, INPUT_DIM)
        _, (last_hidden, _) = self.lstm(keypoints)

        # last_hidden shape: (num_layers, batch, hidden_dim) -- take the
        # final layer's hidden state as the summary of the whole clip
        summary = last_hidden[-1]  # (batch, hidden_dim)

        logits = self.classifier(summary)  # (batch, vocab_size)
        return logits