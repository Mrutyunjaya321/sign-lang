"""
STEP 4 of the pipeline.

Trains the PoseTransformer on your recorded, extracted keypoint data.

Run this after extract_keypoints.py has produced your data/keypoints/train folder:
    python train.py

Leave this running -- for a small vocabulary on a laptop, expect it to
take anywhere from under an hour to overnight depending on data size.
"""

import os
import torch
from torch.utils.data import DataLoader
from dataset import SignDataset
from pose_transformer import PoseTransformer

KEYPOINTS_DIR = "data/keypoints/train"
CHECKPOINT_PATH = "outputs/checkpoints/pose_transformer.pt"
NUM_EPOCHS = 50
BATCH_SIZE = 2
LEARNING_RATE = 3e-4


def mpjpe_loss(predicted, target):
    """
    Mean Per Joint Position Error: average distance between predicted
    and real joint positions, across all joints and frames.
    """
    diff = predicted - target
    per_point_error = torch.sqrt((diff ** 2) + 1e-8)  # +1e-8 avoids sqrt(0) issues
    return per_point_error.mean()


def train():
    dataset = SignDataset(KEYPOINTS_DIR)
    if len(dataset) == 0:
        print(f"No training data found in {KEYPOINTS_DIR}. "
              f"Run extract_keypoints.py first.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    vocab_size = len(dataset.word_to_idx)
    print(f"Training on {len(dataset)} clips across {vocab_size} words: "
          f"{list(dataset.word_to_idx.keys())}")

    model = PoseTransformer(vocab_size=vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0

        for gloss_ids, real_keypoints in dataloader:
            optimizer.zero_grad()

            predicted = model(gloss_ids)
            loss = mpjpe_loss(predicted, real_keypoints)

            loss.backward()      # calculates how to adjust every weight
            optimizer.step()     # actually applies that adjustment

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - MPJPE loss: {avg_loss:.4f}")

    os.makedirs("outputs/checkpoints", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "word_to_idx": dataset.word_to_idx,
    }, CHECKPOINT_PATH)
    print(f"Model saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
