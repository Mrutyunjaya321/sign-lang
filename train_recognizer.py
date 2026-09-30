"""
STEP 6 of the pipeline -- NEW, not in the reference project.

Trains the GestureRecognizer on your recorded, extracted keypoint
data. Reuses the exact same dataset.py and data/keypoints/train
folder as train.py (pose generation) -- same data, opposite
direction: here, keypoints are the INPUT and the word is the TARGET.

Run this after extract_keypoints.py has produced your
data/keypoints/train folder:
    python train_recognizer.py

IMPORTANT ON THE PRINTED "train accuracy":
This is accuracy on the SAME clips the model trained on, not a real
held-out test. With very few clips per word it will look
artificially high or unstable -- it only becomes a meaningful
accuracy number once evaluate.py runs this against a separate
data/val/ set the model has never seen.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SignDataset
from gesture_recognizer import GestureRecognizer

KEYPOINTS_DIR = "data/keypoints/train"
CHECKPOINT_PATH = "outputs/checkpoints/gesture_recognizer.pt"
NUM_EPOCHS = 50
BATCH_SIZE = 2
LEARNING_RATE = 3e-4


def train():
    dataset = SignDataset(KEYPOINTS_DIR)
    if len(dataset) == 0:
        print(f"No training data found in {KEYPOINTS_DIR}. "
              f"Run extract_keypoints.py first.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    vocab_size = len(dataset.word_to_idx)
    print(f"Training recognizer on {len(dataset)} clips across {vocab_size} words: "
          f"{list(dataset.word_to_idx.keys())}")

    words_with_data = {w for w, _ in dataset.samples}
    if len(words_with_data) < vocab_size:
        missing = vocab_size - len(words_with_data)
        print(f"NOTE: {missing} word(s) have zero recorded clips so far -- "
              f"the model can't learn those yet.")

    model = GestureRecognizer(vocab_size=vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        correct = 0

        for gloss_ids, keypoints in dataloader:
            optimizer.zero_grad()

            logits = model(keypoints)          # keypoints = input, this direction
            loss = loss_fn(logits, gloss_ids)   # gloss_ids = target, this direction

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted_ids = logits.argmax(dim=1)
            correct += (predicted_ids == gloss_ids).sum().item()

        avg_loss = total_loss / len(dataloader)
        train_accuracy = 100 * correct / len(dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - loss: {avg_loss:.4f} - "
              f"train accuracy: {train_accuracy:.1f}% (NOT held-out accuracy)")

    os.makedirs("outputs/checkpoints", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "word_to_idx": dataset.word_to_idx,
    }, CHECKPOINT_PATH)
    print(f"Model saved to {CHECKPOINT_PATH}")
    print("Remember: the accuracy above is on training data, not a real "
          "test. Build data/val/ and evaluate.py for a trustworthy number.")


if __name__ == "__main__":
    train()