"""
DEBUG TOOL -- inspects what the trained pose model is actually
predicting, to check whether a black output video is a rendering
bug or just an undertrained model.

Run:
    python debug_generation.py hello
"""

import sys
import torch
from gloss_dict import sentence_to_gloss
from pose_transformer import PoseTransformer

CHECKPOINT_PATH = "outputs/checkpoints/pose_transformer.pt"


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_generation.py <word>")
        return

    word = sys.argv[1]

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    word_to_idx = checkpoint["word_to_idx"]

    model = PoseTransformer(vocab_size=len(word_to_idx))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    gloss = sentence_to_gloss(word)
    if gloss is None or gloss not in word_to_idx:
        print(f"'{word}' not found in trained vocabulary.")
        return

    gloss_id = torch.tensor([word_to_idx[gloss]], dtype=torch.long)

    with torch.no_grad():
        predicted = model(gloss_id)[0]  # shape: (frames, 84)

    values = predicted.numpy()

    print(f"Predicted shape: {values.shape}")
    print(f"Min value: {values.min():.4f}")
    print(f"Max value: {values.max():.4f}")
    print(f"Mean value: {values.mean():.4f}")

    # how many (x,y) pairs round to exactly (0,0)-ish per frame, on average
    reshaped = values.reshape(values.shape[0], -1, 2)
    near_zero = ((abs(reshaped[:, :, 0]) < 0.01) & (abs(reshaped[:, :, 1]) < 0.01)).sum()
    total_points = reshaped.shape[0] * reshaped.shape[1]
    print(f"Points near (0,0): {near_zero}/{total_points} "
          f"({100*near_zero/total_points:.0f}%)")

    print(f"\nFirst frame's first 5 points:\n{reshaped[0][:5]}")


if __name__ == "__main__":
    main()