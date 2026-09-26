"""
STEP 6 -- the final step.

Run this after training is done:
    python main.py

Type a sentence, and it will:
  1. Look up its gloss word using gloss_dict.py
  2. Feed that gloss into the trained model to predict keypoints
  3. Render those keypoints into a skeleton video using render.py
"""

import os
import torch
from gloss_dict import sentence_to_gloss
from pose_transformer import PoseTransformer
from render import render_video

CHECKPOINT_PATH = "outputs/checkpoints/pose_transformer.pt"
OUTPUT_VIDEO_PATH = "outputs/videos/output.mp4"


def load_model():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    word_to_idx = checkpoint["word_to_idx"]

    model = PoseTransformer(vocab_size=len(word_to_idx))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()  # switch to prediction mode, not training mode

    return model, word_to_idx


def run(sentence):
    if not os.path.exists(CHECKPOINT_PATH):
        print("No trained model found. Run train.py first.")
        return

    model, word_to_idx = load_model()

    gloss = sentence_to_gloss(sentence)
    if gloss is None:
        print(f"Sorry, '{sentence}' is not in the trained vocabulary.")
        return

    if gloss not in word_to_idx:
        print(f"Gloss '{gloss}' has no trained model data yet.")
        return

    gloss_id = torch.tensor([word_to_idx[gloss]], dtype=torch.long)

    with torch.no_grad():  # no training happening here, so skip gradient tracking
        predicted = model(gloss_id)[0]  # shape: (frames, points*2)

    os.makedirs("outputs/videos", exist_ok=True)
    render_video(predicted.numpy(), OUTPUT_VIDEO_PATH)


if __name__ == "__main__":
    sentence = input("Enter a sentence: ")
    run(sentence)
