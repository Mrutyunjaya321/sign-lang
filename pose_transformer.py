"""
STEP 3 of the pipeline.

Defines the model itself: given a gloss word (as an ID number), predict
a full sequence of keypoint frames representing that sign being performed.

You don't run this file directly -- train.py and main.py import it.
"""

import torch
import torch.nn as nn

MAX_FRAMES = 60
NUM_POINTS = 42
COORD_DIM = 2
OUTPUT_DIM = NUM_POINTS * COORD_DIM  # flattened keypoints per frame


class PoseTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()

        # Turns a word ID (like 3) into a learned vector of numbers
        self.gloss_embedding = nn.Embedding(vocab_size, embed_dim)

        # A learnable "blank sequence" the transformer reshapes into real motion
        self.frame_queries = nn.Parameter(torch.randn(MAX_FRAMES, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Converts the transformer's internal representation into actual (x,y) coordinates
        self.output_layer = nn.Linear(embed_dim, OUTPUT_DIM)

    def forward(self, gloss_ids):
        batch_size = gloss_ids.shape[0]

        word_embed = self.gloss_embedding(gloss_ids)      # (batch, embed_dim)
        word_embed = word_embed.unsqueeze(1)               # (batch, 1, embed_dim)

        frame_seq = self.frame_queries.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch, MAX_FRAMES, embed_dim)

        # Inject the word's meaning into every frame position before processing
        combined = frame_seq + word_embed

        encoded = self.transformer(combined)               # (batch, MAX_FRAMES, embed_dim)
        predicted_keypoints = self.output_layer(encoded)    # (batch, MAX_FRAMES, OUTPUT_DIM)

        return predicted_keypoints
