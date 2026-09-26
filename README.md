# Text-to-Sign Language Project — Run Guide

## 1. Install everything
```
python -m venv venv
venv\Scripts\activate        (Windows)
source venv/bin/activate     (Mac/Linux)
pip install -r requirements.txt
```

## 2. Record your videos
Create this folder structure yourself and put your recordings in it:
```
data/raw_videos/train/HELLO/clip1.mp4
data/raw_videos/train/HELLO/clip2.mp4
data/raw_videos/train/WATER/clip1.mp4
...
```
Record each word 10-15 times for decent results. Folder names must be
in CAPS with underscores instead of spaces (e.g. THANK_YOU).

## 3. Extract keypoints
```
python extract_keypoints.py --input_dir data/raw_videos/train --output_dir data/keypoints/train
```

## 4. Train the model
```
python train.py
```
Watch the printed MPJPE loss — it should go down over the epochs.

## 5. Update the gloss dictionary
Open `gloss_dict.py` and make sure every word you trained has an entry
matching its exact folder name.

## 6. Run it
```
python main.py
```
Type a sentence. Check `outputs/videos/output.mp4` for the result.

## Notes
- This is a simplified, from-scratch version built for learning and a
  minor project — not the full research-grade pipeline.
- Coordinates are drawn as dots only (no connecting lines) to keep the
  rendering code simple; you can extend `render.py` to draw lines
  between specific point indices if you want a more skeleton-like look.
- If a sentence uses an untrained word, `main.py` will tell you it's
  not recognized instead of crashing.
