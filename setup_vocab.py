"""
STEP 1 -- vocabulary setup.

Run this once, before recording anything:
    python setup_vocab.py

It creates the exact folder structure extract_keypoints.py expects,
one folder per word under data/raw_videos/train/, and tells you how
many clips you still need to record for each word.

Re-run it any time -- it's safe: it never deletes existing clips,
it just creates missing folders and re-counts what's already there.
"""

import os

# The 16-word vocabulary for this project.
# Keep this list as the single source of truth -- gloss_dict.py
# is generated from it further down, so you only maintain one list.
VOCABULARY = [
    "HELLO",
    "THANK_YOU",
    "PLEASE",
    "SORRY",
    "HELP",
    "WATER",
    "STOP",
    "YES",
    "NO",
    "GOOD",
    "BAD",
    "NAME",
    "ME",
    "YOU",
    "EAT",
    "MORE",
]

RAW_VIDEOS_DIR = "data/raw_videos/train"
TARGET_CLIPS_PER_WORD = 12  # aim for 10-15; this is the checklist target


def create_folders():
    os.makedirs(RAW_VIDEOS_DIR, exist_ok=True)
    for word in VOCABULARY:
        word_folder = os.path.join(RAW_VIDEOS_DIR, word)
        os.makedirs(word_folder, exist_ok=True)
    print(f"Created/verified {len(VOCABULARY)} word folders under {RAW_VIDEOS_DIR}/")


def count_existing_clips():
    """Counts how many .mp4/.mov/.avi files already exist per word."""
    counts = {}
    for word in VOCABULARY:
        word_folder = os.path.join(RAW_VIDEOS_DIR, word)
        if not os.path.isdir(word_folder):
            counts[word] = 0
            continue
        clips = [
            f for f in os.listdir(word_folder)
            if f.lower().endswith((".mp4", ".mov", ".avi"))
        ]
        counts[word] = len(clips)
    return counts


def print_progress(counts):
    print("\nRecording progress:")
    print(f"{'Word':<12} {'Recorded':>9} {'Target':>7}  Status")
    print("-" * 45)

    total_recorded = 0
    total_target = len(VOCABULARY) * TARGET_CLIPS_PER_WORD

    for word in VOCABULARY:
        n = counts[word]
        total_recorded += n
        status = "OK" if n >= TARGET_CLIPS_PER_WORD else f"need {TARGET_CLIPS_PER_WORD - n} more"
        print(f"{word:<12} {n:>9} {TARGET_CLIPS_PER_WORD:>7}  {status}")

    print("-" * 45)
    print(f"Total: {total_recorded}/{total_target} clips recorded "
          f"across {len(VOCABULARY)} words.")


def sync_gloss_dict():
    """
    Writes gloss_dict.py so its GLOSS_DICT matches VOCABULARY exactly.
    Overwrites the existing sample file -- back it up first if you
    made manual edits to it you want to keep.
    """
    lines = [
        '"""',
        "This replaces \"Stage 1\" (text-to-gloss) with a simple hand-written",
        "dictionary -- a reasonable shortcut for a minor project.",
        "",
        "Generated from setup_vocab.py -- edit VOCABULARY there, not the",
        "dict below, then re-run setup_vocab.py to regenerate this file.",
        '"""',
        "",
        "GLOSS_DICT = {",
    ]
    for word in VOCABULARY:
        phrase = word.replace("_", " ").lower()
        lines.append(f'    "{phrase}": "{word}",')
    lines.append("}")
    lines.append("")
    lines.append("")
    lines.append("def sentence_to_gloss(sentence):")
    lines.append("    sentence = sentence.lower().strip()")
    lines.append("")
    lines.append("    if sentence in GLOSS_DICT:")
    lines.append("        return GLOSS_DICT[sentence]")
    lines.append("")
    lines.append("    # fallback: check if any known phrase appears inside the typed sentence")
    lines.append("    for phrase, gloss in GLOSS_DICT.items():")
    lines.append("        if phrase in sentence:")
    lines.append("            return gloss")
    lines.append("")
    lines.append("    return None  # word/sentence not recognized")
    lines.append("")

    with open("gloss_dict.py", "w") as f:
        f.write("\n".join(lines))

    print(f"\ngloss_dict.py updated with {len(VOCABULARY)} words.")


if __name__ == "__main__":
    create_folders()
    counts = count_existing_clips()
    print_progress(counts)
    sync_gloss_dict()
