"""
This replaces "Stage 1" (text-to-gloss) with a simple hand-written
dictionary -- a reasonable shortcut for a minor project.

Generated from setup_vocab.py -- edit VOCABULARY there, not the
dict below, then re-run setup_vocab.py to regenerate this file.
"""

GLOSS_DICT = {
    "hello": "HELLO",
    "thank you": "THANK_YOU",
    "please": "PLEASE",
    "sorry": "SORRY",
    "help": "HELP",
    "water": "WATER",
    "stop": "STOP",
    "yes": "YES",
    "no": "NO",
    "good": "GOOD",
    "bad": "BAD",
    "name": "NAME",
    "me": "ME",
    "you": "YOU",
    "eat": "EAT",
    "more": "MORE",
}


def sentence_to_gloss(sentence):
    sentence = sentence.lower().strip()

    if sentence in GLOSS_DICT:
        return GLOSS_DICT[sentence]

    # fallback: check if any known phrase appears inside the typed sentence
    for phrase, gloss in GLOSS_DICT.items():
        if phrase in sentence:
            return gloss

    return None  # word/sentence not recognized
