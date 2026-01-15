"""
General helper functions for brain-to-text data processing and transcription.
"""

import re
import numpy as np


PHONEMES = [
    "BLANK",
    "AA", "AE", "AH", "AO", "AW",
    "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G",
    "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW",
    "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V",
    "W", "Y", "Z", "ZH",
    " | ",
]

LOGIT_TO_PHONEME = dict(enumerate(PHONEMES))


def extract_transcription(char_ids: np.ndarray) -> str:
    """
    Extracts a transcription string from an array of character ASCII IDs.

    Args:
        char_ids (np.ndarray): Array of ASCII values.

    Returns:
        str: Decoded transcription string.
    """
    end_idx = np.argwhere(char_ids == 0)
    if end_idx.size == 0:
        end_idx = len(char_ids)
    else:
        end_idx = end_idx[0, 0]

    return "".join(chr(c) for c in char_ids[:end_idx])


def remove_punctuation(sentence: str) -> str:
    """
    Removes punctuation and normalizes string for Word Error Rate (WER) calculation.

    Args:
        sentence (str): The sentence to clean.

    Returns:
        str: Normalized sentence.
    """
    # Remove characters that aren't letters, spaces, hyphens, or apostrophes
    sentence = re.sub(r"[^a-zA-Z\- ']", "", sentence)
    # Normalize spaces and lower case
    sentence = sentence.replace("- ", " ").lower()
    sentence = sentence.replace("--", "").lower()
    sentence = sentence.replace(" '", "'").lower()

    return " ".join(sentence.split()).strip()
