"""Key detection using Krumhansl-Schmuckler key-finding algorithm."""

from collections import Counter

import numpy as np

from muscriptor.tokenizer.notes import Note

# Krumhansl-Kessler major key profiles (normalized)
_MAJOR_PROFILES = {
    0: [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    1: [2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29],
    2: [2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66],
    3: [3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39],
    4: [2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19],
    5: [5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52],
    6: [2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09],
    7: [4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38],
    8: [4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33],
    9: [2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48],
    10: [3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23],
    11: [2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35],
}

# Krumhansl-Kessler minor key profiles (normalized)
_MINOR_PROFILES = {
    0: [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    1: [3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34],
    2: [3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69],
    3: [2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98],
    4: [3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75],
    5: [4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54],
    6: [2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53],
    7: [3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60],
    8: [2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38],
    9: [5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52],
    10: [3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68],
    11: [2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33],
}

_PITCH_CLASS_NAMES = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
]


def _pitch_class_histogram(notes: list[Note]) -> list[float]:
    """Normalised 12-bin pitch-class histogram from note onset counts."""
    pc: Counter[int] = Counter()
    for n in notes:
        if not n.is_drum:
            pc[n.pitch % 12] += 1
    total = sum(pc.values())
    if total == 0:
        return [1.0] * 12
    return [pc.get(i, 0) / total * 100 for i in range(12)]


def _correlate(hist: list[float], profile: list[float]) -> float:
    """Pearson correlation between hist and profile."""
    h = np.array(hist)
    p = np.array(profile)
    result = np.corrcoef(h, p)[0, 1]
    return float(result) if not np.isnan(result) else -1.0


def detect_key(notes: list[Note]) -> tuple[str, str]:
    """Return (key_name, mode) for the most likely musical key.

    Uses Krumhansl-Schmuckler key-finding on the pitch-class histogram.
    Mode is 'major' or 'minor'.

    Returns ('C', 'major') as fallback when there's no data.
    """
    hist = _pitch_class_histogram(notes)

    best_corr = -float("inf")
    best_root = 0
    best_mode = "major"

    for root, profile in _MAJOR_PROFILES.items():
        corr = _correlate(hist, profile)
        if corr > best_corr:
            best_corr = corr
            best_root = root
            best_mode = "major"

    for root, profile in _MINOR_PROFILES.items():
        corr = _correlate(hist, profile)
        if corr > best_corr:
            best_corr = corr
            best_root = root
            best_mode = "minor"

    return _PITCH_CLASS_NAMES[best_root], best_mode


# Circle of fifths: number of sharps (positive) or flats (negative)
_CIRCLE_OF_FIFTHS = {
    "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6,
    "C#": 7, "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7,
}

# Relative major mapping: minor key -> its relative major
_RELATIVE_MAJOR = {
    "A": "C", "A#": "C#", "B": "D", "C": "D#", "C#": "E",
    "D": "F", "D#": "F#", "E": "G", "F": "G#", "F#": "A",
    "G": "A#", "G#": "B",
    "Bb": "Db", "Eb": "Gb", "Ab": "B", "Db": "E", "Gb": "A",
}


def key_signature(key_name: str, mode: str) -> int:
    """Convert key name + mode to MIDI key signature byte.

    Returns a signed integer: positive = sharps, negative = flats.
    Examples:
        key_signature('C', 'major') -> 0
        key_signature('G', 'major') -> 1
        key_signature('F', 'major') -> -1
        key_signature('A', 'minor') -> 0   (relative of C major)
        key_signature('E', 'minor') -> 1   (relative of G major)
    """
    if mode == "minor":
        key_name = _RELATIVE_MAJOR.get(key_name, key_name)
    return _CIRCLE_OF_FIFTHS.get(key_name, 0)