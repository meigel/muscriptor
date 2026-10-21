"""Key detection: Krumhansl-Schmuckler key finding on a pitch-class histogram.

The transcription's notes are folded into twelve pitch-class bins and correlated
against the Krumhansl-Kessler probe-tone profiles, once for each of the 24 keys.
The best correlation wins. On a transcription this is a coarse read of the
music's tonal centre, which is all a key signature claims.
"""

import numpy as np

from muscriptor.tokenizer.notes import Note

# Krumhansl-Kessler probe-tone profiles for C major and C minor, in pitch-class
# order. Every other key is this profile rotated, which is what the algorithm
# does anyway; keeping the rotations implicit avoids 24 hand-written tables.
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

PITCH_CLASS_NAMES = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)

# The tonic's pitch class -> the key's conventional spelling and its key
# signature in fifths (positive sharps, negative flats). Enharmonic keys are
# spelled to keep the signature small: Db major over C#, Eb minor over D#.
_MAJOR_KEYS = {
    0: ("C", 0),
    7: ("G", 1),
    2: ("D", 2),
    9: ("A", 3),
    4: ("E", 4),
    11: ("B", 5),
    6: ("F#", 6),
    1: ("Db", -5),
    5: ("F", -1),
    10: ("Bb", -2),
    3: ("Eb", -3),
    8: ("Ab", -4),
}

_MINOR_KEYS = {
    9: ("A", 0),
    4: ("E", 1),
    11: ("B", 2),
    6: ("F#", 3),
    1: ("C#", 4),
    8: ("G#", 5),
    3: ("Eb", -6),
    10: ("Bb", -5),
    5: ("F", -4),
    0: ("C", -3),
    7: ("G", -2),
    2: ("D", -1),
}

_KEYS_BY_MODE = {"major": _MAJOR_KEYS, "minor": _MINOR_KEYS}


def _rotated(profile: np.ndarray, root: int) -> np.ndarray:
    """`profile` with `root` as its first pitch class."""
    return np.roll(profile, root)


def pitch_class_histogram(notes: list[Note]) -> np.ndarray | None:
    """Twelve bins of how long each pitch class sounds, or None without pitches.

    Durations rather than note counts: a held chord and a quick passing note
    count for what they last, which is what the profiles were fitted to. Drums
    have no pitch and are ignored.
    """
    histogram = np.zeros(12)
    for note in notes:
        if note.is_drum:
            continue
        histogram[note.pitch % 12] += max(note.offset - note.onset, 0.0) or 1.0
    if not histogram.any():
        return None
    return histogram


def detect_key(notes: list[Note]) -> tuple[str, str]:
    """The most likely key of `notes` as (tonic, mode), e.g. ('Eb', 'minor').

    Falls back to C major when no pitched note was transcribed.
    """
    histogram = pitch_class_histogram(notes)
    if histogram is None:
        return "C", "major"

    best_key = ("C", "major")
    best_correlation = -np.inf
    for mode, profile, spellings in (
        ("major", _MAJOR_PROFILE, _MAJOR_KEYS),
        ("minor", _MINOR_PROFILE, _MINOR_KEYS),
    ):
        for root in range(12):
            correlation = np.corrcoef(histogram, _rotated(profile, root))[0, 1]
            if np.isnan(correlation):
                continue
            if correlation > best_correlation:
                best_correlation = correlation
                best_key = (spellings[root][0], mode)
    return best_key


_PITCH_CLASS_OF = {
    name: pitch_class for pitch_class, name in enumerate(PITCH_CLASS_NAMES)
} | {
    "Db": 1,
    "Eb": 3,
    "Gb": 6,
    "Ab": 8,
    "Bb": 10,
    "Cb": 11,
    "Fb": 4,
}


def key_signature(tonic: str, mode: str) -> int:
    """The key's MIDI key-signature value: sharps when positive, flats when negative.

    Any spelling of the tonic is accepted, and enharmonic ones get the smaller
    signature: ('G', 'major') -> 1, ('F', 'major') -> -1, ('A', 'minor') -> 0,
    ('C#', 'major') -> -5 (spelled Db major, five flats over seven sharps).
    """
    if mode not in _KEYS_BY_MODE:
        raise ValueError(f"mode must be 'major' or 'minor', not {mode!r}")
    if tonic not in _PITCH_CLASS_OF:
        raise ValueError(f"unknown key name: {tonic!r}")
    return _KEYS_BY_MODE[mode][_PITCH_CLASS_OF[tonic]][1]


def midi_key_name(tonic: str, mode: str) -> str:
    """The key as mido spells it, which is what a key_signature meta event holds.

    ('G', 'major') -> 'G', ('A', 'minor') -> 'Am', ('Eb', 'minor') -> 'Ebm'.
    En-harmonic tonics get the spelling `key_signature` uses, so the name is
    always one mido can encode.
    """
    if mode not in _KEYS_BY_MODE:
        raise ValueError(f"mode must be 'major' or 'minor', not {mode!r}")
    if tonic not in _PITCH_CLASS_OF:
        raise ValueError(f"unknown key name: {tonic!r}")
    name = _KEYS_BY_MODE[mode][_PITCH_CLASS_OF[tonic]][0]
    return name if mode == "major" else f"{name}m"
