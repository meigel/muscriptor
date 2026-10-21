"""Tests for muscriptor/utils/key.py."""

import pytest

from muscriptor.tokenizer.notes import Note
from muscriptor.utils.key import (
    detect_key,
    key_signature,
    midi_key_name,
    pitch_class_histogram,
)


def _chord(pitches, onset=0.0, duration=1.0, program=0):
    return [
        Note(
            is_drum=False,
            program=program,
            onset=onset,
            offset=onset + duration,
            pitch=pitch,
        )
        for pitch in pitches
    ]


def _progression(chord_pitches, bars=4):
    """Each chord held for a beat, repeated `bars` times over."""
    notes = []
    for index, pitches in enumerate(chord_pitches):
        notes += _chord(pitches, onset=float(index), duration=1.0)
    return notes


def test_histogram_weights_durations():
    notes = _chord([60], duration=3.0) + _chord([62], onset=3.0, duration=1.0)
    histogram = pitch_class_histogram(notes)
    assert histogram[0] == pytest.approx(3.0)
    assert histogram[2] == pytest.approx(1.0)


def test_histogram_ignores_drums():
    drums = [Note(is_drum=True, program=128, onset=0.0, offset=0.01, pitch=36)]
    assert pitch_class_histogram(drums) is None


def test_detect_key_c_major():
    # C - F - G - C, the plainest C major cadence there is.
    notes = _progression([[60, 64, 67], [65, 69, 72], [67, 71, 74], [60, 64, 67]])
    assert detect_key(notes) == ("C", "major")


def test_detect_key_a_minor():
    # Am - Dm - E - Am: the raised seventh of E points at A, not C.
    notes = _progression([[57, 60, 64], [62, 65, 69], [64, 68, 71], [57, 60, 64]])
    assert detect_key(notes) == ("A", "minor")


def test_detect_key_eb_major():
    # Eb - Ab - Bb - Eb, spelled with flats.
    notes = _progression([[63, 67, 70], [68, 72, 75], [70, 74, 77], [63, 67, 70]])
    assert detect_key(notes) == ("Eb", "major")


def test_detect_key_without_pitched_notes_falls_back_to_c_major():
    assert detect_key([]) == ("C", "major")
    assert detect_key(
        [Note(is_drum=True, program=128, onset=0, offset=1, pitch=36)]
    ) == (
        "C",
        "major",
    )


def test_key_signature_counts():
    assert key_signature("C", "major") == 0
    assert key_signature("G", "major") == 1
    assert key_signature("F", "major") == -1
    assert key_signature("A", "minor") == 0
    assert key_signature("E", "minor") == 1
    assert key_signature("Eb", "minor") == -6


def test_key_signature_spells_enharmonics_the_plain_way():
    # Db major has five flats; C# major would have seven sharps.
    assert key_signature("C#", "major") == -5


def test_midi_key_name_is_what_mido_encodes():
    assert midi_key_name("C", "major") == "C"
    assert midi_key_name("A", "minor") == "Am"
    assert midi_key_name("Eb", "minor") == "Ebm"
    assert midi_key_name("C#", "major") == "Db"


def test_key_signature_rejects_nonsense():
    with pytest.raises(ValueError):
        key_signature("C", "dorian")
    with pytest.raises(ValueError):
        key_signature("H", "major")
