"""Tests for muscriptor/utils/chords.py."""

from muscriptor.tokenizer.notes import Note
from muscriptor.utils.chords import detect_chords


def _note(pitch: int, onset: float, offset: float) -> Note:
    return Note(is_drum=False, program=0, onset=onset, offset=offset, pitch=pitch)


def _drum(pitch: int, onset: float) -> Note:
    return Note(is_drum=True, program=128, onset=onset, offset=onset + 0.01, pitch=pitch)


class TestDetectChords:
    def test_c_major(self):
        """C, E, G active at beat 0 → 'C'."""
        notes = [
            _note(60, 0.0, 0.5),  # C
            _note(64, 0.0, 0.5),  # E
            _note(67, 0.0, 0.5),  # G
        ]
        chords = detect_chords(notes, [0.0])
        assert chords[0][1] == "C"

    def test_a_minor(self):
        """A, C, E active → 'Am'."""
        notes = [
            _note(57, 0.0, 0.5),  # A
            _note(60, 0.0, 0.5),  # C
            _note(64, 0.0, 0.5),  # E
        ]
        chords = detect_chords(notes, [0.0])
        assert chords[0][1] in ("Am", "A")

    def test_g_major(self):
        """G, B, D active → 'G'."""
        notes = [
            _note(55, 0.0, 0.5),  # G
            _note(59, 0.0, 0.5),  # B
            _note(62, 0.0, 0.5),  # D
        ]
        chords = detect_chords(notes, [0.0])
        assert chords[0][1] == "G"

    def test_dom7_chord(self):
        """C, E, G, Bb → 'C7'."""
        notes = [
            _note(60, 0.0, 0.5),  # C
            _note(64, 0.0, 0.5),  # E
            _note(67, 0.0, 0.5),  # G
            _note(70, 0.0, 0.5),  # Bb
        ]
        chords = detect_chords(notes, [0.0])
        assert chords[0][1] == "C7"

    def test_empty_notes(self):
        chords = detect_chords([], [0.0, 0.5])
        assert chords[0][1] == "N.C."

    def test_multi_beat(self):
        """Two different chords over two beats."""
        notes = [
            _note(60, 0.0, 0.4),  # C (ends before beat 1)
            _note(64, 0.0, 0.4),  # E
            _note(67, 0.0, 0.4),  # G
            _note(57, 0.5, 1.0),  # A (starts at beat 1)
            _note(60, 0.5, 1.0),  # C
            _note(64, 0.5, 1.0),  # E
        ]
        chords = detect_chords(notes, [0.0, 0.5])
        assert len(chords) == 2
        assert "C" in chords[0][1]
        assert "Am" in chords[1][1] or "A" in chords[1][1]

    def test_drums_ignored(self):
        """Drum notes should not affect chord detection."""
        notes = [
            _note(60, 0.0, 0.5),
            _note(64, 0.0, 0.5),
            _note(67, 0.0, 0.5),
            _drum(36, 0.0),
            _drum(42, 0.25),
        ]
        chords = detect_chords(notes, [0.0])
        assert chords[0][1] == "C"

    def test_empty_beat_times(self):
        chords = detect_chords([_note(60, 0.0, 0.5)], [])
        assert chords == []

    def test_chord_transition(self):
        """Chord changes across beats."""
        notes = [
            _note(60, 0.0, 0.4),   # C (beat 0 only)
            _note(64, 0.0, 0.4),   # E
            _note(67, 0.0, 0.4),   # G
            _note(65, 0.5, 1.0),   # F (beat 1+)
            _note(69, 0.5, 1.0),   # A
            _note(72, 0.5, 1.0),   # C
        ]
        chords = detect_chords(notes, [0.0, 0.5, 1.0])
        assert len(chords) == 3
        assert "C" in chords[0][1]  # C major
        assert "F" in chords[1][1]  # F major
        assert chords[2][1] == chords[1][1]  # F sustains
