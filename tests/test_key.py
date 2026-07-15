"""Tests for muscriptor/utils/key.py."""

from muscriptor.tokenizer.notes import Note
from muscriptor.utils.key import detect_key, key_signature


def _note(pitch: int, onset: float = 0.0) -> Note:
    return Note(is_drum=False, program=0, onset=onset, offset=onset + 0.2, pitch=pitch)


class TestDetectKey:
    def test_c_major(self):
        """C major: mostly C, E, G notes."""
        notes = [_note(60 + c, i * 0.5) for i, c in enumerate([0, 4, 7] * 5)]  # C, E, G
        key, mode = detect_key(notes)
        assert key == "C"
        assert mode == "major"

    def test_a_minor(self):
        """A minor: mostly A, C, E notes."""
        notes = [_note(57 + c, i * 0.5) for i, c in enumerate([0, 3, 7] * 5)]  # A, C, E
        key, mode = detect_key(notes)
        assert key == "A"
        assert mode == "minor"

    def test_g_major(self):
        """G major: G, B, D notes (pitch 55, 59, 62 → G3, B3, D4)."""
        notes = [_note(55 + c, i * 0.5) for i, c in enumerate([0, 4, 7] * 5)]  # G, B, D
        key, mode = detect_key(notes)
        assert key == "G"
        assert mode == "major"

    def test_d_minor(self):
        """D minor: D, F, A notes."""
        notes = [_note(62 + c, i * 0.5) for i, c in enumerate([0, 3, 7] * 5)]  # D, F, A
        key, mode = detect_key(notes)
        assert key == "D"
        assert mode == "minor"

    def test_empty_notes(self):
        key, mode = detect_key([])
        assert isinstance(key, str)
        assert isinstance(mode, str)

    def test_drums_only(self):
        notes = [Note(is_drum=True, program=128, onset=0.0, offset=0.01, pitch=i) for i in range(10)]
        key, mode = detect_key(notes)
        assert isinstance(key, str)
        assert isinstance(mode, str)


class TestKeySignature:
    def test_c_major(self):
        assert key_signature("C", "major") == 0

    def test_g_major(self):
        assert key_signature("G", "major") == 1

    def test_f_major(self):
        assert key_signature("F", "major") == -1

    def test_d_major(self):
        assert key_signature("D", "major") == 2

    def test_bb_major(self):
        assert key_signature("Bb", "major") == -2

    def test_fsharp_major(self):
        assert key_signature("F#", "major") == 6

    def test_a_minor(self):
        """A minor = relative of C major → 0 sharps."""
        assert key_signature("A", "minor") == 0

    def test_e_minor(self):
        """E minor = relative of G major → 1 sharp."""
        assert key_signature("E", "minor") == 1

    def test_d_minor(self):
        """D minor = relative of F major → 1 flat."""
        assert key_signature("D", "minor") == -1

    def test_case_sensitivity(self):
        # MIDI standard uses uppercase for sharps
        assert key_signature("C#", "major") == 7
