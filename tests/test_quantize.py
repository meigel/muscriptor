"""Tests for muscriptor/utils/quantize.py."""

from muscriptor.tokenizer.notes import Note
from muscriptor.utils.quantize import quantize_notes


def _note(onset: float, offset: float, pitch: int = 60, is_drum: bool = False) -> Note:
    return Note(
        is_drum=is_drum,
        program=128 if is_drum else 0,
        onset=onset,
        offset=offset,
        pitch=pitch,
    )


class TestQuantizeNotes:
    def test_snaps_to_nearest(self):
        """Notes snap to the nearest 8th-note grid."""
        grid = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
        notes = [_note(0.12, 0.26)]
        q = quantize_notes(notes, grid)
        assert q[0].onset == 0.125
        assert q[0].offset == 0.25

    def test_snaps_onset_down(self):
        """Onset at 0.06 → snaps to 0.0 (closer than 0.125)."""
        grid = [0.0, 0.125, 0.25]
        notes = [_note(0.06, 0.2)]
        q = quantize_notes(notes, grid)
        assert q[0].onset == 0.0

    def test_equal_distance_prefers_earlier(self):
        """At exactly 0.0625 (midpoint of 0.0 and 0.125), snaps to 0.0."""
        grid = [0.0, 0.125]
        notes = [_note(0.0625, 0.2)]
        q = quantize_notes(notes, grid)
        assert q[0].onset == 0.0  # equal distance, earlier wins

    def test_minimum_duration(self):
        """If offset snaps to same as onset, enforce 0.01 minimum."""
        grid = [0.0, 0.5, 1.0]
        notes = [_note(0.48, 0.51)]
        q = quantize_notes(notes, grid)
        assert q[0].onset == 0.5
        assert q[0].offset > q[0].onset

    def test_empty_grid(self):
        notes = [_note(0.5, 0.8)]
        q = quantize_notes(notes, grid=[])
        assert q == notes

    def test_empty_notes(self):
        assert quantize_notes([], [0.0, 0.5]) == []

    def test_drum_notes_quantized(self):
        """Drums are quantized too."""
        grid = [0.0, 0.125, 0.25, 0.375, 0.5]
        notes = [_note(0.12, 0.13, pitch=36, is_drum=True)]
        q = quantize_notes(notes, grid)
        assert q[0].onset == 0.125

    def test_preserves_pitch_and_program(self):
        grid = [0.0, 0.5]
        notes = [_note(0.45, 0.9, pitch=72)]
        q = quantize_notes(notes, grid)
        assert q[0].pitch == 72
        assert q[0].program == 0

    def test_multi_note_quantization(self):
        grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        notes = [
            _note(0.02, 0.23),
            _note(0.27, 0.48),
            _note(0.51, 0.74),
            _note(0.76, 0.99),
        ]
        q = quantize_notes(notes, grid)
        for i, expected_onset in enumerate([0.0, 0.25, 0.5, 0.75]):
            assert q[i].onset == expected_onset, f"Note {i} onset mismatch"

    def test_onset_after_last_grid_point(self):
        """Onset after last grid point snaps to last point; offset enforces min duration."""
        grid = [0.0, 0.5, 1.0]
        notes = [_note(1.5, 2.0)]
        q = quantize_notes(notes, grid)
        assert q[0].onset == 1.0
        assert q[0].offset == 1.01  # minimum duration enforced
