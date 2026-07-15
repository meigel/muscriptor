"""Tests for muscriptor/utils/tempo.py."""

import pytest

from muscriptor.tokenizer.notes import Note
from muscriptor.utils.tempo import beat_times, detect_tempo


def _note(onset: float, pitch: int = 60, is_drum: bool = False) -> Note:
    return Note(
        is_drum=is_drum,
        program=128 if is_drum else 0,
        onset=onset,
        offset=onset + 0.1,
        pitch=pitch,
    )


class TestDetectTempo:
    def test_quarter_notes_120bpm(self):
        """120 BPM: quarter notes every 0.5s."""
        notes = [_note(i * 0.5) for i in range(8)]
        assert detect_tempo(notes) == 120.0

    def test_eighth_notes_120bpm(self):
        """Eighth notes every 0.25s — both 120 and 240 BPM are valid."""
        notes = [_note(i * 0.25) for i in range(16)]
        bpm = detect_tempo(notes)
        assert bpm in (120.0, 240.0)

    def test_quarter_notes_90bpm(self):
        """90 BPM: quarter notes every 0.666...s."""
        notes = [_note(i * 60.0 / 90.0) for i in range(12)]
        assert detect_tempo(notes) == pytest.approx(90.0, abs=2.0)

    def test_quarter_notes_160bpm(self):
        """160 BPM: quarter notes every 0.375s."""
        notes = [_note(i * 60.0 / 160.0) for i in range(12)]
        assert detect_tempo(notes) == pytest.approx(160.0, abs=2.0)

    def test_empty_notes(self):
        assert detect_tempo([]) == 120.0

    def test_too_few_notes(self):
        notes = [_note(0.0), _note(1.0)]
        assert detect_tempo(notes) == 120.0

    def test_drums_only(self):
        notes = [_note(i * 0.5, is_drum=True) for i in range(8)]
        assert detect_tempo(notes) == 120.0

    def test_out_of_range_ioi(self):
        """IOI < 0.05s filtered; remaining 0.48s IOI → 125 BPM."""
        notes = [_note(0.0), _note(0.01), _note(0.02), _note(0.5)]
        assert detect_tempo(notes) == pytest.approx(125.0, abs=2.0)

    def test_swing_feel(self):
        """Swing: uneven 8ths detect pulse (120 or 240 BPM both valid)."""
        notes = []
        for i in range(4):
            notes.append(_note(i * 0.5))  # on beat
            notes.append(_note(i * 0.5 + 0.3))  # swung 8th
        bpm = detect_tempo(notes)
        assert bpm in (120.0, 240.0)

    def test_slow_40bpm(self):
        """40 BPM: quarter notes every 1.5s."""
        notes = [_note(i * 1.5) for i in range(6)]
        assert detect_tempo(notes) == pytest.approx(40.0, abs=1.0)

    def test_fast_220bpm(self):
        """220 BPM: quarter notes every ~0.273s."""
        notes = [_note(i * 60.0 / 220.0) for i in range(12)]
        assert detect_tempo(notes) == pytest.approx(220.0, abs=2.0)


class TestBeatTimes:
    def test_basic_grid(self):
        beats = beat_times([0.0, 0.5, 1.0, 1.5, 2.0], 120.0)
        assert len(beats) >= 4
        for i in range(1, len(beats)):
            assert abs(beats[i] - beats[i - 1] - 0.5) < 0.01

    def test_empty_onsets(self):
        assert beat_times([], 120.0) == []

    def test_zero_bpm(self):
        assert beat_times([0.0, 1.0], 0.0) == []

    def test_negative_bpm(self):
        assert beat_times([0.0, 1.0], -10.0) == []

    def test_cover_range(self):
        """Beats should span from before first onset to after last onset."""
        beats = beat_times([1.0, 2.0, 3.0], 120.0)
        assert beats[0] <= 1.0
        assert beats[-1] >= 3.0

    def test_single_onset(self):
        beats = beat_times([1.0], 120.0)
        assert len(beats) >= 2  # at least before and after
