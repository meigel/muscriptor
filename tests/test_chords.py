"""Tests for muscriptor/utils/chords.py."""

from muscriptor.tokenizer.notes import Note
from muscriptor.utils.chords import (
    NO_CHORD,
    chord_changes,
    chord_chart,
    chord_for_mask,
    chord_labels,
)


def _mask(*pitch_classes):
    return sum(1 << pitch_class for pitch_class in pitch_classes)


def _notes(pitches, onset=0.0, duration=1.0, program=0, is_drum=False):
    return [
        Note(
            is_drum=is_drum,
            program=program,
            onset=onset,
            offset=onset + duration,
            pitch=pitch,
        )
        for pitch in pitches
    ]


def _triad(root, quality=(0, 4, 7)):
    return [root + interval for interval in quality]


def test_chord_for_mask_triads():
    assert chord_for_mask(_mask(0, 4, 7)) == "C"
    assert chord_for_mask(_mask(9, 0, 4)) == "Am"
    assert chord_for_mask(_mask(2, 5, 9)) == "Dm"


def test_chord_for_mask_sevenths():
    assert chord_for_mask(_mask(0, 4, 7, 11)) == "Cmaj7"
    assert chord_for_mask(_mask(0, 4, 7, 10)) == "C7"
    assert chord_for_mask(_mask(0, 3, 7, 10)) == "Cm7"


def test_chord_for_mask_suspended_and_diminished():
    assert chord_for_mask(_mask(0, 5, 7)) == "Csus4"
    assert chord_for_mask(_mask(0, 3, 6)) == "Cdim"


def test_chord_for_mask_is_empty_for_silence():
    assert chord_for_mask(0) is None


def test_chord_for_mask_ignores_voicing():
    """The same pitch classes an octave apart are the same chord."""
    assert chord_for_mask(_mask(0, 4, 7)) == chord_for_mask(_mask(0, 4, 7))


def test_chord_labels_change_where_the_harmony_does():
    notes = _notes(_triad(60), onset=0.0, duration=2.0) + _notes(
        _triad(65), onset=2.0, duration=2.0
    )
    assert chord_labels(notes, [0, 1, 2, 3]) == [
        (0, "C"),
        (1, "C"),
        (2, "F"),
        (3, "F"),
    ]


def test_chord_labels_hold_across_a_rest():
    notes = _notes(_triad(60), onset=0.0, duration=1.0)
    assert chord_labels(notes, [0, 1, 2]) == [(0, "C"), (1, "C"), (2, "C")]


def test_chord_labels_start_with_no_chord():
    notes = _notes(_triad(60), onset=1.0, duration=1.0)
    assert chord_labels(notes, [0, 1]) == [(0, NO_CHORD), (1, "C")]


def test_chord_labels_ignore_drums():
    drums = _notes([36, 38], onset=0.0, duration=1.0, is_drum=True)
    assert chord_labels(drums, [0]) == [(0, NO_CHORD)]


def test_short_note_still_names_its_beat():
    """A staccato note ending on the beat is found by the widened window."""
    notes = _notes(_triad(60), onset=0.9, duration=0.1)
    assert chord_labels(notes, [1.0]) == [(1.0, "C")]


def test_chord_changes_drops_repeats():
    labels = [(0, "C"), (1, "C"), (2, "Am"), (3, "Am"), (4, "F")]
    assert chord_changes(labels) == [(0, "C"), (2, "Am"), (4, "F")]


def test_chord_chart_one_cell_per_bar():
    chords = [(0, "C"), (4, "Am"), (8, "F"), (12, "G")]
    assert chord_chart(chords) == "| C  | Am | F  | G  |"


def test_chord_chart_packs_two_chords_into_a_bar():
    assert chord_chart([(0, "C"), (2, "Am")]) == "| C Am |"


def test_chord_chart_wraps_lines():
    chords = [(bar * 4, label) for bar, label in enumerate(["C", "Am", "F", "G"])]
    assert chord_chart(chords, bars_per_line=2).splitlines() == [
        "| C  | Am |",
        "| F  | G  |",
    ]


def test_chord_chart_respects_the_meter():
    chords = [(0, "C"), (3, "F")]
    assert chord_chart(chords, beats_per_bar=3) == "| C | F |"


def test_chord_chart_without_chords():
    assert chord_chart([]) == NO_CHORD
