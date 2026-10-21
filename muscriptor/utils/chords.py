"""Chord labelling from transcribed notes, one label per beat.

Each chord template is a twelve-bit mask of pitch classes. A beat's sounding
notes become a mask too, and the template that explains it with the fewest
notes left over wins. That is a coarse harmonic read — the transcription's
notes, not the recording's timbre — but it is what a chord chart claims.
"""

from collections.abc import Sequence

from muscriptor.tokenizer.notes import Note

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

# How far outside a note, in the units of its times, a beat still counts as
# covered by it. Small: chords change on beats, so a wide window smears the
# old and the new chord together.
WINDOW = 0.05

# Written when nothing has sounded yet, or in a passage without pitched notes.
NO_CHORD = "N.C."


def _mask(*positions: int) -> int:
    mask = 0
    for position in positions:
        mask |= 1 << position
    return mask


# Quality -> (suffix, interval mask). Simplest first: on a tie (a bare fifth is
# both a major triad's outer notes and a sus4's), the earlier entry wins.
TEMPLATES: dict[str, tuple[str, int]] = {
    "major": ("", _mask(0, 4, 7)),
    "minor": ("m", _mask(0, 3, 7)),
    "diminished": ("dim", _mask(0, 3, 6)),
    "augmented": ("aug", _mask(0, 4, 8)),
    "sus4": ("sus4", _mask(0, 5, 7)),
    "sus2": ("sus2", _mask(0, 2, 7)),
    "dominant7": ("7", _mask(0, 4, 7, 10)),
    "major7": ("maj7", _mask(0, 4, 7, 11)),
    "minor7": ("m7", _mask(0, 3, 7, 10)),
    "half_diminished": ("m7b5", _mask(0, 3, 6, 10)),
    "diminished7": ("dim7", _mask(0, 3, 6, 9)),
}


def _sounding_mask(notes: Sequence[Note], time: float, window: float) -> int:
    """Pitch classes of the pitched notes sounding at `time`.

    Notes count strictly while they sound, so a note ending on `time` is not
    part of the chord that starts there — otherwise every chord change would be
    heard as the old and the new chord together. `window` only widens the search
    for beats where nothing is sounding strictly, e.g. after a staccato note, so
    a short note still names its beat.
    """
    strict = 0
    relaxed = 0
    for note in notes:
        if note.is_drum:
            continue
        if note.onset <= time < note.offset:
            strict |= 1 << (note.pitch % 12)
        elif note.onset - window <= time <= note.offset + window:
            relaxed |= 1 << (note.pitch % 12)
    return strict or relaxed


def chord_for_mask(pitch_classes: int) -> str | None:
    """The chord label for a twelve-bit pitch-class mask, or None if it is empty.

    Every root is tried against every template; the score rewards the template's
    notes being present and charges for the notes it doesn't explain, with a
    hair of complexity so a triad beats the seventh chord that contains it.
    """
    if pitch_classes == 0:
        return None

    best_label: str | None = None
    best_score = float("-inf")
    for root in range(12):
        # Rotate so the candidate root sits at bit 0.
        rotated = ((pitch_classes >> root) | (pitch_classes << (12 - root))) & 0xFFF
        for suffix, mask in TEMPLATES.values():
            matched = (rotated & mask).bit_count()
            unexplained = (rotated & ~mask).bit_count()
            score = 2 * matched - unexplained - 0.01 * mask.bit_count()
            if score > best_score:
                best_score = score
                best_label = f"{PITCH_CLASS_NAMES[root]}{suffix}"
    return best_label


def chord_labels(
    notes: Sequence[Note], beats: Sequence[float], window: float = WINDOW
) -> list[tuple[float, str]]:
    """Best chord label at each of `beats`, as (beat, label) pairs.

    `notes` and `beats` share a time unit (seconds of the recording, or beats of
    the written score — see `notes_to_midi`). A chord is held across rests and
    gaps, the way a chart reads: the label only changes where the harmony does.
    """
    labels: list[tuple[float, str]] = []
    current = NO_CHORD
    for beat in beats:
        label = chord_for_mask(_sounding_mask(notes, beat, window))
        if label is not None:
            current = label
        labels.append((float(beat), current))
    return labels


def chord_changes(
    labels: Sequence[tuple[float, str]],
) -> list[tuple[float, str]]:
    """The labels that start a new chord, dropping the ones that repeat it."""
    changes: list[tuple[float, str]] = []
    for beat, label in labels:
        if not changes or changes[-1][1] != label:
            changes.append((beat, label))
    return changes


def chord_chart(
    chords: Sequence[tuple[float, str]],
    beats_per_bar: int = 4,
    bars_per_line: int = 4,
) -> str:
    """A one-line-per-group-of-bars chord chart from (beat, label) change points.

    Each bar gets one cell holding its chords, in order and without repeats, so
    a bar of held C reads `C` and a bar of C then Am reads `C Am`.
    """
    if not chords:
        return NO_CHORD

    # Beats are counted from the start of the score, so bar 0 holds beats
    # [0, beats_per_bar). Chords before it (a pickup) go with the first bar.
    bars: dict[int, list[str]] = {}
    for beat, label in chords:
        bar = max(int(beat // beats_per_bar), 0)
        chords_in_bar = bars.setdefault(bar, [])
        if not chords_in_bar or chords_in_bar[-1] != label:
            chords_in_bar.append(label)

    last_bar = max(bars)
    cells = [" ".join(bars.get(bar, [])) for bar in range(last_bar + 1)]
    width = max(len(cell) for cell in cells)
    lines = []
    for start in range(0, len(cells), bars_per_line):
        row = cells[start : start + bars_per_line]
        lines.append("".join(f"| {cell:<{width}} " for cell in row) + "|")
    return "\n".join(lines)
