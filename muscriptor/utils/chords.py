"""Chord detection from transcribed notes at beat boundaries."""

from muscriptor.tokenizer.notes import Note

# Chord templates as 12-bit masks (bit 0 = C, bit 1 = C#, ..., bit 11 = B)
# Each mask is built with bit shifts for readability.
def _bits(*positions: int) -> int:
    m = 0
    for p in positions:
        m |= 1 << p
    return m


_CHORD_TEMPLATES: dict[str, tuple[str, int]] = {
    "maj":   ("",     _bits(0, 4, 7)),       # 0-4-7
    "min":   ("m",    _bits(0, 3, 7)),       # 0-3-7
    "dim":   ("dim",  _bits(0, 3, 6)),       # 0-3-6
    "aug":   ("aug",  _bits(0, 4, 8)),       # 0-4-8
    "sus4":  ("sus4", _bits(0, 5, 7)),       # 0-5-7
    "dom7":  ("7",    _bits(0, 4, 7, 10)),   # 0-4-7-10
    "maj7":  ("maj7", _bits(0, 4, 7, 11)),   # 0-4-7-11
    "min7":  ("m7",   _bits(0, 3, 7, 10)),   # 0-3-7-10
    "dim7":  ("dim7", _bits(0, 3, 6, 9)),    # 0-3-6-9
}

_PITCH_NAMES = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
]


def _active_pitch_class_mask(notes: list[Note], t: float, window: float = 0.05) -> int:
    """Bitmask of pitch classes sounding at time t (within ±window)."""
    mask = 0
    for n in notes:
        if n.is_drum:
            continue
        if n.onset - window <= t <= n.offset + window:
            mask |= 1 << (n.pitch % 12)
    return mask


def _best_chord(pc_mask: int) -> tuple[str, str] | None:
    """Find best-matching (root_name, quality_suffix) for a pitch class mask.

    Tries every root (0-11) against every chord template and picks the
    combination with the highest score (match_weight - unmatch_penalty).

    Returns None if the mask is empty.
    """
    if pc_mask == 0:
        return None

    best_root = 0
    best_suffix = ""
    best_score = float("-inf")

    for root in range(12):
        # Rotate mask so root is at bit 0
        rotated = ((pc_mask >> root) | (pc_mask << (12 - root))) & 0xFFF
        for qname, (qsuffix, qmask) in _CHORD_TEMPLATES.items():
            match = (rotated & qmask).bit_count()
            extra = (rotated & ~qmask).bit_count()
            # Primary score rewards matches, penalizes extra notes.
            # Tiny complexity penalty (0.01 per template bit) breaks ties
            # in favour of simpler chords (triads before 7ths).
            complexity = qmask.bit_count()
            score = match * 2 - extra - complexity * 0.01
            if score > best_score:
                best_score = score
                best_root = root
                best_suffix = qsuffix

    return _PITCH_NAMES[best_root], best_suffix


def detect_chords(
    notes: list[Note],
    beat_times: list[float],
) -> list[tuple[float, str]]:
    """Label each beat with its best-matching chord.

    Args:
        notes: Transcribed notes from the model.
        beat_times: Beat grid timestamps (from ``tempo.beat_times``).

    Returns:
        List of (beat_time, chord_label) tuples, e.g.
        ``[(0.0, 'C'), (0.5, 'Am'), (1.0, 'F'), ...]``.
    """
    if not beat_times:
        return []

    result: list[tuple[float, str]] = []
    for bt in beat_times:
        mask = _active_pitch_class_mask(notes, bt)
        chord = _best_chord(mask)
        if chord is not None:
            label = f"{chord[0]}{chord[1]}"
        else:
            label = result[-1][1] if result else "N.C."
        result.append((bt, label))
    return result


def format_chord_progression(
    chords: list[tuple[float, str]],
    bpm: float = 120,
    time_signature: tuple[int, int] = (4, 4),
    bars_per_line: int = 4,
) -> str:
    """Format a chord progression as a readable chord chart.

    Groups chords into bars based on BPM and time signature, one line per
    group of bars.

    Args:
        chords: List of (beat_time, chord_label) from :func:`detect_chords`.
        bpm: Tempo in beats per minute for bar grouping.
        time_signature: (numerator, denominator), default (4, 4).
        bars_per_line: How many bars to show per line.

    Returns:
        Formatted string like::

            | C  | Am | F  | G  |
            | C  | Am | F  | G  |
    """
    if not chords:
        return "N.C."

    beats_per_bar = time_signature[0]
    label_width = max(len(label) for _, label in chords) + 1

    lines: list[str] = []
    bar_chords: list[str] = []
    chord_idx = 0

    while chord_idx < len(chords):
        bar_labels: list[str] = []
        while chord_idx < len(chords) and len(bar_labels) < beats_per_bar:
            _, label = chords[chord_idx]
            bar_labels.append(label)
            chord_idx += 1

        if bar_labels:
            bar_str = " | ".join(f"{l:<{label_width}}" for l in bar_labels)
            bar_chords.append(f"| {bar_str} |")

        if len(bar_chords) >= bars_per_line:
            lines.append(" ".join(bar_chords))
            bar_chords = []

    if bar_chords:
        lines.append(" ".join(bar_chords))

    return "\n".join(lines)
