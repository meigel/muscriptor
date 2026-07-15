"""BPM detection and beat-grid generation from transcribed notes."""

from muscriptor.tokenizer.notes import Note

_MIN_BPM = 40
_MAX_BPM = 320


def _grid_alignment_score(onsets: list[float], bpm: float) -> int:
    """Count how many onsets land near the beat grid.

    Tolerance is proportional to beat duration (15%) so scoring is fair
    across fast and slow tempos.
    """
    if bpm <= 0 or not onsets:
        return 0
    beat_dur = 60.0 / bpm
    tol = beat_dur * 0.15
    start = min(onsets)
    aligned = 0
    for o in onsets:
        phase = (o - start) % beat_dur
        if phase < tol or phase > beat_dur - tol:
            aligned += 1
    return aligned


def _harmonic_clarity(onsets: list[float], bpm: float) -> float:
    """Fraction of *all* onsets that fall on the beat grid (within 8 % of a
    grid line).

    A high fraction means most notes articulate the pulse — the hallmark of
    the *true* tempo.  Harmonics and sub-multiples get fewer on-beat
    onsets because their grid lines are either too tight (fast) or too
    sparse (slow) to catch the main rhythmic activity.
    """
    if bpm <= 0 or not onsets:
        return 0.0
    beat_dur = 60.0 / bpm
    tol = beat_dur * 0.08
    start = min(onsets)
    on_beat = 0
    for o in onsets:
        phase = (o - start) % beat_dur
        if phase < tol or phase > beat_dur - tol:
            on_beat += 1
    return on_beat / len(onsets)


def detect_tempo(notes: list[Note]) -> float:
    """Estimate tempo in BPM from note onset times.

    Generates candidate BPMs from consecutive inter-onset intervals (IOIs),
    their sub/super harmonics, and adjacent IOI pair-sums (to catch swing).
    Scores each candidate by grid alignment, then breaks ties by harmonic
    clarity (the cleanest on-beat vs near-beat ratio wins).

    Returns 120.0 if too few notes (< 3) to estimate.
    """
    onsets = sorted({n.onset for n in notes if not n.is_drum})
    if len(onsets) < 3:
        return 120.0

    # Consecutive IOIs, filtering out micro-timing noise
    io_is = [
        onsets[i + 1] - onsets[i]
        for i in range(len(onsets) - 1)
        if onsets[i + 1] - onsets[i] > 0.05
    ]
    if not io_is:
        return 120.0

    # Collect candidate BPMs from each IOI and its harmonics
    candidates: set[int] = set()
    for ioi in io_is:
        base = 60.0 / ioi
        for mult in [0.125, 0.25, 0.5, 1, 2, 4, 8]:
            bpm = base * mult
            if _MIN_BPM <= bpm <= _MAX_BPM:
                candidates.add(round(bpm))

    # Also add candidates from sums of adjacent IOI pairs (catches swing:
    # 0.3 + 0.2 = 0.5s -> 120 BPM)
    for i in range(0, len(io_is) - 1, 2):
        pair_sum = io_is[i] + io_is[i + 1]
        base = 60.0 / pair_sum
        for mult in [0.25, 0.5, 1, 2, 4]:
            bpm = base * mult
            if _MIN_BPM <= bpm <= _MAX_BPM:
                candidates.add(round(bpm))

    if not candidates:
        return 120.0

    # Score by grid alignment, then break ties by harmonic clarity,
    # then prefer the slower tempo (fundamental over harmonic).
    strongly_better = 1.05  # 5% better -> unambiguous winner
    close_enough = 0.95     # within 5% -> compare clarity

    best_bpm = 120.0
    best_raw = 0
    best_clarity = 0.0

    for b in sorted(candidates):
        raw = _grid_alignment_score(onsets, float(b))
        if best_raw == 0:
            best_raw, best_bpm, best_clarity = (
                raw,
                b,
                _harmonic_clarity(onsets, float(b)),
            )
        elif raw >= best_raw * strongly_better:
            # Clearly better -> take it
            best_raw, best_bpm = raw, b
            best_clarity = _harmonic_clarity(onsets, float(b))
        elif raw >= best_raw * close_enough:
            clarity = _harmonic_clarity(onsets, float(b))
            if clarity > best_clarity + 0.01:
                # Better clarity -> better tempo
                best_raw, best_bpm, best_clarity = raw, b, clarity
            elif abs(clarity - best_clarity) <= 0.01:
                # Same clarity -> prefer slower (the fundamental)
                if b < best_bpm:
                    best_raw, best_bpm, best_clarity = raw, b, clarity

    return float(best_bpm)


def beat_times(
    onset_times: list[float],
    bpm: float,
    time_signature: tuple[int, int] = (4, 4),
) -> list[float]:
    """Generate beat times from onset list and BPM.

    Aligns the beat grid so the strongest beat (beat 1) coincides with
    the densest onset region.

    Returns sorted list of absolute beat times in seconds.
    """
    if bpm <= 0 or not onset_times:
        return []

    beat_dur = 60.0 / bpm
    start = min(onset_times)
    end = max(onset_times)

    # Scan phase to align grid with onsets
    best_phase = 0.0
    best_score = -1
    for offset in range(int(beat_dur * 10)):
        phase = offset / 10.0
        score = sum(
            1
            for o in onset_times
            if abs((o - start - phase) % beat_dur) < 0.03
        )
        if score > best_score:
            best_score = score
            best_phase = phase

    grid_start = start + best_phase
    beats = []
    t = grid_start
    while t <= end + beat_dur:
        beats.append(round(t, 4))
        t += beat_dur
    return beats
