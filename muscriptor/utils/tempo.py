"""BPM detection and beat-grid generation from transcribed notes."""

from __future__ import annotations

from muscriptor.tokenizer.notes import Note

_MIN_BPM = 40
_MAX_BPM = 320


def _best_phase(onsets: list[float], beat_dur: float, steps: int = 50) -> float:
    """Find the phase offset (0..beat_dur) that maximises grid alignment.

    Scans *steps* evenly spaced phases.  The best one is returned so the
    caller can use it to compute the true grid-alignment score.
    """
    start = min(onsets)
    best_phase = 0.0
    best_count = -1
    tol = beat_dur * 0.12
    for s in range(steps):
        phase = (beat_dur / steps) * s
        count = 0
        for o in onsets:
            p = (o - start + phase) % beat_dur
            if p < tol or p > beat_dur - tol:
                count += 1
        if count > best_count:
            best_count = count
            best_phase = phase
    return best_phase


def _grid_alignment_score(
    onsets: list[float], beat_dur: float, phase: float
) -> int:
    """Number of onsets within 12 % of a grid line at the given phase."""
    tol = beat_dur * 0.12
    start = min(onsets)
    n = 0
    for o in onsets:
        p = (o - start + phase) % beat_dur
        if p < tol or p > beat_dur - tol:
            n += 1
    return n


def _measure_clarity(
    onsets: list[float], beat_dur: float, phase: float
) -> float:
    """Ratio of onsets on strong beats (1 & 3) to weak beats (2 & 4).

    A high ratio means the tempo correctly captures the 4/4 measure
    structure — onsets cluster on downbeats rather than off-beats.
    """
    bar_dur = beat_dur * 4
    tol = beat_dur * 0.10
    start = min(onsets)
    strong = 0
    weak = 0
    for o in onsets:
        p = (o - start + phase) % bar_dur
        # Beat 1 (downbeat): near 0 or near bar_dur
        if p < tol or p > bar_dur - tol:
            strong += 1
        # Beat 3: near 2 × beat_dur
        elif abs(p - 2 * beat_dur) < tol:
            strong += 1
        # Beats 2 and 4: near beat_dur or 3 × beat_dur
        elif abs(p - beat_dur) < tol or abs(p - 3 * beat_dur) < tol:
            weak += 1
        # Off-grid — ignore (could be syncopation)
    total = strong + weak
    return strong / total if total > 0 else 0.0


def detect_tempo(notes: list[Note]) -> float:
    """Estimate tempo in BPM from note onset times.

    Collects candidate BPMs from inter-onset intervals (IOIs) and their
    harmonics.  For each candidate the optimal beat-grid phase is found,
    then candidates are scored by (1) grid alignment and (2) measure
    clarity (downbeat strength).  Candidates within 5 % of the best raw
    alignment are compared by clarity; the clearest wins.

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

    # Collect candidate BPMs as floats (no rounding — avoids drift)
    candidates: set[float] = set()
    for ioi in io_is:
        base = 60.0 / ioi
        for mult in [0.125, 0.25, 0.5, 1, 2, 4, 8]:
            bpm = base * mult
            if _MIN_BPM <= bpm <= _MAX_BPM:
                candidates.add(bpm)

    # Pair-sum candidates for swing
    for i in range(0, len(io_is) - 1, 2):
        pair_sum = io_is[i] + io_is[i + 1]
        base = 60.0 / pair_sum
        for mult in [0.25, 0.5, 1, 2, 4]:
            bpm = base * mult
            if _MIN_BPM <= bpm <= _MAX_BPM:
                candidates.add(bpm)

    if not candidates:
        return 120.0

    # Rank by grid alignment (with optimal phase), then break ties by
    # measure clarity.  Clarity must improve by more than 0.5 to override
    # the preference for the slower (fundamental) tempo — this resolves
    # harmonic ambiguity in perfectly regular onsets while preserving the
    # clarity advantage for real music where accent patterns are richer.
    strongly_better = 1.05
    close_enough = 0.95
    _CLARITY_THRESH = 0.5  # minimum improvement to override slower-preference

    best_bpm = 120.0
    best_raw = 0
    best_clarity = 0.0

    for bpm in sorted(candidates):
        beat_dur = 60.0 / bpm
        phase = _best_phase(onsets, beat_dur)
        raw = _grid_alignment_score(onsets, beat_dur, phase)

        if best_raw == 0:
            best_raw, best_bpm = raw, bpm
            best_clarity = _measure_clarity(onsets, beat_dur, phase)
        elif raw >= best_raw * strongly_better:
            best_raw, best_bpm = raw, bpm
            best_clarity = _measure_clarity(onsets, beat_dur, phase)
        elif raw >= best_raw * close_enough:
            clarity = _measure_clarity(onsets, beat_dur, phase)
            if clarity > best_clarity + _CLARITY_THRESH:
                # Clearly stronger beat structure → good tempo
                best_raw, best_bpm, best_clarity = raw, bpm, clarity
            elif bpm < best_bpm:
                # Otherwise prefer slower (fundamental pulse)
                best_raw, best_bpm, best_clarity = raw, bpm, clarity

    return round(best_bpm)


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
    phase = _best_phase(sorted(onset_times), beat_dur)

    grid_start = start + phase
    beats = []
    t = grid_start
    while t <= end + beat_dur:
        beats.append(round(t, 4))
        t += beat_dur
    return beats
