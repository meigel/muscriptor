"""Note quantization to a beat grid."""

from muscriptor.tokenizer.notes import Note


def _binary_search_nearest(grid: list[float], t: float) -> float:
    """Find the nearest grid point to time t via binary search.

    Returns the grid value closest to t.
    """
    if not grid:
        return t
    if t <= grid[0]:
        return grid[0]
    if t >= grid[-1]:
        return grid[-1]

    lo, hi = 0, len(grid) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if grid[mid] < t:
            lo = mid + 1
        else:
            hi = mid

    # Compare lo and lo-1
    best = grid[lo]
    best_dist = abs(best - t)
    if lo > 0:
        d = abs(grid[lo - 1] - t)
        if d <= best_dist:  # prefer earlier when tied
            best = grid[lo - 1]
            best_dist = d
    if lo + 1 < len(grid):
        d = abs(grid[lo + 1] - t)
        if d < best_dist:
            best = grid[lo + 1]
    return best


def quantize_notes(
    notes: list[Note],
    grid: list[float],
) -> list[Note]:
    """Snap all note onsets and offsets to the nearest grid points.

    Args:
        notes: Raw transcribed notes.
        grid: Subdivision grid timestamps in seconds (e.g. eighth-note
            boundaries from ``tempo.beat_times`` doubled in density).

    Returns:
        New list of quantized Note objects.
    """
    if not grid:
        return list(notes)

    quantized: list[Note] = []
    for n in notes:
        q_onset = _binary_search_nearest(grid, n.onset)
        q_offset = _binary_search_nearest(grid, n.offset)
        # Ensure minimum duration after quantization
        if q_offset <= q_onset:
            q_offset = q_onset + 0.01
        quantized.append(Note(
            is_drum=n.is_drum,
            program=n.program,
            onset=round(q_onset, 4),
            offset=round(q_offset, 4),
            pitch=n.pitch,
        ))
    return quantized
