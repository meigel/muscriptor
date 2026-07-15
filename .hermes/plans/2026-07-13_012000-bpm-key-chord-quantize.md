# BPM / Key / Chord Detection + Quantization for MuScriptor

> **For Hermes:** Implement task-by-task. No subagent delegation unless specified.

**Goal:** Add BPM detection, key detection, chord labeling, and note quantization as post-processing on MuScriptor transcription output, surfaced via MIDI meta-events, CLI output, and web UI.

**Architecture:** Six pure-Python utility modules in `muscriptor/utils/` that operate on the `Note` list produced by the transcription pipeline. No new model dependencies — all detection is heuristic (IOI clustering, Krumhansl-Schmuckler profiles, chord template matching). The new `muscriptor/utils/midi.py` is augmented to emit MIDI meta-events. The CLI and web server are updated to pass through the new data.

**Tech Stack:** numpy, mido (already deps), standard library. No librosa — keep dependency-free.

**Current state:**
- `muscriptor/tokenizer/notes.py` — `Note(is_drum, program, onset, offset, pitch)`, `note_event2midi()` emits no meta-events
- `muscriptor/utils/midi.py` — `notes_to_midi()` wraps with hardcoded tempo_bpm=120
- `muscriptor/transcription_model.py` — `events_to_midi_bytes()` calls `notes_to_midi(notes)` with no tempo/key
- `muscriptor/main.py` — CLI transcribe command, `--format jsonl` outputs event stream
- `muscriptor/server.py` — FastAPI SSE endpoint
- `web/src/` — React piano roll UI

---

### Task 1: BPM detection from note onsets

**Objective:** Estimate tempo (BPM) from a list of `Note` objects using inter-onset-interval (IOI) histogram clustering.

**Files:**
- Create: `muscriptor/utils/tempo.py`
- Test: `tests/test_tempo.py`

**Implementation:**

```python
"""BPM detection and beat-grid generation from transcribed notes."""

from collections import Counter
from muscriptor.tokenizer.notes import Note

_MIN_BPM = 40
_MAX_BPM = 240


def detect_tempo(notes: list[Note]) -> float:
    """Estimate tempo in BPM from note onset IOI clustering.

    Strategy:
      1. Collect all inter-onset intervals (IOIs) from non-drum notes.
      2. For each IOI, compute candidate BPM = 60.0 / IOI.
      3. Bin candidates in a 1 BPM histogram, pick the dominant bin.
      4. Return the candidate BPM (clamped to [40, 240]).

    Returns 120.0 if too few notes (< 3) to estimate.
    """
    onsets = sorted(set(n.onset for n in notes if not n.is_drum))
    if len(onsets) < 3:
        return 120.0

    io_is = [onsets[i+1] - onsets[i] for i in range(len(onsets)-1) if onsets[i+1] - onsets[i] > 0.05]
    if not io_is:
        return 120.0

    candidates = []
    for ioi in io_is:
        bpm = 60.0 / ioi
        if _MIN_BPM <= bpm <= _MAX_BPM:
            candidates.append(round(bpm))

    if not candidates:
        return 120.0

    counter = Counter(candidates)
    best_bpm, _ = counter.most_common(1)[0]
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
        score = sum(1 for o in onset_times if abs((o - start - phase) % beat_dur) < 0.03)
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
```

**Tests (`tests/test_tempo.py`):**
```python
from muscriptor.tokenizer.notes import Note
from muscriptor.utils.tempo import detect_tempo, beat_times

def test_detect_tempo_quarter_notes():
    # 120 BPM: quarter notes every 0.5s
    notes = [Note(is_drum=False, program=0, onset=i*0.5, offset=i*0.5+0.1, pitch=60) for i in range(8)]
    assert detect_tempo(notes) == 120.0

def test_detect_tempo_eighth_notes():
    # 120 BPM: eighth notes every 0.25s → detects 120
    notes = [Note(is_drum=False, program=0, onset=i*0.25, offset=i*0.25+0.05, pitch=60) for i in range(16)]
    assert detect_tempo(notes) == 120.0

def test_detect_tempo_empty_notes():
    assert detect_tempo([]) == 120.0

def test_beat_times_length():
    beats = beat_times([0.0, 0.5, 1.0, 1.5, 2.0], 120.0)
    assert len(beats) >= 4

def test_beat_times_spacing():
    beats = beat_times([0.0, 0.5, 1.0], 120.0)
    for i in range(1, len(beats)):
        assert abs(beats[i] - beats[i-1] - 0.5) < 0.01
```

**Run:** `cd /home/martin/work/code/AI/muscriptor && source ~/work/venv/tinyTT/bin/activate && python3 -m pytest tests/test_tempo.py -v`
Expected: 5 passed

---

### Task 2: Key detection from pitch histogram

**Objective:** Determine the musical key (C major / A minor etc.) using the Krumhansl-Schmuckler key-finding algorithm on the transcribed notes.

**Files:**
- Create: `muscriptor/utils/key.py`
- Test: `tests/test_key.py`

**Implementation:**

```python
"""Key detection using Krumhansl-Schmuckler profiles."""

from collections import Counter
from muscriptor.tokenizer.notes import Note

# Krumhansl-Kessler major key profiles (normalized)
_MAJOR_PROFILES = {
    0: [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],  # C
    1: [2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29],  # C#
    2: [2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66],  # D
    3: [3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39],  # D#/Eb
    4: [2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19],  # E
    5: [5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52],  # F
    6: [2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09],  # F#/Gb
    7: [4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38],  # G
    8: [4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33],  # G#/Ab
    9: [2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48],  # A
    10: [3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23],  # A#/Bb
    11: [2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35],  # B
}

_MINOR_PROFILES = {
    0: [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],  # Am
    1: [3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34],  # A#m/Bbm
    2: [3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69],  # Bm
    3: [2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98],  # Cm
    4: [3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75],  # C#m/Dbm
    5: [4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54],  # Dm
    6: [2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53],  # D#m/Ebm
    7: [3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60],  # Em
    8: [2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38],  # Fm
    9: [5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52],  # F#m/Gbm
    10: [3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68],  # Gm
    11: [2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33],  # G#m/Abm
}

_PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _pitch_class_histogram(notes: list[Note]) -> list[float]:
    """Normalised 12-bin pitch-class histogram from note onset counts."""
    pc = Counter()
    for n in notes:
        if not n.is_drum:
            pc[n.pitch % 12] += 1
    total = sum(pc.values())
    if total == 0:
        return [1.0] * 12
    return [pc.get(i, 0) / total * 100 for i in range(12)]


def _correlate(hist: list[float], profile: list[float]) -> float:
    """Pearson-like correlation between hist and profile."""
    import numpy as np
    h = np.array(hist)
    p = np.array(profile)
    return float(np.corrcoef(h, p)[0, 1])


def detect_key(notes: list[Note]) -> tuple[str, str]:
    """Return (key_name, mode) e.g. ('C', 'major') or ('A', 'minor')."""
    hist = _pitch_class_histogram(notes)

    best_corr = -float("inf")
    best_key = 0
    best_mode = "major"

    for root, profile in _MAJOR_PROFILES.items():
        corr = _correlate(hist, profile)
        if corr > best_corr:
            best_corr = corr
            best_key = root
            best_mode = "major"

    for root, profile in _MINOR_PROFILES.items():
        corr = _correlate(hist, profile)
        if corr > best_corr:
            best_corr = corr
            best_key = root
            best_mode = "minor"

    return _PITCH_CLASS_NAMES[best_key], best_mode


def key_signature(key_name: str, mode: str) -> int:
    """Convert key name + mode to MIDI key signature byte.

    Returns a signed integer: negative = flats, positive = sharps.
    e.g. key_signature('C', 'major') -> 0
         key_signature('G', 'major') -> 1
         key_signature('F', 'major') -> -1
         key_signature('A', 'minor') -> 0
    """
    # Circle of fifths for major keys
    major_sharps = {"C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6,
                    "C#": 7, "F": -1, "Bb": -2, "Eb": -3, "Ab": -4,
                    "Db": -5, "Gb": -6, "Cb": -7}
    # Relative minor = major - 3 semitones
    relative_map = {"C": "A", "C#": "A#", "D": "B", "D#": "C", "E": "C#",
                    "F": "D", "F#": "D#", "G": "E", "G#": "F", "A": "F#",
                    "A#": "G", "B": "G#"}
    if mode == "minor":
        key_name = relative_map.get(key_name, key_name)
    return major_sharps.get(key_name, 0)
```

**Tests:**
```python
from muscriptor.tokenizer.notes import Note
from muscriptor.utils.key import detect_key, key_signature

def test_detect_key_c_major():
    # C major: mostly C, E, G notes
    notes = [Note(is_drum=False, program=0, onset=i*0.5, offset=i*0.5+0.2, pitch=(60 + [0, 4, 7][i % 3])) for i in range(15)]
    key, mode = detect_key(notes)
    assert key == "C"
    assert mode == "major"

def test_detect_key_a_minor():
    # A minor: mostly A, C, E notes
    notes = [Note(is_drum=False, program=0, onset=i*0.5, offset=i*0.5+0.2, pitch=(57 + [0, 3, 7][i % 3])) for i in range(15)]
    key, mode = detect_key(notes)
    assert key == "A"
    assert mode == "minor"

def test_detect_key_empty():
    key, mode = detect_key([])
    assert isinstance(key, str)
    assert isinstance(mode, str)

def test_key_signature_c_major():
    assert key_signature("C", "major") == 0

def test_key_signature_g_major():
    assert key_signature("G", "major") == 1

def test_key_signature_a_minor():
    assert key_signature("A", "minor") == 0  # relative to C
```

---

### Task 3: Chord detection at beat boundaries

**Objective:** At each beat, collect active pitch classes and label with the best-matching chord template.

**Files:**
- Create: `muscriptor/utils/chords.py`
- Test: `tests/test_chords.py`

```python
"""Chord detection from transcribed notes at beat boundaries."""

from muscriptor.tokenizer.notes import Note

# Chord templates as 12-bit masks
_CHORD_TEMPLATES: dict[str, tuple[str, int]] = {
    "maj":  ("",    0b100010010000),  # 0-4-7
    "min":  ("m",   0b100100010000),  # 0-3-7
    "dim":  ("dim", 0b100100001000),  # 0-3-6
    "aug":  ("aug", 0b100010001000),  # 0-4-8
    "sus4": ("sus4",0b10001000100),   # 0-5-7
    "dom7": ("7",   0b100010010001),  # 0-4-7-10
    "maj7": ("maj7",0b100010010100),  # 0-4-7-11
    "min7": ("m7",  0b100100010001),  # 0-3-7-10
    "dim7": ("dim7",0b100100001001),  # 0-3-6-9
}

_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _active_pitch_classes(notes: list[Note], t: float, window: float = 0.05) -> int:
    """Bitmask of pitch classes active at time t (within ±window)."""
    mask = 0
    for n in notes:
        if n.is_drum:
            continue
        # Note is "active" if t is between onset and offset
        if n.onset - window <= t <= n.offset + window:
            mask |= 1 << (n.pitch % 12)
    return mask


def _best_chord(pc_mask: int) -> tuple[str, str] | None:
    """Find best matching chord root and quality for a pitch class mask."""
    if pc_mask == 0:
        return None

    best_root = 0
    best_quality = ""
    best_score = -1

    for root in range(12):
        rotated = ((pc_mask >> root) | (pc_mask << (12 - root))) & 0xFFF
        for qname, (qsuffix, qmask) in _CHORD_TEMPLATES.items():
            # Score: intersection size minus notes outside template
            match = bin(rotated & qmask).count("1")
            extra = bin(rotated & ~qmask).count("1")
            score = match - extra
            if score > best_score:
                best_score = score
                best_root = root
                best_quality = qsuffix

    return _PITCH_NAMES[best_root], best_quality


def detect_chords(
    notes: list[Note],
    beat_times: list[float],
) -> list[tuple[float, str]]:
    """Label each beat with its best-matching chord.

    Returns list of (beat_time, chord_label) tuples, e.g.
    [(0.0, 'C'), (0.5, 'Am'), (1.0, 'F')].
    """
    result = []
    for bt in beat_times:
        chord = _best_chord(_active_pitch_classes(notes, bt))
        if chord is not None:
            result.append((bt, f"{chord[0]}{chord[1]}"))
        else:
            if result:
                result.append((bt, result[-1][1]))  # sustain last chord
            else:
                result.append((bt, "N.C."))
    return result
```

**Tests:**
```python
from muscriptor.tokenizer.notes import Note
from muscriptor.utils.chords import detect_chords

def test_detect_c_major_chord():
    # C, E, G notes active at beat 0
    notes = [
        Note(is_drum=False, program=0, onset=0.0, offset=0.5, pitch=60),  # C
        Note(is_drum=False, program=0, onset=0.0, offset=0.5, pitch=64),  # E
        Note(is_drum=False, program=0, onset=0.0, offset=0.5, pitch=67),  # G
    ]
    chords = detect_chords(notes, [0.0])
    assert len(chords) >= 1
    assert "C" in chords[0][1]

def test_detect_am_chord():
    notes = [
        Note(is_drum=False, program=0, onset=0.0, offset=0.5, pitch=57),  # A
        Note(is_drum=False, program=0, onset=0.0, offset=0.5, pitch=60),  # C
        Note(is_drum=False, program=0, onset=0.0, offset=0.5, pitch=64),  # E
    ]
    chords = detect_chords(notes, [0.0])
    assert len(chords) >= 1
    assert "Am" in chords[0][1] or "A" in chords[0][1]
```

---

### Task 4: Quantization — snap notes to nearest beat grid

**Objective:** Quantize note onset/offset times to the nearest grid subdivision (16th notes at detected BPM).

**Files:**
- Create: `muscriptor/utils/quantize.py`
- Test: `tests/test_quantize.py`

```python
"""Note quantization to a beat grid."""

from muscriptor.tokenizer.notes import Note


def quantize_time(t: float, grid: list[float]) -> float:
    """Snap a time value to the nearest grid point."""
    if not grid:
        return t
    # Binary search for nearest
    lo, hi = 0, len(grid) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if grid[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    # Check neighbours
    best = grid[lo]
    if lo > 0 and abs(grid[lo - 1] - t) < abs(best - t):
        best = grid[lo - 1]
    if lo + 1 < len(grid) and abs(grid[lo + 1] - t) < abs(best - t):
        best = grid[lo + 1]
    return best


def quantize_notes(
    notes: list[Note],
    beat_times_eighth: list[float],
) -> list[Note]:
    """Snap all note onsets and offsets to the nearest eighth-note grid.

    Args:
        notes: Raw transcribed notes.
        beat_times_eighth: Beat grid at eighth-note resolution
            (2× quarter-note beat density).

    Returns:
        New list of quantized Note objects.
    """
    if not beat_times_eighth:
        return list(notes)

    quantized = []
    for n in notes:
        q_onset = quantize_time(n.onset, beat_times_eighth)
        q_offset = quantize_time(n.offset, beat_times_eighth)
        if q_offset <= q_onset:
            q_offset = q_onset + 0.01  # minimum duration
        quantized.append(Note(
            is_drum=n.is_drum,
            program=n.program,
            onset=q_onset,
            offset=q_offset,
            pitch=n.pitch,
        ))
    return quantized
```

**Tests:**
```python
from muscriptor.tokenizer.notes import Note
from muscriptor.utils.quantize import quantize_notes

def test_quantize_snaps_to_grid():
    grid = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    notes = [Note(is_drum=False, program=0, onset=0.12, offset=0.26, pitch=60)]
    q = quantize_notes(notes, grid)
    assert q[0].onset == 0.125
    assert q[0].offset == 0.25

def test_quantize_empty_grid():
    notes = [Note(is_drum=False, program=0, onset=0.5, offset=0.8, pitch=60)]
    q = quantize_notes(notes, [])
    assert q == notes

def test_quantize_preserves_drums():
    grid = [0.0, 0.5, 1.0]
    notes = [Note(is_drum=True, program=128, onset=0.45, offset=0.46, pitch=36)]
    q = quantize_notes(notes, grid)
    assert q[0].onset == 0.5
```

---

### Task 5: Augment MIDI export with tempo, key signature, time signature meta-events

**Objective:** `notes_to_midi()` and `save_midi()` accept optional `bpm`, `key`, `mode`, `time_signature` parameters and emit corresponding `MetaMessage` events.

**Files:**
- Modify: `muscriptor/utils/midi.py`
- Test: update `tests/test_midi.py`

**Changes to `muscriptor/utils/midi.py`:**

```python
"""MIDI output utilities."""

from pathlib import Path

from mido import MetaMessage, MidiFile, MidiTrack
from muscriptor.tokenizer.notes import Note, note2note_event, note_event2midi


def notes_to_midi(
    notes: list[Note],
    velocity: int = 100,
    tempo_bpm: int = 120,
    key: str | None = None,
    key_mode: str | None = None,
    time_signature: tuple[int, int] | None = None,
) -> MidiFile:
    """Convert a list of Note objects to a mido MidiFile.

    If key/key_mode are provided, a KeySignature meta event is written.
    If time_signature is provided, a TimeSignature meta event is written.
    Tempo is always written as a SetTempo meta event.
    """
    from muscriptor.utils.key import key_signature  # avoid circular import on first load

    note_events = note2note_event(notes)
    tempo_us = int(60_000_000 / tempo_bpm)
    return note_event2midi(
        note_events,
        output_file=None,
        velocity=velocity,
        tempo=tempo_us,
        key=key,
        key_mode=key_mode,
        time_sig=time_signature,
    )


def save_midi(
    notes: list[Note],
    path: str | Path,
    velocity: int = 100,
    tempo_bpm: int = 120,
    key: str | None = None,
    key_mode: str | None = None,
    time_signature: tuple[int, int] | None = None,
) -> None:
    """Save a list of Note objects as a MIDI file."""
    midi = notes_to_midi(
        notes,
        velocity=velocity,
        tempo_bpm=tempo_bpm,
        key=key,
        key_mode=key_mode,
        time_signature=time_signature,
    )
    midi.save(str(path))
```

And update `note_event2midi` in `muscriptor/tokenizer/notes.py` to accept and emit meta events:

**Changes to `muscriptor/tokenizer/notes.py`:**

In `note_event2midi()`, add parameters:
```python
def note_event2midi(
    note_events: list[NoteEvent],
    output_file: str | os.PathLike | None = None,
    velocity: int = 100,
    ticks_per_beat: int = 480,
    tempo: int = 500000,
    key: str | None = None,
    key_mode: str | None = None,
    time_sig: tuple[int, int] | None = None,
) -> MidiFile:
```

Insert after `midi = MidiFile(ticks_per_beat=ticks_per_beat, type=0)` and `track = MidiTrack()`:

```python
    # Tempo meta event
    track.append(MetaMessage("set_tempo", tempo=tempo, time=0))
    # Key signature meta event
    if key is not None and key_mode is not None:
        from muscriptor.utils.key import key_signature
        sf = key_signature(key, key_mode)
        track.append(MetaMessage("key_signature", key=sf, time=0))
    # Time signature meta event
    if time_sig is not None:
        track.append(MetaMessage(
            "time_signature",
            numerator=time_sig[0],
            denominator=time_sig[1],
            time=0,
        ))
```

**Add imports to notes.py:** `from mido import MetaMessage` (add to the existing `from mido import ...` line).

**New tests:**
```python
def test_notes_to_midi_with_meta_events():
    notes = _sample_notes()
    midi = notes_to_midi(notes, tempo_bpm=120, key="C", key_mode="major", time_signature=(4, 4))
    track = midi.tracks[0]
    # Check meta messages exist
    meta_types = [msg.type for msg in track if msg.is_meta]
    assert "set_tempo" in meta_types
    assert "key_signature" in meta_types
    assert "time_signature" in meta_types
```

---

### Task 6: Integrate detection pipeline in transcription output

**Objective:** After transcribing, run BPM → beat grid → key → chords → quantization, then emit enhanced MIDI. Add optional `--detect` / `--quantize` flags to CLI.

**Files:**
- Modify: `muscriptor/transcription_model.py` — add `detect_bpm`, `detect_key`, `detect_chords` methods
- Modify: `muscriptor/main.py` — add `--detect`, `--quantize`, `--bpm`, `--key`, `--chords` CLI flags
- Test: `tests/test_transcription_model.py` (new integration-like tests)

**Key additions to `muscriptor/transcription_model.py`:**

```python
from muscriptor.utils.tempo import detect_tempo, beat_times as generate_beat_times
from muscriptor.utils.key import detect_key as detect_key_from_notes
from muscriptor.utils.chords import detect_chords as detect_chords_from_notes
from muscriptor.utils.quantize import quantize_notes as quantize_notes_func
from muscriptor.utils.midi import notes_to_midi as enhanced_notes_to_midi


class TranscriptionModel:
    # ... existing code ...

    def events_to_midi_bytes(
        self,
        events: Iterator[NoteStartEvent | NoteEndEvent | ProgressEvent],
        detect: bool = False,
        quantize: bool = False,
    ) -> bytes:
        """Reassemble Notes and optionally run detection + quantization."""
        notes: list[Note] = []
        open_notes: dict[int, Note] = {}
        for ev in events:
            if isinstance(ev, ProgressEvent):
                continue
            if isinstance(ev, NoteStartEvent):
                is_drum = ev.instrument == "drums"
                program = (
                    DRUM_PROGRAM
                    if is_drum
                    else self._program_for_instrument(ev.instrument)
                )
                note = Note(
                    is_drum=is_drum,
                    program=program,
                    onset=ev.start_time,
                    offset=ev.start_time,
                    pitch=ev.pitch,
                )
                open_notes[ev.index] = note
            else:
                note = open_notes.pop(ev.start_event_index)
                note.offset = ev.end_time
                notes.append(note)

        notes = validate_notes(notes, fix=True)
        notes = trim_overlapping_notes(notes, sort=True)

        bpm = None
        key = None
        key_mode = None
        time_sig = None
        chord_info = None

        if detect or quantize:
            bpm = detect_tempo(notes)
            eighth_beats = generate_beat_times(
                [n.onset for n in notes], bpm
            )
            # Eighth note grid: beat_times at half-beat intervals
            eighth_times = []
            for b in eighth_beats:
                eighth_times.append(b)
                eighth_times.append(b + 30.0 / bpm)  # half beat
            eighth_times = sorted(set(round(t, 4) for t in eighth_times))

            if quantize:
                notes = quantize_notes_func(notes, eighth_times)

            if detect:
                key, key_mode = detect_key_from_notes(notes)
                chord_info = detect_chords_from_notes(notes, eighth_beats)
                time_sig = (4, 4)  # default; could be detected later

        midi = enhanced_notes_to_midi(
            notes,
            tempo_bpm=int(round(bpm)) if bpm else 120,
            key=key,
            key_mode=key_mode,
            time_signature=time_sig,
        )
        buf = io.BytesIO()
        midi.save(file=buf)
        return buf.getvalue()
```

**CLI flags (`muscriptor/main.py`):**

```python
detect: Annotated[
    bool,
    typer.Option("--detect", help="Detect BPM, key, and chords from transcription")
] = False,

quantize: Annotated[
    bool,
    typer.Option("--quantize", help="Quantize notes to detected beat grid")
] = False,
```

Pass `detect` and `quantize` to `model.transcribe_to_midi()`.

When `--detect` is used, print info to stderr:
```
BPM: 120, Key: C major, Chords: C | Am | F | G
```

---

### Task 7: Display BPM/key/chords in web UI (frontend)

**Objective:** The web UI piano roll gains a status bar showing detected BPM, key, and the current chord label as playback progresses.

**Files:**
- Modify: `web/src/App.tsx` (or equivalent main component)
- Modify: `muscriptor/server.py` — expose detection endpoint or enhance SSE

**Backend change (server.py):** Add optional `detect=true` query parameter to `/transcribe` SSE endpoint. When set, after transcription completes, send a final metadata event:
```
event: metadata
data: {"bpm": 120, "key": "C", "mode": "major", "chords": [[0.0, "C"], [0.5, "Am"], ...]}
```

**Frontend change:** Parse the `metadata` SSE event. Display BPM and key in a status bar. During playback, consult the chord list to show the current chord label (using the current playback time to look up the chord active at that time).

**Note:** This task requires TypeScript and React knowledge. The piano roll component is in the existing web codebase under `web/src/`.

---

### Task 8: Verify end-to-end with real audio

**Objective:** Run the full pipeline on the test audio file and confirm output.

**Files:**
- Run: the `test_c4e4.wav` file already in the repo

**Command:**
```bash
cd /home/martin/work/code/AI/muscriptor
source ~/work/venv/tinyTT/bin/activate
HF_TOKEN="..." python3 -m muscriptor transcribe test_c4e4.wav \
  --model small --detect --quantize -o test_enhanced.mid
```

Check MIDI file structure:
```bash
python3 -c "
from mido import MidiFile
m = MidiFile('test_enhanced.mid')
for i, track in enumerate(m.tracks):
    for msg in track:
        print(msg)
"
```

Verify meta events (set_tempo, key_signature) appear and note timings are grid-aligned.

---

### Verification checklist

- [ ] `detect_tempo()` returns 120 for quarter-note patterns at 0.5s intervals
- [ ] `beat_times()` returns evenly spaced times at the correct interval
- [ ] `detect_key()` identifies C major / A minor from synthetic note lists
- [ ] `key_signature()` maps key names to signed integers
- [ ] `detect_chords()` labels C major and A minor triads correctly
- [ ] `quantize_notes()` snaps to the nearest grid point
- [ ] MIDI export with meta events: set_tempo, key_signature, time_signature in track 0
- [ ] CLI `--detect` prints BPM/key/chords to stderr
- [ ] CLI `--quantize` output has beat-aligned timestamps
- [ ] Web UI status bar shows BPM/key (if Task 7 done)
- [ ] All unit tests pass
- [ ] `git commit` after each task
