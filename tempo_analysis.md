# Tempo Evaluation: Electric Callboy - PUMP IT Velvet Remix

## Track Metadata
- **Track**: Electric Callboy - PUMP IT Velvet Remix (hard electronic remix)
- **Duration**: 164 seconds
- **Total notes**: 2981 (note_on events before quantization)
- **Key**: F minor
- **Detected tempo**: 174 BPM
- **Dominant IOI**: ~0.349s (most common gap between consecutive note onsets)
- **Detected chords**: 449 (one per beat at chosen BPM)

---

## 1. IOI Analysis

### Computation
Base BPM from dominant IOI: `60 / 0.349 = 171.9 BPM`

The algorithm generates candidates by multiplying base BPMs by factors {0.125, 0.25, 0.5, 1, 2, 4, 8}. From the 0.349s IOI:

| Multiplier | BPM candidate | Note value at this tempo | Status |
|------------|---------------|--------------------------|--------|
| 0.125      | 21.5 BPM      | Whole note               | ❌ Below _MIN_BPM (40) |
| 0.25       | 43 BPM        | Half note                | ✅ Candidate |
| 0.5        | 86 BPM        | Quarter note             | ✅ Candidate |
| 1          | **172 BPM**   | Quarter note at 172      | ✅ Candidate (≈174) |
| 2          | 344 BPM       | Eighth note              | ❌ Above _MAX_BPM (320) |
| 4          | 688 BPM       | 16th note                | ❌ Clipped |
| 8          | 1376 BPM      | 32nd note                | ❌ Clipped |

### What Each Candidate Means for the Dominant 0.349s IOI

| Candidate BPM | 0.349s gap = | Quarter note duration |
|---------------|--------------|----------------------|
| **174 BPM**   | Quarter note (0.345s) | 0.345s ✅ near-perfect match |
| **86 BPM**    | Eighth note (0.349s) | 0.698s |
| **43 BPM**    | 16th note (0.349s) | 1.395s |

**Key insight**: The dominant 0.349s IOI maps to a quarter note at ~172 BPM with negligible error (0.349 vs 0.345 = 1.1% difference). This is the most natural interpretation — the most common spacing between non-drum notes is exactly one quarter note.

---

## 2. Algorithm Behavior Analysis

The source code in `musciptor/utils/tempo.py` reveals the tiebreaking logic:

1. **Line 143**: If raw alignment ≥ 105% of best → higher wins regardless
2. **Line 146-153**: If alignment ≥ 95% of best (close):
   - If measure clarity > best + 0.5 → higher clarity wins
   - **Otherwise → prefer SLOWER tempo**

Since the algorithm **explicitly prefers slower tempos when scores are close**, and it still chose **174 BPM over 86 BPM**, this means:

> **174 BPM had significantly better grid alignment (>5% better) OR dramatically better measure clarity (>0.5 ratio better) than 86 BPM.**

This is strong evidence that 174 BPM is the correct musical tempo — the algorithm had to be forced out of its slow-tempo bias.

---

## 3. Genre Conventions

| Genre | Typical BPM range |
|-------|-------------------|
| House / Big Room | 120-130 BPM |
| Trance | 128-140 BPM |
| Hardstyle | 148-155 BPM |
| Dubstep | 140-150 BPM (half-time feel) |
| **Hardcore / Gabber** | **160-200 BPM** |
| Frenchcore | 180-220 BPM |
| Terror / Speedcore | 200-300+ BPM |

Electric Callboy (formerly Eskimo Callboy) is a German **electronicore/metalcore** band. Their original "Pump It" (from the *Tekkno* album, 2022) blends metalcore with heavy EDM elements. A "hard electronic remix" pushes this further into hardcore/hardstyle territory.

**174 BPM is firmly in hardcore/gabber territory** — a natural tempo for a hard electronic remix of a high-energy metalcore track. This is exactly what one would expect.

**86 BPM** would place this in trip-hop/lo-fi territory — completely wrong for a "hard electronic remix."
**128 BPM** is standard house/trance — plausible but not "hard electronic," and unsupported by the IOI data.

---

## 4. Why 128 BPM Fails

The user explicitly asked about 128 BPM. At 128 BPM:
- Quarter note = 0.469s
- The dominant 0.349s IOI → 0.349/0.469 = **0.744 of a quarter note** (≈ dotted eighth)
- A dotted eighth as the *dominant* note spacing is unusual in any genre
- No natural harmonic: 128 BPM doesn't appear from any multiplier of 60/0.349
- The algorithm's multiplier set {0.125, 0.25, 0.5, 1, 2, 4, 8} does not include 0.75 (dotted), so 128 BPM would only arise from a *different* IOI in the distribution — not the dominant one

**128 BPM is unsupported by the primary rhythmic evidence.**

---

## 5. Chord Density as Consistency Check

The chord detector (`musciptor/utils/chords.py`) labels **one chord per beat** (line 94: `for bt in beat_times`). So 449 chords = 449 beat positions.

### At 174 BPM:
- 449 beats × (60/174) seconds/beat = **154.8 seconds** of active music
- Track is 164 seconds → ~9s of silence at edges ✅ **Very reasonable**

### At 86 BPM:
- Same 449 beats would span 449 × (60/86) = **313 seconds** — nearly double the track length ❌
- In reality, at 86 BPM the algorithm would detect ~238 beats for a 164s track

The 449 detected chord labels are consistent with ~174 BPM.

---

## 6. Musical Structure Considerations

- **2981 notes in 164s** = ~18.2 notes/second (all notes including drums)
- At 174 BPM (2.9 beats/sec): ~6.3 notes per beat — dense but standard for hard electronic
- At 86 BPM (1.43 beats/sec): ~12.7 notes per beat — implausibly dense for non-drum material
- The key of **F minor** is very common in hardstyle/hardcore production (dark, aggressive tonality)

---

## VERDICT: 174 BPM ✅

**174 BPM is the correct tempo.** Here's the evidence chain:

1. **IOI evidence**: The dominant 0.349s interval maps to quarter notes at ~172 BPM with 1.1% error — a near-perfect fit
2. **Algorithm behavior**: The algorithm*prefers slower tempos* as a tiebreaker, yet chose 174 over 86, meaning 174 had significantly better alignment or clarity
3. **Genre fit**: 174 BPM sits squarely in hardcore/gabber territory — exactly what a "hard electronic remix" of a metalcore band should be
4. **Chord consistency**: 449 beat positions span ~155s of a 164s track, correct for 174 BPM
5. **Note density**: ~6 notes/beat is appropriate for dense electronic percussion and synths

| Candidate | Verdict | Reason |
|-----------|---------|--------|
| **174 BPM** | ✅ **Correct** | Primary IOI = quarter note; genre-appropriate; algorithm evidence |
| **86 BPM** | ❌ Wrong | Too slow for hard electronic; implausible note density; algorithm rejected it despite slow preference |
| **43 BPM** | ❌ Wrong | Impractically slow; 16th note spacing as dominant gap makes no musical sense |
| **128 BPM** | ❌ Wrong | No natural harmonic from IOI; 0.349s ≠ clean subdivision at 128; wrong genre |
