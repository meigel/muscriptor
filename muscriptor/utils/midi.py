"""MIDI output utilities."""

from pathlib import Path

from muscriptor.tokenizer.notes import Note, note2note_event, note_event2midi


def notes_to_midi(
    notes: list[Note],
    velocity: int = 100,
    tempo_bpm: int = 120,
    key: str | None = None,
    key_mode: str | None = None,
    time_signature: tuple[int, int] | None = None,
    program_names: dict[int, str] | None = None,
) -> "MidiFile":
    """Convert a list of Note objects to a mido MidiFile.

    Args:
        notes: Note objects to convert.
        velocity: Default note-on velocity.
        tempo_bpm: Tempo in beats per minute.
        key: Key name (e.g. 'C', 'G', 'F#') for key signature meta event.
        key_mode: 'major' or 'minor' for key signature meta event.
        time_signature: (numerator, denominator) for time signature meta event.
        program_names: Optional dict mapping program numbers to track names.
    """
    from mido import MidiFile

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
        program_names=program_names,
    )


def save_midi(
    notes: list[Note],
    path: str | Path,
    velocity: int = 100,
    tempo_bpm: int = 120,
    key: str | None = None,
    key_mode: str | None = None,
    time_signature: tuple[int, int] | None = None,
    program_names: dict[int, str] | None = None,
) -> None:
    """Save a list of Note objects as a MIDI file."""
    midi = notes_to_midi(
        notes,
        velocity=velocity,
        tempo_bpm=tempo_bpm,
        key=key,
        key_mode=key_mode,
        time_signature=time_signature,
        program_names=program_names,
    )
    midi.save(str(path))
