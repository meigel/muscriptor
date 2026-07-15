"""Tests for muscriptor/utils/midi.py."""

import tempfile
from pathlib import Path

from mido import MidiFile

from muscriptor.tokenizer.notes import Note
from muscriptor.utils.midi import notes_to_midi, save_midi


def _sample_notes():
    return [
        Note(is_drum=False, program=0, onset=0.0, offset=0.5, pitch=60),
        Note(is_drum=False, program=0, onset=0.5, offset=1.0, pitch=64),
        Note(is_drum=True, program=128, onset=0.0, offset=0.01, pitch=36),
    ]


def test_notes_to_midi_returns_midi_file():
    midi = notes_to_midi(_sample_notes())
    assert isinstance(midi, MidiFile)


def test_notes_to_midi_has_tracks():
    midi = notes_to_midi(_sample_notes())
    assert len(midi.tracks) > 0


def test_notes_to_midi_custom_tempo():
    midi = notes_to_midi(_sample_notes(), tempo_bpm=90)
    assert isinstance(midi, MidiFile)


def test_notes_to_midi_empty_notes():
    midi = notes_to_midi([])
    assert isinstance(midi, MidiFile)


def test_save_midi_creates_file():
    notes = _sample_notes()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "out.mid"
        save_midi(notes, path)
        assert path.exists()
        assert path.stat().st_size > 0


def test_save_midi_is_valid_midi():
    notes = _sample_notes()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "out.mid"
        save_midi(notes, path)
        loaded = MidiFile(str(path))
        assert len(loaded.tracks) > 0


def test_save_midi_string_path():
    notes = _sample_notes()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "out.mid")
        save_midi(notes, path)
        assert Path(path).exists()


def test_notes_to_midi_with_meta_events():
    """MIDI should contain set_tempo, key_signature, time_signature meta events."""
    notes = _sample_notes()
    midi = notes_to_midi(
        notes,
        tempo_bpm=120,
        key="C",
        key_mode="major",
        time_signature=(4, 4),
    )
    track = midi.tracks[0]
    meta_types = [msg.type for msg in track if msg.is_meta]
    assert "set_tempo" in meta_types
    assert "key_signature" in meta_types
    assert "time_signature" in meta_types


def test_notes_to_midi_key_signature_g_major():
    """G major meta event key string should be 'G'."""
    notes = _sample_notes()
    midi = notes_to_midi(notes, key="G", key_mode="major")
    for msg in midi.tracks[0]:
        if msg.type == "key_signature":
            assert msg.key == "G"
            return
    assert False, "no key_signature meta event found"


def test_notes_to_midi_key_signature_a_minor():
    """A minor meta event key string should be 'Am'."""
    notes = _sample_notes()
    midi = notes_to_midi(notes, key="A", key_mode="minor")
    for msg in midi.tracks[0]:
        if msg.type == "key_signature":
            assert msg.key == "Am"
            return
    assert False, "no key_signature meta event found"


def test_notes_to_midi_tempo_change():
    """Tempo meta event should reflect the requested BPM."""
    notes = _sample_notes()
    midi = notes_to_midi(notes, tempo_bpm=90)
    for msg in midi.tracks[0]:
        if msg.type == "set_tempo":
            expected_us = int(60_000_000 / 90)
            assert msg.tempo == expected_us
            return
    assert False, "no set_tempo meta event found"
