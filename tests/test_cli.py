"""CLI tests — checks that `-o -` writes a clean JSONL stream to stdout
and that all progress/timing chatter goes to stderr.

Patches the model loader with a fake so the test stays hermetic.
"""

import json
import wave
from pathlib import Path

import pytest
from typer.testing import CliRunner

import muscriptor.main as main_mod
from muscriptor.events import NoteEndEvent, NoteStartEvent
from muscriptor.utils.midi import ExportOptions


class _FakeInner:
    """Stand-in for the inner torch module; the CLI casts it with `.to(...)`."""

    def to(self, *_args, **_kwargs):
        return self


class _FakeModel:
    # kwargs of the most recent transcribe() call (reset by `patched_model`).
    last_kwargs: dict | None = None

    def __init__(self):
        self._model = _FakeInner()

    @classmethod
    def load_model(cls, **_):
        return cls()

    def transcribe(self, **kwargs):
        type(self).last_kwargs = kwargs
        s0 = NoteStartEvent(pitch=60, start_time=0.0, index=0, instrument="piano")
        s1 = NoteStartEvent(pitch=64, start_time=0.5, index=1, instrument="guitar")
        yield s0
        yield NoteEndEvent(end_time=0.4, start_event=s0)
        yield s1
        yield NoteEndEvent(end_time=0.9, start_event=s1)

    def transcribe_and_postprocess(self, **kwargs):
        type(self).last_kwargs = kwargs
        return b"FAKE_MIDI", None


@pytest.fixture
def fake_audio(tmp_path: Path) -> Path:
    p = tmp_path / "silent.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 100)
    return p


@pytest.fixture
def patched_model(monkeypatch):
    _FakeModel.last_kwargs = None
    monkeypatch.setattr(main_mod, "TranscriptionModel", _FakeModel)


def test_jsonl_to_stdout_is_pure_jsonl(patched_model, fake_audio):
    """`-o -` with --format jsonl writes only JSON objects (one per line) to stdout."""
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        ["transcribe", str(fake_audio), "-f", "jsonl", "-o", "-"],
    )
    assert result.exit_code == 0, result.stderr

    # Every non-empty stdout line must be a complete JSON object.
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) == 4
    parsed = [json.loads(ln) for ln in lines]
    assert parsed[0] == {
        "type": "start",
        "pitch": 60,
        "start_time": 0.0,
        "index": 0,
        "instrument": "piano",
    }
    assert parsed[1] == {"type": "end", "end_time": 0.4, "start_event_index": 0}


def test_jsonl_stdout_has_no_chatter(patched_model, fake_audio):
    """Stdout must contain nothing except JSON — no "Loading model…" etc."""
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        ["transcribe", str(fake_audio), "-f", "jsonl", "-o", "-"],
    )
    assert result.exit_code == 0
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        # Throws if the line isn't valid JSON — proves no banner / "Saved to" /
        # timing line leaked through.
        json.loads(line)


def test_progress_messages_go_to_stderr(patched_model, fake_audio):
    """Loading / transcribing banners must appear on stderr, never stdout."""
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        ["transcribe", str(fake_audio), "-f", "jsonl", "-o", "-"],
    )
    assert result.exit_code == 0
    assert "Loading model" in result.stderr
    assert "Transcribing" in result.stderr
    assert "Loading model" not in result.stdout
    assert "Transcribing" not in result.stdout


def test_jsonl_to_file_keeps_progress_on_stderr(patched_model, fake_audio, tmp_path):
    """Even when writing to a file, banners and "Saved JSONL to …" go to stderr."""
    out = tmp_path / "out.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        ["transcribe", str(fake_audio), "-f", "jsonl", "-o", str(out)],
    )
    assert result.exit_code == 0
    assert result.stdout == ""
    assert "Saved JSONL" in result.stderr
    assert out.read_text().splitlines() == [
        json.dumps(d)
        for d in [
            {
                "type": "start",
                "pitch": 60,
                "start_time": 0.0,
                "index": 0,
                "instrument": "piano",
            },
            {"type": "end", "end_time": 0.4, "start_event_index": 0},
            {
                "type": "start",
                "pitch": 64,
                "start_time": 0.5,
                "index": 1,
                "instrument": "guitar",
            },
            {"type": "end", "end_time": 0.9, "start_event_index": 1},
        ]
    ]


def test_batching_requires_disabling_prelude_forcing(patched_model, fake_audio):
    """--batch-size > 1 without --no-prelude-forcing is rejected up front,
    before any model is loaded."""
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        ["transcribe", str(fake_audio), "-b", "4", "-f", "jsonl", "-o", "-"],
    )
    assert result.exit_code == 1
    assert "--no-prelude-forcing" in result.stderr
    assert "Loading model" not in result.stderr
    assert _FakeModel.last_kwargs is None


def test_batching_allowed_with_prelude_forcing_disabled(patched_model, fake_audio):
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        [
            "transcribe",
            str(fake_audio),
            "-b",
            "4",
            "--no-prelude-forcing",
            "-f",
            "jsonl",
            "-o",
            "-",
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert _FakeModel.last_kwargs["batch_size"] == 4
    assert _FakeModel.last_kwargs["prelude_forcing"] is False


def test_instruments_passed_to_model(patched_model, fake_audio):
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        [
            "transcribe",
            str(fake_audio),
            "--instruments",
            "violin,drums",
            "-f",
            "jsonl",
            "-o",
            "-",
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert _FakeModel.last_kwargs["instruments"] == ["violin", "drums"]


def test_midi_timing_options_reach_the_export(patched_model, fake_audio, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        [
            "transcribe",
            str(fake_audio),
            "-o",
            str(tmp_path / "out.mid"),
            "--bpm",
            "92.5",
            "--beats-per-bar",
            "3",
            "--first-downbeat",
            "1.25",
            "--quantize",
            "--grid",
            "3",
            "--quantize-strength",
            "0.75",
            "--fill-gaps",
            "0.5",
            "--trim-leading-bars",
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert _FakeModel.last_kwargs["export"] == ExportOptions(
        bpm=92.5,
        beats_per_bar=3,
        first_downbeat=1.25,
        quantize=True,
        subdivision=3,
        strength=0.75,
        fill_gaps=0.5,
        trim_leading_bars=True,
    )


def test_midi_is_not_quantized_by_default(patched_model, fake_audio, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app, ["transcribe", str(fake_audio), "-o", str(tmp_path / "out.mid")]
    )
    assert result.exit_code == 0, result.stderr
    assert _FakeModel.last_kwargs["export"] == ExportOptions()


def test_tempo_map_options_reach_the_export(patched_model, fake_audio, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app,
        [
            "transcribe",
            str(fake_audio),
            "-o",
            str(tmp_path / "out.mid"),
            "--tempo-map",
            "--beat-scale",
            "0.5",
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert _FakeModel.last_kwargs["export"] == ExportOptions(
        tempo_map=True, beat_scale=0.5
    )


def test_a_bad_beat_scale_fails_before_transcribing(patched_model, fake_audio):
    runner = CliRunner()
    result = runner.invoke(
        main_mod.app, ["transcribe", str(fake_audio), "--beat-scale", "3"]
    )
    assert result.exit_code == 1
    assert "power of two" in result.stderr
    assert _FakeModel.last_kwargs is None


### --yt and the harmony flags


def test_no_input_is_an_error(patched_model):
    result = CliRunner().invoke(main_mod.app, ["transcribe"])
    assert result.exit_code == 1
    assert "no input" in result.stderr


def test_a_file_and_a_url_are_exclusive(patched_model, fake_audio):
    result = CliRunner().invoke(
        main_mod.app, ["transcribe", str(fake_audio), "--yt", "https://example.com/x"]
    )
    assert result.exit_code == 1
    assert "not both" in result.stderr


def test_yt_without_yt_dlp_says_how_to_install_it(patched_model, monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    result = CliRunner().invoke(
        main_mod.app, ["transcribe", "--yt", "https://example.com/watch?v=abc"]
    )
    assert result.exit_code == 1
    assert "yt-dlp" in result.stderr


def test_yt_download_failure_is_reported(patched_model, monkeypatch):
    import shutil
    import subprocess

    class _Failed:
        returncode = 1
        stderr = "ERROR: Video unavailable"

    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/yt-dlp")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Failed())
    result = CliRunner().invoke(
        main_mod.app, ["transcribe", "--yt", "https://example.com/watch?v=abc"]
    )
    assert result.exit_code == 1
    assert "Video unavailable" in result.stderr


def test_yt_saves_next_to_the_cwd_not_in_the_temp_dir(
    patched_model, fake_audio, tmp_path, monkeypatch
):
    """The download is temporary, so the default output must not go with it."""
    monkeypatch.setattr(main_mod, "_download_audio", lambda _url: fake_audio)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        main_mod.app, ["transcribe", "--yt", "https://youtu.be/dQw4w9WgXcQ"]
    )
    assert result.exit_code == 0, result.stderr
    assert (tmp_path / "dQw4w9WgXcQ.mid").exists()


def test_default_output_stem_is_filesystem_safe():
    stem = main_mod._default_output_stem
    assert stem("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert stem("https://www.youtube.com/watch?v=abc123&t=30") == "watch"
    assert stem("https://example.com/") == "example.com"
    assert stem("https://example.com/clip%20one") == "clip_20one"


def test_harmony_flags_need_a_midi_output(patched_model, fake_audio):
    result = CliRunner().invoke(
        main_mod.app,
        ["transcribe", str(fake_audio), "-f", "json", "--detect-key"],
    )
    assert result.exit_code == 1
    assert "--format midi" in result.stderr


def test_detect_flags_reach_the_export_options(patched_model, fake_audio, tmp_path):
    out = tmp_path / "out.mid"
    result = CliRunner().invoke(
        main_mod.app,
        [
            "transcribe",
            str(fake_audio),
            "-o",
            str(out),
            "--detect-key",
            "--detect-chords",
        ],
    )
    assert result.exit_code == 0, result.stderr
    export = _FakeModel.last_kwargs["export"]
    assert export.detect_key is True
    assert export.detect_chords is True
    assert _FakeModel.last_kwargs["harmony"] is not None


def test_chords_file_implies_detect_chords(patched_model, fake_audio, tmp_path):
    out = tmp_path / "out.mid"
    result = CliRunner().invoke(
        main_mod.app,
        [
            "transcribe",
            str(fake_audio),
            "-o",
            str(out),
            "--chords-file",
            str(tmp_path / "c.txt"),
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert _FakeModel.last_kwargs["export"].detect_chords is True


def test_runs_without_harmony_leave_it_alone(patched_model, fake_audio, tmp_path):
    result = CliRunner().invoke(
        main_mod.app, ["transcribe", str(fake_audio), "-o", str(tmp_path / "out.mid")]
    )
    assert result.exit_code == 0, result.stderr
    assert _FakeModel.last_kwargs["harmony"] is None


def test_report_harmony_prints_and_saves_the_chart(tmp_path, capsys):
    from muscriptor.utils.midi import Harmony

    harmony = Harmony(key=("C", "major"), chords=[(0.0, "C"), (4.0, "Am")])
    chords_file = tmp_path / "chords.txt"
    main_mod._report_harmony(harmony, chords_file, None)
    captured = capsys.readouterr()
    assert "Key: C major" in captured.err
    assert "| C  | Am |" in captured.err
    assert chords_file.read_text() == "| C  | Am |\n"


def test_report_harmony_without_chords_saves_nothing(tmp_path, capsys):
    from muscriptor.utils.midi import Harmony

    chords_file = tmp_path / "chords.txt"
    main_mod._report_harmony(Harmony(), chords_file, None)
    assert capsys.readouterr().err == ""
    assert not chords_file.exists()
