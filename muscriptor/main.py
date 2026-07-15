"""CLI for muscriptor: audio → MIDI transcription."""

import dataclasses
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent
from muscriptor.tokenizer.mt3 import (
    MT3_FULL_PLUS_GROUP_NAMES,
    resolve_instrument_names,
)
from muscriptor.transcription_model import TranscriptionModel
from muscriptor.utils.download import ModelDownloadError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="muscriptor — audio-to-MIDI transcription",
)


def _load_model(model_path: str | None, device: str | None) -> TranscriptionModel:
    """load_model with CLI-friendly failure: known download problems (missing
    HuggingFace authentication, …) print a plain message instead of a traceback."""
    try:
        return TranscriptionModel.load_model(weights_path=model_path, device=device)
    except ModelDownloadError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


class OutputFormat(str, Enum):
    midi = "midi"
    json = "json"
    jsonl = "jsonl"


def _event_to_dict(ev: NoteStartEvent | NoteEndEvent) -> dict:
    if isinstance(ev, NoteStartEvent):
        return {"type": "start", **dataclasses.asdict(ev)}
    return {
        "type": "end",
        "end_time": ev.end_time,
        "start_event_index": ev.start_event_index,
    }


@app.command()
def transcribe(
    audio_file: Annotated[
        Path | None,
        typer.Argument(help="Input audio file (wav, mp3, flac, …) — omit if --yt is used"),
    ] = None,
    yt_url: Annotated[
        str | None,
        typer.Option(
            "--yt",
            "--youtube",
            help="Download audio from a YouTube/any URL and transcribe (requires yt-dlp)",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help=(
                "Output file path. Use '-' to write to stdout (all progress / "
                "timing info is sent to stderr in that case). "
                "Default: <audio_file>.<ext> where ext matches --format."
            ),
        ),
    ] = None,
    format: Annotated[
        OutputFormat,
        typer.Option(
            "--format",
            "-f",
            help=(
                "Output format: midi (default), json (single array of events), "
                "or jsonl (one event per line, streamed as transcription progresses)"
            ),
            case_sensitive=False,
        ),
    ] = OutputFormat.midi,
    notes: Annotated[
        bool, typer.Option("--notes", help="Print decoded events to stdout")
    ] = False,
    sampling: Annotated[
        bool,
        typer.Option(
            "--sampling", help="Use temperature sampling instead of greedy decoding"
        ),
    ] = False,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature", "-t", help="Sampling temperature (only with --sampling)"
        ),
    ] = 1.0,
    cfg_coef: Annotated[
        float, typer.Option("--cfg-coef", help="Classifier-free guidance coefficient")
    ] = 1.0,  # todo: make it dynamic
    model_path: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help=(
                "Model size ('small', 'medium', 'large'; default: large), "
                "a local safetensors path, or an hf:// / http(s):// URL"
            ),
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(
            "--device", "-d", help="Device: 'auto', 'cpu', 'cuda', 'cuda:0', …"
        ),
    ] = "auto",
    batch_size: Annotated[
        int | None,
        typer.Option(
            "--batch-size",
            "-b",
            help="Batch size for generation (default: 1 on CPU, 4 on GPU)",
        ),
    ] = None,
    strict_eos: Annotated[
        bool,
        typer.Option(
            "--strict-eos",
            help="Raise an error if a chunk fails to emit EOS within the generation budget (default: downgrade to a warning)",
        ),
    ] = False,
    beam_size: Annotated[
        int,
        typer.Option(
            "--beam-size",
            help="Beam search width (1 = greedy/sampling, ≥2 enables beam search)",
        ),
    ] = 1,
    auralize: Annotated[
        Path | None,
        typer.Option(
            "--auralize",
            help=(
                "Write a stereo auralization (L=original audio, R=MIDI synthesis) to "
                "this path. Requires fluidsynth on PATH. Extension determines format: "
                ".wav (default) or .mp3. Only valid with --format midi."
            ),
        ),
    ] = None,
    soundfont: Annotated[
        Path | None,
        typer.Option(
            "--soundfont",
            help=(
                "Path to a .sf2 SoundFont for auralization. Defaults to "
                "MuseScore_General.sf2, downloaded once and cached locally."
            ),
        ),
    ] = None,
    instruments: Annotated[
        str | None,
        typer.Option(
            "--instruments",
            help=(
                "Comma-separated list of expected instrument group names. "
                "When given, every instrument not in the list is forbidden "
                "from being decoded at all. Case-insensitive; unambiguous "
                "abbreviations are accepted (e.g. 'timp,cello,dist'). Run "
                "'muscriptor list-instruments' to see all available names."
            ),
        ),
    ] = None,
    detect_tempo: Annotated[
        bool,
        typer.Option(
            "--detect-tempo",
            help=(
                "Estimate BPM from transcribed note onsets and write a "
                "set_tempo MIDI meta event with the detected tempo."
            ),
        ),
    ] = False,
    detect_key: Annotated[
        bool,
        typer.Option(
            "--detect-key",
            help=(
                "Estimate musical key (Krumhansl-Schmuckler algorithm) and "
                "write a key_signature MIDI meta event."
            ),
        ),
    ] = False,
    detect_chords: Annotated[
        bool,
        typer.Option(
            "--detect-chords",
            help=(
                "Detect chord labels at each beat and print the chord "
                "progression to stderr. Implies --detect-tempo."
            ),
        ),
    ] = False,
    quantize: Annotated[
        bool,
        typer.Option(
            "--quantize",
            help=(
                "Snap note onsets and offsets to the nearest grid "
                "subdivision (16th notes by default). Implies "
                "--detect-tempo so the grid matches the music."
            ),
        ),
    ] = False,
    subdivision: Annotated[
        int,
        typer.Option(
            "--subdivision",
            help=(
                "Grid resolution for --quantize: 4 = 16th notes (default), "
                "2 = eighth notes, 8 = 32nd notes. Only meaningful with "
                "--quantize."
            ),
        ),
    ] = 4,
    chords_file: Annotated[
        Path | None,
        typer.Option(
            "--chords-file",
            help=(
                "Save detected chord progression as a text file. "
                "Implies --detect-chords."
            ),
        ),
    ] = None,
    bpm: Annotated[
        int | None,
        typer.Option(
            "--bpm",
            help="Manual BPM override (skips tempo detection). "
            "Use --detect-tempo to re-enable detection.",
        ),
    ] = None,
) -> None:
    """Transcribe an audio file to MIDI."""
    instrument_names: list[str] | None = None
    if instruments is not None:
        tokens = [n for n in instruments.split(",") if n.strip()]
        try:
            instrument_names = resolve_instrument_names(tokens)
        except ValueError as e:
            typer.echo(
                f"Error: {e}. "
                "Run 'muscriptor list-instruments' to see available names.",
                err=True,
            )
            raise typer.Exit(1)
        typer.echo(f"Instruments: {', '.join(instrument_names)}", err=True)

    # YouTube / URL download — WAV is saved to current directory
    if yt_url is not None:
        try:
            import yt_dlp  # noqa: F401
        except ImportError:
            typer.echo(
                "Error: --yt requires yt-dlp. Install it with: pip install yt-dlp",
                err=True,
            )
            raise typer.Exit(1)

        outdir = Path.cwd() / "wav"
        outdir.mkdir(exist_ok=True)
        out_pattern = str(outdir / "%(title)s.%(ext)s")
        typer.echo(f"Downloading audio from {yt_url} …", err=True)
        import subprocess

        result = subprocess.run(
            ["yt-dlp", "-x", "--audio-format", "wav", "-o", out_pattern, yt_url],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            typer.echo(f"yt-dlp failed:\n{result.stderr}", err=True)
            raise typer.Exit(1)
        # Find the downloaded WAV — yt-dlp prints the filename on stderr
        import re as _re

        wav_match = _re.search(r"\[ExtractAudio\] Destination: (.+\.wav)", result.stderr)
        if wav_match:
            wav_path = Path(wav_match.group(1))
        else:
            # Fallback: pick the most recent .wav in the wav/ directory
            wavs = sorted(
                (Path.cwd() / "wav").glob("*.wav"),
                key=lambda p: p.stat().st_mtime,
            )
            if not wavs:
                typer.echo("Could not find downloaded WAV file", err=True)
                raise typer.Exit(1)
            wav_path = wavs[-1]
        audio_file = wav_path.resolve()
        typer.echo(f"Downloaded {audio_file.name}", err=True)

    if audio_file is None:
        typer.echo(
            "Error: provide an audio file or --yt URL",
            err=True,
        )
        raise typer.Exit(1)

    if not audio_file.exists():
        typer.echo(f"Error: file not found: {audio_file}", err=True)
        raise typer.Exit(1)

    is_stdout = output is not None and str(output) == "-"

    if output is None:
        suffix = {
            OutputFormat.midi: ".mid",
            OutputFormat.json: ".json",
            OutputFormat.jsonl: ".jsonl",
        }[format]
        outdir = Path("midi")
        outdir.mkdir(exist_ok=True)
        output = outdir / f"{audio_file.stem}{suffix}"

    _device = None if device == "auto" else device

    # All chatty progress/timing info goes to stderr — stdout is reserved for
    # the actual output when `-o -` is used.
    typer.echo("Loading model…", err=True)
    model = _load_model(model_path, _device)
    import torch

    model._model = model._model.to(torch.float32)

    typer.echo(f"Transcribing {audio_file} …", err=True)

    if auralize is not None and format != OutputFormat.midi:
        typer.echo("Error: --auralize requires --format midi", err=True)
        raise typer.Exit(1)

    kwargs = dict(
        audio=audio_file,
        use_sampling=sampling,
        temperature=temperature,
        cfg_coef=cfg_coef,
        instruments=instrument_names,
        batch_size=batch_size,
        no_eos_is_ok=not strict_eos,
        beam_size=beam_size,
    )

    if format == OutputFormat.midi:
        midi_bytes = model.transcribe_to_midi(
            **kwargs,
            detect_tempo=detect_tempo,
            detect_key=detect_key,
            detect_chords=detect_chords or chords_file is not None,
            quantize=quantize,
            subdivision=subdivision,
            chords_file=chords_file,
            manual_bpm=bpm,
        )
        if is_stdout:
            sys.stdout.buffer.write(midi_bytes)
            sys.stdout.buffer.flush()
        else:
            output.write_bytes(midi_bytes)
            typer.echo(f"Saved MIDI to {output}", err=True)
        if notes:
            typer.echo(
                "Re-run with --format json to inspect the event stream.", err=True
            )
        if auralize is not None and not is_stdout:
            from muscriptor.utils.auralization import auralize as do_auralize

            typer.echo(f"Auralizing → {auralize} …", err=True)
            do_auralize(
                midi_path=output,
                original_audio_path=audio_file,
                output_path=auralize,
                soundfont_path=soundfont,
            )
            typer.echo(f"Saved auralization to {auralize}", err=True)
    elif format == OutputFormat.jsonl:
        # Stream one JSON object per line, flushing after each event so the
        # file (or stdout pipe) can be consumed live.
        if is_stdout:
            sink = sys.stdout
            close_after = False
        else:
            sink = output.open("w")
            close_after = True
        try:
            for e in model.transcribe(**kwargs):
                if isinstance(e, ProgressEvent):
                    continue
                sink.write(json.dumps(_event_to_dict(e)) + "\n")
                sink.flush()
                if notes:
                    typer.echo(str(e), err=True)
        finally:
            if close_after:
                sink.close()
        if not is_stdout:
            typer.echo(f"Saved JSONL to {output}", err=True)
    else:  # json
        events = [
            e
            for e in model.transcribe(**kwargs)
            if not isinstance(e, ProgressEvent)
        ]
        payload = json.dumps([_event_to_dict(e) for e in events], indent=2)
        if is_stdout:
            sys.stdout.write(payload + "\n")
            sys.stdout.flush()
        else:
            output.write_text(payload)
            typer.echo(f"Saved JSON to {output}", err=True)
        if notes:
            for e in events:
                typer.echo(str(e), err=True)


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", help="Bind address")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to listen on")] = 8222,
    model_path: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help=(
                "Model size ('small', 'medium', 'large'; default: large), "
                "a local safetensors path, or an hf:// / http(s):// URL"
            ),
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(
            "--device", "-d", help="Device: 'auto', 'cpu', 'cuda', 'cuda:0', …"
        ),
    ] = "auto",
):
    """Run the HTTP transcription server (POST /transcribe → SSE event stream)."""
    import uvicorn

    from muscriptor.server import create_app

    _device = None if device == "auto" else device
    typer.echo("Loading model…")
    model = _load_model(model_path, _device)
    web_dir = Path(__file__).resolve().parent / "web_dist"
    fastapi_app = create_app(model, web_dir=web_dir if web_dir.is_dir() else None)
    uvicorn.run(fastapi_app, host=host, port=port)


@app.command()
def list_instruments():
    """List the instrument group names accepted by --instruments."""
    for name in MT3_FULL_PLUS_GROUP_NAMES:
        typer.echo(name)


@app.command()
def inspect(
    midi_file: Annotated[
        Path,
        typer.Argument(help="MIDI file to inspect", exists=True, dir_okay=False),
    ],
    notes: Annotated[
        bool,
        typer.Option("--notes", "-n", help="Show individual note events"),
    ] = False,
    summary: Annotated[
        bool,
        typer.Option(
            "--summary", "-s", help="Show only a compact summary per track"
        ),
    ] = False,
) -> None:
    """Print a readable dump of a MIDI file's structure and events."""
    import mido

    mid = mido.MidiFile(midi_file)
    typer.echo(f"File:   {midi_file}")
    typer.echo(f"Format: {mid.type}")
    typer.echo(f"Tracks: {len(mid.tracks)}")
    typer.echo(f"Length: {mid.length:.1f}s")
    typer.echo(f"PPQN:   {mid.ticks_per_beat}")
    typer.echo()

    for i, track in enumerate(mid.tracks):
        note_ons = sum(
            1 for e in track if e.type == "note_on" and e.velocity > 0
        )
        typer.echo(
            f"── Track {i}: {track.name or '(unnamed)'}"
            f"  ({len(track)} msgs, {note_ons} notes) ──"
        )

        if summary:
            # One-line summary: count by message type
            counts: dict[str, int] = {}
            for msg in track:
                if msg.is_meta:
                    counts.setdefault(f"meta:{msg.type}", 0)
                    counts[f"meta:{msg.type}"] += 1
                else:
                    counts[msg.type] = counts.get(msg.type, 0) + 1
            parts = [f"{k}={v}" for k, v in sorted(counts.items())]
            typer.echo("   " + ", ".join(parts))
            continue

        tick = 0
        for msg in track:
            tick += msg.time
            if msg.is_meta:
                if msg.type in (
                    "track_name",
                    "set_tempo",
                    "key_signature",
                    "time_signature",
                    "end_of_track",
                ):
                    typer.echo(
                        f"  @{tick:>8d}  [{msg.type}]  {msg}"
                    )
            elif notes and msg.type in ("note_on", "note_off"):
                typer.echo(
                    f"  @{tick:>8d}  {msg.type:<8s}"
                    f"  note={msg.note:>3d}  vel={msg.velocity:>3d}"
                    f"  ch={msg.channel}"
                )
        typer.echo()


def main():
    app()


if __name__ == "__main__":
    main()
