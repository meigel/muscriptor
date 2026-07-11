"""muscriptor — audio-to-MIDI transcription."""

# The public API is imported lazily (PEP 562): the console entry point
# (muscriptor.launcher) must be importable in environments where torch is
# absent — on Intel macs with a too-recent Python, torch is deliberately not
# installed and the launcher re-execs under a supported interpreter.
# `from muscriptor import TranscriptionModel` still works (and pulls torch).

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from muscriptor.events import NoteEndEvent, NoteStartEvent
    from muscriptor.tokenizer.notes import Note
    from muscriptor.transcription_model import TranscriptionModel

__all__ = ["TranscriptionModel", "Note", "NoteStartEvent", "NoteEndEvent"]

_SUBMODULES = {
    "TranscriptionModel": "muscriptor.transcription_model",
    "Note": "muscriptor.tokenizer.notes",
    "NoteStartEvent": "muscriptor.events",
    "NoteEndEvent": "muscriptor.events",
}


def __getattr__(name: str):
    if name in _SUBMODULES:
        import importlib

        value = getattr(importlib.import_module(_SUBMODULES[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
