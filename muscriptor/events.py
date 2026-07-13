"""Public streaming events and per-chunk event builder.

`TranscriptionModel.transcribe` is a generator that yields these dataclasses
one at a time. Every :class:`NoteStartEvent` is guaranteed to be followed by
exactly one matching :class:`NoteEndEvent` (same `index`) later in the stream.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from muscriptor.tokenizer.notes import (
    MINIMUM_NOTE_DURATION_SEC,
    Event,
)


_DRUM_INSTRUMENT = "drums"


@dataclass
class NoteStartEvent:
    pitch: int
    start_time: float
    index: int
    instrument: str


@dataclass
class NoteEndEvent:
    end_time: float
    start_event: NoteStartEvent

    @property
    def start_event_index(self) -> int:
        return self.start_event.index


@dataclass
class ProgressEvent:
    """A coarse transcription-progress signal, woven into the event stream.

    Marks that ``completed`` of ``total`` fixed-size audio chunks have been
    transcribed (``completed == 0`` is emitted once up front so consumers learn
    ``total`` and get a timing baseline; ``completed == total`` marks the end).
    These are deliberately coarse anchors — the frontend smooths between them
    and derives an ETA, since wall-clock time per chunk is only observable
    there. Advisory only: consumers that build notes/MIDI ignore them.
    """

    completed: int
    total: int


@dataclass
class ChunkBoundary:
    """Marks the start of a new model-output chunk in the token stream.

    ``seek_time`` is the chunk's start time in seconds; ``next_seek_time`` is
    the following chunk's start (``None`` for the last chunk), used to drop
    events the model emits past its window.
    """

    seek_time: float
    next_seek_time: float | None


def decode_model_tokens(
    stream: Iterator[int | ChunkBoundary | ProgressEvent],
    vocab: list[Event],
    instrument_for_program: Callable[[int], str],
    frame_rate: int = 100,
) -> Iterator[NoteStartEvent | NoteEndEvent | ProgressEvent]:
    """Stream model token indices straight into NoteStart/NoteEnd events.

    ``stream`` interleaves :class:`ChunkBoundary` markers with token indices:
    each boundary starts a new chunk, followed by that chunk's tokens (EOS and
    anything after it already stripped). All state — open notes, the running
    note index, the per-chunk decode state — lives in this generator's frame
    and persists across chunks.

    Tokens are consumed strictly in order: no buffering, no end-of-chunk sort.
    Each chunk begins with a *tie prologue* — ``(program, pitch)`` pairs for
    notes sustained from the previous chunk, terminated by a ``tie`` token —
    after which any prior open note not in that tie set is closed at the chunk
    boundary. The rest of the chunk drives note onsets/offsets directly.
    """
    open_notes: dict[tuple[int, int], NoteStartEvent] = {}
    next_index = 0

    def mint(pitch: int, start_time: float, instrument: str) -> NoteStartEvent:
        nonlocal next_index
        ev = NoteStartEvent(
            pitch=pitch, start_time=start_time, index=next_index, instrument=instrument
        )
        next_index += 1
        return ev

    # Per-chunk state (reset at every ChunkBoundary).
    seek_time = 0.0
    next_seek_time: float | None = None
    start_tick = 0
    tick_state = 0
    program_state: int | None = None
    velocity_state: int | None = None
    in_prologue = True
    skip_rest = False
    tie_set: set[tuple[int, int]] = set()
    chunk_started = False

    for item in stream:
        if isinstance(item, ProgressEvent):
            # Advisory progress signal — pass straight through, untouched by the
            # decode state machine.
            yield item
            continue
        if isinstance(item, ChunkBoundary):
            # If the previous chunk never closed its tie prologue (malformed:
            # no `tie` token before it ended), treat its tie set as empty so
            # every still-open note ends at that chunk's boundary.
            if chunk_started and in_prologue:
                for key in list(open_notes):
                    yield NoteEndEvent(
                        end_time=seek_time, start_event=open_notes.pop(key)
                    )
            seek_time = item.seek_time
            next_seek_time = item.next_seek_time
            start_tick = round(seek_time * frame_rate)
            tick_state = start_tick
            program_state = None
            velocity_state = None
            in_prologue = True
            skip_rest = False
            tie_set = set()
            chunk_started = True
            continue

        event = vocab[item]
        etype = event.type

        if in_prologue:
            if etype == "tie":
                # End of the tie section: close prior notes not sustained here.
                in_prologue = False
                velocity_state = None
                for key in list(open_notes):
                    if key not in tie_set:
                        yield NoteEndEvent(
                            end_time=seek_time, start_event=open_notes.pop(key)
                        )
            elif etype == "shift":
                # No tie token: the chunk is malformed. Close all open notes at
                # the boundary and drop the rest of the chunk.
                in_prologue = False
                skip_rest = True
                for key in list(open_notes):
                    yield NoteEndEvent(
                        end_time=seek_time, start_event=open_notes.pop(key)
                    )
            elif etype == "program":
                program_state = event.value
            elif etype == "pitch" and program_state is not None:
                tie_set.add((program_state, event.value))
            continue

        if skip_rest:
            continue

        if etype == "shift":
            if event.value > 0:
                tick_state = start_tick + event.value
        elif etype == "program":
            program_state = event.value
        elif etype == "velocity":
            velocity_state = event.value
        elif etype == "drum":
            time = tick_state / frame_rate
            if next_seek_time is None or time < next_seek_time:
                start = mint(event.value, time, _DRUM_INSTRUMENT)
                yield start
                yield NoteEndEvent(
                    end_time=time + MINIMUM_NOTE_DURATION_SEC, start_event=start
                )
        elif etype == "pitch":
            if program_state is None or velocity_state is None:
                continue
            time = tick_state / frame_rate
            if next_seek_time is not None and time >= next_seek_time:
                continue
            key = (program_state, event.value)
            if key in open_notes:
                yield NoteEndEvent(end_time=time, start_event=open_notes.pop(key))
            if velocity_state > 0:
                start = mint(event.value, time, instrument_for_program(program_state))
                open_notes[key] = start
                yield start

    # End of stream: close anything still open. A well-formed final chunk uses
    # the minimum-duration fallback; a chunk that ended mid-prologue closes at
    # its boundary (matching the malformed-chunk behavior above).
    if chunk_started and in_prologue:
        for key in list(open_notes):
            yield NoteEndEvent(end_time=seek_time, start_event=open_notes.pop(key))
    else:
        for ev in list(open_notes.values()):
            yield NoteEndEvent(
                end_time=ev.start_time + MINIMUM_NOTE_DURATION_SEC, start_event=ev
            )
        open_notes.clear()


class OpenNoteTracker:
    """Mirror of :func:`decode_model_tokens`'s open-note bookkeeping.

    The token generator feeds it the same ChunkBoundary markers and token
    indices it sends downstream; :meth:`open_keys` then returns the
    ``(program, pitch)`` pairs the decoder holds open at that point — exactly
    the notes the next chunk's tie prologue must declare as sustained (see
    ``MT3Tokenizer.tie_section_token_ids``). Kept next to
    :func:`decode_model_tokens` because the two state machines must not drift:
    every rule here (tie prologue, malformed chunks, the next_seek_time
    window, retriggers) matches a branch of the decoder above.
    """

    def __init__(self, vocab: list[Event], frame_rate: int = 100):
        self._vocab = vocab
        self._frame_rate = frame_rate
        self._open: set[tuple[int, int]] = set()
        # Per-chunk state, reset at every ChunkBoundary.
        self._next_seek_time: float | None = None
        self._start_tick = 0
        self._tick_state = 0
        self._program: int | None = None
        self._velocity: int | None = None
        self._in_prologue = True
        self._skip_rest = False
        self._tie_set: set[tuple[int, int]] = set()
        self._chunk_started = False

    def feed(self, item: "int | ChunkBoundary") -> None:
        if isinstance(item, ChunkBoundary):
            # A chunk that never closed its tie prologue leaves the decoder
            # with an empty tie set: every open note ends at its boundary.
            if self._chunk_started and self._in_prologue:
                self._open.clear()
            self._next_seek_time = item.next_seek_time
            self._start_tick = round(item.seek_time * self._frame_rate)
            self._tick_state = self._start_tick
            self._program = None
            self._velocity = None
            self._in_prologue = True
            self._skip_rest = False
            self._tie_set = set()
            self._chunk_started = True
            return

        event = self._vocab[item]
        etype = event.type

        if self._in_prologue:
            if etype == "tie":
                self._in_prologue = False
                self._velocity = None
                self._open &= self._tie_set
            elif etype == "shift":
                # Malformed chunk: decoder closes everything and drops the rest.
                self._in_prologue = False
                self._skip_rest = True
                self._open.clear()
            elif etype == "program":
                self._program = event.value
            elif etype == "pitch" and self._program is not None:
                self._tie_set.add((self._program, event.value))
            return

        if self._skip_rest:
            return

        if etype == "shift":
            if event.value > 0:
                self._tick_state = self._start_tick + event.value
        elif etype == "program":
            self._program = event.value
        elif etype == "velocity":
            self._velocity = event.value
        elif etype == "pitch":
            # Drums are instantaneous and never held open, so only pitch
            # events move the open set.
            if self._program is None or self._velocity is None:
                return
            time = self._tick_state / self._frame_rate
            if self._next_seek_time is not None and time >= self._next_seek_time:
                return
            key = (self._program, event.value)
            self._open.discard(key)
            if self._velocity > 0:
                self._open.add(key)

    def open_keys(self) -> list[tuple[int, int]]:
        """Sorted ``(program, pitch)`` pairs currently held open."""
        return sorted(self._open)
