"""Tests for TranscriptionModel._generate_token_stream.

These check the streaming contract of the token stream without a real model:
a fake `generate()` yields one row (`[batch]`) per timestep and records how
many timesteps have been pulled, so we can assert that a chunk's events come
out *as soon as that chunk finishes* — before the rest of the batch is even
generated.
"""

import threading
from types import SimpleNamespace

import pytest
import torch

from muscriptor.events import ChunkBoundary, ProgressEvent
from muscriptor.transcription_model import TranscriptionModel

EOS = 99


def _run(batches, *, batch_size, seek_times, no_eos_is_ok=False):
    """Drive _generate_token_stream with a fake model.

    ``batches`` is one list of rows per expected ``generate()`` call; each row
    is the per-chunk token for one timestep. Returns ``(stream, pulled)`` where
    ``pulled`` grows by one entry every time the fake yields a timestep, so its
    length is how far generation has progressed.
    """
    pulled: list[list[int]] = []
    calls = iter(batches)

    def generate(**kwargs):
        for row in next(calls):
            pulled.append(row)
            yield torch.tensor(row)

    fake = SimpleNamespace(
        _model=SimpleNamespace(generate=generate),
        _tokenizer=SimpleNamespace(eos_id=EOS),
    )
    conditions = [object()] * len(seek_times)
    stream = TranscriptionModel._generate_token_stream(
        fake,
        conditions,
        seek_times,
        batch_size,
        max_gen_len=64,
        use_sampling=False,
        temperature=1.0,
        cfg_coef=2.0,
        no_eos_is_ok=no_eos_is_ok,
    )
    return stream, pulled


# ---------------------------------------------------------------------------
# Emitted as soon as possible
# ---------------------------------------------------------------------------


def test_first_chunk_streams_before_the_batch_finishes():
    # batch of 2 chunks: chunk 0 ends at row 2, chunk 1 only at row 4.
    rows = [[10, 20], [11, 21], [EOS, 22], [12, 23], [13, EOS]]
    stream, pulled = _run([rows], batch_size=2, seek_times=[0.0, 5.0])
    it = iter(stream)

    assert next(it) == ChunkBoundary(0.0, 5.0)
    assert len(pulled) == 0  # the boundary is emitted before any generation
    assert next(it) == 10
    assert len(pulled) == 1  # first token after a single timestep
    assert next(it) == 11
    # Chunk 0 is fully streamed having generated only its own timesteps —
    # chunk 1 (which finishes at row 4) has not been generated to completion.
    assert len(pulled) == 2


def test_single_chunk_streams_token_by_token():
    rows = [[10], [11], [12], [EOS]]
    stream, pulled = _run([rows], batch_size=1, seek_times=[0.0])
    it = iter(stream)

    assert next(it) == ChunkBoundary(0.0, None)
    assert len(pulled) == 0
    for expected, count in [(10, 1), (11, 2), (12, 3)]:
        assert next(it) == expected
        assert len(pulled) == count


# ---------------------------------------------------------------------------
# Ordering and buffering
# ---------------------------------------------------------------------------


def test_full_stream_order_for_a_batch():
    rows = [[10, 20], [11, 21], [EOS, 22], [12, 23], [13, EOS]]
    stream, _ = _run([rows], batch_size=2, seek_times=[0.0, 5.0])
    assert list(stream) == [
        ChunkBoundary(0.0, 5.0),
        10,
        11,
        ChunkBoundary(5.0, None),
        20,
        21,
        22,
        23,
        # End of the (only) batch: both chunks done.
        ProgressEvent(completed=2, total=2),
    ]


def test_later_chunk_finishing_first_is_buffered_until_its_turn():
    # chunk 1 hits EOS (row 1) before chunk 0 (row 3); its tokens must wait.
    rows = [[10, 20], [11, EOS], [12, 88], [EOS, 88]]
    stream, _ = _run([rows], batch_size=2, seek_times=[0.0, 5.0])
    assert list(stream) == [
        ChunkBoundary(0.0, 5.0),
        10,
        11,
        12,
        ChunkBoundary(5.0, None),
        20,
        ProgressEvent(completed=2, total=2),
    ]


def test_chunks_across_multiple_batches_stay_in_order():
    # batch_size=1 → one generate() call per chunk.
    batches = [[[10], [11], [EOS]], [[20], [EOS]]]
    stream, _ = _run(batches, batch_size=1, seek_times=[0.0, 5.0])
    assert list(stream) == [
        ChunkBoundary(0.0, 5.0),
        10,
        11,
        # batch_size=1 => a completion anchor trails each chunk.
        ProgressEvent(completed=1, total=2),
        ChunkBoundary(5.0, None),
        20,
        ProgressEvent(completed=2, total=2),
    ]


# ---------------------------------------------------------------------------
# Missing EOS
# ---------------------------------------------------------------------------


def test_missing_eos_raises_by_default():
    rows = [[10, 20], [11, 21]]  # neither chunk emits EOS
    stream, _ = _run([rows], batch_size=2, seek_times=[0.0, 5.0])
    with pytest.raises(RuntimeError, match="did not emit EOS"):
        list(stream)


def test_missing_eos_warns_and_still_emits_when_allowed():
    rows = [[10, 20], [11, 21]]
    stream, _ = _run([rows], batch_size=2, seek_times=[0.0, 5.0], no_eos_is_ok=True)
    with pytest.warns(RuntimeWarning, match="did not emit EOS"):
        events = list(stream)
    assert events == [
        ChunkBoundary(0.0, 5.0),
        10,
        11,
        ChunkBoundary(5.0, None),
        20,
        21,
        ProgressEvent(completed=2, total=2),
    ]


# ---------------------------------------------------------------------------
# Parallel decoding (_generate_token_stream_parallel)
# ---------------------------------------------------------------------------


def _run_parallel(
    chunk_tokens, *, num_workers, seek_times, no_eos_is_ok=False, gates=None
):
    """Drive _generate_token_stream_parallel with a fake model.

    ``chunk_tokens`` is one token list per chunk (include EOS explicitly).
    The fake ``generate()`` receives a single-chunk ``conditions`` list and
    looks its rows up by condition identity, so it is safe to call from any
    worker thread in any order. ``gates`` optionally maps
    ``(chunk_index, step_index)`` to a ``threading.Event`` the fake waits on
    before yielding that step — used to force a specific completion order.
    """
    conditions = [object() for _ in seek_times]
    index_of = {id(c): i for i, c in enumerate(conditions)}

    def generate(conditions, **kwargs):
        (cond,) = conditions
        idx = index_of[id(cond)]
        for t, tok in enumerate(chunk_tokens[idx]):
            if gates and (idx, t) in gates:
                assert gates[(idx, t)].wait(timeout=5), "test gate never opened"
            yield torch.tensor([tok])

    fake = SimpleNamespace(
        _model=SimpleNamespace(generate=generate),
        _tokenizer=SimpleNamespace(eos_id=EOS),
    )
    return TranscriptionModel._generate_token_stream_parallel(
        fake,
        conditions,
        seek_times,
        num_workers,
        max_gen_len=64,
        use_sampling=False,
        temperature=1.0,
        cfg_coef=2.0,
        no_eos_is_ok=no_eos_is_ok,
    )


def test_parallel_keeps_chunk_order_and_content():
    chunks = [[10, 11, EOS], [20, EOS], [30, 31, 32, EOS]]
    prev_threads = torch.get_num_threads()
    events = list(
        _run_parallel(chunks, num_workers=2, seek_times=[0.0, 5.0, 10.0])
    )
    assert torch.get_num_threads() == prev_threads  # restored after the stream

    tokens_and_boundaries = [e for e in events if not isinstance(e, ProgressEvent)]
    assert tokens_and_boundaries == [
        ChunkBoundary(0.0, 5.0),
        10,
        11,
        ChunkBoundary(5.0, 10.0),
        20,
        ChunkBoundary(10.0, None),
        30,
        31,
        32,
    ]
    # Progress anchors are timing-dependent in count but must be monotonic,
    # carry the right total, and end at completion.
    progress = [e for e in events if isinstance(e, ProgressEvent)]
    assert all(p.total == 3 for p in progress)
    counts = [p.completed for p in progress]
    assert counts == sorted(counts)
    assert counts[-1] == 3


def test_parallel_first_chunk_streams_while_later_chunk_is_stuck():
    gate = threading.Event()
    chunks = [[10, 11, EOS], [20, 21, EOS]]
    # Chunk 1 blocks before its second token; chunk 0 must stream regardless.
    stream = _run_parallel(
        chunks, num_workers=2, seek_times=[0.0, 5.0], gates={(1, 1): gate}
    )
    it = iter(stream)
    try:
        assert next(it) == ChunkBoundary(0.0, 5.0)
        assert next(it) == 10
        assert next(it) == 11  # all of chunk 0, while chunk 1 is still gated
    finally:
        gate.set()
    rest = [e for e in it if not isinstance(e, ProgressEvent)]
    assert rest == [ChunkBoundary(5.0, None), 20, 21]


def test_parallel_later_chunk_finishing_first_is_buffered_until_its_turn():
    gate = threading.Event()
    chunks = [[10, 11, EOS], [20, EOS]]
    # Chunk 0 blocks before its second token, so chunk 1 finishes first; its
    # tokens must still come out after all of chunk 0's.
    stream = _run_parallel(
        chunks, num_workers=2, seek_times=[0.0, 5.0], gates={(0, 1): gate}
    )
    it = iter(stream)
    try:
        assert next(it) == ChunkBoundary(0.0, 5.0)
        assert next(it) == 10
    finally:
        gate.set()
    rest = [e for e in it if not isinstance(e, ProgressEvent)]
    assert rest == [11, ChunkBoundary(5.0, None), 20]


def test_parallel_missing_eos_raises_by_default():
    chunks = [[10, 11], [20, EOS]]  # chunk 0 never emits EOS
    stream = _run_parallel(chunks, num_workers=2, seek_times=[0.0, 5.0])
    with pytest.raises(RuntimeError, match="did not emit EOS"):
        list(stream)


def test_parallel_missing_eos_warns_and_still_emits_when_allowed():
    chunks = [[10, 11], [20, EOS]]
    stream = _run_parallel(
        chunks, num_workers=2, seek_times=[0.0, 5.0], no_eos_is_ok=True
    )
    with pytest.warns(RuntimeWarning, match="did not emit EOS"):
        events = list(stream)
    tokens_and_boundaries = [e for e in events if not isinstance(e, ProgressEvent)]
    assert tokens_and_boundaries == [
        ChunkBoundary(0.0, 5.0),
        10,
        11,
        ChunkBoundary(5.0, None),
        20,
    ]


def test_parallel_early_close_stops_workers():
    gate = threading.Event()
    chunks = [[10, 11, EOS], [20, 21, EOS]]
    # Thread count is captured before the generator body first runs (it is
    # lazy), so this is the value the close must restore.
    prev_threads = torch.get_num_threads()
    stream = _run_parallel(
        chunks, num_workers=2, seek_times=[0.0, 5.0], gates={(1, 1): gate}
    )
    it = iter(stream)
    assert next(it) == ChunkBoundary(0.0, 5.0)
    assert next(it) == 10
    gate.set()
    # Closing mid-stream must return promptly (workers observe the cancel
    # flag) and restore torch's thread count.
    it.close()
    assert torch.get_num_threads() == prev_threads
