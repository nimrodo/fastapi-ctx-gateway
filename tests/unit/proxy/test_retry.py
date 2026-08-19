"""Tests for stream_with_retry: bounded pre-stream retry, no retry once streaming."""

import json

from fastapi_ctx_gateway.proxy.streaming import StreamAccumulator, stream_with_retry

CHUNK = b'data: {"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},'
CHUNK += b'"finishReason":"STOP"}]}\n\n'
PARTIAL_CHUNK = (
    b'data: {"candidates":[{"content":{"parts":[{"text":"partial"}],"role":"model"}}]}\n\n'
)


def _one_good_chunk():
    async def gen():
        yield CHUNK

    return gen()


def _always_fails_before_any_bytes():
    async def gen():
        raise ConnectionError("upstream unreachable")
        yield b""  # pragma: no cover - unreachable, makes this a generator

    return gen()


def _fails_after_one_chunk():
    async def gen():
        yield PARTIAL_CHUNK
        raise ConnectionError("connection dropped mid-stream")

    return gen()


def _error_payload(chunks: list[bytes]) -> dict | None:
    for chunk in chunks:
        if b'"error"' in chunk:
            return json.loads(chunk.removeprefix(b"data: ").strip())
    return None


async def test_succeeds_on_first_attempt_no_retry_needed() -> None:
    calls = []

    def open_stream():
        calls.append(1)
        return _one_good_chunk()

    accumulator = StreamAccumulator()
    chunks = [c async for c in stream_with_retry(open_stream, accumulator)]

    assert chunks == [CHUNK]
    assert len(calls) == 1
    assert accumulator.finished_cleanly


async def test_retries_once_on_pre_stream_failure_then_succeeds() -> None:
    calls = []

    def open_stream():
        calls.append(1)
        if len(calls) == 1:
            return _always_fails_before_any_bytes()
        return _one_good_chunk()

    accumulator = StreamAccumulator()
    chunks = [c async for c in stream_with_retry(open_stream, accumulator)]

    assert chunks == [CHUNK]
    assert len(calls) == 2
    assert accumulator.finished_cleanly


async def test_gives_up_after_one_retry_and_yields_terminal_error() -> None:
    calls = []

    def open_stream():
        calls.append(1)
        return _always_fails_before_any_bytes()

    accumulator = StreamAccumulator()
    chunks = [c async for c in stream_with_retry(open_stream, accumulator)]

    assert len(calls) == 2  # one initial attempt + one retry, no more
    assert not accumulator.finished_cleanly
    error = _error_payload(chunks)
    assert error is not None
    assert "error" in error


async def test_no_retry_once_bytes_have_streamed() -> None:
    calls = []

    def open_stream():
        calls.append(1)
        return _fails_after_one_chunk()

    accumulator = StreamAccumulator()
    chunks = [c async for c in stream_with_retry(open_stream, accumulator)]

    assert len(calls) == 1  # never retried - bytes already reached the client
    assert chunks[0] == PARTIAL_CHUNK
    assert not accumulator.finished_cleanly
    error = _error_payload(chunks)
    assert error is not None
