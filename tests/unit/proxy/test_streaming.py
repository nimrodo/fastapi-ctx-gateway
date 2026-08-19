"""Tests for tee_stream's forwarding, accumulation, and cancellation behavior."""

from fastapi_ctx_gateway.proxy.streaming import StreamAccumulator, tee_stream

EVENT_1 = b'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}],"role":"model"}}]}\n\n'
EVENT_2 = (
    b'data: {"candidates":[{"content":{"parts":[{"text":"lo"}],"role":"model"},'
    b'"finishReason":"STOP"}],"usageMetadata":{"totalTokenCount":5}}\n\n'
)


async def _upstream(chunks: list[bytes]):
    for chunk in chunks:
        yield chunk


async def test_forwards_every_chunk_unchanged() -> None:
    acc = StreamAccumulator()
    forwarded = [chunk async for chunk in tee_stream(_upstream([EVENT_1, EVENT_2]), acc)]
    assert forwarded == [EVENT_1, EVENT_2]


async def test_accumulates_delta_text_and_finish_reason() -> None:
    acc = StreamAccumulator()
    async for _ in tee_stream(_upstream([EVENT_1, EVENT_2]), acc):
        pass
    assert acc.text == "Hello"
    assert acc.finish_reason == "STOP"
    assert acc.usage == {"totalTokenCount": 5}
    assert acc.finished_cleanly is True


async def test_no_finish_reason_when_stream_truncated() -> None:
    acc = StreamAccumulator()
    async for _ in tee_stream(_upstream([EVENT_1]), acc):
        pass
    assert acc.text == "Hel"
    assert acc.finish_reason is None
    assert acc.finished_cleanly is False


async def test_stopping_iteration_early_closes_upstream() -> None:
    closed = False

    async def upstream():
        nonlocal closed
        try:
            yield EVENT_1
            yield EVENT_2
        finally:
            closed = True

    acc = StreamAccumulator()
    gen = tee_stream(upstream(), acc)
    await gen.__anext__()  # consume only the first chunk
    await gen.aclose()  # simulate the client disconnecting

    assert closed is True
    assert acc.finish_reason is None
