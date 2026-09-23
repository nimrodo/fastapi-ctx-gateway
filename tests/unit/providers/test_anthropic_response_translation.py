"""Tests for AnthropicProvider's SDK-event -> neutral SSE translation and stream()."""

import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from fastapi_ctx_gateway.providers.anthropic import AnthropicProvider, _translate_events
from fastapi_ctx_gateway.schemas.neutral import NeutralGenerateRequest, TextPart, Turn

# --- fake SDK stream events (translation only touches attributes) ---


def _message_start(input_tokens: int, *, cache_read: int = 0, cache_creation: int = 0):
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
    )
    return SimpleNamespace(type="message_start", message=SimpleNamespace(usage=usage))


def _text_delta(text: str):
    return SimpleNamespace(
        type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=text)
    )


def _thinking_delta(text: str):
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="thinking_delta", thinking=text),
    )


def _message_delta(stop_reason: str | None, output_tokens: int, input_tokens: int | None = None):
    return SimpleNamespace(
        type="message_delta",
        delta=SimpleNamespace(stop_reason=stop_reason),
        usage=SimpleNamespace(output_tokens=output_tokens, input_tokens=input_tokens),
    )


def _bare(event_type: str):
    return SimpleNamespace(type=event_type)


async def _aiter(events):
    for event in events:
        yield event


def _payload(chunk: bytes) -> dict:
    return json.loads(chunk.removeprefix(b"data: ").strip())


async def _translate(events) -> list[dict]:
    return [_payload(c) async for c in _translate_events(_aiter(events))]


# --- _translate_events ---


async def test_message_start_yields_nothing_but_seeds_prompt_tokens() -> None:
    out = await _translate([_message_start(10), _message_delta("end_turn", output_tokens=5)])
    assert len(out) == 1
    assert out[0]["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }


async def test_prompt_tokens_include_cache_tokens() -> None:
    out = await _translate(
        [
            _message_start(10, cache_read=3, cache_creation=2),
            _message_delta("end_turn", output_tokens=5),
        ]
    )
    assert out[0]["usage"]["prompt_tokens"] == 15
    assert out[0]["usage"]["total_tokens"] == 20


async def test_text_delta_becomes_one_neutral_delta_event() -> None:
    out = await _translate([_text_delta("Hel"), _text_delta("lo")])
    assert [e["delta"]["parts"][0]["text"] for e in out] == ["Hel", "lo"]
    assert all(e["delta"]["role"] == "assistant" for e in out)


async def test_thinking_delta_is_swallowed() -> None:
    out = await _translate([_text_delta("hi"), _thinking_delta("hmm"), _text_delta("!")])
    assert [e["delta"]["parts"][0]["text"] for e in out] == ["hi", "!"]


async def test_empty_text_delta_is_swallowed() -> None:
    out = await _translate([_text_delta("")])
    assert out == []


@pytest.mark.parametrize(
    ("native", "expected"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "max_tokens"),
        ("refusal", "safety"),
        ("tool_use", "other"),
        ("pause_turn", "other"),
        ("something_new", "other"),
    ],
)
async def test_stop_reason_mapping(native: str, expected: str) -> None:
    out = await _translate([_message_start(1), _message_delta(native, output_tokens=1)])
    assert out[0]["finish_reason"] == expected


async def test_message_delta_without_stop_reason_has_no_finish_reason() -> None:
    out = await _translate([_message_start(1), _message_delta(None, output_tokens=2)])
    assert "finish_reason" not in out[0]
    assert out[0]["usage"]["completion_tokens"] == 2


async def test_message_delta_input_tokens_override_seed() -> None:
    out = await _translate(
        [_message_start(10), _message_delta("end_turn", output_tokens=5, input_tokens=12)]
    )
    assert out[0]["usage"]["prompt_tokens"] == 12
    assert out[0]["usage"]["total_tokens"] == 17


async def test_structural_events_are_swallowed() -> None:
    out = await _translate(
        [
            _message_start(1),
            _bare("content_block_start"),
            _text_delta("x"),
            _bare("content_block_stop"),
            _bare("ping"),
            _message_delta("end_turn", output_tokens=1),
            _bare("message_stop"),
        ]
    )
    assert len(out) == 2  # one text delta + one message_delta
    assert out[0]["delta"]["parts"][0]["text"] == "x"
    assert out[1]["finish_reason"] == "stop"


# --- stream() with a stubbed client ---


class _FakeStream:
    def __init__(self, events, *, raise_mid=None):
        self._events = events
        self._raise_mid = raise_mid
        self.closed = False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for event in self._events:
            yield event
        if self._raise_mid is not None:
            raise self._raise_mid

    async def close(self):
        self.closed = True


class _FakeMessages:
    def __init__(self, result):
        self._result = result
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _FakeClient:
    def __init__(self, result):
        self.messages = _FakeMessages(result)
        self.closed = False

    async def close(self):
        self.closed = True


def _request() -> NeutralGenerateRequest:
    return NeutralGenerateRequest(turns=[Turn(role="user", parts=[TextPart(text="hi")])])


def _provider(result, api_key: str = "") -> tuple[AnthropicProvider, _FakeClient]:
    client = _FakeClient(result)
    return (
        AnthropicProvider(client=client, default_max_tokens=4096, api_key=api_key),  # type: ignore[arg-type]
        client,
    )


async def test_stream_translates_and_closes_the_sdk_stream() -> None:
    fake = _FakeStream(
        [_message_start(2), _text_delta("hi"), _message_delta("end_turn", output_tokens=3)]
    )
    provider, client = _provider(fake)
    chunks = [c async for c in provider.stream("claude-opus-5", _request())]

    assert [_payload(c).get("delta", {}).get("parts", [{}])[0].get("text") for c in chunks[:1]] == [
        "hi"
    ]
    assert _payload(chunks[-1])["usage"]["total_tokens"] == 5
    assert fake.closed is True
    assert client.messages.calls[0]["model"] == "claude-opus-5"
    assert client.messages.calls[0]["stream"] is True
    assert client.messages.calls[0]["max_tokens"] == 4096


async def test_stream_yields_neutral_error_event_on_api_status_error() -> None:
    exc = anthropic.APIStatusError(
        "bad model",
        response=httpx2.Response(404, request=httpx2.Request("POST", "https://api.anthropic.com")),
        body={"error": {"message": "model not found"}},
    )
    provider, _ = _provider(exc)
    chunks = [c async for c in provider.stream("claude-opus-5", _request())]

    assert len(chunks) == 1
    payload = _payload(chunks[0])
    assert payload["error"]["provider_status"] == 404
    assert "model not found" in payload["error"]["message"]


async def test_stream_redacts_the_configured_api_key_from_a_relayed_error() -> None:
    exc = anthropic.APIStatusError(
        "bad key",
        response=httpx2.Response(401, request=httpx2.Request("POST", "https://api.anthropic.com")),
        body={"error": {"message": "invalid x-api-key: sk-ant-mysecretgatewaykey123"}},
    )
    provider, _ = _provider(exc, api_key="sk-ant-mysecretgatewaykey123")
    chunks = [c async for c in provider.stream("claude-opus-5", _request())]

    payload = _payload(chunks[0])
    assert "sk-ant-mysecretgatewaykey123" not in payload["error"]["message"]
    assert "[REDACTED]" in payload["error"]["message"]


async def test_stream_yields_neutral_error_event_on_connection_error() -> None:
    exc = anthropic.APIConnectionError(
        message="boom", request=httpx2.Request("POST", "https://api.anthropic.com")
    )
    provider, _ = _provider(exc)
    chunks = [c async for c in provider.stream("claude-opus-5", _request())]

    assert len(chunks) == 1
    assert _payload(chunks[0])["error"]["provider_status"] is None


async def test_stream_emits_terminal_error_on_mid_stream_drop_and_still_closes() -> None:
    exc = anthropic.APIConnectionError(
        message="dropped", request=httpx2.Request("POST", "https://api.anthropic.com")
    )
    fake = _FakeStream([_message_start(1), _text_delta("par")], raise_mid=exc)
    provider, _ = _provider(fake)
    chunks = [c async for c in provider.stream("claude-opus-5", _request())]

    assert _payload(chunks[0])["delta"]["parts"][0]["text"] == "par"
    assert _payload(chunks[-1])["error"]["type"] == "upstream_error"
    assert fake.closed is True


async def test_aclose_closes_the_client() -> None:
    provider, client = _provider(_FakeStream([]))
    await provider.aclose()
    assert client.closed is True
