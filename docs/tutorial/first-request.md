# Your first request

The gateway speaks its own **neutral** request/response contract, not a given provider's native wire format — a pluggable [`Provider`][fastapi_ctx_gateway.providers.base.Provider] translates to/from whichever upstream API you're calling. Gemini is the built-in provider today (path segment `gemini`); see [ADR-0006](../adr/0006-neutral-schema-and-provider-abstraction.md) for why.

## Make a request

The route shape is `POST /v1/{provider}/{model}:streamGenerateContent`.

=== "curl"

    ```bash
    curl -N -X POST \
      http://localhost:8000/v1/gemini/gemini-3.7-flash:streamGenerateContent \
      -H "x-gateway-api-key: my-gateway-key" \
      -H "Content-Type: application/json" \
      -d '{
        "turns": [
          {"role": "user", "parts": [{"type": "text", "text": "Explain HTTP streaming in one sentence."}]}
        ]
      }'
    ```

=== "Python (httpx)"

    ```python
    import httpx

    with httpx.stream(
        "POST",
        "http://localhost:8000/v1/gemini/gemini-3.7-flash:streamGenerateContent",
        headers={"x-gateway-api-key": "my-gateway-key"},
        json={
            "turns": [
                {
                    "role": "user",
                    "parts": [{"type": "text", "text": "Explain HTTP streaming in one sentence."}],
                }
            ]
        },
    ) as response:
        for chunk in response.iter_bytes():
            print(chunk.decode(), end="")
    ```

## What comes back

An SSE stream of neutral `data: {...}` events — each carrying a `delta.parts[]` text fragment, with a terminal chunk carrying `finish_reason` and `usage`. The gateway translates the upstream provider's own streamed events into these one-for-one as they arrive; it doesn't buffer the whole response first.

## The `X-Cache` header

Every response carries an `X-Cache` header:

- `X-Cache: MISS` — the gateway called the provider for this request.
- `X-Cache: HIT` — a semantically similar cached response was replayed; the provider was never called. See [Semantic caching](semantic-caching.md).

## Missing or invalid key

```bash
curl -i -X POST http://localhost:8000/v1/gemini/gemini-3.7-flash:streamGenerateContent \
  -d '{"turns": []}'
# HTTP/1.1 401 Unauthorized
```

## Unknown provider

```bash
curl -i -X POST http://localhost:8000/v1/openai/gpt-4o:streamGenerateContent \
  -H "x-gateway-api-key: my-gateway-key" -d '{"turns": []}'
# HTTP/1.1 404 Not Found
```

Next: [Configuration](configuration.md), to see every setting the gateway understands.
