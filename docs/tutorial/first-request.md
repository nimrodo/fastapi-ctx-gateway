# Your first request

The gateway proxies Gemini's own `streamGenerateContent` wire schema unchanged — if you already know Gemini's API, you already know this one. The only two differences from calling Gemini directly:

1. You call the **gateway's** URL, not `generativelanguage.googleapis.com`.
2. You authenticate with a **gateway-issued** key (`x-gateway-api-key`), not your Gemini key — the gateway holds the real Gemini key and never exposes it to clients.

## Make a request

=== "curl"

    ```bash
    curl -N -X POST \
      http://localhost:8000/v1/gemini-2.5-flash:streamGenerateContent \
      -H "x-gateway-api-key: my-gateway-key" \
      -H "Content-Type: application/json" \
      -d '{
        "contents": [
          {"role": "user", "parts": [{"text": "Explain HTTP streaming in one sentence."}]}
        ]
      }'
    ```

=== "Python (httpx)"

    ```python
    import httpx

    with httpx.stream(
        "POST",
        "http://localhost:8000/v1/gemini-2.5-flash:streamGenerateContent",
        headers={"x-gateway-api-key": "my-gateway-key"},
        json={
            "contents": [
                {"role": "user", "parts": [{"text": "Explain HTTP streaming in one sentence."}]}
            ]
        },
    ) as response:
        for chunk in response.iter_bytes():
            print(chunk.decode(), end="")
    ```

## What comes back

A standard Gemini SSE stream — incremental `data: {...}` events, each carrying a delta of `candidates[0].content.parts[0].text`, with a terminal chunk carrying `finishReason` and `usageMetadata`. The gateway forwards these bytes to you as Gemini sends them; it doesn't buffer the whole response first.

## The `X-Cache` header

Every response carries an `X-Cache` header:

- `X-Cache: MISS` — the gateway called Gemini for this request.
- `X-Cache: HIT` — a semantically similar cached response was replayed; Gemini was never called. See [Semantic caching](semantic-caching.md).

## Missing or invalid key

```bash
curl -i -X POST http://localhost:8000/v1/gemini-2.5-flash:streamGenerateContent \
  -d '{"contents": []}'
# HTTP/1.1 401 Unauthorized
```

Next: [Configuration](configuration.md), to see every setting the gateway understands.
