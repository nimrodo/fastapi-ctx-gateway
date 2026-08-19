# Using it as a library

`fastapi-ctx-gateway` is a proper importable package, not just a runnable service — another project can depend on it and use individual pieces.

## The app factory

There's no module-level app instance anywhere in the package. Every caller builds its own via [`create_app()`][fastapi_ctx_gateway.app.create_app]:

```python
from fastapi_ctx_gateway import create_app, Settings

app = create_app(Settings())
```

This means you can construct multiple independently-configured instances — one per test, or mounted as a sub-app inside a larger service:

```python
from fastapi import FastAPI
from fastapi_ctx_gateway import create_app, Settings

parent = FastAPI()
parent.mount("/gateway", create_app(Settings(gemini_upstream_key="...")))
```

## Standalone components

Every reusable piece is importable on its own, with no FastAPI or routing dependency pulled in:

```python
from fastapi_ctx_gateway import (
    TokenBudgetPruner,
    RateLimiter,
    TokenEstimator,
    CircuitBreaker,
    SemanticCache,
    OnnxVectorizer,
)
```

For example, using the pruner completely standalone:

```python
from fastapi_ctx_gateway import TokenBudgetPruner
from fastapi_ctx_gateway.config import TokenBudgetConfig
from fastapi_ctx_gateway.schemas.gemini import Content, Part

pruner = TokenBudgetPruner(TokenBudgetConfig(default=1000))
result = pruner.prune(
    contents=[Content(role="user", parts=[Part(text="hi")])],
    system_instruction=None,
    model="gemini-2.5-flash",
)
print(result.pruned, result.dropped_turn_count)
```

## What's *not* exported

`fastapi_ctx_gateway.cli.main` (the CLI entrypoint) is deliberately **not** re-exported from the top-level package — importing `fastapi_ctx_gateway` never has the side effect of parsing the environment or being runnable as a server. Call it explicitly if you need it: `from fastapi_ctx_gateway.cli import main`.

See the [Reference](../reference/index.md) for the full public API surface.
