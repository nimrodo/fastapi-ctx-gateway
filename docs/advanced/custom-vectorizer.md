# Custom vectorizer

[`OnnxVectorizer`][fastapi_ctx_gateway.cache.OnnxVectorizer] wraps an injected ONNX `InferenceSession` and a `tokenize` function, conforming to RedisVL's `BaseVectorizer` interface. It's the natural extension point if you want to swap in a different embedding backend.

## The interface

Subclasses of RedisVL's `BaseVectorizer` implement two methods:

```python
def _embed(self, content: str, **kwargs) -> list[float]: ...
async def _aembed(self, content: str, **kwargs) -> list[float]: ...
```

`OnnxVectorizer._aembed` hands the sync ONNX call to a worker thread via `asyncio.to_thread` — never call the sync `_embed` path directly from a request handler, since ONNX Runtime's inference call is blocking and would stall the event loop.

## Wiring in a real model

```python
import onnxruntime as ort
from fastapi_ctx_gateway.cache import OnnxVectorizer

session = ort.InferenceSession("/path/to/real-model.onnx")


def tokenize(text: str) -> list[int]:
    # Replace with a real HF tokenizer matched to your model
    return my_tokenizer.encode(text)


vectorizer = OnnxVectorizer(session=session, tokenize=tokenize, dims=384)
```

The placeholder [`simple_char_code_tokenize`][fastapi_ctx_gateway.cache.vectorizer.simple_char_code_tokenize] used when no real tokenizer is configured is dependency-light and good enough to exercise the pipeline, but isn't semantically meaningful — swap it out once you have a real model and matching tokenizer deployed.

## A non-ONNX backend

Nothing in `SemanticCache` depends on ONNX specifically — it only needs an object with `embed`/`aembed`. Subclass RedisVL's `BaseVectorizer` directly for a hosted embedding API, a different local runtime, or anything else:

```python
from redisvl.utils.vectorize.base import BaseVectorizer


class MyVectorizer(BaseVectorizer):
    model: str = "my-model"

    def _embed(self, content: str, **kwargs) -> list[float]: ...

    async def _aembed(self, content: str, **kwargs) -> list[float]: ...
```

Then pass it to [`SemanticCache`][fastapi_ctx_gateway.cache.SemanticCache] the same way `app.py` wires up the default one.
