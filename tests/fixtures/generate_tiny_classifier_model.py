"""One-off generator for the tiny ONNX fixture used by guardrails classifier tests.

Not a real prompt-injection model — a deterministic bag-of-words linear
classifier over a fixed 20-word vocabulary (Gather + ReduceSum), small enough
to keep tests fast and offline. Exists only to exercise the wiring
(tokenize -> ONNX inference -> softmax -> threshold -> fail-open), the same
role tests/fixtures/tiny_onnx_model plays for the embedding vectorizer.

Regenerate with: `uv run python tests/fixtures/generate_tiny_classifier_model.py`
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

# Word -> id. id 0 is reserved for unknown/out-of-vocab tokens. Chosen to
# cover the three categories the old regex heuristic matched, so tests can
# show the classifier still catches phrasing the anchored regex would miss
# (reordered words, extra filler, mixed case) without being a real model.
VOCAB: dict[str, int] = {
    "<unk>": 0,
    "ignore": 1,
    "disregard": 2,
    "forget": 3,
    "override": 4,
    "previous": 5,
    "prior": 6,
    "instructions": 7,
    "dan": 8,
    "jailbreak": 9,
    "developer": 10,
    "restrictions": 11,
    "roleplay": 12,
    "system": 13,
    "please": 14,
    "summarize": 15,
    "weather": 16,
    "recipe": 17,
    "meeting": 18,
    "thanks": 19,
}
VOCAB_SIZE = len(VOCAB)
NUM_CLASSES = 2  # [benign, injection]
OUTPUT_PATH = Path(__file__).parent / "tiny_classifier_model" / "model.onnx"


def build() -> None:
    # Injection-signaling words get a strong positive weight on the
    # "injection" column and a negative one on "benign"; everything else
    # (including <unk>) is weighted toward "benign". Purely deterministic,
    # not learned.
    table = np.zeros((VOCAB_SIZE, NUM_CLASSES), dtype=np.float32)
    table[:, 0] = 0.1  # small constant benign bias for every token
    injection_words = {"ignore", "disregard", "forget", "override", "dan", "jailbreak", "roleplay"}
    for word, idx in VOCAB.items():
        if word in injection_words:
            table[idx] = [-1.0, 3.0]

    table_initializer = helper.make_tensor(
        name="embedding_table",
        data_type=TensorProto.FLOAT,
        dims=table.shape,
        vals=table.flatten().tolist(),
    )

    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["seq_len"])
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [NUM_CLASSES])

    # ReduceSum moved `axes` from an attribute to a second input in opset 13
    # (unlike ReduceMean, still attribute-based there) — pass it as an
    # initializer rather than a graph input.
    axes_initializer = helper.make_tensor(
        name="reduce_axes", data_type=TensorProto.INT64, dims=[1], vals=[0]
    )
    gather_node = helper.make_node(
        "Gather", inputs=["embedding_table", "input_ids"], outputs=["gathered"], axis=0
    )
    sum_node = helper.make_node(
        "ReduceSum", inputs=["gathered", "reduce_axes"], outputs=["logits"], keepdims=0
    )

    graph = helper.make_graph(
        nodes=[gather_node, sum_node],
        name="tiny-fixture-injection-classifier",
        inputs=[input_ids],
        outputs=[logits],
        initializer=[table_initializer, axes_initializer],
    )
    model = helper.make_model(graph, producer_name="fastapi-ctx-gateway-test-fixture")
    model.opset_import[0].version = 13  # ReduceSum's `axes` is still an attribute here
    onnx.checker.check_model(model)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(OUTPUT_PATH))
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
