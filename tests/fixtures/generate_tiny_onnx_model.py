"""One-off generator for the tiny ONNX fixture used by vectorizer tests.

Not a real embedding model — just a deterministic Gather+ReduceMean over a
fixed random embedding table, small enough to keep tests fast and offline.
Regenerate with: `uv run python tests/fixtures/generate_tiny_onnx_model.py`
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

VOCAB_SIZE = 1000
DIMS = 384
OUTPUT_PATH = Path(__file__).parent / "tiny_onnx_model" / "model.onnx"


def build() -> None:
    rng = np.random.default_rng(seed=42)
    table = rng.standard_normal((VOCAB_SIZE, DIMS)).astype(np.float32)

    table_initializer = helper.make_tensor(
        name="embedding_table",
        data_type=TensorProto.FLOAT,
        dims=table.shape,
        vals=table.flatten().tolist(),
    )

    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["seq_len"])
    embedding = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [DIMS])

    gather_node = helper.make_node(
        "Gather", inputs=["embedding_table", "input_ids"], outputs=["gathered"], axis=0
    )
    mean_node = helper.make_node(
        "ReduceMean", inputs=["gathered"], outputs=["embedding"], axes=[0], keepdims=0
    )

    graph = helper.make_graph(
        nodes=[gather_node, mean_node],
        name="tiny-fixture-embedder",
        inputs=[input_ids],
        outputs=[embedding],
        initializer=[table_initializer],
    )
    model = helper.make_model(graph, producer_name="fastapi-ctx-gateway-test-fixture")
    model.opset_import[0].version = 13  # ReduceMean's `axes` is still an attribute here
    onnx.checker.check_model(model)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(OUTPUT_PATH))
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
