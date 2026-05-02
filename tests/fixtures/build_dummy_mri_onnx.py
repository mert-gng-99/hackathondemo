"""Build a tiny ONNX MRI classifier fixture for API/model tests."""
from __future__ import annotations

from pathlib import Path


def build(path: Path, logits: tuple[float, float] = (0.1, 2.0)) -> Path:
    """Write an ONNX model that returns constant logits for any MRI tensor."""
    import onnx
    from onnx import TensorProto, helper

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 8, 8, 8])
    output_info = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])
    value = helper.make_tensor("const_logits", TensorProto.FLOAT, [1, 2], list(logits))
    node = helper.make_node("Constant", inputs=[], outputs=["logits"], value=value)
    graph = helper.make_graph([node], "dummy_mri_classifier", [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save(model, path)
    return path
