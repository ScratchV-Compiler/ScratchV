#!/usr/bin/env python3
"""Generate a minimal CNN ONNX model for CI testing when cnn.onnx is missing."""
import onnx
import numpy as np
from onnx import helper, TensorProto

MODEL_PATH = "models/graph/cnn.onnx"

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 64, 64])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [32, 3, 3, 3])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [32])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 32, 62, 62])
w_init = np.random.randn(32, 3, 3, 3).astype(np.float32)
b_init = np.random.randn(32).astype(np.float32)
conv = helper.make_node('Conv', ['X', 'W', 'B'], ['Y'], kernel_shape=[3, 3])
graph = helper.make_graph(
    [conv], 'graph', [X, W, B], [Y],
    initializer=[
        helper.make_tensor('W', TensorProto.FLOAT, [32, 3, 3, 3], w_init.tobytes(), raw=True),
        helper.make_tensor('B', TensorProto.FLOAT, [32], b_init.tobytes(), raw=True),
    ],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
onnx.save(model, MODEL_PATH)
print(f'  Generated minimal CNN: {MODEL_PATH}')
