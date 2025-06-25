import onnx
from onnx import helper, TensorProto

# Load original model
model = onnx.load("mobilenetv2-12.onnx")
graph = model.graph

# Find which tensor is feeding the Gemm (the 1×1280 feature)
reshape_output_name = None
for node in graph.node:
    if node.op_type == "Gemm":
        reshape_output_name = node.input[0]   # e.g. "Reshape_123"
        break

if reshape_output_name is None:
    raise RuntimeError("Could not find Gemm node in graph")

# create an Identity node
id_node = helper.make_node(
    "Identity",
    inputs=[reshape_output_name],
    outputs=["features"],
    name="MakeFeatures",
)
graph.node.append(id_node)

# Clear the existing outputs
graph.ClearField("output")

# Declare a new output for your embedding
feat_output = helper.make_tensor_value_info(
    name="features", # reshape_output_name,
    elem_type=TensorProto.FLOAT,
    shape=[1, 1280],     # batch=1, 1280 channels
)
graph.output.append(feat_output)

# Save out modified model
onnx.save(model, "mobilenetv2-embedding.onnx")
