import onnx

model_path = '/home/dean/Downloads/red_arrow.onnx'

try:
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    print(f"The model at {model_path} is a valid ONNX model.")
except onnx.onnx_cpp2py_export.checker.ValidationError as e:
    print(f"Error: The model at {model_path} is not a valid ONNX model. {e}")
