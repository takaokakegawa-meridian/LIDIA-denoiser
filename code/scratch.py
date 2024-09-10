# https://github.com/sithu31296/PyTorch-ONNX-TFLite

import onnx

print("imports success")

# load in the model and set to eval.
model = Model()
model.load_state_dict(torch.load(pt_model_path, map_location='cpu')).eval()

sample_input = torch.rand((batch_size, channels, height, width))

torch.onnx.export(
    model,                  # PyTorch Model
    sample_input,                    # Input tensor
    onnx_model_path,        # Output file (eg. 'output_model.onnx')
    opset_version=12,       # Operator support version
    input_names=['input'],   # Input tensor name (arbitary)
    output_names=['output'] # Output tensor name (arbitary)
)


onnx_model = onnx.load(onnx_model_path)

tf_rep = prepare(onnx_model)

tf_rep.export_graph(tf_model_path)

converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
tflite_model = converter.convert()

# Save the model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)