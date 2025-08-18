import onnxruntime as ort
session = ort.InferenceSession("runs/train/yolov5n_d076_sppNone_g1004/weights/best.onnx")
print("Outputs:", [o.name for o in session.get_outputs()])