import onnxruntime as ort
import numpy as np
import cv2

# === 1. モデルと画像のパス ===
onnx_path = "yolov5/runs/train/yolov5n_voc_baseline/weights/best.onnx" # モデルのパスを指定
img_path = "yolov5/data/images/bus.jpg"

# === 2. ONNXモデルをロード ===
session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name

# === 3. 画像読み込み & 前処理 ===
img = cv2.imread(img_path)  # BGRで読み込まれる
img = cv2.resize(img, (640, 640))  # モデルの入力サイズに合わせる
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGBに変換
img = img.astype(np.float32) / 255.0  # 正規化
img = np.transpose(img, (2, 0, 1))  # HWC → CHW
img = np.expand_dims(img, axis=0)  # バッチ次元追加 → [1, 3, 640, 640]
input_tensor = img.copy()  # onnxruntimeはcontiguousな配列を好む

# === 4. 推論実行 ===
outputs = session.run(None, {input_name: input_tensor})

# === 5. 出力確認 ===
output_array = outputs[0]  # 通常は1つだけ返ってくる
print(f"Output shape: {output_array.shape}")
print(f"First few outputs:\n{output_array[0][:5]}")

