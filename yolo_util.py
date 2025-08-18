import cv2
import numpy as np
import onnxruntime as ort

class_names = {
    0: "__background__",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    12: "stop sign",
    13: "parking meter",
    14: "bench",
    15: "bird",
    16: "cat",
    17: "dog",
    18: "horse",
    19: "sheep",
    20: "cow",
    21: "elephant",
    22: "bear",
    23: "zebra",
    24: "giraffe",
    25: "backpack",
    26: "umbrella",
    27: "handbag",
    28: "tie",
    29: "suitcase",
    30: "frisbee",
    31: "skis",
    32: "snowboard",
    33: "sports ball",
    34: "kite",
    35: "baseball bat",
    36: "baseball glove",
    37: "skateboard",
    38: "surfboard",
    39: "tennis racket",
    40: "bottle",
    41: "wine glass",
    42: "cup",
    43: "fork",
    44: "knife",
    45: "spoon",
    46: "bowl",
    47: "banana",
    48: "apple",
    49: "sandwich",
    50: "orange",
    51: "broccoli",
    52: "carrot",
    53: "hot dog",
    54: "pizza",
    55: "donut",
    56: "cake",
    57: "chair",
    58: "couch",
    59: "potted plant",
    60: "bed",
    61: "dining table",
    62: "toilet",
    63: "tv",
    64: "laptop",
    65: "mouse",
    66: "remote",
    67: "keyboard",
    68: "cell phone",
    69: "microwave",
    70: "oven",
    71: "toaster",
    72: "sink",
    73: "refrigerator",
    74: "book",
    75: "clock",
    76: "vase",
    77: "scissors",
    78: "teddy bear",
    79: "hair drier",
    80: "toothbrush",
}


def xyxy2xywh(x):
    y = x.copy()
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    y = x.copy()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


class YOLO:
    """
    NMS込みONNXモデルを実行する簡易クラス。
    - __call__(frame) で生frame(BGR)を渡せば前処理→session.run()→ONNXの生出力(list)を返す。
    - 既に作ったblob(1,3,H,W)を渡してもOK。.
    """

    def __init__(self, model_path="yolov5n_with_nms.onnx", imgsz=640, providers=("CPUExecutionProvider",)):
        self.session = ort.InferenceSession(model_path, providers=list(providers))
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = int(imgsz)

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # (3,H,W)
        return np.expand_dims(blob, axis=0)  # (1,3,H,W)

    def __call__(self, x):
        if isinstance(x, np.ndarray) and x.ndim == 4:
            blob = x.astype(np.float32)
        else:
            blob = self.preprocess(x)
        return self.session.run(None, {self.input_name: blob})
