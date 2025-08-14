# yolo_webcam.py
import time
import numpy as np
import cv2 as cv
from yolo_util import YOLO  # ← NMS込みONNXを呼ぶクラス（frameを受け取って前処理する想定）

ONNX_PATH = "yolov5n_with_nms.onnx"  # ★ NMS込みONNX
IMG_SIZE  = 640                      # ★ エクスポート時の入力サイズに合わせる
CONF_THRES = 0.25                    # 追加で絞る場合

def parse_outputs(outputs):
    """
    NMS込みONNXの代表的な2パターンを吸収して [N,6](xyxy,conf,cls) に揃える
    1) 4出力: [num_dets(1), boxes(1,M,4), scores(1,M), labels(1,M)]
    2) 1出力: (M,6) or (1,M,6)
    """
    if isinstance(outputs, (list, tuple)) and len(outputs) == 4:
        num_dets, boxes, scores, labels = outputs
        m = int(num_dets[0])
        if m == 0:
            return np.zeros((0, 6), dtype=np.float32)
        dets = []
        for i in range(m):
            conf = float(scores[0][i])
            if conf < CONF_THRES:
                continue
            x1, y1, x2, y2 = boxes[0][i]
            cls_id = float(labels[0][i])
            dets.append([x1, y1, x2, y2, conf, cls_id])
        return np.array(dets, dtype=np.float32)
    else:
        dets = outputs if isinstance(outputs, np.ndarray) else np.array(outputs)
        if dets.ndim == 3 and dets.shape[0] == 1:
            dets = dets[0]
        if dets.ndim == 2 and dets.shape[1] >= 6:
            return dets[:, :6].astype(np.float32)
        return np.zeros((0, 6), dtype=np.float32)

def draw_dets(frame, dets):
    """入力IMG_SIZE基準のxyxyをframe実サイズへスケールして描画"""
    h, w = frame.shape[:2]
    sx, sy = w / IMG_SIZE, h / IMG_SIZE
    for x1, y1, x2, y2, conf, cls_id in dets:
        x1, x2 = int(x1 * sx), int(x2 * sx)
        y1, y2 = int(y1 * sy), int(y2 * sy)
        cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv.putText(frame, f"{int(cls_id)}:{conf:.2f}", (x1, max(0, y1 - 6)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv.LINE_AA)

def main():
    yolo = YOLO(ONNX_PATH, imgsz=IMG_SIZE)  # ★ frameを渡せば内部で前処理→ONNX実行→出力list/ndarrayを返す想定

    cap = cv.VideoCapture(0, cv.CAP_V4L2)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    # 速度安定用（環境で調整）
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))  # 映らなければコメントアウト

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Can't receive frame. Exiting...")
            break

        t0 = time.time()
        outputs = yolo(frame)             # ★ ここは前処理しないでframeをそのまま渡す
        fps = 1.0 / (time.time() - t0)

        dets = parse_outputs(outputs)     # NMS込み出力を統一形に
        draw_dets(frame, dets)            # そのまま描画

        cv.putText(frame, f"FPS: {fps:.1f}", (5, 22),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.imshow("YOLOv5 ONNX (with NMS)", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
