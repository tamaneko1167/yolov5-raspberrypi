import numpy as np
import cv2 as cv
from yolo_util import YOLO, non_max_suppression, box_iou, nms
import time

print(cv.getBuildInformation())


def augment_img(img, out, fps):
    pred = non_max_suppression(out, 0.45, 0.45)
    for i, det in enumerate(pred):  # per image
        for *xyxy, conf, cls in reversed(det):
            xyxy = [int(i) for i in xyxy]
            cv.rectangle(img, xyxy[0:2], xyxy[2:4], color=(255, 0, 0))
            #cv.putText(img, f"{class_names[int(cls)+1]}", xyxy[0:2], cv.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    cv.putText(img, "FPS: {:.1f}".format(fps), (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, 255)

yolo = YOLO("yolov5n.onnx") 

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # start time measurement
    start_time = time.time()

    # perform yolo classification
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (320, 320))
    imgf = img.astype(np.float32) / 255.
    imgf = np.ascontiguousarray(imgf)
    imgf = np.expand_dims(imgf, 0)
    
    pred = yolo(imgf)
    
    # calculate the FPS
    fps = 1.0 / (time.time() - start_time)

    # augment the frame with bboxes
    augment_img(img, pred, fps)
    augmentedFrame = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    augmentedFrame = cv.resize(augmentedFrame, (640, 640))
    
    # Display the resulting frame
    cv.imshow('frame', augmentedFrame)
    if cv.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
