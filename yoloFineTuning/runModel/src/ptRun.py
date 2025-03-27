import cv2
from ultralytics import YOLO
import time
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def binning_2x2(image):
    return cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_NEAREST)

def enhance_image_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def adaptive_brightness_contrast(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)

    if mean_brightness < 50:
        alpha = 0.7
        beta = 30
    elif mean_brightness > 200:
        alpha = 1.2
        beta = -30
    else:
        alpha = 1.0
        beta = -30

    enhanced_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    return enhanced_frame

def process_frame(frame, model):
    frame = adaptive_brightness_contrast(frame)  # Apply brightness and contrast adjustment
    enhanced_frame = enhance_image_clahe(frame)   # Apply CLAHE enhancement
    results = model.track(enhanced_frame, persist=True, conf=0.15, tracker="bytetrack.yaml")
    return results[0].plot()

def droneDetection(confidence : float = 0.3 , iou_thres :float = 0.3 , modelPath :str = "models/1.pt" , frameSkips : int = 3) :
    try:
        model = YOLO(modelPath)
        model.fuse()
        print("Model loaded and fused")
        model = model.cpu()
    except Exception as e:
        print(e)
        return

    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    except Exception as e:
        print(e)
        return
    
    prev_time = cv2.getTickCount()
    frame_counter = 0
    executor = ThreadPoolExecutor(max_workers=12)  

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            frameSkips += 1

            if frame_counter % frameSkips == 0:
                future = executor.submit(process_frame, frame, model)
                annotated_frame = future.result()

                curr_time = cv2.getTickCount()
                time_in_secs = (curr_time - prev_time) / cv2.getTickFrequency()
                fps = 1 / time_in_secs
                prev_time = curr_time

                cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown()
