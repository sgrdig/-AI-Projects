
import cv2
import numpy as np
import onnxruntime as ort

"""Ce code est base sur le code presents dans ce repo https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-ONNXRuntime/main.py Adapte pour l'utilisation d'un camera en temps reel"""

def letterbox(img: np.ndarray, new_shape: tuple[int, int] = (640, 640)) -> tuple[np.ndarray, tuple[int, int]]:
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, (top, left)

def draw_detections(img: np.ndarray, box: list[float], score: float, class_id: int, color_palette: np.ndarray, classes: list[str]) -> None:
    x1, y1, w, h = box
    color = color_palette[class_id]
    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
    label = f"{classes[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def preprocess(img: np.ndarray, input_width: int, input_height: int) -> tuple[np.ndarray, tuple[int, int], np.ndarray, int, int]:
    img_height, img_width = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, pad = letterbox(img, (input_width, input_height))
    image_data = np.array(img) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    return image_data, pad, img, img_height, img_width

def postprocess(input_image: np.ndarray, output: list[np.ndarray], pad: tuple[int, int], img_height: int, img_width: int, confidence_thres: float, iou_thres: float, color_palette: np.ndarray, classes: list[str]) -> np.ndarray:
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []
    gain = min(input_image.shape[0] / img_height, input_image.shape[1] / img_width)
    outputs[:, 0] -= pad[1]
    outputs[:, 1] -= pad[0]
    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)
        if max_score >= confidence_thres:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            print(x , y, w , h)
            left = int((x - w / 2) / gain)
            top = int((y - h / 2) / gain)
            width = int(w / gain)
            height = int(h / gain)
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)
    for i in indices:
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        draw_detections(input_image, box, score, class_id, color_palette, classes)
    return input_image

# def detect(onnx_model: str, input_image: str, confidence_thres: float = 0.1, iou_thres: float = 0.5) -> np.ndarray:
#     session = ort.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])
#     model_inputs = session.get_inputs()
#     input_shape = model_inputs[0].shape
#     input_width = input_shape[2]
#     input_height = input_shape[3]
#     classes = "Drones"
#     color_palette = np.random.uniform(0, 255, size=(len(classes), 3))
#     img_data, pad, img, img_height, img_width = preprocess(input_image, input_width, input_height)
#     outputs = session.run(None, {model_inputs[0].name: img_data})
#     return postprocess(img, outputs, pad, img_height, img_width, confidence_thres, iou_thres, color_palette, classes)


def detect(session, img: np.ndarray, input_width: int, input_height: int, confidence_thres: float, iou_thres: float, color_palette: np.ndarray, classes: list[str]) -> np.ndarray:
    img_data, pad, img, img_height, img_width = preprocess(img, input_width, input_height)
    outputs = session.run(None, {session.get_inputs()[0].name: img_data})
    return postprocess(img, outputs, pad, img_height, img_width, confidence_thres, iou_thres, color_palette, classes)

# Example usage
# onnx_model_path = "../models/VSModel_1.2_40_640_11s.onnx"
# confidence_thres = 0.1
# iou_thres = 0.5

# classes = "Drones"
# blue_dark = np.array([0, 0, 139])
# color_palette = np.tile(blue_dark, (len(classes), 1))    


# session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
# model_inputs = session.get_inputs()
# input_shape = model_inputs[0].shape
# input_width = input_shape[2]
# input_height = input_shape[3]

# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     output_image = detect(session, frame, input_width, input_height, confidence_thres, iou_thres, color_palette, classes)
#     cv2.imshow("Output", output_image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



def main(onnx_model_path : str , confidence_thres :  float ,iou_thres : float  ):

    classes = ["Drones"]
    blue_dark = np.array([0, 0, 139])
    color_palette = np.tile(blue_dark, (len(classes), 1))    

    session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    model_inputs = session.get_inputs()
    input_shape = model_inputs[0].shape
    input_width = input_shape[2]
    input_height = input_shape[3]


    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        output_image = detect(session, frame, input_width, input_height, confidence_thres, iou_thres, color_palette, classes)
        cv2.imshow("Output", output_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
