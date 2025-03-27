from ultralytics import YOLO


def convert(modelPath : str):
    try : 
        model = YOLO(modelPath)
        model.export(format="onnx")
    except Exception as e :
        print(f"An error occured : {e}")