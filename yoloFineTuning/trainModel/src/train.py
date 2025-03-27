from ultralytics import YOLO

def train(modelPath, dataPath, epochs=32, batchSize=16, imgSize=320):
    model = YOLO("yolov8n.pt")  
    
    model.train(
        data=dataPath, 
        epochs=epochs,
        batch=batchSize,
        imgsz=imgSize,
        save=True,
        project="runs",  
        name="Vsmodel"  
    )

    print("Training complete!")