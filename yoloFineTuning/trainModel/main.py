#Import local lib
from src.dataModification.renamingData import verifyImageLabels 
from src.train import train
import os
import torch
import numpy as np
import random




if __name__ =="__main__":
    seed = 42
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    print(os.getcwd())    
    verifyImageLabels(
        imageDir = "datasets/images/train",
        labelDir ="datasets/labels/train"
    )
    verifyImageLabels(
        imageDir = "datasets/images/val",
        labelDir = "datasets/labels/val"
    )
    train(
        modelPath = "models/yolov8n.pt",
        dataPath = "src/data.yaml")
    

    pass