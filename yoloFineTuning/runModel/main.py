from src.ptRun import droneDetection
from src.onnxRun import main
import argparse
from src.convert import convert

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", action="store_true", help="Use Onnx or pytorch")
    parser.add_argument("--modelPath", type= str , default = "models/1.onnx" , help = "Path to ur model pt or Onnx")
    parser.add_argument("--conf-thres", type=float, default=0.41, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--convert", action="store_true", help="convert .pt to .onnx files")

    args = parser.parse_args()

    if  args.pt :
        print("Pt Model...")
        droneDetection(confidence=args.conf_thres, iou_thres=args.iou_thres)
   
    elif args.convert:
        print("Converting Model...")
        convert(modelPath = args.modelPath)

    else : 
        print("Onnx Model...")
        main(onnx_model_path = args.modelPath ,confidence_thres = args.conf_thres , iou_thres = args.iou_thres )
