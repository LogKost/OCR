import os
from ultralytics import YOLO

def main():
    PROJECT_ROOT = os.path.abspath("models")
    model = YOLO("yolov9t.pt")

    model.train(
        data="data/data.yaml",
        epochs=30,
        imgsz=640,
        project=PROJECT_ROOT,
        name="yolov9_ocr_det",
        device=0
    )
if __name__ == "__main__": main()