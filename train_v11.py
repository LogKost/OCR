import os
from ultralytics import YOLO


def main():
    PROJECT_ROOT = os.path.abspath("models")

    model = YOLO("yolo11n.pt")

    results = model.train(
        data="data/data.yaml",
        epochs=30,
        imgsz=640,
        project=PROJECT_ROOT,
        name="yolo11_ocr_det",
        device=0,
    )


if __name__ == "__main__":
    main()
