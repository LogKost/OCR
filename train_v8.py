import os
from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")

    PROJECT_ROOT = os.path.abspath("models")

    results = model.train(
        data="data/data.yaml",
        epochs=30,
        imgsz=640,
        project=PROJECT_ROOT,
        name="yolov8_ocr_det",
        device=0,
    )
    print("Обучение успешно завершено!")


if __name__ == "__main__":
    main()
