import os
from ultralytics import YOLO


def main():
    PROJECT_ROOT = os.path.abspath("models")

    # Просто передаем путь к файлу. Ultralytics сама создаст 'models/pretrained/' и скачает туда веса!
    model_path = os.path.join("models", "pretrained", "yolo11n.pt")
    model = YOLO(model_path)

    model.train(
        data="data/data.yaml",
        epochs=30,
        imgsz=640,
        project=PROJECT_ROOT,
        name="yolo11_ocr_det",
        device=0,
    )


if __name__ == "__main__":
    main()
