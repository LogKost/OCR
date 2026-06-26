import os
from ultralytics import RTDETR


def main():
    PROJECT_ROOT = os.path.abspath("models")
    run_name = "rtdetr_ocr_det"

    # Для RT-DETR логика точно такая же
    model_path = os.path.join("models", "pretrained", "rtdetr-l.pt")
    model = RTDETR(model_path)

    model.train(
        data="data/data.yaml",
        epochs=30,
        imgsz=640,
        project=PROJECT_ROOT,
        name=run_name,
        device=0,
        batch=4,
        workers=2,
        amp=True,
    )


if __name__ == "__main__":
    main()
