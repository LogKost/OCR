import os
from ultralytics import RTDETR


def main():
    PROJECT_ROOT = os.path.abspath("models")
    run_name = "rtdetr_ocr_det"

    model = RTDETR("rtdetr-l.pt")

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
