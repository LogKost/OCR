import os
import time
import torch


class UniversalInferenceFactory:
    def __init__(self, path, device_type):
        self.path = path
        self.device_type = device_type
        self.model_object = self._initialize_model()

    def _initialize_model(self):
        if not self.path:
            return None
        from ultralytics import YOLO

        model = YOLO(os.path.abspath(self.path))
        model.to(self.device_type)
        return model

    def execute_detection(self, pil_image):
        if self.model_object is None:
            return [], None, 0.0

        start_yolo = time.time()

        with torch.no_grad():
            outputs = self.model_object(pil_image, imgsz=640)

        yolo_duration = time.time() - start_yolo

        annotated_image = outputs[0].plot()[:, :, ::-1]

        detected_boxes = []
        for box in outputs[0].boxes:
            conf = box.conf[0].item() * 100
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detected_boxes.append({"coordinates": [x1, y1, x2, y2], "confidence": conf})

        return detected_boxes, annotated_image, yolo_duration
