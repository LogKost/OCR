import streamlit as st
from PIL import Image
import time
import os
from datetime import datetime
import json
import torch
import easyocr
import numpy as np

st.set_page_config(layout="wide")

st.title("Оптическая система распознавания текста в городской среде")
st.write("Модуль автоматической локализации и распознавания текстовой информации на объектах инфраструктуры.")

st.sidebar.header("Конфигурация системы")


@st.cache_resource
def determine_hardware_device():
    cuda_available = torch.cuda.is_available()
    return "cuda" if cuda_available else "cpu", cuda_available, "CUDA" if cuda_available else "CPU"


DEVICE_TYPE, IS_GPU, DEVICE_LABEL = determine_hardware_device()
st.sidebar.text(f"Среда: {DEVICE_LABEL}")

model_options = {}

models_dir = "models"
if os.path.exists(models_dir):
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith(".pt") and "last.pt" not in file:
                arch_name = os.path.basename(root) if os.path.basename(root) != "weights" else os.path.basename(
                    os.path.dirname(root))
                model_options[f"{arch_name} ({file})"] = os.path.relpath(os.path.join(root, file))

for file in os.listdir("."):
    if file.endswith(".pt") and os.path.isfile(file):
        model_options[f"Корень ({file})"] = file

if model_options:
    selected_model_label = st.sidebar.selectbox("Архитектура модели:", list(model_options.keys()))
    model_path = model_options[selected_model_label]

    if os.path.exists(model_path):
        weight_size = os.path.getsize(model_path) / (1024 * 1024)
        st.sidebar.text(f"Размер модели: {weight_size:.2f} МБ")
else:
    st.sidebar.warning("Файлы .pt не обнаружены")
    model_path = None


@st.cache_resource
def load_ocr_engine():
    return easyocr.Reader(['ru', 'en'], gpu=IS_GPU)


recognizer_engine = load_ocr_engine()


class UniversalInferenceFactory:
    def __init__(self, path, device_type, recognizer):
        self.path = path
        self.device_type = device_type
        self.recognizer = recognizer
        self.model_object = self._initialize_model()

    def _initialize_model(self):
        if not self.path:
            return None

        if self.path.endswith(".pt"):
            from ultralytics import YOLO
            model = YOLO(self.path)
            model.to(self.device_type)
            return model
        raise ValueError(f"Неподдерживаемый формат модели: {self.path}")

    def execute_inference(self, pil_image):
        if self.model_object is None:
            return [], None, 0.0

        img_np = np.array(pil_image)
        parsed_results = []

        start_time = time.time()
        outputs = self.model_object(pil_image, imgsz=640)
        inference_duration = time.time() - start_time

        annotated_image = outputs[0].plot()

        for box in outputs[0].boxes:
            conf = box.conf[0].item() * 100
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            cropped = img_np[max(0, y1):min(img_np.shape[0], y2), max(0, x1):min(img_np.shape[1], x2)]
            text_output = "Н/Д"

            if cropped.size > 0:
                ocr_res = self.recognizer.readtext(cropped)
                if ocr_res:
                    text_output = " ".join([res[1] for res in ocr_res])

            if conf >= 60.0 and text_output != "Н/Д":
                auto_quality_mark = "Корректно"
            elif conf < 40.0:
                auto_quality_mark = "Ошибка локализации (Низкий Confidence)"
            else:
                auto_quality_mark = "Ошибка распознавания текста (OCR)"

            parsed_results.append({
                "coordinates": [x1, y1, x2, y2],
                "confidence": conf,
                "text": text_output,
                "evaluation": auto_quality_mark
            })

        return parsed_results, annotated_image, inference_duration


engine = UniversalInferenceFactory(model_path, DEVICE_TYPE, recognizer_engine) if model_path else None

uploaded_files = st.file_uploader(
    "Загрузка изображений для пакетного анализа:",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files and engine is not None:
    st.success(f"Успешно загружено файлов для анализа: {len(uploaded_files)}")

    session_records = []

    for file_idx, uploaded_file in enumerate(uploaded_files):
        st.markdown(f"## Обработка файла: `{uploaded_file.name}`")

        col1, col2 = st.columns(2)
        source_image = Image.open(uploaded_file)

        with col1:
            st.subheader("Исходное изображение")
            st.image(source_image, use_container_width=True)

        detected_blocks, visual_output, exec_time = engine.execute_inference(source_image)

        st.subheader("Аналитический отчет по кадру")
        total_segments = len(detected_blocks)

        if total_segments > 0:
            for idx, block in enumerate(detected_blocks):
                st.markdown(
                    f"**Сегмент №{idx + 1}** | "
                    f"Уверенность детекции: `{block['confidence']:.2f}%` | "
                    f"Текст: `{block['text']}` | "
                    f"Авто-статус: **{block['evaluation']}**"
                )

                log_data = {
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "filename": uploaded_file.name,
                    "selected_architecture": selected_model_label,
                    "hardware_environment": DEVICE_LABEL,
                    "inference_time_seconds": round(exec_time, 4),
                    "detection_confidence_percent": round(block['confidence'], 2),
                    "bounding_box_geometry": block['coordinates'],
                    "ocr_output_text": block['text'],
                    "auto_quality_mark": block['evaluation'],
                    "is_tp": 1 if block['evaluation'] == "Корректно" else 0,
                    "is_fp": 1 if block['evaluation'] == "Ошибка локализации (Низкий Confidence)" else 0,
                    "is_fn": 1 if block['evaluation'] == "Ошибка распознавания текста (OCR)" else 0
                }
                session_records.append(log_data)

            with col2:
                st.subheader("Результат локализации (Text Detection)")
                if visual_output is not None:
                    st.image(visual_output, channels="BGR", use_container_width=True)
        else:
            with col2:
                st.subheader("Результат обработки")
                st.image(source_image, use_container_width=True)
            st.warning("Текстовые области на данном кадре не обнаружены.")

            log_data = {
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "filename": uploaded_file.name,
                "selected_architecture": selected_model_label,
                "hardware_environment": DEVICE_LABEL,
                "inference_time_seconds": round(exec_time, 4),
                "detection_confidence_percent": 0.0,
                "bounding_box_geometry": [],
                "ocr_output_text": "Н/Д",
                "auto_quality_mark": "Объекты не найдены",
                "is_tp": 0,
                "is_fp": 0,
                "is_fn": 1
            }
            session_records.append(log_data)

        st.text(f"Время анализа кадра: {exec_time:.4f} сек. | Количество найденных сегментов: {total_segments}")
        st.markdown("---")

    if session_records:
        os.makedirs("results", exist_ok=True)
        history_path = "results/history.json"

        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as f:
                try:
                    history_records = json.load(f)
                except json.JSONDecodeError:
                    history_records = []
        else:
            history_records = []

        history_records.extend(session_records)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_records, f, ensure_ascii=False, indent=4)
        st.sidebar.success(f"Добавлено записей в лог: {len(session_records)}")