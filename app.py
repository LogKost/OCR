import json
import os
import re
import time
from datetime import datetime
import easyocr
import numpy as np
import streamlit as st
import torch
from PIL import Image

st.set_page_config(layout="wide")


def clean_mixed_text(text):
    if text == "Н/Д":
        return text
    text = re.sub(r"\[ential", "Central", text)
    text = re.sub(r"^\s*[\(\[\{]\s*", "C", text)
    text = re.sub(r"\s*[\)\]\}]$", "", text)

    words = text.split()
    fixed_words = []
    for word in words:
        lat_count = len(re.findall(r"[a-zA-Z]", word))
        cyr_count = len(re.findall(r"[а-яА-ЯёЁ]", word))
        if lat_count > cyr_count and cyr_count > 0:
            replacements = {
                "в": "re",
                "а": "a",
                "е": "e",
                "о": "o",
                "р": "p",
                "с": "c",
                "х": "x",
                "Т": "T",
            }
            for cyr_char, lat_char in replacements.items():
                word = word.replace(cyr_char, lat_char)
        fixed_words.append(word)

    text = " ".join(fixed_words)
    text = re.sub(r"\b[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    sticks_count = len(re.findall(r"[Ii1l|!\[\]\(\)]", text))
    total_chars = len(text.replace(" ", ""))
    if total_chars > 0 and (sticks_count / total_chars) > 0.6:
        return "Н/Д"
    return text if text else "Н/Д"


st.title("Оптическая система распознавания текста в городской среде")
st.write(
    "Модуль автоматической локализации и распознавания текстовой информации на объектах инфраструктуры."
)

st.sidebar.header("Конфигурация системы")


@st.cache_resource
def determine_hardware_device():
    cuda_available = torch.cuda.is_available()
    return (
        "cuda" if cuda_available else "cpu",
        cuda_available,
        "CUDA" if cuda_available else "CPU",
    )


DEVICE_TYPE, IS_GPU, DEVICE_LABEL = determine_hardware_device()
st.sidebar.text(f"Среда: {DEVICE_LABEL}")

models_dir = "models"


@st.cache_data
def scan_for_models(directory):
    options = {}
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".pt") and "last.pt" not in file:
                    arch_name = (
                        os.path.basename(root)
                        if os.path.basename(root) != "weights"
                        else os.path.basename(os.path.dirname(root))
                    )
                    options[f"{arch_name} ({file})"] = os.path.relpath(
                        os.path.join(root, file)
                    )
    return options


model_options = scan_for_models(models_dir)

if model_options:
    selected_model_label = st.sidebar.selectbox(
        "Архитектура модели:", list(model_options.keys())
    )
    model_path = model_options[selected_model_label]

    if os.path.exists(model_path):
        weight_size = os.path.getsize(model_path) / (1024 * 1024)
        st.sidebar.text(f"Размер модели: {weight_size:.2f} МБ")
else:
    st.sidebar.warning("Файлы моделей не найдены в 'models'")
    model_path = None


@st.cache_resource
def load_ocr_engine():
    return easyocr.Reader(["ru", "en"], gpu=IS_GPU)


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
        from ultralytics import YOLO

        model = YOLO(os.path.abspath(self.path))
        model.to(self.device_type)
        return model

    def execute_inference(self, pil_image):
        if self.model_object is None:
            return [], None, 0.0, 0.0
        img_np = np.array(pil_image)
        parsed_results = []

        start_yolo = time.time()
        outputs = self.model_object(pil_image, imgsz=640)
        yolo_duration = time.time() - start_yolo

        annotated_image = outputs[0].plot()

        start_ocr = time.time()
        for box in outputs[0].boxes:
            conf = box.conf[0].item() * 100
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cropped = img_np[
                max(0, y1) : min(img_np.shape[0], y2),
                max(0, x1) : min(img_np.shape[1], x2),
            ]
            text_output = "Н/Д"
            ocr_conf = 0.0

            if cropped.size > 0:
                ocr_res = self.recognizer.readtext(
                    cropped,
                    decoder="beamsearch",
                    beamWidth=5,
                    workers=0,
                    adjust_contrast=0.8,
                )
                if ocr_res:
                    raw_text = " ".join([res[1] for res in ocr_res])
                    text_output = clean_mixed_text(raw_text)
                    ocr_conf = np.mean([res[2] for res in ocr_res]) * 100

            if conf < 50.0:
                auto_quality_mark = "Ошибка локализации (Низкая уверенность)"
            elif text_output == "Н/Д" or ocr_conf < 50.0:
                auto_quality_mark = "Ошибка распознавания текста (Низкая точность OCR)"
            else:
                auto_quality_mark = "Корректно"

            parsed_results.append(
                {
                    "coordinates": [x1, y1, x2, y2],
                    "confidence": conf,
                    "ocr_confidence": ocr_conf,
                    "text": text_output,
                    "evaluation": auto_quality_mark,
                }
            )
        ocr_duration = time.time() - start_ocr
        return parsed_results, annotated_image, yolo_duration, ocr_duration


engine = (
    UniversalInferenceFactory(model_path, DEVICE_TYPE, recognizer_engine)
    if model_path
    else None
)

uploaded_files = st.file_uploader(
    "Загрузка изображений для пакетного анализа:",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files and engine is not None:
    st.success(f"Успешно загружено файлов для анализа: {len(uploaded_files)}")
    session_records = []

    for uploaded_file in uploaded_files:
        st.markdown(f"## Обработка файла: `{uploaded_file.name}`")
        col1, col2 = st.columns(2)
        source_image = Image.open(uploaded_file)

        with col1:
            st.subheader("Исходное изображение")
            st.image(source_image, use_container_width=True)

        detected_blocks, visual_output, yolo_time, ocr_time = engine.execute_inference(
            source_image
        )
        total_frame_time = yolo_time + ocr_time
        total_segments = len(detected_blocks)

        frame_segments_log = []

        if total_segments > 0:
            m1, m2 = st.columns(2)
            with m1:
                st.metric(
                    label="Всего найдено текстовых областей",
                    value=total_segments,
                )
            with m2:
                max_conf = max([b["confidence"] for b in detected_blocks])
                st.metric(
                    label="Максимальная уверенность детектора",
                    value=f"{max_conf:.1f}%",
                )

            st.write("---")

            for idx, block in enumerate(detected_blocks):
                card_title = f"Сегмент №{idx + 1} | Текст: «{block['text']}»"

                with st.expander(card_title, expanded=True):
                    c_col1, c_col2 = st.columns(2)
                    with c_col1:
                        st.markdown("**Модель детекции:**")
                        st.write(f"Точность локализации: `{block['confidence']:.2f}%`")
                        st.write(
                            f"Геометрия рамки `[x1, y1, x2, y2]`: `{block['coordinates']}`"
                        )
                    with c_col2:
                        st.markdown("**Результат распознавания:**")
                        st.info(f"Распознанный текст: **{block['text']}**")
                        st.write(
                            f"Точность OCR (уверенность): `{block['ocr_confidence']:.2f}%`"
                        )
                        st.write(f"Авто-оценка качества: `{block['evaluation']}`")

                frame_segments_log.append(
                    {
                        "detection_confidence_percent": round(block["confidence"], 2),
                        "ocr_confidence_percent": round(block["ocr_confidence"], 2),
                        "bounding_box_geometry": block["coordinates"],
                        "ocr_output_text": block["text"],
                        "auto_quality_mark": block["evaluation"],
                    }
                )
            with col2:
                st.subheader("Результат локализации")
                if visual_output is not None:
                    st.image(visual_output, channels="BGR", use_container_width=True)
        else:
            with col2:
                st.subheader("Результат обработки")
                st.image(source_image, use_container_width=True)
            st.warning("Текстовые области на данном кадре не обнаружены.")

            frame_segments_log.append(
                {
                    "detection_confidence_percent": 0.0,
                    "ocr_confidence_percent": 0.0,
                    "bounding_box_geometry": [],
                    "ocr_output_text": "Н/Д",
                    "auto_quality_mark": "Объекты не найдены",
                }
            )

        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": uploaded_file.name,
            "image_resolution": f"{source_image.width}x{source_image.height}",
            "selected_architecture": selected_model_label,
            "model_file_path": model_path,
            "hardware_environment": DEVICE_LABEL,
            "yolo_inference_time_seconds": round(yolo_time, 4),
            "ocr_processing_time_seconds": round(ocr_time, 4),
            "total_frame_time_seconds": round(total_frame_time, 4),
            "total_segments_found": total_segments,
            "detected_segments": frame_segments_log,
        }
        session_records.append(log_data)

        st.text(
            f"Время анализа кадра: {total_frame_time:.4f} сек. (YOLO: {yolo_time:.4f} c. | OCR: {ocr_time:.4f} c.)"
        )
        st.markdown("---")

    if session_records:
        os.makedirs("results", exist_ok=True)
        with open("results/history.jsonl", "a", encoding="utf-8") as f:
            for record in session_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        st.sidebar.success(f"Успешно сохранено кадров: {len(session_records)}")
