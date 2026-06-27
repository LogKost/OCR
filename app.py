import os
import time
from datetime import datetime
import numpy as np
import streamlit as st
import torch
import easyocr
from PIL import Image

from services.inference import UniversalInferenceFactory
from utils.text_processing import clean_mixed_text
from utils.evaluator import evaluate_quality
from utils.logger import save_single_log

st.set_page_config(layout="wide")

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

engine = UniversalInferenceFactory(model_path, DEVICE_TYPE) if model_path else None

uploaded_files = st.file_uploader(
    "Загрузка изображений для пакетного анализа:",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files and engine is not None:
    st.success(f"Успешно загружено файлов для анализа: {len(uploaded_files)}")
    saved_counter = 0

    for uploaded_file in uploaded_files:
        st.markdown(f"## Обработка файла: `{uploaded_file.name}`")
        col1, col2 = st.columns(2)
        source_image = Image.open(uploaded_file)
        img_np = np.array(source_image)

        with col1:
            st.subheader("Исходное изображение")
            st.image(source_image, width="stretch")

        detected_blocks, visual_output, yolo_time = engine.execute_detection(
            source_image
        )

        start_ocr = time.time()
        frame_segments_log = []
        total_segments = len(detected_blocks)

        if total_segments > 0:
            m1, m2 = st.columns(2)
            with m1:
                st.metric(
                    label="Всего найдено текстовых областей", value=total_segments
                )
            with m2:
                max_conf = max([b["confidence"] for b in detected_blocks], default=0.0)
                st.metric(
                    label="Максимальная уверенность детектора", value=f"{max_conf:.1f}%"
                )

            st.write("---")

            for idx, block in enumerate(detected_blocks):
                x1, y1, x2, y2 = block["coordinates"]

                cropped = img_np[
                    max(0, y1) : min(img_np.shape[0], y2),
                    max(0, x1) : min(img_np.shape[1], x2),
                ]

                text_output = "Н/Д"
                ocr_conf = 0.0

                if cropped.size > 0:
                    ocr_res = recognizer_engine.readtext(
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

                auto_quality_mark = evaluate_quality(
                    block["confidence"], ocr_conf, text_output, [x1, y1, x2, y2]
                )

                card_title = f"Сегмент №{idx + 1} | Текст: «{text_output}»"
                with st.expander(card_title, expanded=True):
                    c_col1, c_col2 = st.columns(2)
                    with c_col1:
                        st.markdown("**Модель детекции:**")
                        st.write(f"Точность локализации: `{block['confidence']:.2f}%`")
                        st.write(
                            f"Геометрия рамки `[x1, y1, x2, y2]`: `{[x1, y1, x2, y2]}`"
                        )
                    with c_col2:
                        st.markdown("**Результат распознавания:**")
                        st.info(f"Распознанный текст: **{text_output}**")
                        st.write(f"Точность OCR (уверенность): `{ocr_conf:.2f}%`")
                        st.write(f"Авто-оценка качества: `{auto_quality_mark}`")

                frame_segments_log.append(
                    {
                        "detection_confidence_percent": round(block["confidence"], 2),
                        "ocr_confidence_percent": round(ocr_conf, 2),
                        "bounding_box_geometry": [x1, y1, x2, y2],
                        "ocr_output_text": text_output,
                        "auto_quality_mark": auto_quality_mark,
                    }
                )

            with col2:
                st.subheader("Результат локализации")
                if visual_output is not None:
                    st.image(visual_output, width="stretch")
        else:
            with col2:
                st.subheader("Результат обработки")
                st.image(source_image, width="stretch")
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

        ocr_time = time.time() - start_ocr
        total_frame_time = yolo_time + ocr_time

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

        save_single_log(log_data)
        saved_counter += 1

        st.text(
            f"Время анализа кадра: {total_frame_time:.4f} сек. (Модель: {yolo_time:.4f} c. | OCR: {ocr_time:.4f} c.)"
        )
        st.markdown("---")

        if IS_GPU:
            torch.cuda.empty_cache()

    st.sidebar.success(f"Успешно обработано и сохранено кадров: {saved_counter}")
