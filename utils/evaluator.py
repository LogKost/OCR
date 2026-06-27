def evaluate_quality(det_conf, ocr_conf, text_output, box_coordinates):
    MIN_YOLO_CONF = 35.0
    STRONG_YOLO_CONF = 70.0
    MIN_OCR_CONF = 50.0
    STRONG_OCR_CONF = 75.0

    if text_output == "Н/Д":
        return "Ошибка распознавания текста (Данные не найдены)"

    x1, y1, x2, y2 = box_coordinates
    width = x2 - x1
    height = y2 - y1
    aspect_ratio = width / max(1, height)

    if aspect_ratio < 0.2:
        return "Подозрительная геометрия (Возможно ложное срабатывание)"

    if det_conf < MIN_YOLO_CONF and ocr_conf >= STRONG_OCR_CONF:
        return "Корректно (Восстановлено по OCR)"

    if det_conf < MIN_YOLO_CONF:
        return "Ошибка локализации (Низкая уверенность детектора)"

    if det_conf >= STRONG_YOLO_CONF and ocr_conf < MIN_OCR_CONF:
        if len(text_output) > 3:
            return "Корректно (Низкий приоритет OCR)"
        return "Ошибка распознавания текста (Низкая точность OCR)"

    if ocr_conf < MIN_OCR_CONF:
        return "Ошибка распознавания текста (Низкая точность OCR)"

    return "Корректно"
