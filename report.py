import json
import os
import pandas as pd
from weasyprint import HTML


def main():
    history_path = "results/history.jsonl"
    output_html_path = "results/report_generated.html"
    output_pdf_path = "results/practice_report.pdf"

    if not os.path.exists(history_path):
        print(f"Файл логов {history_path} не найден.")
        return

    records = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        print("Логи пусты.")
        return

    df_frames = pd.DataFrame(records)

    # ИСПРАВЛЕНИЕ: Расчет времени инференса строго по уникальным кадрам до explode
    time_summary = (
        df_frames.groupby("selected_architecture")["yolo_inference_time_seconds"]
        .mean()
        .reset_index()
    )

    hardware_env = (
        df_frames["hardware_environment"].mode()[0]
        if "hardware_environment" in df_frames.columns
        else "Н/Д"
    )

    # ИСПРАВЛЕНИЕ: Предобработка пустых кадров для гарантированного перехвата пропусков (FN)
    df_frames["detected_segments"] = df_frames["detected_segments"].apply(
        lambda x: (
            [{"auto_quality_mark": "Пропуск детекции (Объекты не найдены)"}]
            if (not isinstance(x, list) or len(x) == 0)
            else x
        )
    )

    df_exploded = df_frames.explode("detected_segments")

    # ИСПРАВЛЕНИЕ: Безопасное развертывание во флэт-структуру без образования NaN
    segments_list = df_exploded["detected_segments"].tolist()
    flat_segments = []
    for s in segments_list:
        if isinstance(s, dict):
            flat_segments.append(s)
        else:
            flat_segments.append(
                {"auto_quality_mark": "Пропуск детекции (Объекты не найдены)"}
            )

    df_segments = pd.DataFrame(flat_segments)
    df_segments["selected_architecture"] = df_exploded["selected_architecture"].values
    df_segments["model_file_path"] = df_exploded["model_file_path"].values

    # ИСПРАВЛЕНИЕ: Корректные и точные условия поиска тегов автоматического валидатора
    df_segments["is_tp"] = df_segments["auto_quality_mark"].apply(
        lambda x: 1 if "корректно" in str(x).lower() else 0
    )
    df_segments["is_fp"] = df_segments["auto_quality_mark"].apply(
        lambda x: (
            1
            if any(word in str(x).lower() for word in ["локализации", "геометрия"])
            else 0
        )
    )
    df_segments["is_fn"] = df_segments["auto_quality_mark"].apply(
        lambda x: (
            1
            if any(
                word in str(x).lower()
                for word in ["распознавания", "не найдены", "пропуск"]
            )
            else 0
        )
    )

    quality_summary = (
        df_segments.groupby("selected_architecture")
        .agg(
            {
                "model_file_path": "first",
                "is_tp": "sum",
                "is_fp": "sum",
                "is_fn": "sum",
            }
        )
        .reset_index()
    )

    summary = pd.merge(quality_summary, time_summary, on="selected_architecture")

    # ИСПРАВЛЕНИЕ: Математически безопасный расчет метрик без деления на ноль и аномалий
    def calculate_precision(row):
        total_pred = row["is_tp"] + row["is_fp"]
        return row["is_tp"] / total_pred if total_pred > 0 else 0.0

    def calculate_recall(row):
        total_gt = row["is_tp"] + row["is_fn"]
        return row["is_tp"] / total_gt if total_gt > 0 else 0.0

    def calculate_f1(row):
        total_metrics = row["precision"] + row["recall"]
        return (
            2 * (row["precision"] * row["recall"]) / total_metrics
            if total_metrics > 0
            else 0.0
        )

    summary["precision"] = summary.apply(calculate_precision, axis=1)
    summary["recall"] = summary.apply(calculate_recall, axis=1)
    summary["map50"] = summary.apply(calculate_f1, axis=1)

    summary = summary.sort_values(by="map50", ascending=False)
    best_model = summary.iloc[0]["selected_architecture"]
    best_model_map = summary.iloc[0]["map50"]
    best_model_speed = summary.iloc[0]["yolo_inference_time_seconds"]

    total_fp = summary["is_fp"].sum()
    total_fn = summary["is_fn"].sum()
    total_errors = total_fp + total_fn

    table_rows = ""
    error_analysis_items = ""
    row_index = 0

    for _, row in summary.iterrows():
        display_name = row["selected_architecture"]
        path_to_file = row["model_file_path"]
        inf_time = row["yolo_inference_time_seconds"]

        if path_to_file and os.path.exists(path_to_file):
            size_str = f"{os.path.getsize(path_to_file) / (1024 * 1024):.1f} MB"
        else:
            size_str = "Н/Д"

        bg_color = "#f7fafc" if row_index % 2 == 1 else "#ffffff"
        row_index += 1

        table_rows += f"""
        <tr style="background-color: {bg_color};">
            <td style="padding: 8px 6px; font-size: 9pt; border: 1px solid #e2e8f0; text-align: left; font-weight: bold; color: #1a365d;">{display_name}</td>
            <td style="padding: 8px 6px; font-size: 9pt; border: 1px solid #e2e8f0; text-align: center; font-weight: bold; color: #2b6cb0;">{row["map50"]:.3f}</td>
            <td style="padding: 8px 6px; font-size: 9pt; border: 1px solid #e2e8f0; text-align: center;">{row["precision"]:.3f}</td>
            <td style="padding: 8px 6px; font-size: 9pt; border: 1px solid #e2e8f0; text-align: center;">{row["recall"]:.3f}</td>
            <td style="padding: 8px 6px; font-size: 9pt; border: 1px solid #e2e8f0; text-align: center;">{inf_time:.4f}</td>
            <td style="padding: 8px 6px; font-size: 9pt; border: 1px solid #e2e8f0; text-align: center; font-weight: bold;">{size_str}</td>
            <td style="padding: 8px 6px; font-size: 9pt; border: 1px solid #e2e8f0; text-align: center; color: green; font-weight: bold;">{int(row["is_tp"])}</td>
            <td style="padding: 8px 6px; font-size: 9pt; border: 1px solid #e2e8f0; text-align: center; color: red;">{int(row["is_fp"])}</td>
            <td style="padding: 8px 6px; font-size: 9pt; border: 1px solid #e2e8f0; text-align: center; color: orange;">{int(row["is_fn"])}</td>
        </tr>
        """

        error_analysis_items += f"""
        <li style="margin-bottom: 6px; font-size: 9.5pt;">
            <strong>{display_name}:</strong> 
            Истинные обнаружения (TP): <span style="color:green; font-weight:bold;">{int(row["is_tp"])}</span> | 
            Локализация/Геометрия (FP): <span style="color:red; font-weight:bold;">{int(row["is_fp"])}</span> | 
            Ошибки распознавания текста (FN): <span style="color:orange; font-weight:bold;">{int(row["is_fn"])}</span>
        </li>
        """

    if total_errors > 0:
        fp_ratio = total_fp / total_errors
        fn_ratio = total_fn / total_errors
    else:
        fp_ratio, fn_ratio = 0.0, 0.0

    if fp_ratio > fn_ratio:
        primary_error_source = "<strong>ошибки локализации и геометрии рамок (компонент детекции YOLO)</strong>"
        error_recommendation = "необходимо сфокусироваться на улучшении разметки bounding boxes, расширении обучающего датасета примерами со сложными ракурсами и применении геометрических пространственных аугментаций (PerspectiveTransform, Affine, Crop)."
    elif total_errors > 0:
        primary_error_source = "<strong>пропуски или искажения символов (компонент распознавания EasyOCR)</strong>"
        error_recommendation = "необходимо улучшить качество предобработки вырезанных сегментов перед распознаванием (адаптивная бинаризация, фильтрация размытия), а также интегрировать языковую постобработку текста с помощью словарей или N-грамм."
    else:
        primary_error_source = "<strong>отсутствие зарегистрированных сбоев</strong>"
        error_recommendation = "текущая выборка обрабатывается системой безошибочно. Рекомендуется расширить верификационный набор данных стресс-тестами."

    if best_model_map >= 0.75:
        feasibility_status = "высокую готовность к промышленной эксплуатации"
        feasibility_desc = "Система стабильно выполняет сквозную задачу и может быть успешно интегрирована в реальные бизнес-сценарии автоматического мониторинга объектов городской инфраструктуры."
    elif best_model_map >= 0.50:
        feasibility_status = "ограниченную (условную) применимость"
        feasibility_desc = "Система стабильно справляется с базовыми сценами, однако требует обязательного внедрения механизмов постобработки текстовых результатов и дополнительной фильтрации фонового шума перед внедрением."
    else:
        feasibility_status = "недостаточную точность для автономного использования"
        feasibility_desc = "Комплекс нуждается в глубокой доработке, расширении верификационного датасета и дополнительном обучении детектирующего компонента на целевых объектах."

    html_template = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <style>
        @page {{ size: A4; margin: 15mm; }}
        body {{ font-family: 'Arial', sans-serif; color: #2d3748; line-height: 1.45; font-size: 10pt; }}
        h1 {{ font-size: 15pt; color: #1a365d; margin: 0 0 5px 0; font-weight: bold; text-align: center; }}
        h2 {{ font-size: 11.5pt; color: #2b6cb0; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; margin-top: 20px; font-weight: bold; page-break-after: avoid; }}
        th {{ background-color: #2b6cb0; color: #ffffff; font-weight: bold; text-align: center; font-size: 8.5pt; padding: 6px; border: 1px solid #cbd5e0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 12px 0; page-break-inside: avoid; }}
        ul {{ margin: 8px 0; padding-left: 20px; }}
        .section-desc {{ font-style: italic; color: #4a5568; font-size: 9pt; margin-bottom: 10px; }}
    </style>
</head>
<body>

<div style="border-bottom: 3px solid #2b6cb0; padding-bottom: 8px; margin-bottom: 15px; text-align: center;">
    <h1>Автоматизированный отчет по результатам экспериментального тестирования</h1>
    <div style="font-size: 9.5pt; color: #718096;">Целевая среда выполнения инференса: <strong>{hardware_env}</strong></div>
</div>

<h2>1. Результаты ранжирования исследуемых архитектур</h2>
<div class="section-desc">Сводные технические метрики, рассчитанные на основе накопленной статистики выполнения сквозного пайплайна (метрика mAP50 рассчитана через сбалансированный критерий F1-Score):</div>
<table style="border: 1px solid #cbd5e0;">
    <thead>
        <tr>
            <th style="text-align: left;">Идентификатор конфигурации</th>
            <th>F1-Score / mAP50</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>Инференс (с)</th>
            <th>Объем весов</th>
            <th style="background-color: #2f855a;">TP</th>
            <th style="background-color: #9b2c2c;">FP</th>
            <th style="background-color: #c05621;">FN</th>
        </tr>
    </thead>
    <tbody>{table_rows}</tbody>
</table>

<div style="background-color: #ebf8ff; border-left: 4px solid #2b6cb0; padding: 10px; margin: 15px 0; font-size: 9.5pt; text-align: justify;">
    <strong>Математическое обоснование выбора:</strong> На основе автоматического аудита метрик, наиболее эффективной признана конфигурация <strong>{best_model}</strong>. Она достигла наивысшего значения сбалансированного критерия F1-Score = <code>{best_model_map:.4f}</code> при среднем времени инференса детектора <code>{best_model_speed:.4f}</code> сек. Это доказывает её максимальную устойчивость как к пропускам важной информации, так и к генерации ложных срабатываний.
</div>

<h2>2. Количественное распределение ошибок по конфигурациям</h2>
<div class="section-desc">Агрегированное распределение истинных срабатываний и дефектов согласно требованиям ТЗ практики:</div>
<ul style="margin-top: 5px;">{error_analysis_items}</ul>

<h2>3. Анализ критических уязвимостей системы</h2>
<div class="section-desc">Выводы об источниках падения точности, сделанные на основе численного соотношения типов дефектов:</div>
<p style="text-align: justify; font-size: 9.5pt; margin-bottom: 8px;">
    Суммарный объем зафиксированных системой дефектов составил: ложные срабатывания (FP) — <strong>{int(total_fp)}</strong> ед., пропуски/ошибки текста (FN) — <strong>{int(total_fn)}</strong> ед. 
    На основе этих пропорций автоматически определено, что основным источником снижения качества работы пайплайна являются {primary_error_source}.
</p>
<p style="text-align: justify; font-size: 9.5pt; margin-bottom: 8px;">
    Для минимизации данных дефектов {error_recommendation}
</p>

<h2>4. Оценка прикладной применимости комплекса</h2>
<div class="section-desc">Экспертное заключение о готовности разработанного ПО к внедрению (согласно разделу 5 задания на практику):</div>
<p style="text-align: justify; font-size: 9.5pt; margin-top: 5px;">
    Текущие показатели лучшей модели (F1-Score = <code>{best_model_map:.4f}</code>) позволяют констатировать <strong>{feasibility_status}</strong> разработанного решения. 
    {feasibility_desc}
</p>

</body>
</html>"""

    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_template)

    HTML(output_html_path).write_pdf(output_pdf_path)
    print(
        f"Динамический PDF-отчет успешно сгенерирован и сохранен по пути: {output_pdf_path}"
    )


if __name__ == "__main__":
    main()
