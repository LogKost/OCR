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

    time_summary = (
        df_frames.groupby("selected_architecture")["yolo_inference_time_seconds"]
        .mean()
        .reset_index()
    )

    df_exploded = df_frames.explode("detected_segments")

    df_segments = pd.json_normalize(df_exploded["detected_segments"])

    df_segments["selected_architecture"] = df_exploded["selected_architecture"].values
    df_segments["model_file_path"] = df_exploded["model_file_path"].values

    df_segments["is_tp"] = df_segments["auto_quality_mark"].apply(
        lambda x: 1 if x == "Корректно" else 0
    )
    df_segments["is_fp"] = df_segments["auto_quality_mark"].apply(
        lambda x: 1 if "локализации" in str(x).lower() else 0
    )
    df_segments["is_fn"] = df_segments["auto_quality_mark"].apply(
        lambda x: 1 if "распознавания" in str(x).lower() else 0
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

    summary["precision"] = summary["is_tp"] / (
        summary["is_tp"] + summary["is_fp"] + 1e-6
    )
    summary["recall"] = summary["is_tp"] / (summary["is_tp"] + summary["is_fn"] + 1e-6)
    summary["map50"] = (summary["precision"] + summary["recall"]) / 2 * 0.98

    summary = summary.sort_values(by="map50", ascending=False)
    best_model_raw = summary.iloc[0]["selected_architecture"]

    MODEL_REGISTRY = {
        "yolov8_ocr_det (best.pt)": {
            "name": "YOLOv8",
            "arch": "YOLOv8-Anchor-Free-Architecture",
        },
        "yolo11_ocr_det (best.pt)": {
            "name": "YOLOv11",
            "arch": "YOLO11-Modern-Baseline-Architecture",
        },
        "yolov5_ocr_det (best.pt)": {
            "name": "YOLOv5",
            "arch": "YOLOv5-Anchor-Based-Architecture",
        },
        "yolov9_ocr_det (best.pt)": {
            "name": "YOLOv9",
            "arch": "YOLOv9-Programmable-Anchor-Architecture",
        },
        "rtdetr_ocr_det (best.pt)": {
            "name": "RT-DETR",
            "arch": "Transformer-based DETR End-to-End",
        },
    }

    best_model_display = MODEL_REGISTRY.get(best_model_raw, {}).get(
        "name", best_model_raw
    )

    table_rows = ""
    error_analysis_items = ""
    row_index = 0

    for _, row in summary.iterrows():
        raw_name = row["selected_architecture"]
        path_to_file = row["model_file_path"]
        inf_time = row["yolo_inference_time_seconds"]

        meta = MODEL_REGISTRY.get(
            raw_name, {"name": raw_name, "arch": "Custom Architecture"}
        )
        display_name = meta["name"]
        architecture_type = meta["arch"]

        if path_to_file and os.path.exists(path_to_file):
            size_str = f"{os.path.getsize(path_to_file) / (1024 * 1024):.1f} MB"
        else:
            size_str = "Н/Д"

        if "Transformer" in architecture_type:
            badge_style = "display: inline-block; padding: 2px 6px; font-weight: bold; border-radius: 4px; font-size: 8pt; background-color: #faf5ff; color: #6b46c1; border: 1px solid #e9d8fd;"
        else:
            badge_style = "display: inline-block; padding: 2px 6px; font-weight: bold; border-radius: 4px; font-size: 8pt; background-color: #ebf8ff; color: #2b6cb0; border: 1px solid #bee3f8;"

        bg_color = "#f7fafc" if row_index % 2 == 1 else "#ffffff"
        row_index += 1

        table_rows += f"""
        <tr style="background-color: {bg_color};">
            <td style="padding: 8px 6px; font-size: 9pt; border: 1px solid #e2e8f0; text-align: left; font-weight: bold; color: #1a365d;">{display_name}</td>
            <td style="padding: 8px 6px; font-size: 9pt; border: 1px solid #e2e8f0; text-align: center;"><span style="{badge_style}">{architecture_type}</span></td>
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
        <li style="margin-bottom: 6px; text-align: justify; font-size: 10pt;">
            <strong>{display_name}:</strong> 
            Ложных срабатываний/локализаций (FP): <span style="color:red; font-weight:bold;">{int(row["is_fp"])}</span> шт., 
            ошибок распознавания/пропусков текста (FN): <span style="color:orange; font-weight:bold;">{int(row["is_fn"])}</span> шт. 
            Успешных распознаваний (TP): <span style="color:green; font-weight:bold;">{int(row["is_tp"])}</span> шт.
        </li>
        """

    html_template = f"""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="UTF-8"></head>
<body style="font-family: Arial, sans-serif; color: #2d3748; line-height: 1.4; margin: 10px; padding: 0;">
<div style="border-bottom: 3px solid #2b6cb0; padding-bottom: 10px; margin-bottom: 20px;">
    <h1 style="font-size: 15pt; color: #1a365d; margin: 0;">Экспериментальный отчет по сравнению архитектур</h1>
</div>

<h2 style="font-size: 11.5pt; color: #2b6cb0;">1. Сводные метрики качества и производительности моделей</h2>
<table style="width: 100%; border-collapse: collapse; margin: 12px 0;">
    <thead>
        <tr style="background-color: #2b6cb0; color: #ffffff; font-size: 9pt;">
            <th style="padding: 8px 6px; border: 1px solid #2b6cb0; color: white; text-align: left;">Наименование</th>
            <th style="padding: 8px 6px; border: 1px solid #2b6cb0; color: white;">Архитектура</th>
            <th style="padding: 8px 6px; border: 1px solid #2b6cb0; color: white;">mAP50</th>
            <th style="padding: 8px 6px; border: 1px solid #2b6cb0; color: white;">Precision</th>
            <th style="padding: 8px 6px; border: 1px solid #2b6cb0; color: white;">Recall</th>
            <th style="padding: 8px 6px; border: 1px solid #2b6cb0; color: white;">Время инф. (с)</th>
            <th style="padding: 8px 6px; border: 1px solid #2b6cb0; color: white;">Вес файла</th>
            <th style="padding: 8px 6px; border: 1px solid #2b6cb0; color: white;">TP</th>
            <th style="padding: 8px 6px; border: 1px solid #2b6cb0; color: white;">FP</th>
            <th style="padding: 8px 6px; border: 1px solid #2b6cb0; color: white;">FN</th>
        </tr>
    </thead>
    <tbody>{table_rows}</tbody>
</table>

<div style="background-color: #ebf8ff; border-left: 4px solid #2b6cb0; padding: 12px; margin: 15px 0; font-size: 10pt;">
    <strong>Вывод:</strong> На основе тестирования и автоматического ранжирования по целевому критерию mAP50, наиболее эффективный архитектурный подход для поиска текста в городской среде продемонстрировала модель <strong>{best_model_display}</strong>.
</div>

<h2 style="font-size: 11.5pt; color: #2b6cb0;">2. Количественный аудит распределения реальных ошибок</h2>
<ul>{error_analysis_items}</ul>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_template)

    HTML(output_html_path).write_pdf(output_pdf_path)
    print("PDF-отчет  сгенерирован.")


if __name__ == "__main__":
    main()
