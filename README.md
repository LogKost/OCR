# OCR and Object Detection Project

This project focuses on training and evaluating various YOLO models (v5, v8, v9, v11) and RT-DETR for object detection tasks, with an interactive Streamlit web application for inference.

---

## Installation & Setup

Follow these steps to set up the environment and run the project locally.

### 1. Install PyTorch with CUDA (Optional)
If you have a compatible NVIDIA GPU and want to use CUDA for faster training, install PyTorch by following the official instructions:
* Go to [pytorch.org](https://pytorch.org) to get the exact command for your CUDA version.

### 2. Install Dependencies
Install all the required Python libraries using `pip`:
```bash
pip install -r requirements.txt
```

Additionally, this project requires **GTK 3** to be installed on your system (e.g., for reporting or UI components). 
* **Windows Users:** Download and run the [GTK 3 Runtime Environment Installer (v2022-01-04)](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases/tag/2022-01-04). Follow the setup wizard to complete the installation.

---

## Training Models

You can train each model separately. Run the corresponding script for the model you want to train:

```bash
# Train YOLOv5
python train_v5.py

# Train YOLOv8
python train_v8.py

# Train YOLOv9
python train_v9.py

# Train YOLOv11
python train_v11.py

# Train RT-DETR
python train_rtdetr.py
```

---

## Running the Web App & Testing

Before evaluating the models, you need to launch the interactive Streamlit application and upload your test dataset for each model:

```bash
streamlit run app.py
```

1. Open the application in your browser.
2. Upload the test datasets into the UI to run inference for each trained model.
3. The app will automatically save the execution data to `history.json`.

---

## Evaluation & Metrics Generation

Once you have gathered the test results through the Streamlit app, run the evaluation script to process `history.json` and generate the final reports:

```bash
python evaluate.py
```

---

## Project Outputs & Results

After running `evaluate.py`, you can find the generated performance summaries in the `results/` directory:

* **`history.json`** — Contains the raw training and evaluation logs gathered during the process.
* **`practice_report.pdf`** — An automatically generated PDF report summarizing the performance metrics.
* **`report_generated.html`** — An interactive HTML version of the generated evaluation report.
