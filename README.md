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

## Evaluation & Metrics

Once the training is complete, collect and analyze the final metrics for all models by running:

```bash
python evaluate.py
```

---

## Running the Web App

To launch the interactive Streamlit application for testing and visualization, run:

```bash
streamlit run app.py
```
