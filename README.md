# Chest X-Ray Tuberculosis Detection

A Convolutional Neural Network (CNN) pipeline for identifying tuberculosis (TB) in chest X-ray images.

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()  [![License: MIT](https://img.shields.io/badge/license-MIT-green)]()

---

## 🚀 Project Overview

Early TB detection via chest X-ray analysis can improve patient outcomes. This repository implements a clear, extensible workflow to:

1. Preprocess and augment X-ray images (resize, normalize, rotate, flip)
2. Extract deep features using a ResNet backbone or custom CNN
3. Train, validate, and evaluate classification models
4. Visualize performance (accuracy, confusion matrix)
5. Explain model predictions with SHAP
6. Generate an HTML report with key charts
7. Serve predictions through a FastAPI endpoint

---

## 📂 Repository Structure

```
sravansai-10/                              # Repository root
├── cnn-feature-engineering-and-predictive-modeling.ipynb  # Main notebook
├── data/                                  # Raw and processed data (not tracked)
├── img/                                   # Remote image files (optional)
├── Confusion Matrix.png                   # Confusion matrix example
├── Normal Image.png                       # Sample normal X-ray
├── Normal Vs Tuberclosis.png              # Comparison image
├── TB Image.png                           # Sample TB-positive X-ray
├── src/                                   # Python scripts and modules
├── models/                                # Saved model checkpoints
├── reports/                               # Generated HTML and figures
├── requirements.txt                       # Python dependencies
├── LICENSE                                # MIT License
└── README.md                              # Project documentation (this file)
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/sravansai-10.git
   cd sravansai-10
   ```
2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the notebook**

   ```bash
   jupyter notebook cnn-feature-engineering-and-predictive-modeling.ipynb
   ```

---

## ▶️ Usage

### Training via Notebook

1. Open and run each section in the Jupyter notebook:

   * Data loading and preprocessing
   * Feature extraction
   * Model training and evaluation
   * SHAP explainability
   * Report generation

### Training via CLI

```bash
python src/model.py \
  --data_dir data/processed \
  --epochs 25 \
  --batch_size 32 \
  --lr 1e-4 \
  --save_path models/best_model.pth
```

### Inference API

1. Start the FastAPI server:

   ```bash
   uvicorn src.inference_api:app --reload
   ```
2. Send an image to the `/predict` endpoint:

   ```bash
   curl -X POST http://127.0.0.1:8000/predict \
     -F "image=@./img/TB Image.png"
   ```

---

## 📊 Sample Results

![Confusion Matrix](./Confusion%20Matrix.png)

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 94.8% |
| Precision | 95.2% |
| Recall    | 93.7% |
| F1-Score  | 94.4% |
| ROC AUC   |  0.98 |

---

## 🔧 Future Improvements

* Hyperparameter tuning with Optuna
* Cross-validation and learning curve plots
* Containerize service with Docker
* Explore Grad-CAM for deeper explainability

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
