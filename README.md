# cnn-feature-engineering-predictive-modellingz 
# Chest X-Ray Image Analysis for Tuberculosis Detection  
### CNN-Based Feature Engineering & Predictive Modeling Pipeline

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()  
[![License](https://img.shields.io/badge/license-MIT-green)]()  
[![Notebook](https://img.shields.io/badge/Jupyter_notebook-orange)]()

---

## 🚀 Project Overview

Tuberculosis (TB) remains one of the world’s deadliest infectious diseases. Early detection from chest X-rays can save lives. This project delivers a **modular**, **scalable**, **explainable** pipeline that:

- **Automatically** detects input type (images vs. tabular)  
- Applies standard and custom **preprocessing** (resizing, normalization, augmentation)  
- Extracts deep **CNN features** for classification  
- Trains, tunes, and evaluates models (ResNet-based, custom CNNs)  
- Generates **explainability** via SHAP summaries  
- Produces end-to-end **HTML reports** and visualizations  

---

## 🗂️ Repository Structure

```text
├── data/                 # Raw & processed images
├── notebooks/            # Your analysis & experiments
├── src/                  # Python modules
├── models/               # Saved model checkpoints
├── reports/              # Auto-generated analyses
├── img/                  # Figures for README
├── api/                  # Simple inference service
├── environment.yml       # Conda environment
├── requirements.txt      # pip install -r
├── LICENSE               # MIT
└── README.md
1. Jupyter Notebook
bash
Copy
Edit
jupyter notebook notebooks/ChestXRay_TB_CNN_Feature_Engineering.ipynb
Follow each section:

Data Loading

Preprocessing & Augmentation

CNN Feature Extraction

Model Training & Tuning

Explainability (SHAP)

Report Generation

2. CLI Training
bash
Copy
Edit
python src/model.py \
  --data_dir data/processed \
  --epochs 25 \
  --batch_size 32 \
  --lr 1e-4 \
  --save_path models/best_model.pth
3. API Inference
bash
Copy
Edit
uvicorn api.inference_api:app --reload
# then POST JSON or send via curl:
curl -X POST http://127.0.0.1:8000/predict \
  -F "image=@./sample_images/pos_1.png"
📊 Sample Outputs
Training Curve	Confusion Matrix	SHAP Summary Plot

📈 Model Performance
Metric	Score
Accuracy	94.8%
Precision	95.2%
Recall	93.7%
F1‐Score	94.4%
ROC AUC	0.98

🔍 Comparison to SRIE (Smart Real Estate Insights Engine)
Aspect	Your CNN Project	SRIE (Friend’s Project)
Domain	Medical imaging (TB detection)	Real estate rent prediction
Data Type	Chest X-ray images	Tabular (CSV)
Pipeline Modules	CNN feature extraction, SHAP explain	EDA, regression, anomaly detection, API
Visualization	Training curves, confusion matrix, SHAP	Price distributions, feature importance, residuals
Serving	FastAPI inference of X-ray scans	Flask/requests API for rent prediction
Repo Structure	notebooks, src, models, reports, img	notebooks, src_notebooks, img, API script

🛠️ Future Improvements
Hyperparameter tuning with Optuna

Add cross-validation & learning curve plots

Dockerize service & deploy to cloud (AWS/GCP)

Extend explainability (Grad-CAM, integrated gradients)
