
🧠 Chest X-Ray Tuberculosis Detection Using CNN Feature Engineering & Predictive Modeling

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)]()

---

## 📌 Project Overview

This project presents a modular and adaptive machine learning pipeline designed to detect **tuberculosis (TB)** from chest X-ray images. It combines **deep learning-based feature extraction** using **ResNet50** with **classical machine learning classifiers** (SVM, Random Forest, Logistic Regression) to build a flexible, explainable, and high-performance detection system.

Key features include:

* Automatic input type detection (images or tabular data)
* Image preprocessing and augmentation (resizing, normalization, rotation, flipping)
* Feature extraction using CNN embeddings without full CNN training
* Dimensionality reduction with PCA and feature selection using Random Forest importances
* AutoML-driven model selection evaluating multiple classifiers
* Model evaluation via accuracy, precision, recall, F1-score, confusion matrix, ROC/AUC
* Explainability via SHAP global and local feature impact visualization
* End-to-end reporting with HTML visualizations
* FastAPI-based inference service for easy deployment

---

## 📁 Dataset

* **Name**: TB Chest Radiography Database
* **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
* **Classes**:

  * TB Positive (`TB`): 700 images
  * TB Negative (`Normal`): 3500 images

---

## 🖼️ Sample Results (Balanced Dataset)

| Metric    | TB Negative | TB Positive | Macro Average |
| --------- | ----------- | ----------- | ------------- |
| Precision | 0.97        | 0.98        | 0.98          |
| Recall    | 0.98        | 0.97        | 0.97          |
| F1-Score  | 0.98        | 0.97        | 0.97          |
| Accuracy  |             |             | 0.97          |

* **AUC**: 0.98
* **Best Model**: SVM with RBF kernel, C=10

---

## ⚙️ Installation & Usage

1. **Clone the repo**

   ```bash
   git clone https://github.com/your-username/sravansai-10.git
   cd sravansai-10
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter notebook**

   ```bash
   jupyter notebook cnn-feature-engineering-and-predictive-modeling.ipynb
   ```

   Follow notebook sections in order: data loading → preprocessing → feature extraction → training → explainability → report generation

4. **Train via command line interface (CLI)**

   ```bash
   python src/model.py \
     --data_dir data/processed \
     --epochs 25 \
     --batch_size 32 \
     --lr 1e-4 \
     --save_path models/best_model.pth
   ```

5. **Run inference API**

   ```bash
   uvicorn src.inference_api:app --reload
   ```

   Test with:

   ```bash
   curl -X POST http://127.0.0.1:8000/predict \
     -F "image=@./img/TB Image.png"
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

## 🔧 Future Improvements

* Hyperparameter tuning with Optuna
* Cross-validation and learning curve plots
* Docker containerization for easy deployment
* Advanced explainability methods such as Grad-CAM and integrated gradients

---

## 📜 License

This project is licensed under the MIT License. Please cite the dataset source when using this work.

---

## 🙋 Questions or Feedback?

Feel free to open issues or contact any authors for help with using the pipeline or dataset.

