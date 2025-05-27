Chest X-Ray Tuberculosis Detection

This project shows how to use a Convolutional Neural Network (CNN) to find signs of tuberculosis (TB) in chest X-ray images. It’s designed to be easy to follow and extend:

Detects whether you’re using images or tables as input

Prepares and augments images (resize, normalize, flip, rotate)

Extracts features from images using ResNet or custom CNNs

Trains and tests the model, then reports accuracy

Explains model decisions with SHAP

Creates simple HTML reports and charts

Includes a FastAPI-based inference service

Folder Layout

data/           Original and processed images
notebooks/      Experiment notebooks
src/            Main Python code
models/         Saved model files
reports/        Generated reports
img/            Images used in the README
api/            Inference service code (FastAPI)
environment.yml Conda setup file
requirements.txt Python dependencies
LICENSE         Project license (MIT)
README.md       This file

How to Get Started

Clone the repo

git clone <your-repo-url>
cd cnn-feature-engineering-predictive-modeling

Install dependencies

conda env create -f environment.yml
conda activate tb_detection
# or
pip install -r requirements.txt

Run the notebook

jupyter notebook notebooks/ChestXRay_TB_CNN_Feature_Engineering.ipynb

Work through: data loading → preprocessing → feature extraction → training → explainability → reporting

Train from the command line

python src/model.py \
  --data_dir data/processed \
  --epochs 25 \
  --batch_size 32 \
  --lr 1e-4 \
  --save_path models/best_model.pth

Start the API

uvicorn api.inference_api:app --reload

Test with:

curl -X POST http://127.0.0.1:8000/predict \
  -F "image=@./sample_images/pos_1.png"

Results Example

Metric

Score

Accuracy

94.8%

Precision

95.2%

Recall

93.7%

F1-Score

94.4%

ROC AUC

0.98

You’ll also find:

Training and validation loss plots

Confusion matrix

SHAP feature importance plot

Comparison with a Friend’s Project

This project: TB detection using chest X-rays and CNNs

Friend’s project: Real estate price prediction on tabular data

Both have clear folder layouts and API services, but this one focuses on images and explainability.

What’s Next

Add hyperparameter tuning with Optuna

Include cross-validation and learning curve charts

Dockerize the API for easy deployment

Try other explainability methods like Grad-CAM

© 2025 Your Name • MIT License

