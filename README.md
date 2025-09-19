# Deepfake-Detection-Model
A deepfake detection model using TensorFlow/Keras with a modular training pipeline.

🔍 Overview

This project detects deepfakes using a Convolutional Neural Network (CNN). It includes:

Automated dataset download (via Kaggle API)

Preprocessing and image loading

CNN model with batch normalization & dropout

Training with callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)

Evaluation using accuracy, precision, and recall

Training history visualization

🚀 Getting Started

Clone the repo
git clone https://github.com/yourusername/deepfake-detector.git
cd deepfake-detector

Install dependencies

pip install -r requirements.txt

Set up Kaggle API

Place your kaggle.json in the project directory or configure it globally.

Run training

python train.py

🧠 Model

Input: 128x128 RGB images

Architecture: 4 convolutional layers → dense layers → sigmoid output

Optimizer: Adam

Loss: Binary Crossentropy

📊 Metrics

Accuracy

Precision

Recall

📁 Structure

dataset_handler.py – Downloads and loads dataset

model.py – Builds and compiles the CNN

train.py – Orchestrates training and evaluation

📌 Notes

Make sure you have access to the Kaggle dataset before running.

The best model is saved automatically during training.

📝 License

MIT License
