# smart-waste-segregation-system-
Real-time AI-based waste classification using CNN and OpenCV
# Smart Waste Segregation System

Real-time AI-based waste classification using CNN and OpenCV

## About the Project
This project uses a Convolutional Neural Network (CNN) to classify 
waste into 6 categories in real-time using a laptop webcam.

## Waste Categories
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy

## Dataset
Download the dataset from Kaggle:
[Garbage Classification Dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)

After downloading, extract and rename the folder to `dataset` 
and place it in the project root.

## How to Run

### Install Dependencies
pip install -r requirements.txt

### Train the Model (Optional - model already provided)
python train_model.py

### Run Real-Time Detection
python detect_waste.py

## Project Structure
smart-waste-segregation-system/
├── dataset/          ← Download from Kaggle (not included)
├── waste_model.h5    ← Pre-trained model
├── train_model.py    ← CNN training script
├── detect_waste.py   ← Real-time detection script
└── requirements.txt  ← Dependencies

## Results
- 6 waste categories classified in real-time
- Confidence score displayed on webcam feed
- Validation Accuracy: ~92%

## Developed By
Sabari — B.E. ECE, Mepco Schlenk Engineering College, Sivakasi
GYAN-MITRA Hackathon 2026
