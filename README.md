brain-tumor-mri-detection-ai
│
├── app.py                     # Streamlit web application
├── predict.py                 # Model prediction logic
├── train_model.py             # Script to train CNN model
├── requirements.txt           # Python dependencies
├── README.md
│
├── models
│   └── class_indices.json     # Label mapping file
│
├── utils
│   ├── preprocessing.py       # Image preprocessing functions
│   ├── gradcam.py             # Grad-CAM heatmap generation
│   └── tumor_info.py          # Tumor description data
│
├── images
│   └── app_demo.png           # Screenshots for README
│
└── dataset
    ├── Training
    └── Testing

    Brain Tumor MRI Detection using Deep Learning
Overview

This project is an AI-powered medical imaging system that detects brain tumors from MRI scans using a Convolutional Neural Network (CNN).

The application allows users to upload MRI images through a Streamlit web interface. The trained deep learning model predicts the tumor type and visualizes important regions of the image using Grad-CAM heatmaps.

This project demonstrates practical applications of deep learning, computer vision, and explainable AI in medical image analysis.

Features

MRI image classification using CNN

Detects four classes:

Glioma

Meningioma

Pituitary tumor

No tumor

Grad-CAM heatmap visualization

Interactive Streamlit web application

Modular project structure

Training pipeline included

Technologies Used

Python
TensorFlow / Keras
OpenCV
NumPy
Streamlit
Matplotlib
Grad-CAM (Explainable AI)

Dataset

The dataset contains MRI images categorized into four classes.

Due to GitHub file size limitations, the dataset is not included in this repository.

Download dataset here:

https://drive.google.com/file/d/1YeqPKkKmFSWG2VGNe0UdJ2payUJbypZU/view?usp=drive_link

After downloading, extract the ZIP file and place it inside the project folder like this:

dataset/
   Training/
   Testing/
Pretrained Model

The trained model file is excluded from this repository due to GitHub file size limits.

You can train the model yourself using:

python train_model.py

This will generate:

models/brain_tumor_model.h5
Installation

Clone the repository

git clone https://github.com/tadakavikas/brain-tumor-mri-detection-ai.git
cd brain-tumor-mri-detection-ai

Create virtual environment

python3 -m venv .venv
source .venv/bin/activate

Install dependencies

pip install -r requirements.txt
Run the Application

Start the Streamlit app:

streamlit run app.py

Open browser:

http://localhost:8501

Upload an MRI scan to get tumor prediction.

Application Demo

Add your screenshots here.

Example:

![App Interface](/Users/tadakavikas/Desktop/Screenshot 2026-03-16 at 6.35.46 AM.png)
![Prediction Result](/Users/tadakavikas/Desktop/Screenshot 2026-03-16 at 6.36.03 AM.png)
![GradCAM Heatmap](/Users/tadakavikas/Desktop/Screenshot 2026-03-16 at 6.36.22 AM.png)
Model Workflow

Load MRI image

Preprocess image

CNN model predicts tumor class

Grad-CAM generates attention heatmap

Results displayed in Streamlit app

Future Improvements

Use transfer learning (ResNet / EfficientNet)

Improve dataset size

Deploy model on cloud API

Add tumor segmentation

Add patient report generation

Author
Vikas Thadaka

GitHub
https://github.com/tadakavikas

LinkedIn
https://www.linkedin.com/in/tadakavikas

Disclaimer
This project is intended for educational purposes only and should not be used for medical diagnosis.