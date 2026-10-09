# Brain Tumor Detection & Personalized Clinical Guidance System

This project is an advanced AI-powered medical web application that classifies brain MRI scans (Glioma, Meningioma, Pituitary, and No Tumor) using Deep Learning (MobileNetV2 Transfer Learning), analyzes tumor severity using OpenCV, and provides personalized medical, dietary, and lifestyle guidance based on the patient's age and gender.

## 📊 Dataset Overview
The dataset is balanced and structured into Training and Testing directories:
- **Total Dataset Size:** 7,200 MRI Images across 4 classes
- **Training Set:** 5,600 images (1,400 per class)
- **Testing Set:** 1,600 images (400 per class)

| Class Name | Training Images | Testing Images | Total Images |
| :--- | :---: | :---: | :---: |
| **Glioma** | 1,400 | 400 | 1,800 |
| **Meningioma** | 1,400 | 400 | 1,800 |
| **No Tumor** | 1,400 | 400 | 1,800 |
| **Pituitary** | 1,400 | 400 | 1,800 |
| **Total** | **5,600** | **1,600** | **7,200** |

## 🚀 Model Performance & Results
- **Architecture:** MobileNetV2 (Transfer Learning)
- **Test Accuracy:** **90.13%** (Loss: 0.3896)
- **Peak Validation Accuracy:** ~93.69%

### Per-Class Classification Metrics:
| Class Name | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Glioma** | 0.934 | 0.780 | 0.850 |
| **Meningioma** | 0.846 | 0.863 | 0.854 |
| **No Tumor** | 0.879 | 0.995 | 0.933 |
| **Pituitary** | 0.956 | 0.968 | 0.961 |

## ✨ Key Features
- **Tumor Classification:** Accurately detects and classifies brain tumors from MRI scans.
- **Severity Analysis:** Computes tumor region percentage using OpenCV thresholding and categorizes severity (Low, Moderate, High).
- **Personalized Recommendations:** Generates tailored doctor advice, dietary tips, and lifestyle guidance customized for different age groups (Child, Teen, Adult, Middle-age, Senior) and gender.
- **Interactive Web App:** Built using Streamlit for a smooth graphical interface.

## 📂 Project Structure
