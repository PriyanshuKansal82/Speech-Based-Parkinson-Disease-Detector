# 🧠 Speech-Based Parkinson’s Disease Detection using Deep Learning

An end-to-end Deep learning pipeline that detects Parkinson’s Disease from speech signals using audio feature extraction and a neural network.

---

## 🚀 Overview

This project uses speech signal processing and deep learning to detect Parkinson’s disease from voice recordings. It extracts meaningful features from raw audio and classifies them using a trained neural network.

---

## 📊 Dataset

- Total samples: 1700 audio files
- Classes:
  - Healthy: 850
  - Parkinson’s Disease (PD): 850
- Balanced dataset across multiple vowel sounds

---

## ⚙️ Pipeline

1. Data Loading  
   - Organized dataset by vowels and conditions (Healthy / PD)

2. Preprocessing  
   - Resampling to 16 kHz  
   - Normalization  
   - Fixed audio duration (3 seconds)

3. Feature Extraction (26 Features)  
   - Statistical features (mean, variance, skewness, kurtosis, RMS, etc.)  
   - Wavelet features + entropy  
   - MFCC features (mean & std)  
   - Spectral features (centroid, ZCR)  
   - Signal entropy  

4. Data Splitting  
   - Train: 2380  
   - Validation: 340  
   - Test: 680  

5. Model  
   - Fully connected neural network (PyTorch)  
   - Architecture: 128 → 64 → 2  
   - Dropout regularization  

6. Evaluation  
   - Accuracy, Precision, Recall, F1-score
     
---

## 🛠️ Tech Stack

- Python  
- PyTorch  
- Librosa  
- NumPy, SciPy  
- Scikit-learn  
- PyWavelets  
- Matplotlib, Seaborn  


