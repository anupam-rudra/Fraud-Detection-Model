# Credit Card Fraud Detection Using Machine Learning, Deep Learning & Ensemble Learning

A complete end-to-end **Credit Card Fraud Detection System** built using advanced Machine Learning, Deep Learning, Anomaly Detection, and Ensemble Learning techniques to identify fraudulent financial transactions from highly imbalanced datasets.

This project combines multiple supervised and unsupervised models including:

- Logistic Regression
- Random Forest
- XGBoost
- Feed Forward Neural Network (FFNN)
- Autoencoder Anomaly Detection
- Dynamic Weighted Ensemble
- Stacking Ensemble Learning

along with advanced fraud detection techniques such as:

- SMOTE Oversampling
- Dynamic Threshold Optimization
- Precision-Recall Analysis
- ROC-AUC Evaluation
- PR-AUC Based Ensemble Weighting
- Reconstruction Error Detection
- Meta-Learning for Stacking Ensemble

---

# Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Problem Statement](#problem-statement)
- [Project Workflow](#project-workflow)
- [Technologies Used](#technologies-used)
- [Machine Learning Models](#machine-learning-models)
- [Deep Learning Models](#deep-learning-models)
- [Ensemble Learning](#ensemble-learning)
- [Stacking Ensemble Learning](#stacking-ensemble-learning)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualizations](#visualizations)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Key Features](#key-features)
- [Future Improvements](#future-improvements)
- [Conclusion](#conclusion)
- [Author](#author)
- [License](#license)

---

# Project Overview

Credit card fraud detection is one of the most important real-world applications of Machine Learning in the financial industry.

Fraudulent transactions represent only a tiny fraction of all transactions, making fraud detection a highly imbalanced classification problem.

The main objective of this project is to build an intelligent fraud detection system capable of:

- Detecting fraudulent transactions accurately
- Minimizing false positives
- Handling severe class imbalance
- Improving recall without sacrificing precision
- Combining multiple models for better robustness
- Detecting anomalies using reconstruction error
- Optimizing prediction thresholds dynamically

This project implements both supervised and unsupervised learning approaches and combines them into advanced ensemble systems for improved fraud detection performance.

---

# Dataset

The dataset used in this project is the popular **Credit Card Fraud Detection Dataset**.

## Dataset Features

- Transactions made by European cardholders
- Numerical input variables only
- Features `V1` to `V28` are PCA-transformed variables
- Additional raw features:
  - `Time`
  - `Amount`
- Target column:
  - `Class`
    - `0` → Genuine Transaction
    - `1` → Fraudulent Transaction

## Dataset Characteristics

- Extremely imbalanced dataset
- Fraud transactions represent a very small percentage of total records
- Suitable for anomaly detection and imbalanced learning research

---

# Problem Statement

Traditional machine learning models often struggle with highly imbalanced datasets because they become biased toward the majority class.

Key challenges addressed in this project:

- Extreme class imbalance
- High false-negative risk
- Fraud anomaly detection
- Threshold optimization
- Precision vs Recall tradeoff
- Model generalization
- Ensemble robustness

---

# Project Workflow

## 1. Data Loading

- Load dataset using Pandas
- Inspect dataset shape and structure

---

## 2. Exploratory Data Analysis (EDA)

- Display dataset information
- Check missing values
- Analyze class distribution
- Visualize fraud imbalance

---

## 3. Data Preprocessing

### Feature Scaling

`StandardScaler` is applied to:

- `Time`
- `Amount`

### Train-Test Split

Dataset is split into:

- 80% training
- 20% testing

using stratified sampling.

---

## 4. Handling Imbalanced Data

SMOTE (Synthetic Minority Oversampling Technique) is used to generate synthetic fraud samples for better model learning.

---

## 5. Model Training

The project trains multiple models:

### Supervised Learning Models

- Logistic Regression
- Random Forest
- XGBoost
- Feed Forward Neural Network

### Unsupervised Learning Model

- Autoencoder

---

## 6. Threshold Optimization

Instead of using a fixed threshold (`0.5`), the project dynamically finds the optimal threshold using:

- Precision-Recall Curve
- F1 Score Maximization

---

## 7. Ensemble Learning

The project combines multiple model predictions using:

- Dynamic Weighted Ensemble
- Stacking Ensemble

---

# Technologies Used

## Programming Language

- Python

## Libraries

### Data Analysis

- NumPy
- Pandas

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost
- Imbalanced-learn

### Deep Learning

- TensorFlow
- Keras

### Model Persistence

- Joblib

---

# Machine Learning Models

# 1. Logistic Regression

A baseline linear classification model.

## Features

- Fast and interpretable
- Probability-based predictions
- Dynamic threshold optimization
- Precision-Recall analysis

## Evaluation Metrics

- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

---

# 2. Random Forest Classifier

An ensemble tree-based classification model.

## Features

- Handles nonlinear relationships
- Feature importance analysis
- Balanced class weighting
- Parallelized training

## Advantages

- Robust against overfitting
- Strong tabular performance
- High fraud detection capability

---

# 3. XGBoost Classifier

A gradient boosting algorithm optimized for structured data.

## Features

- Boosting-based learning
- Handles imbalance using `scale_pos_weight`
- Regularization support
- High predictive performance

## Advantages

- Excellent fraud detection accuracy
- State-of-the-art tabular learning performance

---

# Deep Learning Models

# 4. Feed Forward Neural Network (FFNN)

A fully connected deep neural network for fraud classification.

## Architecture

```text
Input Layer
   ↓
Dense(64) + ReLU
   ↓
Dropout(0.3)
   ↓
Dense(32) + ReLU
   ↓
Dropout(0.3)
   ↓
Dense(16) + ReLU
   ↓
Output Layer (Sigmoid)
```

## Features

- Dropout regularization
- Early stopping
- Binary cross-entropy loss
- Precision/Recall/AUC monitoring
- Class-weight handling

---

# 5. Autoencoder

An unsupervised anomaly detection model.

## Working Principle

- Train only on genuine transactions
- Learn normal transaction patterns
- Fraudulent transactions generate higher reconstruction error

## Reconstruction Error

```text
MSE = Mean((Original - Reconstructed)^2)
```

## Advantages

- Useful for anomaly detection
- Works even with limited fraud labels
- Detects unseen fraud patterns

---

# Ensemble Learning

# Dynamic Weighted Ensemble Model

The project combines predictions from:

- Logistic Regression
- Random Forest
- XGBoost
- Feed Forward Neural Network

## Ensemble Strategy

Weights are dynamically assigned based on PR-AUC scores.

## Dynamic Weight Formula

```text
Weight_i = PR-AUC_i / Total PR-AUC
```

## Ensemble Probability Formula

```text
Ensemble Probability =
(w1 × LR) +
(w2 × RF) +
(w3 × XGB) +
(w4 × NN)
```

## Benefits

- Improves robustness
- Reduces model bias
- Enhances fraud detection accuracy
- Balances precision and recall
- Leverages strengths of multiple models

---

# Stacking Ensemble Learning

The project also implements an advanced stacking ensemble architecture.

## Meta Features

The stacking model uses:

- Logistic Regression probabilities
- Random Forest probabilities
- XGBoost probabilities
- Neural Network probabilities
- Autoencoder anomaly scores

as meta-features.

## Meta Model

- Logistic Regression is used as the final meta-classifier.

## Advantages

- Learns relationships between model predictions
- Improves ensemble intelligence
- Enhances fraud classification performance

---

# Evaluation Metrics

Since fraud detection is an imbalanced classification problem, accuracy alone is not sufficient.

The following metrics are used:

# Precision

Measures how many predicted frauds are actually fraud.

# Recall

Measures how many actual frauds are detected.

# F1 Score

Harmonic mean of Precision and Recall.

# ROC-AUC

Measures model discrimination capability.

# PR-AUC (Precision-Recall AUC)

Most important metric for highly imbalanced datasets.

# Confusion Matrix

Shows:

- True Positives
- False Positives
- True Negatives
- False Negatives

---

# Visualizations

The project includes multiple visualizations:

- Class distribution plots
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Feature importance plots
- Neural network training curves
- Ensemble performance plots

---

# Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── creditcard.csv
├── fraud_detection.ipynb
├── fraud_detection_rf.pkl
├── fraud_detection_ffnn.h5
├── dynamic_ensemble_config.pkl
├── stacking_meta_model.pkl
├── autoencoder_mse_scaler.pkl
├── README.md
│
└── images/
    ├── imbalance.png
    ├── roc_curve.png
    ├── pr_curve.png
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── training_loss.png
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
```

---

## Navigate to Project Folder

```bash
cd Credit-Card-Fraud-Detection
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost tensorflow joblib
```

---

# How to Run

## Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
fraud_detection.ipynb
```

Run all cells sequentially.

---

# Results

The project successfully:

- Detects fraudulent transactions with high recall
- Handles severe class imbalance effectively
- Optimizes classification thresholds dynamically
- Compares multiple ML and DL approaches
- Uses anomaly detection for fraud analysis
- Improves performance through ensemble learning
- Builds advanced stacking ensemble architecture

## Best Performing Models

The following models achieved strong fraud detection performance:

- XGBoost
- Dynamic Weighted Ensemble
- Stacking Ensemble

The ensemble models achieved the best overall balance between:

- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

---

# Key Features

# Advanced Fraud Detection Pipeline

- Data preprocessing
- SMOTE balancing
- Feature scaling
- Threshold tuning
- Deep learning integration
- Anomaly detection
- Ensemble learning
- Stacking ensemble architecture

---

# Dynamic Threshold Optimization

Instead of using a fixed threshold of `0.5`, the system automatically finds the optimal threshold using F1 score maximization from the Precision-Recall curve.

---

# Real-World Fraud Detection Techniques

- PR-AUC optimization
- Class weighting
- Reconstruction-error anomaly detection
- Weighted soft-voting ensemble
- Meta-learning based stacking

---

# Future Improvements

Possible future enhancements include:

- Real-time fraud detection API
- Streamlit web application
- Hyperparameter tuning using Optuna
- Stratified cross-validation
- Explainable AI using SHAP/LIME
- Docker deployment
- Cloud deployment (AWS/GCP/Azure)
- Real-time transaction streaming
- Graph Neural Networks (GNNs)
- Transformer-based anomaly detection

---

# Conclusion

This project demonstrates a complete real-world fraud detection pipeline using traditional machine learning, deep learning, anomaly detection, and ensemble learning techniques.

By combining multiple supervised and unsupervised models into dynamic ensemble systems, the project achieves robust fraud detection performance on highly imbalanced financial datasets.

The implementation showcases practical applications of:

- Imbalanced learning
- Deep learning
- Ensemble learning
- Stacking models
- Anomaly detection
- Threshold optimization
- Precision-Recall analysis
- Model evaluation

making it a strong end-to-end machine learning portfolio project for real-world financial fraud detection.


---

# License

This project is licensed under the MIT License.
