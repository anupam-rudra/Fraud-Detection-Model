# Credit Card Fraud Detection Using Machine Learning & Deep Learning

A complete end-to-end **Credit Card Fraud Detection System** built using multiple Machine Learning, Deep Learning, and Ensemble Learning techniques to identify fraudulent transactions from highly imbalanced financial datasets.

This project combines:

- Logistic Regression
- Random Forest
- XGBoost
- Feed Forward Neural Network (FFNN)
- Autoencoder Anomaly Detection
- Dynamic Weighted Ensemble Model

with advanced techniques such as:

- SMOTE oversampling
- Threshold optimization
- Precision-Recall analysis
- ROC-AUC evaluation
- Dynamic ensemble weighting

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
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Visualizations](#visualizations)
- [Future Improvements](#future-improvements)
- [Conclusion](#conclusion)

---

# Project Overview

Credit card fraud detection is one of the most important applications of machine learning in the financial sector.

Fraudulent transactions are extremely rare compared to genuine transactions, making this a highly imbalanced classification problem.

The main objective of this project is to build an intelligent fraud detection system capable of:

- Detecting fraudulent transactions accurately
- Minimizing false positives
- Handling class imbalance effectively
- Improving prediction performance using ensemble learning

This project implements multiple supervised and unsupervised learning approaches and combines them into a powerful dynamic ensemble model.

---

# Dataset

The dataset used in this project is the popular **Credit Card Fraud Detection Dataset**.

## Dataset Features

- Transactions made by European cardholders
- Contains numerical input variables
- Features `V1` to `V28` are PCA-transformed variables
- Additional features:
  - `Time`
  - `Amount`
- Target column:
  - `Class`
    - `0` → Genuine Transaction
    - `1` → Fraudulent Transaction

## Dataset Characteristics

- Highly imbalanced dataset
- Fraud cases represent a very small percentage of total transactions

---

# Problem Statement

Traditional machine learning models often struggle with highly imbalanced datasets because they become biased toward the majority class.

Challenges addressed in this project:

- Extreme class imbalance
- High false-negative risk
- Threshold optimization
- Precision vs Recall tradeoff
- Fraud anomaly detection
- Model robustness

---

# Project Workflow

## 1. Data Loading

- Load dataset using Pandas
- Inspect dataset shape and structure

## 2. Exploratory Data Analysis (EDA)

- Display dataset information
- Check missing values
- Analyze class distribution
- Visualize fraud imbalance

## 3. Data Preprocessing

### Feature Scaling

StandardScaler is applied to:
- `Time`
- `Amount`

### Train-Test Split

Dataset split into:
- 80% training
- 20% testing

using stratified sampling.

## 4. Handling Imbalanced Data

SMOTE (Synthetic Minority Oversampling Technique) is used to generate synthetic fraud samples for better learning.

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

## 1. Logistic Regression

A baseline linear classification model.

### Features
- Fast and interpretable
- Probability-based prediction
- Dynamic threshold optimization

### Evaluation
- Precision
- Recall
- F1 Score
- ROC-AUC
- Precision-Recall Curve

---

## 2. Random Forest Classifier

An ensemble tree-based model.

### Features
- Handles nonlinear relationships
- Feature importance analysis
- Balanced class weighting
- Parallel training support

### Advantages
- Robust against overfitting
- High accuracy on tabular data

---

## 3. XGBoost Classifier

A gradient boosting algorithm optimized for structured data.

### Features
- Boosting-based learning
- Handles imbalance using `scale_pos_weight`
- High predictive performance
- Regularization support

### Advantages
- Excellent fraud detection capability
- State-of-the-art tabular performance

---

# Deep Learning Models

## 4. Feed Forward Neural Network (FFNN)

A fully connected deep neural network for fraud classification.

### Architecture

Input Layer → Dense(64) → Dense(32) → Dense(16) → Output Layer

### Features
- Dropout regularization
- Early stopping
- Binary cross-entropy loss
- Precision/Recall/AUC monitoring

---

## 5. Autoencoder

An unsupervised anomaly detection model.

### Working Principle

- Train only on genuine transactions
- Learn normal transaction patterns
- Fraudulent transactions produce high reconstruction error

### Advantages
- Useful for anomaly detection
- Works even with limited fraud labels

---

# Ensemble Learning

## Dynamic Weighted Ensemble Model

The final system combines predictions from:

- Logistic Regression
- Random Forest
- XGBoost
- Neural Network

### Ensemble Strategy

Weights are dynamically assigned based on PR-AUC scores.

### Weighted Probability Formula

```text
Ensemble Probability =
(w1 × LR) +
(w2 × RF) +
(w3 × XGB) +
(w4 × NN)
```

### Benefits

- Improves robustness
- Reduces model bias
- Enhances fraud detection accuracy
- Balances precision and recall

---

# Evaluation Metrics

Since fraud detection is an imbalanced classification problem, traditional accuracy is not sufficient.

The following metrics are used:

## Precision
Measures how many predicted frauds are actually fraud.

## Recall
Measures how many actual frauds are detected.

## F1 Score
Harmonic mean of Precision and Recall.

## ROC-AUC
Measures model discrimination ability.

## Precision-Recall Curve
Most important metric for highly imbalanced datasets.

## Confusion Matrix
Shows:
- True Positives
- False Positives
- True Negatives
- False Negatives

---

# Visualizations

The project includes:

- Class distribution plots
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Feature importance plots
- Neural network training curves

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
├── README.md
│
└── images/
    ├── imbalance.png
    ├── roc_curve.png
    ├── pr_curve.png
    ├── confusion_matrix.png
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
```

## Navigate to Project Folder

```bash
cd Credit-Card-Fraud-Detection
```

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

## Run Jupyter Notebook

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
- Handles severe class imbalance
- Optimizes thresholds dynamically
- Improves performance using ensemble learning
- Compares ML and Deep Learning approaches

The ensemble model achieved the best overall balance between:

- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

---

# Key Features

## Advanced Fraud Detection Pipeline

- Data preprocessing
- SMOTE balancing
- Threshold tuning
- Deep learning integration
- Ensemble learning

## Dynamic Threshold Optimization

Instead of using a fixed threshold of `0.5`, the system automatically finds the optimal threshold using F1 score maximization.

## Real-World Fraud Detection Techniques

- PR-AUC optimization
- Class weighting
- Reconstruction-error anomaly detection
- Weighted soft voting ensemble

---

# Future Improvements

Possible enhancements include:

- Real-time fraud detection API
- Streamlit web application
- Hyperparameter tuning using Optuna
- Cross-validation framework
- Explainable AI (SHAP/LIME)
- Docker deployment
- Cloud deployment (AWS/GCP/Azure)
- Real-time transaction streaming
- Graph neural networks for fraud detection

---

# Conclusion

This project demonstrates a complete fraud detection pipeline using both traditional machine learning and deep learning methods.

By combining multiple models into a dynamic weighted ensemble system, the project achieves robust fraud detection performance on highly imbalanced financial data.

The implementation showcases practical applications of:

- Imbalanced learning
- Ensemble learning
- Deep learning
- Anomaly detection
- Threshold optimization
- Model evaluation

making it a strong real-world machine learning portfolio project.

---

# Author

## Anupam Naskar

Machine Learning | Deep Learning | Data Science

---

# License

This project is licensed under the MIT License.
