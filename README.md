# Breast Cancer Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop and evaluate machine learning classification models to predict whether a breast tumour is benign or malignant using diagnostic features.

Five machine learning algorithms are implemented and compared using standard classification evaluation metrics.

## 2. Dataset Description

**Dataset:** Breast Cancer Wisconsin Diagnostic Dataset

- Total instances: 569
- Input features: 30
- Target variable: diagnosis
- Classes: Benign and Malignant
- Missing values: 0
- Duplicate records: 0

The dataset contains numerical measurements related to breast cell nuclei.

## 3. Machine Learning Models

The following five algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The dataset was divided into 80% training data and 20% testing data.

## 4. Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| SVM | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

## 5. Observations

- Logistic Regression achieved the highest AUC of 0.9960.
- Decision Tree achieved an accuracy of 0.9298.
- KNN achieved an accuracy of 0.9561.
- Gaussian Naive Bayes achieved the highest precision of 1.0000, tied with SVM.
- SVM achieved the highest accuracy, F1 Score and MCC.

## 6. Overall Best Model

Based on the test results, Support Vector Machine (SVM) performed best overall, achieving:

- Accuracy: 97.37%
- AUC: 99.47%
- Precision: 100%
- Recall: 92.86%
- F1 Score: 96.30%
- MCC: 0.9442

## 7. Application

A Streamlit application was developed to allow users to:

- Upload test datasets in CSV format.
- Select one of the five trained models.
- View classification evaluation metrics.
- View the confusion matrix.
- View the classification report.

The application loads previously saved models and evaluates the uploaded test dataset without retraining.

## 8. Project Structure

```text
ML_Assignment_2/
├── dataset/
├── model/
├── app.py
├── prepare_dataset.py
├── data_exploration.py
├── data_preprocessing.py
├── model_training.py
├── model_comparison.csv
├── requirements.txt
└── test_data.csv
```

## 9. GitHub Repository

https://github.com/Ramyashree-m/BITS_ML_Assignment_2

## 10. Streamlit Application

Live application link: To be added after deployment.
