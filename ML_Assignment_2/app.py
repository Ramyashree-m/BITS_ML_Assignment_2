
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Breast Cancer ML Model Evaluation",
    page_icon="🔬",
    layout="wide"
)

st.title("Breast Cancer Classification")
st.subheader("Machine Learning Model Evaluation Dashboard")

st.write(
    "Upload test data, select a trained model, "
    "and evaluate its classification performance."
)

# Model file mapping
model_files = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Gaussian Naive Bayes": "model/gaussian_naive_bayes.joblib",
    "SVM": "model/svm.joblib"
}

# Upload test dataset
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

# Select model
selected_model = st.selectbox(
    "Select Machine Learning Model",
    list(model_files.keys())
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("### Uploaded Dataset Preview")
    st.dataframe(df.head())

    if "diagnosis" not in df.columns:
        st.error("The uploaded CSV must contain a diagnosis column.")

    else:

        # Separate input features and target
        X_test = df.drop("diagnosis", axis=1)

        y_test = df["diagnosis"].map({
            "benign": 0,
            "malignant": 1
        })

        if y_test.isnull().any():
            st.error("Diagnosis must contain benign or malignant.")
            st.stop()

        # Load selected trained model
        model = joblib.load(model_files[selected_model])

        # Check feature compatibility
        if list(X_test.columns) != list(model.feature_names_in_):
            st.error(
                "Uploaded dataset features do not match "
                "the features used during training."
            )
            st.stop()

        # Predict using saved model
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.write("### Evaluation Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # Confusion matrix
        st.write("### Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

        st.pyplot(fig)
        plt.close(fig)

        # Classification report
        st.write("### Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            target_names=["Benign", "Malignant"],
            output_dict=True
        )

        st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("Please upload your test_data.csv file to begin evaluation.")
