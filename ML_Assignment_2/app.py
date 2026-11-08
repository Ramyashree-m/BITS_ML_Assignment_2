
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

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

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer ML Model Evaluation",
    page_icon="🔬",
    layout="wide"
)

# --------------------------------------------------
# PROJECT DIRECTORY
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title("Breast Cancer Classification")

st.subheader("Machine Learning Model Evaluation Dashboard")

st.write(
    "Upload test data, select a trained model, "
    "and evaluate its classification performance."
)

st.divider()

# --------------------------------------------------
# MODEL FILE MAPPING
# --------------------------------------------------

model_files = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "KNN": MODEL_DIR / "knn.joblib",
    "Gaussian Naive Bayes": MODEL_DIR / "gaussian_naive_bayes.joblib",
    "SVM": MODEL_DIR / "svm.joblib"
}

# --------------------------------------------------
# TEST DATA UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

selected_model = st.selectbox(
    "Select Machine Learning Model",
    list(model_files.keys())
)

# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

if uploaded_file is not None:

    try:
        # Read uploaded CSV
        df = pd.read_csv(uploaded_file)

        st.write("### Uploaded Dataset Preview")
        st.dataframe(df.head())

        # Check target column
        if "diagnosis" not in df.columns:
            st.error(
                "The uploaded CSV must contain a diagnosis column."
            )
            st.stop()

        # Separate features and target
        X_test = df.drop("diagnosis", axis=1)

        y_test = df["diagnosis"].map({
            "benign": 0,
            "malignant": 1
        })

        # Validate target values
        if y_test.isnull().any():
            st.error(
                "Diagnosis must contain only benign or malignant."
            )
            st.stop()

        # Check model file existence
        model_path = model_files[selected_model]

        if not model_path.exists():
            st.error(
                f"Model file not found: {model_path}"
            )
            st.stop()

        # Load saved trained model
        model = joblib.load(model_path)

        # Validate input features
        if list(X_test.columns) != list(model.feature_names_in_):
            st.error(
                "Uploaded dataset features do not match "
                "the features used during training."
            )
            st.stop()

        # Make predictions
        y_pred = model.predict(X_test)

        y_prob = model.predict_proba(X_test)[:, 1]

        # --------------------------------------------------
        # EVALUATION METRICS
        # --------------------------------------------------

        accuracy = accuracy_score(y_test, y_pred)

        auc = roc_auc_score(y_test, y_prob)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        mcc = matthews_corrcoef(y_test, y_pred)

        # --------------------------------------------------
        # DISPLAY RESULTS
        # --------------------------------------------------

        st.divider()

        st.write("### Evaluation Results")

        st.write(f"**Selected Model:** {selected_model}")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # --------------------------------------------------
        # CONFUSION MATRIX
        # --------------------------------------------------

        st.divider()

        st.write("### Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(7, 5))

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
        ax.set_title("Confusion Matrix")

        st.pyplot(fig)

        plt.close(fig)

        # --------------------------------------------------
        # CLASSIFICATION REPORT
        # --------------------------------------------------

        st.divider()

        st.write("### Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Benign", "Malignant"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df)

        st.success(
            f"{selected_model} evaluated successfully!"
        )

    except Exception as e:

        st.error(
            f"An error occurred during evaluation: {str(e)}"
        )

else:

    st.info(
        "Please upload your test_data.csv file to begin evaluation."
    )

    st.write(
        "The application uses previously trained models "
        "and does not retrain them."
    )
