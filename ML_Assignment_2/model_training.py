
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# Create model directory if it does not exist
Path("model").mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(
    "dataset/breast_cancer_wisconsin_diagnostic.csv"
)

# Separate features and target
X = df.drop("diagnosis", axis=1)
y = LabelEncoder().fit_transform(df["diagnosis"])

# Split data using the same random state as preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Define five machine learning models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Gaussian Naive Bayes": GaussianNB(),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ))
    ])
}

results = []

# Train and evaluate each model
for name, model in models.items():

    print(f"\nTraining: {name}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    results.append(metrics)

    # Save trained model
    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, f"model/{filename}")

    print("Model saved:", filename)

# Create comparison table
results_df = pd.DataFrame(results)

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nMODEL PERFORMANCE COMPARISON")
print(results_df.to_string(index=False))

print("\nAll five models trained and saved successfully!")
