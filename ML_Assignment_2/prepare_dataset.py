
import pandas as pd
from sklearn.datasets import load_breast_cancer
from pathlib import Path

# Load the Breast Cancer Wisconsin dataset
data = load_breast_cancer(as_frame=True)

# Convert features into a DataFrame
df = data.data.copy()

# Add the target column
df["diagnosis"] = data.target.map({
    0: "malignant",
    1: "benign"
})

# Save the dataset inside the dataset folder
output_path = Path("dataset/breast_cancer_wisconsin_diagnostic.csv")
df.to_csv(output_path, index=False)

# Display dataset information
print("Dataset created successfully!")
print("Dataset shape:", df.shape)
print("\nFirst 5 records:")
print(df.head())

print("\nTarget distribution:")
print(df["diagnosis"].value_counts())

print("\nMissing values:")
print(df.isnull().sum().sum())
