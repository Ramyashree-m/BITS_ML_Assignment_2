
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("dataset/breast_cancer_wisconsin_diagnostic.csv")

# 1. Dataset dimensions
print("DATASET SHAPE")
print(df.shape)

# 2. Column names
print("\nCOLUMN NAMES")
print(df.columns.tolist())

# 3. Dataset information
print("\nDATASET INFORMATION")
df.info()

# 4. Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# 5. Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# 6. Duplicate records
print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

# 7. Target distribution
print("\nTARGET DISTRIBUTION")
print(df["diagnosis"].value_counts())

# 8. Target distribution visualization
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="diagnosis")
plt.title("Distribution of Breast Cancer Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig("dataset/target_distribution.png")
plt.show()

print("\nEDA completed successfully!")
