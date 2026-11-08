
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv(
    "dataset/breast_cancer_wisconsin_diagnostic.csv"
)

# Separate input features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display split information
print("DATA PREPROCESSING COMPLETED")

print("\nOriginal Dataset Shape:", X.shape)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nTraining Target Distribution:")
print(pd.Series(y_train).value_counts())

print("\nTesting Target Distribution:")
print(pd.Series(y_test).value_counts())

print("\nClass Encoding:")
print(dict(zip(label_encoder.classes_,
               label_encoder.transform(label_encoder.classes_))))

# Save test data with original target labels
test_df = X_test.copy()
test_df["diagnosis"] = label_encoder.inverse_transform(y_test)

test_df.to_csv("test_data.csv", index=False)

print("\nTest data saved successfully as test_data.csv")
