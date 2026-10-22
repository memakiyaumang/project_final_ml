import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import joblib


# ==================================================
# Paths
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "performance_dataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model",
    "performance"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "performance_model.pkl"
)


# ==================================================
# Create model directory
# ==================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ==================================================
# Load Dataset
# ==================================================

print("Loading performance dataset...")

data = pd.read_csv(
    DATA_PATH
)

print(
    f"Dataset loaded successfully: {len(data)} records"
)


# ==================================================
# Features
# ==================================================

features = [
    "courseProgress",
    "completedLessons",
    "totalLessons",
    "quizAverage",
    "quizAttempts",
    "passedQuizzes"
]


X = data[features]

y = data["performance"]


# ==================================================
# Split Dataset
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print(
    f"Training records: {len(X_train)}"
)

print(
    f"Testing records: {len(X_test)}"
)


# ==================================================
# Create Random Forest Model
# ==================================================

model = RandomForestClassifier(

    n_estimators=100,

    random_state=42,

    max_depth=6,

    min_samples_split=2

)


# ==================================================
# Train Model
# ==================================================

print("Training Random Forest model...")

model.fit(
    X_train,
    y_train
)


# ==================================================
# Test Model
# ==================================================

predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    predictions
)


print(
    f"\nModel Accuracy: {accuracy * 100:.2f}%"
)


# ==================================================
# Classification Report
# ==================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)


# ==================================================
# Save Model
# ==================================================

joblib.dump(
    model,
    MODEL_PATH
)


print(
    "\nModel saved successfully!"
)

print(
    f"Model path: {MODEL_PATH}"
)