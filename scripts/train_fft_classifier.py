
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

# Load extracted features
df = pd.read_csv(RESULTS / "fft_features.csv")

FEATURES = [
    "energy_low",
    "energy_mid_low",
    "energy_mid_high",
    "energy_high",
    "ratio_high_low",
    "ratio_high_mid",
    "ratio_mid_low",
    "spectral_slope",
]

# Use train for learning and validation for evaluation
train = df[df["split"] == "train"]
validation = df[df["split"] == "validation"]

X_train = train[FEATURES]
y_train = train["label"]

X_val = validation[FEATURES]
y_val = validation["label"]

# Standardize features and train classifier
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

model.fit(X_train, y_train)

# Evaluate the model
def evaluate(name, X, y):
    scores = model.predict_proba(X)[:, 1]
    predictions = (scores >= 0.5).astype(int)

    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(y, predictions), 4))
    print("Average Precision:", round(average_precision_score(y, scores), 4))
    print("ROC AUC:", round(roc_auc_score(y, scores), 4))

evaluate("Training results", X_train, y_train)
evaluate("Validation results", X_val, y_val)

# Save classifier for later fusion with CNNDetection
joblib.dump(model, MODELS / "fft_classifier.joblib")

print("\nModel saved to:", MODELS / "fft_classifier.joblib")
