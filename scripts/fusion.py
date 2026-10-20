
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

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

# Load the three datasets
fft_data = pd.read_csv(RESULTS / "fft_features.csv")
cnn_train = pd.read_csv(RESULTS / "cnn_train_scores.csv")
cnn_val = pd.read_csv(RESULTS / "cnn_validation_scores.csv")

cnn_scores = pd.concat(
    [cnn_train, cnn_val],
    ignore_index=True
)

# Join by filename, rather than relying on row order
data = fft_data.merge(
    cnn_scores[["filename", "label", "cnn_score"]],
    on=["filename", "label"],
    how="inner",
    validate="one_to_one"
)

expected = len(cnn_train) + len(cnn_val)
if len(data) != expected:
    raise ValueError("Some CNN scores are missing or duplicated")

train = data[data["split"] == "train"].copy()
val = data[data["split"] == "validation"].copy()

if len(train) != 120 or len(val) != 40:
    raise ValueError("Unexpected dataset sizes")

X_train = train[FEATURES]
y_train = train["label"].to_numpy()

X_val = val[FEATURES]
y_val = val["label"].to_numpy()

# FFT classifier: same configuration as before
def create_fft_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )

# Out-of-fold FFT predictions for training fusion
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fft_train_scores = cross_val_predict(
    create_fft_model(),
    X_train,
    y_train,
    cv=cv,
    method="predict_proba"
)[:, 1]

# Use the FFT model already trained on all training images
fft_model = joblib.load(MODELS / "fft_classifier.joblib")
fft_val_scores = fft_model.predict_proba(X_val)[:, 1]

# CNN probabilities are extremely small.
# Transform them to log-odds before fusion.
def cnn_to_logit(scores):
    p = np.clip(
        np.asarray(scores, dtype=float),
        1e-10,
        1 - 1e-10
    )
    return np.log(p / (1 - p))

def combine_inputs(cnn, fft):
    return np.column_stack([
        cnn_to_logit(cnn),
        fft
    ])

fusion_train = combine_inputs(
    train["cnn_score"],
    fft_train_scores
)

fusion_val = combine_inputs(
    val["cnn_score"],
    fft_val_scores
)

# Small fusion classifier
fusion_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=1000,
        random_state=42
    )
)

fusion_model.fit(fusion_train, y_train)

fusion_scores = fusion_model.predict_proba(fusion_val)[:, 1]

# Threshold previously selected on validation
CNN_THRESHOLD = 0.0000013586

def evaluate(name, scores, threshold=0.5):
    predictions = (np.asarray(scores) >= threshold).astype(int)

    print(f"\n=== {name} ===")
    print("Accuracy:", round(accuracy_score(y_val, predictions), 4))
    print("Average Precision:", round(
        average_precision_score(y_val, scores), 4
    ))
    print("ROC AUC:", round(
        roc_auc_score(y_val, scores), 4
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, predictions, labels=[0, 1]))

# Compare all methods
evaluate("Original CNN", val["cnn_score"])
evaluate(
    "CNN with adapted threshold",
    val["cnn_score"],
    CNN_THRESHOLD
)
evaluate("FFT classifier", fft_val_scores)
evaluate("CNN + FFT fusion", fusion_scores)

# Save fusion model and validation predictions
joblib.dump(fusion_model, MODELS / "fusion_classifier.joblib")

output = pd.DataFrame({
    "filename": val["filename"].to_numpy(),
    "label": y_val,
    "cnn_score": val["cnn_score"].to_numpy(),
    "fft_score": fft_val_scores,
    "fusion_score": fusion_scores,
})

output.to_csv(
    RESULTS / "fusion_validation_results.csv",
    index=False
)

print("\nFusion completed!")
print("Model saved to:", MODELS / "fusion_classifier.joblib")
print("Results saved to:", RESULTS / "fusion_validation_results.csv")
