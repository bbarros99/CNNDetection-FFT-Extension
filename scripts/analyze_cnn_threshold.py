
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

df = pd.read_csv(RESULTS / "cnn_validation_scores.csv")

y = df["label"].to_numpy()
scores = df["cnn_score"].to_numpy()

# Compare the score distributions
for label, name in [(0, "Real"), (1, "Fake")]:
    values = scores[y == label]

    print(f"\n{name} images")
    print(f"Minimum: {values.min():.10f}")
    print(f"Median:  {np.median(values):.10f}")
    print(f"Maximum: {values.max():.10f}")

# Find candidate thresholds between consecutive scores
unique_scores = np.unique(scores)

thresholds = [0.0]
thresholds += [
    (float(a) + float(b)) / 2
    for a, b in zip(unique_scores[:-1], unique_scores[1:])
]
thresholds += [1.0]

best_threshold = 0.5
best_accuracy = -1

for threshold in thresholds:
    predictions = (scores >= threshold).astype(int)
    accuracy = accuracy_score(y, predictions)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print("\n=== CNN threshold analysis ===")
print("Original threshold: 0.5")
print("Original accuracy:", accuracy_score(
    y, (scores >= 0.5).astype(int)
))

print(f"\nBest validation threshold: {best_threshold:.10f}")
print(f"Best validation accuracy: {best_accuracy:.4f}")

print("\nConfusion matrix with new threshold:")
print(confusion_matrix(
    y, (scores >= best_threshold).astype(int),
    labels=[0, 1]
))

print("\nROC AUC:", roc_auc_score(y, scores))
