
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

# Directories
ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "mini_project"
RESULTS = PROJECT / "results"
MODELS = PROJECT / "models"

sys.path.insert(0, str(ROOT))
from networks.resnet import resnet50

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

# Fixed threshold selected on validation
CNN_THRESHOLD = 0.0000013586

# Load test images and FFT features
df = pd.read_csv(RESULTS / "fft_features.csv")
test = df[df["split"] == "test"].copy()

assert len(test) == 40, "Expected 40 test images"

y_test = test["label"].to_numpy()

# Load FFT and fusion classifiers
fft_model = joblib.load(
    MODELS / "fft_classifier.joblib"
)

fusion_model = joblib.load(
    MODELS / "fusion_classifier.joblib"
)

# Load original CNNDetection
device = torch.device("cpu")

model = resnet50(num_classes=1)
checkpoint = torch.load(
    ROOT / "weights" / "blur_jpg_prob0.5.pth",
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# CNN predictions
cnn_scores = []

print("Evaluating CNNDetection on test images...")

for i, row in enumerate(test.itertuples(), 1):
    image_path = PROJECT / "data" / row.filename

    with Image.open(image_path) as img:
        image = img.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        score = torch.sigmoid(model(tensor)).item()

    cnn_scores.append(score)
    print(f"[{i}/{len(test)}] {row.filename}: {score:.8f}")

cnn_scores = np.asarray(cnn_scores)

# FFT predictions
fft_scores = fft_model.predict_proba(
    test[FEATURES]
)[:, 1]

# Apply the same CNN transformation as in fusion.py
def cnn_to_logit(scores):
    p = np.clip(
        np.asarray(scores, dtype=float),
        1e-10,
        1 - 1e-10
    )
    return np.log(p / (1 - p))

fusion_inputs = np.column_stack([
    cnn_to_logit(cnn_scores),
    fft_scores
])

fusion_scores = fusion_model.predict_proba(
    fusion_inputs
)[:, 1]

# Evaluation
def evaluate(name, scores, threshold=0.5):
    predictions = (scores >= threshold).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    ap = average_precision_score(y_test, scores)
    auc = roc_auc_score(y_test, scores)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ))

    return {
        "method": name,
        "accuracy": accuracy,
        "average_precision": ap,
        "roc_auc": auc,
    }


results = []

results.append(evaluate(
    "Original CNN", cnn_scores
))

results.append(evaluate(
    "CNN adapted threshold",
    cnn_scores,
    CNN_THRESHOLD
))

results.append(evaluate(
    "FFT classifier", fft_scores
))

results.append(evaluate(
    "CNN + FFT fusion", fusion_scores
))

# Save results
pd.DataFrame(results).to_csv(
    RESULTS / "final_metrics.csv",
    index=False
)

pd.DataFrame({
    "filename": test["filename"].to_numpy(),
    "label": y_test,
    "cnn_score": cnn_scores,
    "fft_score": fft_scores,
    "fusion_score": fusion_scores,
}).to_csv(
    RESULTS / "final_predictions.csv",
    index=False
)

print("\nFinal evaluation completed!")
print("Results saved in:", RESULTS)
