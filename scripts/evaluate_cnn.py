
import sys
import argparse

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

from project_paths import CNN_ROOT, DATA, RESULTS, WEIGHTS

# --------------------------------------------------
# Original CNNDetection model
# --------------------------------------------------

sys.path.insert(0, str(CNN_ROOT))
from networks.resnet import resnet50

# --------------------------------------------------
# Command-line arguments
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description="Evaluate CNNDetection on a selected dataset split."
)

parser.add_argument(
    "--split",
    choices=["train", "validation", "test"],
    required=True,
    help="Dataset split to evaluate."
)

args = parser.parse_args()
split = args.split

# --------------------------------------------------
# Check required files
# --------------------------------------------------

if not WEIGHTS.is_file():
    raise FileNotFoundError(
        f"CNNDetection weights not found: {WEIGHTS}"
    )

features_path = RESULTS / "fft_features.csv"

if not features_path.is_file():
    raise FileNotFoundError(
        f"FFT features not found: {features_path}"
    )

RESULTS.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load original CNNDetection model
# --------------------------------------------------

device = torch.device("cpu")

print("Loading CNNDetection model...")

model = resnet50(num_classes=1)

checkpoint = torch.load(
    WEIGHTS,
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# --------------------------------------------------
# Load selected dataset
# --------------------------------------------------

df = pd.read_csv(features_path)

split_data = df[df["split"] == split].copy()

if split_data.empty:
    raise ValueError(
        f"No images found for split: {split}"
    )

print(f"\nEvaluating {len(split_data)} {split} images...")

# --------------------------------------------------
# CNN predictions
# --------------------------------------------------

scores = []

for i, row in enumerate(split_data.itertuples(), start=1):

    image_path = DATA / row.filename

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(image_path) as img:
        image = img.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        score = torch.sigmoid(model(tensor)).item()

    scores.append(score)

    print(
        f"[{i}/{len(split_data)}] "
        f"{row.filename}: {score:.8f}"
    )

# --------------------------------------------------
# Evaluation metrics
# --------------------------------------------------

split_data["cnn_score"] = scores

y_true = split_data["label"]

predictions = (
    split_data["cnn_score"] >= 0.5
).astype(int)

print(f"\n=== CNNDetection {split.capitalize()} Results ===")

print(
    "Accuracy:",
    round(accuracy_score(y_true, predictions), 4)
)

print(
    "Average Precision:",
    round(average_precision_score(y_true, scores), 4)
)

print(
    "ROC AUC:",
    round(roc_auc_score(y_true, scores), 4)
)

print("\nConfusion matrix [Real, Fake]:")

print(
    confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )
)

# --------------------------------------------------
# Save results
# --------------------------------------------------

output = RESULTS / f"cnn_{split}_scores.csv"

split_data[
    ["filename", "label", "cnn_score"]
].to_csv(output, index=False)

print(f"\n{split.capitalize()} evaluation completed!")
print("Scores saved to:", output)
