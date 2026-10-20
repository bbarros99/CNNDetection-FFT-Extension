
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score

# --------------------------------------------------
# Directories
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
DATA = ROOT / "data"

FIGURES.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load final test results
# --------------------------------------------------

metrics = pd.read_csv(RESULTS / "final_metrics.csv")
predictions = pd.read_csv(RESULTS / "final_predictions.csv")

y_true = predictions["label"].to_numpy()

print("Generating figures...")
print(f"Test images: {len(predictions)}")

# --------------------------------------------------
# Figure 1: Performance comparison
# --------------------------------------------------

metric_names = [
    "accuracy",
    "average_precision",
    "roc_auc",
]

metric_labels = [
    "Accuracy",
    "Average Precision",
    "ROC AUC",
]

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 6))

for i, (metric, label) in enumerate(
    zip(metric_names, metric_labels)
):
    values = metrics[metric].to_numpy() * 100

    bars = ax.bar(
        x + (i - 1) * width,
        values,
        width,
        label=label,
    )

    ax.bar_label(
        bars,
        fmt="%.1f",
        padding=3,
        fontsize=9,
    )

ax.set_xticks(x)
ax.set_xticklabels(metrics["method"])
ax.set_ylabel("Score (%)")
ax.set_title("Detection Performance Comparison")
ax.set_ylim(0, 105)
ax.legend()
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig(
    FIGURES / "performance_comparison.png",
    dpi=200,
)
plt.close()

# --------------------------------------------------
# Figure 2: ROC curves
# --------------------------------------------------

methods = {
    "CNNDetection": "cnn_score",
    "FFT": "fft_score",
    "CNN + FFT": "fusion_score",
}

plt.figure(figsize=(8, 6))

for name, column in methods.items():

    scores = predictions[column].to_numpy()

    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{name} (AUC = {auc:.3f})",
    )

plt.plot(
    [0, 1],
    [0, 1],
    "--",
    color="gray",
    label="Random classifier",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Test Dataset")
plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(
    FIGURES / "roc_curves.png",
    dpi=200,
)
plt.close()

# --------------------------------------------------
# Figure 3: Score distributions
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, column) in zip(axes, methods.items()):

    real = predictions.loc[
        predictions["label"] == 0,
        column,
    ].to_numpy()

    fake = predictions.loc[
        predictions["label"] == 1,
        column,
    ].to_numpy()

    # CNN values span several orders of magnitude
    if column == "cnn_score":
        real = np.log10(np.maximum(real, 1e-10))
        fake = np.log10(np.maximum(fake, 1e-10))
        xlabel = "Log10(CNN score)"
    else:
        xlabel = "Predicted score"

    ax.hist(
        real,
        bins=12,
        alpha=0.6,
        label="Real",
    )

    ax.hist(
        fake,
        bins=12,
        alpha=0.6,
        label="Fake",
    )

    ax.set_title(name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of images")
    ax.legend()
    ax.grid(alpha=0.2)

plt.suptitle("Score Distributions - Real vs Fake")
plt.tight_layout()
plt.savefig(
    FIGURES / "score_distributions.png",
    dpi=200,
)
plt.close()

# --------------------------------------------------
# Figure 4: Fusion classification errors
# --------------------------------------------------

predictions["fusion_prediction"] = (
    predictions["fusion_score"] >= 0.5
).astype(int)

errors = predictions[
    predictions["fusion_prediction"] !=
    predictions["label"]
]

print(f"Fusion errors: {len(errors)}")

# Display up to 8 wrongly classified images
selected = errors.head(8)

if len(selected) > 0:

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(13, 7),
    )

    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for ax, row in zip(axes, selected.itertuples()):

        image_path = DATA / row.filename

        with Image.open(image_path) as img:
            image = img.convert("RGB").copy()

        ax.imshow(image)

        actual = "Fake" if row.label == 1 else "Real"
        predicted = (
            "Fake" if row.fusion_prediction == 1
            else "Real"
        )

        ax.set_title(
            f"Actual: {actual}\n"
            f"Predicted: {predicted}\n"
            f"Score: {row.fusion_score:.3f}",
            fontsize=10,
        )

    plt.suptitle(
        "Examples of Misclassified Images - CNN + FFT"
    )

    plt.tight_layout()
    plt.savefig(
        FIGURES / "classification_errors.png",
        dpi=200,
    )
    plt.close()

# --------------------------------------------------
# Finished
# --------------------------------------------------

print("\nFigures generated successfully!")

for path in sorted(FIGURES.glob("*.png")):
    print(path)
