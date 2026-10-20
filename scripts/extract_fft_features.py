
import csv
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SIZE = 256

FEATURE_NAMES = [
    "energy_low",
    "energy_mid_low",
    "energy_mid_high",
    "energy_high",
    "ratio_high_low",
    "ratio_high_mid",
    "ratio_mid_low",
    "spectral_slope",
]


def extract_features(image_path):
    # Load image and convert to grayscale
    image = Image.open(image_path).convert("L")
    image = image.resize((SIZE, SIZE), Image.Resampling.BICUBIC)
    pixels = np.asarray(image, dtype=np.float64) / 255.0

    # Remove the average brightness
    pixels -= pixels.mean()

    # Apply a Hann window to reduce border artifacts
    window = np.hanning(SIZE)
    pixels *= np.outer(window, window)

    # Compute the 2D Fourier transform
    fft = np.fft.fftshift(np.fft.fft2(pixels))
    power = np.abs(fft) ** 2

    # Distance from the center of the frequency spectrum
    y, x = np.indices((SIZE, SIZE))
    radius = np.sqrt(
        (x - SIZE // 2) ** 2 +
        (y - SIZE // 2) ** 2
    )

    # Four frequency bands
    bands = [(1, 16), (16, 32), (32, 64), (64, 128)]

    energies = []

    for minimum, maximum in bands:
        mask = (radius >= minimum) & (radius < maximum)
        energies.append(power[mask].sum())

    energies = np.array(energies)
    relative = energies / (energies.sum() + 1e-12)

    # Ratios between frequency bands
    low, mid_low, mid_high, high = relative

    ratio_high_low = high / (low + 1e-12)
    ratio_high_mid = high / (mid_high + 1e-12)
    ratio_mid_low = mid_high / (mid_low + 1e-12)

    # Average power at each frequency radius
    r = radius.astype(int)
    radial_sum = np.bincount(
        r.ravel(), weights=power.ravel()
    )
    radial_count = np.bincount(r.ravel())

    profile = radial_sum / np.maximum(radial_count, 1)

    # Measure how fast spectral power decreases
    frequencies = np.arange(2, 129)
    values = profile[frequencies]

    slope = np.polyfit(
        np.log(frequencies),
        np.log(values + 1e-12),
        1
    )[0]

    return [
        *relative.tolist(),
        ratio_high_low,
        ratio_high_mid,
        ratio_mid_low,
        float(slope),
    ]


rows = []

for split in ["train", "validation", "test"]:
    for category in ["real", "fake"]:
        folder = DATA / split / category
        images = sorted(folder.glob("*.png"))

        print(f"{split}/{category}: {len(images)} images")

        for image_path in images:
            features = extract_features(image_path)

            row = {
                "filename": str(image_path.relative_to(DATA)),
                "split": split,
                "label": 0 if category == "real" else 1,
            }

            row.update(dict(zip(FEATURE_NAMES, features)))
            rows.append(row)

output = RESULTS / "fft_features.csv"

with open(output, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["filename", "split", "label"] + FEATURE_NAMES
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nExtraction completed!")
print("Total images:", len(rows))
print("Saved to:", output)
