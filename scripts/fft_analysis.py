
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Dossiers du projet
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "mini_project" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

images = {
    "Real": ROOT / "examples" / "real.png",
    "Fake": ROOT / "examples" / "fake.png",
}

def analyze_fft(path):
    # Convert image to grayscale and use the same size
    image = Image.open(path).convert("L")
    image = image.resize((256, 256))
    pixels = np.asarray(image, dtype=np.float64)

    # Remove average brightness
    pixels = pixels - pixels.mean()

    # Compute 2D Fourier transform
    fft = np.fft.fft2(pixels)

    # Move low frequencies to the center
    fft_shifted = np.fft.fftshift(fft)

    # Compute magnitude spectrum
    magnitude = np.abs(fft_shifted)
    spectrum = np.log1p(magnitude)

    # Compute average magnitude for each frequency radius
    height, width = magnitude.shape
    y, x = np.indices((height, width))
    radius = np.sqrt(
        (x - width // 2) ** 2 +
        (y - height // 2) ** 2
    ).astype(int)

    radial_sum = np.bincount(
        radius.ravel(),
        weights=magnitude.ravel()
    )
    radial_count = np.bincount(radius.ravel())
    radial_profile = radial_sum / np.maximum(radial_count, 1)

    return pixels, spectrum, radial_profile


results = {}

for name, path in images.items():
    results[name] = analyze_fft(path)

# Display original grayscale images and their FFT spectra
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for row, (name, data) in enumerate(results.items()):
    pixels, spectrum, profile = data

    axes[row, 0].imshow(pixels, cmap="gray")
    axes[row, 0].set_title(f"{name} image")
    axes[row, 0].axis("off")

    axes[row, 1].imshow(spectrum, cmap="gray")
    axes[row, 1].set_title(f"{name} - FFT spectrum")
    axes[row, 1].axis("off")

plt.tight_layout()
plt.savefig(RESULTS / "fft_spectra.png", dpi=150)
plt.close()

# Compare frequency distributions
plt.figure(figsize=(9, 5))

for name, (_, _, profile) in results.items():
    plt.plot(profile[:128], label=name)

plt.xlabel("Spatial frequency radius")
plt.ylabel("Average FFT magnitude")
plt.title("Frequency spectrum comparison")
plt.yscale("log")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS / "fft_comparison.png", dpi=150)
plt.close()

print("FFT analysis completed.")
print("Results saved in:", RESULTS)
