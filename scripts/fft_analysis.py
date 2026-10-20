
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from project_paths import DATA, RESULTS

# --------------------------------------------------
# Directories
# --------------------------------------------------

RESULTS.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Select one real and one fake image
# --------------------------------------------------

real_images = sorted(
    (DATA / "train" / "real").glob("*.png")
)

fake_images = sorted(
    (DATA / "train" / "fake").glob("*.png")
)

if not real_images or not fake_images:
    raise FileNotFoundError(
        "Real or fake training images not found. "
        "Please download the dataset first."
    )

images = {
    "Real": real_images[0],
    "Fake": fake_images[0],
}

# --------------------------------------------------
# Fourier analysis
# --------------------------------------------------

def analyze_fft(path):

    # Convert image to grayscale
    image = Image.open(path).convert("L")

    # Use the same size for both images
    image = image.resize((256, 256))

    pixels = np.asarray(
        image,
        dtype=np.float64
    )

    # Remove average brightness
    pixels = pixels - pixels.mean()

    # Compute 2D Fourier transform
    fft = np.fft.fft2(pixels)

    # Move low frequencies to the center
    fft_shifted = np.fft.fftshift(fft)

    # Compute magnitude spectrum
    magnitude = np.abs(fft_shifted)
    spectrum = np.log1p(magnitude)

    # Compute radial frequency profile
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

    radial_count = np.bincount(
        radius.ravel()
    )

    radial_profile = (
        radial_sum /
        np.maximum(radial_count, 1)
    )

    return pixels, spectrum, radial_profile


# --------------------------------------------------
# Analyze both images
# --------------------------------------------------

results = {}

for name, path in images.items():

    print(f"Analyzing {name}: {path}")

    results[name] = analyze_fft(path)

# --------------------------------------------------
# Figure 1: Original images and FFT spectra
# --------------------------------------------------

fig, axes = plt.subplots(
    2, 2,
    figsize=(10, 8)
)

for row, (name, data) in enumerate(results.items()):

    pixels, spectrum, profile = data

    axes[row, 0].imshow(
        pixels,
        cmap="gray"
    )

    axes[row, 0].set_title(
        f"{name} image"
    )

    axes[row, 0].axis("off")

    axes[row, 1].imshow(
        spectrum,
        cmap="gray"
    )

    axes[row, 1].set_title(
        f"{name} - FFT spectrum"
    )

    axes[row, 1].axis("off")

plt.tight_layout()

plt.savefig(
    RESULTS / "fft_spectra.png",
    dpi=150
)

plt.close()

# --------------------------------------------------
# Figure 2: Frequency distributions
# --------------------------------------------------

plt.figure(figsize=(9, 5))

for name, (_, _, profile) in results.items():

    plt.plot(
        profile[:128],
        label=name
    )

plt.xlabel("Spatial frequency radius")
plt.ylabel("Average FFT magnitude")

plt.title(
    "Frequency spectrum comparison"
)

plt.yscale("log")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS / "fft_comparison.png",
    dpi=150
)

plt.close()

print("\nFFT analysis completed!")
print("Results saved in:", RESULTS)
