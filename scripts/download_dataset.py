
from datasets import load_dataset
from pathlib import Path
import hashlib
import csv
import io

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

DATASET = "TheKernel01/Tiny-GenImage"

GENERATORS = {
    0: "Real",
    1: "ADM",
    2: "BigGAN",
    3: "GLIDE",
    4: "Midjourney",
    5: "SD14",
    6: "SD15",
    7: "VQDM",
    8: "Wukong",
}

# Number of images per class
TARGETS = {
    "train": 60,
    "validation": 20,
    "test": 20,
}

metadata = []
seen_hashes = set()


def collect(source_split, destination, target):
    print(f"\nLoading {source_split} for {destination}...")

    dataset = load_dataset(
        DATASET,
        split=source_split,
        streaming=True
    )

    counts = {"real": 0, "fake": 0}

    for row in dataset:
        label = int(row["label"])
        category = "real" if label == 0 else "fake"

        if counts[category] >= target:
            if all(n >= target for n in counts.values()):
                break
            continue

        image = row["image"].convert("RGB")

        # Avoid storing exactly identical images twice
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_hash = hashlib.sha256(image_bytes).hexdigest()

        if image_hash in seen_hashes:
            continue

        seen_hashes.add(image_hash)

        folder = DATA / destination / category
        folder.mkdir(parents=True, exist_ok=True)

        index = counts[category]
        filename = f"{category}_{index:03d}.png"
        path = folder / filename

        path.write_bytes(image_bytes)

        metadata.append({
            "split": destination,
            "filename": str(path.relative_to(DATA)),
            "label": label,
            "generator": GENERATORS.get(
                int(row["generator"]), "Unknown"
            ),
            "sha256": image_hash,
        })

        counts[category] += 1

        if all(n >= target for n in counts.values()):
            break

    print("Saved:", counts)

    if not all(n == target for n in counts.values()):
        raise RuntimeError(
            f"Not enough images for {destination}: {counts}"
        )


# Training examples come from the official training split
collect("train", "train", TARGETS["train"])

# Validation and final test use separate images
# from the official validation split
collect("validation", "validation", TARGETS["validation"])
collect("validation", "test", TARGETS["test"])

# Save metadata
with open(DATA / "metadata.csv", "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "split", "filename", "label",
            "generator", "sha256"
        ]
    )
    writer.writeheader()
    writer.writerows(metadata)

print("\nDownload completed!")
print("Total images:", len(metadata))
print("Dataset location:", DATA)
