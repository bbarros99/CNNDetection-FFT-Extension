
#!/usr/bin/env bash
set -euo pipefail

# Run from the project root
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

echo "=== Step 1: Dataset ==="
if [ ! -f data/metadata.csv ]; then
    "$PYTHON" scripts/download_dataset.py
else
    echo "Dataset already present."
fi

echo "=== Step 2: FFT features ==="
"$PYTHON" scripts/extract_fft_features.py

echo "=== Step 3: FFT classifier ==="
"$PYTHON" scripts/train_fft_classifier.py

echo "=== Step 4: Original CNN scores ==="
"$PYTHON" scripts/evaluate_cnn.py --split train
"$PYTHON" scripts/evaluate_cnn.py --split validation

echo "=== Step 5: Threshold analysis ==="
"$PYTHON" scripts/analyze_cnn_threshold.py

echo "=== Step 6: CNN + FFT fusion ==="
"$PYTHON" scripts/fusion.py

echo "=== Step 7: Final evaluation ==="
"$PYTHON" scripts/final_evaluation.py

echo "=== Step 8: Figures ==="
"$PYTHON" scripts/generate_figures.py
"$PYTHON" scripts/fft_analysis.py

echo "=== Pipeline completed! ==="
