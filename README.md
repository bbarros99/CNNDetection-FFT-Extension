
# CNNDetection + FFT Extension

Digital Forensics mini-project  
University of Geneva — Master in Computer Science  
Benjamin Barros — October 2026

## 1. Project overview

This project is based on the paper:

**CNN-Generated Images Are Surprisingly Easy to Spot... for Now**  
Wang et al., CVPR 2020

Original repository:
https://github.com/PeterWang512/CNNDetection

The goal of this mini-project is to test the original CNN detector and try to improve its performance using information from the Fourier transform.

The proposed method extracts eight frequency-based features from each image and combines them with the original CNN prediction using logistic regression.

The original CNN model is not retrained.

## 2. Dataset

The experiment uses 200 images from Tiny-GenImage:

https://huggingface.co/datasets/TheKernel01/Tiny-GenImage

The dataset contains:

| Split | Real | Fake | Total |
|---|---:|---:|---:|
| Train | 60 | 60 | 120 |
| Validation | 20 | 20 | 40 |
| Test | 20 | 20 | 40 |

Images come from several AI generator families.

The same generator families can appear in different splits, so this experiment does not measure generalization to completely unseen generators.

The dataset images are not included in this repository.

## 3. Proposed improvement

The project compares four methods:

1. Original CNNDetection with threshold 0.5.
2. CNNDetection with a threshold adapted on validation data.
3. A classifier trained on eight FFT-based features.
4. A fusion model combining CNN and FFT scores.

The FFT features describe the distribution of image energy across spatial frequencies and the slope of the frequency spectrum.

The FFT and fusion classifiers use logistic regression.

## 4. Experimental results

Results on the 40-image test set:

| Method | Accuracy | Average Precision | ROC AUC |
|---|---:|---:|---:|
| Original CNN | 52.5% | 0.7363 | 0.7325 |
| CNN adapted threshold | 65.0% | 0.7363 | 0.7325 |
| FFT classifier | 62.5% | 0.7753 | 0.7175 |
| CNN + FFT fusion | 72.5% | 0.8110 | 0.7775 |

The CNN + FFT fusion obtained the best performance on this small test set.

However, the results are preliminary. The sample size is small, and the experiment does not establish that the method works better on all AI-generated images.

## 5. Installation

Python 3 is required.

First, clone the original CNNDetection repository:

```bash
git clone https://github.com/PeterWang512/CNNDetection.git
```

Then clone this extension next to it:

```bash
git clone https://github.com/bbarros99/CNNDetection-FFT-Extension.git
cd CNNDetection-FFT-Extension
```

The two repositories should be in the same parent directory:

```text
project/
├── CNNDetection/
└── CNNDetection-FFT-Extension/
```

Create a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required libraries:

```bash
python -m pip install -r requirements.txt
```

The original CNNDetection pretrained checkpoint is also required.

Download the original `blur_jpg_prob0.5.pth` checkpoint following the instructions in the CNNDetection repository and place it at:

```text
CNNDetection/weights/blur_jpg_prob0.5.pth
```

The checkpoint is not included in this repository.

## 6. Running the experiment

To run the complete pipeline:

```bash
./run_pipeline.sh
```

Alternatively, execute the scripts individually:

```bash
python scripts/download_dataset.py
python scripts/extract_fft_features.py
python scripts/train_fft_classifier.py

python scripts/evaluate_cnn.py --split train
python scripts/evaluate_cnn.py --split validation

python scripts/analyze_cnn_threshold.py
python scripts/fusion.py
python scripts/final_evaluation.py

python scripts/generate_figures.py
python scripts/fft_analysis.py
```

The experiment produces evaluation tables and figures in the `results/` directory.

The threshold for the adapted CNN baseline was selected on validation data. The test set was kept separate from this selection.

## 7. Repository structure

```text
scripts/          Python scripts for the experiments
results/          Evaluation results and figures
data/             Downloaded images (not included in Git)
models/           Trained classifiers (not included in Git)
requirements.txt  Python dependencies
run_pipeline.sh   Full experiment pipeline
```

## 8. Limitations

- Only 200 images were used.
- The final test contains 40 images.
- Generator families are not completely separated between train and test.
- Image resolution, compression, and dataset artifacts may influence FFT features.
- The experiment does not test the latest image generators.
- Results may change with another dataset or environment.

## 9. References

Wang et al. (2020). *CNN-Generated Images Are Surprisingly Easy to Spot... for Now*. CVPR.

Original implementation:
https://github.com/PeterWang512/CNNDetection

Dataset:
https://huggingface.co/datasets/TheKernel01/Tiny-GenImage
