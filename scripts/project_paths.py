
from pathlib import Path

# Our GitHub project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Original CNNDetection repository
CNN_ROOT = PROJECT_ROOT.parent / "CNNDetection"

# Project directories
DATA = PROJECT_ROOT / "data"
RESULTS = PROJECT_ROOT / "results"
MODELS = PROJECT_ROOT / "models"
FIGURES = RESULTS / "figures"

# Original CNNDetection pretrained weights
WEIGHTS = CNN_ROOT / "weights" / "blur_jpg_prob0.5.pth"

