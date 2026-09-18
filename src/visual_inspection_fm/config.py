"""Central paths and constants — import this instead of hardcoding paths elsewhere.

Following the cookiecutter-data-science / Eric Ma gist convention: one config
module, everything else imports from it.
"""

from pathlib import Path

# Project root = two levels up from this file (src/visual_inspection_fm/config.py -> project root)
PROJ_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Decision-routing threshold from the mock prototype (sbi_prototype.py) —
# revisit once calibration.py gives us a principled, data-driven threshold
# instead of a hand-picked constant.
UNCERTAINTY_THRESHOLD = 0.5
DEFECT_SEVERITY_THRESHOLD = 0.5

IMAGE_SIZE = 224  # matches the ResNet18 input size used in modeling/sbi_model.py
