"""Shared paths and schema constants."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DOCS_DIR = PROJECT_ROOT / "docs"
WEB_DIR = PROJECT_ROOT / "web"

DATASET_FILENAME = "Aqua_Epoch_BIS_Water_Quality_Dataset.csv"
CLASSIFICATION_TARGET = "Potability"
REGRESSION_TARGET = "pH"

# Both Aqua-Epoch Random Forest models were trained on these six measurements.
MODEL_FEATURE_COLUMNS = (
    "Turbidity",
    "Hardness",
    "TDS",
    "Fluoride",
    "Nitrate",
    "Conductivity",
)

CLASSIFICATION_FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS
REGRESSION_FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS

SAFETY_UNCERTAINTY_BAND = (0.30, 0.70)

# Additional chemistry fields remain accepted by the API for compatibility.
FEATURE_COLUMNS = (
    "Turbidity",
    "Hardness",
    "TDS",
    "Fluoride",
    "Nitrate",
    "Conductivity",
    "Alkalinity",
)

# Keep the public form focused on three common measurements. The other model
# features use training-set medians unless an API client supplies them.
USER_INPUT_FEATURES = ("Turbidity", "Hardness", "TDS")
USER_INPUT_SPECS = {
    "Turbidity": {
        "label": "Turbidity",
        "unit": "NTU",
        "min": 0.0,
        "max": 1000.0,
        "acceptable_limit": 1.0,
        "permissible_limit": 5.0,
        "tooltip": "BIS IS 10500:2012: acceptable up to 1 NTU; permissible up to 5 NTU where no alternate source is available.",
    },
    "Hardness": {
        "label": "Total hardness",
        "unit": "mg/L as CaCO₃",
        "min": 0.0,
        "max": 2000.0,
        "acceptable_limit": 200.0,
        "permissible_limit": 600.0,
        "tooltip": "BIS IS 10500:2012: acceptable up to 200 mg/L; permissible up to 600 mg/L where no alternate source is available.",
    },
    "TDS": {
        "label": "Total dissolved solids (TDS)",
        "unit": "mg/L",
        "min": 0.0,
        "max": 50000.0,
        "acceptable_limit": 500.0,
        "permissible_limit": 2000.0,
        "tooltip": "BIS IS 10500:2012: acceptable up to 500 mg/L; permissible up to 2000 mg/L where no alternate source is available.",
    },
}

BIS_REFERENCE = "IS 10500:2012 (Second revision) — drinking-water specification"

DISPLAY_NAMES = {
    "TDS": "Total dissolved solids (TDS)",
    "Organic_carbon": "Organic carbon",
    "Trihalomethanes": "Trihalomethanes",
}

DISCLAIMER = (
    "Prototype decision-support output only. It is not a BIS or laboratory "
    "certificate of water safety; confirm results through accredited testing."
)


def source_data_candidates() -> list[Path]:
    """Locations used by the CLI when --data is not supplied."""
    return [
        DATA_DIR / DATASET_FILENAME,
        PROJECT_ROOT.parent / DATASET_FILENAME,
    ]
