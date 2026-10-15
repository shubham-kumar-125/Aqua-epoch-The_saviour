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

# Task A follows the compact six-feature model in main2.ipynb.
CLASSIFICATION_FEATURE_COLUMNS = (
    "Turbidity",
    "Hardness",
    "TDS",
    "Fluoride",
    "Nitrate",
    "Conductivity",
)

# Task B uses every available non-target chemical measurement, as required by
# the challenge brief. The classifier's predicted Safe probability is appended
# at training/inference time as a leakage-safe derived model feature.
REGRESSION_BASE_FEATURE_COLUMNS = (
    "Hardness",
    "TDS",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
    "Calcium",
    "Magnesium",
    "Sodium",
    "Potassium",
    "Chloride",
    "Nitrate",
    "Fluoride",
    "Iron",
    "Alkalinity",
)
REGRESSION_DERIVED_FEATURE = "predicted_safe_probability"
REGRESSION_FEATURE_COLUMNS = REGRESSION_BASE_FEATURE_COLUMNS + (REGRESSION_DERIVED_FEATURE,)

# Kept as the public physical-chemistry input schema used by validation, model
# metadata, and the API. It deliberately excludes the derived probability.
FEATURE_COLUMNS = REGRESSION_BASE_FEATURE_COLUMNS

# The competition's web requirement calls for a simple user-facing form. The
# public form collects three measurements, while the pH model uses the full
# chemical panel. The API fills the other 14 laboratory inputs with training-set
# medians and discloses that approximation in every response. The limits below
# validate physically plausible *form input*, while the BIS values are guidance
# tooltips rather than the model's decision rule.
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
        "max": 10000.0,
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
