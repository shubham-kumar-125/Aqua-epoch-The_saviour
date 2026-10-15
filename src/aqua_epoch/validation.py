"""Validation shared by the REST layer and tests."""
from __future__ import annotations

import math
from typing import Any

from .config import FEATURE_COLUMNS, USER_INPUT_FEATURES, USER_INPUT_SPECS


def validate_payload(
    payload: Any,
    feature_ranges: dict[str, dict[str, float]],
) -> tuple[dict[str, float], dict[str, str], list[str]]:
    """Return cleaned data, field errors, and non-blocking range warnings.

    The public form requires three easy-to-measure metrics. A REST client may
    additionally supply any of the remaining chemistry fields for a more
    specific full-panel prediction; those values are validated here as well.
    The caller fills only genuinely unentered model inputs with disclosed
    defaults.
    """
    if not isinstance(payload, dict):
        return {}, {"body": "JSON object expected."}, []

    errors: dict[str, str] = {}
    cleaned: dict[str, float] = {}
    warnings: list[str] = []
    for feature in USER_INPUT_FEATURES:
        raw_value = payload.get(feature)
        if raw_value is None or raw_value == "":
            errors[feature] = "This laboratory metric is required."
            continue
        if isinstance(raw_value, bool):
            errors[feature] = "Enter a numeric value, not true/false."
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            errors[feature] = "Enter a numeric value."
            continue
        if not math.isfinite(value):
            errors[feature] = "Enter a finite number."
            continue
        spec = USER_INPUT_SPECS[feature]
        if value < spec["min"] or value > spec["max"]:
            errors[feature] = (
                f"Enter a realistic value from {spec['min']:g} to {spec['max']:g} {spec['unit']}."
            )
            continue

        cleaned[feature] = value
        range_info = feature_ranges.get(feature, {})
        lower = range_info.get("p01", range_info.get("min"))
        upper = range_info.get("p99", range_info.get("max"))
        if lower is not None and upper is not None and (value < lower or value > upper):
            warnings.append(
                f"{feature} is outside the central 98% of the model's training distribution "
                f"({lower:.3g}–{upper:.3g}); interpret this extrapolation carefully."
            )

    optional_features = tuple(feature for feature in FEATURE_COLUMNS if feature not in USER_INPUT_FEATURES)
    for feature in optional_features:
        if feature not in payload or payload[feature] is None or payload[feature] == "":
            continue
        raw_value = payload[feature]
        if isinstance(raw_value, bool):
            errors[feature] = "Enter a numeric value, not true/false."
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            errors[feature] = "Enter a numeric value."
            continue
        if not math.isfinite(value):
            errors[feature] = "Enter a finite number."
            continue
        if value < 0:
            errors[feature] = "Enter a non-negative chemical concentration."
            continue

        # The public three-field form has explicit BIS-oriented limits. For an
        # optional laboratory field, use a deliberately broad data-informed
        # ceiling to reject obvious unit mistakes while still warning, rather
        # than rejecting, a valid value just outside the training distribution.
        range_info = feature_ranges.get(feature, {})
        observed_maximum = range_info.get("max")
        if observed_maximum is not None and value > max(float(observed_maximum) * 5, 1.0):
            errors[feature] = (
                f"Enter a realistic value no greater than {max(float(observed_maximum) * 5, 1.0):.3g}."
            )
            continue
        cleaned[feature] = value
        lower = range_info.get("p01", range_info.get("min"))
        upper = range_info.get("p99", range_info.get("max"))
        if lower is not None and upper is not None and (value < lower or value > upper):
            warnings.append(
                f"{feature} is outside the central 98% of the model's training distribution "
                f"({lower:.3g}–{upper:.3g}); interpret this extrapolation carefully."
            )

    unexpected = sorted(set(payload) - set(FEATURE_COLUMNS))
    if unexpected:
        warnings.append("Ignored unexpected field(s): " + ", ".join(unexpected) + ".")
    return cleaned, errors, warnings
