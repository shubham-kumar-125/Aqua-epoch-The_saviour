"""Small REST API and static-file server for the Aqua-Epoch dashboard."""
from __future__ import annotations

import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import pandas as pd

from .config import (
    ARTIFACTS_DIR,
    BIS_REFERENCE,
    CLASSIFICATION_FEATURE_COLUMNS,
    DISCLAIMER,
    MODEL_FEATURE_COLUMNS,
    MODELS_DIR,
    REGRESSION_FEATURE_COLUMNS,
    SAFETY_UNCERTAINTY_BAND,
    USER_INPUT_FEATURES,
    USER_INPUT_SPECS,
    WEB_DIR,
)
from .validation import validate_payload

# Some sandboxed macOS environments do not expose physical-core metadata to
# joblib. This stable default avoids a benign warning when the RF predicts.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")


class RequestValidationError(ValueError):
    def __init__(self, errors: dict[str, str], warnings: list[str]) -> None:
        super().__init__("Invalid prediction request")
        self.errors = errors
        self.warnings = warnings


class ModelService:
    """Loads model artifacts once and exposes a typed prediction operation."""

    def __init__(self) -> None:
        metadata_path = ARTIFACTS_DIR / "metadata.json"
        classifier_path = MODELS_DIR / "potability_classifier.joblib"
        regressor_path = MODELS_DIR / "ph_regressor.joblib"
        if not all(path.is_file() for path in (metadata_path, classifier_path, regressor_path)):
            raise FileNotFoundError("Model artifacts are missing. Run `python run.py train` before serving.")
        self.metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.classifier = joblib.load(classifier_path)
        self.regressor = joblib.load(regressor_path)

    def info(self) -> dict[str, Any]:
        """Return front-end-safe metadata and recorded experiment metrics."""
        return {
            "project": self.metadata["project"],
            "model_version": self.metadata["model_version"],
            "trained_on": self.metadata["trained_on"],
            "feature_columns": self.metadata["feature_columns"],
            "model_feature_columns": self.metadata.get(
                "model_feature_columns",
                list(MODEL_FEATURE_COLUMNS),
            ),
            "defaulted_feature_columns": self.metadata.get(
                "defaulted_feature_columns",
                [],
            ),
            "classification_feature_columns": self.metadata.get("classification_feature_columns", list(CLASSIFICATION_FEATURE_COLUMNS)),
            "regression_feature_columns": self.metadata.get("regression_feature_columns", list(REGRESSION_FEATURE_COLUMNS)),
            "user_input_features": list(USER_INPUT_FEATURES),
            "user_input_specs": USER_INPUT_SPECS,
            "limits_reference": BIS_REFERENCE,
            "feature_ranges": self.metadata["feature_ranges"],
            "defaults": self.metadata["defaults"],
            "classification": self.metadata["classification"],
            "regression": self.metadata["regression"],
            "disclaimer": DISCLAIMER,
        }

    def predict(self, payload: Any) -> dict[str, Any]:
        cleaned, errors, warnings = validate_payload(payload, self.metadata["feature_ranges"])
        if errors:
            raise RequestValidationError(errors, warnings)
        model_inputs = {
            feature: cleaned.get(feature, float(self.metadata["defaults"][feature]))
            for feature in MODEL_FEATURE_COLUMNS
        }
        defaulted_features = [
            feature for feature in MODEL_FEATURE_COLUMNS if feature not in cleaned
        ]
        if defaulted_features:
            default_summary = ", ".join(
                f"{feature}={model_inputs[feature]:g}" for feature in defaulted_features
            )
            warnings.append(
                "The website collects only Turbidity, Hardness, and TDS. Training-set "
                f"medians were used for missing model inputs: {default_summary}. "
                "The binary classification is a model estimate based on substituted values "
                "and may be less reliable than a result using measured values."
            )
        classification_frame = pd.DataFrame(
            [model_inputs],
            columns=CLASSIFICATION_FEATURE_COLUMNS,
        )
        probabilities = self.classifier.predict_proba(classification_frame)[0]
        classes = list(self.classifier.classes_)
        safe_probability = float(probabilities[classes.index(1)])
        unsafe_probability = float(probabilities[classes.index(0)])
        threshold = float(self.metadata["classification"]["metrics"]["threshold"])
        uncertainty_policy = self.metadata["classification"].get(
            "uncertainty_policy",
            {
                "lower_safe_probability": SAFETY_UNCERTAINTY_BAND[0],
                "upper_safe_probability": SAFETY_UNCERTAINTY_BAND[1],
            },
        )
        lower_threshold = float(uncertainty_policy["lower_safe_probability"])
        upper_threshold = float(uncertainty_policy["upper_safe_probability"])
        safety_code = int(safe_probability >= threshold)
        safety_label = "Safe" if safety_code == 1 else "Unsafe"
        if lower_threshold < safe_probability < upper_threshold:
            warnings.append(
                f"The model's Safe probability ({safe_probability:.1%}) is within its "
                f"{lower_threshold:.0%}–{upper_threshold:.0%} uncertainty band, but the "
                f"displayed binary estimate uses the configured {threshold:.0%} threshold. "
                "Treat the estimate cautiously and confirm with laboratory testing."
            )
        confusion = self.metadata["classification"]["confusion_matrix"]
        held_out_unsafe = sum(confusion[0])
        held_out_false_safe = confusion[0][1]
        safety_prediction_caution = held_out_unsafe > 0 and held_out_false_safe > 0
        if safety_prediction_caution:
            false_safe_rate = held_out_false_safe / held_out_unsafe
            warnings.append(
                f"At the standard 50% decision threshold, held-out evaluation marked "
                f"{held_out_false_safe} of {held_out_unsafe} unsafe samples as Safe "
                f"(false-safe rate {false_safe_rate:.1%}); "
                "do not use this prediction alone to determine water safety."
            )
        unused_metrics = [feature for feature in cleaned if feature not in MODEL_FEATURE_COLUMNS]
        if unused_metrics:
            warnings.append(
                "The current models do not use optional measurement(s): "
                + ", ".join(unused_metrics)
                + "."
            )
        regression_frame = pd.DataFrame([model_inputs], columns=REGRESSION_FEATURE_COLUMNS)
        predicted_ph = float(self.regressor.predict(regression_frame)[0])
        regression = self.metadata["regression"]
        ph_prediction_caution = regression.get("signal_status") == "weak"
        if ph_prediction_caution:
            r2 = float(regression["metrics"]["r2"])
            warnings.append(
                f"The pH model has weak held-out predictive signal (R² {r2:.3f}); "
                "treat this as an experimental estimate and verify pH with laboratory testing."
            )
        if not 0 <= predicted_ph <= 14:
            warnings.append(
                "The predicted pH lies outside the physical 0–14 interval; verify the reading with laboratory testing."
            )
        guidance = []
        for feature in USER_INPUT_FEATURES:
            spec = USER_INPUT_SPECS[feature]
            value = cleaned[feature]
            if value <= spec["acceptable_limit"]:
                band = "Acceptable guidance"
                status = "acceptable"
            elif value <= spec["permissible_limit"]:
                band = "Permissible only when no alternate source is available"
                status = "permissible"
            else:
                band = "Above the listed permissible guidance"
                status = "above_limit"
            guidance.append(
                {
                    "metric": spec["label"],
                    "value": value,
                    "unit": spec["unit"],
                    "status": status,
                    "guidance": band,
                }
            )
        return {
            "safety_label": safety_label,
            "safety_code": safety_code,
            "safety_prediction_caution": safety_prediction_caution,
            "safe_probability": round(safe_probability, 6),
            "unsafe_probability": round(unsafe_probability, 6),
            "classification_threshold": threshold,
            "safety_uncertainty_band": [lower_threshold, upper_threshold],
            "predicted_ph": round(predicted_ph, 4),
            "ph_input_features": list(REGRESSION_FEATURE_COLUMNS),
            "ph_prediction_caution": ph_prediction_caution,
            "ph_within_physical_range": 0 <= predicted_ph <= 14,
            "model_inputs": model_inputs,
            "entered_metrics": cleaned,
            "unused_metrics": unused_metrics,
            "defaulted_features": defaulted_features,
            "input_guidance": guidance,
            "warnings": warnings,
            "model_version": self.metadata["model_version"],
            "disclaimer": DISCLAIMER,
        }


class AquaEpochHandler(BaseHTTPRequestHandler):
    server_version = "AquaEpoch/1.0"
    service: ModelService | None = None

    def log_message(self, format: str, *args: object) -> None:
        # Short operational log without request bodies or laboratory values.
        print(f"[{self.log_date_time_string()}] {self.address_string()} {format % args}")

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, requested: str) -> None:
        relative = "index.html" if requested in {"", "/"} else requested.lstrip("/")
        candidate = (WEB_DIR / relative).resolve()
        try:
            candidate.relative_to(WEB_DIR.resolve())
        except ValueError:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return
        if not candidate.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return
        content = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type + ("; charset=utf-8" if content_type.startswith("text/") or content_type == "application/javascript" else ""))
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/health":
            if self.service is None:
                self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"status": "not_ready"})
                return
            self._send_json(HTTPStatus.OK, {"status": "ok", "models": "loaded", "model_version": self.service.metadata["model_version"]})
            return
        if path == "/api/model-info":
            if self.service is None:
                self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Models are not loaded."})
                return
            self._send_json(HTTPStatus.OK, self.service.info())
            return
        self._send_file(path)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/predict":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return
        if self.service is None:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Models are not loaded."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= 32_768:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Request body must be between 1 and 32768 bytes."})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Malformed JSON body."})
            return
        try:
            result = self.service.predict(payload)
        except RequestValidationError as error:
            self._send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "Validation failed.", "fields": error.errors, "warnings": error.warnings})
            return
        except Exception:
            # Avoid exposing internal implementation details or model paths.
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Prediction could not be completed."})
            return
        self._send_json(HTTPStatus.OK, result)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the local HTTP server. Interrupt with Ctrl+C."""
    AquaEpochHandler.service = ModelService()
    server = ThreadingHTTPServer((host, port), AquaEpochHandler)
    print(f"Aqua-Epoch dashboard: http://{host}:{port}")
    print("API health: /api/health | model metadata: /api/model-info")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    serve()
