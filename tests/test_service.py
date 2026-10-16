import unittest
from pathlib import Path

from aqua_epoch.api import ModelService, RequestValidationError
from aqua_epoch.config import (
    ARTIFACTS_DIR,
    FEATURE_COLUMNS,
    REGRESSION_FEATURE_COLUMNS,
    USER_INPUT_FEATURES,
)


@unittest.skipUnless((ARTIFACTS_DIR / "metadata.json").is_file(), "run training before model-service smoke test")
class ModelServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = ModelService()

    def test_predict_returns_both_tasks(self) -> None:
        payload = {"Turbidity": 1, "Hardness": 180, "TDS": 450}
        result = self.service.predict(payload)
        self.assertIn(result["safety_code"], (0, 1, None))
        self.assertEqual(
            result["safety_prediction_caution"],
            sum(self.service.metadata["classification"]["confusion_matrix"][0]) > 0
            and self.service.metadata["classification"]["confusion_matrix"][0][1] > 0,
        )
        self.assertGreaterEqual(result["safe_probability"], 0)
        self.assertLessEqual(result["safe_probability"], 1)
        self.assertIsInstance(result["predicted_ph"], float)
        self.assertEqual(
            result["ph_prediction_caution"],
            self.service.metadata["regression"]["signal_status"] == "weak",
        )
        if result["ph_prediction_caution"]:
            self.assertTrue(any("weak held-out predictive signal" in warning for warning in result["warnings"]))
        expected_defaulted = set(self.service.info()["model_feature_columns"]) - set(USER_INPUT_FEATURES)
        self.assertEqual(set(result["defaulted_features"]), expected_defaulted)
        for feature in expected_defaulted:
            self.assertEqual(result["model_inputs"][feature], self.service.metadata["defaults"][feature])
        self.assertEqual(result["ph_input_features"], list(REGRESSION_FEATURE_COLUMNS))
        self.assertTrue(any("Training-set medians were used" in warning for warning in result["warnings"]))
        self.assertEqual(len(result["input_guidance"]), 3)
        self.assertTrue(any("outside the central 98%" in warning and "TDS" in warning for warning in result["warnings"]))

    def test_screenshot_input_uses_training_medians_for_unentered_model_features(self) -> None:
        result = self.service.predict(
            {"Turbidity": 2.86, "Hardness": 212.12, "TDS": 15569.18}
        )
        self.assertEqual(result["safety_label"], "Inconclusive")
        self.assertIsNone(result["safety_code"])
        self.assertEqual(set(result["defaulted_features"]), {"Fluoride", "Nitrate", "Conductivity"})
        self.assertEqual(result["safety_uncertainty_band"], [0.3, 0.7])
        self.assertTrue(
            any("requires all six model measurements" in warning for warning in result["warnings"])
        )
        self.assertTrue(result["ph_prediction_caution"])

    def test_predict_rejects_incomplete_panel(self) -> None:
        payload = {"Turbidity": 1, "Hardness": 100}
        with self.assertRaises(RequestValidationError):
            self.service.predict(payload)

    def test_predict_accepts_an_optional_full_chemistry_panel(self) -> None:
        payload = {
            feature: self.service.metadata["defaults"][feature]
            for feature in FEATURE_COLUMNS
        }
        payload.update({"Turbidity": 1, "Hardness": 180, "TDS": 450})
        result = self.service.predict(payload)
        self.assertEqual(result["defaulted_features"], [])
        self.assertEqual(set(result["entered_metrics"]), set(FEATURE_COLUMNS))
        self.assertEqual(
            set(result["unused_metrics"]),
            {"Alkalinity"},
        )
        self.assertTrue(
            any("do not use optional measurement" in warning for warning in result["warnings"])
        )

    def test_both_models_use_the_aqua_epoch_six_feature_panel(self) -> None:
        self.assertEqual(
            self.service.info()["classification_feature_columns"],
            ["Turbidity", "Hardness", "TDS", "Fluoride", "Nitrate", "Conductivity"],
        )
        self.assertEqual(
            REGRESSION_FEATURE_COLUMNS,
            ("Turbidity", "Hardness", "TDS", "Fluoride", "Nitrate", "Conductivity"),
        )
        self.assertIn("Alkalinity", FEATURE_COLUMNS)
        self.assertNotIn("Potability", FEATURE_COLUMNS)
        self.assertEqual(self.service.info()["regression_feature_columns"], list(REGRESSION_FEATURE_COLUMNS))

    def test_model_info_exposes_updated_tds_form_limit(self) -> None:
        self.assertEqual(self.service.info()["user_input_specs"]["TDS"]["min"], 0.0)
        self.assertEqual(self.service.info()["user_input_specs"]["TDS"]["max"], 50000.0)


if __name__ == "__main__":
    unittest.main()
