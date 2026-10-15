import unittest
from pathlib import Path

from aqua_epoch.api import ModelService, RequestValidationError
from aqua_epoch.config import ARTIFACTS_DIR, FEATURE_COLUMNS, USER_INPUT_FEATURES


@unittest.skipUnless((ARTIFACTS_DIR / "metadata.json").is_file(), "run training before model-service smoke test")
class ModelServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = ModelService()

    def test_predict_returns_both_tasks(self) -> None:
        payload = {"Turbidity": 1, "Hardness": 180, "TDS": 450}
        result = self.service.predict(payload)
        self.assertIn(result["safety_code"], (0, 1))
        self.assertGreaterEqual(result["safe_probability"], 0)
        self.assertLessEqual(result["safe_probability"], 1)
        self.assertIsInstance(result["predicted_ph"], float)
        self.assertEqual(len(result["defaulted_features"]), 14)
        self.assertEqual(len(result["input_guidance"]), 3)

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


if __name__ == "__main__":
    unittest.main()
