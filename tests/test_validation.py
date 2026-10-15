import unittest

from aqua_epoch.config import USER_INPUT_FEATURES
from aqua_epoch.validation import validate_payload


def ranges() -> dict[str, dict[str, float]]:
    return {feature: {"min": 1.0, "max": 100.0} for feature in USER_INPUT_FEATURES}


class ValidationTests(unittest.TestCase):
    def test_accepts_complete_nonnegative_panel(self) -> None:
        payload = {"Turbidity": 4, "Hardness": 180, "TDS": 450}
        cleaned, errors, warnings = validate_payload(payload, ranges())
        self.assertEqual(errors, {})
        self.assertEqual(len(cleaned), len(USER_INPUT_FEATURES))
        self.assertIsInstance(warnings, list)

    def test_rejects_missing_negative_and_nonfinite_values(self) -> None:
        payload = {"Turbidity": 4, "Hardness": 180, "TDS": 450}
        payload["Alkalinity"] = -0.1
        payload["Turbidity"] = "not-a-number"
        del payload["Hardness"]
        _, errors, _ = validate_payload(payload, ranges())
        self.assertIn("Alkalinity", errors)
        self.assertIn("Turbidity", errors)
        self.assertIn("Hardness", errors)

    def test_flags_extrapolation_without_rejecting_it(self) -> None:
        payload = {"Turbidity": 4, "Hardness": 180, "TDS": 450}
        payload["TDS"] = 1000
        _, errors, warnings = validate_payload(payload, ranges())
        self.assertEqual(errors, {})
        self.assertTrue(any("TDS" in warning for warning in warnings))

    def test_flags_tail_values_inside_observed_minimum_and_maximum(self) -> None:
        feature_ranges = ranges()
        feature_ranges["TDS"] = {"min": 1.0, "max": 100.0, "p01": 5.0, "p99": 90.0}
        payload = {"Turbidity": 4, "Hardness": 180, "TDS": 2}
        _, errors, warnings = validate_payload(payload, feature_ranges)
        self.assertEqual(errors, {})
        self.assertTrue(any("TDS" in warning and "central 98%" in warning for warning in warnings))

    def test_accepts_optional_alkalinity_but_ignores_client_potability(self) -> None:
        payload = {"Turbidity": 4, "Hardness": 180, "TDS": 450, "Alkalinity": 100, "Potability": 0}
        cleaned, errors, warnings = validate_payload(payload, ranges())
        self.assertEqual(errors, {})
        self.assertEqual(cleaned["Alkalinity"], 100)
        self.assertNotIn("Potability", cleaned)
        self.assertTrue(any("Potability" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
