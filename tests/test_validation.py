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
        payload["Iron"] = -0.1
        payload["Turbidity"] = "not-a-number"
        del payload["Hardness"]
        _, errors, _ = validate_payload(payload, ranges())
        self.assertIn("Iron", errors)
        self.assertIn("Turbidity", errors)
        self.assertIn("Hardness", errors)

    def test_flags_extrapolation_without_rejecting_it(self) -> None:
        payload = {"Turbidity": 4, "Hardness": 180, "TDS": 450}
        payload["TDS"] = 1000
        _, errors, warnings = validate_payload(payload, ranges())
        self.assertEqual(errors, {})
        self.assertTrue(any("TDS" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
