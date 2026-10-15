# Round 2 live-demo runbook

1. Start the app with `python run.py serve` and open `http://127.0.0.1:8000`.
2. Show `http://127.0.0.1:8000/api/health` to establish that both persisted pipelines are loaded.
3. Open `/api/model-info`; point out the fixed model version, held-out metrics, pH cross-validation results, median defaults and central training ranges.
4. In the UI select **Load median sample**, then click **Run prediction**. Explain that the interface asks only for Turbidity, Total Hardness and TDS, as required by the brief, and uses median defaults for the other four measured model inputs.
5. Hover over **ⓘ** to show the BIS guidance. Demonstrate validation by making `Turbidity` negative or entering an unrealistic value; explain that the browser blocks the request and the API repeats the same checks server-side.
6. Restore the sample and submit. Explain that the pH regressor uses the classifier's predicted Potability label rather than a laboratory label, and that its held-out R² is weak; neither output replaces laboratory testing.
7. Show the generated feature-importance and diagnostic figures in `artifacts/`; explain the six-input Random Forest, pH model selection by training-only cross-validation, and false-safe classification errors.
8. Close with limitations: 3,990 missing pH labels, 143 out-of-range pH labels, a cleaned TDS median near 20,773 mg/L, need to verify source units and labels, external validation, calibration, and secure production controls.

Use `python run.py train` as the supported trainer. It reproduces the two deployed model definitions in `notebooks/main2.ipynb`.
