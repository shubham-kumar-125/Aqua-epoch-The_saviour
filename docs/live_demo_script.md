# Round 2 live-demo runbook

1. Start the app with `python run.py serve` and open `http://127.0.0.1:8000`.
2. Show `http://127.0.0.1:8000/api/health` to establish that both persisted pipelines are loaded.
3. Open `/api/model-info`; point out the model version, held-out metrics, pH cross-validation results and central training ranges.
4. In the UI select **Load median sample**, then click **Run prediction**. Explain that both current models use the three entered measurements directly.
5. Hover over **ⓘ** to show the BIS guidance. Demonstrate validation by making `Turbidity` negative or entering an unrealistic value; explain that the browser blocks the request and the API repeats the same checks server-side.
6. Restore the sample and submit. Explain that the pH regressor uses the classifier's probability rather than a laboratory label, and that its held-out R² is weak; neither output replaces laboratory testing.
7. Show the generated feature-importance and diagnostic figures in `artifacts/`; explain the three-input classifier selection by training-only cross-validation and the observed false-safe classification errors.
8. Close with limitations: 3,990 missing pH labels, 143 out-of-range pH labels, a cleaned TDS median near 20,773 mg/L, need to verify source units and labels, external validation, calibration, and secure production controls.

Use `python run.py train` as the supported trainer. It compares and trains models on the same three measurements collected by the dashboard; `notebooks/main2.ipynb` remains the original experiment.
