# Round 2 live-demo runbook

1. Start the app with `python run.py serve` and open `http://127.0.0.1:8000`.
2. Show `http://127.0.0.1:8000/api/health` to establish that both persisted pipelines are loaded.
3. Open `/api/model-info`; point out the fixed model version, test metrics, median defaults and central training ranges.
4. In the UI select **Load median sample**, then click **Run prediction**. Explain that the interface asks only for Turbidity, Total Hardness and TDS, as required by the brief.
5. Hover over **ⓘ** to show the BIS guidance. Demonstrate validation by making `Turbidity` negative or entering an unrealistic value; explain that the browser blocks the request and the API repeats the same checks server-side.
6. Restore the sample and submit. Explain that 14 unentered metrics receive clearly disclosed training-median defaults, then explain safety label, probability, pH prediction, and the laboratory-certification disclaimer.
7. Show the generated feature-importance and diagnostic figures in `artifacts/`; defend the model choice, imputation-in-pipeline design, stratified Task A split, and the reported metrics.
8. Close with limitations: submitted-data scope, non-causal feature importance, pH-label quality, need for external validation, calibration and secure production controls.
