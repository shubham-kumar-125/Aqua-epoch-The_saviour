# Aqua-Epoch — Water Intelligence Platform

A complete, reproducible submission for the Aqua-Epoch machine-learning and web-development challenge. It trains two independent models from the supplied BIS water-quality dataset and exposes them through a validated REST API and browser dashboard.

## Delivered

- **Task A — Classification:** restores the Aqua-Epoch Random Forest trained on Turbidity, Hardness, TDS, Fluoride, Nitrate, and Conductivity.
- **Task B — Regression:** restores the Aqua-Epoch Random Forest pH estimator using the same six measurements.
- **Validated REST API:** `GET /api/health`, `GET /api/model-info`, and `POST /api/predict`.
- **Web application:** responsive form, client/server validation, sample loading, and result cards.
- **Round 1 materials:** a reproducible PDF technical deck, Markdown report, figures, evaluation metrics, and demo script generated from the trained artifacts.

## Important scientific limitation

This is a decision-support prototype, not a drinking-water certification system. A model prediction must never replace BIS testing, local regulation, or laboratory confirmation. The supplied dataset has major quality concerns: pH labels are missing or physically out of range, and the cleaned TDS median is about 20,773 mg/L (BIS acceptable guidance is 500 mg/L). Verify the source, units, and labels before relying on any result.

The dashboard continues to ask for **Turbidity, Total Hardness, and TDS**. The restored Aqua-Epoch models require three additional chemistry features; when those values are not supplied, the backend fills Fluoride, Nitrate, and Conductivity with medians calculated from the training partition. The API response lists these defaulted fields and their applied values. Predictions with median-filled inputs can be less reliable than predictions with all six measurements entered, so the live safety label is **Inconclusive** whenever any model feature is median-filled; with all six measured values, scores between 30–70% are also inconclusive. The pH model has weak held-out signal even with all six features; treat its numeric output as experimental, not reliable.

API clients may additionally provide Fluoride, Nitrate, Conductivity, and Alkalinity in the same `POST /api/predict` JSON body for compatibility; the current models ignore these optional values and disclose that in the response. Inputs outside the central 98% of the training distribution are flagged.

## Model selection

`python run.py train` restores the Random Forest architectures from `notebooks/aqua_epoch_simple_model.ipynb`, trained on all six measured features. The held-out metrics shown for deployment replace the three unentered chemistry readings with training-partition medians, matching the website's input policy; full-feature cross-validation results are separately marked in the generated metadata. The pH model does not consume classifier predictions.

The three core measurements alone cannot reproduce the simple notebook's all-six-feature evaluation. Filling missing chemistry with a median is a product choice, not a measured value, and the UI warns about the lower-confidence input policy. The pH model's weak held-out R² is a data-quality finding, not a reason to claim high accuracy. The project reports held-out MAE, RMSE, R², and cross-validation results transparently.

## Quick start

The supplied dataset has already been copied into `data/`. From this project directory:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py train
python run.py serve
```

Open <http://127.0.0.1:8000>. Training creates models, plots, the technical deck, and all result files. To train from a different CSV:

```bash
python run.py train --data /absolute/path/to/Aqua_Epoch_BIS_Water_Quality_Dataset.csv
```

## Deploy to Render

The repository includes a Render Blueprint (`render.yaml`) and Docker configuration. The Docker image trains the models during its build, so generated model files do not need to be committed.

1. Push this project directory to a GitHub repository.
2. In Render, choose **New → Blueprint**, connect that repository, and apply the `aqua-epoch` service.
3. Wait for the Docker build and deploy to finish, then open the service URL shown in Render.

Render supplies the service port through `PORT`; the container binds to that port and exposes `/api/health` for health checks. The free web service may spin down when idle and take a short time to wake.

Run verification after training:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## API example

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "Turbidity": 1.0,
    "Hardness": 180,
    "TDS": 450
  }'
```

Response fields include `safety_label`, `safe_probability`, `predicted_ph`, validation warnings, the metrics defaulted by the API, model version, and an explicit non-certification disclaimer. See [BIS_LIMITS.md](docs/BIS_LIMITS.md) for the tooltip source and interpretation.

## Project structure

```text
.
├── data/                         # supplied source CSV
├── models/                       # generated, versioned model pipelines
├── artifacts/                    # generated metrics, plots, and metadata
├── docs/                         # generated Round 1 deck and report
├── notebooks/                    # main2 model source and archived experiments
├── src/aqua_epoch/
│   ├── train.py                  # repeatable training + evaluation
│   ├── report.py                 # PDF technical deck generator
│   ├── api.py                    # dependency-light REST server
│   └── validation.py             # shared request validation
├── web/                          # zero-build dashboard
├── tests/                        # smoke tests
└── run.py                        # train, report, and serve commands
```

`main2.ipynb` and the earlier experiments are retained for reference; `python run.py train` is the supported deployment-training entry point for the Aqua-Epoch Random Forest models served by the dashboard.

## Submission checklist

1. Run `python run.py train` once in the final environment.
2. Replace the clearly marked identity placeholders in `docs/team_details.json` (see `docs/TEAM_DETAILS.md`).
3. Run `python run.py report` to refresh `docs/round1_technical_deck.pdf`.
4. Submit that single PDF for Round 1. Use `docs/live_demo_script.md` for Round 2.
5. Do not claim model output is a laboratory or BIS certification.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | liveness and model readiness |
| `GET` | `/api/model-info` | expected metrics, defaults, ranges, and held-out metrics |
| `POST` | `/api/predict` | safety classification and pH regression |

The server intentionally uses Python's standard-library HTTP server so the deployed inference service has a small attack surface and does not require a separate web framework. Model training requires the packages listed in `requirements.txt`.
