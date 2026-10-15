# Aqua-Epoch — Water Intelligence Platform

A complete, reproducible submission for the Aqua-Epoch machine-learning and web-development challenge. It trains two independent models from the supplied BIS water-quality dataset and exposes them through a validated REST API and browser dashboard.

## Delivered

- **Task A — Classification:** estimates `Safe (1)` / `Unsafe (0)` with probability outputs from a class-balanced Random Forest.
- **Task B — Regression:** estimates continuous pH from all 17 non-pH chemical measurements plus Task A's predicted Safe probability.
- **Validated REST API:** `GET /api/health`, `GET /api/model-info`, and `POST /api/predict`.
- **Web application:** responsive form, client/server validation, sample loading, and result cards.
- **Round 1 materials:** a reproducible PDF technical deck, Markdown report, figures, evaluation metrics, and demo script generated from the trained artifacts.

## Important scientific limitation

This is a decision-support prototype, not a drinking-water certification system. A model prediction must never replace BIS testing, local regulation, or laboratory confirmation. The model is trained only on the supplied dataset; performance and safe operating ranges must be revalidated before operational use.

The dashboard intentionally asks only for **Turbidity, Total Hardness, and TDS**, as required by the brief. Task A uses six features; Task B uses all 17 chemical features plus the classifier's predicted Safe probability. The API fills the 14 unentered chemical measurements with training-set median defaults and discloses this in every response. Treat this simplified demonstration as less specific than a full laboratory-panel prediction.

For the most specific pH result, an API client may also send any or all of the remaining 14 chemistry fields in the same `POST /api/predict` JSON body. The dashboard remains deliberately unchanged and simple; omitted fields still receive disclosed median defaults.

## Model selection

The persisted models are chosen using validation on the training partition; the final 23% hold-out is used only for the reported evaluation. Task A retains a class-balanced Random Forest because it gave the most dependable safe-class F1 and balanced accuracy for the compact six-feature input design. For Task B, robust Huber gradient boosting was selected after comparison with histogram gradient boosting, Random Forest, Extra Trees, LightGBM and XGBoost. It produced the lowest validation pH error among the reliable candidates.

pH has weak predictive signal in this supplied dataset, so a low R² is an important data-quality finding rather than a reason to invent a high accuracy. The project reports MAE, RMSE and R² transparently.

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
├── src/aqua_epoch/
│   ├── train.py                  # repeatable training + evaluation
│   ├── report.py                 # PDF technical deck generator
│   ├── api.py                    # dependency-light REST server
│   └── validation.py             # shared request validation
├── web/                          # zero-build dashboard
├── tests/                        # smoke tests
└── run.py                        # train, report, and serve commands
```

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
