# Aqua-Epoch — Water Intelligence Platform

A complete, reproducible submission for the Aqua-Epoch machine-learning and web-development challenge. It trains two independent models from the supplied BIS water-quality dataset and exposes them through a validated REST API and browser dashboard.

## Delivered

- **Task A — Classification:** estimates `Safe (1)` / `Unsafe (0)` from the three measurements collected by the dashboard; five-fold cross-validation selects the classifier.
- **Task B — Regression:** estimates pH from those same three measurements and a cross-fitted Safe probability from Task A.
- **Validated REST API:** `GET /api/health`, `GET /api/model-info`, and `POST /api/predict`.
- **Web application:** responsive form, client/server validation, sample loading, and result cards.
- **Round 1 materials:** a reproducible PDF technical deck, Markdown report, figures, evaluation metrics, and demo script generated from the trained artifacts.

## Important scientific limitation

This is a decision-support prototype, not a drinking-water certification system. A model prediction must never replace BIS testing, local regulation, or laboratory confirmation. The supplied dataset has major quality concerns: pH labels are missing or physically out of range, and the cleaned TDS median is about 20,773 mg/L (BIS acceptable guidance is 500 mg/L). Verify the source, units, and labels before relying on any result.

The dashboard asks only for **Turbidity, Total Hardness, and TDS**. Both deployed models are trained on these same values, so they do not depend on hidden median-filled chemistry inputs. The live safety result is **Inconclusive** for intermediate Safe probabilities (30–70%); this avoids making a confident binary claim when the three inputs are ambiguous. The pH regressor consumes a cross-fitted Safe probability from the classifier. Its weak held-out signal is prominent in the app; treat its numeric output as experimental, not reliable.

API clients may additionally provide Fluoride, Nitrate, Conductivity, and Alkalinity in the same `POST /api/predict` JSON body for compatibility; the current models ignore these optional values and disclose that in the response. Inputs outside the central 98% of the training distribution are flagged.

## Model selection

`python run.py train` evaluates candidate models using the exact three-field deployment input. Task A selects by five-fold stratified ROC-AUC and Task B by five-fold MAE. Both are reported on a shared 23% held-out split with seed 15. The pH model uses cross-fitted classifier probabilities, not observed Potability labels, during training.

The three-input classifier has conflicting labels among near-identical samples, so its probability is not a definitive water-safety determination. Adding Fluoride, Nitrate, and Conductivity substantially improved offline classification, but the simplified form was intentionally retained. The pH model's weak held-out R² is a data-quality finding, not a reason to claim high accuracy. With only the form measurements, the supplied dataset does not contain enough signal for reliable pH estimation. The project reports held-out MAE, RMSE, R², and cross-validation results transparently.

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

`main2.ipynb` is retained as the original experiment; `python run.py train` is the supported deployment-training entry point for the three-input models served by the dashboard.

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
