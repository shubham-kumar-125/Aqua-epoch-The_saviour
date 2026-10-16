"""Generate the Round 1 technical deck from saved experiment artifacts."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .config import ARTIFACTS_DIR, DISCLAIMER, DOCS_DIR


NAVY = "#12304A"
TEAL = "#087E8B"
ORANGE = "#F28C45"
INK = "#1D2935"
MUTED = "#5D6B78"
PALE = "#F2F6F8"
WHITE = "#FFFFFF"
TOTAL_PAGES = 8


def _load_metadata() -> dict[str, Any]:
    path = ARTIFACTS_DIR / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError("No training metadata found. Run `python run.py train` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _team_details() -> dict[str, str]:
    """Read submitted team information without publishing placeholder values."""
    details = {
        "team_name": "Not provided",
        "members": "Not provided",
        "contact": "Not provided",
        "spoc": "Prakshaale",
    }
    path = DOCS_DIR / "team_details.json"
    if not path.is_file():
        return details

    supplied = json.loads(path.read_text(encoding="utf-8"))
    for key in ("team_name", "members", "contact", "spoc"):
        value = str(supplied.get(key, "")).strip()
        if value and not (value.startswith("[") and value.endswith("]")):
            details[key] = value
    return details


def _page(title: str, subtitle: str, page_number: int) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(PALE)
    ax.set_axis_off()
    fig.text(0.06, 0.947, title, fontsize=23, fontweight="bold", color=NAVY, va="top")
    fig.text(0.06, 0.895, subtitle, fontsize=10.5, color=MUTED, va="top")
    fig.add_artist(Line2D([0.06, 0.94], [0.858, 0.858], transform=fig.transFigure, color="#D7E1E6", linewidth=1))
    fig.text(0.06, 0.035, "AQUA-EPOCH  /  ROUND 1 TECHNICAL SUBMISSION", fontsize=8, color=MUTED)
    fig.text(0.94, 0.035, f"{page_number:02d}  /  {TOTAL_PAGES:02d}", fontsize=8, color=MUTED, ha="right")
    return fig, ax


def _card(
    fig: plt.Figure,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    value: str,
    detail: str = "",
    accent: str = TEAL,
) -> None:
    fig.patches.append(
        Rectangle(
            (x, y),
            width,
            height,
            transform=fig.transFigure,
            facecolor=WHITE,
            edgecolor="#D7E1E6",
            linewidth=0.8,
        )
    )
    fig.patches.append(
        Rectangle((x, y), 0.012, height, transform=fig.transFigure, facecolor=accent, edgecolor=accent)
    )
    fig.text(x + 0.025, y + height - 0.045, title.upper(), fontsize=8.5, color=MUTED, fontweight="bold")
    fig.text(x + 0.025, y + height - 0.105, value, fontsize=22, color=accent, fontweight="bold")
    if detail:
        fig.text(x + 0.025, y + 0.025, detail, fontsize=8.5, color=INK, va="bottom")


def _text_block(
    fig: plt.Figure,
    x: float,
    y: float,
    heading: str,
    body: str,
    width: int = 62,
    heading_color: str = TEAL,
    body_size: float = 11,
) -> None:
    fig.text(x, y, heading, fontsize=12, color=heading_color, fontweight="bold", va="top")
    fig.text(
        x,
        y - 0.045,
        textwrap.fill(body, width=width),
        fontsize=body_size,
        color=INK,
        va="top",
        linespacing=1.4,
    )


def _add_image(fig: plt.Figure, filename: str, rect: list[float]) -> bool:
    image_path = ARTIFACTS_DIR / filename
    if not image_path.is_file():
        return False
    axis = fig.add_axes(rect)
    axis.imshow(mpimg.imread(image_path))
    axis.set_axis_off()
    return True


def _write_markdown(metadata: dict[str, Any]) -> Path:
    classification = metadata["classification"]
    regression = metadata["regression"]
    cm = classification["metrics"]
    rm = regression["metrics"]
    quality = metadata["data_quality"]
    dataset = metadata["dataset"]
    team = _team_details()
    path = DOCS_DIR / "round1_submission.md"
    matrix = classification["confusion_matrix"]
    content = f"""# Aqua-Epoch — Round 1 Technical Submission

## Project and team

- **Project:** Aqua-Epoch Water Intelligence Platform
- **Event:** techFEST'26, SLIET Longowal
- **Team:** {team['team_name']}
- **Members:** {team['members']}
- **Contact:** {team['contact']}
- **SPOC:** {team['spoc']}
- **Model training date:** {metadata['trained_on']}
- **Model version:** `{metadata['model_version']}`

## Problem and solution

The prototype predicts a binary water-safety label (`Unsafe=0`, `Safe=1`) and estimates continuous pH. A browser form and REST API provide input validation, prediction results, BIS guidance for three measurements, and visible warnings. Outputs are decision support only, not a laboratory certificate.

## Data and preprocessing

- Source CSV: `{dataset['filename']}` ({dataset['source_rows']:,} source rows, {dataset['columns']} columns).
- {dataset['rows']:,} unique source rows remain after exact-duplicate removal; {quality['rows_removed_by_basic_cleaning']:,} duplicate rows were removed. Task-specific physical-range checks are applied only to each task's target and three user-entered measurements.
- Task A has {quality['classification_labeled_rows']:,} labelled rows ({cm['test_rows']:,} held out). Its inputs are the same three measurements collected by the website: Turbidity, Hardness, and TDS.
- Task B has {quality['regression_labeled_rows']:,} usable pH targets ({rm['test_rows']:,} held out). It uses those three measurements and a cross-fitted Safe probability from Task A.
- Both tasks use a fixed-seed 77/23 hold-out split. Task A candidates are selected by five-fold stratified ROC-AUC; pH candidates use five-fold MAE cross-validation on the training partition. Hold-outs are used only for final reporting. Imputation and scaling are fitted inside pipelines.
- {quality['source_missing_ph_labels']:,} source pH labels are missing and {quality['pH_outside_0_14']:,} are outside the physical 0–14 interval. Missing chemistry values are imputed within the pipeline.
- The source contains {quality['source_exact_duplicate_rows']:,} exact duplicate rows; {quality['retained_exact_duplicate_rows']:,} remain after basic cleaning. The cleaned TDS median is {metadata['defaults']['TDS']:,.0f} mg/L, versus BIS acceptable guidance of 500 mg/L, so the dataset's units and representativeness require verification.

## Models and held-out results

Task A selects among three classifiers by training-only five-fold stratified ROC-AUC. The chosen `{classification['display_name']}` uses the same three measurements as the live form. The hold-out metrics below use a 0.50 Safe-probability threshold; the live service only reports a determinate label at or below the conservative Unsafe cutoff or at or above the Safe cutoff.

| Classification metric | Held-out result |
| --- | ---: |
| Accuracy | {cm['accuracy']:.4f} |
| Balanced accuracy | {cm['balanced_accuracy']:.4f} |
| ROC-AUC | {cm['roc_auc']:.4f} |
| Safe-class precision | {cm['precision_safe']:.4f} |
| Safe-class recall | {cm['recall_safe']:.4f} |
| Safe-class F1 | {cm['f1_safe']:.4f} |
| Confusion matrix | `[[{matrix[0][0]}, {matrix[0][1]}], [{matrix[1][0]}, {matrix[1][1]}]]` |

At this threshold the model marks {matrix[0][1]:,} of {sum(matrix[0]):,} held-out unsafe samples as Safe. This false-safe error is safety-critical; the model is not suitable for operational certification.

The live service reports **Inconclusive** when the Safe probability falls between {classification['uncertainty_policy']['lower_safe_probability']:.0%} and {classification['uncertainty_policy']['upper_safe_probability']:.0%}. On separate training-only out-of-fold predictions, the policy resolved {classification['uncertainty_policy']['coverage']:.1%} of rows with {classification['uncertainty_policy']['resolved_accuracy']:.1%} accuracy; unresolved cases require laboratory testing. These are internal validation results, not a guarantee for new water sources.

Task B selects a regression pipeline using the lowest mean absolute error in five-fold cross-validation on the training partition. The selected model is `{regression['model']}`. Its training inputs include cross-fitted classifier probabilities rather than observed Potability labels, matching the value available at inference.

| Regression metric | Held-out result |
| --- | ---: |
| MAE | {rm['mae']:.4f} pH |
| RMSE | {rm['rmse']:.4f} pH |
| R² | {rm['r2']:.4f} |
| Training-mean baseline R² | {rm['training_mean_baseline_r2']:.4f} |

The near-zero R² indicates weak explanatory/predictive signal for pH using only the three measurements the website collects. Cross-validation results are recorded in `artifacts/metadata.json`; MAE and RMSE should not be interpreted as evidence of reliable laboratory-level pH estimation.

## Prototype input behavior

The browser form requests Turbidity, Total Hardness, and TDS, and both deployed models use those three measurements. The pH model receives the classifier's Safe probability. API clients may provide optional chemistry fields for compatibility, but the current models ignore them and the response identifies those values as unused.

## Limitations and responsible use

The dataset may not represent other regions, seasons, instruments, or water sources. Feature importance is not causal evidence, and results require external validation, calibration, expert-selected thresholds, drift monitoring, and secure operational controls before any deployment.

{DISCLAIMER}

## References

- Supplied dataset: `{dataset['filename']}`.
- Bureau of Indian Standards, **IS 10500:2012 — Drinking Water Specification**, cited by the user-interface guidance in `BIS_LIMITS.md`.
- Reproducible evaluation artifacts: `artifacts/metadata.json`, `artifacts/classification_metrics.csv`, and `artifacts/regression_metrics.csv`.
"""
    path.write_text(content, encoding="utf-8")
    return path


def _cover(pdf: PdfPages, metadata: dict[str, Any], team: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.set_axis_off()
    fig.patch.set_facecolor(WHITE)
    fig.patches.append(Rectangle((0, 0.38), 1, 0.62, transform=fig.transFigure, facecolor=NAVY, edgecolor=NAVY))
    fig.patches.append(Rectangle((0, 0.38), 0.018, 0.62, transform=fig.transFigure, facecolor=TEAL, edgecolor=TEAL))
    fig.patches.append(plt.Circle((0.88, 0.77), 0.13, transform=fig.transFigure, facecolor=TEAL, alpha=0.75, edgecolor="none"))
    fig.patches.append(plt.Circle((0.92, 0.64), 0.075, transform=fig.transFigure, facecolor=ORANGE, alpha=0.9, edgecolor="none"))
    fig.text(0.07, 0.91, "AQUA-EPOCH  /  TECHNICAL SUBMISSION", fontsize=10, color="#B7DDE1", fontweight="bold")
    fig.text(0.07, 0.78, "Water intelligence\nfor informed decisions", fontsize=34, color=WHITE, fontweight="bold", va="top", linespacing=1.12)
    fig.text(0.07, 0.54, "Round 1  |  techFEST'26  |  SLIET Longowal", fontsize=14, color="#D9E7EC")
    fig.text(0.07, 0.29, "PROJECT", fontsize=8, color=MUTED, fontweight="bold")
    fig.text(0.07, 0.245, "Aqua-Epoch Water Intelligence Platform", fontsize=16, color=NAVY, fontweight="bold")
    fig.text(0.07, 0.18, f"Team: {team['team_name']}     Members: {team['members']}", fontsize=10.5, color=INK)
    fig.text(0.07, 0.135, f"Contact: {team['contact']}     SPOC: {team['spoc']}", fontsize=10.5, color=INK)
    fig.text(0.07, 0.09, f"Model {metadata['model_version']}  |  Trained {metadata['trained_on']}", fontsize=9, color=MUTED)
    fig.text(0.94, 0.035, "01  /  08", fontsize=8, color=MUTED, ha="right")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_report(metadata: dict[str, Any] | None = None) -> Path:
    """Create the Round 1 PDF and its detailed companion Markdown report."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = metadata or _load_metadata()
    _write_markdown(metadata)
    classification = metadata["classification"]
    regression = metadata["regression"]
    cm = classification["metrics"]
    rm = regression["metrics"]
    quality = metadata["data_quality"]
    dataset = metadata["dataset"]
    team = _team_details()
    output = DOCS_DIR / "round1_technical_deck.pdf"

    with PdfPages(output) as pdf:
        _cover(pdf, metadata, team)

        # 2 — objective and audience
        fig, _ = _page("The challenge and our response", "Two predictive tasks, one usable prototype, clear decision-support boundaries", 2)
        _card(fig, 0.06, 0.52, 0.275, 0.25, "Task A", "Water safety", "Binary label: Unsafe (0) or Safe (1).")
        _card(fig, 0.362, 0.52, 0.275, 0.25, "Task B", "pH estimate", "Continuous regression from chemistry.")
        _card(fig, 0.664, 0.52, 0.275, 0.25, "Delivery", "Web + REST API", "Validated inputs and explicit warnings.", ORANGE)
        _text_block(
            fig, 0.08, 0.40, "Why it matters",
            "Water measurements are useful only when they can be interpreted consistently. Aqua-Epoch combines model outputs with user-facing guidance and communicates uncertainty instead of presenting a prediction as a certificate.",
            width=90,
        )
        fig.text(0.08, 0.19, "DESIGN PRINCIPLE", fontsize=8.5, color=MUTED, fontweight="bold")
        fig.text(0.08, 0.14, "Explain the input assumptions. Show the model evidence. Keep laboratory confirmation in the loop.", fontsize=13, color=NAVY, fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 3 — dataset and methodology
        fig, _ = _page("Data and evaluation design", "The reported hold-out results use the same three inputs as the live website", 3)
        _card(fig, 0.06, 0.60, 0.275, 0.18, "Source CSV rows", f"{dataset['source_rows']:,}", f"{dataset['columns']} columns")
        _card(fig, 0.362, 0.60, 0.275, 0.18, "Unique rows", f"{dataset['rows']:,}", f"{quality['rows_removed_by_basic_cleaning']:,} exact duplicates removed", ORANGE)
        _card(fig, 0.664, 0.60, 0.275, 0.18, "Labelled examples", f"{quality['classification_labeled_rows']:,}", f"{cm['test_rows']:,} classification hold-out rows")
        _text_block(
            fig, 0.08, 0.49, "Task A | classification",
            f"{classification['display_name']}; Turbidity, Hardness, and TDS; five-fold stratified model selection; 0.50 probability threshold.",
            width=57, body_size=10.2,
        )
        _text_block(
            fig, 0.53, 0.49, "Task B | pH regression",
            f"Three measured inputs plus cross-fitted Safe probability; {quality['regression_labeled_rows']:,} usable pH targets; shared 77/23 hold-out; candidates selected by training-only five-fold MAE.",
            width=57, heading_color=ORANGE, body_size=10.2,
        )
        fig.text(0.08, 0.24, "CLEANING AND LEAKAGE CONTROLS", fontsize=8.5, color=MUTED, fontweight="bold")
        fig.text(0.08, 0.195, f"{quality['pH_outside_0_14']:,} pH labels outside 0–14 excluded  |  imputation fitted inside training pipelines  |  pH uses classifier Safe probability", fontsize=10.2, color=INK)
        fig.text(0.08, 0.13, f"Class balance after cleaning: {quality['class_distribution']['0']:,} Unsafe and {quality['class_distribution']['1']:,} Safe labels.", fontsize=10, color=MUTED)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 4 — classification results
        fig, _ = _page("Task A | water-safety classification", "Held-out performance at the predeclared 0.50 Safe-probability threshold", 4)
        _card(fig, 0.06, 0.64, 0.205, 0.17, "ROC-AUC", f"{cm['roc_auc']:.3f}", "Ranking quality")
        _card(fig, 0.282, 0.64, 0.205, 0.17, "Balanced accuracy", f"{cm['balanced_accuracy']:.3f}", "Mean class recall")
        _card(fig, 0.504, 0.64, 0.205, 0.17, "Safe-class F1", f"{cm['f1_safe']:.3f}", f"Precision {cm['precision_safe']:.3f}", ORANGE)
        _card(fig, 0.726, 0.64, 0.214, 0.17, "Accuracy", f"{cm['accuracy']:.3f}", f"{cm['test_rows']:,} held-out samples")
        _add_image(fig, "classification_confusion_matrix.png", [0.065, 0.17, 0.43, 0.39])
        matrix = classification["confusion_matrix"]
        false_safe = matrix[0][1]
        unsafe_total = sum(matrix[0])
        _text_block(
            fig, 0.55, 0.52, "Safety-critical error",
            f"{false_safe:,} of {unsafe_total:,} actually Unsafe hold-out samples were predicted Safe. This false-safe rate is {false_safe / unsafe_total:.1%}; it is unacceptable as a stand-alone basis for declaring water safe.",
            width=49, heading_color=ORANGE, body_size=11,
        )
        fig.text(0.55, 0.265, f"Actual Unsafe: {matrix[0][0]:,} correctly flagged  |  {matrix[0][1]:,} false-safe", fontsize=9.5, color=INK)
        fig.text(0.55, 0.22, f"Actual Safe: {matrix[1][1]:,} correctly flagged  |  {matrix[1][0]:,} missed", fontsize=9.5, color=INK)
        fig.text(0.55, 0.15, "Prioritise threshold review and external validation before any real-world use.", fontsize=9.5, color=MUTED, wrap=True)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 5 — classification interpretation
        fig, _ = _page("Task A | what the model learned", "Feature importance describes model behaviour; it does not establish chemical causation", 5)
        _add_image(fig, "classification_feature_importance.png", [0.06, 0.16, 0.57, 0.59])
        top_features = classification["feature_importance"][:4]
        fig.text(0.68, 0.75, "Top model signals", fontsize=14, color=TEAL, fontweight="bold")
        for index, item in enumerate(top_features):
            y = 0.67 - index * 0.095
            fig.text(0.68, y, item["feature"], fontsize=11, color=INK, fontweight="bold")
            fig.text(0.92, y, f"{item['importance']:.3f}", fontsize=11, color=TEAL, ha="right")
        _text_block(
            fig, 0.68, 0.29, "Deployment caveat",
            "The classifier uses exactly the three measurements collected by the browser form. The live service reports Inconclusive for intermediate Safe probabilities; only stronger model scores receive a binary label.",
            width=42, heading_color=ORANGE, body_size=9.5,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 6 — regression results
        fig, _ = _page("Task B | experimental pH estimate", f"{regression['model']} selected using training-only cross-validation; held-out R² remains weak", 6)
        _card(fig, 0.06, 0.64, 0.27, 0.17, "MAE", f"{rm['mae']:.3f}", "pH units | lower is better")
        _card(fig, 0.365, 0.64, 0.27, 0.17, "RMSE", f"{rm['rmse']:.3f}", "pH units | penalises large errors", ORANGE)
        _card(fig, 0.67, 0.64, 0.27, 0.17, "R-squared", f"{rm['r2']:.3f}", f"{rm['test_rows']:,} held-out samples")
        _add_image(fig, "regression_diagnostics.png", [0.055, 0.12, 0.63, 0.43])
        _text_block(
            fig, 0.72, 0.51, "Interpretation",
            f"R² is {rm['r2']:.3f} versus {rm['training_mean_baseline_r2']:.3f} for the training-mean baseline. The supplied features explain little of held-out pH variation; treat the output as experimental, not a reliable laboratory substitute.",
            width=34, heading_color=ORANGE, body_size=10,
        )
        fig.text(0.72, 0.235, f"{quality['pH_outside_0_14']:,} source pH labels fell outside the physical 0–14 range and were excluded.", fontsize=9.5, color=INK, wrap=True)
        fig.text(0.72, 0.15, "More representative measurements and external validation are required to establish utility.", fontsize=9.2, color=MUTED, wrap=True)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 7 — prototype architecture
        fig, _ = _page("From measurement to result", "Inference path and the assumptions behind the simplified web form", 7)
        boxes = [
            (0.07, "01  INPUT", "Turbidity\nHardness\nTDS"),
            (0.30, "02  VALIDATE", "Finite numeric\nPhysical ranges\nBIS guidance"),
            (0.53, "03  PREDICT", "Safety pipeline\npH pipeline\nClassifier probability"),
            (0.76, "04  RESPOND", "Label + probability\npH estimate\nWarnings"),
        ]
        for x, heading, body in boxes:
            fig.patches.append(Rectangle((x, 0.49), 0.17, 0.22, transform=fig.transFigure, facecolor=WHITE, edgecolor="#B8CDD5", linewidth=1))
            fig.patches.append(Rectangle((x, 0.49), 0.17, 0.035, transform=fig.transFigure, facecolor=TEAL if x < 0.53 else ORANGE, edgecolor="none"))
            fig.text(x + 0.014, 0.665, heading, fontsize=9.5, color=NAVY, fontweight="bold")
            fig.text(x + 0.014, 0.62, body, fontsize=9.5, color=INK, va="top", linespacing=1.35)
        for x in (0.25, 0.48, 0.71):
            fig.text(x, 0.57, ">", fontsize=17, color=ORANGE, fontweight="bold")
        _text_block(
            fig, 0.08, 0.39, "What the service uses",
            "Both models use the three form values directly. The pH model receives the classifier's predicted Safe probability; optional API chemistry fields are not consumed.",
            width=68, body_size=10.2,
        )
        _text_block(
            fig, 0.56, 0.39, "Transparency endpoints",
            "GET /api/health checks readiness. GET /api/model-info exposes recorded metrics and feature ranges. POST /api/predict returns both predictions, entered/unused fields, BIS guidance, and limitations.",
            width=55, heading_color=ORANGE, body_size=10.2,
        )
        fig.text(0.08, 0.14, "The pH model uses classifier probabilities during both training and inference; its held-out signal remains weak.", fontsize=9.5, color=MUTED)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 8 — safeguards and next steps
        fig, _ = _page("Limitations and responsible use", "Prototype decision support — not a BIS or laboratory certification", 8)
        _text_block(
            fig, 0.08, 0.76, "Use limits",
            f"The cleaned dataset's TDS median is {metadata['defaults']['TDS']:,.0f} mg/L, well above the 500 mg/L BIS acceptable guidance. Training data may not represent local water; a high Safe probability does not guarantee safety, and false-safe errors make unreviewed automated decisions inappropriate.",
            width=61, heading_color=ORANGE,
        )
        _text_block(
            fig, 0.55, 0.76, "BIS input guidance",
            "Turbidity: 1 / 5 NTU; total hardness: 200 / 600 mg/L as CaCO3; TDS: 500 / 2,000 mg/L. Values show acceptable / permissible limits where no alternate source is available. They are contextual guidance, not the model's decision rule.",
            width=57,
        )
        _text_block(
            fig, 0.08, 0.43, "Before deployment",
            "Validate on independent local laboratory data; calibrate probabilities; select thresholds with water-quality experts; monitor drift; and add authentication, rate limits, audit logging, and human review.",
            width=61,
        )
        _text_block(
            fig, 0.55, 0.43, "Next demonstration",
            "Check API health, review BIS input guidance, submit a sample, inspect the defaulted fields, and explain both the safety-critical classification errors and weak pH R-squared.",
            width=57,
        )
        fig.patches.append(Rectangle((0.06, 0.095), 0.88, 0.105, transform=fig.transFigure, facecolor="#E4F0F2", edgecolor="none"))
        fig.text(0.08, 0.158, DISCLAIMER, fontsize=9.5, color=NAVY, va="center", wrap=True)
        fig.text(0.08, 0.065, "References: supplied BIS water-quality dataset; BIS IS 10500:2012; project evaluation artifacts and methodology.", fontsize=8.5, color=MUTED)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return output


if __name__ == "__main__":
    build_report()
