"""Reproducible training for water-safety classification and pH regression."""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .config import (
    ARTIFACTS_DIR,
    CLASSIFICATION_FEATURE_COLUMNS,
    CLASSIFICATION_TARGET,
    DATASET_FILENAME,
    DOCS_DIR,
    FEATURE_COLUMNS,
    MODELS_DIR,
    REGRESSION_FEATURE_COLUMNS,
    REGRESSION_TARGET,
    source_data_candidates,
)


def _json_ready(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _find_data_path(provided: Path | None) -> Path:
    if provided is not None:
        path = provided.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path
    for candidate in source_data_candidates():
        if candidate.is_file():
            return candidate
    searched = "\n".join(f"  - {candidate}" for candidate in source_data_candidates())
    raise FileNotFoundError(
        f"Could not locate {DATASET_FILENAME}. Searched:\n{searched}\n"
        "Use: python run.py train --data /path/to/dataset.csv"
    )


def _make_dirs() -> None:
    for directory in (MODELS_DIR, ARTIFACTS_DIR, DOCS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _make_classifier() -> Pipeline:
    """Task A Random Forest from main2.ipynb."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features=1.0,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def _make_regression_candidates() -> dict[str, Pipeline]:
    """pH candidates from main2.ipynb, selected by training-fold MAE."""
    return {
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=350,
                        min_samples_leaf=2,
                        max_features=1.0,
                        random_state=45,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Extra Trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=350,
                        min_samples_leaf=2,
                        max_features=0.9,
                        random_state=45,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=250,
                        learning_rate=0.04,
                        max_depth=2,
                        loss="huber",
                        random_state=45,
                    ),
                ),
            ]
        ),
        "Histogram Gradient Boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=250,
                        learning_rate=0.06,
                        l2_regularization=1,
                        random_state=45,
                    ),
                ),
            ]
        ),
        "Ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "Polynomial Ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("model", Ridge(alpha=100.0)),
            ]
        ),
        "PCA + Ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=0.95)),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
    }


def _safe_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return the probability for Safe (class 1), independent of class order."""
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("The safety classifier was not fitted with class 1.")
    return model.predict_proba(features)[:, classes.index(1)]


def _plot_confusion(matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_xticks([0, 1], ["Unsafe (0)", "Safe (1)"])
    ax.set_yticks([0, 1], ["Unsafe (0)", "Safe (1)"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Task A — held-out confusion matrix")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=15)
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "classification_confusion_matrix.png", dpi=180)
    plt.close(fig)


def _plot_importance(
    importances: np.ndarray,
    feature_columns: tuple[str, ...],
    filename: str,
    title: str,
) -> list[dict[str, float]]:
    series = pd.Series(importances, index=feature_columns).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.3))
    ax.barh(series.index, series.values, color="#0a6f78")
    ax.set_xlabel("Relative feature importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / filename, dpi=180)
    plt.close(fig)
    return [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in series.sort_values(ascending=False).items()
    ]


def _plot_regression(y_true: pd.Series, y_pred: np.ndarray) -> None:
    residuals = y_true.to_numpy() - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    low = min(float(y_true.min()), float(np.min(y_pred)))
    high = max(float(y_true.max()), float(np.max(y_pred)))
    axes[0].scatter(y_true, y_pred, s=10, alpha=0.32, color="#116c7a")
    axes[0].plot([low, high], [low, high], "--", color="#ee7d32", label="ideal")
    axes[0].set(xlabel="Actual pH", ylabel="Predicted pH", title="Task B — held-out predictions")
    axes[0].legend()
    axes[1].hist(residuals, bins=34, color="#3b8c68", edgecolor="white")
    axes[1].axvline(0, color="#333333", linestyle="--")
    axes[1].set(xlabel="Residual (actual − predicted pH)", ylabel="Count", title="Residual distribution")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "regression_diagnostics.png", dpi=180)
    plt.close(fig)


def train(data_path: Path | None = None, seed: int = 15) -> dict[str, object]:
    """Train both deployed models using the model definitions in main2.ipynb."""
    os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACTS_DIR / ".matplotlib"))
    _make_dirs()
    dataset_path = _find_data_path(data_path)
    df = pd.read_csv(dataset_path)
    required = set(FEATURE_COLUMNS) | {CLASSIFICATION_TARGET, REGRESSION_TARGET, "Potassium"}
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError("Dataset lacks required column(s): " + ", ".join(missing_columns))

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    source_rows = len(df)
    source_exact_duplicate_rows = int(df.duplicated().sum())
    source_missing_ph = int(df[REGRESSION_TARGET].isna().sum())
    source_ph_outside_range = int(((df[REGRESSION_TARGET] < 0) | (df[REGRESSION_TARGET] > 14)).sum())
    valid_rows = (
        (df[REGRESSION_TARGET].isna() | df[REGRESSION_TARGET].between(0, 14))
        & (df["Hardness"].isna() | (df["Hardness"] >= 0))
        & df["TDS"].between(50, 50_000)
        & (df["Fluoride"].isna() | (df["Fluoride"] >= 0))
        & (df["Nitrate"].isna() | (df["Nitrate"] <= 100))
        & (df["Potassium"].isna() | (df["Potassium"] <= 40))
    )
    df = df.loc[valid_rows].drop_duplicates().copy()
    retained_exact_duplicate_rows = int(df.duplicated().sum())

    X_all = df.loc[:, FEATURE_COLUMNS]
    class_rows = df[CLASSIFICATION_TARGET].isin([0, 1])
    pH_rows = df[REGRESSION_TARGET].notna() & df[CLASSIFICATION_TARGET].isin([0, 1])

    X_class = df.loc[class_rows, CLASSIFICATION_FEATURE_COLUMNS]
    y_class = df.loc[class_rows, CLASSIFICATION_TARGET].astype(int)
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_class, y_class, test_size=0.23, random_state=seed
    )
    classifier_evaluation = _make_classifier()
    print(f"Training classification model on {len(Xc_train):,} rows …")
    classifier_evaluation.fit(Xc_train, yc_train)
    class_probabilities = _safe_probability(classifier_evaluation, Xc_test)
    class_predictions = (class_probabilities >= 0.50).astype(int)
    class_metrics = {
        "accuracy": accuracy_score(yc_test, class_predictions),
        "balanced_accuracy": balanced_accuracy_score(yc_test, class_predictions),
        "precision_safe": precision_score(yc_test, class_predictions, zero_division=0),
        "recall_safe": recall_score(yc_test, class_predictions, zero_division=0),
        "f1_safe": f1_score(yc_test, class_predictions, zero_division=0),
        "roc_auc": roc_auc_score(yc_test, class_probabilities),
        "threshold": 0.50,
        "test_rows": int(len(yc_test)),
    }
    class_matrix = confusion_matrix(yc_test, class_predictions, labels=[0, 1])

    # The notebook's pH model uses Potability as a feature. At inference, the
    # API supplies the classifier's predicted class for this model input.
    regression_data = df.loc[pH_rows]
    X_reg = regression_data.loc[:, REGRESSION_FEATURE_COLUMNS]
    y_reg = regression_data.loc[:, REGRESSION_TARGET]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=45,
    )

    print(f"Selecting main2.ipynb pH model from {len(Xr_train):,} training rows …")
    regression_candidates = _make_regression_candidates()
    regression_cv = KFold(n_splits=5, shuffle=True, random_state=45)
    regression_cv_results: dict[str, dict[str, object]] = {}
    regression_mean_mae: dict[str, float] = {}
    for candidate_name, candidate in regression_candidates.items():
        fold_scores = cross_val_score(
            candidate,
            Xr_train,
            yr_train,
            cv=regression_cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        mean_mae = float(-fold_scores.mean())
        regression_mean_mae[candidate_name] = mean_mae
        regression_cv_results[candidate_name] = {
            "mean_mae": mean_mae,
            "std_mae": float(fold_scores.std()),
            "fold_mae": (-fold_scores).tolist(),
        }
    regression_model_name = min(
        regression_mean_mae,
        key=lambda name: regression_mean_mae[name],
    )
    regressor_evaluation = regression_candidates[regression_model_name]
    regressor_evaluation.fit(Xr_train, yr_train)
    reg_predictions = np.asarray(regressor_evaluation.predict(Xr_test), dtype=float)
    baseline_prediction = np.full(len(yr_test), float(yr_train.mean()))
    reg_metrics = {
        "mae": mean_absolute_error(yr_test, reg_predictions),
        "rmse": float(np.sqrt(mean_squared_error(yr_test, reg_predictions))),
        "r2": r2_score(yr_test, reg_predictions),
        "training_mean_baseline_r2": r2_score(yr_test, baseline_prediction),
        "test_rows": int(len(yr_test)),
    }

    # Refit persisted models on all labelled rows only after held-out evaluation.
    classifier = _make_classifier()
    classifier.fit(X_class, y_class)
    from sklearn.base import clone

    regressor = clone(regressor_evaluation)
    regressor.fit(X_reg, y_reg)

    class_model = classifier_evaluation.named_steps["model"]
    class_importance = _plot_importance(
        class_model.feature_importances_,
        CLASSIFICATION_FEATURE_COLUMNS,
        "classification_feature_importance.png",
        "Task A — compact Random Forest feature importance",
    )
    reg_permutation = permutation_importance(
        regressor_evaluation,
        Xr_test,
        yr_test,
        n_repeats=3,
        random_state=seed,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    reg_importance = _plot_importance(
        reg_permutation.importances_mean,
        REGRESSION_FEATURE_COLUMNS,
        "regression_feature_importance.png",
        "Task B — held-out permutation importance",
    )
    _plot_confusion(class_matrix)
    _plot_regression(yr_test, reg_predictions)

    ranges: dict[str, dict[str, float]] = {}
    defaults: dict[str, float] = {}
    missingness: dict[str, int] = {}
    for feature in FEATURE_COLUMNS:
        column = X_all[feature]
        ranges[feature] = {
            "min": float(column.min(skipna=True)),
            "max": float(column.max(skipna=True)),
            "p01": float(column.quantile(0.01)),
            "p99": float(column.quantile(0.99)),
            "median": float(column.median()),
        }
        defaults[feature] = float(column.median())
        missingness[feature] = int(column.isna().sum())

    metadata: dict[str, object] = {
        "project": "Aqua-Epoch Water Intelligence Platform",
        "model_version": "4.0.0-main2",
        "trained_on": str(date.today()),
        "dataset": {
            "filename": dataset_path.name,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "source_rows": int(source_rows),
        },
        "feature_columns": list(FEATURE_COLUMNS),
        "classification_feature_columns": list(CLASSIFICATION_FEATURE_COLUMNS),
        "regression_feature_columns": list(REGRESSION_FEATURE_COLUMNS),
        "feature_ranges": ranges,
        "defaults": defaults,
        "data_quality": {
            "classification_labeled_rows": int(class_rows.sum()),
            "regression_labeled_rows": int(pH_rows.sum()),
            "rows_removed_by_basic_cleaning": int(source_rows - len(df)),
            "source_missing_ph_labels": source_missing_ph,
            "source_exact_duplicate_rows": source_exact_duplicate_rows,
            "retained_exact_duplicate_rows": retained_exact_duplicate_rows,
            "missing_by_feature": missingness,
            "class_distribution": {str(key): int(value) for key, value in y_class.value_counts().sort_index().items()},
            "pH_outside_0_14": source_ph_outside_range,
        },
        "classification": {
            "target": CLASSIFICATION_TARGET,
            "meaning": {"0": "Unsafe", "1": "Safe"},
            "display_name": "Random Forest",
            "model": "RandomForestClassifier (200 trees, max depth 12, min_samples_split 10, min_samples_leaf 5)",
            "selection": "Fixed Random Forest from main2.ipynb using the six-feature input schema.",
            "metrics": class_metrics,
            "confusion_matrix": class_matrix.tolist(),
            "feature_importance": class_importance,
        },
        "regression": {
            "target": REGRESSION_TARGET,
            "model": regression_model_name,
            "display_name": regression_model_name,
            "features": list(REGRESSION_FEATURE_COLUMNS),
            "selection": (
                "Selected by the lowest mean MAE in five-fold cross-validation on the training "
                "partition, comparing the seven regression pipelines in main2.ipynb. The held-out "
                "test partition was not used for selection."
            ),
            "cross_validation": regression_cv_results,
            "weak_signal_r2_threshold": 0.10,
            "signal_status": "weak" if reg_metrics["r2"] < 0.10 else "measurable",
            "metrics": reg_metrics,
            "feature_importance": reg_importance,
            "potability_input": "Predicted by the classification model at inference time.",
        },
    }
    metadata = _json_ready(metadata)
    joblib.dump(classifier, MODELS_DIR / "potability_classifier.joblib")
    joblib.dump(regressor, MODELS_DIR / "ph_regressor.joblib")
    (ARTIFACTS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame([class_metrics]).to_csv(ARTIFACTS_DIR / "classification_metrics.csv", index=False)
    pd.DataFrame([reg_metrics]).to_csv(ARTIFACTS_DIR / "regression_metrics.csv", index=False)

    from .report import build_report

    build_report(metadata)
    print("\nTraining complete.")
    print(f"Task A ROC-AUC: {class_metrics['roc_auc']:.4f}; F1-safe: {class_metrics['f1_safe']:.4f}")
    print(f"Task B MAE: {reg_metrics['mae']:.4f}; RMSE: {reg_metrics['rmse']:.4f}; R²: {reg_metrics['r2']:.4f}")
    print(f"Models: {MODELS_DIR}")
    print(f"Round 1 PDF: {DOCS_DIR / 'round1_technical_deck.pdf'}")
    return metadata


if __name__ == "__main__":
    train()
