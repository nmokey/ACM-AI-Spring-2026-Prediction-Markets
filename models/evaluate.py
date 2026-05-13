"""
models/evaluate.py
────────────────────
Evaluation utilities — Team 2's Week 4 deliverable.

Computes Brier score, log-loss, plots calibration curve, and prints
feature importance. Called from train.py and notebooks/model_eval.ipynb.

Team 2 owns this file.

Key concepts:
    Brier Score: mean((p_model - actual_outcome)^2). Range [0, 1].
                 Lower is better. Random guessing = 0.25. Perfect = 0.0.
    Calibration: if our model says 70%, do those contracts resolve YES ~70% of the time?
                 sklearn.calibration.calibration_curve gives you the data to plot this.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

with open(Path(__file__).parents[1] / "config" / "settings.yaml") as _f:
    _CFG = yaml.safe_load(_f)

BRIER_TARGET = 0.20
_METRICS_PATH = Path(_CFG["data"]["model_metrics_path"])
_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str] | None = None,
    n_train: int | None = None,
    n_test: int | None = None,
) -> dict[str, float]:
    proba = model.predict_proba(X_test)[:, 1]
    brier = float(brier_score_loss(y_test, proba))
    ll = float(log_loss(y_test, proba))

    brier_status = "PASS ✓" if brier < BRIER_TARGET else "FAIL ✗"
    print(f"Brier Score: {brier:.4f}  [{brier_status} — target < {BRIER_TARGET}]")
    print(f"Log Loss:    {ll:.4f}")

    importances = _get_feature_importances(model, feature_names)
    _print_feature_importance(importances)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "brier_score": brier,
        "log_loss": ll,
        "n_train": n_train if n_train is not None else int(len(y_test)),
        "n_test": n_test if n_test is not None else int(len(y_test)),
        "feature_importances": importances,
    }
    with open(_METRICS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    return {"brier_score": brier, "log_loss": ll}


def _get_feature_importances(model, feature_names: list[str] | None) -> dict[str, float]:
    if hasattr(model, "calibrated_classifiers_"):
        estimator = model.calibrated_classifiers_[0].estimator
    else:
        estimator = model
    if not hasattr(estimator, "feature_importances_"):
        return {}
    names = feature_names or [f"feature_{i}" for i in range(len(estimator.feature_importances_))]
    return {n: float(v) for n, v in zip(names, estimator.feature_importances_)}


def _print_feature_importance(importances: dict[str, float]) -> None:
    if not importances:
        print("Model does not support feature importances.")
        return
    ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    max_imp = ranked[0][1] if ranked else 1.0
    print("\nFeature Importances:")
    for name, score in ranked:
        bar = "█" * int((score / max_imp) * 30)
        print(f"  {name:<30} {score:.4f}  {bar}")


def calibration_data(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bins: int = 10,
):
    proba = model.predict_proba(X_test)[:, 1]
    return calibration_curve(y_test, proba, n_bins=n_bins, strategy="uniform")
