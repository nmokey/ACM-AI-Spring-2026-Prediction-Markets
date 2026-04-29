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

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

BRIER_TARGET = 0.20


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    from sklearn.metrics import brier_score_loss, log_loss
    proba = model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, proba)

    ll = log_loss(y_test, proba)
    brier_status = "PASS ✓" if brier < 0.20 else "FAIL ✗"
    print(f"Brier Score: {brier:.4f}  [{brier_status} — target < 0.20]")
    print(f"Log Loss:    {ll:.4f}")

    _print_feature_importance(model, feature_names)

    return {"brier_score": brier, "log_loss": ll}


def _print_feature_importance(model, feature_names: list[str] | None) -> None:
    if hasattr(model, "calibrated_classifiers_"):
        estimator = model.calibrated_classifiers_[0].estimator
    else:
        estimator = model

    if not hasattr(estimator, "feature_importances_"):
        print("Model does not support feature importances.")
        return

    importances = estimator.feature_importances_
    names = feature_names or [f"feature_{i}" for i in range(len(importances))]

    ranked = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)

    print("\nFeature Importances:")
    max_imp = ranked[0][1] if ranked else 1.0
    for name, score in ranked:
        bar_len = int((score / max_imp) * 30)
        bar = "█" * bar_len
        print(f"  {name:<30} {score:.4f}  {bar}")


def calibration_data(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bins: int = 10,
):
    from sklearn.calibration import calibration_curve

    proba = model.predict_proba(X_test)[:, 1]
    return calibration_curve(y_test, proba, n_bins=n_bins, strategy="uniform")
