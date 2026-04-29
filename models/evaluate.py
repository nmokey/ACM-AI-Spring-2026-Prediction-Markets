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
    """
    Evaluate a fitted classifier on a held-out test set and print results.
    """
    proba = model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, proba)
    ll = log_loss(y_test, proba)

    status = "PASS" if brier < BRIER_TARGET else "FAIL"
    print(f"\n── Model Evaluation ──────────────────────")
    print(f"  Brier score : {brier:.4f}  (target < {BRIER_TARGET}) [{status}]")
    print(f"  Log-loss    : {ll:.4f}")
    print(f"──────────────────────────────────────────\n")

    _print_feature_importance(model, feature_names)

    return {"brier_score": brier, "log_loss": ll}


def _print_feature_importance(model, feature_names: list[str] | None) -> None:
    """
    Print a ranked feature importance table.
    """
    try:
        # CalibratedClassifierCV averages across folds; grab the first estimator
        base = model.calibrated_classifiers_[0].estimator
        importances = base.feature_importances_
    except AttributeError:
        print("  [feature importance] model structure not recognised — skipping")
        return

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(importances))]

    pairs = sorted(zip(importances, feature_names), reverse=True)
    max_imp = pairs[0][0] if pairs else 1.0

    print("── Feature Importance ────────────────────")
    for imp, name in pairs:
        bar_len = round((imp / max_imp) * 20)
        print(f"  {name:<25} {'█' * bar_len} {imp:.4f}")
    print("──────────────────────────────────────────\n")


def calibration_data(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bins: int = 10,
):
    """
    Return (fraction_of_positives, mean_predicted_value) for a calibration curve.

    Use this in notebooks/model_eval.ipynb:
        fop, mpv = calibration_data(model, X_test, y_test)
        plt.plot(mpv, fop, marker='o', label='Model')
        plt.plot([0, 1], [0, 1], '--', label='Perfect calibration')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
    """
    proba = model.predict_proba(X_test)[:, 1]
    return calibration_curve(y_test, proba, n_bins=n_bins, strategy="uniform")
