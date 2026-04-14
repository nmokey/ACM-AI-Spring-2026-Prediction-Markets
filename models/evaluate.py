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


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    """
    Evaluate a fitted classifier on a held-out test set and print results.

    TODO (Week 4):
        1. Call model.predict_proba(X_test)[:, 1] to get P(YES) predictions
        2. Compute brier_score_loss(y_test, proba) from sklearn.metrics
        3. Compute log_loss(y_test, proba) from sklearn.metrics
        4. Print both metrics with a clear pass/fail against target (Brier < 0.20)
        5. Call _print_feature_importance(model, feature_names)
        6. Return {"brier_score": ..., "log_loss": ...}
    """
    raise NotImplementedError


def _print_feature_importance(model, feature_names: list[str] | None) -> None:
    """
    Print a ranked feature importance table.

    TODO (Week 4):
        - Try to access model.estimator.feature_importances_ (XGBoost exposes this)
        - The CalibratedClassifierCV wrapper stores the base model in
          model.calibrated_classifiers_[0].estimator
        - Sort features by importance and print a simple text bar chart
    """
    raise NotImplementedError


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

    TODO (Week 4):
        from sklearn.calibration import calibration_curve
        proba = model.predict_proba(X_test)[:, 1]
        return calibration_curve(y_test, proba, n_bins=n_bins, strategy="uniform")
    """
    raise NotImplementedError
