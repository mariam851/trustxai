"""
trustxai.conformal.nonconformity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Module implementing Split Conformal Prediction for classification models
to produce mathematically guaranteed prediction sets.
"""

from typing import List, Tuple
import numpy as np


class ConformalClassifier:
    """
    Split Conformal Prediction calibration layer for multi-class classification.
    Calculates conformity scores using non-conformity measure:
        s_i = 1 - f(x_i)_{y_i}
    and constructs prediction sets:
        C(x) = {y : f(x)_y >= 1 - q_hat}
    """

    def __init__(self, alpha: float = 0.1):
        """
        Parameters
        ----------
        alpha : float, default=0.1
            Significance level (target coverage rate is 1 - alpha).
            e.g., alpha=0.1 guarantees >= 90% marginal coverage.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError("Significance level alpha must be strictly in (0, 1).")
        self.alpha = alpha
        self.q_hat: float = 1.0
        self.is_calibrated: bool = False

    def calibrate(self, probs_cal: np.ndarray, y_cal: np.ndarray) -> "ConformalClassifier":
        """
        Calibrates the threshold q_hat using a held-out calibration set.

        Parameters
        ----------
        probs_cal : np.ndarray
            Predicted class probabilities of shape (N_cal, K).
        y_cal : np.ndarray
            Ground truth integer labels of shape (N_cal,).
        """
        probs_cal = np.asarray(probs_cal, dtype=float)
        y_cal = np.asarray(y_cal, dtype=int)

        n = len(y_cal)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")
        if probs_cal.shape[0] != n:
            raise ValueError("Mismatch between probabilities and labels count.")

        # Extract predicted probability assigned to the true class: f(x_i)_{y_i}
        true_class_probs = probs_cal[np.arange(n), y_cal]

        # Non-conformity score: s_i = 1 - f(x_i)_{y_i}
        nonconformity_scores = 1.0 - true_class_probs

        # Finite-sample adjusted quantile: ceil((n + 1) * (1 - alpha)) / n
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(level, 1.0)

        # q_hat is the (1 - alpha) empirical quantile of nonconformity scores
        self.q_hat = float(np.quantile(nonconformity_scores, level, method="higher"))
        self.is_calibrated = True
        return self

    def predict_set(self, probs: np.ndarray) -> List[List[int]]:
        """
        Constructs prediction sets for test instances.

        Parameters
        ----------
        probs : np.ndarray
            Predicted probabilities of shape (N, K).

        Returns
        -------
        List[List[int]]
            List of sets containing included class indices for each sample.
        """
        if not self.is_calibrated:
            raise RuntimeError("Model must be calibrated before generating prediction sets.")

        probs = np.asarray(probs, dtype=float)
        # Inclusion criterion: 1 - probs <= q_hat  <=>  probs >= 1 - q_hat
        threshold = 1.0 - self.q_hat
        prediction_sets = []

        for row in probs:
            classes = np.where(row >= threshold)[0].tolist()
            # Guarantees non-empty set by falling back to argmax if empty
            if not classes:
                classes = [int(np.argmax(row))]
            prediction_sets.append(classes)

        return prediction_sets

    def evaluate_coverage(
        self, probs_test: np.ndarray, y_test: np.ndarray
    ) -> Tuple[float, float]:
        """
        Evaluates empirical coverage and average set size on test data.

        Returns
        -------
        Tuple[float, float]
            (empirical_coverage, average_set_size)
        """
        y_test = np.asarray(y_test, dtype=int)
        prediction_sets = self.predict_set(probs_test)

        covered = [
            y_true in p_set
            for y_true, p_set in zip(y_test, prediction_sets)
        ]
        empirical_coverage = float(np.mean(covered))
        average_set_size = float(np.mean([len(s) for s in prediction_sets]))

        return empirical_coverage, average_set_size
    