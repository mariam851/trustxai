"""
tests.test_conformal
~~~~~~~~~~~~~~~~~~~~
Unit and empirical property tests for ConformalClassifier.
"""

import numpy as np
import pytest
from trustxai.conformal.nonconformity import ConformalClassifier


def test_uncalibrated_error():
    """Predicting without prior calibration must raise RuntimeError."""
    clf = ConformalClassifier(alpha=0.1)
    probs = np.array([[0.8, 0.2], [0.1, 0.9]])
    with pytest.raises(RuntimeError):
        clf.predict_set(probs)


def test_empirical_coverage_guarantee():
    """Empirical coverage on synthetic well-calibrated data must satisfy >= 1 - alpha."""
    np.random.seed(42)
    n_cal = 1000
    n_test = 1000
    k_classes = 3

    # Generate Dirichlet distributed mock probabilities
    probs_cal = np.random.dirichlet(np.ones(k_classes), size=n_cal)
    # Sample labels proportionally to class probabilities
    y_cal = np.array([np.random.choice(k_classes, p=p) for p in probs_cal])

    probs_test = np.random.dirichlet(np.ones(k_classes), size=n_test)
    y_test = np.array([np.random.choice(k_classes, p=p) for p in probs_test])

    alpha = 0.1  # 90% expected coverage
    clf = ConformalClassifier(alpha=alpha).calibrate(probs_cal, y_cal)

    coverage, avg_size = clf.evaluate_coverage(probs_test, y_test)

    # Coverage should comfortably satisfy target (allowing minor finite-sample variance)
    assert coverage >= (1.0 - alpha) - 0.03
    assert avg_size >= 1.0


def test_alpha_sensitivity():
    """Higher desired confidence (smaller alpha) must expand prediction set sizes."""
    probs_cal = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
        [0.9, 0.05, 0.05]
    ])
    y_cal = np.array([0, 1, 2, 0])

    probs_test = np.array([[0.4, 0.35, 0.25]])

    clf_strict = ConformalClassifier(alpha=0.05).calibrate(probs_cal, y_cal)
    clf_relaxed = ConformalClassifier(alpha=0.50).calibrate(probs_cal, y_cal)

    set_strict = clf_strict.predict_set(probs_test)[0]
    set_relaxed = clf_relaxed.predict_set(probs_test)[0]

    # Stricter guarantee requires equal or larger set
    assert len(set_strict) >= len(set_relaxed)
    