"""
trustxai.core.benchmark
~~~~~~~~~~~~~~~~~~~~~~~
Top-level TrustBenchmark orchestrator uniting stability, conformal prediction, and index scoring.
"""

from typing import List
import numpy as np

from trustxai.core.stability import ExplanationStabilityEstimator
from trustxai.conformal.nonconformity import ConformalClassifier
from trustxai.core.index import UnifiedTrustAnalyzer, TrustEvaluationResult
from trustxai.explainers.wrappers import BaseExplainerWrapper


class TrustBenchmark:
    """
    Main evaluation pipeline conducting full-spectrum trust audits on models.
    """

    def __init__(
        self,
        explainer: BaseExplainerWrapper,
        alpha: float = 0.1,
        epsilon: float = 0.05,
        num_perturbations: int = 40,
        lipschitz_threshold: float = 2.0,
    ):
        self.explainer = explainer
        self.stability_estimator = ExplanationStabilityEstimator(
            explainer_func=self.explainer.explain,
            epsilon=epsilon,
            num_perturbations=num_perturbations,
        )
        self.conformal_clf = ConformalClassifier(alpha=alpha)
        self.analyzer = UnifiedTrustAnalyzer(lipschitz_threshold=lipschitz_threshold)

    def calibrate(
        self,
        X_train: np.ndarray,
        probs_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "TrustBenchmark":
        """Fits reference distribution for OOD detection and calibrates conformal layer."""
        self.analyzer.fit_reference(X_train)
        self.conformal_clf.calibrate(probs_cal, y_cal)
        return self

    def audit(
        self,
        X_test: np.ndarray,
        probs_test: np.ndarray,
    ) -> List[TrustEvaluationResult]:
        """
        Executes end-to-end evaluation over test samples.
        """
        lipschitz_bounds = self.stability_estimator.compute_batch_stability(X_test)
        pred_sets = self.conformal_clf.predict_set(probs_test)
        return self.analyzer.evaluate_batch(lipschitz_bounds, pred_sets, X_test)
    