"""
trustxai.core.index
~~~~~~~~~~~~~~~~~~~
Computes the Unified Trust Index (UTI) and classifies explanation reliability.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import numpy as np


class TrustStatus(str, Enum):
    CERTIFIED = "CERTIFIED"
    FRAGILE = "FRAGILE"
    AMBIGUOUS_OR_OOD = "AMBIGUOUS_OR_OOD"


@dataclass
class TrustEvaluationResult:
    """Stores full audit results for a test instance."""
    uti_score: float
    status: TrustStatus
    prediction_set: List[int]
    lipschitz_bound: float
    ood_score: float


class UnifiedTrustAnalyzer:
    """
    Synthesizes Lipschitz explanation stability, conformal prediction sets,
    and out-of-distribution (OOD) distance to audit explanation trustworthiness.
    """

    def __init__(
        self,
        lipschitz_threshold: float = 2.0,
        ood_percentile_cutoff: float = 95.0,
    ):
        """
        Parameters
        ----------
        lipschitz_threshold : float, default=2.0
            Upper bound on acceptable explanation sensitivity.
        ood_percentile_cutoff : float, default=95.0
            Percentile threshold above which an instance is flagged as OOD.
        """
        self.lipschitz_threshold = lipschitz_threshold
        self.ood_percentile_cutoff = ood_percentile_cutoff
        self.reference_ood_threshold: Optional[float] = None

    def fit_reference(self, X_train: np.ndarray) -> "UnifiedTrustAnalyzer":
        """
        Fits reference distribution metrics (e.g. Mahalanobis or Euclidean distance)
        to identify out-of-distribution points.
        """
        X_train = np.asarray(X_train, dtype=float)
        self.train_mean = np.mean(X_train, axis=0)
        # Compute distances of training points to distribution centroid
        train_distances = np.linalg.norm(X_train - self.train_mean, axis=1)
        self.reference_ood_threshold = float(
            np.percentile(train_distances, self.ood_percentile_cutoff)
        )
        return self

    def compute_ood_scores(self, X: np.ndarray) -> np.ndarray:
        """Computes relative distance score to training centroid."""
        if self.reference_ood_threshold is None:
            raise RuntimeError("Must call fit_reference before computing OOD scores.")
        X = np.asarray(X, dtype=float)
        distances = np.linalg.norm(X - self.train_mean, axis=1)
        # Normalized score where values > 1.0 are considered OOD
        return distances / (self.reference_ood_threshold + 1e-12)

    def evaluate_instance(
        self,
        lipschitz_bound: float,
        prediction_set: List[int],
        ood_score: float,
    ) -> TrustEvaluationResult:
        """
        Evaluates a single instance across all 3 pillars and computes UTI:
            UTI = (1 / (1 + ood_score)) * (1 / (1 + lipschitz_bound)) * Indicator(|C(x)| == 1)
        """
        is_single_class = (len(prediction_set) == 1)
        is_in_distribution = (ood_score <= 1.0)
        is_stable = (lipschitz_bound <= self.lipschitz_threshold)

        # Mathematical formulation of UTI in [0, 1]
        uti = (
            (1.0 / (1.0 + max(0.0, ood_score)))
            * (1.0 / (1.0 + max(0.0, lipschitz_bound)))
            * (1.0 if is_single_class else 0.0)
        )

        if not is_single_class or not is_in_distribution:
            status = TrustStatus.AMBIGUOUS_OR_OOD
        elif not is_stable:
            status = TrustStatus.FRAGILE
        else:
            status = TrustStatus.CERTIFIED

        return TrustEvaluationResult(
            uti_score=float(uti),
            status=status,
            prediction_set=prediction_set,
            lipschitz_bound=float(lipschitz_bound),
            ood_score=float(ood_score),
        )

    def evaluate_batch(
        self,
        lipschitz_bounds: np.ndarray,
        prediction_sets: List[List[int]],
        X_test: np.ndarray,
    ) -> List[TrustEvaluationResult]:
        """Runs audit across a batch of samples."""
        ood_scores = self.compute_ood_scores(X_test)
        results = [
            self.evaluate_instance(l_bound, p_set, ood)
            for l_bound, p_set, ood in zip(lipschitz_bounds, prediction_sets, ood_scores)
        ]
        return results
