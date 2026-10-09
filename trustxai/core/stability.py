"""
trustxai.core.stability
~~~~~~~~~~~~~~~~~~~~~~~
Module for estimating local Lipschitz continuity and stability metrics
of model explanation vectors under bounded input perturbations.
"""

from typing import Callable, Optional
import numpy as np


class ExplanationStabilityEstimator:
    """
    Estimates the local Lipschitz stability bound for feature attribution methods:
        L_E(x) = max_{x' in B_epsilon(x)} ||E(x) - E(x')||_2 / ||x - x'||_2
    """

    def __init__(
        self,
        explainer_func: Callable[[np.ndarray], np.ndarray],
        epsilon: float = 0.05,
        num_perturbations: int = 50,
        feature_scales: Optional[np.ndarray] = None,
        random_state: Optional[int] = 42,
    ):
        """
        Parameters
        ----------
        explainer_func : Callable[[np.ndarray], np.ndarray]
            Function taking a 2D numpy array (N, D) and returning
            feature attribution vectors of shape (N, D).
        epsilon : float, default=0.05
            Maximum relative perturbation radius (fraction of feature standard deviation).
        num_perturbations : int, default=50
            Number of Monte Carlo perturbation samples around each instance.
        feature_scales : np.ndarray, optional
            Pre-computed scale factors (e.g. feature std deviations) of shape (D,).
        random_state : int, optional
            Seed for reproducibility.
        """
        self.explainer_func = explainer_func
        self.epsilon = epsilon
        self.num_perturbations = num_perturbations
        self.feature_scales = feature_scales
        self.rng = np.random.default_rng(random_state)

    def _sample_ball(self, x: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """
        Generates samples uniformly inside an L2 ball around x, scaled per-feature.
        """
        dim = x.shape[0]
        # Uniform sampling within a unit sphere
        gaussian_directions = self.rng.normal(0, 1, size=(self.num_perturbations, dim))
        norms = np.linalg.norm(gaussian_directions, axis=1, keepdims=True) + 1e-12
        normalized_directions = gaussian_directions / norms

        # Uniform radius scaling in dim dimensions: r ~ U(0, 1)^(1/dim)
        uniform_radii = self.rng.uniform(0, 1, size=(self.num_perturbations, 1)) ** (1.0 / dim)
        radii = self.epsilon * uniform_radii

        # Apply perturbation scaled by feature standard deviations
        perturbations = normalized_directions * radii * scale
        return x + perturbations

    def compute_instance_stability(self, x: np.ndarray) -> dict:
        """
        Computes local stability metrics for a single instance x.

        Parameters
        ----------
        x : np.ndarray
            Input instance vector of shape (D,).

        Returns
        -------
        dict
            Dictionary containing:
            - 'lipschitz_bound': Maximum observed ratio ||Delta E|| / ||Delta x||.
            - 'mean_shift': Average explanation shift ||Delta E|| across samples.
            - 'max_shift': Maximum explanation shift ||Delta E||.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError(f"Expected 1D instance vector, got shape {x.shape}")

        scale = self.feature_scales if self.feature_scales is not None else np.ones_like(x)
        scale = np.where(scale == 0, 1.0, scale)

        # Baseline explanation for original x
        base_explanation = self.explainer_func(x.reshape(1, -1))[0]

        # Generate perturbed instances
        x_perturbed = self._sample_ball(x, scale)

        # Compute explanations on perturbed instances
        perturbed_explanations = self.explainer_func(x_perturbed)

        # Vector differences
        delta_x = x_perturbed - x
        delta_e = perturbed_explanations - base_explanation

        # Norms
        norm_delta_x = np.linalg.norm(delta_x, axis=1) + 1e-12
        norm_delta_e = np.linalg.norm(delta_e, axis=1)

        # Ratios
        ratios = norm_delta_e / norm_delta_x
        lipschitz_bound = float(np.max(ratios))

        return {
            "lipschitz_bound": lipschitz_bound,
            "mean_shift": float(np.mean(norm_delta_e)),
            "max_shift": float(np.max(norm_delta_e)),
        }

    def compute_batch_stability(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the Lipschitz stability bound for a batch of instances.

        Parameters
        ----------
        X : np.ndarray
            Batch of instances of shape (N, D).

        Returns
        -------
        np.ndarray
            Array of shape (N,) containing the estimated lipschitz_bound per sample.
        """
        X = np.asarray(X, dtype=float)
        if self.feature_scales is None:
            self.feature_scales = np.std(X, axis=0)

        bounds = [self.compute_instance_stability(instance)["lipschitz_bound"] for instance in X]
        return np.array(bounds, dtype=float)
    