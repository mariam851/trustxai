"""
trustxai.perturbations.continuous
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Continuous perturbation strategies including bounded L2 ball and Gaussian noise.
"""

from typing import Optional
import numpy as np
from trustxai.perturbations.base import BasePerturbationGenerator


class L2BallPerturbation(BasePerturbationGenerator):
    """
    Samples instances uniformly within a scaled L2 ball of radius epsilon:
        B_epsilon(x) = { x' : ||(x' - x) / scale||_2 <= epsilon }
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        num_samples: int = 50,
        feature_scales: Optional[np.ndarray] = None,
        random_state: Optional[int] = 42,
    ):
        if epsilon <= 0:
            raise ValueError("Perturbation radius epsilon must be positive.")
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2.")

        self.epsilon = epsilon
        self.num_samples = num_samples
        self.feature_scales = feature_scales
        self.rng = np.random.default_rng(random_state)

    def generate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        dim = x.shape[0]

        # Sample uniform random directions via normalized Gaussian vectors
        gaussian = self.rng.normal(0.0, 1.0, size=(self.num_samples, dim))
        norms = np.linalg.norm(gaussian, axis=1, keepdims=True) + 1e-12
        directions = gaussian / norms

        # Radial scaling for uniform volume sampling in d-dimensional sphere
        radii = self.epsilon * (self.rng.uniform(0.0, 1.0, size=(self.num_samples, 1)) ** (1.0 / dim))

        scale = self.feature_scales if self.feature_scales is not None else np.ones_like(x)
        scale = np.where(scale == 0, 1.0, scale)

        perturbations = directions * radii * scale
        return x + perturbations
    