"""
trustxai.perturbations.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Abstract base class for input perturbation strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BasePerturbationGenerator(ABC):
    """
    Abstract interface for local neighborhood perturbation generators.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        num_samples: int = 50,
        feature_scales: Optional[np.ndarray] = None,
        random_state: Optional[int] = 42,
    ):
        """
        Parameters
        ----------
        epsilon : float, default=0.05
            Maximum perturbation radius or scale factor.
        num_samples : int, default=50
            Number of synthetic instances to generate around the anchor point.
        feature_scales : np.ndarray, optional
            Per-feature scaling array of shape (D,) to prevent dominating dimensions.
        random_state : int, optional
            Seed for pseudo-random number generation.
        """
        if epsilon <= 0.0:
            raise ValueError("Perturbation radius epsilon must be strictly positive.")
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2.")

        self.epsilon = epsilon
        self.num_samples = num_samples
        self.feature_scales = feature_scales
        self.rng = np.random.default_rng(random_state)

    @abstractmethod
    def generate(self, x: np.ndarray) -> np.ndarray:
        """
        Generates perturbed samples in the immediate topological neighborhood of x.

        Parameters
        ----------
        x : np.ndarray
            Anchor instance vector of shape (D,).

        Returns
        -------
        np.ndarray
            Perturbed batch matrix of shape (num_samples, D).
        """
        pass
    