"""
trustxai.explainers.base
~~~~~~~~~~~~~~~~~~~~~~~~
Abstract base class defining the required contract for explanation wrappers.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


class BaseExplainerWrapper(ABC):
    """
    Abstract interface for feature attribution and explanation methods.
    
    Any custom explainer integrated into TrustXAI must inherit from this class
    and implement the `explain` method, returning a 2D numpy array.
    """

    def __init__(self, model: Any, background_data: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        model : Any
            The trained predictive model (e.g., scikit-learn, XGBoost, PyTorch module).
        background_data : np.ndarray, optional
            Reference baseline samples needed by perturbation-based explainers.
        """
        self.model = model
        self.background_data = background_data

    @abstractmethod
    def explain(self, X: np.ndarray) -> np.ndarray:
        """
        Computes local feature attribution vectors for a given batch of instances.

        Parameters
        ----------
        X : np.ndarray
            Input matrix of shape (N, D), where N is sample count and D is feature dimension.

        Returns
        -------
        np.ndarray
            Attribution matrix of shape (N, D) representing the relative contribution
            of each feature for each instance.
        """
        pass
