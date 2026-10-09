"""
trustxai.explainers.wrappers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unified wrappers for feature attribution methods (e.g. SHAP, TreeExplainer, KernelExplainer).
Ensures consistent numpy array outputs of shape (N, D) for downstream stability estimators.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
import numpy as np


class BaseExplainerWrapper(ABC):
    """Abstract base class defining the required contract for explanation wrappers."""

    @abstractmethod
    def explain(self, X: np.ndarray) -> np.ndarray:
        """
        Computes feature attribution vectors.

        Parameters
        ----------
        X : np.ndarray
            Input matrix of shape (N, D).

        Returns
        -------
        np.ndarray
            Attribution matrix of shape (N, D).
        """
        pass


class ShapWrapper(BaseExplainerWrapper):
    """
    Unified SHAP interface for Tree-based models and generic black-box estimators.
    """

    def __init__(
        self,
        model: Any,
        background_data: Optional[np.ndarray] = None,
        explainer_type: str = "auto",
        target_class: int = 1,
    ):
        """
        Parameters
        ----------
        model : Any
            Fitted classifier exposing predict_proba or predict.
        background_data : np.ndarray, optional
            Reference samples for KernelExplainer or TreeExplainer interventional trees.
        explainer_type : str, default="auto"
            One of ['auto', 'tree', 'kernel'].
        target_class : int, default=1
            Class index whose explanation vector should be tracked for multi-class problems.
        """
        import shap  # Deferred import to keep lightweight dependencies

        self.model = model
        self.target_class = target_class

        if explainer_type == "tree":
            self.explainer = shap.TreeExplainer(model, data=background_data)
        elif explainer_type == "kernel":
            if background_data is None:
                raise ValueError("KernelExplainer strictly requires background_data.")
            predict_fn = getattr(model, "predict_proba", model.predict)
            self.explainer = shap.KernelExplainer(predict_fn, background_data)
        elif explainer_type == "auto":
            try:
                self.explainer = shap.TreeExplainer(model)
            except Exception:
                if background_data is None:
                    raise ValueError(
                        "Automatic fallback to black-box explainer requires background_data."
                    )
                predict_fn = getattr(model, "predict_proba", model.predict)
                self.explainer = shap.KernelExplainer(predict_fn, background_data)
        else:
            raise ValueError(f"Unsupported explainer_type: {explainer_type}")

    def explain(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        shap_values = self.explainer.shap_values(X)

        # Handle list outputs for multi-class classifiers
        if isinstance(shap_values, list):
            target_idx = min(self.target_class, len(shap_values) - 1)
            raw_vals = np.array(shap_values[target_idx])
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # Shape: (N, D, num_classes)
            target_idx = min(self.target_class, shap_values.shape[2] - 1)
            raw_vals = shap_values[:, :, target_idx]
        else:
            raw_vals = np.array(shap_values)

        return raw_vals


class FunctionalExplainerWrapper(BaseExplainerWrapper):
    """
    Lightweight wrapper accepting any arbitrary function: X -> attributions.
    Ideal for unit tests, custom gradients, or integrated gradients.
    """

    def __init__(self, func: Callable[[np.ndarray], np.ndarray]):
        self.func = func

    def explain(self, X: np.ndarray) -> np.ndarray:
        out = self.func(np.asarray(X, dtype=float))
        return np.asarray(out, dtype=float)
    