"""
tests.test_stability
~~~~~~~~~~~~~~~~~~~~
Unit tests for ExplanationStabilityEstimator.
"""

import numpy as np
from trustxai.core.stability import ExplanationStabilityEstimator


def test_constant_explainer_zero_lipschitz():
    """If explanation never changes, Lipschitz bound must strictly be zero."""
    dummy_x = np.array([1.0, 2.0, 3.0])

    def constant_explainer(X_batch):
        # Always returns fixed explanation vector [0.5, 0.5, 0.5]
        return np.tile([0.5, 0.5, 0.5], (len(X_batch), 1))

    estimator = ExplanationStabilityEstimator(
        explainer_func=constant_explainer,
        epsilon=0.1,
        num_perturbations=30,
        random_state=42
    )

    result = estimator.compute_instance_stability(dummy_x)
    assert np.isclose(result["lipschitz_bound"], 0.0, atol=1e-6)
    assert np.isclose(result["max_shift"], 0.0, atol=1e-6)


def test_linear_identity_explainer():
    """If explanation equals input (E(x) = x), Lipschitz bound must equal 1.0."""
    dummy_x = np.array([2.5, -1.0, 0.4])

    def identity_explainer(X_batch):
        return np.copy(X_batch)

    estimator = ExplanationStabilityEstimator(
        explainer_func=identity_explainer,
        epsilon=0.05,
        num_perturbations=50,
        random_state=42
    )

    result = estimator.compute_instance_stability(dummy_x)
    assert np.isclose(result["lipschitz_bound"], 1.0, atol=1e-5)


def test_batch_computation_shape():
    """Batch calculation should return 1D array with matching sample length."""
    X_sample = np.random.randn(10, 4)

    def simple_explainer(X_batch):
        return X_batch * 0.2

    estimator = ExplanationStabilityEstimator(
        explainer_func=simple_explainer,
        epsilon=0.05,
        num_perturbations=20
    )

    bounds = estimator.compute_batch_stability(X_sample)
    assert bounds.shape == (10,)
    assert np.all(bounds >= 0.0)
