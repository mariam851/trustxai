"""
tests.test_integration
~~~~~~~~~~~~~~~~~~~~~~~
Integration tests covering perturbations, LaTeX reporting, and end-to-end benchmark.
"""

import numpy as np
import pytest
from trustxai.perturbations.continuous import L2BallPerturbation
from trustxai.reporting.latex import AcademicReporter
from trustxai.core.benchmark import TrustBenchmark
from trustxai.explainers.wrappers import FunctionalExplainerWrapper
from trustxai.core.index import TrustEvaluationResult, TrustStatus
from sklearn.ensemble import RandomForestClassifier
from trustxai.explainers.wrappers import ShapWrapper


def test_perturbation_generator_properties():
    """Verify perturbation output shapes and non-trivial perturbation magnitudes."""
    gen = L2BallPerturbation(epsilon=0.05, num_samples=25, random_state=42)
    x = np.array([1.0, 2.0, 3.0])
    perturbed = gen.generate(x)
    assert perturbed.shape == (25, 3)
    assert not np.allclose(perturbed, x)


def test_latex_reporter_output():
    """Verify LaTeX table string generation."""
    mock_results = [
        TrustEvaluationResult(
            uti_score=0.8,
            status=TrustStatus.CERTIFIED,
            prediction_set=[0],
            lipschitz_bound=0.5,
            ood_score=0.2
        ),
        TrustEvaluationResult(
            uti_score=0.0,
            status=TrustStatus.AMBIGUOUS_OR_OOD,
            prediction_set=[0, 1],
            lipschitz_bound=1.2,
            ood_score=1.5
        )
    ]
    table = AcademicReporter.generate_summary_table({"MockModel": mock_results})
    assert "\\begin{table*}" in table
    assert "MockModel" in table
    assert "\\end{table*}" in table


def test_end_to_end_benchmark_pipeline():
    """Test full pipeline execution using TrustBenchmark and FunctionalExplainer."""
    np.random.seed(42)
    X_train = np.random.randn(50, 4)
    probs_cal = np.array([[0.8, 0.2]] * 20)
    y_cal = np.array([0] * 20)

    X_test = np.random.randn(10, 4)
    probs_test = np.array([[0.7, 0.3]] * 10)

    explainer = FunctionalExplainerWrapper(func=lambda x: x * 0.5)
    bench = TrustBenchmark(explainer=explainer, alpha=0.1, epsilon=0.05, num_perturbations=10)
    bench.calibrate(X_train, probs_cal, y_cal)
    results = bench.audit(X_test, probs_test)

    assert len(results) == 10
    assert isinstance(results[0], TrustEvaluationResult)

def test_shap_wrapper_tree():
    """Verify ShapWrapper with a real RandomForest model."""
    X = np.random.randn(20, 4)
    y = np.random.choice([0, 1], size=20)
    model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)

    wrapper = ShapWrapper(model=model, explainer_type="tree", target_class=1)
    attributions = wrapper.explain(X[:5])

    assert attributions.shape == (5, 4)
    assert not np.isnan(attributions).any()


def test_perturbation_validation_errors():
    """Verify perturbation generator raises appropriate errors for invalid parameters."""
    with pytest.raises(ValueError):
        L2BallPerturbation(epsilon=-0.1)
    with pytest.raises(ValueError):
        L2BallPerturbation(num_samples=1)
