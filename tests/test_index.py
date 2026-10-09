"""
tests.test_index
~~~~~~~~~~~~~~~~
Validation of UnifiedTrustAnalyzer and UTI score computation.
"""

import numpy as np
from trustxai.core.index import UnifiedTrustAnalyzer, TrustStatus


def test_certified_case_low_lipschitz_and_in_distribution():
    X_train = np.random.normal(0, 1, size=(200, 4))
    analyzer = UnifiedTrustAnalyzer(lipschitz_threshold=2.0).fit_reference(X_train)

    x_test = np.array([0.05, -0.05, 0.01, -0.02])
    ood_score = analyzer.compute_ood_scores(x_test.reshape(1, -1))[0]

    result = analyzer.evaluate_instance(
        lipschitz_bound=0.4,
        prediction_set=[1],
        ood_score=ood_score
    )

    assert result.status == TrustStatus.CERTIFIED
    assert 0.0 < result.uti_score <= 1.0


def test_ambiguous_prediction_set_drops_uti():
    analyzer = UnifiedTrustAnalyzer()
    result = analyzer.evaluate_instance(
        lipschitz_bound=0.1,
        prediction_set=[0, 1],  # Uncertain prediction set
        ood_score=0.2
    )

    assert result.status == TrustStatus.AMBIGUOUS_OR_OOD
    assert result.uti_score == 0.0


def test_fragile_explanation():
    analyzer = UnifiedTrustAnalyzer(lipschitz_threshold=1.5)
    result = analyzer.evaluate_instance(
        lipschitz_bound=4.5,  # Exceeds threshold
        prediction_set=[0],
        ood_score=0.3
    )

    assert result.status == TrustStatus.FRAGILE
    assert result.uti_score > 0.0
    