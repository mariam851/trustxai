"""
TrustXAI: A Unified Diagnostic Framework for Assessing Explanation Stability,
Uncertainty Calibration, and Robustness in Machine Learning.
"""

from trustxai.core.benchmark import TrustBenchmark
from trustxai.core.index import TrustStatus, TrustEvaluationResult
from trustxai.reporting.latex import AcademicReporter

__version__ = "0.1.0"
__all__ = [
    "TrustBenchmark",
    "TrustStatus",
    "TrustEvaluationResult",
    "AcademicReporter",
]
