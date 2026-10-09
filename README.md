# TrustXAI: Diagnostic Benchmark Suite for Explainable AI

[![CI](https://github.com/mariam851/trustxai/actions/workflows/ci.yml/badge.svg)](https://github.com/mariam851/trustxai/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A diagnostic benchmark framework to audit machine learning models for **Explanation Fragility**, **Statistical Uncertainty**, and **Out-of-Distribution (OOD) Vulnerability**.

---

## Key Pillars

1. **Explanation Fragility:** Evaluates local Lipschitz sensitivity ($L_E$) under bounded perturbations ($\epsilon$-ball) to verify whether attribution vectors remain stable.
2. **Conformal Uncertainty Guarantees:** Constructs non-empty prediction sets $C(x)$ using finite-sample distribution-free bounds guaranteeing:
   $$P(Y \in C(X)) \ge 1 - \alpha$$
3. **Out-of-Distribution Auditing:** Rejects uncalibrated inferences on distributional shifts.
4. **Unified Trust Index (UTI):** A composite metric classifying each instance as:
   - `CERTIFIED`: Stable explanation, valid conformal coverage, in-distribution.
   - `FRAGILE`: Confident prediction but unstable feature attributions.
   - `AMBIGUOUS_OR_OOD`: Prediction set exceeds single label or out-of-distribution.

---

## Quickstart

### Installation

```bash
git clone [https://github.com/mariam851/trustxai.git](https://github.com/mariam851/trustxai.git)
cd trustxai
pip install -e .[explainers]
```

### Usage Example

```python
from trustxai.core.benchmark import TrustBenchmark
from trustxai.explainers.wrappers import ShapWrapper
from trustxai.reporting.latex import AcademicReporter

# 1. Initialize explainer and diagnostic benchmark
explainer = ShapWrapper(model=model, explainer_type="tree", target_class=1)
benchmark = TrustBenchmark(
    explainer=explainer,
    alpha=0.10,          # 90% confidence target
    epsilon=0.05,        # 5% perturbation radius
    lipschitz_threshold=1.5
)

# 2. Conformal calibration on validation holdout
benchmark.calibrate(X_train=X_train, probs_cal=probs_cal, y_cal=y_cal)

# 3. Audit test data
results = benchmark.audit(X_test=X_test, probs_test=probs_test)

# 4. Generate publication-ready LaTeX table
latex_code = AcademicReporter.generate_summary_table({"Random Forest": results})
print(latex_code)
```

---

## Benchmark Metrics & Interpretation

`TrustXAI` decomposes model trustworthiness into reproducible diagnostic indicators:

| Status | Mathematical Condition | Interpretation |
| :--- | :--- | :--- |
| **`CERTIFIED`** | $L_E \le \tau$, $\Vert{}C(x)\Vert{} = 1$, $D_{OOD}(x) \le 1.0$ | Reliable prediction, bounded attribution shift, calibrated. |
| **`FRAGILE`** | $L_E > \tau$, $\Vert{}C(x)\Vert{} = 1$, $D_{OOD}(x) \le 1.0$ | Prediction is confident, but explanation flips under perturbation. |
| **`AMBIGUOUS_OR_OOD`** | $\Vert{}C(x)\Vert{} > 1$ or $D_{OOD}(x) > 1.0$ | High statistical uncertainty or shifted sample; explanations rejected. |

---

## Running Test Suite

Verify complete test coverage and stress resilience across all diagnostic layers:

```bash
# Run unit & stress tests with coverage report
pytest tests/ -v --cov=trustxai --cov-report=term-missing
```

---

## Citation

If you use **TrustXAI** in your academic research or benchmarking pipelines, please cite:

```bibtex
@software{trustxai2026,
  author = {Mariam Zakaria},
  title = {TrustXAI: A Diagnostic Benchmark Suite for Auditing Explanation Stability and Conformal Uncertainty Guarantees},
  url = {[https://github.com/mariam851/trustxai](https://github.com/mariam851/trustxai)},
  year = {2026}
}
```

---

## License

This project is open-sourced under the [MIT License](LICENSE).
