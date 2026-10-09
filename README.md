<div align="center">

# TrustXAI: Rigorous Auditing of Explanation Stability and Statistical Uncertainty

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A benchmark framework bridging Explainable AI (XAI) and Conformal Prediction to rigorously evaluate the empirical reliability of feature attribution methods.**

</div>

---

## Motivation & Theoretical Foundation

Post-hoc explainability techniques (e.g., SHAP, LIME, Integrated Gradients) are frequently deployed in high-stakes decision workflows. However, standard validation typically tests prediction metrics alone (e.g., AUC, F1), ignoring two failure modes:
1. **Explanation Fragility:** Minor input shifts drastically alter attribution vectors despite identical output labels (high local Lipschitz sensitivity).
2. **Unguaranteed Decisions:** Models emit confident explanations on ambiguous or out-of-distribution (OOD) instances where multi-class predictions lack statistical calibration.

`TrustXAI` enforces a tripartite diagnostic framework combining:
* **Local Lipschitz Explanation Continuity ($L_E$):** Monte Carlo estimation of maximal attribution shift over an $\epsilon$-ball.
* **Split Conformal Prediction Coverage ($C(x)$):** Finite-sample distribution-free bounds guaranteeing $P(Y \in C(X)) \ge 1 - \alpha$.
* **Unified Trust Index (UTI):** A composite score penalizing distribution shift and attribution fragility, accompanied by an explicit tripartite status: `CERTIFIED`, `FRAGILE`, or `AMBIGUOUS_OR_OOD`.

---

## Quickstart

### Installation

```bash
git clone [https://github.com/MariamZakaria/trustxai.git](https://github.com/MariamZakaria/trustxai.git)
cd trustxai
pip install -e .[explainers]
