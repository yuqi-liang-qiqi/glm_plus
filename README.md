<p align="center">
  <img src="assets/logo/logo.jp2" alt="GLM Plus" width="160" />
</p>

`GLM Plus` is a Python toolbox for extended GLMs, focused on reliable implementations and clear APIs.

- **Bayesian Ordinal Quantile Regression (OQR)**: `OR1` (J≥4) and `OR2` (J=3), translated and optimized from the R package `bqror`
- **Panel extensions**: end-to-end pipeline with year fixed effects and gender×year interactions
- **Frequentist OQR (TORQUE)**: single/two-index approximations with interval prediction
- Roadmap: rare-events logistic regression (Relogit), Firth penalized logistic regression, zero-inflated counts (ZIP/ZINB)

### Table of contents
- [Features](#features)
- [Quick start](#quick-start)
- [Tutorials](#tutorials)
- [Dependencies](#dependencies)
- [Project structure](#project-structure)
- [References](#references)

## Features
- **OR1 / OR2 (Bayesian)**: full posterior sampling, DIC, marginal likelihood, parameter summaries, and covariate effect utilities
- **Panel-OQR**: build design matrices, fit at multiple quantiles, extract yearly gender gaps, compare top vs bottom, and visualize
- **Frequentist OQR (TORQUE)**:
  - Single-index with optional two-index extension
  - Monotone transform per-quantile and quantile-regression-based estimation
  - Simple API and interval prediction

## Quick start

### OR1 (J≥4) example
```python
import numpy as np
from glm_plus.ordinal_quantile_regression.ori import quantregOR1

n = 200
np.random.seed(42)
x = np.column_stack([np.ones(n), np.random.randn(n), np.random.randn(n)])
latent = x @ np.array([0.0, 1.0, -0.5]) + np.random.randn(n)
y = np.ones((n, 1), dtype=int)
y[latent > -1] = 2
y[latent > 0] = 3
y[latent > 1] = 4

k = x.shape[1]
J = len(np.unique(y))
result = quantregOR1(
    y=y,
    x=x,
    b0=np.zeros((k, 1)),
    B0=10 * np.eye(k),
    d0=np.zeros((J - 2, 1)),
    D0=0.25 * np.eye(J - 2),
    burn=500,
    mcmc=1000,
    p=0.5,
    verbose=True,
)
```

### Frequentist OQR (TORQUE) example
```python
import numpy as np
from glm_plus.frequentist import FrequentistOQR

X = np.random.randn(500, 4)
y = np.random.randint(1, 6, size=500)

model = FrequentistOQR(quantiles=(0.25, 0.5, 0.75), use_two_index=True, random_state=123)
model.fit(X, y)
lo, hi = model.predict_interval(X[:5], 0.25, 0.75)
```

## Tutorials
- Ordinal quantile regression tutorial: `tutorial_ordinal_quantile_regression.md`
- Panel OQR guide: `tutorial_panel_oqr.md`
- Module docs:
  - `glm_plus/ordinal_quantile_regression/README.md`
  - `glm_plus/frequentist/README.md`

## Dependencies
- Core: `numpy`, `scipy`, `pandas`
- Optional parallelism: `joblib`
- Frequentist OQR: `scikit-learn`, `statsmodels`, `scipy`

## Project structure
- `glm_plus/ordinal_quantile_regression/`: OR1, OR2, and panel extensions
- `glm_plus/frequentist/`: TORQUE implementation and examples
- `bqror_r_package/`: reference R package and data
- `tests/`: examples and notebooks
- `assets/logo/`: project logo

## References
- Hong, H. G., & Zhou, J. (2013). A multi-index model for quantile regression with ordinal data. Journal of Applied Statistics, 40(6), 1231–1245.
- Rahman, M. A. (2016). Bayesian Quantile Regression for Ordinal Models. Bayesian Analysis, 11(1), 1–27.
