<div align="center">
  <img src="assets/logo/logo.png" alt="GLM Plus Logo" width="200"/>
  
  <h1>GLM Plus</h1>
  <h3>Extended GLMs for Python</h3>
  
  <p>
    <strong>Reliable implementations with clear APIs.</strong><br/>
  </p>
  
  <p>
    <a href="#installation"><img src="https://img.shields.io/badge/Quick%20Start-Installation-blue" alt="Quick Start"/></a>
    <a href="#features"><img src="https://img.shields.io/badge/Features-Overview-brightgreen" alt="Features"/></a>
    <a href="#tutorials"><img src="https://img.shields.io/badge/Tutorials-Docs-orange" alt="Docs"/></a>
  </p>
  
  <p>
    <a href="#installation">Quick Start</a> •
    <a href="#usage">Usage</a> •
    <a href="#examples">Examples</a> •
    <a href="#tutorials">Documentation</a> •
    <a href="#contributing">Contributing</a>
  </p>
</div>

---

## Features

<table>
<tr>
<td width="50%">

### <strong>Core Features</strong>
- **Generalized Ordered Logit**: Python implementation of Stata's `gologit2` with robust inference
- **Bayesian OQR**: `OR1` (J≥4) and `OR2` (J=3), translated and optimized from R `bqror`
- **Panel OQR**: year fixed effects and gender×year interactions
- **Frequentist OQR (TORQUE)**: single/two-index approximations with interval prediction
- **Brant Test**: proportional odds assumption testing for ordered models
- **Utilities**: posterior summaries, DIC, marginal likelihood, covariate effects
- **Clean, minimal APIs** focused on practical workflows

</td>
<td width="50%">

### <strong>Who is it for?</strong>
- Applied researchers using ordered outcomes
- Social science and health analytics
- Users who need quantile effects beyond means/medians
- Python users who prefer clear, lightweight interfaces

</td>
</tr>
</table>

### Documentation

See the in-repo guides:

- `tutorial_ordinal_quantile_regression.md` — end-to-end OQR tutorial
- `tutorial_panel_oqr.md` — panel OQR with time trends and comparisons
- Module docs:
  - `glm_plus/gologit2/README.md` — generalized ordered logit models (Stata gologit2 equivalent)
  - `glm_plus/ordinal_quantile_regression/README.md` — Bayesian and frequentist OQR
  - `glm_plus/brant_test/` — proportional odds assumption testing

---

## Table of Contents

<details>
<summary>Navigation</summary>

- [About](#about)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)

</details>

---

## About

GLM Plus provides comprehensive tools for ordinal outcome modeling in Python, including:
- **Generalized ordered logit models** (equivalent to Stata's gologit2) with robust statistical inference
- **Ordinal quantile regression** in both Bayesian and frequentist flavors
- **Model diagnostics** including Brant tests for proportional odds assumptions
- **Panel-data helpers** and clear Python APIs

It aims to be practical and numerically stable while staying close to established methodology.

<div align="center">
  <table>
    <tr>
      <td align="center"><strong>Author</strong></td>
      <td align="center"><strong>Contact</strong></td>
      <td align="center"><strong>GitHub</strong></td>
    </tr>
    <tr>
      <td align="center">Yuqi Liang</td>
      <td align="center"><a href="mailto:dawson1900@live.com">dawson1900@live.com</a></td>
      <td align="center"><a href="https://github.com/Liang-Team">@Liang-Team</a></td>
    </tr>
  </table>
</div>

## Project Structure

This repository contains Python implementations and supporting materials.

```
glm_plus/
├── README.md
├── assets/
│   └── logo/
│       └── logo.png
├── glm_plus/
│   ├── gologit2/
│   │   ├── gologit2.py   # Generalized ordered logit models
│   │   └── README.md
│   ├── brant_test/
│   │   ├── brant_test.py # Proportional odds assumption testing
│   ├── ordinal_quantile_regression/
│   │   ├── bayesian_version/
│   │   │   ├── ori.py        # OR1 (J≥4)
│   │   │   ├── orii.py       # OR2 (J=3)
│   │   │   ├── panel_oqr.py  # Panel helpers
│   │   │   └── README.md
│   │   └── frequentist_version/
│   │       ├── torque.py     # TORQUE implementation
│   │       └── README.md
├── tests/
│   ├── seniority/        # Seniority analysis examples
│   ├── frequentist.ipynb # OQR examples
│   └── seniority.ipynb   # Panel analysis
├── tutorial_ordinal_quantile_regression.md
└── tutorial_panel_oqr.md
```

## Requirements

- Python 3.9+
- Core: `numpy`, `scipy`, `pandas`
- Optional: `joblib` (parallel tasks in examples)
- Frequentist OQR: `scikit-learn`, `statsmodels`, `scipy`

## Installation

This repository does not yet ship as a Python package. Use one of the following:

1) Work from the repo root so `glm_plus` is importable:
```bash
git clone https://github.com/your-org/glm_plus.git
cd glm_plus
python -c "import glm_plus; print('ok')"
```

2) Or add the repo root to `PYTHONPATH`:
```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

## Usage

### Generalized Ordered Logit (gologit2) example
```python
import numpy as np
from glm_plus.gologit2 import GeneralizedOrderedModel

# Generate sample data
np.random.seed(42)
n = 1000
X = np.random.randn(n, 3)
linear = X @ np.array([0.8, -0.5, 1.2])
y = np.ones(n)
y[linear > -0.5] = 2
y[linear > 0.5] = 3
y[linear > 1.5] = 4

# Fit model with mixed constraints
model = GeneralizedOrderedModel(
    link="logit", 
    pl_vars=["x1"]  # x1 has same effect across thresholds
)
result = model.fit(X, y, feature_names=["x1", "x2", "x3"])

# View results with standard errors and p-values
print(result.summary())

# Make predictions
probs = model.predict_proba(X[:10])
```

### Ordinal Quantile Regression: OR1 (J≥4) example
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

## Examples

- Run the example notebook: `tests/seniority.ipynb`
- See tutorials in the repo root for step-by-step guides

## FAQ

**When to use gologit2 vs OQR?** Use **gologit2** for standard ordered logit with relaxed proportional odds assumptions. Use **OQR** when you want to model quantile effects (distributional impacts beyond the mean).

**Which OQR model should I pick?** Use `OR2` for J=3; use `OR1` for J≥4. The frequentist TORQUE implementation provides a practical alternative for larger datasets.

**How does gologit2 compare to Stata?** Our implementation provides equivalent results with some improvements: robust numerical methods, comprehensive diagnostics, and automatic handling of edge cases. Parameter interpretations are identical, but cutpoint signs may differ (clearly documented).

**Performance considerations?** 
- **gologit2**: Optimized for reliability over speed; includes comprehensive diagnostics
- **OQR**: Vectorized latent sampling with optional parallelism for Bayesian methods

See module READMEs for detailed performance notes and optimization tips.

## Contributing

Contributions are welcome. Please open an issue or pull request with a clear description. Keep APIs simple and add tests or examples when possible.

## License

This repository currently has no explicit license. For uses beyond personal or research purposes, please contact the author.

