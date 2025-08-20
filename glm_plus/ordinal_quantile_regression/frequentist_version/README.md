# Frequentist Ordinal Quantile Regression (OQR)

This module provides a practical frequentist implementation of TORQUE-style ordinal quantile regression:

- Single-index estimator with a learned monotone transformation `h` of the jittered ordinal response. By default we use a rank-based engineering approximation (threshold search along CANCOR directions + isotonic rearrangement) to mimic the spirit of Eq. (11) in the paper.
- Optional two-index extension with the same structure as the paper: `Qτ(Y|X) = floor( h^{-1}( Xβ1 + g^{-1}( Xβ2,τ ) ) )`. Here `g` is estimated via an engineering approximation to Eq. (12); `β1` is fixed at τ=0.5; `β2,τ` is obtained by quantile regression of `g(r1)` on `X`.

Dependencies: `numpy`, `scikit-learn`, `statsmodels`, `scipy`.

## Quick start

```python
import numpy as np
from glm_plus.frequentist import FrequentistOQR

# Simulated ordinal outcome 1..5 and features X (with no intercept)
X = np.random.randn(500, 4)
y = np.random.randint(1, 6, size=500)

model = FrequentistOQR(quantiles=(0.25, 0.5, 0.75), use_two_index=True, auto_select_k=True, random_state=123)
model.fit(X, y)

# Predict 50% interval
lo, hi = model.predict_interval(X[:5], 0.25, 0.75)
print(lo, hi)
```

## Notes / Differences from the paper

- Jittering (adding Unif[0,1) noise) converts ordinal outcomes to continuous.
- The estimators for `h` and `g` use a rank-motivated approximation (threshold grid + isotonic rearrangement) instead of the paper's exact rank-based optimization (Eqs. 11 and 12). We therefore do not claim the asymptotic results in Proposition 2.2 for this implementation.
- `β1` is fixed at τ=0.5 and does not vary with τ; `β2,τ` varies with τ.
- Automatic dimension selection uses a simplified CANCOR + sequential χ² test and currently supports only k ∈ {1, 2}.

## Reference

Hong, H. G., & Zhou, J. (2013). A multi-index model for quantile regression with ordinal data. Journal of Applied Statistics, 40(6), 1231–1245.


