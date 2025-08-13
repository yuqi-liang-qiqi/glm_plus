"""
Frequentist Ordinal Quantile Regression (OQR)

This package provides a practical frequentist alternative to the Bayesian
implementations under `glm_plus.ordinal_quantile_regression`.

Currently included:
- `torque.FrequentistOQR`: single-index TORQUE with an optional two-index
  extension inspired by Hong & Zhou (2013), using monotone transforms via
  isotonic regression and quantile regression for coefficient estimation.

Dependencies: numpy, scikit-learn, statsmodels, scipy
"""

from .torque import FrequentistOQR

__all__ = ["FrequentistOQR"]


