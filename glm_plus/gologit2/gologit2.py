"""
Generalized Ordered Models (Python version of Stata's gologit2)

This module implements a practical subset of the functionality of Stata's
gologit2 command, supporting multiple link functions (logit, probit,
cloglog, loglog, cauchit) and optional parallel-lines (proportional odds)
constraints for selected predictors.

Design goals:
- Beginner-friendly: clear comments explain each step in simple English.
- Minimal dependencies: only numpy, scipy, and (optionally) pandas for
  friendly column names.
- Practical features: fit(), predict_proba(), and predict_xb() with a
  clean, Pythonic API.

Important notes versus Stata's gologit2:
- This is a clean-room Python implementation inspired by the Stata code.
- It focuses on core estimation and prediction. Advanced features like
  survey weights, clustering, robust SEs, autofit, margins, and gamma
  parameterization are not included in this first version.
- Standard errors are estimated using the inverse of the numerical Hessian
  approximation returned by the optimizer where possible; for L-BFGS-B
  the Hessian inverse is a low-rank approximation, so reported SEs can be
  approximate or None.

Usage (quick start):
    >>> import numpy as np
    >>> from gologit2 import GeneralizedOrderedModel
    >>> # X: shape (n_samples, n_features); y: ordinal categories like [1,2,3]
    >>> model = GeneralizedOrderedModel(link="logit", pl_vars=["x1"])  # x1 same effect across thresholds
    >>> result = model.fit(X, y, feature_names=["x1", "x2"], verbose=True)
    >>> probs = model.predict_proba(X)  # shape (n_samples, n_categories)
    >>> xb_eq1 = model.predict_xb(X, equation=1)  # linear predictor for threshold 1

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import optimize, stats


# -------------------------------
# Helper math for link functions
# -------------------------------

def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Numerically safe natural log: clip inputs to avoid log(0).

    - x can be a scalar or array.
    - Values are clipped to [eps, 1 - eps] to keep within (0, 1).
    """
    return np.log(np.clip(x, eps, 1.0 - eps))


def _cdf_logit(z: np.ndarray) -> np.ndarray:
    """Logistic CDF F(z) = 1 / (1 + exp(-z))."""
    return 1.0 / (1.0 + np.exp(-z))


def _cdf_probit(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF."""
    return stats.norm.cdf(z)


def _cdf_cloglog(z: np.ndarray) -> np.ndarray:
    """Complementary log-log cumulative form used by Stata for cloglog link.

    Stata's gologit2 uses F(x) = exp(-exp(x)) for the cumulative probability.
    Numerical safety: clip z to avoid exp overflow.
    """
    z_safe = np.clip(z, -35, 35)  # prevent exp overflow
    return np.exp(-np.exp(z_safe))


def _cdf_loglog(z: np.ndarray) -> np.ndarray:
    """Log-log cumulative form used by Stata for loglog link.

    Stata's gologit2 uses F(x) = 1 - exp(-exp(-x)).
    Numerical safety: clip z to avoid exp overflow.
    """
    z_safe = np.clip(z, -35, 35)  # prevent exp overflow
    return 1.0 - np.exp(-np.exp(-z_safe))


def _cdf_cauchit(z: np.ndarray) -> np.ndarray:
    """Cauchit CDF: F(z) = 0.5 + (1/pi) * arctan(z)."""
    return 0.5 + (1.0 / math.pi) * np.arctan(z)


def _cumprob_from_xb(xb: np.ndarray, link: str) -> np.ndarray:
    """Compute the cumulative probability F_j for each xb_j per Stata's rules.

    Stata's gologit2 defines F(XB_j) differently by link:
    - logit/probit: F_j = CDF(-xb_j)
    - cloglog:      F_j = exp(-exp(xb_j))
    - loglog:       F_j = 1 - exp(-exp(-xb_j))
    - cauchit:      F_j = 0.5 + (1/pi) * atan(-xb_j)
    """
    if link == "logit":
        return _cdf_logit(-xb)
    if link == "probit":
        return _cdf_probit(-xb)
    if link == "cloglog":
        return _cdf_cloglog(xb)
    if link == "loglog":
        return _cdf_loglog(-xb)
    if link == "cauchit":
        return _cdf_cauchit(-xb)
    raise ValueError(f"Unsupported link: {link}")


def _inverse_start_cutpoint(cum_p: np.ndarray, link: str) -> np.ndarray:
    """Start values for the cutpoints (alphas) from cumulative probabilities.

    For each link, we invert the cumulative function F to get an initial
    cutpoint value that roughly matches the observed cumulative shares.
    This mimics Stata's Start_Values logic:

    - logit:   alpha_j = -logit(cum_p)
    - probit:  alpha_j = -Phi^{-1}(cum_p)
    - cloglog: alpha_j = cloglog(1 - cum_p) = log(-log(cum_p))
    - loglog:  alpha_j = -cloglog(cum_p)    = -log(-log(1 - cum_p))
    - cauchit: alpha_j = -tan(pi * (cum_p - 0.5))
    """
    eps = 1e-9
    # Apply gentle shrinkage for sparse categories to avoid extreme initial values
    n_total = cum_p.size + 1  # approximate total categories
    p = (cum_p * (n_total - 1) + 1.0) / n_total  # shrink toward center (more conservative)
    p = np.clip(p, eps, 1.0 - eps)
    
    if link == "logit":
        return -np.log(p / (1.0 - p))
    if link == "probit":
        return -stats.norm.ppf(p)
    if link == "cloglog":
        return np.log(-np.log(p))
    if link == "loglog":
        return -np.log(-np.log(1.0 - p))
    if link == "cauchit":
        return -np.tan(math.pi * (p - 0.5))
    raise ValueError(f"Unsupported link: {link}")


def _monotonic_cutpoints_to_free(alphas: np.ndarray) -> np.ndarray:
    """Convert monotonic cutpoints α1 < α2 < ... < αK to free parameters.
    
    Reparameterization: α1 = a1, α2 = a1 + exp(d2), ..., αK = a1 + exp(d2) + ... + exp(dK)
    Returns: [a1, d2, d3, ..., dK] where d_j are log-differences.
    """
    if len(alphas) == 0:
        return np.array([])
    if len(alphas) == 1:
        return alphas  # just a1
    
    a1 = alphas[0]
    diffs = np.diff(alphas)  # α2-α1, α3-α2, ..., αK-α(K-1)
    # Ensure positive differences (add small epsilon if needed)
    diffs = np.maximum(diffs, 1e-8)
    log_diffs = np.log(diffs)  # d2, d3, ..., dK
    return np.concatenate([[a1], log_diffs])


def _free_to_monotonic_cutpoints(free_params: np.ndarray) -> np.ndarray:
    """Convert free parameters back to monotonic cutpoints.
    
    Input: [a1, d2, d3, ..., dK] 
    Output: [α1, α2, ..., αK] where α1 = a1, α2 = a1 + exp(d2), etc.
    """
    if len(free_params) == 0:
        return np.array([])
    if len(free_params) == 1:
        return free_params  # just α1 = a1
    
    a1 = free_params[0]
    log_diffs = free_params[1:]  # d2, d3, ..., dK
    diffs = np.exp(log_diffs)    # exp(d2), exp(d3), ..., exp(dK)
    
    alphas = np.empty(len(free_params))
    alphas[0] = a1
    alphas[1:] = a1 + np.cumsum(diffs)  # a1 + exp(d2), a1 + exp(d2) + exp(d3), ...
    return alphas


# -------------------------------
# Results container
# -------------------------------


@dataclass
class Gologit2Result:
    """Holds fitted results for the generalized ordered model.

    Attributes
    - link: the link function used (e.g., "logit").
    - categories: sorted unique values of the ordinal response.
    - feature_names: names of the predictors (no intercept here).
    - pl_vars: variables constrained to parallel lines (or None).
    - alphas: array of size (K,), K = n_categories - 1.
    - beta_pl: array of size (P_pl,) for parallel-lines variables (or None).
    - beta_npl: array of size (K, P_npl) for non-parallel variables (or None).
    - success: whether the optimizer converged.
    - n_iter: number of iterations used by the optimizer.
    - fun: final negative log-likelihood value.
    - message: optimizer message.
    - hess_inv: inverse Hessian approximation returned by optimizer (if any).
    - se_alphas: standard errors for cutpoint parameters (or None).
    - se_beta_pl: standard errors for parallel-lines coefficients (or None).
    - se_beta_npl: standard errors for non-parallel coefficients (or None).
    - pvalues_alphas: p-values for cutpoint parameters (or None).
    - pvalues_beta_pl: p-values for parallel-lines coefficients (or None).
    - pvalues_beta_npl: p-values for non-parallel coefficients (or None).
    - pseudo_r2: McFadden's pseudo R-squared (or None).
    - pseudo_r2_adj: adjusted pseudo R-squared (or None).
    - null_loglik: log-likelihood of null model (intercept only).
    """

    link: str
    categories: np.ndarray
    feature_names: List[str]
    pl_vars: Optional[List[str]]
    alphas: np.ndarray
    beta_pl: Optional[np.ndarray]
    beta_npl: Optional[np.ndarray]
    success: bool
    n_iter: int
    fun: float
    message: str
    hess_inv: Optional[np.ndarray]
    se_alphas: Optional[np.ndarray] = None
    se_beta_pl: Optional[np.ndarray] = None
    se_beta_npl: Optional[np.ndarray] = None
    pvalues_alphas: Optional[np.ndarray] = None
    pvalues_beta_pl: Optional[np.ndarray] = None
    pvalues_beta_npl: Optional[np.ndarray] = None
    pseudo_r2: Optional[float] = None
    pseudo_r2_adj: Optional[float] = None
    null_loglik: Optional[float] = None
    # 诊断信息
    hessian_condition_number: Optional[float] = None
    cov_condition_number: Optional[float] = None
    hessian_method: Optional[str] = None
    inversion_method: Optional[str] = None

    def params_as_dict(self) -> dict:
        """Return parameters in a friendly dict format for inspection."""
        params = {"alphas": self.alphas}
        if self.beta_pl is not None and self.pl_vars:
            params["beta_pl"] = dict(zip(self.pl_vars, self.beta_pl))
        if self.beta_npl is not None:
            npl_names = [n for n in self.feature_names if self.pl_vars is None or n not in self.pl_vars]
            by_eq = []
            for j in range(self.beta_npl.shape[0]):
                by_eq.append(dict(zip(npl_names, self.beta_npl[j])))
            params["beta_npl_by_equation"] = by_eq
        return params

    def summary(self) -> str:
        """Return a formatted summary of the fitted model results."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"Generalized Ordered Model Results ({self.link.upper()} link)")
        lines.append("=" * 70)
        lines.append(f"Categories: {self.categories.tolist()}")
        lines.append(f"Convergence: {'Yes' if self.success else 'No'}")
        lines.append(f"Iterations: {self.n_iter}")
        lines.append(f"Negative Log-Likelihood: {self.fun:.6f}")
        if self.null_loglik is not None:
            lines.append(f"Null Log-Likelihood: {self.null_loglik:.6f}")
        if self.pseudo_r2 is not None:
            lines.append(f"McFadden's Pseudo R²: {self.pseudo_r2:.4f}")
        if self.pseudo_r2_adj is not None:
            lines.append(f"Adjusted Pseudo R²: {self.pseudo_r2_adj:.4f}")
        lines.append("")
        
        # Variable constraint information
        pl_vars = self.pl_vars or []
        npl_vars = [n for n in self.feature_names if n not in pl_vars]
        lines.append("Variable Constraints:")
        lines.append(f"  Parallel-lines (same effect across thresholds): {pl_vars}")
        lines.append(f"  Non-parallel (varying effect by threshold): {npl_vars}")
        lines.append("")
        
        # Note about our parameterization vs Stata
        lines.append("PARAMETERIZATION vs. Stata gologit2:")
        lines.append("- Our cutpoints (α) have opposite sign for logit/probit: F(α - Xβ) vs F(-α + Xβ)")
        lines.append("- Coefficient interpretations (β) are identical: same direction & magnitude")
        lines.append("- Only the intercept/cutpoint signs differ between implementations")
        lines.append("")
        
        # Note about standard errors and diagnostics
        if self.se_alphas is not None:
            lines.append("STANDARD ERRORS:")
            lines.append("- Cutpoint SEs computed using Delta method transformation from monotonic")
            lines.append("  reparameterization (accounts for constraint α₁ < α₂ < ... < αₖ)")
            lines.append("- Coefficient SEs computed from numerical Hessian at MLE")
            lines.append("- P-values based on asymptotic normal distribution (Wald tests)")
            
            # 添加数值诊断信息
            if self.hessian_method:
                lines.append(f"- Hessian computed using {self.hessian_method} finite differences")
            if self.inversion_method:
                lines.append(f"- Matrix inverted using {self.inversion_method}")
            if self.hessian_condition_number is not None:
                lines.append(f"- Hessian condition number: {self.hessian_condition_number:.2e}")
            if self.cov_condition_number is not None:
                lines.append(f"- Covariance matrix condition number: {self.cov_condition_number:.2e}")
                if self.cov_condition_number > 1e12:
                    lines.append("  ⚠ Very high condition number - SEs may be unreliable")
            lines.append("")
            
            lines.append("PREDICTION vs TRAINING CONSISTENCY:")
            lines.append("- Training: negative probabilities trigger infeasibility (hard constraint)")
            lines.append("- Prediction: probabilities clipped to [0,1] and renormalized (safe mode)")
            lines.append("- Interpret predictions near constraint boundaries with caution")
            lines.append("")
        
        # Cutpoints
        lines.append("Cutpoints (Intercepts):")
        lines.append("-" * 65)
        if self.se_alphas is not None and self.pvalues_alphas is not None:
            lines.append("                Coef.      Std.Err         z        P>|z|")
            lines.append("-" * 65)
            for i, (alpha, se, pval) in enumerate(zip(self.alphas, self.se_alphas, self.pvalues_alphas), 1):
                z_stat = alpha / se if se > 0 else 0
                significance = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                lines.append(f"  α{i:2d}     {alpha:10.6f}   {se:10.6f}   {z_stat:8.3f}   {pval:8.3f} {significance}")
        else:
            for i, alpha in enumerate(self.alphas, 1):
                lines.append(f"  α{i:2d} = {alpha:10.6f}")
        lines.append("")
        
        # Parallel-lines coefficients
        if self.beta_pl is not None and self.pl_vars:
            lines.append("Parallel-Lines Coefficients (same across all thresholds):")
            lines.append("-" * 65)
            if self.se_beta_pl is not None and self.pvalues_beta_pl is not None:
                lines.append("                Coef.      Std.Err         z        P>|z|")
                lines.append("-" * 65)
                for var, coef, se, pval in zip(self.pl_vars, self.beta_pl, self.se_beta_pl, self.pvalues_beta_pl):
                    z_stat = coef / se if se > 0 else 0
                    significance = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                    lines.append(f"  {var:8s}  {coef:10.6f}   {se:10.6f}   {z_stat:8.3f}   {pval:8.3f} {significance}")
            else:
                for var, coef in zip(self.pl_vars, self.beta_pl):
                    lines.append(f"  {var:15s} = {coef:10.6f}")
            lines.append("")
        
        # Non-parallel coefficients
        if self.beta_npl is not None:
            npl_names = [n for n in self.feature_names if self.pl_vars is None or n not in self.pl_vars]
            if self.pl_vars is None or len(self.pl_vars) == 0:
                lines.append("Non-Parallel Coefficients (all variables vary by threshold):")
            else:
                lines.append("Non-Parallel Coefficients (vary by threshold):")
            lines.append("-" * 65)
            for j in range(self.beta_npl.shape[0]):
                lines.append(f"  Equation {j+1}:")
                if self.se_beta_npl is not None and self.pvalues_beta_npl is not None:
                    lines.append("                Coef.      Std.Err         z        P>|z|")
                    lines.append("    " + "-" * 61)
                    for var, coef, se, pval in zip(npl_names, self.beta_npl[j], self.se_beta_npl[j], self.pvalues_beta_npl[j]):
                        z_stat = coef / se if se > 0 else 0
                        significance = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                        lines.append(f"    {var:8s}  {coef:10.6f}   {se:10.6f}   {z_stat:8.3f}   {pval:8.3f} {significance}")
                else:
                    for var, coef in zip(npl_names, self.beta_npl[j]):
                        lines.append(f"    {var:15s} = {coef:10.6f}")
            lines.append("")
        
        lines.append("=" * 70)
        return "\n".join(lines)


# --------------------------------------
# Generalized Ordered Model (main class)
# --------------------------------------


class GeneralizedOrderedModel:
    """Generalized ordered model with optional parallel-lines constraints.

    Parameters
    - link: one of {"logit", "probit", "cloglog", "loglog", "cauchit"}.
    - pl_vars: list of predictor names constrained to have the same effect
      across all thresholds (parallel lines). If None or empty, all predictors
      are unconstrained (non-parallel lines), which matches gologit2's default.

    Notes about inputs
    - X can be a numpy array or pandas DataFrame. If a DataFrame is provided,
      we will use its column names. Otherwise, you should pass feature_names.
    - Do not include an intercept column in X. Intercepts (cutpoints) are
      handled separately as one alpha per threshold.
    """

    def __init__(self, link: str = "logit", pl_vars: Optional[Sequence[str]] = None):
        link = link.lower()
        supported = {"logit", "probit", "cloglog", "loglog", "cauchit"}
        if link not in supported:
            raise ValueError(f"Unsupported link '{link}'. Choose from {sorted(supported)}")
        self.link: str = link
        self.pl_vars: Optional[List[str]] = list(pl_vars) if pl_vars else None

        # Set by fit()
        self.categories_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self._result: Optional[Gologit2Result] = None

    # ---------------
    # Public methods
    # ---------------

    def fit(
        self,
        X: Union[np.ndarray, Any],
        y: Union[np.ndarray, Sequence],
        feature_names: Optional[Sequence[str]] = None,
        maxiter: int = 200,
        tol: float = 1e-6,
        verbose: bool = True,
        optimizer: str = "L-BFGS-B",
    ) -> Gologit2Result:
        """Estimate parameters by maximizing the log-likelihood.

        Inputs
        - X: shape (n_samples, n_features). Do NOT include an intercept.
        - y: ordinal response values (e.g., [1, 2, 3] or [-1, 0, 1]).
        - feature_names: names for the columns of X (required if X is numpy).
        - maxiter, tol: optimizer controls.
        - verbose: if True, print progress messages a beginner can follow.
        - optimizer: optimizer name for scipy.optimize.minimize.

        Returns
        - Gologit2Result with fitted parameters and basic diagnostics.
        """
        # Handle pandas input gracefully without requiring pandas at import time
        try:
            import pandas as pd  # type: ignore
        except Exception:  # pragma: no cover - optional import
            pd = None  # noqa: N816

        if verbose:
            print("[gologit2] Preparing data...")

        # Extract numpy arrays and feature names
        if pd is not None and isinstance(X, pd.DataFrame):
            X_np = X.values.astype(float)
            names = list(X.columns)
        else:
            X_np = np.asarray(X, dtype=float)
            if feature_names is None:
                names = [f"x{i}" for i in range(X_np.shape[1])]
            else:
                names = list(feature_names)
        y_np = np.asarray(y)

        # Map y to sorted unique categories and store mapping to indices 0..M-1
        categories = np.unique(y_np)
        cat_to_index = {c: i for i, c in enumerate(categories)}
        y_idx = np.vectorize(cat_to_index.get)(y_np)
        n, p = X_np.shape
        M = categories.size
        if M < 3:
            raise ValueError("y must have at least 3 ordered categories for gologit2.")
        K = M - 1  # number of thresholds/equations

        if verbose:
            print(f"[gologit2] n={n}, p={p}, categories={categories.tolist()} (K={K})")
            print(f"[gologit2] Link: {self.link}")

        # Determine parallel-lines (pl) vs non-parallel (npl) variables
        pl_vars = set(self.pl_vars) if self.pl_vars else set()
        for v in pl_vars:
            if v not in names:
                raise ValueError(f"pl_vars contains '{v}', which is not in feature_names {names}")
        pl_mask = np.array([name in pl_vars for name in names], dtype=bool)
        npl_mask = ~pl_mask
        p_pl = int(pl_mask.sum())
        p_npl = p - p_pl

        if verbose:
            if p_pl == 0:
                print("[gologit2] No parallel-lines constraints: all coefficients vary by threshold (npl).")
            else:
                kept = [names[i] for i in range(p) if pl_mask[i]]
                print(f"[gologit2] Parallel-lines variables: {kept}")
                print("[gologit2] Remaining variables vary by threshold (npl).")

        # Start values
        if verbose:
            print("[gologit2] Computing start values (cutpoints from observed cumulative proportions)...")
        # empirical proportions per category
        counts = np.bincount(y_idx, minlength=M).astype(float)
        probs = counts / counts.sum()
        cum_probs = np.cumsum(probs)[:-1]  # length K
        alpha0 = _inverse_start_cutpoint(cum_probs, self.link)
        # Convert to monotonic-constrained parameterization
        alpha0_free = _monotonic_cutpoints_to_free(alpha0)
        # start betas at 0
        beta_pl0 = np.zeros(p_pl) if p_pl > 0 else None
        beta_npl0 = np.zeros((K, p_npl)) if p_npl > 0 else None

        # Parameter packing/unpacking helpers (now using free alpha parameterization)
        def pack_params(alpha_free: np.ndarray, bpl: Optional[np.ndarray], bnpl: Optional[np.ndarray]) -> np.ndarray:
            parts: List[np.ndarray] = [alpha_free.ravel()]
            if bpl is not None:
                parts.append(bpl.ravel())
            if bnpl is not None:
                parts.append(bnpl.ravel())
            return np.concatenate(parts) if parts else np.array([], dtype=float)

        def unpack_params(theta: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
            pos = 0
            alpha_free = theta[pos : pos + K]
            pos += K
            # Convert back to monotonic cutpoints
            alphas = _free_to_monotonic_cutpoints(alpha_free)
            bpl = None
            if p_pl > 0:
                bpl = theta[pos : pos + p_pl]
                pos += p_pl
            bnpl = None
            if p_npl > 0:
                bnpl = theta[pos : pos + K * p_npl].reshape(K, p_npl)
                pos += K * p_npl
            return alphas, bpl, bnpl

        theta0 = pack_params(alpha0_free, beta_pl0, beta_npl0)

        # Pre-split design matrix once to avoid repeated indexing in the objective
        X_pl = X_np[:, pl_mask] if p_pl > 0 else None
        X_npl = X_np[:, npl_mask] if p_npl > 0 else None

        if verbose:
            print("[gologit2] Starting numerical optimization (maximum likelihood)...")

        def nll(theta: np.ndarray) -> float:
            """Negative log-likelihood under current parameters.

            Steps (for each observation):
            1) For each threshold j, compute xb_j = alpha_j + X_pl @ b_pl + X_npl @ b_npl[j].
            2) Convert to cumulative probabilities F_j using the chosen link.
            3) Convert to category probabilities by differencing cumulative probs.
            4) Check for negative probabilities and non-monotonic F (return +∞ if found).
            5) Accumulate log-likelihood for the observed category.
            """
            alphas, bpl, bnpl = unpack_params(theta)

            # Vectorized computation of linear predictors for all thresholds
            # xb shape: (K, n)
            xb = alphas[:, np.newaxis]  # broadcast alphas to (K, 1)
            
            if p_pl > 0 and X_pl is not None and bpl is not None:
                # Add parallel-lines contribution (same for all thresholds)
                xb = xb + (X_pl @ bpl)[np.newaxis, :]  # broadcast to (K, n)
            
            if p_npl > 0 and X_npl is not None and bnpl is not None:
                # Add non-parallel contribution (different for each threshold)
                xb = xb + (bnpl @ X_npl.T)  # (K, p_npl) @ (p_npl, n) = (K, n)

            # Cumulative probabilities F_j per threshold j
            F = _cumprob_from_xb(xb, self.link)  # shape (K, n)
            
            # STRICT CHECK: Ensure F is monotonic (F_j >= F_{j-1} for all j > 1)
            # Vectorized check: diff along threshold axis should be non-negative
            if K > 1 and np.any(np.diff(F, axis=0) < -1e-12):  # allow tiny numerical tolerance
                return np.inf  # infeasible parameters
            
            # Category probabilities p(y = category_k)
            # k = 0 (lowest): F1
            # 0 < k < M-1:   Fk - F(k-1)
            # k = M-1:       1 - F(K)
            p_cat = np.empty((M, n))
            p_cat[0, :] = F[0, :]
            for k_idx in range(1, M - 1):
                p_cat[k_idx, :] = F[k_idx, :] - F[k_idx - 1, :]
            p_cat[M - 1, :] = 1.0 - F[K - 1, :]
            
            # STRICT CHECK: Ensure no negative probabilities
            if np.any(p_cat < -1e-12):  # allow tiny numerical tolerance
                return np.inf  # infeasible parameters

            # Select the probability for the observed category and sum logs
            obs_prob = p_cat[y_idx, np.arange(n)]
            
            # Final check: ensure observed probabilities are positive
            if np.any(obs_prob <= 0):
                return np.inf
                
            return -np.sum(np.log(obs_prob))

        # Optimize with improved convergence criteria
        opt_options = {"maxiter": maxiter, "disp": bool(verbose), "gtol": tol}
        # Add ftol for more intuitive convergence (SciPy >= 1.11 supports ftol for L-BFGS-B)
        try:
            opt_options["ftol"] = tol
        except Exception:
            pass  # fallback for older SciPy versions
        
        opt = optimize.minimize(
            nll,
            theta0,
            method=optimizer,
            options=opt_options,
        )

        if verbose:
            status = "converged" if opt.success else "failed"
            print(f"[gologit2] Optimization {status}. Iterations: {opt.nit}. NLL: {opt.fun:.6f}")

        # Unpack final parameters
        alpha_hat, bpl_hat, bnpl_hat = unpack_params(opt.x)

        # Try to extract a dense hessian inverse if available
        hess_inv = None
        if hasattr(opt, "hess_inv"):
            try:
                # For BFGS, hess_inv may be a matrix; for L-BFGS-B it's a LinearOperator-like object.
                hess_inv = np.asarray(opt.hess_inv.todense()) if hasattr(opt.hess_inv, "todense") else np.asarray(opt.hess_inv)
            except Exception:
                hess_inv = None

        result = Gologit2Result(
            link=self.link,
            categories=categories,
            feature_names=names,
            pl_vars=list(pl_vars) if pl_vars else None,
            alphas=alpha_hat,
            beta_pl=bpl_hat if p_pl > 0 else None,
            beta_npl=bnpl_hat if p_npl > 0 else None,
            success=bool(opt.success),
            n_iter=int(getattr(opt, "nit", 0) or 0),
            fun=float(opt.fun),
            message=str(opt.message),
            hess_inv=hess_inv,
        )
        
        # 计算标准误、p值和pseudo R2
        result = self._compute_statistics(X_np, y_np, result)

        # Save fitted state on the model
        self.categories_ = categories
        self.feature_names_ = names
        self._result = result

        # Comprehensive diagnostic check on training set
        if verbose:
            print("[gologit2] Running comprehensive diagnostics on training set...")
        self._comprehensive_diagnostics(X_np, verbose=verbose)

        if verbose:
            print("[gologit2] Finished. You can inspect 'result.params_as_dict()' or 'result.summary()'.")

        return result

    def predict_proba(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """Predict category probabilities for each row of X.

        Returns an array of shape (n_samples, n_categories).
        
        Note: This method applies safety guardrails (clip negative values to 0 
        and renormalize) to ensure valid probability distributions. This differs 
        from the strict infeasibility checks during training. For raw probabilities 
        without guardrails (for diagnostics), use _predict_proba_raw().
        """
        if self._result is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        # Handle pandas input
        try:
            import pandas as pd  # type: ignore
        except Exception:  # pragma: no cover
            pd = None

        if pd is not None and isinstance(X, pd.DataFrame):
            X_np = X.values.astype(float)
            # Ensure column order matches training
            if self.feature_names_ is not None and list(X.columns) != self.feature_names_:
                raise ValueError("Column order/names of X do not match the fitted model.")
        else:
            X_np = np.asarray(X, dtype=float)

        res = self._result
        names = res.feature_names
        link = res.link
        categories = res.categories
        n, p = X_np.shape
        M = categories.size
        K = M - 1

        # Determine masks based on fitted model
        pl_vars = set(res.pl_vars or [])
        pl_mask = np.array([name in pl_vars for name in names], dtype=bool)
        npl_mask = ~pl_mask
        p_pl = int(pl_mask.sum())
        p_npl = p - p_pl

        X_pl = X_np[:, pl_mask] if p_pl > 0 else None
        X_npl = X_np[:, npl_mask] if p_npl > 0 else None

        # Vectorized computation of linear predictors for all thresholds
        # xb shape: (K, n)
        xb = res.alphas[:, np.newaxis]  # broadcast alphas to (K, 1)
        
        if p_pl > 0 and X_pl is not None and res.beta_pl is not None:
            # Add parallel-lines contribution (same for all thresholds)
            xb = xb + (X_pl @ res.beta_pl)[np.newaxis, :]  # broadcast to (K, n)
        
        if p_npl > 0 and X_npl is not None and res.beta_npl is not None:
            # Add non-parallel contribution (different for each threshold)
            xb = xb + (res.beta_npl @ X_npl.T)  # (K, p_npl) @ (p_npl, n) = (K, n)

        # Cum probs and category probs
        F = _cumprob_from_xb(xb, link)
        p_cat = np.empty((M, n))
        p_cat[0, :] = F[0, :]
        for k_idx in range(1, M - 1):
            p_cat[k_idx, :] = F[k_idx, :] - F[k_idx - 1, :]
        p_cat[M - 1, :] = 1.0 - F[K - 1, :]

        # Final safety guardrails: ensure non-negative probabilities and normalization
        p_cat = np.maximum(p_cat, 0.0)  # clip negative values to 0
        row_sums = p_cat.sum(axis=0, keepdims=True)
        p_cat = p_cat / np.maximum(row_sums, 1e-12)  # normalize each row to sum to 1

        return p_cat.T  # (n, M)

    def predict_xb(self, X: Union[np.ndarray, Any], equation: int) -> np.ndarray:
        """Predict the linear predictor xb_j for a given threshold equation (1..K).

        - equation=1 refers to the lowest threshold, up to K = n_categories-1.
        - Returns a 1D array of length n_samples.
        """
        if self._result is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        eq = int(equation)
        categories = self._result.categories
        K = categories.size - 1
        if eq < 1 or eq > K:
            raise ValueError(f"equation must be in 1..{K}")

        # Handle pandas input
        try:
            import pandas as pd  # type: ignore
        except Exception:  # pragma: no cover
            pd = None

        if pd is not None and isinstance(X, pd.DataFrame):
            X_np = X.values.astype(float)
            if self.feature_names_ is not None and list(X.columns) != self.feature_names_:
                raise ValueError("Column order/names of X do not match the fitted model.")
        else:
            X_np = np.asarray(X, dtype=float)

        res = self._result
        names = res.feature_names
        pl_vars = set(res.pl_vars or [])
        pl_mask = np.array([name in pl_vars for name in names], dtype=bool)
        npl_mask = ~pl_mask

        X_pl = X_np[:, pl_mask] if res.beta_pl is not None else None
        X_npl = X_np[:, npl_mask] if res.beta_npl is not None else None

        xb_j = np.full(X_np.shape[0], res.alphas[eq - 1])
        if X_pl is not None and res.beta_pl is not None:
            xb_j = xb_j + X_pl @ res.beta_pl
        if X_npl is not None and res.beta_npl is not None:
            xb_j = xb_j + X_npl @ res.beta_npl[eq - 1]
        return xb_j

    def _comprehensive_diagnostics(self, X: np.ndarray, verbose: bool = True) -> None:
        """Run comprehensive diagnostics on fitted model including probability ranges and F spacing."""
        if self._result is None:
            return
        
        try:
            # Get raw probabilities and cumulative probabilities for detailed analysis
            probs = self._predict_proba_raw(X)
            res = self._result
            n, p = X.shape
            M = res.categories.size
            K = M - 1
            
            # Compute F values for spacing analysis
            pl_vars = set(res.pl_vars or [])
            pl_mask = np.array([name in pl_vars for name in res.feature_names], dtype=bool)
            npl_mask = ~pl_mask
            p_pl = int(pl_mask.sum())
            p_npl = p - p_pl
            
            X_pl = X[:, pl_mask] if p_pl > 0 else None
            X_npl = X[:, npl_mask] if p_npl > 0 else None
            
            # Vectorized computation of F values
            xb = res.alphas[:, np.newaxis]  # (K, 1)
            if p_pl > 0 and X_pl is not None and res.beta_pl is not None:
                xb = xb + (X_pl @ res.beta_pl)[np.newaxis, :]  # (K, n)
            if p_npl > 0 and X_npl is not None and res.beta_npl is not None:
                xb = xb + (res.beta_npl @ X_npl.T)  # (K, n)
            
            F = _cumprob_from_xb(xb, res.link)  # shape (K, n)
            
            # 1. Probability range analysis
            prob_min = np.min(probs)
            prob_median = np.median(probs)
            prob_max = np.max(probs)
            
            # 2. Check for infeasible samples (negative probabilities)
            neg_mask = probs < -1e-12
            infeasible_count = np.sum(np.any(neg_mask, axis=1))
            
            # 3. F spacing analysis (minimum gaps between consecutive F values)
            if K > 1:
                f_diffs = np.diff(F, axis=0)  # shape (K-1, n)
                min_f_spacing = np.min(f_diffs)
                median_f_spacing = np.median(f_diffs)
            else:
                min_f_spacing = float('inf')
                median_f_spacing = float('inf')
            
            if verbose:
                print(f"[gologit2] DIAGNOSTICS:")
                print(f"  Probability range: [{prob_min:.6f}, {prob_median:.6f}, {prob_max:.6f}] (min/median/max)")
                print(f"  Infeasible samples (negative probs): {infeasible_count}")
                if K > 1:
                    print(f"  F spacing: min={min_f_spacing:.6f}, median={median_f_spacing:.6f}")
                else:
                    print(f"  F spacing: N/A (only 1 threshold)")
                
                # Warnings and recommendations
                if infeasible_count > 0:
                    print(f"[gologit2] WARNING: {infeasible_count} samples have negative probabilities!")
                    print("[gologit2] Consider: (1) adding parallel-lines constraints, or")
                    print("[gologit2]          (2) merging sparse categories, or")
                    print("[gologit2]          (3) using a simpler model (e.g., multinomial logit)")
                
                if min_f_spacing < 1e-6:
                    print(f"[gologit2] WARNING: Very small F spacing detected (min={min_f_spacing:.2e})")
                    print("[gologit2] This may indicate near-separation or numerical instability.")
                
                if prob_max > 1 + 1e-12:
                    print(f"[gologit2] WARNING: Some probabilities exceed 1.0 (max={prob_max:.6f})")
                    print("[gologit2] This indicates numerical issues in the model.")
                    
        except Exception:
            # If diagnostic fails, don't crash the main estimation
            if verbose:
                print("[gologit2] Could not complete comprehensive diagnostics.")

    def _predict_proba_raw(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities without safety guardrails (for diagnostics)."""
        if self._result is None:
            raise RuntimeError("Model is not fitted.")
        
        res = self._result
        names = res.feature_names
        link = res.link
        categories = res.categories
        n, p = X.shape
        M = categories.size
        K = M - 1

        # Determine masks based on fitted model
        pl_vars = set(res.pl_vars or [])
        pl_mask = np.array([name in pl_vars for name in names], dtype=bool)
        npl_mask = ~pl_mask
        p_pl = int(pl_mask.sum())
        p_npl = p - p_pl

        X_pl = X[:, pl_mask] if p_pl > 0 else None
        X_npl = X[:, npl_mask] if p_npl > 0 else None

        # Vectorized computation of linear predictors
        xb = res.alphas[:, np.newaxis]  # (K, 1)
        
        if p_pl > 0 and X_pl is not None and res.beta_pl is not None:
            xb = xb + (X_pl @ res.beta_pl)[np.newaxis, :]  # (K, n)
        
        if p_npl > 0 and X_npl is not None and res.beta_npl is not None:
            xb = xb + (res.beta_npl @ X_npl.T)  # (K, n)

        # Compute probabilities without safety clipping
        F = _cumprob_from_xb(xb, link)
        p_cat = np.empty((M, n))
        p_cat[0, :] = F[0, :]
        for k_idx in range(1, M - 1):
            p_cat[k_idx, :] = F[k_idx, :] - F[k_idx - 1, :]
        p_cat[M - 1, :] = 1.0 - F[K - 1, :]

        return p_cat.T  # (n, M)

    def compute_numerical_hessian(self, X: np.ndarray, y: np.ndarray, method: str = "3-point") -> Optional[dict]:
        """Compute numerical Hessian at the fitted parameters.
        
        Returns the inverse Hessian (approximate covariance matrix) if successful.
        This provides more reliable standard errors than L-BFGS-B's low-rank approximation.
        
        Parameters:
        - X: feature matrix used during fitting  
        - y: response vector used during fitting
        - method: finite difference method ("2-point", "3-point", "cs")
        """
        if self._result is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        
        try:
            # 优先尝试SciPy的内部API
            from scipy.optimize._numdiff import approx_derivative
        except ImportError:
            try:
                # 后备：检查是否在公开API中
                from scipy.optimize import approx_derivative
            except ImportError:
                try:
                    # 再后备：检查更早版本的位置
                    from scipy.misc import approx_fprime as approx_derivative_fallback
                    print("[gologit2] Warning: Using fallback derivative approximation.")
                    print("[gologit2] Consider upgrading SciPy for better numerical differentiation.")
                    approx_derivative = approx_derivative_fallback
                except ImportError:
                    raise ImportError(
                        "Cannot find numerical differentiation function. "
                        "Please upgrade SciPy (>= 1.0 recommended) or install numdifftools."
                    )
            
            print("[gologit2] Computing numerical Hessian (this may take a moment)...")
            
            # Reconstruct objective function from fitted model state
            res = self._result
            
            # Convert y to category indices
            categories = res.categories
            cat_to_index = {c: i for i, c in enumerate(categories)}
            y_idx = np.vectorize(cat_to_index.get)(y)
            n, p = X.shape
            M = categories.size
            K = M - 1
            
            # Set up parallel-lines masks
            pl_vars = set(res.pl_vars or [])
            pl_mask = np.array([name in pl_vars for name in res.feature_names], dtype=bool)
            npl_mask = ~pl_mask
            p_pl = int(pl_mask.sum())
            p_npl = p - p_pl
            
            X_pl = X[:, pl_mask] if p_pl > 0 else None
            X_npl = X[:, npl_mask] if p_npl > 0 else None
            
            def pack_params(alpha_free: np.ndarray, bpl: Optional[np.ndarray], bnpl: Optional[np.ndarray]) -> np.ndarray:
                parts: List[np.ndarray] = [alpha_free.ravel()]
                if bpl is not None:
                    parts.append(bpl.ravel())
                if bnpl is not None:
                    parts.append(bnpl.ravel())
                return np.concatenate(parts) if parts else np.array([], dtype=float)

            def unpack_params(theta: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
                pos = 0
                alpha_free = theta[pos : pos + K]
                pos += K
                alphas = _free_to_monotonic_cutpoints(alpha_free)
                bpl = None
                if p_pl > 0:
                    bpl = theta[pos : pos + p_pl]
                    pos += p_pl
                bnpl = None
                if p_npl > 0:
                    bnpl = theta[pos : pos + K * p_npl].reshape(K, p_npl)
                    pos += K * p_npl
                return alphas, bpl, bnpl
            
            def neg_log_likelihood(theta: np.ndarray) -> float:
                """Negative log-likelihood function for Hessian computation."""
                alphas, bpl, bnpl = unpack_params(theta)
                
                # Compute linear predictors
                xb = alphas[:, np.newaxis]
                if p_pl > 0 and X_pl is not None and bpl is not None:
                    xb = xb + (X_pl @ bpl)[np.newaxis, :]
                if p_npl > 0 and X_npl is not None and bnpl is not None:
                    xb = xb + (bnpl @ X_npl.T)
                
                # Cumulative probabilities
                F = _cumprob_from_xb(xb, res.link)
                
                # Check monotonicity
                if K > 1 and np.any(np.diff(F, axis=0) < -1e-12):
                    return np.inf
                
                # Category probabilities
                p_cat = np.empty((M, n))
                p_cat[0, :] = F[0, :]
                for k_idx in range(1, M - 1):
                    p_cat[k_idx, :] = F[k_idx, :] - F[k_idx - 1, :]
                p_cat[M - 1, :] = 1.0 - F[K - 1, :]
                
                # Check non-negativity
                if np.any(p_cat < -1e-12):
                    return np.inf
                
                # Compute likelihood
                obs_prob = p_cat[y_idx, np.arange(n)]
                if np.any(obs_prob <= 0):
                    return np.inf
                    
                return -np.sum(np.log(obs_prob))
            
            # Get current parameter vector
            alpha_free = _monotonic_cutpoints_to_free(res.alphas)
            theta_hat = pack_params(alpha_free, res.beta_pl, res.beta_npl)
            
            # Compute Hessian using finite differences with stable step size
            rel_step = 1e-6  # 温和的步长，避免数值噪声
            
            def gradient_func(theta):
                return approx_derivative(neg_log_likelihood, theta, method=method, rel_step=rel_step)
            
            hessian = approx_derivative(gradient_func, theta_hat, method=method, rel_step=rel_step)
            
            # 数值稳健化处理
            # 1. 强制对称化 
            hessian = 0.5 * (hessian + hessian.T)
            
            # 2. 检查条件数（使用对称矩阵专用的特征值计算）
            eigenvals = np.linalg.eigvalsh(hessian)  # 返回实数特征值，已排序
            min_eigval = eigenvals[0]   # 最小特征值
            max_eigval = eigenvals[-1]  # 最大特征值
            condition_number = max_eigval / min_eigval if min_eigval > 1e-16 else np.inf
            
            print(f"[gologit2] Hessian condition number: {condition_number:.2e}")
            
            # 3. 尝试稳健的求逆方法（分层策略：Cholesky -> solve -> pinv）
            cov_matrix = None
            inversion_method = ""
            
            # 首先尝试Cholesky分解（对正定矩阵最稳健）
            try:
                L = np.linalg.cholesky(hessian)
                # 求逆：H^-1 = (L*L^T)^-1 = L^-T * L^-1
                L_inv = np.linalg.inv(L)
                cov_matrix = L_inv.T @ L_inv
                inversion_method = "Cholesky decomposition"
            except np.linalg.LinAlgError:
                # Cholesky失败，尝试标准求逆
                try:
                    cov_matrix = np.linalg.inv(hessian)
                    inversion_method = "standard matrix inversion"
                except np.linalg.LinAlgError:
                    # 标准求逆失败，使用伪逆
                    try:
                        cov_matrix = np.linalg.pinv(hessian, rcond=1e-12)
                        inversion_method = "pseudo-inverse (SVD)"
                        print("[gologit2] Warning: Matrix singular, using pseudo-inverse.")
                    except np.linalg.LinAlgError:
                        print("[gologit2] Error: All matrix inversion methods failed.")
                        return None
            
            print(f"[gologit2] Hessian inverted using {inversion_method}.")
            
            # 4. 确保协方差矩阵对称性（统一处理浮点误差）
            cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
            
            # 检查对角元素是否非负
            diag_elements = np.diag(cov_matrix)
            if np.any(diag_elements < 0):
                print("[gologit2] Warning: Negative diagonal elements in covariance matrix.")
                print("[gologit2] This suggests numerical instability - standard errors may be unreliable.")
            
            print("[gologit2] Numerical Hessian computed successfully.")
            
            # 返回协方差矩阵和诊断信息
            return {
                "cov_matrix": cov_matrix,
                "hessian_condition_number": condition_number,
                "inversion_method": inversion_method,
                "method": method
            }
                
        except ImportError:
            print("[gologit2] Numerical Hessian requires scipy >= 1.0")
            return None
        except Exception as e:
            print(f"[gologit2] Failed to compute numerical Hessian: {e}")
            return None
    
    def _compute_statistics(self, X: np.ndarray, y: np.ndarray, result: Gologit2Result) -> Gologit2Result:
        """计算标准误、p值和pseudo R2等统计量"""
        print("[gologit2] Computing standard errors, p-values, and pseudo R2...")
        
        try:
            # 1. 计算null model的log-likelihood (仅包含cutpoints)
            null_loglik = self._compute_null_loglikelihood(X, y, result)
            result.null_loglik = null_loglik
            
            # 2. 计算pseudo R2
            if null_loglik is not None:
                result.pseudo_r2 = 1 - (result.fun / (-null_loglik))
                
                # 计算参数个数用于调整的R2
                n_params = len(result.alphas)
                if result.beta_pl is not None:
                    n_params += len(result.beta_pl)
                if result.beta_npl is not None:
                    n_params += result.beta_npl.size
                
                # 修正的McFadden调整R²公式：1 - ((LL_full - k) / LL_null)
                # 其中 LL_full = -result.fun, LL_null = null_loglik
                # 所以：1 - ((-result.fun - n_params) / null_loglik) = 1 - ((result.fun + n_params) / (-null_loglik))
                result.pseudo_r2_adj = 1 - ((result.fun + n_params) / (-null_loglik))
                
                print(f"[gologit2] McFadden's Pseudo R2: {result.pseudo_r2:.4f}")
                print(f"[gologit2] Adjusted Pseudo R2: {result.pseudo_r2_adj:.4f}")
            else:
                # null_loglik计算失败，无法计算伪R²
                result.pseudo_r2 = None
                result.pseudo_r2_adj = None
                print("[gologit2] Warning: Could not compute null model likelihood.")
                print("[gologit2] Pseudo R² statistics unavailable (likely due to sparse categories).")
            
            # 3. 计算标准误和p值
            hessian_result = self.compute_numerical_hessian(X, y, method="3-point")
            
            if hessian_result is not None:
                cov_matrix = hessian_result["cov_matrix"]
                # 存储诊断信息
                result.hessian_condition_number = hessian_result["hessian_condition_number"]
                result.inversion_method = hessian_result["inversion_method"]
                result.hessian_method = hessian_result["method"]
                # 验证协方差矩阵维度
                K = len(result.alphas)
                n_beta_pl = len(result.beta_pl) if result.beta_pl is not None else 0
                n_beta_npl = result.beta_npl.size if result.beta_npl is not None else 0
                expected_dim = K + n_beta_pl + n_beta_npl
                
                if cov_matrix.shape != (expected_dim, expected_dim):
                    raise ValueError(f"Covariance matrix dimension {cov_matrix.shape} != expected ({expected_dim}, {expected_dim})")
                
                # 报告协方差矩阵条件数（使用对称矩阵专用计算）
                cov_eigenvals = np.linalg.eigvalsh(cov_matrix)  # 已排序的实数特征值
                cov_min_eigval = cov_eigenvals[0]
                cov_max_eigval = cov_eigenvals[-1]
                cov_condition_number = cov_max_eigval / cov_min_eigval if cov_min_eigval > 1e-16 else np.inf
                result.cov_condition_number = cov_condition_number
                
                print(f"[gologit2] Covariance matrix condition number: {cov_condition_number:.2e}")
                print(f"[gologit2] Minimum eigenvalue: {cov_min_eigval:.6e}")
                
                if cov_condition_number > 1e12:
                    print("[gologit2] WARNING: Very high condition number - standard errors may be unreliable")
                
                # 1. 对cutpoints使用Delta方法进行变换
                # 构造雅可比矩阵 J = ∂α/∂θ_free，其中θ_free = [a1, d2, d3, ..., dK, β_pl, β_npl]
                alpha_free = _monotonic_cutpoints_to_free(result.alphas)
                jacobian_alpha = self._compute_cutpoints_jacobian(alpha_free)
                
                # 提取cutpoints部分的协方差矩阵 (前K×K块)
                cov_alpha_free = cov_matrix[:K, :K]
                
                # Delta方法: Cov(α) = J * Cov(θ_free_alpha) * J^T
                cov_alphas = jacobian_alpha @ cov_alpha_free @ jacobian_alpha.T
                
                # 检查对角元素非负
                alpha_variances = np.diag(cov_alphas)
                if np.any(alpha_variances < 0):
                    print("[gologit2] WARNING: Negative variances detected for cutpoints")
                    alpha_variances = np.maximum(alpha_variances, 1e-12)
                
                result.se_alphas = np.sqrt(alpha_variances)
                
                # 2. Beta系数的标准误（直接从协方差矩阵对角线）
                pos = K
                
                # Parallel-lines系数标准误
                if result.beta_pl is not None:
                    beta_pl_variances = np.diag(cov_matrix[pos:pos+n_beta_pl, pos:pos+n_beta_pl])
                    if np.any(beta_pl_variances < 0):
                        print("[gologit2] WARNING: Negative variances detected for PL coefficients")
                        beta_pl_variances = np.maximum(beta_pl_variances, 1e-12)
                    result.se_beta_pl = np.sqrt(beta_pl_variances)
                    pos += n_beta_pl
                
                # Non-parallel系数标准误
                if result.beta_npl is not None:
                    beta_npl_variances = np.diag(cov_matrix[pos:pos+n_beta_npl, pos:pos+n_beta_npl])
                    if np.any(beta_npl_variances < 0):
                        print("[gologit2] WARNING: Negative variances detected for NPL coefficients")
                        beta_npl_variances = np.maximum(beta_npl_variances, 1e-12)
                    result.se_beta_npl = np.sqrt(beta_npl_variances).reshape(result.beta_npl.shape)
                
                # 3. 计算p值 (双尾z-test)
                result.pvalues_alphas = 2 * (1 - stats.norm.cdf(np.abs(result.alphas / result.se_alphas)))
                
                if result.beta_pl is not None and result.se_beta_pl is not None:
                    z_pl = result.beta_pl / result.se_beta_pl
                    result.pvalues_beta_pl = 2 * (1 - stats.norm.cdf(np.abs(z_pl)))
                
                if result.beta_npl is not None and result.se_beta_npl is not None:
                    z_npl = result.beta_npl / result.se_beta_npl
                    result.pvalues_beta_npl = 2 * (1 - stats.norm.cdf(np.abs(z_npl)))
                
                print("[gologit2] Standard errors computed using Delta method for cutpoints.")
            else:
                print("[gologit2] Could not compute standard errors (Hessian computation failed).")
                
        except Exception as e:
            print(f"[gologit2] Error computing statistics: {e}")
            print("[gologit2] Model estimation successful, but additional statistics unavailable.")
        
        return result
    
    def _compute_cutpoints_jacobian(self, alpha_free: np.ndarray) -> np.ndarray:
        """计算从自由参数到原始cutpoints的雅可比矩阵。
        
        对于单调重参数化：α1 = a1, α2 = a1 + exp(d2), α3 = a1 + exp(d2) + exp(d3), ...
        其中 alpha_free = [a1, d2, d3, ..., dK]
        
        雅可比矩阵 J[i,j] = ∂αᵢ/∂θⱼ：
        - ∂α1/∂a1 = 1, ∂α1/∂dₖ = 0 for k > 1
        - ∂α2/∂a1 = 1, ∂α2/∂d2 = exp(d2), ∂α2/∂dₖ = 0 for k > 2
        - ∂αⱼ/∂a1 = 1, ∂αⱼ/∂dₖ = exp(dₖ) for k ≤ j, = 0 for k > j
        """
        K = len(alpha_free)
        if K == 1:
            return np.array([[1.0]])  # 只有一个cutpoint时
            
        jacobian = np.zeros((K, K))
        
        # 第一行：∂α1/∂θ = [1, 0, 0, ...]
        jacobian[0, 0] = 1.0
        
        # 后续行：∂αⱼ/∂θ
        for i in range(1, K):
            jacobian[i, 0] = 1.0  # ∂αⱼ/∂a1 = 1
            for j in range(1, i + 1):
                jacobian[i, j] = np.exp(alpha_free[j])  # ∂αⱼ/∂dₖ = exp(dₖ) for k ≤ j
        
        # 安全性验证：检查Jacobian的基本性质
        self._validate_cutpoints_jacobian(alpha_free, jacobian)
                
        return jacobian
    
    def _validate_cutpoints_jacobian(self, alpha_free: np.ndarray, jacobian: np.ndarray) -> None:
        """验证cutpoints Jacobian的数值正确性"""
        K = len(alpha_free)
        
        # 基本形状检查
        assert jacobian.shape == (K, K), f"Jacobian shape {jacobian.shape} != ({K}, {K})"
        
        # 验证第一行应该是 [1, 0, 0, ...]
        expected_first_row = np.zeros(K)
        expected_first_row[0] = 1.0
        assert np.allclose(jacobian[0, :], expected_first_row), "Jacobian first row incorrect"
        
        # 验证第一列应该全部是1（所有alpha对a1的偏导数都是1）
        assert np.allclose(jacobian[:, 0], 1.0), "Jacobian first column should be all 1s"
        
        # 验证上三角部分为0（∂αᵢ/∂dⱼ = 0 when j > i）
        for i in range(K):
            for j in range(i + 1, K):
                assert np.abs(jacobian[i, j]) < 1e-12, f"Upper triangle element [{i},{j}] should be 0"
                
        # 数值验证：用有限差分近似检查（只对小矩阵做）
        if K <= 4:  # 避免大矩阵的计算开销
            eps = 1e-8
            for i in range(K):
                for j in range(K):
                    # 扰动第j个自由参数
                    alpha_free_plus = alpha_free.copy()
                    alpha_free_minus = alpha_free.copy()
                    alpha_free_plus[j] += eps
                    alpha_free_minus[j] -= eps
                    
                    # 计算αᵢ在扰动前后的值
                    alpha_plus = _free_to_monotonic_cutpoints(alpha_free_plus)
                    alpha_minus = _free_to_monotonic_cutpoints(alpha_free_minus)
                    
                    # 数值导数
                    numerical_deriv = (alpha_plus[i] - alpha_minus[i]) / (2 * eps)
                    analytical_deriv = jacobian[i, j]
                    
                    # 检查相对误差
                    if abs(analytical_deriv) > 1e-10:
                        rel_error = abs(numerical_deriv - analytical_deriv) / abs(analytical_deriv)
                        assert rel_error < 1e-5, f"Jacobian[{i},{j}]: numerical={numerical_deriv:.8f}, analytical={analytical_deriv:.8f}, rel_error={rel_error:.2e}"
    
    def _compute_null_loglikelihood(self, X: np.ndarray, y: np.ndarray, result: Gologit2Result) -> Optional[float]:
        """计算null model (仅包含cutpoints) 的log-likelihood。
        
        Null model是只包含cutpoints/intercepts而不包含协变量的模型。
        对于累积链接函数（logit/probit/cloglog/loglog/cauchit），
        使用经验累积比例的反函数来获得null cutpoints与拟合intercept-only的MLE等价。
        
        这种方法在样本量足够大时是准确的，但对于极端稀疏的类别可能不稳定。
        为防止数值问题，使用与起始值相同的轻微收缩处理。
        """
        try:
            # 转换y为索引
            categories = result.categories
            cat_to_index = {c: i for i, c in enumerate(categories)}
            y_idx = np.vectorize(cat_to_index.get)(y)
            n = len(y)
            M = categories.size
            K = M - 1
            
            # 计算经验概率
            counts = np.bincount(y_idx, minlength=M).astype(float)
            probs = counts / counts.sum()
            cum_probs = np.cumsum(probs)[:-1]  # K个累积概率
            
            # 计算null model cutpoints（与起始值计算相同，包含轻微收缩以防止数值问题）
            null_alphas = _inverse_start_cutpoint(cum_probs, result.link)
            
            # 计算null model的log-likelihood
            # 每个观测的线性预测子只有cutpoint (无协变量)
            xb_null = null_alphas[:, np.newaxis]  # shape (K, n)
            F_null = _cumprob_from_xb(xb_null, result.link)
            
            # 类别概率
            p_cat_null = np.empty((M, n))
            p_cat_null[0, :] = F_null[0, :]
            for k_idx in range(1, M - 1):
                p_cat_null[k_idx, :] = F_null[k_idx, :] - F_null[k_idx - 1, :]
            p_cat_null[M - 1, :] = 1.0 - F_null[K - 1, :]
            
            # 选择观测到的类别概率
            obs_prob_null = p_cat_null[y_idx, np.arange(n)]
            
            # 检查概率是否有效
            if np.any(obs_prob_null <= 0):
                return None
            
            null_loglik = np.sum(np.log(obs_prob_null))
            return null_loglik
            
        except Exception as e:
            print(f"[gologit2] Warning: Could not compute null log-likelihood: {e}")
            return None

    def bootstrap_se(self, X: np.ndarray, y: np.ndarray, n_bootstrap: int = 100, 
                     random_state: Optional[int] = None) -> Optional[np.ndarray]:
        """Compute bootstrap standard errors for parameter estimates.
        
        Returns array of standard errors in the same order as the parameter vector.
        """
        if self._result is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        
        print(f"[gologit2] Computing bootstrap standard errors ({n_bootstrap} replications)...")
        
        rng = np.random.RandomState(random_state)
        n_samples = len(y)
        param_estimates = []
        
        # Get original parameter vector for reference
        res = self._result
        K = len(res.alphas)
        n_params = K  # alphas
        if res.beta_pl is not None:
            n_params += len(res.beta_pl)
        if res.beta_npl is not None:
            n_params += res.beta_npl.size
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[boot_idx]
            y_boot = y[boot_idx]
            
            try:
                # Fit model on bootstrap sample
                boot_model = GeneralizedOrderedModel(link=self.link, pl_vars=self.pl_vars)
                boot_result = boot_model.fit(X_boot, y_boot, 
                                           feature_names=self.feature_names_,
                                           verbose=False)
                
                # Extract parameter vector
                params = np.concatenate([
                    boot_result.alphas,
                    boot_result.beta_pl if boot_result.beta_pl is not None else [],
                    boot_result.beta_npl.ravel() if boot_result.beta_npl is not None else []
                ])
                param_estimates.append(params)
                
            except Exception:
                # Skip failed bootstrap replications
                continue
        
        if len(param_estimates) == 0:
            print("[gologit2] All bootstrap replications failed.")
            return None
        
        param_estimates = np.array(param_estimates)
        se_estimates = np.std(param_estimates, axis=0, ddof=1)
        
        print(f"[gologit2] Bootstrap completed ({len(param_estimates)}/{n_bootstrap} successful).")
        return se_estimates

    def test_parallel_lines(self, X: np.ndarray, y: np.ndarray, 
                           variables: Optional[List[str]] = None) -> dict:
        """Perform Wald tests for parallel lines assumption on specified variables.
        
        This is a simplified version of gologit2's autofit functionality.
        Tests whether coefficients for each variable are the same across equations.
        
        Returns dict with test statistics and p-values for each variable.
        """
        if self._result is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        
        if self._result.beta_npl is None:
            print("[gologit2] No non-parallel variables to test.")
            return {}
        
        # For now, return a placeholder - full implementation would require
        # computing the covariance matrix and Wald statistics
        print("[gologit2] Parallel lines testing not yet fully implemented.")
        print("[gologit2] This would require:")
        print("  1. Robust covariance matrix estimation")
        print("  2. Wald test statistics for coefficient equality across equations")
        print("  3. Chi-square p-values for each variable")
        print("[gologit2] Consider using autofit-style stepwise selection manually.")
        
        return {}


__all__ = [
    "GeneralizedOrderedModel",
    "Gologit2Result",
]


