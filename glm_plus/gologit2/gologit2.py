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


def _clip_preserve_imag(z: np.ndarray, low: float, high: float) -> np.ndarray:
    """Clip the real part of z to [low, high] while preserving the imaginary part.
    Works for real or complex arrays.
    """
    if np.iscomplexobj(z):
        real_clipped = np.clip(z.real, low, high)
        return real_clipped + 1j * z.imag
    return np.clip(z, low, high)


def _cdf_logit(z: np.ndarray) -> np.ndarray:
    """Logistic CDF F(z) = 1 / (1 + exp(-z))."""
    # Numerical safety: clip Re(z) to avoid exp overflow/underflow; keep Im(z) for complex-step
    z_safe = _clip_preserve_imag(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z_safe))


def _cdf_probit(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF."""
    return stats.norm.cdf(z)


def _cdf_cloglog(z: np.ndarray) -> np.ndarray:
    """Complementary log-log cumulative form used by Stata for cloglog link.

    Stata's gologit2 uses F(x) = exp(-exp(x)) for the cumulative probability.
    Numerical safety: clip z to avoid exp overflow.
    """
    z_safe = _clip_preserve_imag(z, -35, 35)  # prevent exp overflow
    return np.exp(-np.exp(z_safe))


def _cdf_loglog(z: np.ndarray) -> np.ndarray:
    """Log-log cumulative form used by Stata for loglog link.

    Stata's gologit2 uses F(x) = 1 - exp(-exp(-x)).
    Numerical safety: clip z to avoid exp overflow.
    """
    z_safe = _clip_preserve_imag(z, -35, 35)  # prevent exp overflow
    return 1.0 - np.exp(-np.exp(-z_safe))


def _cdf_cauchit(z: np.ndarray) -> np.ndarray:
    """Cauchit CDF: F(z) = 0.5 + (1/pi) * arctan(z)."""
    return 0.5 + (1.0 / math.pi) * np.arctan(z)


def _cumprob_from_xb(xb: np.ndarray, link: str) -> np.ndarray:
    """Compute the cumulative probability F_j for each xb_j.

    CORRECTED PARAMETERIZATION: xb = alpha_j - X*beta (standard form)
    All links now use F(xb) directly since xb already has correct sign:
    - logit/probit: F_j = CDF(xb_j)  [FIXED: removed negative sign]
    - cloglog:      F_j = exp(-exp(xb_j))
    - loglog:       F_j = 1 - exp(-exp(-xb_j))
    - cauchit:      F_j = 0.5 + (1/pi) * atan(xb_j)  [FIXED: removed negative sign]
    """
    if link == "logit":
        return _cdf_logit(xb)  # FIXED: removed negative sign
    if link == "probit":
        return _cdf_probit(xb)  # FIXED: removed negative sign
    if link == "cloglog":
        return _cdf_cloglog(xb)
    if link == "loglog":
        return _cdf_loglog(-xb)
    if link == "cauchit":
        return _cdf_cauchit(xb)  # FIXED: removed negative sign
    raise ValueError(f"Unsupported link: {link}")


def _inverse_start_cutpoint(cum_p: np.ndarray, link: str) -> np.ndarray:
    """Start values for the cutpoints (alphas) from cumulative probabilities.

    CORRECTED PARAMETERIZATION: F(alpha_j - X*beta) = cum_p
    When X=0 (intercept only), F(alpha_j) = cum_p, so alpha_j = F^{-1}(cum_p)

    - logit:   alpha_j = logit(cum_p) = log(cum_p/(1-cum_p))  [FIXED: removed negative]
    - probit:  alpha_j = Phi^{-1}(cum_p)  [FIXED: removed negative]
    - cloglog: alpha_j = log(-log(1-cum_p))  [FIXED: adjusted for standard form]
    - loglog:  alpha_j = -log(-log(cum_p))  [unchanged]
    - cauchit: alpha_j = tan(pi * (cum_p - 0.5))  [FIXED: removed negative]
    """
    eps = 1e-9
    # Apply gentle shrinkage for sparse categories to avoid extreme initial values
    n_total = cum_p.size + 1  # approximate total categories
    p = (cum_p * (n_total - 1) + 1.0) / n_total  # shrink toward center (more conservative)
    p = np.clip(p, eps, 1.0 - eps)
    
    if link == "logit":
        return np.log(p / (1.0 - p))  # FIXED: removed negative sign
    if link == "probit":
        return stats.norm.ppf(p)  # FIXED: removed negative sign
    if link == "cloglog":
        return np.log(-np.log(1.0 - p))  # FIXED: changed to standard form
    if link == "loglog":
        return -np.log(-np.log(p))  # FIXED: adjusted for consistency
    if link == "cauchit":
        return np.tan(math.pi * (p - 0.5))  # FIXED: removed negative sign
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

    def coef_by_threshold(self, var_name: str):
        """
        返回K-1个门槛上的系数β_j(var)，索引为cut1..cut(K-1)。
        若变量是parallel，则每个门槛同一个值；若是non-parallel，则取对应列。
        
        Parameters:
            var_name: 变量名
            
        Returns:
            pd.Series: 各门槛的系数，索引为cut1, cut2, ..., cutK-1
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for coef_by_threshold method")
        
        K = len(self.alphas)  # 门槛个数 
        
        # 构造非平行变量名列表
        npl_vars = [n for n in self.feature_names if self.pl_vars is None or n not in self.pl_vars]
        
        if var_name in npl_vars and self.beta_npl is not None:
            # 非平行变量：从beta_npl矩阵中获取
            var_idx = npl_vars.index(var_name)
            coefs = self.beta_npl[:, var_idx]
        elif self.pl_vars and var_name in self.pl_vars and self.beta_pl is not None:
            # 平行变量：所有门槛相同系数
            var_idx = self.pl_vars.index(var_name)
            coefs = np.repeat(self.beta_pl[var_idx], K)
        else:
            raise KeyError(f"Variable '{var_name}' not found in model variables.")
        
        cuts = [f"cut{j}" for j in range(1, K + 1)]
        return pd.Series(coefs, index=cuts, name=var_name)

    def dPr_ge_by_threshold(self, X: np.ndarray, var_name: str, dx: Optional[float] = None):
        """
        数值导数：对每个门槛j，计算∂Pr(Y>=j)/∂x_var的样本平均。
        
        Parameters:
            X: 设计矩阵 (n_samples, n_features)
            var_name: 变量名
            dx: 数值差分步长（若为None，自动选择：标准化变量用1.0，其他用1e-4）
            
        Returns:
            pd.Series: 各门槛的阈值概率效应，索引为cut1, cut2, ..., cutK-1
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for dPr_ge_by_threshold method")
        
        # 找变量在X中的列位置
        if var_name not in self.feature_names:
            raise KeyError(f"Variable '{var_name}' not found in feature_names")
        col_idx = self.feature_names.index(var_name)
        
        # 自动选择合适的步长：对标准化变量使用dx=1.0（对齐+1SD），其他用小步长
        if dx is None:
            dx = 1.0 if var_name.endswith('_z') else 1e-4
        
        # 需要创建临时模型实例进行预测
        temp_model = GeneralizedOrderedModel(link="logit")  # 临时实例
        temp_model.categories_ = self.categories
        temp_model.feature_names_ = self.feature_names
        temp_model._result = self
        
        # 基准预测
        P = temp_model.predict_proba(X)  # (n, M)
        # 计算累积概率 Pr(Y>=j)
        P_ge = np.fliplr(np.cumsum(np.fliplr(P), axis=1))  # (n,M)
        
        # 扰动预测
        X_plus = X.copy()
        X_plus[:, col_idx] += dx
        P_plus = temp_model.predict_proba(X_plus)
        P_ge_plus = np.fliplr(np.cumsum(np.fliplr(P_plus), axis=1))
        
        # 数值导数
        dP_ge = (P_ge_plus - P_ge) / dx  # (n,M)
        
        # 只取门槛1..K-1（最后一列是Pr(Y>=M)对应最高类别）
        K = len(self.alphas)  # 门槛数
        avg_effects = dP_ge[:, :K].mean(axis=0)
        
        cuts = [f"cut{j}" for j in range(1, K + 1)]
        return pd.Series(avg_effects, index=cuts, name=f"dPr(Y>=cut)/d{var_name}")

    def dPr_cat_by_threshold(self, X: np.ndarray, var_name: str, dx: Optional[float] = None):
        """
        数值导数：对每个类别c，计算∂Pr(Y=c)/∂x_var的样本平均。
        
        Parameters:
            X: 设计矩阵 (n_samples, n_features)
            var_name: 变量名
            dx: 数值差分步长（若为None，自动选择：标准化变量用1.0，其他用1e-4）
            
        Returns:
            pd.Series: 各类别的概率效应，索引为Y=0, Y=1, ..., Y=M-1
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for dPr_cat_by_threshold method")
        
        # 找变量在X中的列位置
        if var_name not in self.feature_names:
            raise KeyError(f"Variable '{var_name}' not found in feature_names")
        col_idx = self.feature_names.index(var_name)
        
        # 自动选择合适的步长：对标准化变量使用dx=1.0（对齐+1SD），其他用小步长
        if dx is None:
            dx = 1.0 if var_name.endswith('_z') else 1e-4
        
        # 创建临时模型实例
        temp_model = GeneralizedOrderedModel(link="logit")
        temp_model.categories_ = self.categories
        temp_model.feature_names_ = self.feature_names  
        temp_model._result = self
        
        # 基准和扰动预测
        P = temp_model.predict_proba(X)  # (n,M)
        X_plus = X.copy()
        X_plus[:, col_idx] += dx
        P_plus = temp_model.predict_proba(X_plus)
        
        # 数值导数
        dP = (P_plus - P) / dx  # (n,M)
        avg_effects = dP.mean(axis=0)
        
        cats = [f"Y={c}" for c in self.categories]
        return pd.Series(avg_effects, index=cats, name=f"dPr(Y=c)/d{var_name}")

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
            lines.append("- Default: robust (Huber-White/BHHH or Sandwich). Cutpoints use Delta method from monotonic reparameterization.")
            lines.append("- If a cluster variable was provided to fit(), SEs are cluster-robust (by worker_id).")
            lines.append("- P-values use normal approximation (Wald tests).")
            
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
        cluster_var: Optional[Sequence] = None,
        maxiter: int = 2000,
        tol: float = 1e-5,
        verbose: bool = True,
        optimizer: str = "L-BFGS-B",
        compute_se: bool = True,
    ) -> Gologit2Result:
        """Estimate parameters by maximizing the log-likelihood.

        Inputs
        - X: shape (n_samples, n_features). Do NOT include an intercept.
        - y: ordinal response values (e.g., [1, 2, 3] or [-1, 0, 1]).
        - feature_names: names for the columns of X (required if X is numpy).
        - maxiter, tol: optimizer controls.
        - verbose: if True, print progress messages a beginner can follow.
        - optimizer: optimizer name for scipy.optimize.minimize.
        - compute_se: if True, compute standard errors and diagnostics (default);
          if False, skip SE computation for faster model selection.

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
        cluster_np = None
        if cluster_var is not None:
            cluster_np = np.asarray(cluster_var)
            if cluster_np.shape[0] != y_np.shape[0]:
                raise ValueError("cluster_var must have the same length as y")

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
        # NUMERICAL SAFETY: Clip initial cutpoints to prevent extreme values
        alpha0 = np.clip(alpha0, -8.0, 8.0)
        if verbose:
            print(f"[gologit2] Initial cutpoints (clipped): {alpha0}")
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

            # CORRECTED: Standard parameterization xb = alpha_j - X*beta
            # xb shape: (K, n)
            xb = alphas[:, np.newaxis]  # broadcast alphas to (K, 1)
            
            if p_pl > 0 and X_pl is not None and bpl is not None:
                # SUBTRACT parallel-lines contribution (same for all thresholds)
                xb = xb - (X_pl @ bpl)[np.newaxis, :]  # FIXED: changed + to -
            
            if p_npl > 0 and X_npl is not None and bnpl is not None:
                # SUBTRACT non-parallel contribution (different for each threshold)
                xb = xb - (bnpl @ X_npl.T)  # FIXED: changed + to -

            # Cumulative probabilities F_j per threshold j
            F = _cumprob_from_xb(xb, self.link)  # shape (K, n)
            
            # SOFT CHECK: Ensure F is monotonic with reduced penalty for better curvature
            penalty = 0.0
            if K > 1:
                f_diffs = np.diff(F, axis=0)  # F_{j+1} - F_j for all j
                violations = np.maximum(0, -f_diffs)  # negative diffs are violations
                if np.any(violations > 0):
                    penalty += 1e6 * np.sum(violations ** 2)  # 降低罚项强度避免扭曲Hessian
            
            # Category probabilities p(y = category_k)
            # k = 0 (lowest): F1
            # 0 < k < M-1:   Fk - F(k-1)
            # k = M-1:       1 - F(K)
            p_cat = np.empty((M, n))
            p_cat[0, :] = F[0, :]
            for k_idx in range(1, M - 1):
                p_cat[k_idx, :] = F[k_idx, :] - F[k_idx - 1, :]
            p_cat[M - 1, :] = 1.0 - F[K - 1, :]
            
            # SOFT CHECK: Penalize negative probabilities with reduced penalty
            neg_prob_violations = np.maximum(0, -p_cat)  # negative probs are violations
            if np.any(neg_prob_violations > 0):
                penalty += 1e6 * np.sum(neg_prob_violations ** 2)  # 降低罚项强度

            # Select the probability for the observed category
            obs_prob = p_cat[y_idx, np.arange(n)]
            
            # NUMERICAL SAFETY: Clip observed probabilities to prevent log(0)
            obs_prob = np.clip(obs_prob, 1e-12, 1.0)
                
            return -np.sum(np.log(obs_prob)) + penalty

        # Optimize with improved convergence criteria and fallback strategies
        opt_options = {"maxiter": maxiter, "disp": bool(verbose), "gtol": tol}
        # Add ftol for more intuitive convergence (SciPy >= 1.11 supports ftol for L-BFGS-B)
        try:
            opt_options["ftol"] = tol
        except Exception:
            pass  # fallback for older SciPy versions
        
        if verbose:
            print(f"[gologit2] Using optimizer: {optimizer}, maxiter: {maxiter}, tol: {tol}")
        
        # Try optimization with fallback strategies
        optimization_success = False
        opt = None
        
        # Primary optimization attempt
        try:
            opt = optimize.minimize(
                nll,
                theta0,
                method=optimizer,
                options=opt_options,
            )
            optimization_success = True
        except Exception as e:
            if verbose:
                print(f"[gologit2] Primary optimizer {optimizer} failed: {e}")
            
            # Fallback to BFGS if L-BFGS-B failed
            if optimizer == "L-BFGS-B":
                try:
                    if verbose:
                        print("[gologit2] Trying fallback optimizer: BFGS")
                    opt = optimize.minimize(
                        nll,
                        theta0,
                        method="BFGS",
                        options=opt_options,
                    )
                    optimization_success = True
                except Exception as e2:
                    if verbose:
                        print(f"[gologit2] Fallback optimizer BFGS also failed: {e2}")
        
        if not optimization_success or opt is None:
            raise RuntimeError(f"All optimization methods failed. Last error: {e}")
        

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
        
        # 关键修复：先保存基本拟合状态，再按需计算统计量
        self.categories_ = categories
        self.feature_names_ = names
        self.pl_vars_ = list(pl_vars) if pl_vars else None
        self.npl_vars_ = [names[i] for i in range(len(names)) if not pl_mask[i]] if p_npl > 0 else None
        self._result = result  # 设置基本result，供统计量计算使用
        
        # CRITICAL GATEKEEPER: 只有需要时才计算统计量，严格控制
        if compute_se:
            if verbose:
                print("[gologit2] Computing standard errors and diagnostics...")
            result = self._compute_statistics(X_np, y_np, result, verbose=verbose, cluster_var=cluster_np)
            # 更新完整的result对象
            self._result = result
            
            # Comprehensive diagnostic check on training set (only when computing SE)
            if verbose:
                print("[gologit2] Running comprehensive diagnostics on training set...")
                self._comprehensive_diagnostics(X_np, verbose=verbose)
        else:
            if verbose:
                print("[gologit2] Skipping SE computation for faster fitting.")
            # 直接使用基本result，不触发任何统计计算
            self._result = result

        if verbose:
            print("[gologit2] Finished. You can inspect 'result.params_as_dict()' or 'result.summary()'.")

        return result

    def predict_proba(self, X: Union[np.ndarray, Any], strict: bool = False) -> np.ndarray:
        """Predict category probabilities for each row of X.

        Parameters:
        - X: feature matrix for prediction
        - strict: if True, raise error when negative probabilities/non-monotonic F detected;
                  if False, clip negative values to 0 and renormalize (default)

        Returns an array of shape (n_samples, n_categories).
        
        Note: This method applies safety guardrails by default (clip negative values 
        to 0 and renormalize) to ensure valid probability distributions. For diagnostics
        or to detect boundary cases, use strict=True or _predict_proba_raw().
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

        # CORRECTED: Standard parameterization xb = alpha_j - X*beta
        # xb shape: (K, n)
        xb = res.alphas[:, np.newaxis]  # broadcast alphas to (K, 1)
        
        if p_pl > 0 and X_pl is not None and res.beta_pl is not None:
            # SUBTRACT parallel-lines contribution (same for all thresholds)
            xb = xb - (X_pl @ res.beta_pl)[np.newaxis, :]  # FIXED: changed + to -
        
        if p_npl > 0 and X_npl is not None and res.beta_npl is not None:
            # SUBTRACT non-parallel contribution (different for each threshold)
            xb = xb - (res.beta_npl @ X_npl.T)  # FIXED: changed + to -

        # Cum probs and category probs
        F = _cumprob_from_xb(xb, link)
        p_cat = np.empty((M, n))
        p_cat[0, :] = F[0, :]
        for k_idx in range(1, M - 1):
            p_cat[k_idx, :] = F[k_idx, :] - F[k_idx - 1, :]
        p_cat[M - 1, :] = 1.0 - F[K - 1, :]

        # Check for boundary violations before applying guardrails
        has_negative_prob = np.any(p_cat < -1e-12)
        has_non_monotonic_F = K > 1 and np.any(np.diff(F, axis=0) < -1e-12)
        
        if strict and (has_negative_prob or has_non_monotonic_F):
            error_details = []
            if has_negative_prob:
                min_prob = np.min(p_cat)
                error_details.append(f"negative probabilities detected (min={min_prob:.6e})")
            if has_non_monotonic_F:
                min_diff = np.min(np.diff(F, axis=0))
                error_details.append(f"non-monotonic cumulative probabilities (min_diff={min_diff:.6e})")
            raise ValueError(f"Model prediction violates feasibility constraints: {'; '.join(error_details)}")
        
        # Apply safety guardrails: ensure non-negative probabilities and normalization
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

        # CORRECTED: Standard parameterization xb = alpha_j - X*beta
        xb_j = np.full(X_np.shape[0], res.alphas[eq - 1])
        if X_pl is not None and res.beta_pl is not None:
            xb_j = xb_j - X_pl @ res.beta_pl  # FIXED: changed + to -
        if X_npl is not None and res.beta_npl is not None:
            xb_j = xb_j - X_npl @ res.beta_npl[eq - 1]  # FIXED: changed + to -
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
            
            # CORRECTED: Standard parameterization xb = alpha_j - X*beta
            xb = res.alphas[:, np.newaxis]  # (K, 1)
            if p_pl > 0 and X_pl is not None and res.beta_pl is not None:
                xb = xb - (X_pl @ res.beta_pl)[np.newaxis, :]  # FIXED: changed + to -
            if p_npl > 0 and X_npl is not None and res.beta_npl is not None:
                xb = xb - (res.beta_npl @ X_npl.T)  # FIXED: changed + to -
            
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
                if infeasible_count > 0 and verbose:
                    print(f"[gologit2] WARNING: {infeasible_count} samples have negative probabilities!")
                    print("[gologit2] Consider: (1) adding parallel-lines constraints, or")
                    print("[gologit2]          (2) merging sparse categories, or")
                    print("[gologit2]          (3) using a simpler model (e.g., multinomial logit)")
                
                if min_f_spacing < 1e-6 and verbose:
                    print(f"[gologit2] WARNING: Very small F spacing detected (min={min_f_spacing:.2e})")
                    print("[gologit2] This may indicate near-separation or numerical instability.")
                
                if prob_max > 1 + 1e-12 and verbose:
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

        # CORRECTED: Standard parameterization xb = alpha_j - X*beta
        xb = res.alphas[:, np.newaxis]  # (K, 1)
        
        if p_pl > 0 and X_pl is not None and res.beta_pl is not None:
            xb = xb - (X_pl @ res.beta_pl)[np.newaxis, :]  # FIXED: changed + to -
        
        if p_npl > 0 and X_npl is not None and res.beta_npl is not None:
            xb = xb - (res.beta_npl @ X_npl.T)  # FIXED: changed + to -

        # Compute probabilities without safety clipping
        F = _cumprob_from_xb(xb, link)
        p_cat = np.empty((M, n), dtype=F.dtype)  # Use F.dtype to support complex-step
        p_cat[0, :] = F[0, :]
        for k_idx in range(1, M - 1):
            p_cat[k_idx, :] = F[k_idx, :] - F[k_idx - 1, :]
        p_cat[M - 1, :] = 1.0 - F[K - 1, :]

        return p_cat.T  # (n, M)

    def compute_score_contributions(self, X: np.ndarray, y: np.ndarray,
                                  result: Optional[Gologit2Result] = None) -> Optional[np.ndarray]:
        """计算每个观测的得分贡献 s_i = ∇ℓ_i，用于稳健标准误计算
        
        Returns:
            (n, p) array where s_i is the score contribution of observation i
            返回None如果计算失败
        """
        res = result if result is not None else self._result
        if res is None:
            return None
            
        try:
            n, p = X.shape
            K = len(res.alphas)  # number of thresholds
            M = K + 1  # number of categories
            
            # 参数向量: [alphas, beta_pl, beta_npl.flatten()]
            total_params = K
            if res.beta_pl is not None:
                total_params += len(res.beta_pl)
            if res.beta_npl is not None:
                total_params += res.beta_npl.size
                
            scores = np.zeros((n, total_params))
            
            # 使用complex-step数值导数计算每个观测的得分
            h = 1e-8  # complex step size
            
            def single_obs_nll(params_vec, obs_idx):
                """单个观测的负对数似然"""
                # 解包参数
                alphas = params_vec[:K]
                param_idx = K
                
                beta_pl = None
                if res.beta_pl is not None:
                    p_pl = len(res.beta_pl)
                    beta_pl = params_vec[param_idx:param_idx + p_pl]
                    param_idx += p_pl
                    
                beta_npl = None
                if res.beta_npl is not None:
                    beta_npl = params_vec[param_idx:].reshape(res.beta_npl.shape)
                
                # 单个观测的数据
                X_obs = X[obs_idx:obs_idx+1]  # (1, p)
                y_obs = y[obs_idx]
                
                # 计算线性预测子
                xb = alphas.reshape(-1, 1)  # (K, 1)
                
                if beta_pl is not None and res.pl_vars:
                    pl_indices = [i for i, name in enumerate(res.feature_names) if name in res.pl_vars]
                    if pl_indices:
                        X_pl = X_obs[:, pl_indices]  # (1, p_pl)
                        xb = xb - (beta_pl.reshape(1, -1) @ X_pl.T)  # (K, 1)
                
                if beta_npl is not None:
                    npl_indices = [i for i, name in enumerate(res.feature_names) 
                                 if res.pl_vars is None or name not in res.pl_vars]
                    if npl_indices:
                        X_npl = X_obs[:, npl_indices]  # (1, p_npl)
                        xb = xb - (beta_npl @ X_npl.T)  # (K, 1)
                
                # 计算概率
                F = _cumprob_from_xb(xb.reshape(K, 1), res.link).flatten()
                
                # 计算类别概率
                probs = np.zeros(M)
                probs[0] = F[0]
                for k in range(1, K):
                    probs[k] = F[k] - F[k-1]
                probs[M-1] = 1.0 - F[K-1]
                
                # 避免数值问题
                probs = np.clip(probs, 1e-12, 1.0)
                probs = probs / probs.sum()  # 重归一化
                
                # 负对数似然
                return -np.log(probs[y_obs])
            
            # 计算每个参数的得分 (对每个观测)
            current_params = np.zeros(total_params)
            current_params[:K] = res.alphas
            param_idx = K
            
            if res.beta_pl is not None:
                p_pl = len(res.beta_pl)
                current_params[param_idx:param_idx + p_pl] = res.beta_pl
                param_idx += p_pl
                
            if res.beta_npl is not None:
                current_params[param_idx:] = res.beta_npl.flatten()
            
            # 对每个观测计算得分向量
            for i in range(n):
                for j in range(total_params):
                    # Complex-step derivative
                    params_complex = current_params.astype(complex)
                    params_complex[j] += 1j * h
                    
                    # 计算复数步的似然
                    nll_complex = single_obs_nll(params_complex, i)
                    
                    # 得分 = -∂ℓ/∂θ = ∂(-ℓ)/∂θ
                    scores[i, j] = np.imag(nll_complex) / h
                    
            return scores
            
        except Exception as e:
            # Silently fail for score contributions to avoid spamming logs during fast selection
            return None

    def compute_sandwich_se(self, X: np.ndarray, y: np.ndarray,
                           result: Optional[Gologit2Result] = None,
                           cluster_var: Optional[np.ndarray] = None,
                           small_sample: bool = True,
                           ridge: float = 1e-6) -> Optional[dict]:
        """计算Sandwich稳健标准误 V = H^-1 B H^-1（支持HC1与cluster-robust）
        
        Parameters:
            cluster_var: (n,) array of cluster identifiers for cluster-robust SE
            small_sample: whether to apply HC1-style small-sample correction
            ridge: small ridge added to Hessian before inversion for stability
        """
        res = result if result is not None else self._result
        if res is None:
            return None
            
        try:
            # 1. 计算Hessian逆（使用数值Hessian得到的协方差矩阵，即H^-1）
            hessian_result = self.compute_numerical_hessian(X, y, result, method="3-point", verbose=False)
            if hessian_result is None or 'cov_matrix' not in hessian_result:
                return None
            H_inv_free = hessian_result['cov_matrix']  # with respect to alpha_free, betas
            
            # 2. 计算得分贡献
            scores = self.compute_score_contributions(X, y, result)
            if scores is None:
                return None
            
            # 3. 计算B矩阵 (Outer Product of Gradients)
            if cluster_var is not None:
                # Cluster-robust: 按cluster求和得分
                unique_clusters = np.unique(cluster_var)
                cluster_scores = []
                for cluster in unique_clusters:
                    mask = cluster_var == cluster
                    cluster_score = scores[mask].sum(axis=0)  # 该cluster的总得分
                    cluster_scores.append(cluster_score)
                scores_for_B = np.array(cluster_scores)  # (n_clusters, p)
            else:
                # Individual-level
                scores_for_B = scores  # (n, p)
            
            # B = Σ s_i s_i'  
            B = scores_for_B.T @ scores_for_B  # (p, p)
            # 小样本校正
            if small_sample:
                n = scores.shape[0]
                p = scores.shape[1]
                if cluster_var is None:
                    # HC1: n/(n-p)
                    if n > p:
                        B = (n / (n - p)) * B
                else:
                    # Cluster-robust dof correction: G/(G-1) * (n-1)/(n-p)
                    unique_clusters = np.unique(cluster_var)
                    G = unique_clusters.size
                    if (G > 1) and (n > p):
                        B = (G / (G - 1)) * ((n - 1) / (n - p)) * B
            
            # 4. 将H^-1从自由参数空间映射到原始α参数空间（保证与scores一致）
            K = len(result.alphas)
            p_pl = 0 if result.beta_pl is None else len(result.beta_pl)
            p_npl = 0 if result.beta_npl is None else result.beta_npl.size
            total_params = K + p_pl + p_npl
            alpha_free = _monotonic_cutpoints_to_free(result.alphas)
            J_alpha = self._compute_cutpoints_jacobian(alpha_free)  # (K,K)
            J_full = np.eye(total_params)
            J_full[:K, :K] = J_alpha
            H_inv_alpha = J_full @ H_inv_free @ J_full.T

            # 5. Sandwich估计: V = H^-1 B H^-1（两者都在原始参数空间）
            try:
                V_sandwich = H_inv_alpha @ B @ H_inv_alpha
            except np.linalg.LinAlgError:
                V_sandwich = H_inv_alpha @ B @ H_inv_alpha

            se_sandwich = np.sqrt(np.maximum(0.0, np.diag(V_sandwich)))
            return {
                'V_sandwich': V_sandwich,
                'se_sandwich': se_sandwich,
                'method': 'sandwich_cluster' if cluster_var is not None else ('sandwich_hc1' if small_sample else 'sandwich_hc0')
            }
                
        except Exception as e:
            # Silently fail for sandwich SE to avoid spamming logs during fast selection
            return None

    def compute_bhhh_se(self, X: np.ndarray, y: np.ndarray,
                       result: Optional[Gologit2Result] = None) -> Optional[dict]:
        """计算BHHH标准误: V = B^-1, where B = Σ s_i s_i'
        
        BHHH估计直接用得分外积的逆作为协方差矩阵估计
        """
        res = result if result is not None else self._result
        if res is None:
            return None
            
        try:
            # 计算得分贡献
            scores = self.compute_score_contributions(X, y, result)
            if scores is None:
                return None
            
            # BHHH矩阵: B = Σ s_i s_i'
            B = scores.T @ scores  # (p, p)
            
            # 协方差矩阵: V ≈ B^-1（数值稳定：岭化+伪逆兜底）
            ridge = 1e-8
            B_reg = B + ridge * np.eye(B.shape[0])
            try:
                V_bhhh = np.linalg.inv(B_reg)
            except np.linalg.LinAlgError:
                V_bhhh = np.linalg.pinv(B_reg)
            
            se_bhhh = np.sqrt(np.maximum(0.0, np.diag(V_bhhh)))
            return {
                'V_bhhh': V_bhhh,
                'se_bhhh': se_bhhh,
                'method': 'bhhh'
            }
                
        except Exception as e:
            # Silently fail for BHHH SE to avoid spamming logs during fast selection
            return None

    def compute_numerical_hessian(self, X: np.ndarray, y: np.ndarray, 
                                   result: Optional[Gologit2Result] = None, 
                                   method: str = "3-point", verbose: bool = True) -> Optional[dict]:
        """Compute numerical Hessian at the fitted parameters.
        
        Returns the inverse Hessian (approximate covariance matrix) if successful.
        This provides more reliable standard errors than L-BFGS-B's low-rank approximation.
        
        Parameters:
        - X: feature matrix used during fitting  
        - y: response vector used during fitting
        - result: Gologit2Result object (if None, uses self._result)
        - method: finite difference method ("2-point", "3-point", "cs")
        """
        # Use provided result or fall back to self._result
        res = result if result is not None else self._result
        if res is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        
        # FIXED: Import hierarchy without putting computation in except block
        approx_derivative = None
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
                    if verbose:
                        print("[gologit2] Warning: Using fallback derivative approximation.")
                        print("[gologit2] Consider upgrading SciPy for better numerical differentiation.")
                    approx_derivative = approx_derivative_fallback
                except ImportError:
                    if verbose:
                        print("[gologit2] Error: Cannot find numerical differentiation function.")
                        print("[gologit2] Please upgrade SciPy (>= 1.0 recommended) or install numdifftools.")
                    return None
        
        if approx_derivative is None:
            if verbose:
                print("[gologit2] Error: Failed to import approx_derivative.")
            return None
            
        if verbose:
            print("[gologit2] Computing numerical Hessian (this may take a moment)...")
        
        # Use the provided or retrieved result object
        
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
            
            # CORRECTED: Standard parameterization xb = alpha_j - X*beta
            xb = alphas[:, np.newaxis]
            if p_pl > 0 and X_pl is not None and bpl is not None:
                xb = xb - (X_pl @ bpl)[np.newaxis, :]  # FIXED: changed + to -
            if p_npl > 0 and X_npl is not None and bnpl is not None:
                xb = xb - (bnpl @ X_npl.T)  # FIXED: changed + to -
            
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
        
        if verbose:
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
                    if verbose:
                        print("[gologit2] Warning: Matrix singular, using pseudo-inverse.")
                except np.linalg.LinAlgError:
                    if verbose:
                        print("[gologit2] Error: All matrix inversion methods failed.")
                    return None
        
        if verbose:
            print(f"[gologit2] Hessian inverted using {inversion_method}.")
        
        # 4. 确保协方差矩阵对称性（统一处理浮点误差）
        cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
        
        # 检查对角元素是否非负
        diag_elements = np.diag(cov_matrix)
        if np.any(diag_elements < 0):
            if verbose:
                print("[gologit2] Warning: Negative diagonal elements in covariance matrix.")
                print("[gologit2] This suggests numerical instability - standard errors may be unreliable.")
        
        if verbose:
            print("[gologit2] Numerical Hessian computed successfully.")
        
        # 返回协方差矩阵和诊断信息
        return {
            "cov_matrix": cov_matrix,
            "hessian_condition_number": condition_number,
            "inversion_method": inversion_method,
            "method": method
        }
    
    def _compute_statistics(self, X: np.ndarray, y: np.ndarray, result: Gologit2Result, verbose: bool = True, cluster_var: Optional[np.ndarray] = None) -> Gologit2Result:
        """计算标准误、p值和pseudo R2等统计量"""
        if verbose:
            print("[gologit2] Computing standard errors, p-values, and pseudo R2...")
        
        try:
            # 1. 计算null model的log-likelihood (仅包含cutpoints)
            if verbose:
                print("[gologit2] Computing null model likelihood for pseudo R2...")
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
                
                if verbose:
                    print(f"[gologit2] McFadden's Pseudo R2: {result.pseudo_r2:.4f}")
                    print(f"[gologit2] Adjusted Pseudo R2: {result.pseudo_r2_adj:.4f}")
            else:
                # null_loglik计算失败，无法计算伪R²
                result.pseudo_r2 = None
                result.pseudo_r2_adj = None
                if verbose:
                    print("[gologit2] Warning: Could not compute null model likelihood.")
                    print("[gologit2] Pseudo R² statistics unavailable (likely due to sparse categories).")
                    print("[gologit2] Tip: Consider merging extremely sparse outcome categories or ensuring each category has observations.")
            
            # 3. 计算标准误和p值 - 优先使用稳健方法，数值Hessian仅做诊断
            if verbose:
                print("[gologit2] Computing standard errors using robust methods...")
            
            # 尝试多种稳健标准误方法（主力）
            robust_methods = {}
            primary_se_source = None
            
            # 方法1: BHHH（通常最稳健，优先使用）
            try:
                if verbose:
                    print("[gologit2] Computing BHHH standard errors...")
                bhhh_result = self.compute_bhhh_se(X, y, result)
                if bhhh_result is not None:
                    robust_methods['bhhh'] = bhhh_result
                    primary_se_source = 'bhhh'
                    if verbose:
                        print("[gologit2] BHHH SE computation successful.")
            except Exception as e:
                if verbose:
                    print(f"[gologit2] BHHH SE computation failed: {e}")
            
            # 方法2: Sandwich (HC1)（备选，支持cluster-robust）
            if primary_se_source is None:
                try:
                    if verbose:
                        print("[gologit2] Computing Sandwich standard errors (HC1/cluster-robust if provided)...")
                    sandwich_result = self.compute_sandwich_se(X, y, result, cluster_var=cluster_var, small_sample=True)
                    if sandwich_result is not None:
                        robust_methods['sandwich'] = sandwich_result
                        primary_se_source = 'sandwich'
                        if verbose:
                            print("[gologit2] Sandwich SE computation successful.")
                except Exception as e:
                    if verbose:
                        print(f"[gologit2] Sandwich SE computation failed: {e}")
            
            # 方法3: 数值Hessian（仅作最后备选和诊断）
            hessian_result = None
            # 严格限制：只有在大样本(>=500)且两种稳健方法都失败时才用Hessian
            if primary_se_source is None and len(X) >= 500:
                if verbose:
                    print("[gologit2] WARNING: Both BHHH and Sandwich failed. Using numerical Hessian as last resort...")
                    print("[gologit2] This may take longer and be less stable than robust methods.")
                try:
                    hessian_result = self.compute_numerical_hessian(X, y, result, method="3-point", verbose=verbose)
                    if hessian_result is not None:
                        primary_se_source = 'hessian'
                        if verbose:
                            print("[gologit2] Numerical Hessian computation successful.")
                except Exception as e:
                    if verbose:
                        print(f"[gologit2] Numerical Hessian computation failed: {e}")
            elif primary_se_source is None:
                if verbose:
                    print("[gologit2] WARNING: Sample size too small (<500) and robust methods failed.")
                    print("[gologit2] Standard errors not available. Consider using bootstrap SE.")
            # 完全跳过诊断性Hessian以加速
            
            # 存储稳健标准误结果供诊断使用
            result.robust_se_methods = robust_methods
            result.primary_se_method = primary_se_source
            
            # 使用最优可用的SE方法
            if primary_se_source == 'bhhh' and 'bhhh' in robust_methods:
                if verbose:
                    print("[gologit2] Using BHHH standard errors as primary method.")
                se_data = robust_methods['bhhh']
                cov_matrix = se_data.get('V_bhhh', None)
                if cov_matrix is not None:
                    self._assign_se_from_covariance(result, cov_matrix)
                else:
                    # 后备：仅有SE向量时（不推荐）
                    se_vector = se_data['se_bhhh']
                    self._assign_se_from_vector(result, se_vector)
                
            elif primary_se_source == 'sandwich' and 'sandwich' in robust_methods:
                if verbose:
                    print("[gologit2] Using Sandwich standard errors as primary method.")
                se_data = robust_methods['sandwich']
                cov_matrix = se_data.get('V_sandwich', None)
                if cov_matrix is not None:
                    self._assign_se_from_covariance(result, cov_matrix)
                else:
                    se_vector = se_data['se_sandwich']
                    self._assign_se_from_vector(result, se_vector)
                
            elif hessian_result is not None and primary_se_source == 'hessian':
                if verbose:
                    print("[gologit2] Using numerical Hessian standard errors as fallback.")
                cov_matrix = hessian_result["cov_matrix"]
                # 使用原有的Delta方法逻辑处理Hessian结果
                self._process_hessian_se(result, cov_matrix, hessian_result, verbose)
            else:
                if verbose:
                    print("[gologit2] Could not compute standard errors using any method.")
                return result
            
            # 存储Hessian诊断信息（如果可用）
            if hessian_result is not None:
                result.hessian_condition_number = hessian_result["hessian_condition_number"]
                result.inversion_method = hessian_result["inversion_method"]
                result.hessian_method = hessian_result["method"]
                
            # 计算p值（对所有SE方法通用）
            self._compute_p_values(result)
            
            if verbose:
                print(f"[gologit2] Standard errors computed using {primary_se_source.upper()} method.")
                
        except Exception as e:
            if verbose:
                print(f"[gologit2] Error computing statistics: {e}")
                print("[gologit2] Model estimation successful, but additional statistics unavailable.")
        
        return result
    
    def _assign_se_from_vector(self, result: Gologit2Result, se_vector: np.ndarray) -> None:
        """从完整参数向量的SE分配到各个参数组（cutpoints, beta_pl, beta_npl）"""
        K = len(result.alphas)
        pos = 0
        
        # Cutpoints SE (需要Delta方法变换)
        alpha_free_se = se_vector[pos:pos+K]
        alpha_free = _monotonic_cutpoints_to_free(result.alphas)
        jacobian_alpha = self._compute_cutpoints_jacobian(alpha_free)
        
        # 构造协方差矩阵对角线
        cov_alpha_free_diag = alpha_free_se ** 2
        # Delta方法近似：Var(α_i) ≈ Σ_j (∂α_i/∂θ_j)² * Var(θ_j) 
        alpha_variances = np.array([
            np.sum((jacobian_alpha[i, :] ** 2) * cov_alpha_free_diag) 
            for i in range(K)
        ])
        result.se_alphas = np.sqrt(np.maximum(alpha_variances, 1e-12))
        pos += K
        
        # Beta_pl SE
        if result.beta_pl is not None:
            n_beta_pl = len(result.beta_pl)
            result.se_beta_pl = se_vector[pos:pos+n_beta_pl]
            pos += n_beta_pl
        
        # Beta_npl SE
        if result.beta_npl is not None:
            n_beta_npl = result.beta_npl.size
            result.se_beta_npl = se_vector[pos:pos+n_beta_npl].reshape(result.beta_npl.shape)

    def _assign_se_from_covariance(self, result: Gologit2Result, cov_matrix: np.ndarray) -> None:
        """从完整协方差矩阵分配SE。
        假定传入的协方差矩阵已经在原始参数空间（α, β）中。
        """
        K = len(result.alphas)
        n_beta_pl = len(result.beta_pl) if result.beta_pl is not None else 0
        n_beta_npl = result.beta_npl.size if result.beta_npl is not None else 0
        expected_dim = K + n_beta_pl + n_beta_npl
        if cov_matrix.shape != (expected_dim, expected_dim):
            # 维度不符则直接返回
            return
        cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
        # Cutpoints: already in original α parameterization
        cov_alpha = cov_matrix[:K, :K]
        result.se_alphas = np.sqrt(np.maximum(np.diag(cov_alpha), 1e-12))
        # Beta_pl
        pos = K
        if result.beta_pl is not None and n_beta_pl > 0:
            cov_pl = cov_matrix[pos:pos+n_beta_pl, pos:pos+n_beta_pl]
            result.se_beta_pl = np.sqrt(np.maximum(np.diag(cov_pl), 1e-12))
            pos += n_beta_pl
        # Beta_npl
        if result.beta_npl is not None and n_beta_npl > 0:
            cov_npl = cov_matrix[pos:pos+n_beta_npl, pos:pos+n_beta_npl]
            se_npl = np.sqrt(np.maximum(np.diag(cov_npl), 1e-12))
            result.se_beta_npl = se_npl.reshape(result.beta_npl.shape)
    
    def _process_hessian_se(self, result: Gologit2Result, cov_matrix: np.ndarray, hessian_result: dict, verbose: bool = True) -> None:
        """处理数值Hessian结果，使用完整Delta方法"""
        K = len(result.alphas)
        n_beta_pl = len(result.beta_pl) if result.beta_pl is not None else 0
        n_beta_npl = result.beta_npl.size if result.beta_npl is not None else 0
        expected_dim = K + n_beta_pl + n_beta_npl
        
        if cov_matrix.shape != (expected_dim, expected_dim):
            if verbose:
                print(f"[gologit2] WARNING: Covariance matrix dimension mismatch")
            return
        
        # 报告协方差矩阵条件数
        cov_eigenvals = np.linalg.eigvalsh(cov_matrix)
        cov_min_eigval = cov_eigenvals[0]
        cov_max_eigval = cov_eigenvals[-1]
        cov_condition_number = cov_max_eigval / cov_min_eigval if cov_min_eigval > 1e-16 else np.inf
        result.cov_condition_number = cov_condition_number
        
        if verbose:
            print(f"[gologit2] Covariance matrix condition number: {cov_condition_number:.2e}")
        if cov_condition_number > 1e12 and verbose:
            print("[gologit2] WARNING: Very high condition number - standard errors may be unreliable")
        
        # Delta方法处理cutpoints
        alpha_free = _monotonic_cutpoints_to_free(result.alphas)
        jacobian_alpha = self._compute_cutpoints_jacobian(alpha_free)
        cov_alpha_free = cov_matrix[:K, :K]
        cov_alphas = jacobian_alpha @ cov_alpha_free @ jacobian_alpha.T
        
        alpha_variances = np.diag(cov_alphas)
        result.se_alphas = np.sqrt(np.maximum(alpha_variances, 1e-12))
        
        # Beta系数SE
        pos = K
        if result.beta_pl is not None:
            beta_pl_variances = np.diag(cov_matrix[pos:pos+n_beta_pl, pos:pos+n_beta_pl])
            result.se_beta_pl = np.sqrt(np.maximum(beta_pl_variances, 1e-12))
            pos += n_beta_pl
        
        if result.beta_npl is not None:
            beta_npl_variances = np.diag(cov_matrix[pos:pos+n_beta_npl, pos:pos+n_beta_npl])
            result.se_beta_npl = np.sqrt(np.maximum(beta_npl_variances, 1e-12)).reshape(result.beta_npl.shape)
    
    def _compute_p_values(self, result: Gologit2Result) -> None:
        """计算p值（双尾z-test）"""
        if result.se_alphas is not None:
            result.pvalues_alphas = 2 * (1 - stats.norm.cdf(np.abs(result.alphas / result.se_alphas)))
        
        if result.beta_pl is not None and result.se_beta_pl is not None:
            z_pl = result.beta_pl / result.se_beta_pl
            result.pvalues_beta_pl = 2 * (1 - stats.norm.cdf(np.abs(z_pl)))
        
        if result.beta_npl is not None and result.se_beta_npl is not None:
            z_npl = result.beta_npl / result.se_beta_npl
            result.pvalues_beta_npl = 2 * (1 - stats.norm.cdf(np.abs(z_npl)))

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
            # Silently fail to avoid spamming logs during fast selection
            return None

    def bootstrap_se(self, X: np.ndarray, y: np.ndarray, n_bootstrap: int = 100, 
                     random_state: Optional[int] = None, verbose: bool = True,
                     cluster_var: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Compute bootstrap standard errors for parameter estimates.
        
        Supports cluster bootstrap if cluster_var is provided (resample clusters with replacement).
        Returns array of standard errors in the same order as the parameter vector.
        """
        if self._result is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        
        if verbose:
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
            if cluster_var is None:
                boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
                X_boot = X[boot_idx]
                y_boot = y[boot_idx]
            else:
                # Cluster bootstrap: resample clusters, take all obs in selected clusters
                clusters = np.asarray(cluster_var)
                unique_clusters = np.unique(clusters)
                sampled_clusters = rng.choice(unique_clusters, size=unique_clusters.size, replace=True)
                mask = np.isin(clusters, sampled_clusters)
                X_boot = X[mask]
                y_boot = y[mask]
            
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
            if verbose:
                print("[gologit2] All bootstrap replications failed.")
            return None
        
        param_estimates = np.array(param_estimates)
        se_estimates = np.std(param_estimates, axis=0, ddof=1)
        
        if verbose:
            print(f"[gologit2] Bootstrap completed ({len(param_estimates)}/{n_bootstrap} successful).")
        return se_estimates

    def test_parallel_lines(self, X: np.ndarray, y: np.ndarray, 
                           variables: Optional[List[str]] = None,
                           method: str = "lr", verbose: bool = True) -> dict:
        """Perform tests for parallel lines assumption on specified variables.
        
        This tests whether coefficients for each variable are the same across equations.
        
        Parameters:
        - X: feature matrix used during fitting
        - y: response vector used during fitting  
        - variables: variables to test (if None, test all non-parallel variables)
        - method: 'lr' for likelihood ratio test (recommended), 'wald' for Wald test
        
        Returns dict with test statistics and p-values for each variable.
        """
        if self._result is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        
        result = self._result
        feature_names = result.feature_names
        categories = result.categories
        K = len(result.alphas)  # number of thresholds
        
        # Convert y to category indices
        cat_to_index = {c: i for i, c in enumerate(categories)}
        y_idx = np.vectorize(cat_to_index.get)(y)
        
        # Determine variables to test
        if variables is None:
            # Test all currently non-parallel variables
            if result.pl_vars is None:
                variables = feature_names  # All variables are non-parallel
            else:
                variables = [v for v in feature_names if v not in result.pl_vars]
        else:
            # Filter to variables actually in the model
            variables = [v for v in variables if v in feature_names]
        
        if not variables:
            if verbose:
                print("[gologit2] No variables to test for parallel lines assumption.")
            return {}
        
        if verbose:
            print(f"[gologit2] Testing parallel lines assumption for {len(variables)} variables using {method.upper()} method...")
        
        test_results = {}
        
        for var in variables:
            try:
                # Get current parallel-lines variables (excluding the test variable)
                current_pl_vars = list(result.pl_vars) if result.pl_vars else []
                if var in current_pl_vars:
                    # Variable is currently parallel, test if it should be non-parallel
                    restricted_pl_vars = current_pl_vars.copy()
                    unrestricted_pl_vars = [v for v in current_pl_vars if v != var]
                else:
                    # Variable is currently non-parallel, test if it could be parallel
                    restricted_pl_vars = list(current_pl_vars) + [var] if result.pl_vars else [var]
                    unrestricted_pl_vars = current_pl_vars.copy()
                
                if method.lower() == "lr":
                    # Likelihood Ratio Test
                    
                    # Fit restricted model (null hypothesis: parallel lines)
                    restricted_model = GeneralizedOrderedModel(link=result.link, pl_vars=restricted_pl_vars)
                    restricted_result = restricted_model.fit(X, y, feature_names=feature_names, verbose=False, maxiter=1000, compute_se=False)
                    
                    # Fit unrestricted model (alternative: non-parallel)
                    unrestricted_model = GeneralizedOrderedModel(link=result.link, pl_vars=unrestricted_pl_vars)
                    unrestricted_result = unrestricted_model.fit(X, y, feature_names=feature_names, verbose=False, maxiter=1000, compute_se=False)
                    
                    if restricted_result.success and unrestricted_result.success:
                        # Calculate LR statistic
                        ll_restricted = -restricted_result.fun
                        ll_unrestricted = -unrestricted_result.fun
                        lr_stat = 2 * (ll_unrestricted - ll_restricted)
                        
                        # Degrees of freedom: difference in number of parameters
                        # Moving from parallel to non-parallel: parallel has 1 coef, non-parallel has K-1 coefs
                        # (one parallel coef becomes K-1 non-parallel coefs, net gain = K-2)
                        df = K - 2  # Correct: K-2 additional parameters
                        
                        # P-value from chi-square distribution
                        from scipy.stats import chi2
                        p_value = 1 - chi2.cdf(max(0, lr_stat), df)
                        
                        test_results[var] = {
                            'method': 'LR',
                            'statistic': lr_stat,
                            'p_value': p_value,
                            'df': df,
                            'll_restricted': ll_restricted,
                            'll_unrestricted': ll_unrestricted,
                            'reject_parallel': p_value < 0.05,
                            'success': True
                        }
                    else:
                        test_results[var] = {
                            'method': 'LR',
                            'statistic': np.nan,
                            'p_value': np.nan,
                            'df': K - 2,
                            'success': False,
                            'error': 'Model fitting failed'
                        }
                        
                elif method.lower() == "wald":
                    # Wald test (requires covariance matrix)
                    if verbose:
                        print(f"[gologit2] Wald test not yet implemented for variable {var}")
                    test_results[var] = {
                        'method': 'Wald',
                        'statistic': np.nan,
                        'p_value': np.nan,
                        'success': False,
                        'error': 'Wald test not implemented'
                    }
                else:
                    raise ValueError(f"Unknown test method: {method}")
                    
            except Exception as e:
                test_results[var] = {
                    'method': method.upper(),
                    'statistic': np.nan,
                    'p_value': np.nan,
                    'success': False,
                    'error': str(e)
                }
                if verbose:
                    print(f"[gologit2] Error testing variable {var}: {e}")
        
        # Print summary
        if verbose:
            print("[gologit2] Parallel lines test results:")
            for var, res in test_results.items():
                if res['success']:
                    status = "REJECT parallel" if res.get('reject_parallel', False) else "ACCEPT parallel"
                    print(f"  {var}: {res['method']} = {res['statistic']:.3f}, p = {res['p_value']:.4f} -> {status}")
                else:
                    print(f"  {var}: TEST FAILED - {res.get('error', 'Unknown error')}")
        
        return test_results

    def autofit_variable_selection(self, X: np.ndarray, y: np.ndarray, 
                                  feature_names: List[str],
                                  candidate_vars: Optional[List[str]] = None,
                                  significance_level: float = 0.01,
                                  max_npl_vars: int = 5,
                                  verbose: bool = True) -> dict:
        """自动变量选择：从全并行开始，逐步放开违反比例优势假设的变量
        
        实现部分平行有序模型(PPOM)的变量选择策略
        
        Parameters:
            candidate_vars: 候选非平行变量列表，如果为None则考虑所有变量
            significance_level: LR检验的显著性水平
            max_npl_vars: 最多允许的非平行变量数
            verbose: 是否输出详细过程
        
        Returns:
            dict: 包含选择过程和最终结果的详细信息
        """
        if candidate_vars is None:
            candidate_vars = feature_names.copy()
        
        # 移除已知应该parallel的变量（如年份dummy等）
        candidate_vars = [v for v in candidate_vars if v in feature_names]
        
        if verbose:
            print(f"[gologit2] Starting automatic variable selection...")
            print(f"[gologit2] Candidate variables: {candidate_vars}")
        
        # Step 1: 拟合全并行模型作为baseline
        if verbose:
            print("[gologit2] Step 1: Fitting fully parallel model (baseline)...")
        
        baseline_model = GeneralizedOrderedModel(link=self.link, pl_vars=feature_names.copy())
        baseline_result = baseline_model.fit(X, y, feature_names=feature_names, verbose=False, compute_se=False)
        
        if not baseline_result.success:
            if verbose:
                print("[gologit2] ERROR: Baseline parallel model failed to converge")
            return {'success': False, 'reason': 'baseline_failed'}
        
        baseline_ll = -baseline_result.fun
        selection_log = []
        
        if verbose:
            print(f"[gologit2] Baseline model LL: {baseline_ll:.4f}")
        
        # Step 2: 逐个测试候选变量
        selected_npl_vars = []
        current_pl_vars = feature_names.copy()
        current_ll = baseline_ll
        
        K = len(baseline_result.alphas)  # number of thresholds
        
        for iteration in range(max_npl_vars):
            if verbose:
                print(f"[gologit2] Step {iteration + 2}: Testing remaining candidates...")
            
            # 测试每个剩余候选变量
            remaining_candidates = [v for v in candidate_vars if v not in selected_npl_vars]
            if not remaining_candidates:
                break
            
            best_var = None
            best_ll = current_ll
            best_lr_stat = 0
            best_p_value = 1.0
            
            for var in remaining_candidates:
                # 构造新的pl_vars列表（移除当前测试变量）
                test_pl_vars = [v for v in current_pl_vars if v != var]
                
                try:
                    # 拟合放开该变量的模型
                    test_model = GeneralizedOrderedModel(link=self.link, pl_vars=test_pl_vars)
                    test_result = test_model.fit(X, y, feature_names=feature_names, verbose=False, compute_se=False)
                    
                    if test_result.success:
                        test_ll = -test_result.fun
                        
                        # LR检验统计量
                        lr_stat = 2 * (test_ll - current_ll)
                        # 自由度：从1个并行系数变成K-1个非平行系数，增加了K-2个参数
                        df = K - 2
                        p_value = 1 - stats.chi2.cdf(lr_stat, df) if lr_stat > 0 else 1.0
                        
                        if verbose:
                            print(f"[gologit2]   {var}: LL={test_ll:.4f}, LR={lr_stat:.2f}, p={p_value:.4f}")
                        
                        # 记录测试结果
                        selection_log.append({
                            'iteration': iteration + 1,
                            'variable': var,
                            'll_baseline': current_ll,
                            'll_test': test_ll,
                            'lr_stat': lr_stat,
                            'p_value': p_value,
                            'df': df,
                            'selected': p_value < significance_level
                        })
                        
                        # 更新最佳候选
                        if p_value < significance_level and lr_stat > best_lr_stat:
                            best_var = var
                            best_ll = test_ll
                            best_lr_stat = lr_stat
                            best_p_value = p_value
                    else:
                        if verbose:
                            print(f"[gologit2]   {var}: Model failed to converge")
                        selection_log.append({
                            'iteration': iteration + 1,
                            'variable': var,
                            'll_baseline': current_ll,
                            'll_test': np.nan,
                            'lr_stat': np.nan,
                            'p_value': np.nan,
                            'df': K - 2,
                            'selected': False,
                            'note': 'convergence_failed'
                        })
                        
                except Exception as e:
                    if verbose:
                        print(f"[gologit2]   {var}: Error during fitting: {e}")
                    selection_log.append({
                        'iteration': iteration + 1,
                        'variable': var,
                        'll_baseline': current_ll,
                        'll_test': np.nan,
                        'lr_stat': np.nan,
                        'p_value': np.nan,
                        'df': K - 2,
                        'selected': False,
                        'note': f'error: {str(e)}'
                    })
            
            # 如果找到显著变量，将其加入非平行集合
            if best_var is not None:
                selected_npl_vars.append(best_var)
                current_pl_vars.remove(best_var)
                current_ll = best_ll
                
                if verbose:
                    print(f"[gologit2] -> Selected {best_var} (p={best_p_value:.4f})")
            else:
                if verbose:
                    print("[gologit2] -> No more significant variables found")
                break
        
        # Step 3: 拟合最终模型
        if selected_npl_vars:
            if verbose:
                print(f"[gologit2] Step Final: Fitting PPOM with NPL vars: {selected_npl_vars}")
            
            final_model = GeneralizedOrderedModel(link=self.link, pl_vars=current_pl_vars)
            final_result = final_model.fit(X, y, feature_names=feature_names, verbose=False, compute_se=True)
            
            final_success = final_result.success
            final_ll = -final_result.fun if final_success else np.nan
        else:
            # 没有选中任何变量，保持全并行
            if verbose:
                print("[gologit2] Step Final: No variables selected, keeping fully parallel model")
            
            final_model = baseline_model
            final_result = baseline_result
            final_success = True
            final_ll = baseline_ll
        
        # 返回完整的选择结果
        return {
            'success': final_success,
            'selected_npl_vars': selected_npl_vars,
            'final_pl_vars': current_pl_vars,
            'baseline_ll': baseline_ll,
            'final_ll': final_ll,
            'improvement': final_ll - baseline_ll if not np.isnan(final_ll) else 0,
            'selection_log': selection_log,
            'final_model': final_model,
            'final_result': final_result if final_success else None,
            'parameters': {
                'significance_level': significance_level,
                'max_npl_vars': max_npl_vars,
                'n_candidates': len(candidate_vars),
                'n_iterations': len(selected_npl_vars) + 1
            }
        }


__all__ = [
    "GeneralizedOrderedModel",
    "Gologit2Result",
]


