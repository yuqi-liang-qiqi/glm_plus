"""
@Author  : Yuqi Liang 梁彧祺
@File    : ori.py
@Time    : 12/08/2025 21:30
@Desc    : 
Python translation of the OR1 (Ordinal Quantile Regression with 3+ outcomes)
functions from the R package `bqror` (file `ORI.R`).

The goal is to keep the API and behavior close to the R reference while using
common Python scientific libraries (NumPy, SciPy). All functions include clear
English comments that explain inputs, outputs, and the main steps.

Important notes for users new to this code:
- y is assumed to be an integer vector with categories 1..J (as in the R code).
- x is an (n x k) matrix that already includes a column of ones if an
  intercept is needed.
- p is the quantile (or skewness) parameter, 0 < p < 1.
- This file implements: sampling steps, likelihood helpers, DIC, inefficiency
  factor, marginal likelihood, and the main `quantregOR1` function.

Dependencies: numpy, scipy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import math
import numpy as np
from numpy.linalg import inv, cholesky, lstsq
from scipy.stats import norm, truncnorm


# ===============
# Utility helpers
# ===============

def _assert_numeric_array(name: str, arr: np.ndarray) -> None:
    """Validate that an input is a numeric NumPy array.

    Parameters
    - name: descriptive name used in error messages.
    - arr: any array-like input that should contain numeric data.

    Raises
    - TypeError: if the array is not of a numeric dtype.
    """
    if not np.issubdtype(np.asarray(arr).dtype, np.number):
        raise TypeError(f"each entry in {name} must be numeric")


def _unique_sorted_levels(y: np.ndarray) -> np.ndarray:
    """Return sorted unique outcome levels.

    Parameters
    - y: 1D array of length n or 2D array (n x 1) with integer categories 1..J.

    Returns
    - levels: 1D array of sorted unique values in `y`.

    Raises
    - ValueError: if any element of `y` is not an integer (within a tolerance).
    """
    y1 = np.asarray(y).reshape(-1)
    if not np.allclose(y1, np.floor(y1)):
        raise ValueError("each entry of y must be an integer")
    return np.unique(y1)


def _as_matrix(x: np.ndarray) -> np.ndarray:
    """Ensure an array has shape (n, k).

    Parameters
    - x: array-like. If 1D, it will be reshaped to (n, 1).

    Returns
    - mat: NumPy array with at least 2 dimensions.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Numerically safe natural logarithm.

    Parameters
    - x: array of values to take log of.
    - eps: lower bound used to clip `x` before logging.

    Returns
    - logx: element-wise log(clip(x, eps, inf)).
    """
    return np.log(np.clip(x, eps, None))


def _logsumexp(values: np.ndarray) -> float:
    """Stable log-sum-exp for 1D array-like input.

    Parameters
    - values: 1D array-like of log-values.

    Returns
    - logsumexp: scalar log( sum_i exp(values[i]) ).
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size == 0:
        return -np.inf
    vmax = np.max(v)
    if not np.isfinite(vmax):
        return vmax
    return float(vmax + np.log(np.sum(np.exp(v - vmax))))


def _logmeanexp(values: np.ndarray) -> float:
    """Stable log-mean-exp for 1D array-like input.

    Returns log( mean_i exp(values[i]) ).
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size == 0:
        return -np.inf
    return float(_logsumexp(v) - math.log(v.size))


def alcdfstd(x: float, p: float) -> float:
    """CDF of a standard Asymmetric Laplace distribution AL(0, 1, p).

    Parameters
    - x: scalar evaluation point.
    - p: quantile/skewness parameter in (0, 1).

    Returns
    - cdf: probability P(X <= x) for X ~ AL(0, 1, p).

    Raises
    - ValueError: if `p` is not in (0, 1).
    """
    if not (0 < p < 1):
        raise ValueError("parameter p must be between 0 to 1")
    if x <= 0:
        return float(p * math.exp((1.0 - p) * x))
    else:
        return float(1.0 - (1.0 - p) * math.exp(-p * x))


def alcdf(x: float, mu: float, sigma: float, p: float) -> float:
    """CDF of an Asymmetric Laplace distribution AL(mu, sigma, p).

    Parameters
    - x: scalar evaluation point.
    - mu: location parameter.
    - sigma: positive scale parameter.
    - p: quantile/skewness parameter in (0, 1).

    Returns
    - cdf: probability P(X <= x) for X ~ AL(mu, sigma, p).

    Raises
    - ValueError: if `p` is not in (0, 1) or `sigma` <= 0.
    """
    if not (0 < p < 1):
        raise ValueError("parameter p must be between 0 to 1")
    if sigma <= 0:
        raise ValueError("parameter sigma must be positive")
    z = (x - mu) / sigma
    if z <= 0:
        return float(p * math.exp((1.0 - p) * z))
    else:
        return float(1.0 - (1.0 - p) * math.exp(-p * z))


def _mvn_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Log-density of a multivariate normal distribution.

    Parameters
    - x: (d,) array, evaluation point.
    - mean: (d,) mean vector.
    - cov: (d, d) symmetric positive-definite covariance matrix.

    Returns
    - logpdf: scalar log f(x | mean, cov).
    """
    x = np.asarray(x).reshape(-1)
    mean = np.asarray(mean).reshape(-1)
    cov = np.asarray(cov)
    d = x.shape[0]
    L = np.linalg.cholesky(cov)
    # Solve L v = x - mean
    v = np.linalg.solve(L, x - mean)
    quad = v @ v
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)


# ==============================================
# GIG(lambda=1/2) sampling via Inverse-Gaussian
# ==============================================

def _sample_inverse_gaussian(mu: float, lam: float) -> float:
    """Sample from an Inverse-Gaussian distribution IG(mu, lambda).

    Parameters
    - mu: mean parameter (> 0).
    - lam: shape parameter lambda (> 0).

    Returns
    - sample: one draw from IG(mu, lambda) via the Michael–Schucany–Haas method.

    Raises
    - ValueError: if `mu` or `lam` are not positive.
    """
    if mu <= 0 or lam <= 0:
        raise ValueError("IG parameters must be positive")
    v = np.random.normal()
    y = v * v
    mu2 = mu * mu
    x = mu + (mu2 * y) / (2.0 * lam) - (mu / (2.0 * lam)) * math.sqrt(4.0 * mu * lam * y + mu2 * y * y)
    u = np.random.rand()
    if u <= mu / (mu + x):
        return x
    else:
        return mu2 / x


def _rgig_lambda_half(chi: float, psi: float) -> float:
    """Sample from a GIG distribution with lambda = +1/2.

    Parameters
    - chi: non-negative parameter (> 0 in practice; small values will be clipped).
    - psi: positive parameter (> 0; small values will be clipped).

    Returns
    - sample: one draw from GIG(1/2, chi, psi).

    Notes
    - Uses the identity: If X ~ GIG(1/2, chi, psi), then Y = 1/X ~ IG( sqrt(psi/chi), psi ).
    """
    eps = 1e-12
    chi = max(chi, eps)
    psi = max(psi, eps)
    mu_y = math.sqrt(psi / chi)
    lam_y = psi
    y = _sample_inverse_gaussian(mu_y, lam_y)
    return 1.0 / y


# ================================
# Likelihood for the OR1 (J >= 3)
# ================================

def qrnegLogLikensumOR1(y: np.ndarray, x: np.ndarray, beta: np.ndarray, delta: np.ndarray, p: float) -> Dict[str, np.ndarray]:
    """Negative log-likelihood pieces for the OR1 model (J >= 3).

    Parameters
    - y: (n, 1) integer categories in {1, ..., J}.
    - x: (n, k) covariate matrix.
    - beta: (k,) vector of regression coefficients.
    - delta: ((J-2),) vector parameterizing the interior cut-points via exp cum-sums.
    - p: quantile parameter in (0, 1).

    Returns
    - dict with keys:
      - 'nlogl': (n, 1) vector of negative log-likelihood per observation.
      - 'negsumlogl': scalar, the sum over i of nlogl[i].

    Raises
    - ValueError: if `p` is not in (0, 1) or `y` has non-integers.
    """
    if not (0 < p < 1):
        raise ValueError("parameter p must be between 0 to 1")
    y = _as_matrix(y)
    x = _as_matrix(x)
    beta = np.asarray(beta).reshape(-1)
    delta = np.asarray(delta).reshape(-1)

    _assert_numeric_array("delta", delta)
    _assert_numeric_array("x", x)
    _assert_numeric_array("beta", beta)

    J = _unique_sorted_levels(y).shape[0]
    n = x.shape[0]

    # Build cut-points from delta following R code
    expdelta = np.exp(delta)
    q = expdelta.shape[0] + 1  # J-2 + 1 = J-1
    gammacp = np.zeros(q)
    for j in range(2, J):  # j = 2..J-1 (1-based)
        gammacp[j - 1] = float(np.sum(expdelta[: j - 1]))
    # allgammacp = [-Inf, gammacp[1..J-1], Inf], 1-based indexing in R
    allgammacp = np.concatenate((np.array([-np.inf]), gammacp, np.array([np.inf])))  # length J+1

    mu = x @ beta
    lnpdf = np.zeros((n, 1))
    for i in range(n):
        yi = int(y[i, 0])
        meanp = float(mu[i])
        if yi == 1:
            # Guard against underflow: log( max(F, eps) )
            prob = alcdf(0.0, meanp, 1.0, p)
            lnpdf[i, 0] = math.log(max(prob, 1e-300))
        elif yi == J:
            # Guard against round-to-one: log( max(1 - F, eps) )
            tail = 1.0 - alcdf(allgammacp[J - 1], meanp, 1.0, p)
            lnpdf[i, 0] = math.log(max(tail, 1e-300))
        else:
            upper = alcdf(allgammacp[yi], meanp, 1.0, p)      # yi+1 -> index yi (0-based)
            lower = alcdf(allgammacp[yi - 1], meanp, 1.0, p)  # yi   -> index yi-1
            lnpdf[i, 0] = math.log(max(upper - lower, 1e-300))

    nlogl = -lnpdf
    negsumlogl = float(-np.sum(lnpdf))
    return {"nlogl": nlogl, "negsumlogl": negsumlogl}


# =====================================
# Gradient/Hessian-based delta minimizer
# =====================================

def qrminfundtheorem(
    delta_in: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    beta: np.ndarray,
    cri0: float,
    cri1: float,
    stepsize: float,
    maxiter: int,
    h: float,
    dh: float,
    sw: int,
    p: float,
) -> Dict[str, np.ndarray]:
    """Minimize the negative log-likelihood with respect to delta.

    Uses finite-difference gradients and Hessian to update `delta` using a
    convex combination of BHHH and Newton steps (following the R function).

    Parameters
    - delta_in: initial delta vector ((J-2),).
    - y: (n, 1) integer categories 1..J.
    - x: (n, k) covariate matrix.
    - beta: (k,) coefficient vector used during optimization.
    - cri0: initial convergence criterion value.
    - cri1: stopping threshold; smaller means stricter convergence.
    - stepsize: learning rate for parameter updates.
    - maxiter: maximum number of iterations.
    - h: finite-difference step for first derivatives.
    - dh: finite-difference step for second derivatives.
    - sw: iteration number at which to switch from BHHH to Newton steps.
    - p: quantile parameter in (0, 1).

    Returns
    - dict with keys:
      - 'deltamin': minimizing delta vector.
      - 'negsum': negative sum of log-likelihood at deltamin.
      - 'logl': (n, 1) log-likelihood contributions at deltamin.
      - 'G': (n, d) gradient matrix (per-observation scores).
      - 'H': (d, d) Hessian matrix (symmetric).
    """
    _assert_numeric_array("deltaIn", delta_in)
    y = _as_matrix(y)
    x = _as_matrix(x)
    beta = np.asarray(beta).reshape(-1)
    if not (0 < p < 1):
        raise ValueError("parameter p must be between 0 to 1")

    n = y.shape[0]
    d = int(np.asarray(delta_in).size)
    storevn = np.zeros((n, d))
    storevp = np.zeros((n, d))
    der = np.zeros((n, d))  # individual score rows
    dh2 = dh * dh

    delta = np.asarray(delta_in, dtype=float).reshape(-1)
    cri = cri0
    jj = 0
    while (cri > cri1) and (jj < maxiter):
        jj += 1
        vo = -qrnegLogLikensumOR1(y, x, beta, delta, p)["nlogl"]  # (n,1)
        vo = vo.reshape(-1)
        deltao = delta.copy()

        # First derivatives wrt each delta component
        for i in range(d):
            delta[i] = deltao[i] - h
            vn = -qrnegLogLikensumOR1(y, x, beta, delta, p)["nlogl"].reshape(-1)
            delta[i] = deltao[i] + h
            vp = -qrnegLogLikensumOR1(y, x, beta, delta, p)["nlogl"].reshape(-1)
            delta[i] = deltao[i]

            storevn[:, i] = vn
            storevp[:, i] = vp
            der[:, i] = 0.5 * (vp - vn) / h

        # Hessian construction (symmetric)
        hess = np.zeros((d, d))
        for j in range(d):
            for i in range(j + 1):
                if i == j:
                    delta[i] = deltao[i] + dh
                    vp2 = -qrnegLogLikensumOR1(y, x, beta, delta, p)["nlogl"].reshape(-1)
                    delta[i] = deltao[i] - dh
                    vn2 = -qrnegLogLikensumOR1(y, x, beta, delta, p)["nlogl"].reshape(-1)
                    delta[i] = deltao[i]
                    hess[i, j] = np.sum((vp2 + vn2 - 2.0 * vo) / dh2)
                else:
                    f = [i, j]
                    # (i+dh, j+dh)
                    delta[f] = deltao[f] + dh
                    vpp = -qrnegLogLikensumOR1(y, x, beta, delta, p)["nlogl"].reshape(-1)
                    delta[f] = deltao[f] - dh
                    vnn = -qrnegLogLikensumOR1(y, x, beta, delta, p)["nlogl"].reshape(-1)
                    delta[i] = deltao[i] + dh; delta[j] = deltao[j] - dh
                    vpn = -qrnegLogLikensumOR1(y, x, beta, delta, p)["nlogl"].reshape(-1)
                    delta[i] = deltao[i] - dh; delta[j] = deltao[j] + dh
                    vnp = -qrnegLogLikensumOR1(y, x, beta, delta, p)["nlogl"].reshape(-1)
                    delta[i] = deltao[i]; delta[j] = deltao[j]
                    hess[i, j] = 0.25 * np.sum((vpp + vnn - vpn - vnp) / dh2)

        # Symmetrize Hessian
        hess = np.diag(np.diag(hess)) + (hess + hess.T - np.diag(np.diag(hess)))

        # Convergence criterion
        cri = np.sum(np.abs(np.sum(der, axis=0)))

        # BHHH and Newton steps
        G = der  # (n x d)
        # Solve (G'G) dd = G'1 stacked => dd = (G'G)^{-1} G' 1? In R they use colSums(der)
        gg = G.T @ G
        gsum = np.sum(G, axis=0)
        ddeltabhhh = np.linalg.solve(gg, gsum)
        ddeltahess = np.linalg.solve(-hess, gsum)

        weight = min(1.0, max(0.0, jj - sw))
        ddelta = (1.0 - weight) * ddeltabhhh + weight * ddeltahess
        delta = delta + stepsize * ddelta

    deltamin = delta
    out = qrnegLogLikensumOR1(y, x, beta, deltamin, p)
    logl = -out["nlogl"]  # (n,1)
    negsum = out["negsumlogl"]
    return {"deltamin": deltamin, "negsum": negsum, "logl": logl, "G": der, "H": hess}


# ==========================
# Conditional sampling steps
# ==========================

def drawbetaOR1(
    z: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    tau2: float,
    theta: float,
    invB0: np.ndarray,
    invB0b0: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Sample beta | z, w via a multivariate normal.

    Parameters
    - z: (n, 1) latent continuous responses.
    - x: (n, k) design matrix.
    - w: (n, 1) latent weights from the GIG mixture.
    - tau2: scalar, 2 / (p (1 - p)).
    - theta: scalar, (1 - 2 p) / (p (1 - p)).
    - invB0: (k, k) prior precision matrix.
    - invB0b0: (k,) product invB0 @ b0.

    Returns
    - dict with keys:
      - 'beta': (k,) sampled coefficients.
      - 'Btilde': (k, k) posterior covariance.
      - 'btilde': (k,) posterior mean.
    """
    _assert_numeric_array("z", z)
    _assert_numeric_array("x", x)
    _assert_numeric_array("w", w)
    if not np.isscalar(tau2):
        raise TypeError("parameter tau2 must be scalar")
    if not np.isscalar(theta):
        raise TypeError("parameter theta must be scalar")
    _assert_numeric_array("invB0", invB0)
    _assert_numeric_array("invB0b0", invB0b0)

    z = _as_matrix(z)
    x = _as_matrix(x)
    w = _as_matrix(w)
    n, k = x.shape

    # Build sum of per-observation precision contributions
    var_sum = np.zeros((k, k))
    mean_sum = np.zeros(k)
    for i in range(n):
        xi = x[i, :].reshape(-1, 1)
        wi = float(w[i, 0])
        zi = float(z[i, 0])
        if not np.isfinite(wi) or wi <= 0.0 or not np.isfinite(zi):
            continue
        denom = tau2 * max(wi, 1e-12)
        var_sum += (xi @ xi.T) / denom
        mean_sum += (x[i, :] * (zi - theta * wi)) / denom

    Btilde = inv(invB0 + var_sum)
    btilde = Btilde @ (invB0b0.reshape(-1) + mean_sum)
    # Sample from N(btilde, Btilde) via Cholesky
    L = cholesky(Btilde)
    beta = btilde + L @ np.random.normal(size=k)

    return {"beta": beta.reshape(-1), "Btilde": Btilde, "btilde": btilde.reshape(-1)}


def drawwOR1(
    z: np.ndarray,
    x: np.ndarray,
    beta: np.ndarray,
    tau2: float,
    theta: float,
    indexp: float,
) -> np.ndarray:
    """Sample latent weights w from GIG(1/2, chi_i, psi).

    Parameters
    - z: (n, 1) latent continuous responses.
    - x: (n, k) design matrix.
    - beta: (k,) coefficients.
    - tau2: scalar = 2 / (p (1 - p)).
    - theta: scalar = (1 - 2 p) / (p (1 - p)).
    - indexp: kept for parity with R (always 0.5 here).

    Returns
    - w: (n, 1) vector of sampled latent weights.
    """
    _assert_numeric_array("z", z)
    _assert_numeric_array("x", x)
    _assert_numeric_array("beta", beta)
    if not np.isscalar(tau2):
        raise TypeError("parameter tau2 must be scalar")
    if not np.isscalar(theta):
        raise TypeError("parameter theta must be scalar")
    if not np.isscalar(indexp):
        raise TypeError("parameter indexp must be scalar")

    z = _as_matrix(z)
    x = _as_matrix(x)
    beta = np.asarray(beta).reshape(-1)
    n = x.shape[0]

    tildeeta = (theta * theta) / (tau2) + 2.0
    w = np.zeros((n, 1))
    mu_vec = x @ beta
    for i in range(n):
        zi = float(z[i, 0])
        mui = float(mu_vec[i])
        chi_i = ((zi - mui) ** 2) / tau2
        if not np.isfinite(chi_i) or chi_i <= 0.0:
            chi_i = 1e-12
        w[i, 0] = _rgig_lambda_half(chi_i, tildeeta)
    return w


def drawlatentOR1(
    y: np.ndarray,
    x: np.ndarray,
    beta: np.ndarray,
    w: np.ndarray,
    theta: float,
    tau2: float,
    delta: np.ndarray,
) -> np.ndarray:
    """Sample latent variable z from a univariate truncated normal.

    Parameters
    - y: (n, 1) outcome categories 1..J.
    - x: (n, k) design matrix.
    - beta: (k,) coefficients.
    - w: (n, 1) latent weights.
    - theta: scalar (1 - 2 p) / (p (1 - p)).
    - tau2: scalar 2 / (p (1 - p)).
    - delta: ((J-2),) parameter that defines interior cut-points.

    Returns
    - z: (n, 1) latent continuous responses.
    """
    y = _as_matrix(y)
    x = _as_matrix(x)
    beta = np.asarray(beta).reshape(-1)
    w = _as_matrix(w)
    delta = np.asarray(delta).reshape(-1)
    _assert_numeric_array("x", x)
    _assert_numeric_array("beta", beta)
    _assert_numeric_array("w", w)
    _assert_numeric_array("delta", delta)
    if not np.isscalar(theta):
        raise TypeError("parameter theta must be scalar")
    if not np.isscalar(tau2):
        raise TypeError("parameter tau2 must be scalar")

    J = _unique_sorted_levels(y).shape[0]
    n = x.shape[0]

    # Compute cut-point bounds for each category at current delta
    expdelta = np.exp(delta)
    q = expdelta.shape[0] + 1
    gammacp = np.zeros(q)
    for j in range(2, J):
        gammacp[j - 1] = float(np.sum(expdelta[: j - 1]))
    # Include -inf and +inf for convenience (1-based mapping like R)
    bounds = np.concatenate((np.array([-np.inf]), gammacp, np.array([np.inf])))

    # Vectorized truncated normal sampling via inverse-CDF method
    mu = (x @ beta).reshape(-1)
    wv = w.reshape(-1)
    eps = 1e-12
    wv_safe = np.where(np.isfinite(wv) & (wv > eps), wv, eps)
    meanp = mu + theta * wv_safe
    std = np.sqrt(tau2 * wv_safe)

    yi = y.reshape(-1).astype(int)
    a = bounds[yi - 1]
    b = bounds[yi]

    # Standardized truncation limits
    a_std = (a - meanp) / std
    b_std = (b - meanp) / std

    # Handle infinite bounds naturally: norm.cdf(-inf)=0, norm.cdf(+inf)=1
    cdf_a = norm.cdf(a_std)
    cdf_b = norm.cdf(b_std)
    cdf_a = np.clip(cdf_a, eps, 1.0 - eps)
    cdf_b = np.clip(cdf_b, eps, 1.0 - eps)

    # Guard against extremely narrow intervals due to numerical issues
    width = np.maximum(cdf_b - cdf_a, eps)
    u0 = np.random.random(size=n)
    u = np.clip(cdf_a + u0 * width, eps, 1.0 - eps)
    z_std = norm.ppf(u)

    z = (meanp + std * z_std).reshape(-1, 1)
    return z


def drawdeltaOR1(
    y: np.ndarray,
    x: np.ndarray,
    beta: np.ndarray,
    delta0: np.ndarray,
    d0: np.ndarray,
    D0: np.ndarray,
    tune: float,
    Dhat: np.ndarray,
    p: float,
) -> Dict[str, np.ndarray]:
    """Random-walk Metropolis–Hastings update for delta.

    Parameters
    - y: (n, 1) outcome categories 1..J.
    - x: (n, k) design matrix.
    - beta: (k,) current beta draw.
    - delta0: ((J-2),) current delta.
    - d0: ((J-2),) prior mean for delta.
    - D0: ((J-2), (J-2)) prior covariance for delta.
    - tune: scalar step-size multiplier.
    - Dhat: ((J-2), (J-2)) negative inverse Hessian from maximization (scales proposal).
    - p: quantile parameter in (0, 1).

    Returns
    - dict with keys:
      - 'deltareturn': ((J-2),) accepted (or retained) delta.
      - 'accept': 0/1 acceptance indicator.
    """
    y = _as_matrix(y)
    x = _as_matrix(x)
    beta = np.asarray(beta).reshape(-1)
    delta0 = np.asarray(delta0).reshape(-1)
    d0 = np.asarray(d0).reshape(-1)
    D0 = np.asarray(D0)
    Dhat = np.asarray(Dhat)

    J = _unique_sorted_levels(y).shape[0]
    k = J - 2
    L = cholesky(Dhat)
    proposal = delta0 + tune * (L @ np.random.normal(size=k))

    num = qrnegLogLikensumOR1(y, x, beta, proposal, p)
    den = qrnegLogLikensumOR1(y, x, beta, delta0, p)
    pnum = -num["negsumlogl"] + _mvn_logpdf(proposal, d0, D0)
    pden = -den["negsumlogl"] + _mvn_logpdf(delta0, d0, D0)

    if math.log(np.random.rand()) <= (pnum - pden):
        return {"deltareturn": proposal, "accept": 1}
    else:
        return {"deltareturn": delta0, "accept": 0}


# ==================
# Model diagnostics
# ==================

def dicOR1(
    y: np.ndarray,
    x: np.ndarray,
    betadraws: np.ndarray,
    deltadraws: np.ndarray,
    postMeanbeta: np.ndarray,
    postMeandelta: np.ndarray,
    burn: int,
    mcmc: int,
    p: float,
) -> Dict[str, float]:
    """Compute Deviance Information Criterion (DIC) for OR1.

    Parameters
    - y: (n, 1) outcome categories 1..J.
    - x: (n, k) design matrix.
    - betadraws: (k, nsim) matrix of beta draws.
    - deltadraws: ((J-2), nsim) matrix of delta draws.
    - postMeanbeta: (k,) posterior mean of beta.
    - postMeandelta: ((J-2),) posterior mean of delta.
    - burn: number of initial iterations to discard.
    - mcmc: number of post-burn iterations.
    - p: quantile parameter in (0, 1).

    Returns
    - dict with keys:
      - 'DIC': deviance information criterion.
      - 'pd': effective number of parameters.
      - 'dev': deviance at posterior means.
    """
    y = _as_matrix(y)
    x = _as_matrix(x)
    betadraws = _as_matrix(betadraws)
    deltadraws = _as_matrix(deltadraws)
    postMeanbeta = np.asarray(postMeanbeta).reshape(-1)
    postMeandelta = np.asarray(postMeandelta).reshape(-1)
    nsim = int(burn + mcmc)

    ans = qrnegLogLikensumOR1(y, x, postMeanbeta, postMeandelta, p)
    dev = 2.0 * ans["negsumlogl"]

    post_sims = betadraws[:, burn:nsim].shape[1]
    Deviance = np.zeros(post_sims)
    for i in range(post_sims):
        temp = qrnegLogLikensumOR1(y, x, betadraws[:, burn + i], deltadraws[:, burn + i], p)
        Deviance[i] = 2.0 * temp["negsumlogl"]

    avgdDeviance = float(np.mean(Deviance))
    DIC = 2.0 * avgdDeviance - dev
    pd = avgdDeviance - dev
    return {"DIC": DIC, "pd": pd, "dev": dev}


def _acf(series: np.ndarray, maxlags: int) -> np.ndarray:
    """Compute the sample autocorrelation function up to `maxlags`.

    Parameters
    - series: 1D array of draws.
    - maxlags: maximum lag (inclusive).

    Returns
    - ac: 1D array of length `maxlags + 1`; ac[0] = 1.
    """
    x = np.asarray(series).reshape(-1)
    x = x - np.mean(x)
    n = x.size
    # Full autocorrelation via FFT or direct; use direct for clarity
    denom = np.dot(x, x) / n if n > 0 else 1.0
    ac = np.empty(maxlags + 1)
    ac[0] = 1.0
    for lag in range(1, maxlags + 1):
        if lag >= n:
            ac[lag] = 0.0
        else:
            ac[lag] = np.dot(x[:-lag], x[lag:]) / ((n - lag) * denom)
    return ac


def ineffactorOR1(
    x: np.ndarray,
    betadraws: np.ndarray,
    deltadraws: np.ndarray,
    accutoff: float = 0.05,
    maxlags: int = 400,
    verbose: bool = True,
) -> np.ndarray:
    """Compute inefficiency factors using the batch-means method.

    Parameters
    - x: (n, k) design matrix (used only to extract column names upstream; unused here).
    - betadraws: (k, nsim) beta draws.
    - deltadraws: ((J-2), nsim) delta draws.
    - accutoff: ACF cutoff threshold (default 0.05).
    - maxlags: maximum lag for ACF (default 400).
    - verbose: unused here (kept for parity).

    Returns
    - ineff: (k + J - 2, 1) column vector of inefficiency factors for beta then delta.
    """
    betadraws = _as_matrix(betadraws)
    deltadraws = _as_matrix(deltadraws)
    n = betadraws.shape[1]
    k = betadraws.shape[0]

    ineff_beta = np.zeros((k, 1))
    for i in range(k):
        ac = _acf(betadraws[i, :], maxlags)
        # first lag where acf <= cutoff (excluding lag 0)
        idx = np.where(ac[1:] <= accutoff)[0]
        nlags = int(idx[0] + 1) if idx.size > 0 else maxlags
        nbatch = max(1, n // nlags)
        nuse = nbatch * nlags
        b = betadraws[i, :nuse]
        xbatch = b.reshape(nbatch, nlags).T  # (nlags x nbatch)
        mxbatch = np.mean(xbatch, axis=0)
        varxbatch = float(np.sum((mxbatch - np.mean(b)) ** 2) / (nbatch - 1 if nbatch > 1 else 1.0))
        nse = math.sqrt(varxbatch / nbatch) if nbatch > 0 else np.nan
        rne = (np.std(b, ddof=1) / math.sqrt(nuse)) / nse if nuse > 1 and nse > 0 else np.nan
        ineff_beta[i, 0] = 1.0 / rne if rne and not np.isnan(rne) and rne != 0 else np.nan

    k2 = deltadraws.shape[0]
    ineff_delta = np.zeros((k2, 1))
    for i in range(k2):
        ac = _acf(deltadraws[i, :], maxlags)
        idx = np.where(ac[1:] <= accutoff)[0]
        nlags = int(idx[0] + 1) if idx.size > 0 else maxlags
        nbatch = max(1, n // nlags)
        nuse = nbatch * nlags
        d = deltadraws[i, :nuse]
        xbatch = d.reshape(nbatch, nlags).T
        mxbatch = np.mean(xbatch, axis=0)
        varxbatch = float(np.sum((mxbatch - np.mean(d)) ** 2) / (nbatch - 1 if nbatch > 1 else 1.0))
        nse = math.sqrt(varxbatch / nbatch) if nbatch > 0 else np.nan
        rne = (np.std(d, ddof=1) / math.sqrt(nuse)) / nse if nuse > 1 and nse > 0 else np.nan
        ineff_delta[i, 0] = 1.0 / rne if rne and not np.isnan(rne) and rne != 0 else np.nan

    return np.vstack([ineff_beta, ineff_delta])


def covEffectOR1(
    modelOR1: Dict[str, np.ndarray],
    y: np.ndarray,
    xMat1: np.ndarray,
    xMat2: np.ndarray,
    p: float,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Average covariate effect (ACE) over outcome categories (OR1).

    Parameters
    - modelOR1: dict output from `quantregOR1` containing MCMC draws.
    - y: (n, 1) outcome vector.
    - xMat1: (n, k) baseline design matrix.
    - xMat2: (n, k) modified design matrix (e.g., covariate shifted).
    - p: quantile parameter in (0, 1).
    - verbose: unused here; kept for parity.

    Returns
    - dict with key:
      - 'avgDiffProb': (J, 1) average change in predicted probabilities for categories 1..J.
    """
    xMat1 = _as_matrix(xMat1)
    xMat2 = _as_matrix(xMat2)
    y = _as_matrix(y)
    J = _unique_sorted_levels(y).shape[0]
    if J <= 3:
        raise ValueError("This function is only available for models with more than 3 outcome variables.")

    N = modelOR1["betadraws"].shape[1]
    m = int(N / 1.25)
    burn = int(0.25 * m)
    betaBurnt = _as_matrix(modelOR1["betadraws"])[:, burn:]
    deltaBurnt = _as_matrix(modelOR1["deltadraws"])[:, burn:]

    # Build cut-points from the first column of deltaBurnt (following the R code)
    expdeltaBurnt = np.exp(deltaBurnt)
    gammacpCov = np.zeros((J - 1, expdeltaBurnt.shape[1]))
    for j in range(2, J):
        # Note: mirrors the R code which used the first column; here we use each column b
        gammacpCov[j - 1, :] = np.sum(expdeltaBurnt[: j - 1, :], axis=0)

    n = xMat1.shape[0]
    mcmc_used = betaBurnt.shape[1]
    oldProb = np.zeros((n, mcmc_used, J))
    newProb = np.zeros((n, mcmc_used, J))
    oldComp = np.zeros((n, mcmc_used, J - 1))
    newComp = np.zeros((n, mcmc_used, J - 1))

    for j in range(1, J):  # 1..J-1
        for b in range(mcmc_used):
            for i in range(n):
                oldComp[i, b, j - 1] = alcdf(gammacpCov[j - 1, b] - float(xMat1[i, :] @ betaBurnt[:, b]), 0.0, 1.0, p)
                newComp[i, b, j - 1] = alcdf(gammacpCov[j - 1, b] - float(xMat2[i, :] @ betaBurnt[:, b]), 0.0, 1.0, p)
            if j == 1:
                oldProb[:, b, 0] = oldComp[:, b, 0]
                newProb[:, b, 0] = newComp[:, b, 0]
            else:
                oldProb[:, b, j - 1] = oldComp[:, b, j - 1] - oldComp[:, b, j - 2]
                newProb[:, b, j - 1] = newComp[:, b, j - 1] - newComp[:, b, j - 2]

    oldProb[:, :, J - 1] = 1.0 - oldComp[:, :, J - 2]
    newProb[:, :, J - 1] = 1.0 - newComp[:, :, J - 2]

    diffProb = newProb - oldProb
    avgDiffProb = np.mean(diffProb, axis=(0, 1)).reshape(J, 1)
    return {"avgDiffProb": avgDiffProb}


def logMargLikeOR1(
    y: np.ndarray,
    x: np.ndarray,
    b0: np.ndarray,
    B0: np.ndarray,
    d0: np.ndarray,
    D0: np.ndarray,
    postMeanbeta: np.ndarray,
    postMeandelta: np.ndarray,
    betadraws: np.ndarray,
    deltadraws: np.ndarray,
    tune: float,
    Dhat: np.ndarray,
    p: float,
    verbose: bool,
    random_state: Optional[int] = None,
    ) -> float:
    """Estimate the log marginal likelihood using Chib–Jeliazkov (2001).

    Parameters
    - y: (n, 1) outcome categories 1..J.
    - x: (n, k) design matrix.
    - b0: (k, 1) or (k,) prior mean of beta.
    - B0: (k, k) prior covariance of beta.
    - d0: ((J-2), 1) or ((J-2),) prior mean of delta.
    - D0: ((J-2), (J-2)) prior covariance of delta.
    - postMeanbeta: (k,) posterior mean of beta from complete run.
    - postMeandelta: ((J-2),) posterior mean of delta from complete run.
    - betadraws: (k, nsim) full-run beta draws.
    - deltadraws: ((J-2), nsim) full-run delta draws.
    - tune: scalar proposal scale.
    - Dhat: ((J-2), (J-2)) negative inverse Hessian used for proposals.
    - p: quantile parameter in (0, 1).
    - verbose: flag to print progress (unused here).
    - random_state: optional integer seed for reproducibility.

    Returns
    - logMargLike: scalar estimate of log marginal likelihood.
    """
    if random_state is not None:
        np.random.seed(int(random_state))

    x = _as_matrix(x)
    y = _as_matrix(y)
    betadraws = _as_matrix(betadraws)
    deltadraws = _as_matrix(deltadraws)
    postMeanbeta = np.asarray(postMeanbeta).reshape(-1)
    postMeandelta = np.asarray(postMeandelta).reshape(-1)
    b0 = _as_matrix(b0).reshape(-1)
    B0 = np.asarray(B0)
    d0 = _as_matrix(d0).reshape(-1)
    D0 = np.asarray(D0)
    Dhat = np.asarray(Dhat)

    n, k = x.shape
    nsim = betadraws.shape[1]
    burn = (0.25 * nsim) / 1.25

    indexp = 0.5
    theta = (1.0 - 2.0 * p) / (p * (1.0 - p))
    tau = math.sqrt(2.0 / (p * (1.0 - p)))
    tau2 = tau * tau
    invB0 = inv(B0)
    invB0b0 = invB0 @ b0

    # Reduced run storage
    betaStoreRedrun = np.zeros((k, nsim))
    btildeStoreRedrun = np.zeros((k, nsim))
    BtildeStoreRedrun = np.zeros((k, k, nsim))
    # Store proposal delta vectors drawn from q(delta* -> delta)
    deltaPropRed = np.zeros((postMeandelta.shape[0], nsim))

    # Reduced run: sample beta, w, z while holding delta at postMeandelta
    w = np.abs(np.random.normal(loc=2.0, scale=1.0, size=(n, 1)))
    z = np.random.normal(size=(n, 1))
    Lprop = cholesky((tune ** 2) * Dhat)
    for i in range(nsim):
        bd = drawbetaOR1(z, x, w, tau2, theta, invB0, invB0b0)
        betaStoreRedrun[:, i] = bd["beta"]
        btildeStoreRedrun[:, i] = bd["btilde"]
        BtildeStoreRedrun[:, :, i] = bd["Btilde"]
        w = drawwOR1(z, x, betaStoreRedrun[:, i], tau2, theta, indexp)
        z = drawlatentOR1(y, x, betaStoreRedrun[:, i], w, theta, tau2, postMeandelta)
        # draw a proposal delta from q(delta* -> delta)
        deltaPropRed[:, i] = postMeandelta + Lprop @ np.random.normal(size=postMeandelta.shape[0])

    # Build posterior ordinate terms
    j = 0
    postOrddeltanum = np.zeros((nsim - int(burn), 1))
    postOrddeltaden = np.zeros((nsim - int(burn), 1))
    postOrdbetaStore = np.zeros((nsim - int(burn), 1))

    for i in range(int(burn), nsim):
        # Delta ordinate (numerator and denominator)
        E1_num = qrnegLogLikensumOR1(y, x, betadraws[:, i], postMeandelta, p)
        E1_den = qrnegLogLikensumOR1(y, x, betadraws[:, i], deltadraws[:, i], p)
        E1_logNum = -E1_num["negsumlogl"] + _mvn_logpdf(postMeandelta, d0, D0)
        E1_logDen = -E1_den["negsumlogl"] + _mvn_logpdf(deltadraws[:, i], d0, D0)
        # Numerically stable MH acceptance: min(1, exp(log_ratio)) without overflow
        _log_ratio_E1 = E1_logNum - E1_logDen
        E1alphaMH = 1.0 if _log_ratio_E1 >= 0 else math.exp(_log_ratio_E1)
        qpdf = math.exp(_mvn_logpdf(postMeandelta, deltadraws[:, i], (tune ** 2) * Dhat))
        postOrddeltanum[j, 0] = E1alphaMH * qpdf

        E2_num = qrnegLogLikensumOR1(y, x, betaStoreRedrun[:, i], deltaPropRed[:, i], p)
        E2_den = qrnegLogLikensumOR1(y, x, betaStoreRedrun[:, i], postMeandelta, p)
        E2_logNum = -E2_num["negsumlogl"] + _mvn_logpdf(deltaPropRed[:, i], d0, D0)
        E2_logDen = -E2_den["negsumlogl"] + _mvn_logpdf(postMeandelta, d0, D0)
        # Numerically stable MH acceptance for denominator leg
        _log_ratio_E2 = E2_logNum - E2_logDen
        postOrddeltaden[j, 0] = 1.0 if _log_ratio_E2 >= 0 else math.exp(_log_ratio_E2)

        postOrdbetaStore[j, 0] = math.exp(
            _mvn_logpdf(postMeanbeta, btildeStoreRedrun[:, i], BtildeStoreRedrun[:, :, i])
        )
        j += 1

    postOrddelta = float(np.mean(postOrddeltanum) / np.mean(postOrddeltaden))
    postOrdbeta = float(np.mean(postOrdbetaStore))

    priorContbeta = math.exp(_mvn_logpdf(postMeanbeta, b0, B0))
    priorContdelta = math.exp(_mvn_logpdf(postMeandelta, d0, D0))

    logLikeCont = -qrnegLogLikensumOR1(y, x, postMeanbeta, postMeandelta, p)["negsumlogl"]
    logPriorCont = math.log(priorContbeta * priorContdelta)
    logPosteriorCont = math.log(postOrdbeta * postOrddelta)

    return float(logLikeCont + logPriorCont - logPosteriorCont)


# =============================
# Main MCMC: quantregOR1 (J>=3)
# =============================

def quantregOR1(
    y: np.ndarray,
    x: np.ndarray,
    b0: np.ndarray,
    B0: np.ndarray,
    d0: np.ndarray,
    D0: np.ndarray,
    burn: int,
    mcmc: int,
    p: float,
    tune: float = 0.1,
    accutoff: float = 0.05,
    maxlags: int = 400,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Bayesian quantile regression for ordinal outcomes with J >= 3 (OR1).

    Parameters
    - y: (n, 1) integer categories 1..J.
    - x: (n, k) design matrix (include a column of ones if you want an intercept).
    - b0: (k, 1) or (k,) prior mean for beta.
    - B0: (k, k) prior covariance for beta.
    - d0: ((J-2), 1) or ((J-2),) prior mean for delta.
    - D0: ((J-2), (J-2)) prior covariance for delta.
    - burn: number of burn-in iterations.
    - mcmc: number of post burn-in iterations.
    - p: quantile level in (0, 1).
    - tune: MH proposal scaling for delta (default 0.1).
    - accutoff: ACF cutoff used in inefficiency factor (default 0.05).
    - maxlags: maximum lag for ACF (default 400).
    - verbose: print summary at the end (default True).

    Returns
    - dict with keys similar to the R package output:
      - 'summary': stacked summary for beta then delta: mean, sd, 97.5%, 2.5%, ineff.
      - 'postMeanbeta', 'postStdbeta': (k, 1).
      - 'postMeandelta', 'postStddelta': ((J-2), 1).
      - 'gammacp': (J-1, 1) cut-points at posterior mean.
      - 'catprob': (J,) category probabilities at x=mean(x) and beta=postMeanbeta.
      - 'acceptancerate': MH acceptance rate for delta in percent.
      - 'dicQuant': dict with DIC results.
      - 'logMargLike': scalar log marginal likelihood estimate.
      - 'ineffactor': (k+J-2, 1) inefficiency factors.
      - 'betadraws': (k, nsim), 'deltadraws': ((J-2), nsim).
    """
    # Input checks and basic setup
    x = _as_matrix(x)
    y = _as_matrix(y)
    _assert_numeric_array("x", x)
    _assert_numeric_array("B0", B0)
    _assert_numeric_array("D0", D0)
    if not np.isscalar(mcmc):
        raise TypeError("parameter mcmc must be scalar")
    if not isinstance(mcmc, (int, np.integer)):
        raise TypeError("parameter mcmc must be a numeric")
    if not isinstance(burn, (int, np.integer)):
        raise TypeError("parameter burn must be a numeric")
    if not np.isscalar(p):
        raise TypeError("parameter p must be scalar")
    if not (0 < p < 1):
        raise ValueError("parameter p must be between 0 to 1")
    if not np.isscalar(tune) or tune <= 0:
        raise ValueError("parameter tune must be greater than 0")

    J = _unique_sorted_levels(y).shape[0]
    if J < 3:
        raise ValueError("OR1 requires J >= 3 outcome categories")
    n, k = x.shape
    if D0.shape != (J - 2, J - 2):
        raise ValueError("D0 must have size (J-2)x(J-2)")
    if B0.shape != (k, k):
        raise ValueError("B0 must have size kxk")

    nsim = int(burn + mcmc)

    # Initialize delta using category frequencies (as in the R code)
    yvec = y.reshape(-1).astype(int)
    yprob = np.zeros(J)
    for i in range(n):
        yprob[yvec[i] - 1] += 1
    yprob /= n
    gam = norm.ppf(np.cumsum(yprob[: J - 1]))
    deltaIn = np.log(gam[1: J - 1] - gam[: J - 2]).reshape(-1)

    invB0 = inv(B0)
    invB0b0 = invB0 @ np.asarray(b0).reshape(-1)

    beta = np.zeros((k, nsim))
    delta = np.zeros((J - 2, nsim))

    # OLS-like start for beta
    ytemp = y.reshape(-1) - 1.5
    beta[:, 0] = lstsq(x, ytemp, rcond=None)[0]
    delta[:, 0] = deltaIn

    w = np.abs(np.random.normal(loc=2.0, scale=1.0, size=(n, 1)))
    z = np.random.normal(size=(n, 1))

    theta = (1.0 - 2.0 * p) / (p * (1.0 - p))
    tau = math.sqrt(2.0 / (p * (1.0 - p)))
    tau2 = tau * tau
    indexp = 0.5

    # Compute Dhat via delta optimization
    minimize = qrminfundtheorem(
        deltaIn, y, x, beta[:, 0], cri0=1.0, cri1=0.001, stepsize=1.0,
        maxiter=10, h=0.002, dh=0.0002, sw=20, p=p,
    )
    Dhat = -inv(minimize["H"]) * 3.0

    mhacc = 0
    for i in range(1, nsim):
        bd = drawbetaOR1(z, x, w, tau2, theta, invB0, invB0b0)
        beta[:, i] = bd["beta"]

        w = drawwOR1(z, x, beta[:, i], tau2, theta, indexp)

        dd = drawdeltaOR1(y, x, beta[:, i], delta[:, i - 1], d0, D0, tune, Dhat, p)
        delta[:, i] = dd["deltareturn"]
        if i >= burn:
            mhacc += dd["accept"]

        z = drawlatentOR1(y, x, beta[:, i], w, theta, tau2, delta[:, i])

    # Posterior summaries
    postMeanbeta = np.mean(beta[:, burn:], axis=1)
    postStdbeta = np.std(beta[:, burn:], axis=1, ddof=1)
    if J == 3:
        postMeandelta = float(np.mean(delta[burn:]))
        postStddelta = float(np.std(delta[burn:], ddof=1))
    else:
        postMeandelta = np.mean(delta[:, burn:], axis=1)
        postStddelta = np.std(delta[:, burn:], axis=1, ddof=1)

    # Compute cut-points at posterior mean
    gammacp = np.zeros((J - 1, 1))
    expdelta = np.exp(np.asarray(postMeandelta).reshape(-1))
    for j in range(2, J):
        gammacp[j - 1, 0] = float(np.sum(expdelta[: j - 1]))

    acceptrate = (mhacc / mcmc) * 100.0

    # Category probabilities at mean of x and posterior mean of beta
    xbar = np.mean(x, axis=0)
    catprob = np.zeros(J)
    catprob[0] = alcdfstd(float(0.0 - xbar @ postMeanbeta), p)
    for j in range(2, J):
        catprob[j - 1] = (
            alcdfstd(float(gammacp[j - 1, 0] - xbar @ postMeanbeta), p)
            - alcdfstd(float(gammacp[j - 2, 0] - xbar @ postMeanbeta), p)
        )
    catprob[J - 1] = 1.0 - alcdfstd(float(gammacp[J - 2, 0] - xbar @ postMeanbeta), p)

    # DIC
    dicQuant = dicOR1(y, x, beta, delta, postMeanbeta, postMeandelta, burn, mcmc, p)

    # Marginal likelihood
    logMargLike = logMargLikeOR1(
        y, x, b0, B0, d0, D0,
        postMeanbeta, postMeandelta, beta, delta, tune, Dhat, p, verbose,
    )

    # Inefficiency factors
    ineffactor = ineffactorOR1(x, beta, delta, accutoff, maxlags, verbose=False)

    # Credible intervals
    upperCrediblebeta = np.quantile(beta[:, burn:], 0.975, axis=1)
    lowerCrediblebeta = np.quantile(beta[:, burn:], 0.025, axis=1)
    if J == 3:
        upperCredibledelta = float(np.quantile(delta[burn:], 0.975))
        lowerCredibledelta = float(np.quantile(delta[burn:], 0.025))
    else:
        upperCredibledelta = np.quantile(delta[:, burn:], 0.975, axis=1)
        lowerCredibledelta = np.quantile(delta[:, burn:], 0.025, axis=1)

    # Combine summary like R output
    inefficiencyBeta = ineffactor[:k, 0]
    inefficiencyDelta = ineffactor[k:, 0]

    summary_beta = np.column_stack([
        postMeanbeta.reshape(-1, 1),
        postStdbeta.reshape(-1, 1),
        upperCrediblebeta.reshape(-1, 1),
        lowerCrediblebeta.reshape(-1, 1),
        inefficiencyBeta.reshape(-1, 1),
    ])
    if J == 3:
        summary_delta = np.array([[postMeandelta, postStddelta, upperCredibledelta, lowerCredibledelta, inefficiencyDelta[0]]])
    else:
        summary_delta = np.column_stack([
            np.asarray(postMeandelta).reshape(-1, 1),
            np.asarray(postStddelta).reshape(-1, 1),
            np.asarray(upperCredibledelta).reshape(-1, 1),
            np.asarray(lowerCredibledelta).reshape(-1, 1),
            np.asarray(inefficiencyDelta).reshape(-1, 1),
        ])
    summary = np.vstack([summary_beta, summary_delta])

    if verbose:
        print("Summary of MCMC draws:")
        print(np.round(summary, 4))
        print(f"\nMH acceptance rate: {round(acceptrate, 2)}%")
        print(f"Log of Marginal Likelihood: {round(float(logMargLike), 2)}")
        print(f"DIC: {round(float(dicQuant['DIC']), 2)}")

    result = {
        "summary": summary,
        "postMeanbeta": postMeanbeta.reshape(-1, 1),
        "postMeandelta": np.asarray(postMeandelta).reshape(-1, 1 if J > 3 else 1),
        "postStdbeta": postStdbeta.reshape(-1, 1),
        "postStddelta": np.asarray(postStddelta).reshape(-1, 1 if J > 3 else 1),
        "gammacp": gammacp,
        "catprob": catprob,
        "acceptancerate": acceptrate,
        "dicQuant": dicQuant,
        "logMargLike": logMargLike,
        "ineffactor": ineffactor,
        "betadraws": beta,
        "deltadraws": delta,
    }
    return result


# Optional: small helper to pretty-print a summary like the R S3 method
def summary_bqrorOR1(result: Dict[str, np.ndarray], digits: int = 4) -> None:
    """Pretty-print the `summary` matrix from a `quantregOR1` result.

    Parameters
    - result: dict returned by `quantregOR1`.
    - digits: number of decimal places to show (default 4).
    """
    arr = np.asarray(result["summary"])
    with np.printoptions(precision=digits, suppress=True):
        print(arr)


