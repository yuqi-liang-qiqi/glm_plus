"""
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
    if not np.issubdtype(np.asarray(arr).dtype, np.number):
        raise TypeError(f"each entry in {name} must be numeric")


def _unique_sorted_levels(y: np.ndarray) -> np.ndarray:
    """Return sorted unique levels of y as a 1D array.

    y is expected to be a 1D or 2D (n x 1) array of integers 1..J.
    """
    y1 = np.asarray(y).reshape(-1)
    if not np.allclose(y1, np.floor(y1)):
        raise ValueError("each entry of y must be an integer")
    return np.unique(y1)


def _as_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def alcdfstd(x: float, p: float) -> float:
    """CDF of a standard Asymmetric Laplace distribution AL(0, 1, p).

    - x: scalar
    - p: quantile/skewness parameter in (0, 1)

    Returns P(X <= x) for X ~ AL(0, 1, p).
    """
    if not (0 < p < 1):
        raise ValueError("parameter p must be between 0 to 1")
    if x <= 0:
        return float(p * math.exp((1.0 - p) * x))
    else:
        return float(1.0 - (1.0 - p) * math.exp(-p * x))


def alcdf(x: float, mu: float, sigma: float, p: float) -> float:
    """CDF of an Asymmetric Laplace distribution AL(mu, sigma, p).

    - x: scalar point
    - mu: location
    - sigma: scale (> 0)
    - p: quantile/skewness parameter in (0, 1)

    Returns P(X <= x) for X ~ AL(mu, sigma, p).
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
    """Log pdf of multivariate normal N(mean, cov) at vector x.

    Shapes:
    - x: (d,), mean: (d,), cov: (d, d)
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
    """Sample from Inverse-Gaussian IG(mu, lambda) using the
    Michael–Schucany–Haas method.
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
    """Sample from GIG(lambda=+1/2, chi, psi).

    Relation: If X ~ GIG(1/2, chi, psi), then Y = 1/X ~ IG(muY, lambdaY)
    where muY = sqrt(psi/chi), lambdaY = psi. We sample Y and return X = 1/Y.
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
    """Negative log-likelihood for each observation and their sum (OR1 model).

    Inputs
    - y: (n, 1) integer categories in {1, ..., J}
    - x: (n, k)
    - beta: (k,) or (k,1)
    - delta: ((J-2),) or ((J-2),1), parameterization of cut-points
    - p: quantile parameter in (0,1)

    Returns dict with:
    - nlogl: (n, 1) vector of negative log-likelihood per observation
    - negsumlogl: scalar sum of negative log-likelihood
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
            lnpdf[i, 0] = math.log(alcdf(0.0, meanp, 1.0, p))
        elif yi == J:
            lnpdf[i, 0] = math.log(1.0 - alcdf(allgammacp[J - 1], meanp, 1.0, p))
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
    """Minimize negative log-likelihood w.r.t. delta using finite differences.

    This implements the procedure described in the R function `qrminfundtheorem`.
    It computes gradient and Hessian using central differences and updates delta
    using a combination of BHHH and Newton steps.
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
    """Sample beta from its conditional posterior (multivariate normal).

    z: (n,1), x: (n,k), w: (n,1)
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
        var_sum += (xi @ xi.T) / (tau2 * w[i, 0])
        mean_sum += (x[i, :] * (z[i, 0] - theta * w[i, 0])) / (tau2 * w[i, 0])

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
    """Sample latent weights w from GIG with lambda = 1/2.

    Each w_i ~ GIG(1/2, chi_i, psi) where
      chi_i = ((z_i - x_i beta)^2) / tau2
      psi   = (theta^2)/tau2 + 2
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
        chi_i = ((z[i, 0] - mu_vec[i]) ** 2) / tau2
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
    """Sample latent variable z from truncated normal per observation.

    Truncation bounds come from the cut-points implied by delta.
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

    expdelta = np.exp(delta)
    q = expdelta.shape[0] + 1
    gammacp = np.zeros(q)
    for j in range(2, J):
        gammacp[j - 1] = float(np.sum(expdelta[: j - 1]))
    # Include -inf and +inf for convenience (1-based mapping like R)
    bounds = np.concatenate((np.array([-np.inf]), gammacp, np.array([np.inf])))

    z = np.zeros((n, 1))
    mu = x @ beta
    for i in range(n):
        meanp = float(mu[i] + theta * w[i, 0])
        std = math.sqrt(tau2 * w[i, 0])
        yi = int(y[i, 0])
        a = bounds[yi - 1]
        b = bounds[yi]
        a_std = (a - meanp) / std if np.isfinite(a) else -np.inf
        b_std = (b - meanp) / std if np.isfinite(b) else np.inf
        z[i, 0] = truncnorm.rvs(a_std, b_std, loc=meanp, scale=std)
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
    """Random-walk Metropolis-Hastings for delta.

    Proposal: delta1 ~ N(delta0, (tune^2) * Dhat)
    Accept/Reject based on posterior kernel (likelihood + prior).
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
    """Deviance Information Criterion (DIC) for OR1.

    Returns dict with keys 'DIC', 'pd', 'dev'.
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
    """Compute autocorrelation function up to maxlags (inclusive).

    Returns an array a[0..maxlags], where a[0] = 1.
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
    """Compute inefficiency factors using batch-means.

    Returns a column vector of length k + (J-2): inefficiency for each beta and delta.
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
    """Average covariate effect across outcome categories (OR1 model).

    The effect is the change in predicted category probabilities between xMat2
    and xMat1, averaged over MCMC draws (post burn-in) and observations.
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
) -> float:
    """Estimate log marginal likelihood following the R code structure.

    This uses outputs from a complete run and a reduced run to build the
    posterior ordinate terms for beta and delta.
    """
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
    deltaStoreRedrun = np.zeros((1, nsim))  # stores proposal density for delta draws

    # Reduced run: sample beta, w, z while holding delta at postMeandelta
    w = np.abs(np.random.normal(loc=2.0, scale=1.0, size=(n, 1)))
    z = np.random.normal(size=(n, 1))
    for i in range(nsim):
        bd = drawbetaOR1(z, x, w, tau2, theta, invB0, invB0b0)
        betaStoreRedrun[:, i] = bd["beta"]
        btildeStoreRedrun[:, i] = bd["btilde"]
        BtildeStoreRedrun[:, :, i] = bd["Btilde"]
        w = drawwOR1(z, x, betaStoreRedrun[:, i], tau2, theta, indexp)
        z = drawlatentOR1(y, x, betaStoreRedrun[:, i], w, theta, tau2, postMeandelta)
        # delta proposal density at deltadraws[:, i] under N(postMeandelta, tune^2 * Dhat)
        deltaStoreRedrun[0, i] = math.exp(
            _mvn_logpdf(deltadraws[:, i], postMeandelta, (tune ** 2) * Dhat)
        )

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
        E1alphaMH = min(1.0, math.exp(E1_logNum - E1_logDen))
        qpdf = math.exp(_mvn_logpdf(postMeandelta, deltadraws[:, i], (tune ** 2) * Dhat))
        postOrddeltanum[j, 0] = E1alphaMH * qpdf

        E2_num = qrnegLogLikensumOR1(y, x, betaStoreRedrun[:, i], deltaStoreRedrun[:, i], p)
        E2_den = qrnegLogLikensumOR1(y, x, betaStoreRedrun[:, i], postMeandelta, p)
        E2_logNum = -E2_num["negsumlogl"] + _mvn_logpdf(deltaStoreRedrun[:, i], d0, D0)
        E2_logDen = -E2_den["negsumlogl"] + _mvn_logpdf(postMeandelta, d0, D0)
        postOrddeltaden[j, 0] = min(1.0, math.exp(E2_logNum - E2_logDen))

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
    """Bayesian quantile regression in the OR1 model.

    This closely follows the structure of the R function `quantregOR1`.
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
    arr = np.asarray(result["summary"])
    with np.printoptions(precision=digits, suppress=True):
        print(arr)


