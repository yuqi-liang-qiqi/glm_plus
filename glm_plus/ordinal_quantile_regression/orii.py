"""
@Author  : Yuqi Liang 梁彧祺
@File    : orii.py
@Desc    :
Python translation of the OR2 (Ordinal Quantile Regression with exactly 3 outcomes)
functions from the R package `bqror` (file `ORII.R`).

Goal: keep the API and behavior close to the R reference while using NumPy/SciPy,
and provide clear, beginner-friendly English docstrings.

Important for new users:
- y must be an integer vector with categories 1..J. For OR2 we require J == 3.
- x is an (n x k) matrix that already includes a column of ones if you want an intercept.
- p is the quantile (or skewness) parameter, 0 < p < 1.

This file implements: sampling steps for beta, sigma, and nu; truncated-normal
draws for the latent variable z; likelihood helper; DIC; inefficiency factor;
covariate effect; marginal likelihood; and the main `quantregOR2` function.

Dependencies: numpy, scipy
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, List

import math
import numpy as np
from numpy.linalg import inv, cholesky
from scipy.stats import truncnorm

# Reuse helpers from OR1 implementation
from .ori import (
    alcdf,
    alcdfstd,
    _as_matrix,
    _assert_numeric_array,
    _unique_sorted_levels,
    _mvn_logpdf,
    _acf,
    _rgig_lambda_half,
)


def drawlatentOR2(
    y: np.ndarray,
    x: np.ndarray,
    beta: np.ndarray,
    sigma: float,
    nu: np.ndarray,
    theta: float,
    tau2: float,
    gammacp: np.ndarray,
) -> np.ndarray:
    """Sample latent variable z from a univariate truncated normal (OR2).

    Parameters
    - y: (n, 1) observed categories in {1, 2, 3}.
    - x: (n, k) covariate matrix (include a column of ones if needed).
    - beta: (k,) regression coefficients.
    - sigma: positive scalar scale parameter.
    - nu: (n, 1) modified latent weights.
    - theta: scalar, (1 - 2 p) / (p (1 - p)).
    - tau2: scalar, 2 / (p (1 - p)).
    - gammacp: (J+1,) vector of cut-points including [-inf, 0, cp2, inf].

    Returns
    - z: (n, 1) sampled latent continuous responses.

    Simple explanation: For each observation, we draw a hidden continuous value z
    from a normal distribution that is truncated to the interval that matches its
    observed category. This helps connect the ordinal outcome to a continuous
    latent model used in the sampler.
    """
    y = _as_matrix(y)
    x = _as_matrix(x)
    beta = np.asarray(beta).reshape(-1)
    nu = _as_matrix(nu)
    _assert_numeric_array("x", x)
    _assert_numeric_array("beta", beta)
    _assert_numeric_array("nu", nu)
    if not np.isscalar(theta):
        raise TypeError("parameter theta must be scalar")
    if not np.isscalar(tau2):
        raise TypeError("parameter tau2 must be scalar")

    J = _unique_sorted_levels(y).shape[0]
    n = x.shape[0]
    z = np.zeros((n, 1))
    mu = x @ beta
    for i in range(n):
        mean_i = float(mu[i] + theta * nu[i, 0])
        std_i = math.sqrt(tau2 * sigma * nu[i, 0])
        yi = int(y[i, 0])
        a = gammacp[yi - 1]
        b = gammacp[yi]
        a_std = (a - mean_i) / std_i if np.isfinite(a) else -np.inf
        b_std = (b - mean_i) / std_i if np.isfinite(b) else np.inf
        z[i, 0] = truncnorm.rvs(a_std, b_std, loc=mean_i, scale=std_i)
    return z


def drawbetaOR2(
    z: np.ndarray,
    x: np.ndarray,
    sigma: float,
    nu: np.ndarray,
    tau2: float,
    theta: float,
    invB0: np.ndarray,
    invB0b0: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Sample beta | z, sigma, nu via a multivariate normal (OR2).

    Parameters
    - z: (n, 1) latent continuous responses.
    - x: (n, k) design matrix.
    - sigma: positive scalar.
    - nu: (n, 1) modified latent weights.
    - tau2: scalar, 2 / (p (1 - p)).
    - theta: scalar, (1 - 2 p) / (p (1 - p)).
    - invB0: (k, k) prior precision (inverse covariance) matrix for beta.
    - invB0b0: (k,) product invB0 @ b0.

    Returns
    - dict: keys 'beta' (k,), 'Btilde' (k, k), 'btilde' (k,).

    Beginner note: This step updates the regression coefficients using a
    multivariate normal distribution whose mean and covariance are based on
    the data (z, x, nu) and the prior (b0, B0).
    """
    _assert_numeric_array("z", z)
    _assert_numeric_array("x", x)
    _assert_numeric_array("nu", nu)
    if not np.isscalar(tau2):
        raise TypeError("parameter tau2 must be scalar")
    if not np.isscalar(theta):
        raise TypeError("parameter theta must be scalar")
    _assert_numeric_array("invB0", invB0)
    _assert_numeric_array("invB0b0", invB0b0)

    z = _as_matrix(z)
    x = _as_matrix(x)
    nu = _as_matrix(nu)
    n, k = x.shape

    var_sum = np.zeros((k, k))
    mean_sum = np.zeros(k)
    for i in range(n):
        xi = x[i, :].reshape(-1, 1)
        denom = tau2 * sigma * nu[i, 0]
        var_sum += (xi @ xi.T) / denom
        mean_sum += (x[i, :] * (z[i, 0] - theta * nu[i, 0])) / denom

    Btilde = inv(invB0 + var_sum)
    btilde = Btilde @ (invB0b0.reshape(-1) + mean_sum)
    L = cholesky(Btilde)
    beta = btilde + L @ np.random.normal(size=k)
    return {"beta": beta.reshape(-1), "Btilde": Btilde, "btilde": btilde.reshape(-1)}


def drawsigmaOR2(
    z: np.ndarray,
    x: np.ndarray,
    beta: np.ndarray,
    nu: np.ndarray,
    tau2: float,
    theta: float,
    n0: float,
    d0: float,
) -> Dict[str, float]:
    """Sample sigma from an inverse-gamma distribution (OR2).

    Parameters
    - z: (n, 1) latent continuous responses.
    - x: (n, k) design matrix.
    - beta: (k,) regression coefficients.
    - nu: (n, 1) modified latent weights.
    - tau2: scalar = 2 / (p (1 - p)).
    - theta: scalar = (1 - 2 p) / (p (1 - p)).
    - n0, d0: prior hyperparameters for sigma (inverse-gamma prior).

    Returns
    - dict: keys 'sigma' (float) and 'dtilde' (float), the updated scale parameter.

    Beginner note: We use a standard conjugate update for a variance-like
    parameter. We first build a sum of squared residuals term, then sample sigma
    from an inverse-gamma posterior.
    """
    _assert_numeric_array("z", z)
    _assert_numeric_array("x", x)
    _assert_numeric_array("beta", beta)
    _assert_numeric_array("nu", nu)
    if not np.isscalar(tau2):
        raise TypeError("parameter tau2 must be scalar")
    if not np.isscalar(theta):
        raise TypeError("parameter theta must be scalar")
    if not np.isscalar(n0):
        raise TypeError("parameter n0 must be scalar")
    if not np.isscalar(d0):
        raise TypeError("parameter d0 must be scalar")

    z = _as_matrix(z)
    x = _as_matrix(x)
    beta = np.asarray(beta).reshape(-1)
    nu = _as_matrix(nu)
    n = x.shape[0]

    ntilde = n0 + 3.0 * n
    mu_vec = x @ beta
    temp = ((z.reshape(-1) - mu_vec - theta * nu.reshape(-1)) ** 2) / (tau2 * nu.reshape(-1))
    dtilde = float(np.sum(temp) + d0 + 2.0 * np.sum(nu))
    # Sample sigma ~ IG(ntilde/2, 2/dtilde) via sigma = 1 / Gamma(shape=ntilde/2, scale=2/dtilde)
    shape = ntilde / 2.0
    scale = 2.0 / dtilde
    gamma_sample = np.random.gamma(shape=shape, scale=scale)
    sigma = float(1.0 / gamma_sample)
    return {"sigma": sigma, "dtilde": dtilde}


def drawnuOR2(
    z: np.ndarray,
    x: np.ndarray,
    beta: np.ndarray,
    sigma: float,
    tau2: float,
    theta: float,
    indexp: float,
) -> np.ndarray:
    """Sample modified scale factors nu from a GIG distribution (OR2).

    Parameters
    - z: (n, 1) latent continuous responses.
    - x: (n, k) design matrix.
    - beta: (k,) regression coefficients.
    - sigma: positive scalar.
    - tau2: scalar = 2 / (p (1 - p)).
    - theta: scalar = (1 - 2 p) / (p (1 - p)).
    - indexp: index parameter of GIG, equal to 0.5 here.

    Returns
    - nu: (n, 1) sampled from GIG(1/2, chi_i, psi), where chi_i and psi depend on data.

    Beginner note: The Asymmetric Laplace error can be written as a mixture.
    This step draws the per-observation mixture weights.
    """
    _assert_numeric_array("z", z)
    _assert_numeric_array("x", x)
    _assert_numeric_array("beta", beta)
    if not np.isscalar(sigma):
        raise TypeError("parameter sigma must be scalar")
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
    mu_vec = x @ beta
    tildeeta = (theta * theta) / (tau2 * sigma) + (2.0 / sigma)
    nu = np.zeros((n, 1))
    for i in range(n):
        chi_i = ((z[i, 0] - mu_vec[i]) ** 2) / (tau2 * sigma)
        nu[i, 0] = _rgig_lambda_half(chi_i, tildeeta)
    return nu


def qrnegLogLikeOR2(
    y: np.ndarray,
    x: np.ndarray,
    gammacp: np.ndarray,
    betaOne: np.ndarray,
    sigmaOne: float,
    p: float,
) -> float:
    """Negative sum of log-likelihood for the OR2 model.

    Parameters
    - y: (n, 1) integer categories {1, 2, 3}.
    - x: (n, k) covariate matrix.
    - gammacp: (J+1,) vector of cut-points including [-inf, 0, cp2, inf].
    - betaOne: (k,) a beta vector.
    - sigmaOne: positive scalar.
    - p: quantile parameter in (0, 1).

    Returns
    - negsum: scalar, the negative sum of log-likelihood.

    Beginner note: We compute the probability of each observed category using
    the Asymmetric Laplace CDF at the cut-points, then sum the log-probabilities.
    """
    x = _as_matrix(x)
    y = _as_matrix(y)
    betaOne = np.asarray(betaOne).reshape(-1)
    if not (0 < p < 1):
        raise ValueError("parameter p must be between 0 to 1")
    if not np.isscalar(sigmaOne):
        raise TypeError("parameter sigmaOne must be scalar")

    J = _unique_sorted_levels(y).shape[0]
    n = y.shape[0]
    mu = x @ betaOne
    logpdf = np.zeros(n)
    for i in range(n):
        yi = int(y[i, 0])
        meanf = float(mu[i])
        if yi == 1:
            logpdf[i] = math.log(alcdf(0.0, meanf, sigmaOne, p))
        elif yi == J:
            logpdf[i] = math.log(1.0 - alcdf(gammacp[J - 1], meanf, sigmaOne, p))
        else:
            upper = alcdf(gammacp[J - 1], meanf, sigmaOne, p)
            lower = alcdf(gammacp[J - 2], meanf, sigmaOne, p)
            logpdf[i] = math.log(max(upper - lower, 1e-300))
    return float(-np.sum(logpdf))


def dicOR2(
    y: np.ndarray,
    x: np.ndarray,
    betadraws: np.ndarray,
    sigmadraws: np.ndarray,
    gammacp: np.ndarray,
    postMeanbeta: np.ndarray,
    postMeansigma: float,
    burn: int,
    mcmc: int,
    p: float,
) -> Dict[str, float]:
    """Compute DIC for the OR2 model.

    Returns a dict with keys 'DIC', 'pd', and 'dev'.
    """
    y = _as_matrix(y)
    x = _as_matrix(x)
    betadraws = _as_matrix(betadraws)
    sigmadraws = _as_matrix(sigmadraws)
    postMeanbeta = np.asarray(postMeanbeta).reshape(-1)
    if not np.isscalar(p):
        raise TypeError("parameter p must be scalar")
    if not (0 < p < 1):
        raise ValueError("parameter p must be between 0 to 1")

    nsim = int(burn + mcmc)
    dev = 2.0 * qrnegLogLikeOR2(y, x, gammacp, postMeanbeta, float(postMeansigma), p)

    post_sims = betadraws[:, burn:nsim].shape[1]
    Deviance = np.zeros(post_sims)
    for i in range(post_sims):
        Deviance[i] = 2.0 * qrnegLogLikeOR2(
            y,
            x,
            gammacp,
            betadraws[:, burn + i],
            float(sigmadraws[0, burn + i] if sigmadraws.ndim == 2 else sigmadraws[burn + i]),
            p,
        )
    avgDeviance = float(np.mean(Deviance))
    DIC = 2.0 * avgDeviance - dev
    pd = avgDeviance - dev
    return {"DIC": DIC, "pd": pd, "dev": dev}


def rndald(sigma: float, p: float, n: int) -> np.ndarray:
    """Generate random numbers from AL(0, sigma, p) using a normal–exponential mixture.

    Parameters
    - sigma: positive scale.
    - p: quantile parameter in (0, 1).
    - n: number of draws.

    Returns
    - eps: (n,) draws from Asymmetric Laplace.

    Beginner note: An AL random variable can be generated by mixing a normal
    and an exponential random variable in a simple formula.
    """
    if not (0 < p < 1):
        raise ValueError("parameter p must be between 0 to 1")
    if not np.isscalar(sigma):
        raise TypeError("parameter sigma must be scalar")
    if int(n) != n:
        raise ValueError("parameter n must be an integer")
    u = np.random.normal(size=n)
    w = np.random.exponential(scale=1.0, size=n)
    theta = (1.0 - 2.0 * p) / (p * (1.0 - p))
    tau = math.sqrt(2.0 / (p * (1.0 - p)))
    eps = sigma * (theta * w + tau * np.sqrt(w) * u)
    return eps.reshape(-1)


def ineffactorOR2(
    x: np.ndarray,
    betadraws: np.ndarray,
    sigmadraws: np.ndarray,
    accutoff: float = 0.05,
    maxlags: int = 400,
    verbose: bool = True,
) -> np.ndarray:
    """Compute inefficiency factors (batch-means) for beta and sigma in OR2.

    Returns a column vector of length k + 1: k for beta components, 1 for sigma.
    """
    betadraws = _as_matrix(betadraws)
    sigmadraws = _as_matrix(sigmadraws)
    n = betadraws.shape[1]
    k = betadraws.shape[0]

    ineff_beta = np.zeros((k, 1))
    for i in range(k):
        ac = _acf(betadraws[i, :], maxlags)
        idx = np.where(ac[1:] <= accutoff)[0]
        nlags = int(idx[0] + 1) if idx.size > 0 else maxlags
        nbatch = max(1, n // nlags)
        nuse = nbatch * nlags
        b = betadraws[i, :nuse]
        xbatch = b.reshape(nbatch, nlags).T
        mxbatch = np.mean(xbatch, axis=0)
        varxbatch = float(np.sum((mxbatch - np.mean(b)) ** 2) / (nbatch - 1 if nbatch > 1 else 1.0))
        nse = math.sqrt(varxbatch / nbatch) if nbatch > 0 else np.nan
        rne = (np.std(b, ddof=1) / math.sqrt(nuse)) / nse if nuse > 1 and nse > 0 else np.nan
        ineff_beta[i, 0] = 1.0 / rne if rne and not np.isnan(rne) and rne != 0 else np.nan

    # Sigma
    ac2 = _acf(sigmadraws.reshape(-1), maxlags)
    idx2 = np.where(ac2[1:] <= accutoff)[0]
    nlags2 = int(idx2[0] + 1) if idx2.size > 0 else maxlags
    nbatch2 = max(1, n // nlags2)
    nuse2 = nbatch2 * nlags2
    s = sigmadraws.reshape(-1)[:nuse2]
    xbatch2 = s.reshape(nbatch2, nlags2).T
    mxbatch2 = np.mean(xbatch2, axis=0)
    varxbatch2 = float(np.sum((mxbatch2 - np.mean(s)) ** 2) / (nbatch2 - 1 if nbatch2 > 1 else 1.0))
    nse2 = math.sqrt(varxbatch2 / nbatch2) if nbatch2 > 0 else np.nan
    rne2 = (np.std(s, ddof=1) / math.sqrt(nuse2)) / nse2 if nuse2 > 1 and nse2 > 0 else np.nan
    ineff_sigma = np.array([[1.0 / rne2 if rne2 and not np.isnan(rne2) and rne2 != 0 else np.nan]])

    ineff = np.vstack([ineff_beta, ineff_sigma])
    if verbose:
        print("Summary of Inefficiency Factor:")
        print(np.round(ineff, 4))
    return ineff


def covEffectOR2(
    modelOR2: Dict[str, np.ndarray],
    y: np.ndarray,
    xMat1: np.ndarray,
    xMat2: np.ndarray,
    gammacp2: float,
    p: float,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Average covariate effect (ACE) for the OR2 model.

    Parameters
    - modelOR2: dict output from `quantregOR2` that contains 'betadraws' and 'sigmadraws'.
    - y: (n, 1) observed categories {1, 2, 3}.
    - xMat1: (n, k) baseline covariate matrix.
    - xMat2: (n, k) modified covariate matrix (e.g., add 0.02 to a column).
    - gammacp2: the single interior cut-point (besides 0). The full cut vector is [-inf, 0, gammacp2, inf].
    - p: quantile parameter in (0, 1).
    - verbose: print a small summary (default True).

    Returns
    - dict with key 'avgDiffProb': (J, 1) average change in predicted probabilities.

    Beginner note: We compare predicted category probabilities before and after
    changing one covariate, then average the differences over observations and
    a subset of posterior draws.
    """
    xMat1 = _as_matrix(xMat1)
    xMat2 = _as_matrix(xMat2)
    y = _as_matrix(y)
    J = _unique_sorted_levels(y).shape[0]
    if J > 3:
        raise ValueError("This function is only available for models with 3 outcome variables.")
    if not (0 < p < 1):
        raise ValueError("parameter p must be between 0 to 1")

    N = modelOR2["betadraws"].shape[1]
    m = int(N / 1.25)
    burn = int(0.25 * m)
    betaBurnt = _as_matrix(modelOR2["betadraws"])[:, burn:]
    sigmaBurnt = _as_matrix(modelOR2["sigmadraws"]).reshape(-1)[burn:]

    gammacp = np.array([-np.inf, 0.0, float(gammacp2), np.inf])
    n = xMat1.shape[0]
    m_used = betaBurnt.shape[1]

    oldProb = np.zeros((n, m_used, J))
    newProb = np.zeros((n, m_used, J))
    oldComp = np.zeros((n, m_used, J - 1))
    newComp = np.zeros((n, m_used, J - 1))

    for j in range(1, J):  # 1..J-1
        for b in range(m_used):
            for i in range(n):
                oldComp[i, b, j - 1] = alcdf(
                    gammacp[j] - float(xMat1[i, :] @ betaBurnt[:, b]), 0.0, float(sigmaBurnt[b]), p
                )
                newComp[i, b, j - 1] = alcdf(
                    gammacp[j] - float(xMat2[i, :] @ betaBurnt[:, b]), 0.0, float(sigmaBurnt[b]), p
                )
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

    if verbose:
        print("Summary of Covariate Effect:")
        print(np.round(avgDiffProb, 4))
    return {"avgDiffProb": avgDiffProb}


def _invgamma_logpdf(x: float, shape: float, scale: float) -> float:
    """Log-PDF of the inverse-gamma IG(shape, scale) at x.

    Uses the parameterization: f(x) = scale^shape / Gamma(shape) * x^(-shape-1) * exp(-scale / x)
    for x > 0.
    """
    if x <= 0:
        return -np.inf
    return shape * math.log(scale) - math.lgamma(shape) - (shape + 1.0) * math.log(x) - (scale / x)


def logMargLikeOR2(
    y: np.ndarray,
    x: np.ndarray,
    b0: np.ndarray,
    B0: np.ndarray,
    n0: float,
    d0: float,
    postMeanbeta: np.ndarray,
    postMeansigma: float,
    btildeStore: np.ndarray,
    BtildeStore: np.ndarray,
    gammacp2: float,
    p: float,
    verbose: bool,
    random_state: Optional[int] = None,
) -> float:
    """Estimate the log marginal likelihood in OR2 using a reduced run.

    Parameters
    - y, x: data.
    - b0, B0: prior mean and covariance for beta.
    - n0, d0: prior hyperparameters for sigma (inverse-gamma).
    - postMeanbeta, postMeansigma: posterior means from the complete run.
    - btildeStore, BtildeStore: storage from complete run for beta's conditional posterior.
    - gammacp2: the middle cut-point (besides 0).
    - p: quantile parameter.
    - verbose: unused here.

    Returns
    - logMargLike: scalar estimate of log marginal likelihood.
    """
    if random_state is not None:
        np.random.seed(int(random_state))

    x = _as_matrix(x)
    y = _as_matrix(y)
    postMeanbeta = np.asarray(postMeanbeta).reshape(-1)
    n = x.shape[0]
    nsim = btildeStore.shape[1]
    burn = int((0.25 * nsim) / 1.25)

    J = _unique_sorted_levels(y).shape[0]
    gammacp = np.array([-np.inf, 0.0, float(gammacp2), np.inf])
    indexp = 0.5
    theta = (1.0 - 2.0 * p) / (p * (1.0 - p))
    tau = math.sqrt(2.0 / (p * (1.0 - p)))
    tau2 = tau * tau
    ntilde = n0 + 3.0 * n

    sigmaRedrun = np.zeros(nsim)
    dtildeStoreRedrun = np.zeros(nsim)
    z = np.random.normal(size=(n, 1))
    nu = 5.0 * np.ones((n, 1))

    # Reduced run: sample sigma and nu holding beta at postMeanbeta
    for i in range(nsim):
        sig = drawsigmaOR2(z, x, postMeanbeta, nu, tau2, theta, n0, d0)
        sigmaRedrun[i] = sig["sigma"]
        dtildeStoreRedrun[i] = sig["dtilde"]
        nu = drawnuOR2(z, x, postMeanbeta, sigmaRedrun[i], tau2, theta, indexp)
        z = drawlatentOR2(y, x, postMeanbeta, sigmaRedrun[i], nu, theta, tau2, gammacp)

    # Posterior ordinate terms (use log-mean-exp to reduce underflow)
    postOrdbetaLogs = np.zeros(nsim - burn)
    postOrdsigmaLogs = np.zeros(nsim - burn)
    j = 0
    sigmaStar = float(postMeansigma)
    for i in range(burn, nsim):
        postOrdbetaLogs[j] = _mvn_logpdf(postMeanbeta, btildeStore[:, i], BtildeStore[:, :, i])
        shape = ntilde / 2.0
        scale = 2.0 / dtildeStoreRedrun[i]
        postOrdsigmaLogs[j] = _invgamma_logpdf(sigmaStar, shape, scale)
        j += 1

    def _log_mean_exp(log_vals: np.ndarray) -> float:
        m = float(np.max(log_vals))
        return m + math.log(float(np.mean(np.exp(log_vals - m))))

    postOrdbeta_logmean = _log_mean_exp(postOrdbetaLogs)
    postOrdsigma_logmean = _log_mean_exp(postOrdsigmaLogs)

    # Priors
    b0 = _as_matrix(b0).reshape(-1)
    B0 = np.asarray(B0)
    # Note: The original R code does b0 <- array(rep(b0, k), dim=c(k,1)) before the prior
    # density evaluation, which appears unintended. We intentionally keep b0 as length-k here.
    priorContbeta = math.exp(_mvn_logpdf(postMeanbeta, b0, B0))
    shape0 = n0 / 2.0
    scale0 = 2.0 / d0
    priorContsigma = math.exp(_invgamma_logpdf(float(postMeansigma), shape0, scale0))

    # Likelihood at posterior means
    logLikeCont = -qrnegLogLikeOR2(y, x, gammacp, postMeanbeta, float(postMeansigma), p)
    logPriorCont = math.log(priorContbeta * priorContsigma)
    # Use log of mean densities from reduced run
    logPosteriorCont = postOrdbeta_logmean + postOrdsigma_logmean
    return float(logLikeCont + logPriorCont - logPosteriorCont)


def quantregOR2(
    y: np.ndarray,
    x: np.ndarray,
    b0: np.ndarray,
    B0: np.ndarray,
    n0: float = 5.0,
    d0: float = 8.0,
    gammacp2: float = 3.0,
    burn: int = 1000,
    mcmc: int = 5000,
    p: float = 0.5,
    accutoff: float = 0.05,
    maxlags: int = 400,
    verbose: bool = True,
    x_names: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Bayesian quantile regression for ordinal outcomes with exactly 3 categories (OR2).

    Parameters
    - y: (n, 1) integer categories {1, 2, 3}.
    - x: (n, k) design matrix (include a column of ones if you want an intercept).
    - b0, B0: prior mean and covariance for beta.
    - n0, d0: prior shape and scale for sigma's inverse-gamma prior.
    - gammacp2: the single interior cut-point (besides 0); full cut vector is [-inf, 0, gammacp2, inf].
    - burn: number of burn-in iterations.
    - mcmc: number of post burn-in iterations.
    - p: quantile (0 < p < 1).
    - accutoff, maxlags: settings for inefficiency factor.
    - verbose: print a summary at the end.

    Returns
    - dict mirroring the R package output:
      - 'summary': stacked summary for beta (k rows) and sigma (1 row): mean, sd, 97.5%, 2.5%, ineff.
      - 'postMeanbeta', 'postStdbeta': (k, 1).
      - 'postMeansigma', 'postStdsigma': (1, 1).
      - 'dicQuant': dict with DIC results.
      - 'logMargLike': scalar log marginal likelihood estimate.
      - 'ineffactor': (k+1, 1) inefficiency factors.
      - 'betadraws': (k, nsim), 'sigmadraws': (1, nsim).

    Beginner note: The algorithm cycles through four steps at each iteration:
    update beta, update sigma, update nu, and update the latent z. After the
    burn-in period, we keep draws to summarize uncertainty and compute model
    diagnostics (DIC) and an approximation to the marginal likelihood.
    """
    x = _as_matrix(x)
    y = _as_matrix(y)
    _assert_numeric_array("x", x)
    _assert_numeric_array("B0", B0)
    if not isinstance(mcmc, (int, np.integer)):
        raise TypeError("parameter mcmc must be a numeric")
    if not isinstance(burn, (int, np.integer)):
        raise TypeError("parameter burn must be a numeric")
    if not np.isscalar(p):
        raise TypeError("parameter p must be scalar")
    if not (0 < p < 1):
        raise ValueError("parameter p must be between 0 to 1")
    levels = _unique_sorted_levels(y)
    J = levels.shape[0]
    if not (J == 3 and int(levels.min()) == 1 and int(levels.max()) == 3):
        raise ValueError("OR2 requires exactly three outcome levels {1, 2, 3}.")

    n, k = x.shape
    if np.asarray(B0).shape != (k, k):
        raise ValueError("B0 must have size kxk")

    nsim = int(burn + mcmc)
    invB0 = inv(np.asarray(B0))
    invB0b0 = invB0 @ _as_matrix(b0).reshape(-1)

    beta = np.zeros((k, nsim))
    sigma = np.zeros(nsim)
    btildeStore = np.zeros((k, nsim))
    BtildeStore = np.zeros((k, k, nsim))

    beta[:, 0] = 0.0
    sigma[0] = 2.0
    nu = 5.0 * np.ones((n, 1))
    gammacp = np.array([-np.inf, 0.0, float(gammacp2), np.inf])
    indexp = 0.5
    theta = (1.0 - 2.0 * p) / (p * (1.0 - p))
    tau = math.sqrt(2.0 / (p * (1.0 - p)))
    tau2 = tau * tau
    z = np.random.normal(size=(n, 1))

    for i in range(1, nsim):
        bd = drawbetaOR2(z, x, sigma[i - 1], nu, tau2, theta, invB0, invB0b0)
        beta[:, i] = bd["beta"]
        btildeStore[:, i] = bd["btilde"]
        BtildeStore[:, :, i] = bd["Btilde"]

        sd = drawsigmaOR2(z, x, beta[:, i], nu, tau2, theta, float(n0), float(d0))
        sigma[i] = sd["sigma"]

        nu = drawnuOR2(z, x, beta[:, i], sigma[i], tau2, theta, indexp)

        z = drawlatentOR2(y, x, beta[:, i], sigma[i], nu, theta, tau2, gammacp)

    # Posterior summaries
    postMeanbeta = np.mean(beta[:, burn:], axis=1)
    postStdbeta = np.std(beta[:, burn:], axis=1, ddof=1)
    postMeansigma = float(np.mean(sigma[burn:]))
    postStdsigma = float(np.std(sigma[burn:], ddof=1))

    dicQuant = dicOR2(y, x, beta, sigma.reshape(1, -1), gammacp, postMeanbeta, postMeansigma, burn, mcmc, p)

    logMargLike = logMargLikeOR2(
        y,
        x,
        b0,
        B0,
        float(n0),
        float(d0),
        postMeanbeta,
        postMeansigma,
        btildeStore,
        BtildeStore,
        float(gammacp2),
        p,
        verbose,
    )

    ineffactor = ineffactorOR2(x, beta, sigma.reshape(1, -1), accutoff, maxlags, verbose=False)

    # Credible intervals and summary matrix
    upperCrediblebeta = np.quantile(beta[:, burn:], 0.975, axis=1)
    lowerCrediblebeta = np.quantile(beta[:, burn:], 0.025, axis=1)
    upperCrediblesigma = float(np.quantile(sigma[burn:], 0.975))
    lowerCrediblesigma = float(np.quantile(sigma[burn:], 0.025))

    inefficiencyBeta = ineffactor[:k, 0]
    inefficiencySigma = ineffactor[k, 0]

    summary_beta = np.column_stack([
        postMeanbeta.reshape(-1, 1),
        postStdbeta.reshape(-1, 1),
        upperCrediblebeta.reshape(-1, 1),
        lowerCrediblebeta.reshape(-1, 1),
        inefficiencyBeta.reshape(-1, 1),
    ])
    summary_sigma = np.array([[postMeansigma, postStdsigma, upperCrediblesigma, lowerCrediblesigma, inefficiencySigma]])
    summary = np.vstack([summary_beta, summary_sigma])

    if verbose:
        print("Summary of MCMC draws:")
        # Row labels for nicer printout
        def _build_row_names(k: int, names: Optional[Sequence[str]]) -> List[str]:
            if names is not None and len(names) == k:
                return list(names) + ["sigma"]
            return [f"beta_{i+1}" for i in range(k)] + ["sigma"]

        row_names = _build_row_names(k, x_names)
        out = np.round(summary, 4)
        # Simple aligned print with labels
        colhdr = ["Post Mean", "Post Std", "Upper Credible", "Lower Credible", "Inef Factor"]
        print("\t".join([" "] + colhdr))
        for rn, row in zip(row_names, out):
            print("\t".join([rn] + [str(v) for v in row.tolist()]))
        print(f"\nLog of Marginal Likelihood: {round(float(logMargLike), 2)}")
        print(f"DIC: {round(float(dicQuant['DIC']), 2)}")

    # Build row names for external use
    row_names = ([f"beta_{i+1}" for i in range(k)] if not x_names or len(x_names) != k else list(x_names)) + ["sigma"]

    result = {
        "summary": summary,
        "postMeanbeta": postMeanbeta.reshape(-1, 1),
        "postStdbeta": postStdbeta.reshape(-1, 1),
        "postMeansigma": np.array([[postMeansigma]]),
        "postStdsigma": np.array([[postStdsigma]]),
        "dicQuant": dicQuant,
        "logMargLike": float(logMargLike),
        "ineffactor": ineffactor,
        "betadraws": beta,
        "sigmadraws": sigma.reshape(1, -1),
        "row_names": row_names,
    }
    return result


def summary_bqrorOR2(result: Dict[str, np.ndarray], digits: int = 4) -> None:
    """Pretty-print the `summary` matrix from a `quantregOR2` result.

    Parameters
    - result: dict returned by `quantregOR2`.
    - digits: number of decimal places to show (default 4).
    """
    arr = np.asarray(result["summary"])
    with np.printoptions(precision=digits, suppress=True):
        print(arr)


