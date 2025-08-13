from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Sequence

import numpy as np
from numpy.typing import ArrayLike

try:
    # statsmodels for linear quantile regression
    import statsmodels.api as sm
except Exception as _:
    sm = None  # handled at fit time

try:
    # scikit-learn for isotonic regression, CCA, and spline basis
    from sklearn.isotonic import IsotonicRegression
    from sklearn.cross_decomposition import CCA
    from sklearn.preprocessing import SplineTransformer
except Exception as _:
    IsotonicRegression = None  # type: ignore
    CCA = None  # type: ignore
    SplineTransformer = None  # type: ignore


def _as_2d(x: ArrayLike) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


def _jitter_ordinal(y: ArrayLike, random_state: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    y = np.asarray(y).reshape(-1)
    if not np.allclose(y, np.floor(y)):
        raise ValueError("y must be integer-valued categories 1..J")
    u = rng.uniform(0.0, 1.0, size=y.shape[0])
    return y.astype(float) + u


@dataclass
class FrequentistOQR:
    """
    Frequentist TORQUE-like ordinal quantile regression with optional two-index extension.

    Model (single-index): Q_tau(h(Y) | X) = alpha_tau + X beta_tau, with h monotone.
    Optionally fits a second index on residuals per Hong & Zhou (2013) to reduce variance.

    Parameters
    - quantiles: list of tau values in (0,1) to fit.
    - use_two_index: if True, estimate a second index via residual regression.
    - random_state: seed for jittering reproducibility.

    Notes
    - h is estimated via isotonic regression mapping observed jittered Y to an order-preserving
      scale aligned with a preliminary index. The inverse transform is obtained by numerical
      inversion on a grid for prediction back to Y.
    - Cutpoints are not used explicitly; predictions return conditional quantiles on original
      ordinal scale by inverting h and flooring.
    - Identification: we use transform-location constraints h(y0)=0 and, when relevant, g(e10)=0,
      followed by a norm-based scaling where all betas and transform codomains are divided by ||beta1||.
      This is equivalent to fixing the first coefficient up to reparameterization.
    - This implementation focuses on at most two indices (k<=2) even if selection suggests k>2.
    - The rank-based objectives for h and g are approximated via finite grids over thresholds with
      isotonic regression enforcing monotonicity. Grids are controlled by rank_grid_* and t_grid_*
      parameters; optional subsample_n reduces the O(n^2) pairwise cost.
    """

    quantiles: Sequence[float] = (0.25, 0.5, 0.75)
    use_two_index: bool = False
    random_state: Optional[int] = None
    auto_select_k: bool = True
    alpha_cancor: float = 0.05
    n_spline_knots: int = 5
    spline_degree: int = 3
    rank_grid_n: int = 51
    t_grid_n: int = 61
    rank_grid_low: float = 0.05
    rank_grid_high: float = 0.95
    t_grid_low: float = 0.05
    t_grid_high: float = 0.95
    subsample_n: Optional[int] = None

    # learned attributes after fit
    is_fit_: bool = False
    h_iso_: Any = None
    grid_y_: Optional[np.ndarray] = None
    grid_hy_: Optional[np.ndarray] = None
    beta1_: Optional[np.ndarray] = None
    alpha1_: Dict[float, float] | None = None  # deprecated in normalized variant
    beta2_: Optional[np.ndarray] = None
    alpha2_: Dict[float, float] | None = None  # deprecated in normalized variant
    cca_: Any = None
    g_iso_: Any = None
    grid_r1_: Optional[np.ndarray] = None
    grid_gr_: Optional[np.ndarray] = None
    selected_k_: Optional[int] = None
    selected_k_full_: Optional[int] = None
    cancor_gammas_: Optional[np.ndarray] = None
    cancor_pvalue_s1_: Optional[float] = None
    cancor_pvalues_seq_: Optional[np.ndarray] = None
    beta2_init_: Optional[np.ndarray] = None
    beta2_tau_: Optional[Dict[float, np.ndarray]] = None
    _y_jit: Optional[np.ndarray] = None
    _y_min_cat: Optional[int] = None
    _y_max_cat: Optional[int] = None
    _used_weights_: bool = False
    _sum_weights_: Optional[float] = None
    k_truncated_: Optional[bool] = None
    _last_y_: Optional[np.ndarray] = None
    _last_w_: Optional[np.ndarray] = None

    def _check_deps(self) -> None:
        if sm is None:
            raise ImportError("statsmodels is required: pip install statsmodels")
        if IsotonicRegression is None:
            raise ImportError("scikit-learn is required: pip install scikit-learn")
        if self.auto_select_k and (CCA is None or SplineTransformer is None):
            raise ImportError("scikit-learn's CCA and SplineTransformer are required for auto_select_k")

    def _cancor_select_k(self, X: np.ndarray, y_jit: np.ndarray, w: Optional[np.ndarray]) -> int:
        """Weighted CANCOR + sequential chi-square to choose k (supports k>2 selection).

        Weighting is implemented via weighted whitening: rows are centered by weighted means,
        scaled by weighted standard deviations, and multiplied by sqrt(normalized weights),
        then unweighted CCA is applied with scale=False.
        """
        if CCA is None or SplineTransformer is None:
            return 1
        n, p = X.shape
        # Normalize weights to have mean 1 (so sum equals n) for stable testing scale
        if w is None:
            w_norm = np.ones(n, dtype=float)
        else:
            w = w.reshape(-1).astype(float)
            w_norm = w * (n / np.sum(w)) if np.sum(w) > 0 else np.ones(n, dtype=float)
        sw = np.sqrt(w_norm)
        # Build B-spline basis of jittered y
        st = SplineTransformer(n_knots=max(self.n_spline_knots, 3), degree=int(self.spline_degree), include_bias=True)
        Z_raw = st.fit_transform(y_jit.reshape(-1, 1))
        # Weighted centering and scaling
        def _w_center_scale(A: np.ndarray) -> np.ndarray:
            mu = (w_norm[:, None] * A).sum(axis=0, keepdims=True) / np.sum(w_norm)
            A0 = A - mu
            var = (w_norm[:, None] * (A0 ** 2)).sum(axis=0, keepdims=True) / np.sum(w_norm)
            sd = np.sqrt(np.maximum(var, 1e-12))
            return (A0 / sd) * sw[:, None]
        Xw = _w_center_scale(X)
        Zw = _w_center_scale(Z_raw)
        # Fit up to r components
        r = min(p, Zw.shape[1])
        if r < 1:
            return 1
        cca = CCA(n_components=r, scale=False)
        Xc, Zc = cca.fit_transform(Xw, Zw)
        gammas = []
        for i in range(r):
            xi = Xc[:, i]
            zi = Zc[:, i]
            denom = np.std(xi) * np.std(zi)
            rho = 0.0 if denom == 0 else float(np.corrcoef(xi, zi)[0, 1])
            gammas.append(max(0.0, min(0.999999, abs(rho))))
        gammas = np.array(gammas)
        self.cca_ = cca
        self.cancor_gammas_ = gammas
        # Sequential tests for s=0..r-1 (null: exactly s significant)
        # Paper's df: (p - s)(H + m - s - 1). Approximate H as internal knots = n_knots - 2; m = degree + 1
        H_internal = max(int(self.n_spline_knots) - 2, 1)
        m_order = int(self.spline_degree) + 1
        from math import log
        from scipy.stats import chi2
        pvals = []
        chosen = None
        for s in range(0, r):
            tail = gammas[s:]  # includes index s (0-based means gamma_{s+1}..)
            if tail.size == 0:
                continue
            ssum = 0.0
            for g in tail:
                ssum += log(1.0 - g * g + 1e-12)
            stat = -(n - (p + H_internal + m_order + 2) / 2.0) * ssum
            df = max((p - s) * (H_internal + m_order - s - 1), 1)
            pval = float(chi2.sf(stat, df))
            pvals.append(pval)
            if chosen is None and (pval > self.alpha_cancor):
                chosen = s
        if chosen is None:
            chosen = r  # all significant
        self.cancor_pvalues_seq_ = np.array(pvals, dtype=float)
        self.cancor_pvalue_s1_ = float(pvals[1]) if len(pvals) > 1 else (float(pvals[0]) if pvals else None)  # legacy
        self.selected_k_full_ = int(chosen)
        return int(min(chosen, 2))

    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: Optional[ArrayLike] = None, y0: Optional[float] = None, e10: Optional[float] = None) -> "FrequentistOQR":
        self._check_deps()

        X = _as_2d(X).astype(float)
        y = np.asarray(y).reshape(-1)
        n, k = X.shape
        if y.shape[0] != n:
            raise ValueError("X and y have incompatible shapes")
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
        if w is not None and w.shape[0] != n:
            raise ValueError("sample_weight length must match n")
        taus = [float(t) for t in self.quantiles]
        if not all(0 < t < 1 for t in taus):
            raise ValueError("All quantiles must be in (0,1)")

        # Step 0: jitter the ordinal response
        y_jit = _jitter_ordinal(y, random_state=self.random_state)
        self._y_jit = y_jit.copy()

        # Step 1: CANCOR-based initial direction(s) and automatic k selection (optional)
        beta0 = None
        if self.auto_select_k:
            k_hat = self._cancor_select_k(X, y_jit, w)
            self.selected_k_ = int(k_hat)
            # Override two-index based on selection (model currently supports up to 2 indices)
            self.use_two_index = bool((self.selected_k_full_ or k_hat) >= 2)
            self.k_truncated_ = bool((self.selected_k_full_ or k_hat) > 2)
            # Use first CCA X-weight as initial direction if available
            try:
                beta0 = self.cca_.x_weights_[:, 0].reshape(-1)
            except Exception:
                beta0 = None
        if beta0 is None:
            rq0 = sm.QuantReg(y_jit, X).fit(q=0.5, weights=w)
            beta0 = rq0.params.reshape(-1)

        # Step 2: monotone transform h via rank-based objective (approximation to Eq. 11)
        # Choose y0 for location normalization h(y0)=0
        if y0 is None:
            # median of jittered response
            y0 = float(np.quantile(y_jit, 0.5))
        # Precompute S = Xβ0_i - Xβ0_j (with optional subsample to reduce O(n^2))
        sel = None
        if self.subsample_n is not None and n > int(self.subsample_n):
            rng = np.random.default_rng(self.random_state)
            sel = rng.choice(n, size=int(self.subsample_n), replace=False)
            X_rank = X[sel, :]
            y_rank = y_jit[sel]
            wi = (np.ones_like(y_rank) if w is None else w[sel])
            wj = wi
        else:
            X_rank = X
            y_rank = y_jit
            wi = np.ones(n) if w is None else w
            wj = wi
        s_vec = (X_rank @ beta0).reshape(-1)
        S = s_vec[:, None] - s_vec[None, :]
        # Candidate thresholds t over quantiles of S
        flatS = S.reshape(-1)
        t_grid = np.quantile(flatS, np.linspace(self.t_grid_low, self.t_grid_high, int(self.t_grid_n)))
        # Evaluate h at jittered quantile grid
        y_levels = np.quantile(y_jit, np.linspace(self.rank_grid_low, self.rank_grid_high, int(self.rank_grid_n)))
        # Helper to compute objective at given y_thresh and t
        def _obj_h(y_thresh: float, t: float) -> float:
            A = (y_rank >= y_thresh).astype(float)
            B = (y_rank >= y0).astype(float)
            # sum_{i,j: S>=t} w_i w_j (A_i - B_j), exclude i==j
            M = (S >= t).astype(float)
            np.fill_diagonal(M, 0.0)
            term1 = float((wi * A)[:, None] @ (wj * M).sum(axis=1, keepdims=True))
            term2 = float((wj * B)[None, :] @ (wi[:, None] * M).sum(axis=0, keepdims=True).T)
            return term1 - term2
        h_vals = []
        for yl in y_levels:
            best_t = t_grid[0]
            best_val = -1e300
            for t in t_grid:
                val = _obj_h(float(yl), float(t))
                if val > best_val:
                    best_val = val
                    best_t = t
            h_vals.append(best_t)
        # Enforce monotonicity of h over levels via isotonic regression on (y_levels, h_vals)
        self.h_iso_ = IsotonicRegression(increasing=True, out_of_bounds='clip')
        self.h_iso_.fit(y_levels.astype(float), np.array(h_vals, dtype=float))
        # Cache grid for inverse mapping h^{-1}
        y_min, y_max = float(np.min(y)), float(np.max(y) + 1.0)
        self.grid_y_ = np.linspace(y_min, y_max, num=1001)
        self.grid_hy_ = self.h_iso_.transform(self.grid_y_)
        self._y_min_cat = int(np.min(y))
        self._y_max_cat = int(np.max(y))

        # Step 3: fit quantile regressions for h(Y) on X (normalized: no separate alpha storage)
        hy_full = self.h_iso_.transform(self._y_jit)
        # 3a) enforce beta1 from median (tau=0.5), independent of user-specified quantiles
        rq_med = sm.QuantReg(hy_full, X).fit(q=0.5, weights=w)
        self.beta1_ = rq_med.params.reshape(-1)
        # No explicit alpha1 stored; absorbed by h's location normalization
        self.alpha1_ = None

        # Optional two-index: regress residuals r1 = h(Y) - (alpha1_0.5 + X beta1)
        if self.use_two_index:
            # Use median residuals for second-stage index
            r1 = hy_full - (X @ self.beta1_)
            # Determine initial second direction from CANCOR if available
            if (self.cca_ is not None) and hasattr(self.cca_, "x_weights_") and self.cca_.x_weights_.shape[1] >= 2:
                self.beta2_init_ = self.cca_.x_weights_[:, 1].reshape(-1)
            else:
                self.beta2_init_ = beta0.copy()
            # Estimate second monotone transform g via rank-based objective (approx to Eq. 12)
            # Choose e1,0 for normalization g(e1,0)=0
            if e10 is None:
                e10 = float(np.quantile(r1, 0.5))
            # Align residuals to rank sample size to avoid dimension mismatch when subsampling
            r1_rank = r1[sel] if sel is not None else r1
            # Pairwise differences along beta2^0
            s2_vec = (X_rank @ (self.beta2_init_ if self.beta2_init_ is not None else beta0)).reshape(-1)
            S2 = s2_vec[:, None] - s2_vec[None, :]
            flatS2 = S2.reshape(-1)
            t2_grid = np.quantile(flatS2, np.linspace(self.t_grid_low, self.t_grid_high, int(self.t_grid_n)))
            # Residual thresholds grid
            r_grid = np.quantile(r1_rank, np.linspace(self.rank_grid_low, self.rank_grid_high, int(self.rank_grid_n)))
            def _obj_g(s_thresh: float, t: float) -> float:
                A = (r1_rank >= s_thresh).astype(float)
                B = (r1_rank >= e10).astype(float)
                M = (S2 >= t).astype(float)
                np.fill_diagonal(M, 0.0)
                term1 = float((wi * A)[:, None] @ (wj * M).sum(axis=1, keepdims=True))
                term2 = float((wj * B)[None, :] @ (wi[:, None] * M).sum(axis=0, keepdims=True).T)
                return term1 - term2
            g_vals = []
            for rg in r_grid:
                best_t = t2_grid[0]
                best_val = -1e300
                for t in t2_grid:
                    val = _obj_g(float(rg), float(t))
                    if val > best_val:
                        best_val = val
                        best_t = t
                g_vals.append(best_t)
            # Enforce monotone g over residual grid via isotonic
            self.g_iso_ = IsotonicRegression(increasing=True, out_of_bounds='clip')
            self.g_iso_.fit(r_grid.astype(float), np.array(g_vals, dtype=float))
            # Grid for g^{-1}
            rmin, rmax = float(np.min(r1)), float(np.max(r1))
            pad = 1e-6 + 0.05 * (rmax - rmin if rmax > rmin else 1.0)
            self.grid_r1_ = np.linspace(rmin - pad, rmax + pad, num=1001)
            self.grid_gr_ = self.g_iso_.transform(self.grid_r1_)
            # Transform residuals for second-stage regression
            r1g = self.g_iso_.transform(r1)

            # Fit quantile regressions of g(r1) on X for each tau to obtain alpha2_tau, beta2
            self.beta2_tau_ = {}
            beta2_med = None
            for t in taus:
                rq = sm.QuantReg(r1g, X).fit(q=t, weights=w)
                self.beta2_tau_[float(t)] = rq.params.reshape(-1)
                if abs(float(t) - 0.5) < 1e-9:
                    beta2_med = rq.params.reshape(-1)
            # keep a convenience median beta2 if available
            self.beta2_ = beta2_med
            self.alpha2_ = None

            # Enforce decorrelation between first and second indices via weighted Gram–Schmidt
            if (self.beta1_ is not None) and (self.beta2_tau_ is not None):
                # Build weighted design covariance once
                if w is None:
                    w_norm = np.ones(n, dtype=float)
                else:
                    w_norm = w.reshape(-1)
                    w_norm = w_norm * (n / np.sum(w_norm)) if np.sum(w_norm) > 0 else np.ones(n, dtype=float)
                WX = (w_norm[:, None] * X)
                Sigma = X.T @ WX  # X^T W X
                b1 = self.beta1_.reshape(-1, 1)
                denom = float(b1.T @ Sigma @ b1) + 1e-12
                for kq in list(self.beta2_tau_.keys()):
                    b2 = self.beta2_tau_[kq].reshape(-1, 1)
                    coeff = float((b2.T @ Sigma @ b1) / denom)
                    b2_tilde = (b2 - coeff * b1).reshape(-1)
                    self.beta2_tau_[kq] = b2_tilde
                if self.beta2_ is not None:
                    b2 = self.beta2_.reshape(-1, 1)
                    coeff = float((b2.T @ Sigma @ b1) / denom)
                    self.beta2_ = (b2 - coeff * b1).reshape(-1)

        # Identification scale normalization: divide by ||beta1|| and rescale transform codomains
        if self.beta1_ is not None:
            norm_b1 = float(np.linalg.norm(self.beta1_))
            if norm_b1 > 0:
                self.beta1_ = self.beta1_ / norm_b1
                if self.beta2_tau_ is not None:
                    for kq in list(self.beta2_tau_.keys()):
                        self.beta2_tau_[kq] = self.beta2_tau_[kq] / norm_b1
                if self.beta2_ is not None:
                    self.beta2_ = self.beta2_ / norm_b1
                if self.grid_hy_ is not None:
                    self.grid_hy_ = self.grid_hy_ / norm_b1
                if self.grid_gr_ is not None:
                    self.grid_gr_ = self.grid_gr_ / norm_b1

        # cache training X for diagnostics
        try:
            self._last_X_ = X.copy()
        except Exception:
            self._last_X_ = X
        # record whether weights were used
        self._used_weights_ = (w is not None)
        self._sum_weights_ = None if w is None else float(np.sum(w))
        self._last_y_ = y.copy()
        self._last_w_ = None if w is None else w.copy()
        self.is_fit_ = True
        return self

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration and approximation settings for reproducibility."""
        return {
            "quantiles": list(self.quantiles),
            "use_two_index": self.use_two_index,
            "auto_select_k": self.auto_select_k,
            "alpha_cancor": float(self.alpha_cancor),
            "n_spline_knots": int(self.n_spline_knots),
            "spline_degree": int(self.spline_degree),
            "rank_grid": {
                "n": int(self.rank_grid_n),
                "low": float(self.rank_grid_low),
                "high": float(self.rank_grid_high),
            },
            "t_grid": {
                "n": int(self.t_grid_n),
                "low": float(self.t_grid_low),
                "high": float(self.t_grid_high),
            },
            "subsample_n": None if self.subsample_n is None else int(self.subsample_n),
            "k_cap": 2,
        }

    def _hinv(self, v: np.ndarray) -> np.ndarray:
        """Numerical inverse of h using cached grid and linear interpolation."""
        if self.grid_y_ is None or self.grid_hy_ is None:
            raise RuntimeError("Model not fitted: inverse grid missing")
        # Monotone: invert via interpolation swapping axes
        v_clip = np.clip(v, float(np.min(self.grid_hy_)), float(np.max(self.grid_hy_)))
        return np.interp(v_clip, self.grid_hy_, self.grid_y_)

    def _ginv(self, v: np.ndarray) -> np.ndarray:
        """Numerical inverse of g using cached grid and linear interpolation."""
        if self.grid_r1_ is None or self.grid_gr_ is None:
            raise RuntimeError("Two-index components not fitted: g^{-1} grid missing")
        v_clip = np.clip(v, float(np.min(self.grid_gr_)), float(np.max(self.grid_gr_)))
        return np.interp(v_clip, self.grid_gr_, self.grid_r1_)

    def predict_quantiles(self, X: ArrayLike, quantiles: Optional[Sequence[float]] = None, return_continuous: bool = False) -> Dict[float, np.ndarray] | Tuple[Dict[float, np.ndarray], Dict[float, np.ndarray]]:
        if not self.is_fit_:
            raise RuntimeError("Call fit() before predict_quantiles().")
        X = _as_2d(X).astype(float)
        taus = [float(t) for t in (self.quantiles if quantiles is None else quantiles)]
        if not all(0 < t < 1 for t in taus):
            raise ValueError("All quantiles must be in (0,1)")

        preds: Dict[float, np.ndarray] = {}
        preds_cont: Dict[float, np.ndarray] = {}
        xb1 = X @ self.beta1_  # type: ignore[arg-type]
        for t in taus:
            # First index contribution (no explicit alpha due to normalization)
            u1 = xb1
            if self.use_two_index and (self.beta2_tau_ is not None):
                # Second index per paper: r_tau = g^{-1}(X beta2_tau)
                b2t = self.beta2_tau_.get(float(t), self.beta2_)
                if b2t is None:
                    b2t = next(iter(self.beta2_tau_.values()))
                u2 = (X @ b2t)
                r_tau = self._ginv(u2)
                u1 = u1 + r_tau
            y_hat_cont = self._hinv(u1)
            # Clip to category bounds and floor
            if (self._y_min_cat is not None) and (self._y_max_cat is not None):
                y_hat_cont = np.clip(y_hat_cont, self._y_min_cat, self._y_max_cat + 0.999)
            y_floor = np.floor(y_hat_cont).astype(int)
            if (self._y_min_cat is not None) and (self._y_max_cat is not None):
                y_floor = np.clip(y_floor, self._y_min_cat, self._y_max_cat)
            preds[t] = y_floor
            preds_cont[t] = y_hat_cont
        return (preds, preds_cont) if return_continuous else preds

    def predict_quantiles_continuous(self, X: ArrayLike, quantiles: Optional[Sequence[float]] = None) -> Dict[float, np.ndarray]:
        """Return continuous-scale quantile predictions before flooring to categories."""
        out = self.predict_quantiles(X, quantiles=quantiles, return_continuous=True)
        return out[1]  # type: ignore[index]

    def make_prediction_interval(self, X: ArrayLike, tau_low: float = 0.25, tau_high: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
        return self.predict_interval(X, tau_low=tau_low, tau_high=tau_high)

    def evaluate_intervals(self, X: ArrayLike, y_true: ArrayLike, tau_low: float = 0.25, tau_high: float = 0.75, sample_weight: Optional[ArrayLike] = None) -> Dict[str, Any]:
        y_true = np.asarray(y_true).reshape(-1)
        lo, hi = self.make_prediction_interval(X, tau_low, tau_high)
        L = (hi - lo).reshape(-1)
        cover = ((y_true >= lo.reshape(-1)) & (y_true <= hi.reshape(-1))).astype(float)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
        if (w is not None) and (w.shape[0] == y_true.shape[0]):
            w = w / (np.sum(w) if np.sum(w) > 0 else 1.0)
            mean_L = float(np.sum(w * L))
            mean_cover = float(np.sum(w * cover))
        else:
            mean_L = float(np.mean(L))
            mean_cover = float(np.mean(cover))
        # Length histogram counts 0..(max_cat-min_cat)
        max_len = int((self._y_max_cat or int(np.max(y_true))) - (self._y_min_cat or int(np.min(y_true))))
        bins = list(range(0, max_len + 2))
        hist = np.bincount(np.clip(L.astype(int), 0, max_len), minlength=max_len + 1)
        # Coverage by interval length L for L=0..4 (Table 6 style); aggregate L>=5 as "5+"
        L_int = np.clip(L.astype(int), 0, 10**9)
        coverage_by_length: Dict[str, Optional[float]] = {}
        counts_by_length: Dict[str, int] = {}
        for ell in range(0, 5):
            mask = (L_int == ell)
            if w is not None:
                w_denom = float(np.sum(w[mask]))
                cov = None if w_denom <= 0 else float(np.sum(w[mask] * cover[mask]) / w_denom)
                cnt = int(np.sum(mask))
            else:
                cnt = int(np.sum(mask))
                cov = None if cnt == 0 else float(np.mean(cover[mask]))
            coverage_by_length[str(ell)] = cov
            counts_by_length[str(ell)] = cnt
        # Aggregate 5+
        mask5 = (L_int >= 5)
        if np.any(mask5):
            if w is not None:
                w_denom5 = float(np.sum(w[mask5]))
                cov5 = None if w_denom5 <= 0 else float(np.sum(w[mask5] * cover[mask5]) / w_denom5)
            else:
                cnt5 = int(np.sum(mask5))
                cov5 = None if cnt5 == 0 else float(np.mean(cover[mask5]))
            coverage_by_length["5+"] = cov5
            counts_by_length["5+"] = int(np.sum(mask5))
        return {
            "mean_length": mean_L,
            "mean_coverage": mean_cover,
            "length_hist": hist.astype(int).tolist(),
            "coverage_by_length": coverage_by_length,
            "counts_by_length": counts_by_length,
            "tau_low": tau_low,
            "tau_high": tau_high,
        }

    def get_reporting_scaled_betas(self, feature_index: int) -> Dict[str, Any]:
        """
        Create reporting-only rescaled coefficients so that the specified feature's
        coefficient equals 1 in each index vector (when nonzero). This matches
        presentation like Table 3 where a chosen variable is normalized to 1.

        Returns a dict with copies; model parameters are unchanged.
        """
        if self.beta1_ is None:
            raise RuntimeError("Model not fitted.")
        j = int(feature_index)
        out: Dict[str, Any] = {}
        # Beta1
        b1 = self.beta1_.reshape(-1)
        if not (0 <= j < b1.shape[0]):
            raise IndexError("feature_index out of bounds for beta1")
        s1 = None if abs(b1[j]) < 1e-12 else (1.0 / float(b1[j]))
        out["beta1_scale"] = s1
        out["beta1_scaled"] = None if s1 is None else (b1 * s1)
        # Beta2 per tau, if available
        if self.beta2_tau_ is not None:
            b2_scaled: Dict[float, np.ndarray | None] = {}
            b2_scales: Dict[float, Optional[float]] = {}
            for kq, b2 in self.beta2_tau_.items():
                if not (0 <= j < b2.shape[0]):
                    b2_scales[float(kq)] = None
                    b2_scaled[float(kq)] = None
                else:
                    s2 = None if abs(b2[j]) < 1e-12 else (1.0 / float(b2[j]))
                    b2_scales[float(kq)] = s2
                    b2_scaled[float(kq)] = None if s2 is None else (b2 * s2)
            out["beta2_tau_scales"] = b2_scales
            out["beta2_tau_scaled"] = b2_scaled
        else:
            out["beta2_tau_scales"] = None
            out["beta2_tau_scaled"] = None
        return out

    def plot_transforms(self, ax=None):
        try:
            import matplotlib.pyplot as plt
        except Exception:
            raise RuntimeError("matplotlib is required for plot_transforms()")
        hx = self.grid_y_ if self.grid_y_ is not None else np.linspace(0, 1, 100)
        hy = self.grid_hy_ if self.grid_hy_ is not None else hx
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        else:
            # require an axes sequence of length 2 when provided
            if not isinstance(ax, (list, tuple, np.ndarray)) or len(ax) != 2:
                raise ValueError("ax must be a sequence of two matplotlib axes when provided")
        ax0 = ax[0] if isinstance(ax, (list, tuple, np.ndarray)) else ax
        ax0.plot(hx, hy, label='h(y)'); ax0.set_title('Estimated h'); ax0.set_xlabel('y'); ax0.set_ylabel('h(y)')
        if self.grid_r1_ is not None and self.grid_gr_ is not None:
            if isinstance(ax, (list, tuple, np.ndarray)):
                ax1 = ax[1]
            else:
                ax1 = ax0.figure.add_subplot(1, 2, 2)
            ax1.plot(self.grid_r1_, self.grid_gr_, label='g(r1)'); ax1.set_title('Estimated g'); ax1.set_xlabel('r1'); ax1.set_ylabel('g(r1)')
        return ax

    def predict_interval(self, X: ArrayLike, tau_low: float = 0.25, tau_high: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
        if not (0 < tau_low < tau_high < 1):
            raise ValueError("Require 0 < tau_low < tau_high < 1")
        qs = self.predict_quantiles(X, quantiles=(tau_low, tau_high))
        return qs[tau_low], qs[tau_high]

    def summary(self) -> Dict[str, Any]:
        if not self.is_fit_:
            raise RuntimeError("Model not fitted.")
        out: Dict[str, Any] = {
            "beta1": None if self.beta1_ is None else self.beta1_.copy(),
            "alpha1": None if self.alpha1_ is None else dict(self.alpha1_),
            "two_index": self.use_two_index,
        }
        if self.use_two_index:
            out.update({
                "beta2": None if self.beta2_ is None else self.beta2_.copy(),
                "beta2_tau": None if self.beta2_tau_ is None else {float(k): v.copy() for k, v in self.beta2_tau_.items()},
                "alpha2": None if self.alpha2_ is None else dict(self.alpha2_),
            })
        # CANCOR selection diagnostics
        out["selected_k"] = self.selected_k_
        out["selected_k_full"] = self.selected_k_full_
        out["cancor_gammas"] = None if self.cancor_gammas_ is None else self.cancor_gammas_.copy()
        out["cancor_pvalues_seq"] = None if self.cancor_pvalues_seq_ is None else self.cancor_pvalues_seq_.copy()
        # Identification note
        out["identification"] = {
            "location_constraints": True,
            "norm_scaling_by_beta1": True,
        }
        out["k_truncated"] = bool(self.k_truncated_) if self.k_truncated_ is not None else False
        # Approximation settings and implementation limits
        out["approximation"] = {
            "rank_grid": {"n": int(self.rank_grid_n), "low": float(self.rank_grid_low), "high": float(self.rank_grid_high)},
            "t_grid": {"n": int(self.t_grid_n), "low": float(self.t_grid_low), "high": float(self.t_grid_high)},
            "subsample_n": None if self.subsample_n is None else int(self.subsample_n),
        }
        out["k_cap"] = 2
        out["used_weights"] = bool(self._used_weights_)
        # Correlation diagnostics between indices (after orthogonalization)
        try:
            X_last = getattr(self, "_last_X_", None)
            if X_last is None:
                corr_diag = None
            else:
                xb1 = X_last @ self.beta1_ if self.beta1_ is not None else None
                corr_map: Dict[float, float] = {}
                if self.use_two_index and self.beta2_tau_ is not None and xb1 is not None:
                    for kq, b2 in self.beta2_tau_.items():
                        xb2 = X_last @ b2
                        c = float(np.corrcoef(xb1, xb2)[0, 1]) if (np.std(xb1) > 0 and np.std(xb2) > 0) else 0.0
                        corr_map[float(kq)] = c
                corr_diag = corr_map if corr_map else None
        except Exception:
            corr_diag = None
        out["corr_xbeta1_xbeta2"] = corr_diag
        return out

    def bootstrap_inference(self, n_boot: int = 200, block_length: Optional[int] = None, ci: float = 0.95, random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Bootstrap standard errors and CIs for beta1 (q=0.5) and beta2,τ (if two-index).

        Notes
        - Holds h(.) and g(.) fixed at their fitted values; re-fits the quantile regressions
          within each bootstrap replicate. This captures QR-stage variability but not transform
          estimation variability.
        - If block_length is provided (>=1), uses moving block bootstrap on the training order.
        """
        if not self.is_fit_:
            raise RuntimeError("Model not fitted.")
        if self._last_X_ is None or self._last_y_ is None:
            raise RuntimeError("Training data not cached for bootstrap.")
        X = np.asarray(self._last_X_)
        y = np.asarray(self._last_y_)
        w = None if self._last_w_ is None else np.asarray(self._last_w_)
        n = X.shape[0]
        taus = [float(t) for t in self.quantiles]
        rng = np.random.default_rng(random_state)

        def draw_indices_iid() -> np.ndarray:
            if w is not None and np.sum(w) > 0:
                p = w / np.sum(w)
                return rng.choice(n, size=n, replace=True, p=p)
            return rng.choice(n, size=n, replace=True)

        def draw_indices_blocks(blen: int) -> np.ndarray:
            # circular moving blocks
            num_blocks = int(np.ceil(n / blen))
            starts = rng.integers(low=0, high=n, size=num_blocks)
            idx = []
            for s in starts:
                block = [(s + t) % n for t in range(blen)]
                idx.extend(block)
            return np.array(idx[:n], dtype=int)

        b1_dim = self.beta1_.shape[0] if self.beta1_ is not None else X.shape[1]
        b1_boot = np.zeros((n_boot, b1_dim), dtype=float)
        b2_boot: Dict[float, np.ndarray] = {t: np.zeros((n_boot, X.shape[1]), dtype=float) for t in (taus if self.use_two_index else [])}

        for b in range(n_boot):
            idx = draw_indices_blocks(int(block_length)) if (block_length is not None and int(block_length) >= 1) else draw_indices_iid()
            Xb = X[idx, :]
            yb = y[idx]
            wb = None if w is None else w[idx]
            # transform and first-stage QR
            hyb = self.h_iso_.transform(_jitter_ordinal(yb, random_state=rng.integers(0, 2**31 - 1)))  # jitter per replicate
            rq1 = sm.QuantReg(hyb, Xb).fit(q=0.5, weights=wb)
            b1_boot[b, :] = rq1.params.reshape(-1)
            if self.use_two_index and self.g_iso_ is not None:
                r1b = hyb - (Xb @ rq1.params.reshape(-1))
                r1gb = self.g_iso_.transform(r1b)
                for t in taus:
                    rq2 = sm.QuantReg(r1gb, Xb).fit(q=t, weights=wb)
                    b2_boot[t][b, :] = rq2.params.reshape(-1)

        alpha = 1.0 - float(ci)
        def se_and_ci(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            se = np.std(arr, axis=0, ddof=1)
            lo = np.quantile(arr, alpha / 2.0, axis=0)
            hi = np.quantile(arr, 1.0 - alpha / 2.0, axis=0)
            return se, lo, hi

        b1_se, b1_lo, b1_hi = se_and_ci(b1_boot)
        out: Dict[str, Any] = {"beta1": {"se": b1_se, "ci_low": b1_lo, "ci_high": b1_hi, "ci_level": ci}}
        if self.use_two_index:
            b2_out: Dict[float, Dict[str, Any]] = {}
            for t, mat in b2_boot.items():
                se, lo, hi = se_and_ci(mat)
                b2_out[float(t)] = {"se": se, "ci_low": lo, "ci_high": hi, "ci_level": ci}
            out["beta2_tau"] = b2_out
        # cache
        self.beta1_boot_ = {"draws": b1_boot, **out["beta1"]}
        if self.use_two_index:
            self.beta2_tau_boot_ = {float(t): {"draws": b2_boot[t], **b2d} for t, b2d in out["beta2_tau"].items()}
        return out


