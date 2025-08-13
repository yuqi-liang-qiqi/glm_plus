"""
Utilities to extend Bayesian Ordinal Quantile Regression (OQR) to panel/time settings.

This module builds on the existing OR1/OR2 implementations by:
- constructing a design matrix with year fixed effects and gender×year interactions;
- fitting OQR at multiple quantiles using the existing samplers;
- extracting per-year female−male differences with posterior summaries;
- computing probabilities such as P(diff_top > diff_bottom).

Dependencies: numpy, pandas, matplotlib (optional for plotting)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .ori import quantregOR1
from .orii import quantregOR2


@dataclass
class PanelDesign:
    y: np.ndarray                 # (n, 1) integer 1..J
    X: np.ndarray                 # (n, k)
    x_names: List[str]            # length k
    year_levels: List[str]        # ordered year factor levels (strings)
    base_year: str                # year level used as baseline (dropped dummy)
    gender_col: str               # name of gender column in X
    interaction_prefix: str       # prefix used for interaction columns
    gender_is_female_one: bool = True  # interpret effect as female (1) minus male (0)


def build_panel_design(
    df: pd.DataFrame,
    outcome_col: str,
    year_col: str,
    gender_col: str,
    controls: Sequence[str] | None = None,
    cohort_col: Optional[str] = None,
    drop_first_year: bool = True,
    gender_is_female_one: bool = True,
) -> PanelDesign:
    """Create design matrix with year fixed effects and gender×year interactions.

    Inputs
    - df: long panel. Must include integer ordinal `outcome_col` in {1..J}.
    - year_col: will be treated as categorical (factor).
    - gender_col: assumed numeric 0/1. If not 0/1, values are kept as-is.
    - controls: list of additional covariate columns. Numeric columns are used
      as-is; categorical columns are one-hot encoded (drop_first=True).
    - cohort_col: optional cohort factor for additional fixed effects.
    - drop_first_year: whether to drop the baseline year dummy to avoid collinearity.

    Returns
    - PanelDesign with y, X, names, and metadata needed for downstream summaries.
    """
    if controls is None:
        controls = []

    df_local = df.copy()

    # Outcome vector y (n, 1)
    y = df_local[outcome_col].to_numpy().reshape(-1, 1)

    # Ensure year is a categorical with fixed ordering
    df_local[year_col] = pd.Categorical(df_local[year_col], ordered=True)
    year_levels = [str(v) for v in df_local[year_col].cat.categories.tolist()]

    # Gender main effect
    gender_series = df_local[gender_col].astype(float)

    # Year dummies
    year_dummies = pd.get_dummies(df_local[year_col], prefix=f"{year_col}", drop_first=drop_first_year)
    if drop_first_year:
        base_year = year_levels[0]
    else:
        base_year = "(none)"

    # Controls: numeric as-is; categorical -> dummies (drop first)
    control_blocks: List[pd.DataFrame] = []
    for col in controls:
        if pd.api.types.is_numeric_dtype(df_local[col]):
            control_blocks.append(df_local[[col]].astype(float).rename(columns={col: col}))
        else:
            d = pd.get_dummies(df_local[col], prefix=col, drop_first=True)
            control_blocks.append(d)
    controls_mat = pd.concat(control_blocks, axis=1) if control_blocks else pd.DataFrame(index=df_local.index)

    # Gender × year interactions: for each included year dummy
    # Interaction naming convention: f"{gender_col}__X__{year_col}:{year_col}_{level}"
    interaction_prefix = f"{gender_col}__X__{year_col}"
    interactions = pd.DataFrame(index=df_local.index)
    for c in year_dummies.columns:
        interactions[f"{interaction_prefix}:{c}"] = gender_series.values * year_dummies[c].values

    # Assemble X: intercept + gender + year FE + interactions + controls (+ cohort FE)
    X_parts: List[pd.DataFrame] = [
        pd.DataFrame({"Intercept": np.ones(len(df_local), dtype=float)}),
        pd.DataFrame({gender_col: gender_series.astype(float)}),
        year_dummies.astype(float),
        interactions.astype(float),
    ]
    if not controls_mat.empty:
        X_parts.append(controls_mat.astype(float))
    if cohort_col is not None:
        cohort_fe = pd.get_dummies(df_local[cohort_col], prefix=str(cohort_col), drop_first=True)
        X_parts.append(cohort_fe.astype(float))

    X_df = pd.concat(X_parts, axis=1)
    x_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=float)

    return PanelDesign(
        y=y,
        X=X,
        x_names=x_names,
        year_levels=year_levels,
        base_year=base_year,
        gender_col=gender_col,
        interaction_prefix=interaction_prefix,
        gender_is_female_one=gender_is_female_one,
    )


@dataclass
class PanelOQRFit:
    quantiles: List[float]
    fits: Dict[float, Dict[str, np.ndarray]]  # p -> model result dict
    x_names: List[str]
    year_levels: List[str]
    base_year: str
    gender_col: str
    interaction_prefix: str


def fit_panel_oqr(
    design: PanelDesign,
    burn: int = 2000,
    mcmc: int = 8000,
    quantiles: Sequence[float] = (0.2, 0.5, 0.8),
    B0_scale: float = 10.0,
    D0_scale: float = 0.25,
    verbose: bool = False,
) -> PanelOQRFit:
    """Fit Bayesian OQR at selected quantiles using OR1 (J>=3) or OR2 (J==3).

    The design already includes year FE and gender×year interactions.
    """
    y = design.y
    X = design.X
    x_names = design.x_names
    J = int(np.unique(y).size)
    n, k = X.shape

    # Priors
    b0 = np.zeros((k, 1))
    B0 = np.eye(k) * B0_scale

    fits: Dict[float, Dict[str, np.ndarray]] = {}

    if J == 3:
        # OR2 priors
        n0 = 5.0
        d0 = 8.0
        for p in quantiles:
            res = quantregOR2(
                y=y,
                x=X,
                b0=b0,
                B0=B0,
                n0=n0,
                d0=d0,
                burn=burn,
                mcmc=mcmc,
                p=float(p),
                verbose=verbose,
            )
            fits[float(p)] = res
    else:
        # OR1 priors
        d0 = np.zeros((J - 2, 1))
        D0 = np.eye(J - 2) * D0_scale
        for p in quantiles:
            res = quantregOR1(
                y=y,
                x=X,
                b0=b0,
                B0=B0,
                d0=d0,
                D0=D0,
                burn=burn,
                mcmc=mcmc,
                p=float(p),
                tune=0.1,
                verbose=verbose,
            )
            fits[float(p)] = res

    return PanelOQRFit(
        quantiles=list(map(float, quantiles)),
        fits=fits,
        x_names=x_names,
        year_levels=design.year_levels,
        base_year=design.base_year,
        gender_col=design.gender_col,
        interaction_prefix=design.interaction_prefix,
    )


def _gender_effect_draws_per_year(
    betadraws: np.ndarray,
    x_names: List[str],
    year_levels: List[str],
    base_year: str,
    gender_col: str,
    interaction_prefix: str,
) -> Dict[str, np.ndarray]:
    """Return per-year draws for female−male effect as linear combos of beta draws.

    For the baseline year (if dropped), effect = beta_gender.
    For other years, effect = beta_gender + beta_(gender×year_t).
    """
    # Locate column indices deterministically
    name_to_idx = {n: i for i, n in enumerate(x_names)}
    if gender_col not in name_to_idx:
        raise KeyError(f"gender column '{gender_col}' not found in design names")
    idx_gender = name_to_idx[gender_col]
    draws = betadraws  # (k, nsim)

    # Parse year_col from interaction_prefix pattern: "{gender}__X__{year}"
    try:
        year_col = interaction_prefix.split("__X__")[1]
    except Exception as e:
        raise ValueError(f"interaction_prefix format unexpected: {interaction_prefix}") from e

    out: Dict[str, np.ndarray] = {}
    for yv in year_levels:
        if base_year != "(none)" and yv == base_year:
            # With dropped baseline, baseline year's effect is the gender main effect
            eff = draws[idx_gender, :].copy()
        else:
            inter_name = f"{interaction_prefix}:{year_col}_{yv}"
            if inter_name in name_to_idx:
                idx_inter = name_to_idx[inter_name]
                eff = draws[idx_gender, :] + draws[idx_inter, :]
            else:
                # If a specific interaction column is absent (e.g., that level never appears),
                # fall back to the gender main effect for robustness.
                eff = draws[idx_gender, :].copy()
        out[str(yv)] = eff.reshape(1, -1)
    return out


def extract_gender_year_effects(
    fit: PanelOQRFit,
    p: float,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """For a given quantile p, return per-year female−male effect summaries and raw draws.

    Returns
    - summary_df: rows = year, columns = mean, l95, u95
    - draws_dict: mapping year -> (1, nsim) array of posterior draws
    """
    model = fit.fits[float(p)]
    betadraws = model["betadraws"]  # (k, nsim)
    draws_dict = _gender_effect_draws_per_year(
        betadraws=betadraws,
        x_names=fit.x_names,
        year_levels=fit.year_levels,
        base_year=fit.base_year,
        gender_col=fit.gender_col,
        interaction_prefix=fit.interaction_prefix,
    )

    rows = []
    # Preserve original year ordering from design
    for yv in fit.year_levels:
        v = draws_dict[str(yv)].reshape(-1)
        rows.append({
            "year": str(yv),
            "mean": float(np.mean(v)),
            "l95": float(np.quantile(v, 0.025)),
            "u95": float(np.quantile(v, 0.975)),
        })
    summary_df = pd.DataFrame(rows)
    return summary_df, draws_dict


def probability_top_greater_bottom(
    fit: PanelOQRFit,
    p_top: float = 0.8,
    p_bottom: float = 0.2,
    aggregate: bool = True,
) -> Tuple[float, pd.DataFrame]:
    """Compute P(diff_top > diff_bottom).

    If aggregate=True, we average the per-year draws first, then compare.
    Otherwise, we compute probability per year and return the mean of those
    probabilities along with a per-year table.
    """
    _, top_draws = extract_gender_year_effects(fit, p=float(p_top))
    _, bot_draws = extract_gender_year_effects(fit, p=float(p_bottom))

    years = [str(y) for y in fit.year_levels]
    rows = []
    probs_per_year: List[float] = []
    for yv in years:
        vt = top_draws[yv].reshape(-1)
        vb = bot_draws[yv].reshape(-1)
        pr = float(np.mean((vt - vb) > 0.0))
        probs_per_year.append(pr)
        rows.append({"year": yv, "P(top>bottom)": pr})
    table = pd.DataFrame(rows)

    if aggregate:
        # Average draws over years then compare
        vt_all = np.vstack([top_draws[yv] for yv in years]).mean(axis=0).reshape(-1)
        vb_all = np.vstack([bot_draws[yv] for yv in years]).mean(axis=0).reshape(-1)
        p_overall = float(np.mean((vt_all - vb_all) > 0.0))
    else:
        p_overall = float(np.mean(probs_per_year))
    return p_overall, table


def plot_gender_gap_trends(
    fit: PanelOQRFit,
    quantiles: Optional[Sequence[float]] = None,
    ax=None,
    title: Optional[str] = None,
):
    """Plot time trends of female−male differences for selected quantiles.

    Requires matplotlib. Returns the matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    qs = list(map(float, quantiles if quantiles is not None else fit.quantiles))
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    for p in qs:
        df_sum, _ = extract_gender_year_effects(fit, p)
        # Ensure plotting order follows original year_levels
        df_sum["year"] = pd.Categorical(df_sum["year"], categories=[str(y) for y in fit.year_levels], ordered=True)
        df_sum = df_sum.sort_values("year")
        ax.plot(df_sum["year"], df_sum["mean"], marker="o", label=f"p={p}")
        ax.fill_between(df_sum["year"], df_sum["l95"], df_sum["u95"], alpha=0.2)

    ax.axhline(0.0, color="k", lw=1, ls=":")
    ax.set_xlabel("Year")
    ax.set_ylabel("Female − Male (latent readiness)")
    if title:
        ax.set_title(title)
    ax.legend()
    return ax


