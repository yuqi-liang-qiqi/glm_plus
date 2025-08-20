# Gologit2: Generalized Ordered Logit Models for Python

A Python implementation of generalized ordered logit models, inspired by Stata's `gologit2` command. This implementation provides reliable ordered categorical regression with optional parallel-lines constraints and robust statistical inference.

## Features

- **Multiple link functions**: logit, probit, cloglog, loglog, cauchit
- **Flexible constraints**: Mix parallel-lines (proportional odds) and non-parallel variables
- **Robust inference**: Standard errors via numerical Hessian with Delta method for cutpoints
- **Statistical measures**: p-values, pseudo R², model diagnostics
- **Numerical stability**: Advanced matrix inversion strategies and condition number monitoring
- **Stata compatibility**: Equivalent parameter interpretations with clear documentation of differences

## Quick Start

```python
import numpy as np
from glm_plus.gologit2 import GeneralizedOrderedModel

# Generate sample data
np.random.seed(42)
n = 1000
X = np.random.randn(n, 3)
# Create ordered response with coefficients
linear = X @ np.array([0.8, -0.5, 1.2])
y = np.ones(n)
y[linear > -0.5] = 2
y[linear > 0.5] = 3
y[linear > 1.5] = 4

# Fit generalized ordered model
model = GeneralizedOrderedModel(link="logit", pl_vars=["x1"])  # x1 has same effect across thresholds
result = model.fit(X, y, feature_names=["x1", "x2", "x3"])

# View results with standard errors and p-values
print(result.summary())

# Make predictions
probs = model.predict_proba(X[:10])  # Predicted probabilities
xb = model.predict_xb(X[:10], equation=1)  # Linear predictor for threshold 1
```

## Comparison with Stata's gologit2

### Similarities
- **Parameter interpretations**: Coefficient magnitudes and directions are identical
- **Model flexibility**: Same support for mixing parallel-lines and non-parallel variables
- **Link functions**: All major link functions supported (logit, probit, cloglog, loglog, cauchit)
- **Statistical inference**: Provides standard errors, p-values, and pseudo R²

### Key Differences

| Aspect | Our Implementation | Stata gologit2 |
|--------|-------------------|-----------------|
| **Cutpoint signs** | F(α - Xβ) parameterization | F(-α + Xβ) parameterization |
| **Standard errors** | Numerical Hessian + Delta method | Analytical derivatives |
| **Matrix inversion** | Cholesky → Standard → Pseudo-inverse | Standard approach |
| **Monotonic constraints** | Reparameterization with exponential gaps | Direct constraints |
| **Diagnostics** | Extensive condition number reporting | Basic convergence info |
| **Prediction safety** | Automatic probability clipping/normalization | Raw probabilities |

### Technical Notes

**Cutpoint Parameterization**: We use F(α - Xβ) while Stata uses F(-α + Xβ). This means our cutpoints (α) have opposite signs, but coefficient interpretations (β) are identical.

**Standard Errors**: Our cutpoint standard errors use the Delta method to properly account for monotonic reparameterization, ensuring correct inference under the constraint α₁ < α₂ < ... < αₖ.

**Numerical Stability**: We employ multiple safeguards:
- Symmetric matrix eigenvalue computation (`eigvalsh` vs `eigvals`)
- Layered matrix inversion strategies
- Comprehensive condition number monitoring
- 3-point finite differences with optimal step sizes

## API Reference

### GeneralizedOrderedModel

**Parameters:**
- `link`: str, default "logit" - Link function: {"logit", "probit", "cloglog", "loglog", "cauchit"}
- `pl_vars`: list of str, optional - Variables constrained to have parallel lines (same effect across thresholds)

**Methods:**

#### fit(X, y, feature_names=None, maxiter=200, tol=1e-6, verbose=True)
Fit the model to data.

**Parameters:**
- `X`: array-like, shape (n_samples, n_features) - Feature matrix (no intercept)
- `y`: array-like, shape (n_samples,) - Ordinal response variable
- `feature_names`: list of str, optional - Names for features
- `maxiter`: int, default 200 - Maximum optimization iterations
- `tol`: float, default 1e-6 - Convergence tolerance
- `verbose`: bool, default True - Print progress messages

**Returns:** `Gologit2Result` object with fitted parameters and statistics

#### predict_proba(X)
Predict class probabilities.

**Parameters:**
- `X`: array-like, shape (n_samples, n_features) - Feature matrix

**Returns:** ndarray, shape (n_samples, n_classes) - Predicted probabilities

#### predict_xb(X, equation)
Predict linear predictor for specific threshold equation.

**Parameters:**
- `X`: array-like, shape (n_samples, n_features) - Feature matrix  
- `equation`: int - Threshold equation number (1 to K)

**Returns:** ndarray, shape (n_samples,) - Linear predictor values

### Gologit2Result

**Key Attributes:**
- `alphas`: Cutpoint parameters (intercepts)
- `beta_pl`: Parallel-lines coefficients
- `beta_npl`: Non-parallel coefficients (shape: K × p_npl)
- `se_alphas`, `se_beta_pl`, `se_beta_npl`: Standard errors
- `pvalues_alphas`, `pvalues_beta_pl`, `pvalues_beta_npl`: p-values
- `pseudo_r2`: McFadden's pseudo R²
- `pseudo_r2_adj`: Adjusted pseudo R²
- `success`: Convergence indicator
- `hessian_condition_number`: Numerical stability diagnostic

**Methods:**
- `summary()`: Formatted results table with significance stars
- `params_as_dict()`: Dictionary of all parameters

## Advanced Usage

### Mixed Constraints Example

```python
# Some variables with parallel lines, others varying by threshold
model = GeneralizedOrderedModel(
    link="logit", 
    pl_vars=["age", "education"]  # Same effects across thresholds
)
# Variables "income" and "region" will vary by threshold
result = model.fit(X, y, feature_names=["age", "education", "income", "region"])
```

### Different Link Functions

```python
# Probit link for symmetric errors
model_probit = GeneralizedOrderedModel(link="probit")

# Complementary log-log for extreme value distributions  
model_cloglog = GeneralizedOrderedModel(link="cloglog")

# Compare models using pseudo R²
r2_logit = result_logit.pseudo_r2
r2_probit = result_probit.pseudo_r2
```

### Diagnostics and Model Assessment

```python
result = model.fit(X, y, verbose=True)

# Check convergence and numerical stability
if result.success:
    print(f"Model converged in {result.n_iter} iterations")
    print(f"Hessian condition number: {result.hessian_condition_number:.2e}")
    
    # High condition numbers indicate potential numerical issues
    if result.cov_condition_number > 1e12:
        print("Warning: High condition number - consider model simplification")
        
# Model comparison
print(f"McFadden R²: {result.pseudo_r2:.4f}")
print(f"Adjusted R²: {result.pseudo_r2_adj:.4f}")
```

## Installation Notes

This module is part of the `glm_plus` package. Ensure you have:
- Python 3.8+
- NumPy ≥ 1.20
- SciPy ≥ 1.7 (for numerical differentiation)
- Pandas (optional, for DataFrame support)

## Troubleshooting

### Common Issues

**"Optimization failed"**: Try simpler models first:
- Add parallel-lines constraints to reduce parameters
- Check for sparse categories (merge rare outcomes)
- Verify data quality (no missing values, reasonable ranges)

**"High condition number"**: Indicates numerical instability:
- Consider variable scaling/centering
- Remove highly collinear predictors
- Try different link functions

**"Negative probabilities"**: During training indicates infeasible parameters:
- Add more parallel-lines constraints
- Merge sparse outcome categories
- Check predictor ranges for extreme values

### Performance Tips

- Start with parallel-lines constraints for all variables, then relax selectively
- Center and scale continuous predictors
- Use `verbose=True` to monitor convergence
- For large datasets, consider increasing `tol` to 1e-5

## Citation

If you use this implementation in academic work, please cite:

```
Liang, Y. (2024). gologit2: Generalized Ordered Logit Models for Python. 
GLM Plus Package. https://github.com/Liang-Team/glm_plus
```

For the underlying methodology, refer to:
- Williams, R. (2006). Generalized ordered logit/partial proportional odds models for ordinal dependent variables. *The Stata Journal*, 6(1), 58-82.
- Long, J.S. & Freese, J. (2014). *Regression Models for Categorical Dependent Variables Using Stata*, 3rd ed. Stata Press.

## License

Part of the GLM Plus package. See main repository for license details.
