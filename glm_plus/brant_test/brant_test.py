"""
Brant Test for Proportional Odds Assumption in Python

This module implements the Brant test to check the parallel regression assumption
(proportional odds assumption) in ordinal logistic regression models.

The Brant test evaluates whether the relationship between each predictor and 
the logit is the same across all thresholds of the ordinal outcome variable.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cho_factor, cho_solve, LinAlgError as CholeskyError
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from sklearn.preprocessing import LabelEncoder
import warnings
from typing import Dict, List, Tuple, Optional, Union


def brant_test(model=None, X: np.ndarray = None, y: np.ndarray = None, 
               feature_names: List[str] = None, by_var: bool = False,
               variable_groups: Dict[str, List[int]] = None,
               categories_order: List = None, alpha: float = 0.05,
               drop_constant: bool = False, return_internals: bool = False,
               verbose: bool = True) -> Dict:
    """
    Performs the Brant test for proportional odds assumption.
    
    The Brant test checks whether the coefficients are consistent across 
    different thresholds of the ordinal outcome variable. If the test is 
    significant, it suggests violation of the proportional odds assumption.
    
    Parameters:
    -----------
    model : fitted ordinal regression model, optional
        The fitted ordinal logistic regression model (currently unused, 
        provided for API compatibility)
    X : np.ndarray
        Design matrix (predictors) without intercept
    y : np.ndarray  
        Ordinal outcome variable 
    feature_names : List[str], optional
        Names of the predictor variables
    by_var : bool, default=False
        If True, group variables together using variable_groups for joint testing
    variable_groups : Dict[str, List[int]], optional
        Mapping of variable names to column indices for grouped testing.
        E.g., {'education': [0,1,2], 'region': [3,4]} groups dummy variables.
        If None and by_var=True, each column is treated as separate variable.
    categories_order : List, optional
        Explicit ordering for ordinal categories. If None, uses natural sorting.
        Useful for string categories that need custom ordering.
    alpha : float, default=0.05
        Significance threshold for determining statistical significance.
        Tests with p-value < alpha are considered significant.
    drop_constant : bool, default=False
        If True, automatically drop columns with near-zero variance.
        Dropped columns are recorded in diagnostics.
    return_internals : bool, default=False
        If True, return internal matrices (D, var_beta, beta_star) in 
        results['internals'] for debugging and verification.
    verbose : bool, default=True
        If True, print progress and diagnostic information
        
    Returns:
    --------
    Dict : Dictionary containing test results with chi-square statistics,
           degrees of freedom, and p-values
    """
    
    # Step 1: Data preparation and validation
    if verbose:
        print("Step 1: Preparing data for Brant test...")
    
    # Input validation
    if X is None or y is None:
        raise ValueError("X and y must be provided")
    
    # Get data dimensions first
    n_obs, n_features = X.shape
    
    # Ensure y is properly encoded as ordinal starting from 1 
    y_encoded = _encode_ordinal_outcome(y, categories_order=categories_order)
    
    # Validate variable_groups if by_var is used
    if by_var and variable_groups:
        _validate_variable_groups(variable_groups, n_features, feature_names)
    
    # Check for constant or zero columns in X and optionally drop them
    X_variance = np.var(X, axis=0)
    constant_cols = np.where(X_variance < 1e-10)[0]
    dropped_features = []
    
    if len(constant_cols) > 0:
        if drop_constant:
            # Drop constant columns and update feature names
            keep_cols = np.setdiff1d(np.arange(n_features), constant_cols)
            if len(keep_cols) == 0:
                raise ValueError("All columns have zero variance, cannot proceed")
            
            dropped_features = [feature_names[i] if feature_names else f"X{i+1}" for i in constant_cols]
            X = X[:, keep_cols]
            n_features = X.shape[1]
            
            if feature_names:
                feature_names = [feature_names[i] for i in keep_cols]
            
            # Update variable_groups if they exist
            if by_var and variable_groups:
                # Create mapping from old to new indices
                old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_cols)}
                
                new_variable_groups = {}
                for group_name, old_indices in variable_groups.items():
                    new_indices = []
                    for old_idx in old_indices:
                        if old_idx in old_to_new:
                            new_indices.append(old_to_new[old_idx])
                        else:
                            # This index was dropped
                            if verbose:
                                print(f"   - Dropped constant column {old_idx} from group '{group_name}'")
                    
                    if len(new_indices) > 0:
                        new_variable_groups[group_name] = new_indices
                    else:
                        warnings.warn(f"Group '{group_name}' had all columns dropped due to zero variance")
                
                variable_groups = new_variable_groups
            
            if verbose:
                print(f"   - Dropped {len(constant_cols)} constant columns: {dropped_features}")
                
        else:
            warnings.warn(f"Columns {constant_cols} have zero or near-zero variance and may cause numerical issues. "
                         f"Consider setting drop_constant=True to automatically remove them.")
    
    # Check for missing values
    if np.any(np.isnan(X)) or np.any(np.isnan(y_encoded)):
        raise ValueError("Missing values detected in X or y. Please handle missing data before running the test.")
    
    # Set feature names if not provided
    if feature_names is None:
        feature_names = [f"X{i+1}" for i in range(n_features)]
    
    # Setup variable groups for by_var functionality
    if by_var and variable_groups is None:
        # Default: treat each column as separate variable
        variable_groups = {name: [i] for i, name in enumerate(feature_names)}
        if verbose:
            print("   - by_var=True but no variable_groups provided, treating each column separately")
    
    # Get number of outcome categories
    J = int(np.max(y_encoded))
    K = n_features  # Number of predictors (excluding intercept)
    
    # CRITICAL: Brant test requires at least 3 categories
    if J < 3:
        raise ValueError(f"Brant test requires at least 3 ordered categories, but found {J} categories. "
                        f"The test is not meaningful with fewer than 3 categories.")
    
    # Check for large matrices and warn about memory usage
    matrix_size = (J-1) * K
    if matrix_size > 2000:
        memory_mb = (matrix_size ** 2) * 8 / (1024**2)  # 8 bytes per float64
        warnings.warn(f"Large parameter matrix ({matrix_size}×{matrix_size}) will require ~{memory_mb:.1f}MB memory. "
                     f"Consider reducing dimensionality or using by_var grouping.")
    
    if verbose:
        print(f"   - Number of observations: {n_obs}")
        print(f"   - Number of predictors: {K}")
        print(f"   - Number of outcome categories: {J}")
        if by_var and variable_groups:
            print(f"   - Variable groups: {len(variable_groups)} groups")
            for group_name, indices in variable_groups.items():
                print(f"     - {group_name}: columns {indices}")
        if len(dropped_features) > 0:
            print(f"   - Dropped constant columns: {len(dropped_features)}")
        if matrix_size > 1000:
            print(f"   - Parameter matrix size: {matrix_size}×{matrix_size}")
    
    # Step 2: Create binary outcomes for each threshold
    if verbose:
        print("Step 2: Creating binary outcomes for each threshold...")
    
    # Create binary indicators: z_m = 1 if y > m, 0 otherwise
    binary_outcomes = {}
    for m in range(1, J):
        binary_outcomes[f"z{m}"] = (y_encoded > m).astype(int)
        n_above = np.sum(binary_outcomes[f"z{m}"])
        n_below = n_obs - n_above
        if verbose:
            print(f"   - Threshold {m}: {n_above} above, {n_below} below (balance: {n_above/n_obs:.2f})")
        
        # Check for separation issues
        if n_above < 10 or n_below < 10:
            warnings.warn(f"Threshold {m} has very few observations in one category "
                         f"({n_above} above, {n_below} below), which may cause convergence issues")
    
    # Step 3: Fit binary logistic regression models
    if verbose:
        print("Step 3: Fitting binary logistic regression models...")
    
    binary_models = {}
    beta_hat = np.zeros((J-1, K+1))  # +1 for intercept
    var_hat = []
    fitted_values = np.zeros((n_obs, J-1))
    
    convergence_warnings = []
    separation_warnings = []
    per_threshold_diagnostics = {}
    
    for m in range(1, J):
        if verbose:
            print(f"   - Fitting binary model for threshold {m}...")
        
        # Add intercept to design matrix (statsmodels convention: intercept first)
        X_wi = sm.add_constant(X, has_constant='add')
        
        # Fit logistic regression using statsmodels for MLE estimation
        model_m = sm.Logit(binary_outcomes[f"z{m}"], X_wi)
        try:
            result_m = model_m.fit(disp=False, maxiter=100)
            
            # Check convergence
            if not result_m.mle_retvals['converged']:
                convergence_warnings.append(f"Threshold {m}: convergence not achieved")
                if verbose:
                    print(f"     - WARNING: Convergence not achieved for threshold {m}")
            
        except PerfectSeparationError as e:
            # Try to identify which variables might be causing separation
            try:
                # Get predictions from last successful iteration (if any)
                pred_probs = model_m.predict(X_wi, params=model_m.start_params)
                extreme_prob_idx = (pred_probs < 0.001) | (pred_probs > 0.999)
                if np.any(extreme_prob_idx):
                    # Look at feature values for extreme predictions
                    extreme_features = np.where(np.abs(X[extreme_prob_idx].mean(axis=0)) > 2)[0]
                    suspect_vars = [feature_names[i] if feature_names else f"X{i+1}" for i in extreme_features]
                    suspect_info = f" Potential separating variables: {suspect_vars[:3]}" if len(suspect_vars) > 0 else ""
                else:
                    suspect_info = ""
            except:
                suspect_info = ""
            
            msg = f"Threshold {m}: perfect (or quasi) separation detected. " \
                  f"Consider collapsing sparse categories or using penalized logit.{suspect_info}"
            convergence_warnings.append(msg)
            if verbose:
                print("     - " + msg)
            raise RuntimeError(msg)
            
        except Exception as e:
            convergence_warnings.append(f"Threshold {m}: {str(e)[:50]}...")
            if verbose:
                print(f"     - Convergence issues for threshold {m}: {e}")
            # Try with more iterations
            try:
                result_m = model_m.fit(disp=False, maxiter=200, start_params=None)
            except PerfectSeparationError as e2:
                msg = f"Threshold {m}: perfect separation detected after retry. " \
                      f"Consider collapsing categories or using penalized methods."
                raise RuntimeError(msg)
            except Exception as e2:
                raise RuntimeError(f"Failed to fit binary model for threshold {m}: {e2}")
        
        # Store model result
        binary_models[f"model{m}"] = result_m
        
        # Store MLE coefficients (intercept first in statsmodels)
        beta_hat[m-1, 0] = result_m.params[0]    # intercept
        beta_hat[m-1, 1:] = result_m.params[1:]  # feature coefficients
        
        # Store fitted probabilities
        fitted_probs = result_m.predict(X_wi)
        fitted_values[:, m-1] = fitted_probs
        
        # Check for near-separation (fitted probabilities near 0 or 1)
        min_prob = np.min(fitted_probs)
        max_prob = np.max(fitted_probs)
        if min_prob < 0.01 or max_prob > 0.99:
            separation_warnings.append(f"Threshold {m}: fitted probabilities range [{min_prob:.4f}, {max_prob:.4f}]")
            if verbose:
                print(f"     - WARNING: Near-separation detected, prob range [{min_prob:.4f}, {max_prob:.4f}]")
        
        # Use MLE covariance matrix directly from statsmodels (handle type compatibility)
        cov_matrix = result_m.cov_params()
        if hasattr(cov_matrix, 'values'):  # DataFrame
            cov_matrix = cov_matrix.values
        var_hat.append(cov_matrix)
        
        # Store detailed diagnostics for this threshold
        per_threshold_diagnostics[f"threshold_{m}"] = {
            'converged': result_m.mle_retvals.get('converged', False),
            'log_likelihood': result_m.llf,
            'aic': result_m.aic,
            'min_fitted_prob': min_prob,
            'max_fitted_prob': max_prob,
            'n_iterations': result_m.mle_retvals.get('iterations', 'unknown'),
            'n_above_threshold': np.sum(binary_outcomes[f"z{m}"]),
            'n_below_threshold': n_obs - np.sum(binary_outcomes[f"z{m}"])
        }
        
        if verbose:
            print(f"     - Converged: {result_m.mle_retvals.get('converged', 'unknown')}")
            print(f"     - Log-likelihood: {result_m.llf:.2f}")
            print(f"     - AIC: {result_m.aic:.2f}")
            print(f"     - Iterations: {result_m.mle_retvals.get('iterations', 'unknown')}")
            print(f"     - Prob range: [{min_prob:.4f}, {max_prob:.4f}]")
    
    # Step 4: Calculate variance-covariance matrix between thresholds
    if verbose:
        print("Step 4: Calculating cross-threshold variance-covariance matrices...")
    
    # Initialize full variance-covariance matrix
    var_beta = np.zeros(((J-1)*K, (J-1)*K))
    
    # Calculate covariances between different thresholds using fast vectorized approach
    X_with_intercept = np.column_stack([np.ones(n_obs), X])
    
    cross_cov_issues = 0
    for m in range(J-2):
        for l in range(m+1, J-1):
            # Get fitted probabilities for thresholds m and l
            pi_m = fitted_values[:, m]
            pi_l = fitted_values[:, l]
            
            # Use fast covariance calculation (avoids diagonal matrix construction)
            try:
                cov_ml = _calculate_cross_threshold_covariance_fast(
                    X_with_intercept, pi_m, pi_l)
                
                # Store in variance matrix (excluding intercept terms)
                start_m, end_m = m*K, (m+1)*K
                start_l, end_l = l*K, (l+1)*K
                
                var_beta[start_m:end_m, start_l:end_l] = cov_ml
                var_beta[start_l:end_l, start_m:end_m] = cov_ml  # Symmetric
                
                if verbose:
                    cov_norm = np.linalg.norm(cov_ml)
                    print(f"     - Cross-covariance thresholds {m+1}-{l+1}: ||cov|| = {cov_norm:.4f}")
                
            except Exception as e:
                cross_cov_issues += 1
                warnings.warn(f"Numerical issues calculating covariance between thresholds {m+1} and {l+1}: {e}")
                if verbose:
                    print(f"     - Failed to compute cross-covariance for thresholds {m+1}-{l+1}")
    
    # Fill diagonal blocks with individual model variances (exclude intercept)
    for m in range(J-1):
        start_idx = m * K
        end_idx = (m + 1) * K
        # Handle type compatibility for covariance matrix
        cov_m = var_hat[m]
        if hasattr(cov_m, 'values'):  # DataFrame case
            cov_m = cov_m.values
        # Extract covariance matrix excluding intercept (first row/column)
        var_beta[start_idx:end_idx, start_idx:end_idx] = cov_m[1:, 1:]
        
        if verbose:
            diag_norm = np.linalg.norm(cov_m[1:, 1:])
            print(f"     - Diagonal block {m+1}: ||cov|| = {diag_norm:.4f}")
    
    if verbose and cross_cov_issues > 0:
        print(f"   - WARNING: {cross_cov_issues} cross-covariance computations failed")
    
    # Step 5: Construct test statistics
    if verbose:
        print("Step 5: Computing test statistics...")
    
    # Stack coefficients (excluding intercepts)
    beta_star = np.concatenate([beta_hat[m, 1:] for m in range(J-1)])
    
    # Create constraint matrix D for testing equality of coefficients
    D = _create_constraint_matrix(J, K)
    
    # CRITICAL DIMENSION CHECK: Verify consistency
    expected_beta_length = (J-1) * K
    expected_D_beta_length = (J-2) * K
    
    assert len(beta_star) == expected_beta_length, \
        f"beta_star length {len(beta_star)} != expected {expected_beta_length}"
    assert D.shape == ((J-2)*K, (J-1)*K), \
        f"D shape {D.shape} != expected {((J-2)*K, (J-1)*K)}"
    assert var_beta.shape == ((J-1)*K, (J-1)*K), \
        f"var_beta shape {var_beta.shape} != expected {((J-1)*K, (J-1)*K)}"
    
    D_beta = D @ beta_star
    assert len(D_beta) == expected_D_beta_length, \
        f"D @ beta_star length {len(D_beta)} != expected {expected_D_beta_length}"
    
    if verbose:
        print(f"   - Dimension checks passed: beta_star({len(beta_star)}), D{D.shape}, D_beta({len(D_beta)})")
    
    # Calculate omnibus test statistic using numerically stable methods
    try:
        var_D_beta = D @ var_beta @ D.T
        
        # CRITICAL: Ensure numerical symmetry before Cholesky decomposition
        var_D_beta = 0.5 * (var_D_beta + var_D_beta.T)
        
        # Use Cholesky decomposition for numerical stability
        X2_omnibus = _stable_quadratic_form(D_beta, var_D_beta, verbose=verbose)
        df_omnibus = (J-2) * K
        p_omnibus = 1 - stats.chi2.cdf(X2_omnibus, df_omnibus)
        
        if verbose:
            print(f"   - Omnibus test statistic: {X2_omnibus:.4f}")
            print(f"   - Degrees of freedom: {df_omnibus}")
            print(f"   - P-value: {p_omnibus:.4f}")
        
    except Exception as e:
        warnings.warn(f"Numerical issues computing omnibus test statistic: {e}")
        X2_omnibus = np.nan
        df_omnibus = (J-2) * K
        p_omnibus = np.nan
    
    # Step 6: Individual/grouped variable tests
    if verbose:
        print("Step 6: Computing individual/grouped variable tests...")
    
    X2_individual = []
    df_individual = []
    p_individual = []
    test_names = []
    
    if by_var and variable_groups:
        # Implement proper by_var functionality with variable grouping
        if verbose:
            print("   - Using grouped variable testing (by_var=True)")
        
        for group_name, col_indices in variable_groups.items():
            if verbose:
                print(f"   - Testing variable group '{group_name}' (columns {col_indices})...")
            
            # Create constraint matrix for this group  
            D_group = _create_group_constraint_matrix(J, K, col_indices)
            
            # Sanity check: D_group should have (J-2) * len(col_indices) rows
            expected_rows = (J-2) * len(col_indices)
            assert D_group.shape[0] == expected_rows, \
                f"D_group has wrong shape: {D_group.shape[0]} != {expected_rows}"
            
            try:
                D_group_beta = D_group @ beta_star
                var_D_group_beta = D_group @ var_beta @ D_group.T
                
                # Ensure numerical symmetry
                var_D_group_beta = 0.5 * (var_D_group_beta + var_D_group_beta.T)
                
                # Use numerically stable quadratic form computation
                X2_group = _stable_quadratic_form(D_group_beta, var_D_group_beta, verbose=False)
                df_group = (J-2) * len(col_indices)
                p_group = 1 - stats.chi2.cdf(X2_group, df_group)
                
                X2_individual.append(X2_group)
                df_individual.append(df_group)
                p_individual.append(p_group)
                test_names.append(group_name)
                
                if verbose:
                    print(f"     - Group test statistic: {X2_group:.4f}, df: {df_group}, p-value: {p_group:.4f}")
                
            except Exception as e:
                warnings.warn(f"Numerical issues computing test for variable group '{group_name}': {e}")
                X2_individual.append(np.nan)
                df_individual.append((J-2) * len(col_indices))
                p_individual.append(np.nan)
                test_names.append(group_name)
    
    else:
        # Test each predictor individually (original behavior)
        if verbose:
            print("   - Using individual variable testing")
            
        for k in range(K):
            var_name = feature_names[k]
            if verbose:
                print(f"   - Testing variable {var_name}...")
            
            # Create constraint matrix for this variable
            D_k = _create_variable_constraint_matrix(J, K, k)
            
            try:
                D_k_beta = D_k @ beta_star
                var_D_k_beta = D_k @ var_beta @ D_k.T
                
                # Ensure numerical symmetry
                var_D_k_beta = 0.5 * (var_D_k_beta + var_D_k_beta.T)
                
                # Use numerically stable quadratic form computation
                X2_k = _stable_quadratic_form(D_k_beta, var_D_k_beta, verbose=False)
                df_k = J - 2
                p_k = 1 - stats.chi2.cdf(X2_k, df_k)
                
                X2_individual.append(X2_k)
                df_individual.append(df_k)
                p_individual.append(p_k)
                test_names.append(var_name)
                
                if verbose:
                    print(f"     - Test statistic: {X2_k:.4f}, p-value: {p_k:.4f}")
                
            except Exception as e:
                warnings.warn(f"Numerical issues computing test for variable {var_name}: {e}")
                X2_individual.append(np.nan)
                df_individual.append(J-2)
                p_individual.append(np.nan)
                test_names.append(var_name)
    
    # Step 7: Format and return results with diagnostics
    results = _format_test_results(
        X2_omnibus, df_omnibus, p_omnibus,
        X2_individual, df_individual, p_individual,
        test_names, alpha=alpha  # Use configurable significance threshold
    )
    
    # Add comprehensive diagnostic information
    results['diagnostics'] = {
        'convergence_warnings': convergence_warnings,
        'separation_warnings': separation_warnings,
        'cross_covariance_issues': cross_cov_issues,
        'per_threshold': per_threshold_diagnostics,
        'n_observations': n_obs,
        'n_categories': J,
        'n_predictors': K,
        'constant_columns': constant_cols.tolist() if len(constant_cols) > 0 else [],
        'dropped_features': dropped_features,
        'by_var_used': by_var and variable_groups is not None,
        'variable_groups_used': variable_groups if by_var and variable_groups else None,
        'alpha_threshold': alpha
    }
    
    # Keep both test_names and original feature_names for clarity
    if by_var and variable_groups:
        results['feature_names'] = feature_names
        results['test_names'] = test_names
        results['variable_groups'] = variable_groups
        
    # Add internal matrices for debugging/verification if requested
    if return_internals:
        results['internals'] = {
            'constraint_matrix_D': D,
            'variance_covariance_matrix': var_beta,
            'stacked_coefficients': beta_star,
            'binary_models': binary_models,
            'fitted_values': fitted_values,
            'coefficient_matrix': beta_hat
        }
    
    if verbose:
        print("\nBrant Test Results:")
        print("=" * 60)
        _print_test_results(results)
        
        # Print diagnostic summary
        if len(convergence_warnings) > 0 or len(separation_warnings) > 0 or cross_cov_issues > 0:
            print(f"\nDiagnostic Summary:")
            print(f"- Convergence issues: {len(convergence_warnings)}")
            print(f"- Separation warnings: {len(separation_warnings)}")  
            print(f"- Cross-covariance computation failures: {cross_cov_issues}")
            if len(convergence_warnings) > 0:
                print("  Convergence details:", convergence_warnings[:3])  # Show first 3
            if len(separation_warnings) > 0:
                print("  Separation details:", separation_warnings[:3])  # Show first 3
    
    return results


def _stable_quadratic_form(v: np.ndarray, M: np.ndarray, verbose: bool = False) -> float:
    """
    Compute v^T M^{-1} v using numerically stable methods.
    
    First tries Cholesky decomposition, then falls back to solve, then pinv.
    
    Parameters:
    -----------
    v : np.ndarray
        Vector (n,)
    M : np.ndarray
        Positive definite matrix (n x n)
    verbose : bool
        If True, print method used
        
    Returns:
    --------
    float : Quadratic form result
    """
    try:
        # Try Cholesky decomposition (fastest and most stable for PD matrices)
        L, lower = cho_factor(M, check_finite=False)
        M_inv_v = cho_solve((L, lower), v, check_finite=False)
        result = v.T @ M_inv_v
        if verbose:
            print(f"     - Used Cholesky decomposition for quadratic form")
        return result
        
    except (CholeskyError, np.linalg.LinAlgError):
        if verbose:
            print(f"     - Cholesky failed, trying standard solve")
        
        try:
            # Fall back to standard solve
            M_inv_v = np.linalg.solve(M, v)
            result = v.T @ M_inv_v
            return result
            
        except np.linalg.LinAlgError:
            if verbose:
                print(f"     - Standard solve failed, using pseudo-inverse")
            
            # Last resort: pseudo-inverse
            M_inv_v = np.linalg.pinv(M) @ v
            result = v.T @ M_inv_v
            return result


def _create_group_constraint_matrix(J: int, K: int, col_indices: List[int]) -> np.ndarray:
    """
    Create constraint matrix for testing a group of variables jointly across thresholds.
    
    This implements the by_var functionality by creating constraints that test
    whether coefficients for the specified columns are equal across thresholds.
    
    Parameters:
    -----------
    J : int
        Number of outcome categories
    K : int
        Total number of predictors  
    col_indices : List[int]
        Column indices to include in this group
        
    Returns:
    --------
    np.ndarray : Constraint matrix for group testing
    """
    n_group_vars = len(col_indices)
    n_constraints = (J-2) * n_group_vars
    n_params = (J-1) * K
    
    D_group = np.zeros((n_constraints, n_params))
    
    constraint_idx = 0
    for i in range(J-2):  # For each pair of adjacent thresholds
        for col_idx in col_indices:  # For each variable in the group
            # Coefficient for this variable at threshold i+1
            param_idx_1 = i * K + col_idx
            # Coefficient for this variable at threshold i+2
            param_idx_2 = (i + 1) * K + col_idx
            
            # Constraint: beta_{i+1,col_idx} - beta_{i+2,col_idx} = 0
            D_group[constraint_idx, param_idx_1] = 1
            D_group[constraint_idx, param_idx_2] = -1
            
            constraint_idx += 1
    
    return D_group


def _xtwx(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Fast computation of X^T W X where W is diagonal with weights w.
    
    Parameters:
    -----------
    X : np.ndarray
        Design matrix (n x p)
    w : np.ndarray
        Weight vector (n,)
        
    Returns:
    --------
    np.ndarray : X^T W X matrix (p x p)
    """
    # Vectorized: X^T W X = X^T (w * X) where w broadcasts over columns
    return X.T @ (X * w[:, None])


def _calculate_cross_threshold_covariance_fast(X_wi: np.ndarray, pi_m: np.ndarray, 
                                             pi_l: np.ndarray) -> np.ndarray:
    """
    Fast and numerically stable calculation of cross-threshold covariance.
    
    This avoids constructing large diagonal matrices and uses vectorized operations.
    
    Parameters:
    -----------
    X_wi : np.ndarray
        Design matrix with intercept (n x p+1)
    pi_m, pi_l : np.ndarray
        Fitted probabilities for thresholds m and l (n,)
        
    Returns:
    --------
    np.ndarray : Cross-threshold covariance matrix excluding intercept
    """
    # Calculate weight vectors (avoid diagonal matrix construction)
    w_m = pi_m * (1 - pi_m)
    w_l = pi_l * (1 - pi_l) 
    w_ml = pi_l - pi_m * pi_l
    
    # Fast weighted cross-products
    XtWmX = _xtwx(X_wi, w_m)
    XtWlX = _xtwx(X_wi, w_l)
    XtWmlX = _xtwx(X_wi, w_ml)
    
    # Numerically stable computation: inv(X'WmX) @ (X'WmlX) @ inv(X'WlX)
    # Use Cholesky decomposition when possible for better numerical stability
    
    try:
        # Try Cholesky for XtWmX
        L_m, lower_m = cho_factor(XtWmX, check_finite=False)
        A = cho_solve((L_m, lower_m), XtWmlX, check_finite=False)
    except (CholeskyError, np.linalg.LinAlgError):
        try:
            # Fall back to standard solve
            A = np.linalg.solve(XtWmX, XtWmlX)
        except np.linalg.LinAlgError:
            warnings.warn("Numerical issues with XtWmX, using pseudo-inverse")
            A = np.linalg.pinv(XtWmX) @ XtWmlX
    
    try:
        # Try Cholesky for XtWlX  (consistent with XtWmX approach)
        L_l, lower_l = cho_factor(XtWlX, check_finite=False)
        B = cho_solve((L_l, lower_l), np.eye(XtWlX.shape[0]), check_finite=False)
    except (CholeskyError, np.linalg.LinAlgError):
        try:
            # Fall back to standard solve for identity matrix (equivalent to inverse)
            B = np.linalg.solve(XtWlX, np.eye(XtWlX.shape[0]))
        except np.linalg.LinAlgError:
            warnings.warn("Numerical issues with XtWlX, using pseudo-inverse")
            B = np.linalg.pinv(XtWlX)
    
    # Final covariance matrix
    cov_full = A @ B
    
    # Return excluding intercept (first row and column)
    return cov_full[1:, 1:]


def _encode_ordinal_outcome(y: np.ndarray, categories_order: List = None) -> np.ndarray:
    """
    Encode ordinal outcome to ensure it starts from 1 and follows correct ordering.
    
    Parameters:
    -----------
    y : np.ndarray
        Original ordinal outcome
    categories_order : List, optional
        Explicit ordering for categories. If provided, categories will be mapped
        according to this order. Useful for string categories that need specific
        ordering (e.g., ['low', 'medium', 'high'] rather than alphabetical).
        
    Returns:
    --------
    np.ndarray : Encoded outcome starting from 1 with proper ordering
    """
    if categories_order is not None:
        # Use explicit category ordering
        unique_y = np.unique(y)
        
        # Check that all categories in y are in categories_order
        missing_cats = set(unique_y) - set(categories_order)
        if missing_cats:
            raise ValueError(f"Categories {missing_cats} in y are not in categories_order")
        
        # Check that all categories_order are in y  
        extra_cats = set(categories_order) - set(unique_y)
        if extra_cats:
            warnings.warn(f"Categories {extra_cats} in categories_order are not present in y")
        
        # Create mapping from category to integer
        category_to_int = {cat: i+1 for i, cat in enumerate(categories_order)}
        
        # Apply mapping
        encoded = np.array([category_to_int[val] for val in y])
        
    else:
        # Use natural ordering (good for numeric, alphabetical for strings)
        unique_vals = np.unique(y)
        
        if len(unique_vals) < 3:
            warnings.warn("Ordinal outcome has fewer than 3 categories, results may not be meaningful")
        
        # For non-numeric categories, try to convert or require explicit ordering
        if not np.issubdtype(y.dtype, np.number):
            if isinstance(y[0], str):
                # Try to convert numeric strings to numbers
                try:
                    numeric_vals = [float(val) for val in unique_vals]
                    # Check if they're all integers
                    if all(val.is_integer() for val in numeric_vals):
                        numeric_vals = [int(val) for val in numeric_vals]
                    
                    # Create mapping from string to numeric
                    str_to_num = {str_val: num_val for str_val, num_val in zip(unique_vals, numeric_vals)}
                    y_numeric = np.array([str_to_num[str_val] for str_val in y])
                    
                    warnings.warn(f"Converted string categories {list(unique_vals)} to numeric. "
                                f"If this ordering is incorrect, please provide categories_order parameter.")
                    
                    # Proceed with numeric encoding
                    encoder = LabelEncoder()
                    encoded = encoder.fit_transform(y_numeric) + 1  # Start from 1
                    
                except (ValueError, TypeError):
                    # Not numeric strings, require explicit ordering
                    raise ValueError(
                        "Non-numeric ordinal categories detected. You must provide 'categories_order' "
                        "parameter to explicitly specify the correct ordinal ordering. "
                        f"Detected categories: {list(unique_vals)}. "
                        "Example: categories_order=['low', 'medium', 'high'] or convert to numeric first."
                    )
            else:
                # Non-string, non-numeric - require explicit ordering
                raise ValueError(
                    "Non-numeric ordinal categories detected. You must provide 'categories_order' "
                    "parameter to explicitly specify the correct ordinal ordering. "
                    f"Detected categories: {list(unique_vals)}"
                )
        else:
            # For numeric categories, proceed with natural ordering
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(y) + 1  # Start from 1
    
    return encoded


def _validate_variable_groups(variable_groups: Dict[str, List[int]], 
                            n_features: int, feature_names: List[str] = None) -> None:
    """
    Validate variable_groups parameter for robustness.
    
    Checks that all indices are within bounds, no duplicates exist,
    and indices are properly formatted.
    
    Parameters:
    -----------
    variable_groups : Dict[str, List[int]]
        Mapping of group names to column indices
    n_features : int
        Total number of features in X
    feature_names : List[str], optional
        Names of features for better error messages
        
    Raises:
    -------
    ValueError : If validation fails
    """
    if not isinstance(variable_groups, dict):
        raise ValueError("variable_groups must be a dictionary")
    
    if len(variable_groups) == 0:
        raise ValueError("variable_groups cannot be empty when by_var=True")
    
    all_indices = []
    
    for group_name, indices in variable_groups.items():
        if not isinstance(indices, (list, tuple, np.ndarray)):
            raise ValueError(f"Indices for group '{group_name}' must be a list, tuple, or array")
        
        if len(indices) == 0:
            raise ValueError(f"Group '{group_name}' cannot have empty indices")
        
        # Convert to list and check for proper integer types
        indices_list = list(indices)
        for i, idx in enumerate(indices_list):
            if not isinstance(idx, (int, np.integer)) or idx != int(idx):
                raise ValueError(f"All indices must be integers. Found {type(idx).__name__} in group '{group_name}'")
            indices_list[i] = int(idx)
        
        # Check bounds
        for idx in indices_list:
            if idx < 0 or idx >= n_features:
                feat_info = f" (feature: {feature_names[idx]})" if feature_names and 0 <= idx < len(feature_names) else ""
                raise ValueError(f"Index {idx}{feat_info} in group '{group_name}' is out of bounds [0, {n_features-1}]")
        
        # Check for duplicates within group
        if len(set(indices_list)) != len(indices_list):
            duplicates = [idx for idx in set(indices_list) if indices_list.count(idx) > 1]
            raise ValueError(f"Group '{group_name}' contains duplicate indices: {duplicates}")
        
        # Sort indices within group for consistency
        indices_list.sort()
        variable_groups[group_name] = indices_list
        
        all_indices.extend(indices_list)
    
    # Check for overlapping indices across groups
    if len(set(all_indices)) != len(all_indices):
        duplicates = [idx for idx in set(all_indices) if all_indices.count(idx) > 1]
        # Find which groups have overlapping indices
        overlapping_groups = []
        for idx in duplicates:
            groups_with_idx = [name for name, indices in variable_groups.items() if idx in indices]
            overlapping_groups.extend([(idx, groups_with_idx)])
        
        raise ValueError(f"Indices cannot appear in multiple groups. "
                        f"Overlapping indices: {dict(overlapping_groups)}")
    
    # Warn if not all features are covered
    uncovered_indices = set(range(n_features)) - set(all_indices)
    if uncovered_indices:
        uncovered_names = [feature_names[i] if feature_names else f"index_{i}" 
                          for i in sorted(uncovered_indices)]
        warnings.warn(f"Some features are not covered by any group: {uncovered_names}")


# Note: Old Fisher information and diagonal matrix-based functions removed
# Now using statsmodels MLE covariance matrices and vectorized operations


def _create_constraint_matrix(J: int, K: int) -> np.ndarray:
    """
    Create constraint matrix D for testing equality of coefficients across thresholds.
    
    Parameters:
    -----------
    J : int
        Number of outcome categories
    K : int
        Number of predictors
        
    Returns:
    --------
    np.ndarray : Constraint matrix D
    """
    # Create constraint matrix to test if coefficients are equal across thresholds
    # Each row tests if coefficient for threshold i equals coefficient for threshold i+1
    
    n_constraints = (J - 2) * K
    n_params = (J - 1) * K
    
    D = np.zeros((n_constraints, n_params))
    
    constraint_idx = 0
    for i in range(J - 2):  # For each pair of adjacent thresholds
        for k in range(K):  # For each variable
            # Coefficient for threshold i+1
            param_idx_1 = i * K + k
            # Coefficient for threshold i+2  
            param_idx_2 = (i + 1) * K + k
            
            # Constraint: beta_{i+1,k} - beta_{i+2,k} = 0
            D[constraint_idx, param_idx_1] = 1
            D[constraint_idx, param_idx_2] = -1
            
            constraint_idx += 1
    
    return D


def _create_variable_constraint_matrix(J: int, K: int, var_idx: int) -> np.ndarray:
    """
    Create constraint matrix for testing a single variable across thresholds.
    
    Parameters:
    -----------
    J : int
        Number of outcome categories
    K : int  
        Number of predictors
    var_idx : int
        Index of the variable to test
        
    Returns:
    --------
    np.ndarray : Constraint matrix for variable var_idx
    """
    n_constraints = J - 2
    n_params = (J - 1) * K
    
    D_k = np.zeros((n_constraints, n_params))
    
    for i in range(J - 2):
        # Coefficient for variable var_idx at threshold i+1
        param_idx_1 = i * K + var_idx
        # Coefficient for variable var_idx at threshold i+2
        param_idx_2 = (i + 1) * K + var_idx
        
        # Constraint: beta_{i+1,var_idx} - beta_{i+2,var_idx} = 0
        D_k[i, param_idx_1] = 1
        D_k[i, param_idx_2] = -1
    
    return D_k


def _format_test_results(X2_omnibus: float, df_omnibus: int, p_omnibus: float,
                        X2_individual: List[float], df_individual: List[int], 
                        p_individual: List[float], feature_names: List[str], 
                        alpha: float = 0.05) -> Dict:
    """
    Format test results into a structured dictionary with clear hypothesis information.
    
    Parameters:
    -----------
    X2_omnibus : float
        Omnibus test statistic
    df_omnibus : int
        Degrees of freedom for omnibus test
    p_omnibus : float
        P-value for omnibus test
    X2_individual : List[float]
        Individual test statistics
    df_individual : List[int]
        Degrees of freedom for individual tests
    p_individual : List[float]
        P-values for individual tests
    feature_names : List[str]
        Names of predictor variables
        
    Returns:
    --------
    Dict : Formatted test results with hypothesis information
    """
    results = {
        'test_description': 'Brant Test for Proportional Odds Assumption',
        'null_hypothesis': 'H0: Parallel regression assumption holds (coefficients equal across thresholds)',
        'alternative_hypothesis': 'H1: Parallel regression assumption violated (coefficients differ across thresholds)',
        'interpretation': f'If p-value < {alpha}, reject H0 (proportional odds assumption violated)',
        'omnibus': {
            'description': 'Test all variables simultaneously',
            'test_statistic': X2_omnibus,
            'df': df_omnibus,
            'p_value': p_omnibus,
            'significant': p_omnibus < alpha if not np.isnan(p_omnibus) else None
        },
        'individual_tests': {}
    }
    
    for i, name in enumerate(feature_names):
        results['individual_tests'][name] = {
            'description': f'Test variable {name} individually',
            'test_statistic': X2_individual[i],
            'df': df_individual[i],
            'p_value': p_individual[i],
            'significant': p_individual[i] < alpha if not np.isnan(p_individual[i]) else None
        }
    
    return results


def _print_test_results(results: Dict) -> None:
    """
    Print formatted test results to console with enhanced readability.
    
    Parameters:
    -----------
    results : Dict
        Test results dictionary
    """
    print(f"\n{results['test_description']}")
    print("=" * 60)
    print(f"{results['null_hypothesis']}")
    print(f"{results['interpretation']}")
    print("=" * 60)
    
    # Table header
    print(f"{'Test':<20} {'X²':<10} {'df':<5} {'p-value':<12} {'Significant':<12}")
    print("-" * 60)
    
    # Omnibus test
    omnibus = results['omnibus']
    sig_marker = "***" if omnibus.get('significant') else ""
    print(f"{'Omnibus':<20} {omnibus['test_statistic']:<10.3f} {omnibus['df']:<5} "
          f"{omnibus['p_value']:<12.4f} {sig_marker:<12}")
    
    # Individual tests
    for var_name, test_result in results['individual_tests'].items():
        sig_marker = "***" if test_result.get('significant') else ""
        print(f"{var_name:<20} {test_result['test_statistic']:<10.3f} {test_result['df']:<5} "
              f"{test_result['p_value']:<12.4f} {sig_marker:<12}")
    
    print("-" * 60)
    alpha_threshold = results.get('diagnostics', {}).get('alpha_threshold', 0.05)
    print(f"*** = Significant at p < {alpha_threshold} (proportional odds assumption violated)")
    print("If omnibus test is significant, check individual tests to identify")
    print("which variables violate the parallel regression assumption.")


# Example usage and testing function
def test_brant_test():
    """
    Test function to demonstrate usage of the improved Brant test.
    
    This creates simulated ordinal data with known violation of proportional odds
    and demonstrates both individual and grouped variable testing.
    """
    np.random.seed(42)
    
    print("=" * 80)
    print("TESTING IMPROVED BRANT TEST IMPLEMENTATION")
    print("=" * 80)
    
    # Create sample ordinal data  
    n = 1000
    
    # Create predictors with some grouped structure
    # Education variables (3 dummy variables)
    education_base = np.random.randn(n)
    X_education = np.column_stack([
        (education_base > 0.5).astype(int),    # High school
        (education_base > 1.0).astype(int),    # College  
        (education_base > 1.5).astype(int)     # Graduate
    ])
    
    # Region variables (2 dummy variables)
    region_base = np.random.randn(n)
    X_region = np.column_stack([
        (region_base > 0).astype(int),         # Urban
        (region_base > 1).astype(int)          # Metro
    ])
    
    # Continuous variable
    X_income = np.random.randn(n, 1)
    
    # Combine all predictors
    X = np.column_stack([X_education, X_region, X_income])
    feature_names = ['HS', 'College', 'Graduate', 'Urban', 'Metro', 'Income']
    
    # Define variable groups for by_var testing
    variable_groups = {
        'Education': [0, 1, 2],    # Columns 0,1,2 are education dummies
        'Region': [3, 4],          # Columns 3,4 are region dummies  
        'Income': [5]              # Column 5 is income
    }
    
    # Create ordinal outcome with violation in education group
    # Education has different effects across thresholds (violates proportional odds)
    # Region and Income have consistent effects (satisfy assumption)
    
    education_effect_1 = 2.0 * X_education[:, 0] + 1.5 * X_education[:, 1] + 1.0 * X_education[:, 2]
    education_effect_2 = 0.5 * X_education[:, 0] + 0.3 * X_education[:, 1] + 0.1 * X_education[:, 2]
    region_effect = 0.5 * X_region[:, 0] + 0.3 * X_region[:, 1]
    income_effect = 0.4 * X_income[:, 0]
    
    linear_combo_1 = education_effect_1 + region_effect + income_effect  # For threshold 1
    linear_combo_2 = education_effect_2 + region_effect + income_effect  # For threshold 2
    
    # Create thresholds with different education effects
    prob_1 = 1 / (1 + np.exp(-linear_combo_1))
    prob_2 = 1 / (1 + np.exp(-linear_combo_2))
    
    # Generate ordinal outcomes
    u = np.random.uniform(0, 1, n)
    y = np.ones(n, dtype=int)  # Start with category 1
    y[u > prob_1] = 2  # Move to category 2 
    y[u > prob_2] = 3  # Move to category 3
    
    print(f"Generated test data:")
    print(f"- Sample size: {n}")
    print(f"- Predictors: {len(feature_names)}")
    print(f"- Outcome categories: {len(np.unique(y))}")
    print(f"- Variable groups: {variable_groups}")
    print(f"- Expected violation: Education group should be significant")
    
    # Test 1: Individual variable testing
    print("\n" + "="*60)
    print("TEST 1: INDIVIDUAL VARIABLE TESTING")
    print("="*60)
    
    try:
        results_individual = brant_test(
            X=X, y=y, 
            feature_names=feature_names,
            by_var=False,
            verbose=True
        )
        
        print("\n✓ Individual testing completed successfully!")
        
    except Exception as e:
        print(f"Individual testing failed: {e}")
        import traceback
        traceback.print_exc()
        results_individual = None
    
    # Test 2: Grouped variable testing (by_var)
    print("\n" + "="*60) 
    print("TEST 2: GROUPED VARIABLE TESTING (by_var=True)")
    print("="*60)
    
    try:
        results_grouped = brant_test(
            X=X, y=y,
            feature_names=feature_names,
            by_var=True,
            variable_groups=variable_groups,
            verbose=True
        )
        
        print("\n✓ Grouped testing completed successfully!")
        
        # Validate results
        education_sig = results_grouped['individual_tests']['Education'].get('significant', False)
        region_sig = results_grouped['individual_tests']['Region'].get('significant', False)
        income_sig = results_grouped['individual_tests']['Income'].get('significant', False)
        
        print(f"\nValidation Results:")
        print(f"- Education group significant: {education_sig} (Expected: True)")
        print(f"- Region group significant: {region_sig} (Expected: False)")
        print(f"- Income significant: {income_sig} (Expected: False)")
        
        if education_sig and not region_sig:
            print("✓ Test correctly identified education group violation!")
        else:
            print("⚠ Results may differ due to randomness")
            
    except Exception as e:
        print(f"Grouped testing failed: {e}")
        import traceback
        traceback.print_exc()
        results_grouped = None
    
    # Test 3: String categories requiring explicit ordering 
    print("\n" + "="*60)
    print("TEST 3: STRING CATEGORIES VALIDATION")
    print("="*60)
    
    try:
        # Create string outcome categories
        y_string = np.array(['low', 'medium', 'high'])[np.digitize(y, bins=[1.5, 2.5]) - 1]
        
        print("Testing with string categories without categories_order (should fail):")
        try:
            brant_test(X=X[:100], y=y_string[:100], verbose=False)
            print("⚠ Expected error not raised!")
        except ValueError as e:
            print(f"✓ Correctly raised ValueError: {str(e)[:100]}...")
        
        print("\nTesting with string categories WITH categories_order (should work):")
        results_string = brant_test(
            X=X[:100], y=y_string[:100], 
            categories_order=['low', 'medium', 'high'],
            alpha=0.01,  # Test custom alpha
            verbose=True
        )
        print("✓ String categories with explicit ordering worked correctly!")
        
    except Exception as e:
        print(f"String category testing failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Numeric strings (should auto-convert)
    print("\n" + "="*60)
    print("TEST 4: NUMERIC STRING AUTO-CONVERSION")
    print("="*60)
    
    try:
        # Create numeric string categories
        y_numeric_str = np.array(['1', '2', '3'])[y-1]
        
        print("Testing numeric strings (should auto-convert with warning):")
        results_numeric_str = brant_test(
            X=X[:200], y=y_numeric_str[:200],
            drop_constant=True,  # Test drop_constant
            return_internals=True,  # Test internals
            verbose=True
        )
        print("✓ Numeric strings auto-converted successfully!")
        
        # Check internals
        if 'internals' in results_numeric_str:
            print(f"✓ Internals included: {list(results_numeric_str['internals'].keys())}")
        
        return results_grouped
        
    except Exception as e:
        print(f"Numeric string testing failed: {e}")
        import traceback
        traceback.print_exc()
        return results_grouped


if __name__ == "__main__":
    # Run test if script is executed directly
    print("Running Brant Test Implementation...")
    test_brant_test()
