"""
Panel Data Brant Test for Seniority Analysis

This script implements a robust Brant test for proportional odds assumption
using panel data with year as a covariate. Instead of testing each year separately,
this approach uses all years' data together and includes year as a control variable.

Author: Assistant
Date: 2024
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Add the parent directory to path to import brant_test
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from glm_plus.brant_test.brant_test import brant_test


def load_and_prepare_panel_data(file_path=None):
    """
    Load and prepare panel data for Brant test analysis.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to data file. If None, uses sample data structure shown by user.
        
    Returns:
    --------
    pd.DataFrame : Prepared data with all variables
    """
    print("Step 1: Loading panel data...")
    
    if file_path is None:
        # Create realistic sample data to avoid separation problems
        print("   - Using large sample data to avoid perfect separation issues")
        
        np.random.seed(42)  # For reproducibility
        n_workers = 500
        n_years = 10
        total_obs = n_workers * n_years
        
        # Create worker IDs and years
        data = []
        for worker_id in range(n_workers):
            for year in range(1, n_years + 1):
                data.append({
                    'worker_id': f'worker_{worker_id:04d}',
                    'country': 'us',
                    'year': year,
                    'rarity_score': np.random.beta(2, 5),  # Realistic distribution
                    'highest_educational_degree': np.random.choice(['Bachelor', 'Master', 'PhD'], p=[0.6, 0.3, 0.1]),
                    'whether_bachelor_university_prestigious': np.random.choice([True, False], p=[0.2, 0.8]),
                    'internationalization': np.random.choice(['Local', 'National', 'Multinational'], p=[0.4, 0.3, 0.3]),
                    'work_years': max(1, year + np.random.normal(5, 2)),  # Increases with career year
                    'simplified_company_size': np.random.choice(['Small (11-50 employees)', 'Medium (51-200 employees)', 'Large (500+ employees)'], p=[0.3, 0.4, 0.3]),
                    'gender': np.random.choice(['male', 'female'], p=[0.6, 0.4])
                })
        
        df = pd.DataFrame(data)
        
        # Create realistic seniority progression (balanced distribution)
        seniority_probs = []
        for _, row in df.iterrows():
            year = row['year']
            # Probability changes with career year
            if year <= 3:
                probs = [0.7, 0.25, 0.05, 0.0]  # Junior, Regular, Senior, Expert
            elif year <= 6:
                probs = [0.3, 0.5, 0.18, 0.02]
            else:
                probs = [0.1, 0.4, 0.4, 0.1]
            seniority_probs.append(probs)
        
        # Sample seniority levels
        seniority_levels = ['Junior', 'Regular', 'Senior', 'Expert']
        df['seniority'] = [np.random.choice(seniority_levels, p=probs) 
                          for probs in seniority_probs]
        
        print(f"   - Created sample data: {df.shape[0]} observations, {df.shape[1]} variables")
        print(f"   - Seniority distribution: {dict(df['seniority'].value_counts())}")
        
    else:
        # Load actual data file  
        try:
            print(f"   - Attempting to load: {file_path}")
            
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("File must be CSV or Excel format")
                
            print(f"   - Successfully loaded: {df.shape[0]} observations, {df.shape[1]} variables")
            print(f"   - Columns: {list(df.columns)}")
            
            # Check if this is the user's actual data structure with C1-C10 columns
            if 'C1' in df.columns and 'C10' in df.columns:
                print("   - Detected panel data with C1-C10 structure")
                print("   - This appears to be your career sequence data")
                
                # Need to understand what C1-C10 represent
                print("   - Please specify:")
                print("     1. What do C1-C10 columns represent? (seniority levels over 10 years?)")
                print("     2. What variable should be used as the ordinal outcome?")
                print("     3. Are there other relevant covariates in your dataset?")
                
                # Show sample of the data structure
                print(f"   - Sample of first few rows:")
                print(df.head(2).to_string())
                
                # For now, return the dataframe as-is for inspection
                return df
            
            # Check for expected variables
            expected_vars = ['year', 'seniority', 'worker_id']
            missing_vars = [var for var in expected_vars if var not in df.columns]
            
            if missing_vars:
                print(f"   - WARNING: Missing expected variables: {missing_vars}")
                print(f"   - Available columns: {list(df.columns)}")
                print("   - Please adapt the data preparation to match your data structure")
                
                # Try to provide helpful suggestions
                possible_mappings = {}
                if 'worker_id' not in df.columns:
                    id_cols = [col for col in df.columns if 'id' in col.lower() or 'worker' in col.lower()]
                    if id_cols:
                        possible_mappings['worker_id'] = id_cols
                
                if 'year' not in df.columns:
                    year_cols = [col for col in df.columns if 'year' in col.lower() or 'time' in col.lower()]
                    if year_cols:
                        possible_mappings['year'] = year_cols
                
                if possible_mappings:
                    print(f"   - Possible variable mappings: {possible_mappings}")
                
                # Return for manual inspection rather than failing
                return df
            
        except Exception as e:
            print(f"   - Error loading data: {e}")
            print("   - Using large sample data instead")
            return load_and_prepare_panel_data(file_path=None)
    
    # Print basic info
    if 'year' in df.columns:
        print(f"   - Year range: {df['year'].min()}-{df['year'].max()}")
    if 'worker_id' in df.columns:
        print(f"   - Unique workers: {df['worker_id'].nunique()}")
    print(f"   - Total observations: {len(df)}")
    
    return df


def create_year_variables(df, method='dummies'):
    """
    Create year variables as covariates.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Panel data
    method : str
        'dummies' for dummy variables, 'continuous' for linear trend
        
    Returns:
    --------
    np.ndarray : Year variables matrix
    list : Variable names
    """
    print("\nStep 2: Creating year variables...")
    
    if method == 'dummies':
        # Create dummy variables for years (exclude first year as reference)
        years = sorted(df['year'].unique())
        year_dummies = []
        year_names = []
        
        for year in years[1:]:  # Skip first year as reference
            dummy = (df['year'] == year).astype(int)
            year_dummies.append(dummy)
            year_names.append(f'year_{year}')
        
        year_matrix = np.column_stack(year_dummies)
        print(f"   - Created {len(year_names)} year dummy variables")
        print(f"   - Reference year: {years[0]}")
        print(f"   - Year dummies: {year_names}")
        
    elif method == 'continuous':
        # Use year as continuous variable (centered)
        year_continuous = df['year'] - df['year'].mean()
        year_matrix = year_continuous.values.reshape(-1, 1)
        year_names = ['year_continuous']
        print(f"   - Created continuous year variable (centered)")
        print(f"   - Mean year: {df['year'].mean():.1f}")
        
    else:
        raise ValueError("method must be 'dummies' or 'continuous'")
    
    return year_matrix, year_names


def prepare_covariates(df):
    """
    Prepare all other covariates for the model.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Panel data
        
    Returns:
    --------
    np.ndarray : Covariate matrix
    list : Covariate names
    """
    print("\nStep 3: Preparing other covariates...")
    
    covariates = []
    covariate_names = []
    
    # 1. Continuous variables
    continuous_vars = ['rarity_score', 'work_years']
    for var in continuous_vars:
        if var in df.columns:
            # Standardize continuous variables
            standardized = (df[var] - df[var].mean()) / df[var].std()
            covariates.append(standardized.values)
            covariate_names.append(f'{var}_std')
            print(f"   - Added standardized {var}")
    
    # 2. Educational degree (ordinal, but treat as dummies)
    if 'highest_educational_degree' in df.columns:
        education_dummies = pd.get_dummies(df['highest_educational_degree'], prefix='edu')
        # Drop first dummy as reference
        edu_cols = education_dummies.columns[1:]
        for col in edu_cols:
            covariates.append(education_dummies[col].values)
            covariate_names.append(col)
        print(f"   - Added education dummies: {list(edu_cols)}")
    
    # 3. University prestige (binary)
    if 'whether_bachelor_university_prestigious' in df.columns:
        prestigious = df['whether_bachelor_university_prestigious'].astype(int).values
        covariates.append(prestigious)
        covariate_names.append('prestigious_university')
        print(f"   - Added prestigious university indicator")
    
    # 4. Internationalization (categorical)
    if 'internationalization' in df.columns:
        intl_dummies = pd.get_dummies(df['internationalization'], prefix='intl')
        # Drop first dummy as reference
        intl_cols = intl_dummies.columns[1:]
        for col in intl_cols:
            covariates.append(intl_dummies[col].values)
            covariate_names.append(col)
        print(f"   - Added internationalization dummies: {list(intl_cols)}")
    
    # 5. Company size (categorical)
    if 'simplified_company_size' in df.columns:
        size_dummies = pd.get_dummies(df['simplified_company_size'], prefix='size')
        # Drop first dummy as reference
        size_cols = size_dummies.columns[1:]
        for col in size_cols:
            covariates.append(size_dummies[col].values)
            covariate_names.append(col)
        print(f"   - Added company size dummies: {list(size_cols)}")
    
    # 6. Gender (binary)
    if 'gender' in df.columns:
        gender_female = (df['gender'] == 'female').astype(int).values
        covariates.append(gender_female)
        covariate_names.append('female')
        print(f"   - Added female indicator")
    
    # 7. Country (if multiple countries)
    if 'country' in df.columns and df['country'].nunique() > 1:
        country_dummies = pd.get_dummies(df['country'], prefix='country')
        # Drop first dummy as reference
        country_cols = country_dummies.columns[1:]
        for col in country_cols:
            covariates.append(country_dummies[col].values)
            covariate_names.append(col)
        print(f"   - Added country dummies: {list(country_cols)}")
    
    if len(covariates) == 0:
        print("   - WARNING: No covariates prepared!")
        return np.array([]).reshape(len(df), 0), []
    
    covariate_matrix = np.column_stack(covariates)
    print(f"   - Total covariates: {len(covariate_names)}")
    print(f"   - Covariate names: {covariate_names}")
    
    return covariate_matrix, covariate_names


def prepare_ordinal_outcome(df, outcome_var='seniority'):
    """
    Prepare ordinal outcome variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Panel data
    outcome_var : str
        Name of ordinal outcome variable
        
    Returns:
    --------
    np.ndarray : Encoded ordinal outcome
    list : Category labels
    """
    print(f"\nStep 4: Preparing ordinal outcome variable '{outcome_var}'...")
    
    if outcome_var not in df.columns:
        raise ValueError(f"Outcome variable '{outcome_var}' not found in data")
    
    # Get unique values and their counts
    value_counts = df[outcome_var].value_counts()
    print(f"   - Outcome categories: {dict(value_counts)}")
    
    # Define ordinal ordering for seniority levels
    if outcome_var == 'seniority':
        # Seniority ordering based on actual data exploration
        # From lowest to highest level: Assistant -> Junior -> Regular -> Senior -> Leader -> Chief or founder
        seniority_order = [
            'Assistant', 'Junior', 'Regular', 'Senior', 'Leader', 'Chief or founder'
        ]
        
        # Filter to only categories present in data
        present_categories = df[outcome_var].unique()
        ordered_categories = [cat for cat in seniority_order if cat in present_categories]
        
        # Add any categories not in predefined order
        missing_categories = [cat for cat in present_categories if cat not in ordered_categories]
        if missing_categories:
            print(f"   - WARNING: Categories not in predefined order: {missing_categories}")
            print(f"   - Adding them at the end: {missing_categories}")
            ordered_categories.extend(sorted(missing_categories))
        
        print(f"   - Ordinal ordering: {ordered_categories}")
        
        # Encode using explicit ordering
        category_to_int = {cat: i+1 for i, cat in enumerate(ordered_categories)}
        encoded_outcome = np.array([category_to_int[val] for val in df[outcome_var]])
        
    else:
        # For other variables, try automatic encoding
        try:
            encoder = LabelEncoder()
            encoded_outcome = encoder.fit_transform(df[outcome_var]) + 1  # Start from 1
            ordered_categories = encoder.classes_
            print(f"   - Automatic ordering: {ordered_categories}")
            print(f"   - WARNING: Please verify this ordering is correct for your ordinal variable")
            
        except Exception as e:
            raise ValueError(f"Could not encode ordinal outcome '{outcome_var}': {e}")
    
    print(f"   - Encoded categories: 1-{len(ordered_categories)}")
    print(f"   - Distribution: {np.bincount(encoded_outcome)}")
    
    # Check if we have enough categories for Brant test
    n_categories = len(np.unique(encoded_outcome))
    if n_categories < 3:
        raise ValueError(f"Brant test requires at least 3 ordinal categories, found {n_categories}")
    
    return encoded_outcome, ordered_categories


def define_variable_groups(year_names, covariate_names):
    """
    Define variable groups for grouped testing.
    
    Parameters:
    -----------
    year_names : list
        Names of year variables
    covariate_names : list
        Names of other covariates
        
    Returns:
    --------
    dict : Variable groups mapping
    """
    print("\nStep 5: Defining variable groups for grouped testing...")
    
    all_names = year_names + covariate_names
    variable_groups = {}
    current_idx = 0
    
    # Year variables as one group
    if len(year_names) > 0:
        year_indices = list(range(current_idx, current_idx + len(year_names)))
        variable_groups['Year_Effects'] = year_indices
        current_idx += len(year_names)
        print(f"   - Year_Effects: indices {year_indices} ({year_names})")
    
    # Group education variables
    edu_indices = []
    edu_vars = []
    for i, name in enumerate(covariate_names):
        if 'edu_' in name:
            edu_indices.append(current_idx + i)
            edu_vars.append(name)
    if edu_indices:
        variable_groups['Education'] = edu_indices
        print(f"   - Education: indices {edu_indices} ({edu_vars})")
    
    # Group internationalization variables
    intl_indices = []
    intl_vars = []
    for i, name in enumerate(covariate_names):
        if 'intl_' in name:
            intl_indices.append(current_idx + i)
            intl_vars.append(name)
    if intl_indices:
        variable_groups['Internationalization'] = intl_indices
        print(f"   - Internationalization: indices {intl_indices} ({intl_vars})")
    
    # Group company size variables
    size_indices = []
    size_vars = []
    for i, name in enumerate(covariate_names):
        if 'size_' in name:
            size_indices.append(current_idx + i)
            size_vars.append(name)
    if size_indices:
        variable_groups['Company_Size'] = size_indices
        print(f"   - Company_Size: indices {size_indices} ({size_vars})")
    
    # Individual variables
    individual_vars = ['rarity_score_std', 'work_years_std', 'prestigious_university', 'female']
    for var in individual_vars:
        if var in covariate_names:
            var_idx = current_idx + covariate_names.index(var)
            variable_groups[var] = [var_idx]
            print(f"   - {var}: index [{var_idx}]")
    
    print(f"   - Total groups: {len(variable_groups)}")
    return variable_groups


def run_panel_brant_test(data_file=None, year_method='dummies', verbose=True):
    """
    Run comprehensive Brant test on panel data with year controls.
    
    Parameters:
    -----------
    data_file : str, optional
        Path to data file. If None, uses sample data.
    year_method : str
        Method for year variables: 'dummies' or 'continuous'
    verbose : bool
        Print detailed progress information
        
    Returns:
    --------
    dict : Brant test results
    """
    print("=" * 80)
    print("PANEL DATA BRANT TEST FOR PROPORTIONAL ODDS ASSUMPTION")
    print("=" * 80)
    print(f"Year method: {year_method}")
    print(f"Verbose output: {verbose}")
    print()
    
    try:
        # Step 1: Load data
        df = load_and_prepare_panel_data(data_file)
        
        # Step 2: Create year variables
        year_matrix, year_names = create_year_variables(df, method=year_method)
        
        # Step 3: Prepare other covariates
        covariate_matrix, covariate_names = prepare_covariates(df)
        
        # Step 4: Prepare outcome variable
        y_encoded, category_labels = prepare_ordinal_outcome(df, 'seniority')
        
        # Step 5: Combine all predictors
        if year_matrix.size > 0 and covariate_matrix.size > 0:
            X_combined = np.column_stack([year_matrix, covariate_matrix])
        elif year_matrix.size > 0:
            X_combined = year_matrix
        elif covariate_matrix.size > 0:
            X_combined = covariate_matrix
        else:
            raise ValueError("No predictors available for analysis")
        
        all_feature_names = year_names + covariate_names
        
        print(f"\nStep 6: Final data preparation...")
        print(f"   - Total observations: {len(y_encoded)}")
        print(f"   - Total predictors: {X_combined.shape[1]}")
        print(f"   - Outcome categories: {len(np.unique(y_encoded))} ({category_labels})")
        print(f"   - Feature names: {all_feature_names}")
        
        # Step 6: Define variable groups
        variable_groups = define_variable_groups(year_names, covariate_names)
        
        # Step 7: Run Brant test - Individual variables
        print("\n" + "=" * 80)
        print("INDIVIDUAL VARIABLE TESTING")
        print("=" * 80)
        
        results_individual = brant_test(
            X=X_combined,
            y=y_encoded,
            feature_names=all_feature_names,
            by_var=False,
            alpha=0.05,
            drop_constant=True,
            verbose=verbose
        )
        
        # Step 8: Run Brant test - Grouped variables
        print("\n" + "=" * 80)
        print("GROUPED VARIABLE TESTING")
        print("=" * 80)
        
        results_grouped = brant_test(
            X=X_combined,
            y=y_encoded,
            feature_names=all_feature_names,
            by_var=True,
            variable_groups=variable_groups,
            alpha=0.05,
            drop_constant=True,
            verbose=verbose
        )
        
        # Step 9: Summary and interpretation
        print("\n" + "=" * 80)
        print("INTERPRETATION AND RECOMMENDATIONS")
        print("=" * 80)
        
        # Check omnibus results
        omnibus_p_individual = results_individual['omnibus']['p_value']
        omnibus_p_grouped = results_grouped['omnibus']['p_value']
        
        print(f"\nOmnibus Test Results:")
        print(f"   - Individual testing: p = {omnibus_p_individual:.4f}")
        print(f"   - Grouped testing: p = {omnibus_p_grouped:.4f}")
        
        if omnibus_p_grouped < 0.05:
            print(f"\n✗ PROPORTIONAL ODDS ASSUMPTION VIOLATED (p < 0.05)")
            print(f"   The parallel regression assumption does not hold for your model.")
            print(f"   This suggests that the relationship between predictors and the")
            print(f"   log-odds varies across different thresholds of seniority levels.")
            
            # Identify problematic variables
            print(f"\nProblematic variable groups (p < 0.05):")
            for group_name, test_result in results_grouped['individual_tests'].items():
                if test_result.get('significant', False):
                    print(f"   - {group_name}: p = {test_result['p_value']:.4f}")
            
            print(f"\nRecommendations:")
            print(f"   1. Consider using partial proportional odds models")
            print(f"   2. Try multinomial logistic regression instead")
            print(f"   3. Examine interactions between problematic variables and outcome levels")
            print(f"   4. Consider collapsing some seniority categories")
            
        else:
            print(f"\n✓ PROPORTIONAL ODDS ASSUMPTION SATISFIED (p ≥ 0.05)")
            print(f"   The parallel regression assumption holds for your model.")
            print(f"   You can proceed with standard ordinal logistic regression.")
            print(f"   The coefficients have consistent interpretation across all")
            print(f"   seniority level thresholds.")
        
        # Year effect interpretation
        year_significant = False
        if 'Year_Effects' in results_grouped['individual_tests']:
            year_p = results_grouped['individual_tests']['Year_Effects']['p_value']
            year_significant = results_grouped['individual_tests']['Year_Effects'].get('significant', False)
            
            print(f"\nYear Effects:")
            if year_significant:
                print(f"   - Year effects are significant (p = {year_p:.4f})")
                print(f"   - Different years have different baseline seniority distributions")
                print(f"   - Important to control for temporal trends")
            else:
                print(f"   - Year effects are not significant (p = {year_p:.4f})")
                print(f"   - Seniority distributions are stable across years")
        
        return {
            'individual_results': results_individual,
            'grouped_results': results_grouped,
            'data_summary': {
                'n_observations': len(y_encoded),
                'n_predictors': X_combined.shape[1],
                'n_categories': len(category_labels),
                'category_labels': category_labels,
                'feature_names': all_feature_names,
                'variable_groups': variable_groups,
                'year_method': year_method
            }
        }
        
    except Exception as e:
        print(f"\nERROR: Panel Brant test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def explore_user_data(file_path):
    """
    Explore and understand the structure of user's data file.
    
    Parameters:
    -----------
    file_path : str
        Path to the data file
        
    Returns:
    --------
    None : Prints exploration results
    """
    print("=" * 80)
    print("DATA EXPLORATION MODE")
    print("=" * 80)
    
    try:
        df = load_and_prepare_panel_data(file_path)
        
        print(f"\n📊 DATA STRUCTURE ANALYSIS")
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print(f"\n📋 COLUMN INFORMATION:")
        for i, col in enumerate(df.columns):
            dtype = str(df[col].dtype)  # Convert dtype to string for formatting
            n_unique = df[col].nunique()
            missing = df[col].isnull().sum()
            print(f"  {i:2d}. {col:<25} | {dtype:<10} | {n_unique:>6} unique | {missing:>6} missing")
        
        # Check for potential time/year variables
        print(f"\n⏰ POTENTIAL TIME VARIABLES:")
        time_candidates = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['year', 'time', 'date', 'period']):
                time_candidates.append(col)
        
        if time_candidates:
            for col in time_candidates:
                print(f"  - {col}: {df[col].dtype}, range: {df[col].min()} to {df[col].max()}")
        else:
            print("  - No obvious time variables found")
            
        # Check for potential ID variables  
        print(f"\n🆔 POTENTIAL ID VARIABLES:")
        id_candidates = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['id', 'worker', 'person', 'individual']):
                id_candidates.append(col)
                
        if id_candidates:
            for col in id_candidates:
                print(f"  - {col}: {df[col].nunique()} unique values out of {len(df)} rows")
        else:
            print("  - No obvious ID variables found")
        
        # Check for C1-C10 pattern (career sequences)
        c_columns = [col for col in df.columns if col.startswith('C') and col[1:].isdigit()]
        if len(c_columns) >= 5:  # At least 5 C columns
            print(f"\n📈 CAREER SEQUENCE VARIABLES (C1-C{max([int(col[1:]) for col in c_columns])}):")
            print(f"  - Found {len(c_columns)} sequential columns: {sorted(c_columns)}")
            
            # Sample some values to understand what they represent
            print(f"  - Sample values from first worker:")
            if 'worker_id' in df.columns:
                first_worker = df.iloc[0]
                c_values = [first_worker[col] for col in sorted(c_columns)]
                print(f"    {c_values}")
            else:
                first_row_c = [df.iloc[0][col] for col in sorted(c_columns)]
                print(f"    {first_row_c}")
            
            # Check data types and unique values
            print(f"  - Data types: {[df[col].dtype for col in sorted(c_columns)[:3]]}...")
            unique_vals_c1 = df[c_columns[0]].unique()
            print(f"  - Unique values in C1: {sorted(unique_vals_c1) if len(unique_vals_c1) < 20 else f'{len(unique_vals_c1)} unique values'}")
            
            # Check if C columns look like seniority levels
            if len(unique_vals_c1) < 10:  # Could be categorical
                print(f"  - C1 appears categorical (potential seniority levels)")
                for col in sorted(c_columns)[:3]:  # Show first 3
                    vals = df[col].value_counts().head()
                    print(f"    {col} distribution: {dict(vals)}")
        
        # Check for categorical variables that might be ordinal outcomes
        print(f"\n🎯 POTENTIAL ORDINAL OUTCOMES:")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            unique_vals = df[col].unique()
            if 2 < len(unique_vals) < 10:  # Reasonable for ordinal
                print(f"  - {col}: {len(unique_vals)} categories - {list(unique_vals)}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        if c_columns:
            print("  1. Your data appears to have career sequence structure (C1-C10)")
            print("  2. This is likely WIDE format - each row is one worker, C1-C10 are years 1-10")
            print("  3. For Brant test, you'll need to:")
            print("     a. Reshape to LONG format (one row per worker-year)")
            print("     b. Choose one C column as ordinal outcome")
            print("     c. Use year (1-10) as time variable")
            print("     d. Include worker characteristics as covariates")
        else:
            print("  1. Please identify which variable represents the ordinal outcome")
            print("  2. Identify time/year variable for panel structure")
            print("  3. Specify individual ID variable")
            
        print(f"\n🔧 NEXT STEPS:")
        print("  1. If this is career sequence data (C1-C10):")
        print("     - Set RESHAPE_NEEDED = True in main function") 
        print("     - Specify which C column represents seniority levels")
        print("  2. Otherwise, modify data preparation to match your structure")
        
        return df
        
    except Exception as e:
        print(f"❌ Error exploring data: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    """
    Main execution: Run panel Brant test with sample data or your data file.
    
    USAGE MODES:
    1. Data exploration: Set EXPLORE_MODE = True to understand your data structure
    2. Analysis mode: Set EXPLORE_MODE = False to run Brant test
    3. Sample data testing: Set data_file_path = None to use sample data
    """
    
    print("Starting Panel Data Brant Test Analysis...")
    print("=" * 80)
    
    # CONFIGURATION - MODIFY THESE SETTINGS
    EXPLORE_MODE = False  # Set to True to explore data structure first
    
    # Update this path to your actual data file (or None for sample data)
    data_file_path = "/Users/lei/Documents/Sequenzo_all_folders/sequenzo_local/test_data/real_data_my_paper/250808_tree/final_df.csv"
    
    year_treatment = 'dummies'  # 'dummies' or 'continuous'
    
    if EXPLORE_MODE:
        print("🔍 EXPLORATION MODE ENABLED")
        print("This will help you understand your data structure before running Brant test")
        print()
        
        # Data exploration mode
        explore_user_data(data_file_path)
        
        print(f"\n" + "=" * 80)
        print("EXPLORATION COMPLETED")
        print("=" * 80)
        print("💡 To run the actual Brant test:")
        print("   1. Set EXPLORE_MODE = False")
        print("   2. Modify data preparation code based on exploration results")
        print("   3. Run this script again")
    else:
        print("📊 ANALYSIS MODE ENABLED")
        print("Running Brant test for proportional odds assumption...")
        print()
        
        # Analysis mode
        results = run_panel_brant_test(
            data_file=data_file_path,
            year_method=year_treatment,
            verbose=True
        )
        
        # Show results summary
        if results:
            print(f"\n" + "=" * 80)
            print("ANALYSIS COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print(f"Results available in 'results' dictionary with keys:")
            print(f"   - individual_results: Individual variable test results")  
            print(f"   - grouped_results: Grouped variable test results")
            print(f"   - data_summary: Summary of data and model setup")
            
        else:
            print(f"\n" + "=" * 80)
            print("ANALYSIS FAILED")
            print("=" * 80)
            print(f"Please check error messages above and data preparation.")
