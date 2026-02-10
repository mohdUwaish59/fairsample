"""
High-level comparison utilities for resampling techniques.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Any

from ..techniques import *
from .helpers import get_available_techniques, validate_input_data


def compare_techniques(
    X: np.ndarray,
    y: np.ndarray,
    techniques: Union[List[str], str] = 'all',
    complexity_measures: Union[str, List[str]] = 'basic',
    random_state: int = 42,
    include_original: bool = True,
    verbose: bool = True,
    **technique_params
) -> pd.DataFrame:
    """
    Compare multiple resampling techniques based on complexity scores.
    
    This function applies different resampling techniques and calculates
    complexity measures for each, allowing you to compare how each technique
    affects dataset complexity.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Target vector
    techniques : list of str or 'all', default='all'
        List of technique names to compare, or 'all' for all available
    complexity_measures : str or list, default='basic'
        Which complexity measures to calculate:
        - 'basic': Essential measures (N3, N1, N2, F1, F2)
        - 'standard': Common measures for comparison
        - 'all': All available measures (may be slow)
        - 'feature': Only feature overlap measures
        - 'instance': Only instance overlap measures
        - 'structural': Only structural overlap measures
        - list: Specific measure names (e.g., ['N3', 'F1', 'N1', 'kDN'])
    random_state : int, default=42
        Random state for reproducibility
    include_original : bool, default=True
        Whether to include original data complexity in results
    verbose : bool, default=True
        Whether to print progress information
    **technique_params : dict
        Additional parameters for specific techniques
    
    Returns
    -------
    results : pd.DataFrame
        DataFrame with complexity scores for each technique.
        Columns depend on complexity_measures parameter.
    
    Example
    -------
    >>> from fairsample.utils import compare_techniques
    >>> 
    >>> # Basic comparison (fast)
    >>> results = compare_techniques(X, y, ['RFCL', 'NUS', 'URNS'])
    >>> print(results[['samples', 'N3', 'imbalance_ratio']])
    >>> 
    >>> # Get all complexity measures
    >>> all_results = compare_techniques(X, y, ['RFCL', 'NUS'], 
    >>>                                 complexity_measures='all')
    >>> print(all_results.columns)
    >>> 
    >>> # Get only specific measures
    >>> custom = compare_techniques(X, y, ['RFCL', 'NUS'],
    >>>                            complexity_measures=['N3', 'F1', 'kDN', 'borderline'])
    >>> print(custom)
    >>> 
    >>> # Save results
    >>> results.to_csv('technique_comparison.csv')
    """
    from ..complexity import ComplexityMeasures
    
    # Validate input
    X, y = validate_input_data(X, y)
    
    # Get technique list
    if techniques == 'all':
        techniques = get_available_techniques()
    elif isinstance(techniques, str):
        techniques = [techniques]
    
    if verbose:
        print(f"Comparing {len(techniques)} techniques based on complexity scores")
        print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
        print(f"Class distribution: {np.bincount(y)}")
        print(f"Complexity measures: {complexity_measures}")
        print("="*60)
    
    results = []
    
    # Calculate original complexity if requested
    if include_original:
        try:
            if verbose:
                print("Calculating original data complexity...")
            cm_original = ComplexityMeasures(X, y)
            
            # Get complexity measures based on user selection
            if isinstance(complexity_measures, str):
                if complexity_measures in ['basic', 'standard', 'all']:
                    original_scores = cm_original.analyze_overlap(measures=complexity_measures)
                else:
                    # Category selection
                    all_measures = cm_original.get_all_complexity_measures(measures=complexity_measures)
                    original_scores = _flatten_complexity_dict(all_measures)
            elif isinstance(complexity_measures, list):
                # Specific measures requested
                all_measures = cm_original.get_all_complexity_measures(measures=complexity_measures)
                original_scores = _flatten_complexity_dict(all_measures)
            else:
                original_scores = cm_original.analyze_overlap(measures='basic')
            
            # Add basic info
            result_row = {
                'technique': 'Original',
                'samples': len(X),
                'imbalance_ratio': original_scores.get('imbalance_ratio', np.max(np.bincount(y)) / np.min(np.bincount(y)))
            }
            
            # Add complexity scores
            for key, value in original_scores.items():
                if key not in ['n_samples', 'n_features', 'n_classes', 'class_distribution', 'dataset_info']:
                    if isinstance(value, (list, np.ndarray)):
                        try:
                            result_row[key] = float(np.mean(value))
                        except:
                            result_row[key] = str(value)
                    else:
                        result_row[key] = value
            
            results.append(result_row)
            
            if verbose:
                n3_val = result_row.get('N3', 'N/A')
                n3_str = f"{n3_val:.4f}" if isinstance(n3_val, (int, float)) else 'N/A'
                print(f"✓ Original: N3={n3_str}, Imbalance={result_row['imbalance_ratio']:.2f}")
        except Exception as e:
            if verbose:
                print(f"✗ Original complexity calculation failed: {e}")
    
    # Compare each technique
    for tech_name in techniques:
        try:
            # Get technique class
            tech_class = globals().get(tech_name)
            if tech_class is None:
                if verbose:
                    print(f"Warning: Technique '{tech_name}' not found, skipping...")
                continue
            
            # Initialize technique
            tech_params = technique_params.get(tech_name, {})
            tech_instance = tech_class(random_state=random_state, **tech_params)
            
            # Apply resampling
            if verbose:
                print(f"Applying {tech_name}...")
            X_res, y_res = tech_instance.fit_resample(X, y)
            
            # Validate result
            if len(np.unique(y_res)) < 2:
                if verbose:
                    print(f"Warning: {tech_name} resulted in single class, skipping...")
                continue
            
            # Calculate complexity after resampling
            if verbose:
                print(f"Calculating complexity for {tech_name}...")
            cm_resampled = ComplexityMeasures(X_res, y_res)
            
            # Get complexity measures based on user selection
            if isinstance(complexity_measures, str):
                if complexity_measures in ['basic', 'standard', 'all']:
                    resampled_scores = cm_resampled.analyze_overlap(measures=complexity_measures)
                else:
                    all_measures = cm_resampled.get_all_complexity_measures(measures=complexity_measures)
                    resampled_scores = _flatten_complexity_dict(all_measures)
            elif isinstance(complexity_measures, list):
                all_measures = cm_resampled.get_all_complexity_measures(measures=complexity_measures)
                resampled_scores = _flatten_complexity_dict(all_measures)
            else:
                resampled_scores = cm_resampled.analyze_overlap(measures='basic')
            
            # Add basic info
            result_row = {
                'technique': tech_name,
                'samples': len(X_res),
                'imbalance_ratio': resampled_scores.get('imbalance_ratio', np.max(np.bincount(y_res)) / np.min(np.bincount(y_res)))
            }
            
            # Add complexity scores
            for key, value in resampled_scores.items():
                if key not in ['n_samples', 'n_features', 'n_classes', 'class_distribution', 'dataset_info']:
                    if isinstance(value, (list, np.ndarray)):
                        try:
                            result_row[key] = float(np.mean(value))
                        except:
                            result_row[key] = str(value)
                    else:
                        result_row[key] = value
            
            results.append(result_row)
            
            if verbose:
                n3_val = result_row.get('N3', 'N/A')
                n3_str = f"{n3_val:.4f}" if isinstance(n3_val, (int, float)) else 'N/A'
                print(f"✓ {tech_name}: {len(X)} → {len(X_res)} samples, N3={n3_str}")
                
        except Exception as e:
            if verbose:
                print(f"✗ {tech_name} failed: {str(e)}")
            continue
    
    if not results:
        raise ValueError("No techniques were successfully evaluated")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('technique')
    
    if verbose:
        print("\n" + "="*60)
        print("COMPLEXITY COMPARISON RESULTS")
        print("="*60)
        # Show key columns
        display_cols = ['samples', 'imbalance_ratio']
        if 'N3' in results_df.columns:
            display_cols.append('N3')
        if 'N1' in results_df.columns:
            display_cols.append('N1')
        if 'F1' in results_df.columns:
            display_cols.append('F1')
        
        print(results_df[display_cols].round(4))
        print("\nInterpretation:")
        print("  - Lower N3, N1, F2 = Less overlap (better)")
        print("  - Higher F1, SI = Better separation (better)")
        print("  - Imbalance ratio closer to 1.0 = More balanced (usually better)")
        print(f"\nTotal measures calculated: {len(results_df.columns)}")
        print(f"Available columns: {list(results_df.columns)}")
    
    return results_df


def _flatten_complexity_dict(complexity_dict):
    """Flatten nested complexity dictionary."""
    flattened = {}
    
    for category, measures in complexity_dict.items():
        if isinstance(measures, dict):
            for key, value in measures.items():
                if value is not None:
                    flattened[key] = value
        else:
            flattened[category] = measures
    
    return flattened


def get_resampled_data(
    X: np.ndarray,
    y: np.ndarray,
    techniques: Union[List[str], str] = 'all',
    random_state: int = 42,
    verbose: bool = True,
    **technique_params
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Get resampled data from multiple techniques.
    
    This function applies different resampling techniques and returns the
    resampled datasets. Users can then save to CSV, train models, create
    visualizations, or perform any custom analysis.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Target vector
    techniques : list of str or 'all', default='all'
        List of technique names to apply, or 'all' for all available
    random_state : int, default=42
        Random state for reproducibility
    verbose : bool, default=True
        Whether to print progress information
    **technique_params : dict
        Additional parameters for specific techniques
    
    Returns
    -------
    data_dict : dict
        Dictionary with structure:
        {
            'original': {'X': X_original, 'y': y_original},
            'technique_name': {'X': X_resampled, 'y': y_resampled},
            ...
        }
    
    Example
    -------
    >>> from fairsample.utils import get_resampled_data
    >>> import pandas as pd
    >>> 
    >>> # Get resampled data
    >>> data = get_resampled_data(X, y, ['RFCL', 'NUS', 'URNS'])
    >>> 
    >>> # Save to CSV
    >>> for technique, info in data.items():
    >>>     df = pd.DataFrame(info['X'])
    >>>     df['target'] = info['y']
    >>>     df.to_csv(f'{technique}_resampled.csv', index=False)
    >>> 
    >>> # Train your own model
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> clf = RandomForestClassifier()
    >>> clf.fit(data['RFCL']['X'], data['RFCL']['y'])
    >>> 
    >>> # Create custom visualization
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.decomposition import PCA
    >>> pca = PCA(n_components=2)
    >>> X_2d = pca.fit_transform(data['RFCL']['X'])
    >>> plt.scatter(X_2d[:, 0], X_2d[:, 1], c=data['RFCL']['y'])
    >>> plt.show()
    """
    # Validate input
    X, y = validate_input_data(X, y)
    
    # Get technique list
    if techniques == 'all':
        techniques = get_available_techniques()
    elif isinstance(techniques, str):
        techniques = [techniques]
    
    if verbose:
        print(f"Getting resampled data for {len(techniques)} techniques")
        print(f"Original dataset: {len(X)} samples, {np.bincount(y)} class distribution")
        print("="*60)
    
    # Initialize result dictionary with original data
    data_dict = {
        'original': {
            'X': X.copy(),
            'y': y.copy()
        }
    }
    
    # Apply each technique
    for tech_name in techniques:
        try:
            # Get technique class
            tech_class = globals().get(tech_name)
            if tech_class is None:
                if verbose:
                    print(f"Warning: Technique '{tech_name}' not found, skipping...")
                continue
            
            # Initialize with parameters
            tech_params = technique_params.get(tech_name, {})
            tech_instance = tech_class(random_state=random_state, **tech_params)
            
            # Apply resampling
            X_res, y_res = tech_instance.fit_resample(X, y)
            
            # Validate result
            if len(np.unique(y_res)) < 2:
                if verbose:
                    print(f"Warning: {tech_name} resulted in single class, skipping...")
                continue
            
            # Store resampled data
            data_dict[tech_name] = {
                'X': X_res.copy(),
                'y': y_res.copy()
            }
            
            if verbose:
                print(f"✓ {tech_name}: {len(X)} → {len(X_res)} samples")
                
        except Exception as e:
            if verbose:
                print(f"✗ {tech_name} failed: {str(e)}")
            continue
    
    if verbose:
        print(f"\n✅ Successfully processed {len(data_dict)-1} techniques")
        print("You can now:")
        print("  - Save to CSV: pd.DataFrame(data['RFCL']['X']).to_csv('data.csv')")
        print("  - Train models: clf.fit(data['RFCL']['X'], data['RFCL']['y'])")
        print("  - Create plots: plt.scatter(data['RFCL']['X'][:, 0], ...)")
    
    return data_dict