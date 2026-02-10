"""
Utility Functions
================

This module provides high-level convenience functions for working with
resampling techniques and complexity analysis.

Main Components:
- compare_techniques: Compare techniques based on complexity scores
- get_resampled_data: Get resampled data for custom workflows
- Helper functions

Example:
    >>> from fairsample.utils import compare_techniques, get_resampled_data
    >>> 
    >>> # Compare techniques by complexity scores
    >>> results = compare_techniques(X, y, ['RFCL', 'NUS', 'URNS'])
    >>> print(results.sort_values('N3'))  # Lower N3 = less overlap
    >>> 
    >>> # Get resampled data for your own workflow
    >>> data = get_resampled_data(X, y, ['RFCL', 'NUS'])
    >>> 
    >>> # Save to CSV
    >>> import pandas as pd
    >>> df = pd.DataFrame(data['RFCL']['X'])
    >>> df['target'] = data['RFCL']['y']
    >>> df.to_csv('rfcl_resampled.csv')
    >>> 
    >>> # Train your model
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> clf = RandomForestClassifier()
    >>> clf.fit(data['RFCL']['X'], data['RFCL']['y'])
"""

from .comparison import (
    compare_techniques, 
    get_resampled_data
)
from .helpers import get_available_techniques, validate_input_data

__all__ = [
    'compare_techniques',
    'get_resampled_data',
    'get_available_techniques',
    'validate_input_data',
]