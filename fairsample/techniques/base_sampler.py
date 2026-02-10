"""
Base class for all resampling techniques.
"""

import numpy as np
from abc import ABC, abstractmethod
from sklearn.utils import check_X_y


class BaseSampler(ABC):
    """
    Base class for all resampling techniques.
    
    All resampling techniques should inherit from this class and implement
    the fit_resample method.
    
    Parameters
    ----------
    random_state : int, default=None
        Random state for reproducibility
    """
    
    def __init__(self, random_state=None):
        self.random_state = random_state
        self._sampling_type = 'unknown'
    
    @abstractmethod
    def fit_resample(self, X, y):
        """
        Resample the dataset.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        X_resampled : array-like
            Resampled training data
        y_resampled : array-like
            Resampled target values
        """
        pass
    
    def _validate_input(self, X, y):
        """Validate input data."""
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
        
        # Check for binary classification
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(f"Only binary classification supported. Got {len(unique_classes)} classes.")
        
        return X, y
    
    def _get_minority_majority_indices(self, y):
        """Get indices of minority and majority classes."""
        unique_classes, counts = np.unique(y, return_counts=True)
        
        if counts[0] < counts[1]:
            minority_class = unique_classes[0]
            majority_class = unique_classes[1]
        else:
            minority_class = unique_classes[1]
            majority_class = unique_classes[0]
        
        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]
        
        return minority_indices, majority_indices, minority_class, majority_class