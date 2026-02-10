"""
Random Undersampling implementation.
"""

import numpy as np
from .base_sampler import BaseSampler


class RandomUnderSampler(BaseSampler):
    """
    Random Undersampling.
    
    This technique randomly removes majority class samples to balance
    the dataset.
    
    Parameters
    ----------
    sampling_strategy : str or float, default='auto'
        Sampling strategy. If 'auto', balance to 1:1 ratio.
        If float, specify the desired ratio of minority to majority.
    random_state : int, default=None
        Random state for reproducibility
    """
    
    def __init__(self, sampling_strategy='auto', random_state=None):
        super().__init__(random_state=random_state)
        self.sampling_strategy = sampling_strategy
        self._sampling_type = 'undersampling'
    
    def fit_resample(self, X, y):
        """
        Resample the dataset using random undersampling.
        
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
        X, y = self._validate_input(X, y)
        
        # Get minority and majority class indices
        minority_indices, majority_indices, minority_class, majority_class = \
            self._get_minority_majority_indices(y)
        
        # Calculate target number of majority samples
        if self.sampling_strategy == 'auto':
            target_majority_size = len(minority_indices)
        elif isinstance(self.sampling_strategy, (int, float)):
            target_majority_size = int(len(minority_indices) / self.sampling_strategy)
        else:
            raise ValueError("sampling_strategy must be 'auto' or a number")
        
        # If we already have few enough majority samples, return as is
        if len(majority_indices) <= target_majority_size:
            return X.copy(), y.copy()
        
        # Randomly select majority samples to keep
        np.random.seed(self.random_state)
        keep_majority_indices = np.random.choice(
            majority_indices,
            size=target_majority_size,
            replace=False
        )
        
        # Combine minority and selected majority samples
        keep_indices = np.concatenate([minority_indices, keep_majority_indices])
        keep_indices = np.sort(keep_indices)
        
        return X[keep_indices], y[keep_indices]