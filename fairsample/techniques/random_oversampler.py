"""
Random Oversampling implementation.
"""

import numpy as np
from .base_sampler import BaseSampler


class RandomOverSampler(BaseSampler):
    """
    Random Oversampling.
    
    This technique randomly duplicates minority class samples to balance
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
        self._sampling_type = 'oversampling'
    
    def fit_resample(self, X, y):
        """
        Resample the dataset using random oversampling.
        
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
        
        # Calculate target number of minority samples
        if self.sampling_strategy == 'auto':
            target_minority_size = len(majority_indices)
        elif isinstance(self.sampling_strategy, (int, float)):
            target_minority_size = int(len(majority_indices) * self.sampling_strategy)
        else:
            raise ValueError("sampling_strategy must be 'auto' or a number")
        
        # If we already have enough minority samples, return as is
        if len(minority_indices) >= target_minority_size:
            return X.copy(), y.copy()
        
        # Calculate how many samples to generate
        n_samples_to_generate = target_minority_size - len(minority_indices)
        
        # Randomly select minority samples to duplicate
        np.random.seed(self.random_state)
        duplicate_indices = np.random.choice(
            minority_indices,
            size=n_samples_to_generate,
            replace=True
        )
        
        # Create resampled dataset
        X_resampled = np.vstack([X, X[duplicate_indices]])
        y_resampled = np.hstack([y, y[duplicate_indices]])
        
        return X_resampled, y_resampled