"""
Neighbourhood-based Under-Sampling (NUS) implementation.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from .base_sampler import BaseSampler


class NUS(BaseSampler):
    """
    Neighbourhood-based Under-Sampling (NUS).
    
    This technique removes majority class samples that are in the neighborhood
    of minority class samples, helping to reduce class overlap.
    
    Parameters
    ----------
    n_neighbors : int, default=3
        Number of neighbors to consider
    random_state : int, default=None
        Random state for reproducibility
    """
    
    def __init__(self, n_neighbors=3, random_state=None):
        super().__init__(random_state=random_state)
        self.n_neighbors = n_neighbors
        self._sampling_type = 'undersampling'
    
    def fit_resample(self, X, y):
        """
        Resample the dataset using NUS.
        
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
        
        # If dataset is already balanced or minority is larger, return as is
        if len(minority_indices) >= len(majority_indices):
            return X.copy(), y.copy()
        
        # Fit k-NN on minority samples
        X_minority = X[minority_indices]
        
        # Adjust n_neighbors if we have fewer minority samples
        k = min(self.n_neighbors, len(X_minority) - 1)
        if k <= 0:
            return X.copy(), y.copy()
        
        knn = NearestNeighbors(n_neighbors=k + 1)  # +1 because it includes the point itself
        knn.fit(X_minority)
        
        # Find majority samples that are neighbors of minority samples
        X_majority = X[majority_indices]
        distances, indices = knn.kneighbors(X_majority)
        
        # Remove majority samples that are too close to minority samples
        # Use median distance as threshold
        median_distance = np.median(distances[:, 1])  # Skip first column (self)
        
        # Keep majority samples that are far enough from minority samples
        keep_majority_mask = distances[:, 1] > median_distance
        keep_majority_indices = majority_indices[keep_majority_mask]
        
        # Ensure we keep at least as many majority samples as minority samples
        if len(keep_majority_indices) < len(minority_indices):
            # If we removed too many, keep some of the closest ones
            n_additional = len(minority_indices) - len(keep_majority_indices)
            removed_indices = majority_indices[~keep_majority_mask]
            
            if len(removed_indices) > 0:
                # Sort by distance and keep the farthest ones among the removed
                removed_distances = distances[~keep_majority_mask, 1]
                sorted_indices = np.argsort(removed_distances)[::-1]  # Descending order
                additional_indices = removed_indices[sorted_indices[:n_additional]]
                keep_majority_indices = np.concatenate([keep_majority_indices, additional_indices])
        
        # Combine minority and selected majority samples
        keep_indices = np.concatenate([minority_indices, keep_majority_indices])
        keep_indices = np.sort(keep_indices)
        
        return X[keep_indices], y[keep_indices]