"""
Undersampling based on Recursive Neighbourhood Search (URNS) implementation.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from .base_sampler import BaseSampler


class URNS(BaseSampler):
    """
    Undersampling based on Recursive Neighbourhood Search (URNS).
    
    This technique recursively removes majority class samples that are
    in dense regions and close to the decision boundary.
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to consider
    random_state : int, default=None
        Random state for reproducibility
    max_iterations : int, default=10
        Maximum number of recursive iterations
    """
    
    def __init__(self, n_neighbors=5, random_state=None, max_iterations=10):
        super().__init__(random_state=random_state)
        self.n_neighbors = n_neighbors
        self.max_iterations = max_iterations
        self._sampling_type = 'undersampling'
    
    def fit_resample(self, X, y):
        """
        Resample the dataset using URNS.
        
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
        
        # Start with all samples
        current_indices = np.arange(len(X))
        target_majority_size = len(minority_indices)
        
        for iteration in range(self.max_iterations):
            # Get current majority indices
            current_y = y[current_indices]
            current_majority_mask = current_y == majority_class
            current_majority_indices = current_indices[current_majority_mask]
            
            # If we've reached the target size, stop
            if len(current_majority_indices) <= target_majority_size:
                break
            
            # Fit k-NN on current data
            X_current = X[current_indices]
            k = min(self.n_neighbors, len(X_current) - 1)
            if k <= 0:
                break
                
            knn = NearestNeighbors(n_neighbors=k + 1)
            knn.fit(X_current)
            
            # Find neighbors for each sample
            distances, indices = knn.kneighbors(X_current)
            
            # Calculate neighborhood purity for majority samples
            majority_scores = []
            current_majority_local_indices = np.where(current_majority_mask)[0]
            
            for local_idx in current_majority_local_indices:
                # Get neighbors (excluding self)
                neighbor_indices = indices[local_idx, 1:]
                neighbor_labels = current_y[neighbor_indices]
                
                # Calculate purity (fraction of majority class neighbors)
                purity = np.sum(neighbor_labels == majority_class) / len(neighbor_labels)
                majority_scores.append((local_idx, purity))
            
            if not majority_scores:
                break
            
            # Sort by purity (highest first) and remove samples with highest purity
            majority_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Remove a fraction of the most pure majority samples
            n_to_remove = min(
                len(current_majority_indices) - target_majority_size,
                max(1, len(majority_scores) // 4)  # Remove 25% at most
            )
            
            indices_to_remove = [score[0] for score in majority_scores[:n_to_remove]]
            global_indices_to_remove = current_indices[indices_to_remove]
            
            # Update current indices
            current_indices = np.setdiff1d(current_indices, global_indices_to_remove)
        
        return X[current_indices], y[current_indices]