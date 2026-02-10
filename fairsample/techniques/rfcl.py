"""
Random Forest Cleaning Rule (RFCL) implementation.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from .base_sampler import BaseSampler


class RFCL(BaseSampler):
    """
    Random Forest Cleaning Rule (RFCL).
    
    This technique uses Random Forest to identify and remove noisy/overlapping
    majority class samples that are likely to be misclassified.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the random forest
    random_state : int, default=None
        Random state for reproducibility
    cv : int, default=3
        Number of cross-validation folds
    """
    
    def __init__(self, n_estimators=100, random_state=None, cv=3):
        super().__init__(random_state=random_state)
        self.n_estimators = n_estimators
        self.cv = cv
        self._sampling_type = 'undersampling'
    
    def fit_resample(self, X, y):
        """
        Resample the dataset using RFCL.
        
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
        
        # Create Random Forest classifier
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        
        # Get cross-validation predictions
        try:
            y_pred = cross_val_predict(rf, X, y, cv=self.cv)
        except:
            # Fallback: fit on full data and predict
            rf.fit(X, y)
            y_pred = rf.predict(X)
        
        # Identify correctly classified majority samples
        majority_mask = y == majority_class
        correctly_classified_majority = majority_indices[
            y_pred[majority_indices] == y[majority_indices]
        ]
        
        # Keep all minority samples and correctly classified majority samples
        keep_indices = np.concatenate([minority_indices, correctly_classified_majority])
        
        # If we removed too many samples, keep some randomly
        if len(correctly_classified_majority) < len(minority_indices):
            # Keep at least as many majority samples as minority samples
            n_additional = len(minority_indices) - len(correctly_classified_majority)
            incorrectly_classified_majority = majority_indices[
                y_pred[majority_indices] != y[majority_indices]
            ]
            
            if len(incorrectly_classified_majority) > 0:
                np.random.seed(self.random_state)
                additional_indices = np.random.choice(
                    incorrectly_classified_majority,
                    size=min(n_additional, len(incorrectly_classified_majority)),
                    replace=False
                )
                keep_indices = np.concatenate([keep_indices, additional_indices])
        
        # Sort indices to maintain order
        keep_indices = np.sort(keep_indices)
        
        return X[keep_indices], y[keep_indices]