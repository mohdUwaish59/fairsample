"""
Pytest configuration and shared fixtures for fairsample tests.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def sample_binary_data():
    """Generate sample binary classification data."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.8, 0.2],
        class_sep=1.0,
        random_state=42
    )
    return X, y


@pytest.fixture
def sample_binary_dataframe():
    """Generate sample binary classification data as pandas DataFrame."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.8, 0.2],
        class_sep=1.0,
        random_state=42
    )
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    
    return df


@pytest.fixture
def sample_multiclass_data():
    """Generate sample multiclass classification data."""
    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.6, 0.3, 0.1],
        class_sep=0.8,
        random_state=42
    )
    return X, y


@pytest.fixture
def sample_imbalanced_data():
    """Generate highly imbalanced binary classification data."""
    X, y = make_classification(
        n_samples=500,
        n_features=15,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        weights=[0.95, 0.05],
        class_sep=0.5,
        random_state=42
    )
    return X, y


@pytest.fixture
def sample_small_data():
    """Generate small dataset for edge case testing."""
    X, y = make_classification(
        n_samples=50,
        n_features=5,
        n_informative=3,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.7, 0.3],
        class_sep=1.5,
        random_state=42
    )
    return X, y


@pytest.fixture
def sample_categorical_data():
    """Generate mixed categorical and numerical data."""
    np.random.seed(42)
    n_samples = 200
    
    # Numerical features
    X_num = np.random.randn(n_samples, 5)
    
    # Categorical features
    categories = ['A', 'B', 'C']
    X_cat = np.random.choice(categories, size=(n_samples, 3))
    
    # Combine
    X = np.column_stack([X_num, X_cat])
    
    # Create target with some correlation
    y = (X_num[:, 0] + X_num[:, 1] > 0).astype(int)
    
    # Make imbalanced
    minority_indices = np.where(y == 1)[0]
    keep_minority = np.random.choice(minority_indices, size=len(minority_indices)//4, replace=False)
    keep_majority = np.where(y == 0)[0]
    keep_indices = np.concatenate([keep_majority, keep_minority])
    
    return X[keep_indices], y[keep_indices]


@pytest.fixture
def sample_edge_case_data():
    """Generate edge case data for robustness testing."""
    # Single class data
    X_single = np.random.randn(100, 5)
    y_single = np.zeros(100)
    
    # Perfect separation data
    X_perfect = np.vstack([
        np.random.randn(50, 5) - 3,
        np.random.randn(50, 5) + 3
    ])
    y_perfect = np.hstack([np.zeros(50), np.ones(50)])
    
    # Identical features data
    X_identical = np.ones((100, 5))
    y_identical = np.random.choice([0, 1], size=100, p=[0.8, 0.2])
    
    return {
        'single_class': (X_single, y_single),
        'perfect_separation': (X_perfect, y_perfect),
        'identical_features': (X_identical, y_identical)
    }


@pytest.fixture
def technique_names():
    """List of all available technique names."""
    return [
        'RandomOverSampler',
        'RandomUnderSampler',
        'RFCL',
        'URNS',
        'NUS',
        'DeviOCSVM',
        'FCMBoostOBU',
        'SVDDWSMOTE',
        'ODBOT',
        'EHSO',
        'NBUS',
        'NBBasic',
        'NBTomek',
        'NBComm',
        'NBRec',
        'KMeansUndersampling',
        'HKMUndersampling',
        'FCMUndersampling',
        'RKMUndersampling',
        'FRKMUndersampling',
        'OSM'
    ]


@pytest.fixture
def complexity_measures():
    """List of all complexity measures."""
    return [
        'F1', 'F1v', 'F2', 'F3', 'F4',  # Feature-based
        'N1', 'N2', 'N3', 'N4',         # Instance/Structural
        'R_value', 'kDN', 'CM', 'D3',   # Instance-based
        'SI', 'deg_overlap', 'borderline' # Additional instance-based
    ]