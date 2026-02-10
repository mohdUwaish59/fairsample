"""
Helper utility functions.
"""

import numpy as np
from typing import List, Tuple, Any
from sklearn.utils import check_X_y


def get_available_techniques() -> List[str]:
    """
    Get list of all available resampling techniques.
    
    Returns
    -------
    techniques : list of str
        List of available technique names
    """
    return [
        # T1.x - Overlap-based undersampling
        'RFCL',
        'URNS',
        'NUS',
        'DeviOCSVM',
        'FCMBoostOBU',
        
        # T2.x - Hybrid methods
        'SVDDWSMOTE',
        'ODBOT',
        'EHSO',
        
        # T3-T5 - Clustering-based
        'NBUS',
        'KMeansUndersampling',
        
        # T6 - Comprehensive
        'OSM',
        
        # Baseline methods
        'RandomOverSampler',
        'RandomUnderSampler',
    ]


def validate_input_data(X: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and convert input data to proper format.
    
    Parameters
    ----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    
    Returns
    -------
    X : np.ndarray
        Validated feature matrix
    y : np.ndarray
        Validated target vector
    
    Raises
    ------
    ValueError
        If data is invalid
    """
    # Use sklearn's validation
    X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
    
    # Check for minimum requirements
    if len(X) < 10:
        raise ValueError(f"Dataset too small: {len(X)} samples. Need at least 10.")
    
    # Check class distribution
    unique_classes, counts = np.unique(y, return_counts=True)
    
    if len(unique_classes) < 2:
        raise ValueError(f"Need at least 2 classes, got {len(unique_classes)}")
    
    if len(unique_classes) > 2:
        raise ValueError(f"Currently only binary classification supported, got {len(unique_classes)} classes")
    
    # Check for minimum class sizes
    min_class_size = np.min(counts)
    if min_class_size < 2:
        raise ValueError(f"Minimum class has only {min_class_size} samples. Need at least 2.")
    
    return X, y


def get_technique_info(technique_name: str) -> dict:
    """
    Get information about a specific technique.
    
    Parameters
    ----------
    technique_name : str
        Name of the technique
    
    Returns
    -------
    info : dict
        Dictionary with technique information
    """
    technique_info = {
        # T1.x - Overlap-based undersampling
        'RFCL': {
            'full_name': 'Random Forest Cleaning Rule',
            'category': 'Overlap-based Undersampling',
            'type': 'Undersampling',
            'handles_overlap': True,
        },
        'URNS': {
            'full_name': 'Undersampling based on Recursive Neighbourhood Search',
            'category': 'Overlap-based Undersampling', 
            'type': 'Undersampling',
            'handles_overlap': True,
        },
        'NUS': {
            'full_name': 'Neighbourhood-based Under-Sampling',
            'category': 'Overlap-based Undersampling',
            'type': 'Undersampling', 
            'handles_overlap': True,
        },
        'DeviOCSVM': {
            'full_name': 'Devi et al. One-Class SVM',
            'category': 'Overlap-based Undersampling',
            'type': 'Undersampling',
            'handles_overlap': True,
        },
        'FCMBoostOBU': {
            'full_name': 'Fuzzy C-Means Boosted Overlap-Based Undersampling',
            'category': 'Overlap-based Undersampling',
            'type': 'Hybrid',
            'handles_overlap': True,
        },
        
        # T2.x - Hybrid methods
        'SVDDWSMOTE': {
            'full_name': 'SVDD-based Overlap Handler',
            'category': 'Hybrid Methods',
            'type': 'Hybrid',
            'handles_overlap': True,
        },
        'ODBOT': {
            'full_name': 'Outlier Detection-Based Oversampling Technique',
            'category': 'Hybrid Methods',
            'type': 'Oversampling',
            'handles_overlap': True,
        },
        'EHSO': {
            'full_name': 'Evolutionary Hybrid Sampling in Overlapping Scenarios',
            'category': 'Hybrid Methods',
            'type': 'Hybrid',
            'handles_overlap': True,
        },
        
        # T3-T5 - Clustering-based
        'NBUS': {
            'full_name': 'Neighbourhood-Based Undersampling',
            'category': 'Clustering-based',
            'type': 'Undersampling',
            'handles_overlap': True,
        },
        'KMeansUndersampling': {
            'full_name': 'K-Means Based Undersampling',
            'category': 'Clustering-based',
            'type': 'Undersampling',
            'handles_overlap': False,
        },
        
        # T6 - Comprehensive
        'OSM': {
            'full_name': 'Overlap-Separating Model',
            'category': 'Comprehensive',
            'type': 'Hybrid',
            'handles_overlap': True,
        },
        
        # Baseline methods
        'RandomOverSampler': {
            'full_name': 'Random Oversampling',
            'category': 'Baseline',
            'type': 'Oversampling',
            'handles_overlap': False,
        },
        'RandomUnderSampler': {
            'full_name': 'Random Undersampling',
            'category': 'Baseline',
            'type': 'Undersampling',
            'handles_overlap': False,
        },
    }
    
    return technique_info.get(technique_name, {
        'full_name': technique_name,
        'category': 'Unknown',
        'type': 'Unknown',
        'handles_overlap': False,
    })


def print_technique_summary():
    """
    Print a summary of all available techniques.
    """
    techniques = get_available_techniques()
    
    print("Available Resampling Techniques")
    print("=" * 50)
    
    categories = {}
    for tech in techniques:
        info = get_technique_info(tech)
        category = info['category']
        if category not in categories:
            categories[category] = []
        categories[category].append((tech, info))
    
    for category, techs in categories.items():
        print(f"\n{category}:")
        print("-" * len(category))
        for tech_name, info in techs:
            print(f"  • {tech_name}: {info['full_name']}")
            print(f"    Type: {info['type']}, Handles Overlap: {info['handles_overlap']}")
