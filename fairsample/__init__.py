"""
Imbalanced Learning Toolkit
===========================

A comprehensive toolkit for handling imbalanced datasets with class overlap.

This package provides state-of-the-art resampling techniques specifically designed
for imbalanced datasets with class overlap, along with complexity analysis tools
and high-level utility functions.

Main Components:
- techniques: Resampling techniques for imbalanced data
- complexity: Complexity measures for overlap analysis
- utils: High-level convenience functions

Example:
    >>> from fairsample import RFCL, ComplexityMeasures
    >>> sampler = RFCL(random_state=42)
    >>> X_resampled, y_resampled = sampler.fit_resample(X, y)
    >>> cm = ComplexityMeasures(X, y)
    >>> complexity_results = cm.analyze_overlap()
"""

from .__version__ import __version__, __author__, __email__, __description__, __url__

# Import all techniques for easy access
try:
    from .techniques import (
        # Base class
        BaseSampler,
        
        # Overlap-based undersampling (T1.x)
        RFCL,
        URNS,
        NUS,
        DeviOCSVM,
        FCMBoostOBU,
        
        # Hybrid methods (T2.x)
        SVDDWSMOTE,
        ODBOT,
        EHSO,
        
        # Clustering-based (T3-T5)
        NBUS,
        NBBasic,
        NBTomek,
        NBComm,
        NBRec,
        KMeansUndersampling,
        HKMUndersampling,
        FCMUndersampling,
        RKMUndersampling,
        FRKMUndersampling,
        
        # Comprehensive (T6)
        OSM,
        
        # Baseline methods
        RandomOverSampler,
        RandomUnderSampler,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import techniques: {e}")

# Import complexity measures
try:
    from .complexity import ComplexityMeasures, compare_pre_post_overlap
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import complexity measures: {e}")

# Import high-level utilities
try:
    from .utils import (
        compare_techniques,
        get_resampled_data,
        get_available_techniques,
        validate_input_data
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import utility functions: {e}")

# Define what gets imported with "from fairsample import *"
__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',
    '__description__',
    '__url__',
    
    # Base class
    'BaseSampler',
    
    # Techniques (T1.x - Overlap-based undersampling)
    'RFCL',
    'URNS', 
    'NUS',
    'DeviOCSVM',
    'FCMBoostOBU',
    
    # Techniques (T2.x - Hybrid methods)
    'SVDDWSMOTE',
    'ODBOT',
    'EHSO',
    
    # Techniques (T3-T5 - Clustering-based)
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
    
    # Techniques (T6 - Comprehensive)
    'OSM',
    
    # Baseline techniques
    'RandomOverSampler',
    'RandomUnderSampler',
    
    # Complexity measures
    'ComplexityMeasures',
    'compare_pre_post_overlap',
    
    # High-level utilities
    'compare_techniques',
    'get_resampled_data',
    'get_available_techniques',
    'validate_input_data',
]

# Package metadata
__title__ = 'fairsample'
__license__ = 'MIT'
__copyright__ = f'2024 {__author__}'