"""
Resampling Techniques for Imbalanced Data
=========================================

This module contains various resampling techniques specifically designed
for handling imbalanced datasets with class overlap.
"""

# Import base class
from .base_sampler import BaseSampler

# Import overlap-based undersampling techniques (T1.x)
from .rfcl import RFCL
from .urns import URNS
from .nus import NUS
from .devi_ocsvm import DeviOCSVM
from .fcm_boost_obu import FCMBoostOBU

# Import hybrid methods (T2.x)
from .svddwsmote import SVDDWSMOTE
from .odbot import ODBOT
from .ehso import EHSO

# Import clustering-based methods (T3-T5)
from .nbus import NBUS, NBBasic, NBTomek, NBComm, NBRec
from .kmeans_undersampling import (
    KMeansUndersampling, HKMUndersampling, FCMUndersampling,
    RKMUndersampling, FRKMUndersampling
)

# Import comprehensive method (T6)
from .osm import OSM

# Import baseline methods
from .random_oversampler import RandomOverSampler
from .random_undersampler import RandomUnderSampler

__all__ = [
    # Base class
    'BaseSampler',
    
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
    'NBBasic',
    'NBTomek', 
    'NBComm',
    'NBRec',
    'KMeansUndersampling',
    'HKMUndersampling',
    'FCMUndersampling',
    'RKMUndersampling',
    'FRKMUndersampling',
    
    # T6 - Comprehensive
    'OSM',
    
    # Baseline methods
    'RandomOverSampler',
    'RandomUnderSampler',
]