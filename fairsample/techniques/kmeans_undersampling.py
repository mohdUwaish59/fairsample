"""
K-Means based undersampling variants (placeholder).
"""

from .random_undersampler import RandomUnderSampler


class KMeansUndersampling(RandomUnderSampler):
    """
    K-Means based undersampling (placeholder implementation).
    
    Currently uses random undersampling as a placeholder.
    TODO: Implement the actual K-Means undersampling variants.
    """
    
    def __init__(self, variant='basic', random_state=None):
        super().__init__(sampling_strategy='auto', random_state=random_state)
        self.variant = variant
        self._sampling_type = 'undersampling'


class HKMUndersampling(KMeansUndersampling):
    """Hierarchical K-Means undersampling."""
    def __init__(self, random_state=None):
        super().__init__(variant='hkm', random_state=random_state)


class FCMUndersampling(KMeansUndersampling):
    """Fuzzy C-Means undersampling."""
    def __init__(self, random_state=None):
        super().__init__(variant='fcm', random_state=random_state)


class RKMUndersampling(KMeansUndersampling):
    """Rough K-Means undersampling."""
    def __init__(self, random_state=None):
        super().__init__(variant='rkm', random_state=random_state)


class FRKMUndersampling(KMeansUndersampling):
    """Fuzzy Rough K-Means undersampling."""
    def __init__(self, random_state=None):
        super().__init__(variant='frkm', random_state=random_state)