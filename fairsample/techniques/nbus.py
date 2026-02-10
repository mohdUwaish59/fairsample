"""
Neighbourhood-Based Undersampling variants (placeholder).
"""

from .nus import NUS


class NBUS(NUS):
    """
    Neighbourhood-Based Undersampling (placeholder implementation).
    
    Currently uses NUS as a placeholder.
    TODO: Implement the actual NBUS variants.
    """
    
    def __init__(self, variant='basic', random_state=None):
        super().__init__(n_neighbors=3, random_state=random_state)
        self.variant = variant
        self._sampling_type = 'undersampling'


class NBBasic(NBUS):
    """Basic NBUS variant."""
    def __init__(self, random_state=None):
        super().__init__(variant='basic', random_state=random_state)


class NBTomek(NBUS):
    """NBUS with Tomek links."""
    def __init__(self, random_state=None):
        super().__init__(variant='tomek', random_state=random_state)


class NBComm(NBUS):
    """NBUS with common neighbors."""
    def __init__(self, random_state=None):
        super().__init__(variant='common', random_state=random_state)


class NBRec(NBUS):
    """Recursive NBUS."""
    def __init__(self, random_state=None):
        super().__init__(variant='recursive', random_state=random_state)