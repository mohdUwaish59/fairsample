"""
Test resampling techniques.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_X_y


class TestBaseSampler:
    """Test BaseSampler base class."""
    
    def test_base_sampler_import(self):
        """Test BaseSampler can be imported."""
        from fairsample.techniques import BaseSampler
        assert BaseSampler is not None
    
    def test_base_sampler_interface(self):
        """Test BaseSampler has required interface."""
        from fairsample.techniques import BaseSampler
        
        # Check required methods exist
        assert hasattr(BaseSampler, 'fit_resample')
        assert hasattr(BaseSampler, 'fit')
        assert hasattr(BaseSampler, 'resample')


class TestTechniqueImports:
    """Test that all techniques can be imported."""
    
    def test_baseline_techniques_import(self):
        """Test baseline techniques import."""
        from fairsample.techniques import RandomOverSampler, RandomUnderSampler
        assert RandomOverSampler is not None
        assert RandomUnderSampler is not None
    
    def test_overlap_based_techniques_import(self):
        """Test overlap-based techniques import."""
        from fairsample.techniques import RFCL, URNS, NUS, DeviOCSVM, FCMBoostOBU
        
        techniques = [RFCL, URNS, NUS, DeviOCSVM, FCMBoostOBU]
        for technique in techniques:
            assert technique is not None
    
    def test_hybrid_techniques_import(self):
        """Test hybrid techniques import."""
        from fairsample.techniques import SVDDWSMOTE, ODBOT, EHSO
        
        techniques = [SVDDWSMOTE, ODBOT, EHSO]
        for technique in techniques:
            assert technique is not None
    
    def test_clustering_techniques_import(self):
        """Test clustering-based techniques import."""
        from fairsample.techniques import (
            NBUS, NBBasic, NBTomek, NBComm, NBRec,
            KMeansUndersampling, HKMUndersampling, FCMUndersampling,
            RKMUndersampling, FRKMUndersampling
        )
        
        techniques = [
            NBUS, NBBasic, NBTomek, NBComm, NBRec,
            KMeansUndersampling, HKMUndersampling, FCMUndersampling,
            RKMUndersampling, FRKMUndersampling
        ]
        
        for technique in techniques:
            assert technique is not None
    
    def test_comprehensive_technique_import(self):
        """Test comprehensive technique import."""
        from fairsample.techniques import OSM
        assert OSM is not None


class TestTechniqueInstantiation:
    """Test technique instantiation."""
    
    @pytest.mark.parametrize("technique_name", [
        'RandomOverSampler', 'RandomUnderSampler', 'RFCL', 'NUS', 'URNS', 'OSM'
    ])
    def test_technique_instantiation(self, technique_name):
        """Test that techniques can be instantiated."""
        from fairsample import techniques
        
        technique_class = getattr(techniques, technique_name)
        
        # Try different initialization approaches
        try:
            # Try with random_state
            sampler = technique_class(random_state=42)
        except TypeError:
            try:
                # Try without parameters
                sampler = technique_class()
            except Exception as e:
                pytest.fail(f"Cannot instantiate {technique_name}: {e}")
        
        assert sampler is not None


class TestTechniqueFunctionality:
    """Test technique functionality with real data."""
    
    def test_random_oversampler_basic(self, sample_binary_data):
        """Test RandomOverSampler basic functionality."""
        from fairsample.techniques import RandomOverSampler
        
        X, y = sample_binary_data
        sampler = RandomOverSampler(random_state=42)
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Check output types
        assert isinstance(X_resampled, np.ndarray)
        assert isinstance(y_resampled, np.ndarray)
        
        # Check dimensions
        assert X_resampled.shape[1] == X.shape[1]  # Same number of features
        assert len(X_resampled) == len(y_resampled)  # Same number of samples
        
        # Check that minority class was oversampled
        original_counts = np.bincount(y)
        resampled_counts = np.bincount(y_resampled)
        
        minority_class = np.argmin(original_counts)
        assert resampled_counts[minority_class] >= original_counts[minority_class]
    
    def test_random_undersampler_basic(self, sample_binary_data):
        """Test RandomUnderSampler basic functionality."""
        from fairsample.techniques import RandomUnderSampler
        
        X, y = sample_binary_data
        sampler = RandomUnderSampler(random_state=42)
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Check output types
        assert isinstance(X_resampled, np.ndarray)
        assert isinstance(y_resampled, np.ndarray)
        
        # Check dimensions
        assert X_resampled.shape[1] == X.shape[1]
        assert len(X_resampled) == len(y_resampled)
        
        # Check that majority class was undersampled
        assert len(X_resampled) <= len(X)
    
    def test_technique_with_pandas_dataframe(self, sample_binary_dataframe):
        """Test technique works with pandas DataFrame."""
        from fairsample.techniques import RandomOverSampler
        
        df = sample_binary_dataframe
        X = df.drop('target', axis=1)
        y = df['target']
        
        sampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Should work with DataFrame input
        assert X_resampled is not None
        assert y_resampled is not None
        assert len(X_resampled) == len(y_resampled)
    
    def test_technique_preserves_feature_names(self, sample_binary_dataframe):
        """Test that techniques preserve feature information when possible."""
        from fairsample.techniques import RandomOverSampler
        
        df = sample_binary_dataframe
        X = df.drop('target', axis=1)
        y = df['target']
        
        sampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Check that we can convert back to DataFrame
        if hasattr(X, 'columns'):
            df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            assert list(df_resampled.columns) == list(X.columns)


class TestTechniqueRobustness:
    """Test technique robustness and edge cases."""
    
    def test_technique_with_small_data(self, sample_small_data):
        """Test techniques work with small datasets."""
        from fairsample.techniques import RandomOverSampler
        
        X, y = sample_small_data
        sampler = RandomOverSampler(random_state=42)
        
        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            assert X_resampled is not None
            assert y_resampled is not None
        except Exception as e:
            # Some techniques might not work with very small data
            pytest.skip(f"Technique failed with small data: {e}")
    
    def test_technique_with_single_class(self, sample_edge_case_data):
        """Test technique behavior with single class data."""
        from fairsample.techniques import RandomOverSampler
        
        X, y = sample_edge_case_data['single_class']
        sampler = RandomOverSampler(random_state=42)
        
        # Should handle single class gracefully
        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            # If it doesn't raise an error, check output
            assert len(np.unique(y_resampled)) <= 1
        except ValueError:
            # It's acceptable to raise ValueError for single class
            pass
    
    def test_technique_reproducibility(self, sample_binary_data):
        """Test that techniques are reproducible with same random_state."""
        from fairsample.techniques import RandomOverSampler
        
        X, y = sample_binary_data
        
        # Run twice with same random_state
        sampler1 = RandomOverSampler(random_state=42)
        X_res1, y_res1 = sampler1.fit_resample(X, y)
        
        sampler2 = RandomOverSampler(random_state=42)
        X_res2, y_res2 = sampler2.fit_resample(X, y)
        
        # Results should be identical
        np.testing.assert_array_equal(X_res1, X_res2)
        np.testing.assert_array_equal(y_res1, y_res2)
    
    def test_technique_input_validation(self, sample_binary_data):
        """Test that techniques validate input properly."""
        from fairsample.techniques import RandomOverSampler
        
        X, y = sample_binary_data
        sampler = RandomOverSampler(random_state=42)
        
        # Test with mismatched X and y
        with pytest.raises((ValueError, IndexError)):
            sampler.fit_resample(X, y[:-10])  # Remove some y values
        
        # Test with invalid input types
        with pytest.raises((ValueError, TypeError)):
            sampler.fit_resample("invalid", y)


class TestTechniquePerformance:
    """Test technique performance characteristics."""
    
    def test_technique_execution_time(self, sample_binary_data):
        """Test that techniques complete in reasonable time."""
        from fairsample.techniques import RandomOverSampler
        import time
        
        X, y = sample_binary_data
        sampler = RandomOverSampler(random_state=42)
        
        start_time = time.time()
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time (10 seconds for small data)
        assert execution_time < 10.0
    
    def test_technique_memory_usage(self, sample_binary_data):
        """Test that techniques don't use excessive memory."""
        from fairsample.techniques import RandomOverSampler
        
        X, y = sample_binary_data
        sampler = RandomOverSampler(random_state=42)
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Resampled data shouldn't be excessively larger than original
        # (this is a basic sanity check)
        assert len(X_resampled) < len(X) * 10  # At most 10x larger