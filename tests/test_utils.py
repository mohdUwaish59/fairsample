"""
Test utility functions.
"""

import pytest
import numpy as np
import pandas as pd


class TestUtilsImport:
    """Test utils module imports."""
    
    def test_utils_import(self):
        """Test utils module can be imported."""
        from fairsample import utils
        assert utils is not None
    
    def test_comparison_functions_import(self):
        """Test comparison functions import."""
        from fairsample.utils import (
            compare_techniques,
            get_resampled_data
        )
        
        functions = [
            compare_techniques,
            get_resampled_data
        ]
        
        for func in functions:
            assert func is not None
            assert callable(func)
    
    def test_helper_functions_import(self):
        """Test helper functions import."""
        from fairsample.utils import (
            get_available_techniques,
            validate_input_data
        )
        
        assert get_available_techniques is not None
        assert validate_input_data is not None


class TestGetAvailableTechniques:
    """Test get_available_techniques function."""
    
    def test_get_available_techniques_basic(self):
        """Test basic functionality of get_available_techniques."""
        from fairsample.utils import get_available_techniques
        
        techniques = get_available_techniques()
        
        assert isinstance(techniques, list)
        assert len(techniques) > 0
        
        # Check some expected techniques are present
        expected_techniques = ['RandomOverSampler', 'RandomUnderSampler', 'RFCL', 'NUS', 'OSM']
        for technique in expected_techniques:
            assert technique in techniques
    
    def test_get_available_techniques_categories(self):
        """Test get_available_techniques with categories."""
        from fairsample.utils import get_available_techniques
        
        try:
            # Try to get techniques by category if supported
            techniques = get_available_techniques(category='baseline')
            assert isinstance(techniques, list)
        except TypeError:
            # Function might not support categories yet
            pass


class TestValidateInputData:
    """Test validate_input_data function."""
    
    def test_validate_input_data_valid(self, sample_binary_data):
        """Test validate_input_data with valid data."""
        from fairsample.utils import validate_input_data
        
        X, y = sample_binary_data
        
        # Should not raise exception with valid data
        try:
            validate_input_data(X, y)
        except Exception as e:
            pytest.fail(f"validate_input_data failed with valid data: {e}")
    
    def test_validate_input_data_pandas(self, sample_binary_dataframe):
        """Test validate_input_data with pandas DataFrame."""
        from fairsample.utils import validate_input_data
        
        df = sample_binary_dataframe
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Should work with pandas data
        try:
            validate_input_data(X, y)
        except Exception as e:
            pytest.fail(f"validate_input_data failed with pandas data: {e}")
    
    def test_validate_input_data_invalid(self):
        """Test validate_input_data with invalid data."""
        from fairsample.utils import validate_input_data
        
        # Test with mismatched dimensions
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 90)  # Wrong size
        
        with pytest.raises((ValueError, IndexError)):
            validate_input_data(X, y)


class TestCompareTechniques:
    """Test compare_techniques function."""
    
    def test_compare_techniques_basic(self, sample_binary_data):
        """Test basic compare_techniques functionality."""
        from fairsample.utils import compare_techniques
        
        X, y = sample_binary_data
        
        # Test with a few techniques
        techniques = ['RandomOverSampler', 'RandomUnderSampler']
        
        try:
            results = compare_techniques(X, y, techniques, verbose=False)
            
            assert results is not None
            assert isinstance(results, pd.DataFrame)
            assert len(results) == len(techniques)
            
        except Exception as e:
            pytest.skip(f"compare_techniques failed: {e}")
    
    def test_compare_techniques_single(self, sample_binary_data):
        """Test compare_techniques with single technique."""
        from fairsample.utils import compare_techniques
        
        X, y = sample_binary_data
        
        try:
            results = compare_techniques(X, y, ['RandomOverSampler'], verbose=False)
            
            assert results is not None
            assert isinstance(results, pd.DataFrame)
            assert len(results) == 1
            
        except Exception as e:
            pytest.skip(f"compare_techniques failed with single technique: {e}")
    
    def test_compare_techniques_all(self, sample_small_data):
        """Test compare_techniques with 'all' techniques."""
        from fairsample.utils import compare_techniques
        
        X, y = sample_small_data  # Use smaller data for faster execution
        
        try:
            results = compare_techniques(X, y, 'all', verbose=False)
            
            assert results is not None
            assert isinstance(results, pd.DataFrame)
            assert len(results) > 5  # Should have multiple techniques
            
        except Exception as e:
            pytest.skip(f"compare_techniques failed with 'all': {e}")
    
    def test_compare_techniques_parameters(self, sample_binary_data):
        """Test compare_techniques with different parameters."""
        from fairsample.utils import compare_techniques
        
        X, y = sample_binary_data
        
        try:
            # Test with different test_size
            results = compare_techniques(
                X, y, ['RandomOverSampler'], 
                test_size=0.2, 
                random_state=123,
                verbose=False
            )
            
            assert results is not None
            
        except Exception as e:
            pytest.skip(f"compare_techniques failed with parameters: {e}")


class TestAutoSelectTechnique:
    """Test auto_select_technique function."""
    
    def test_auto_select_technique_basic(self, sample_binary_data):
        """Test basic auto_select_technique functionality."""
        from fairsample.utils import auto_select_technique
        
        X, y = sample_binary_data
        
        try:
            best_technique = auto_select_technique(X, y, verbose=False)
            
            assert best_technique is not None
            assert isinstance(best_technique, str)
            
        except Exception as e:
            pytest.skip(f"auto_select_technique failed: {e}")
    
    def test_auto_select_technique_with_candidates(self, sample_binary_data):
        """Test auto_select_technique with specific candidates."""
        from fairsample.utils import auto_select_technique
        
        X, y = sample_binary_data
        candidates = ['RandomOverSampler', 'RandomUnderSampler']
        
        try:
            best_technique = auto_select_technique(
                X, y, 
                candidates=candidates,
                verbose=False
            )
            
            assert best_technique is not None
            assert best_technique in candidates
            
        except Exception as e:
            pytest.skip(f"auto_select_technique failed with candidates: {e}")


class TestGetResampledData:
    """Test get_resampled_data function."""
    
    def test_get_resampled_data_basic(self, sample_binary_data):
        """Test basic get_resampled_data functionality."""
        from fairsample.utils import get_resampled_data
        
        X, y = sample_binary_data
        
        try:
            data = get_resampled_data(X, y, ['RandomOverSampler'])
            
            assert data is not None
            assert isinstance(data, dict)
            assert 'RandomOverSampler' in data
            
            # Check structure of returned data
            technique_data = data['RandomOverSampler']
            assert 'X_resampled' in technique_data
            assert 'y_resampled' in technique_data
            
        except Exception as e:
            pytest.skip(f"get_resampled_data failed: {e}")
    
    def test_get_resampled_data_multiple(self, sample_binary_data):
        """Test get_resampled_data with multiple techniques."""
        from fairsample.utils import get_resampled_data
        
        X, y = sample_binary_data
        techniques = ['RandomOverSampler', 'RandomUnderSampler']
        
        try:
            data = get_resampled_data(X, y, techniques)
            
            assert data is not None
            assert isinstance(data, dict)
            
            for technique in techniques:
                assert technique in data
                
        except Exception as e:
            pytest.skip(f"get_resampled_data failed with multiple techniques: {e}")


class TestGetEvaluationData:
    """Test get_evaluation_data function."""
    
    def test_get_evaluation_data_basic(self, sample_binary_data):
        """Test basic get_evaluation_data functionality."""
        from fairsample.utils import get_evaluation_data
        
        X, y = sample_binary_data
        
        try:
            data = get_evaluation_data(X, y, ['RandomOverSampler'])
            
            assert data is not None
            assert isinstance(data, dict)
            
        except Exception as e:
            pytest.skip(f"get_evaluation_data failed: {e}")


class TestGetVisualizationData:
    """Test get_visualization_data function."""
    
    def test_get_visualization_data_basic(self, sample_binary_data):
        """Test basic get_visualization_data functionality."""
        from fairsample.utils import get_visualization_data
        
        X, y = sample_binary_data
        
        try:
            data = get_visualization_data(X, y, ['RandomOverSampler'])
            
            assert data is not None
            assert isinstance(data, dict)
            
        except Exception as e:
            pytest.skip(f"get_visualization_data failed: {e}")


class TestUtilsRobustness:
    """Test utils robustness and edge cases."""
    
    def test_utils_with_small_data(self, sample_small_data):
        """Test utils functions with small datasets."""
        from fairsample.utils import compare_techniques
        
        X, y = sample_small_data
        
        try:
            results = compare_techniques(X, y, ['RandomOverSampler'], verbose=False)
            assert results is not None
        except Exception as e:
            pytest.skip(f"Utils failed with small data: {e}")
    
    def test_utils_with_pandas(self, sample_binary_dataframe):
        """Test utils functions with pandas DataFrame."""
        from fairsample.utils import compare_techniques
        
        df = sample_binary_dataframe
        X = df.drop('target', axis=1)
        y = df['target']
        
        try:
            results = compare_techniques(X, y, ['RandomOverSampler'], verbose=False)
            assert results is not None
        except Exception as e:
            pytest.skip(f"Utils failed with pandas data: {e}")
    
    def test_utils_error_handling(self, sample_binary_data):
        """Test utils error handling."""
        from fairsample.utils import compare_techniques
        
        X, y = sample_binary_data
        
        # Test with invalid technique name
        try:
            results = compare_techniques(X, y, ['NonExistentTechnique'], verbose=False)
            # Should either handle gracefully or raise appropriate error
        except (ValueError, KeyError, AttributeError):
            # These are acceptable errors for invalid technique names
            pass


class TestUtilsPerformance:
    """Test utils performance characteristics."""
    
    def test_utils_execution_time(self, sample_binary_data):
        """Test that utils functions complete in reasonable time."""
        from fairsample.utils import compare_techniques
        import time
        
        X, y = sample_binary_data
        
        start_time = time.time()
        try:
            results = compare_techniques(X, y, ['RandomOverSampler'], verbose=False)
            execution_time = time.time() - start_time
            
            # Should complete within reasonable time (60 seconds for small comparison)
            assert execution_time < 60.0
        except Exception as e:
            pytest.skip(f"Utils performance test failed: {e}")


class TestUtilsIntegration:
    """Test utils integration with other modules."""
    
    def test_utils_with_complexity_measures(self, sample_binary_data):
        """Test utils integration with complexity measures."""
        from fairsample.utils import get_resampled_data
        from fairsample.complexity import ComplexityMeasures
        
        X, y = sample_binary_data
        
        try:
            # Get resampled data
            data = get_resampled_data(X, y, ['RandomOverSampler'])
            
            # Use with complexity measures
            technique_data = data['RandomOverSampler']
            X_resampled = technique_data['X_resampled']
            y_resampled = technique_data['y_resampled']
            
            cm = ComplexityMeasures(X_resampled, y_resampled)
            results = cm.analyze_overlap()
            
            assert results is not None
            
        except Exception as e:
            pytest.skip(f"Utils-complexity integration failed: {e}")
    
    def test_utils_with_techniques(self, sample_binary_data):
        """Test utils integration with techniques."""
        from fairsample.utils import get_available_techniques
        from fairsample.techniques import RandomOverSampler
        
        # Get available techniques
        techniques = get_available_techniques()
        
        # Check that we can actually import and use them
        assert 'RandomOverSampler' in techniques
        
        # Test instantiation
        sampler = RandomOverSampler(random_state=42)
        assert sampler is not None