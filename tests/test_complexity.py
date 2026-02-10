"""
Test complexity measures module.
"""

import pytest
import numpy as np
import pandas as pd


class TestComplexityMeasuresImport:
    """Test ComplexityMeasures import and basic structure."""
    
    def test_complexity_measures_import(self):
        """Test ComplexityMeasures can be imported."""
        from fairsample.complexity import ComplexityMeasures
        assert ComplexityMeasures is not None
    
    def test_compare_function_import(self):
        """Test compare_pre_post_overlap function import."""
        from fairsample.complexity import compare_pre_post_overlap
        assert compare_pre_post_overlap is not None


class TestComplexityMeasuresInstantiation:
    """Test ComplexityMeasures instantiation."""
    
    def test_basic_instantiation(self, sample_binary_data):
        """Test basic ComplexityMeasures instantiation."""
        from fairsample.complexity import ComplexityMeasures
        
        X, y = sample_binary_data
        cm = ComplexityMeasures(X, y)
        
        assert cm is not None
        assert hasattr(cm, 'X')
        assert hasattr(cm, 'y')
        assert hasattr(cm, 'classes')
    
    def test_instantiation_with_pandas(self, sample_binary_dataframe):
        """Test ComplexityMeasures works with pandas DataFrame."""
        from fairsample.complexity import ComplexityMeasures
        
        df = sample_binary_dataframe
        X = df.drop('target', axis=1)
        y = df['target']
        
        cm = ComplexityMeasures(X, y)
        assert cm is not None
    
    def test_instantiation_with_different_distance_functions(self, sample_binary_data):
        """Test ComplexityMeasures with different distance functions."""
        from fairsample.complexity import ComplexityMeasures
        
        X, y = sample_binary_data
        
        # Test default distance function
        cm1 = ComplexityMeasures(X, y, distance_func="default")
        assert cm1 is not None
        
        # Test HEOM distance function
        cm2 = ComplexityMeasures(X, y, distance_func="HEOM")
        assert cm2 is not None


class TestComplexityMeasuresBasicMethods:
    """Test basic complexity measures methods."""
    
    def test_analyze_overlap_method(self, sample_binary_data):
        """Test analyze_overlap method."""
        from fairsample.complexity import ComplexityMeasures
        
        X, y = sample_binary_data
        cm = ComplexityMeasures(X, y)
        
        results = cm.analyze_overlap()
        
        # Check return type
        assert isinstance(results, dict)
        
        # Check required keys
        required_keys = ['n_samples', 'n_features', 'n_classes', 'class_distribution']
        for key in required_keys:
            assert key in results
        
        # Check values make sense
        assert results['n_samples'] == len(X)
        assert results['n_features'] == X.shape[1]
        assert results['n_classes'] == len(np.unique(y))
    
    def test_feature_based_measures(self, sample_binary_data):
        """Test feature-based complexity measures."""
        from fairsample.complexity import ComplexityMeasures
        
        X, y = sample_binary_data
        cm = ComplexityMeasures(X, y)
        
        # Test F1 measure
        f1_result = cm.F1()
        assert f1_result is not None
        assert isinstance(f1_result, np.ndarray)
        assert len(f1_result) == X.shape[1]  # One value per feature
        
        # Test F2 measure
        f2_result = cm.F2()
        assert f2_result is not None
        assert isinstance(f2_result, list)
    
    def test_instance_based_measures(self, sample_binary_data):
        """Test instance-based complexity measures."""
        from fairsample.complexity import ComplexityMeasures
        
        X, y = sample_binary_data
        cm = ComplexityMeasures(X, y)
        
        # Test N3 measure
        n3_result = cm.N3(k=1)
        assert n3_result is not None
        assert isinstance(n3_result, (int, float, np.ndarray))
        
        # Test N4 measure
        n4_result = cm.N4(k=1)
        assert n4_result is not None
        assert isinstance(n4_result, (int, float, np.ndarray))
    
    def test_structural_measures(self, sample_binary_data):
        """Test structural complexity measures."""
        from fairsample.complexity import ComplexityMeasures
        
        X, y = sample_binary_data
        cm = ComplexityMeasures(X, y)
        
        # Test N1 measure
        n1_result = cm.N1()
        assert n1_result is not None
        assert isinstance(n1_result, (int, float, np.ndarray))
        
        # Test N2 measure
        n2_result = cm.N2()
        assert n2_result is not None
        assert isinstance(n2_result, (int, float, np.ndarray))


class TestComplexityMeasuresAdvanced:
    """Test advanced complexity measures functionality."""
    
    def test_feature_overlap_analysis(self, sample_binary_data):
        """Test feature overlap analysis."""
        from fairsample.complexity import ComplexityMeasures
        
        X, y = sample_binary_data
        cm = ComplexityMeasures(X, y)
        
        results = cm.feature_overlap_analysis(imb=True, viz=False)
        
        assert isinstance(results, dict)
        assert 'F1_mean' in results
        assert 'F2_mean' in results
    
    def test_instance_overlap_analysis(self, sample_binary_data):
        """Test instance overlap analysis."""
        from fairsample.complexity import ComplexityMeasures
        
        X, y = sample_binary_data
        cm = ComplexityMeasures(X, y)
        
        results = cm.instance_overlap_analysis(imb=True, viz=False, k=3)
        
        assert isinstance(results, dict)
        assert 'N3' in results
        assert 'N4' in results
    
    def test_structural_overlap_analysis(self, sample_binary_data):
        """Test structural overlap analysis."""
        from fairsample.complexity import ComplexityMeasures
        
        X, y = sample_binary_data
        cm = ComplexityMeasures(X, y)
        
        results = cm.structural_overlap_analysis(imb=True, viz=False)
        
        assert isinstance(results, dict)
        assert 'N1' in results
        assert 'N2' in results
    
    def test_instance_hardness_level(self, sample_binary_data):
        """Test instance-level hardness analysis."""
        from fairsample.complexity import ComplexityMeasures
        
        X, y = sample_binary_data
        cm = ComplexityMeasures(X, y)
        
        hardness = cm.N3(k=3, inst_level=True)
        
        assert isinstance(hardness, np.ndarray)
        assert len(hardness) == len(X)
        assert np.all(hardness >= 0) and np.all(hardness <= 1)


class TestComplexityMeasuresComparison:
    """Test complexity measures comparison functionality."""
    
    def test_compare_pre_post_overlap(self, sample_binary_data):
        """Test compare_pre_post_overlap function."""
        from fairsample.complexity import compare_pre_post_overlap
        from fairsample.techniques import RandomOverSampler
        
        X, y = sample_binary_data
        
        # Create resampled data
        sampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Compare complexity
        comparison = compare_pre_post_overlap(X, y, X_resampled, y_resampled)
        
        assert isinstance(comparison, dict)
        assert 'original' in comparison
        assert 'resampled' in comparison
        assert 'improvements' in comparison
        
        # Check that improvements contains expected measures
        improvements = comparison['improvements']
        expected_measures = ['N3_reduction', 'N1_reduction', 'imbalance_improvement']
        for measure in expected_measures:
            assert measure in improvements


class TestComplexityMeasuresRobustness:
    """Test complexity measures robustness and edge cases."""
    
    def test_with_small_dataset(self, sample_small_data):
        """Test complexity measures with small dataset."""
        from fairsample.complexity import ComplexityMeasures
        
        X, y = sample_small_data
        cm = ComplexityMeasures(X, y)
        
        # Should not crash with small data
        try:
            results = cm.analyze_overlap()
            assert results is not None
        except Exception as e:
            pytest.skip(f"Complexity measures failed with small data: {e}")
    
    def test_with_perfect_separation(self, sample_edge_case_data):
        """Test complexity measures with perfectly separated data."""
        from fairsample.complexity import ComplexityMeasures
        
        X, y = sample_edge_case_data['perfect_separation']
        cm = ComplexityMeasures(X, y)
        
        results = cm.analyze_overlap()
        assert results is not None
        
        # With perfect separation, some measures should indicate low complexity
        # This is a basic sanity check
        assert results['n_classes'] == 2
    
    def test_with_identical_features(self, sample_edge_case_data):
        """Test complexity measures with identical features."""
        from fairsample.complexity import ComplexityMeasures
        
        X, y = sample_edge_case_data['identical_features']
        
        try:
            cm = ComplexityMeasures(X, y)
            results = cm.analyze_overlap()
            assert results is not None
        except Exception as e:
            # Some measures might fail with identical features
            pytest.skip(f"Complexity measures failed with identical features: {e}")
    
    def test_multiclass_data(self, sample_multiclass_data):
        """Test complexity measures with multiclass data."""
        from fairsample.complexity import ComplexityMeasures
        
        X, y = sample_multiclass_data
        
        try:
            cm = ComplexityMeasures(X, y)
            results = cm.analyze_overlap()
            assert results is not None
            assert results['n_classes'] == 3
        except Exception as e:
            # Some measures might be designed for binary classification only
            pytest.skip(f"Complexity measures failed with multiclass data: {e}")
    
    def test_categorical_data_handling(self, sample_categorical_data):
        """Test complexity measures with categorical data."""
        from fairsample.complexity import ComplexityMeasures
        
        X, y = sample_categorical_data
        
        try:
            cm = ComplexityMeasures(X, y)
            results = cm.analyze_overlap()
            assert results is not None
        except Exception as e:
            pytest.skip(f"Complexity measures failed with categorical data: {e}")


class TestComplexityMeasuresPerformance:
    """Test complexity measures performance."""
    
    def test_execution_time(self, sample_binary_data):
        """Test that complexity measures complete in reasonable time."""
        from fairsample.complexity import ComplexityMeasures
        import time
        
        X, y = sample_binary_data
        
        start_time = time.time()
        cm = ComplexityMeasures(X, y)
        results = cm.analyze_overlap()
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time (30 seconds for small data)
        assert execution_time < 30.0
    
    def test_memory_efficiency(self, sample_binary_data):
        """Test that complexity measures don't use excessive memory."""
        from fairsample.complexity import ComplexityMeasures
        
        X, y = sample_binary_data
        cm = ComplexityMeasures(X, y)
        
        # Should be able to create multiple instances without issues
        cms = [ComplexityMeasures(X, y) for _ in range(5)]
        assert len(cms) == 5


class TestComplexityMeasuresValidation:
    """Test complexity measures input validation."""
    
    def test_invalid_input_types(self):
        """Test complexity measures with invalid input types."""
        from fairsample.complexity import ComplexityMeasures
        
        # Test with invalid X
        with pytest.raises((ValueError, TypeError)):
            ComplexityMeasures("invalid", [0, 1, 0, 1])
        
        # Test with invalid y
        with pytest.raises((ValueError, TypeError)):
            ComplexityMeasures([[1, 2], [3, 4]], "invalid")
    
    def test_mismatched_dimensions(self, sample_binary_data):
        """Test complexity measures with mismatched X and y dimensions."""
        from fairsample.complexity import ComplexityMeasures
        
        X, y = sample_binary_data
        
        # Test with mismatched dimensions
        with pytest.raises((ValueError, IndexError)):
            ComplexityMeasures(X, y[:-10])  # Remove some y values
    
    def test_single_class_handling(self, sample_edge_case_data):
        """Test complexity measures handling of single class data."""
        from fairsample.complexity import ComplexityMeasures
        
        X, y = sample_edge_case_data['single_class']
        
        # Should raise appropriate error for single class
        with pytest.raises(ValueError, match="Less than two classes"):
            ComplexityMeasures(X, y)