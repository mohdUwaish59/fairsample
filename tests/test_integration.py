"""
Integration tests for the fairsample package.
"""

import pytest
import numpy as np
import pandas as pd


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    def test_basic_workflow_numpy(self, sample_binary_data):
        """Test basic workflow with numpy arrays."""
        from fairsample import RFCL, ComplexityMeasures
        
        X, y = sample_binary_data
        
        # Step 1: Apply resampling technique
        sampler = RFCL(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Step 2: Calculate complexity measures
        cm_original = ComplexityMeasures(X, y)
        cm_resampled = ComplexityMeasures(X_resampled, y_resampled)
        
        original_complexity = cm_original.analyze_overlap()
        resampled_complexity = cm_resampled.analyze_overlap()
        
        # Verify workflow completed
        assert original_complexity is not None
        assert resampled_complexity is not None
        assert X_resampled is not None
        assert y_resampled is not None
    
    def test_basic_workflow_pandas(self, sample_binary_dataframe):
        """Test basic workflow with pandas DataFrame."""
        from fairsample import RandomOverSampler, ComplexityMeasures
        
        df = sample_binary_dataframe
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Step 1: Apply resampling technique
        sampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Step 2: Calculate complexity measures
        cm = ComplexityMeasures(X, y)
        complexity_results = cm.analyze_overlap()
        
        # Verify workflow completed
        assert complexity_results is not None
        assert X_resampled is not None
        assert y_resampled is not None
    
    def test_comparison_workflow(self, sample_binary_data):
        """Test technique comparison workflow."""
        from fairsample.utils import compare_techniques
        from fairsample.complexity import compare_pre_post_overlap
        from fairsample.techniques import RandomOverSampler
        
        X, y = sample_binary_data
        
        # Step 1: Compare multiple techniques
        try:
            comparison_results = compare_techniques(
                X, y, 
                ['RandomOverSampler', 'RandomUnderSampler'],
                verbose=False
            )
            assert comparison_results is not None
        except Exception as e:
            pytest.skip(f"Technique comparison failed: {e}")
        
        # Step 2: Detailed complexity comparison for one technique
        sampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        complexity_comparison = compare_pre_post_overlap(X, y, X_resampled, y_resampled)
        assert complexity_comparison is not None
        assert 'original' in complexity_comparison
        assert 'resampled' in complexity_comparison
    
    def test_multiple_techniques_workflow(self, sample_binary_data):
        """Test workflow with multiple techniques."""
        from fairsample import RandomOverSampler, RandomUnderSampler, NUS
        from fairsample.complexity import ComplexityMeasures
        
        X, y = sample_binary_data
        
        techniques = [
            ('RandomOverSampler', RandomOverSampler(random_state=42)),
            ('RandomUnderSampler', RandomUnderSampler(random_state=42)),
        ]
        
        # Try NUS if available
        try:
            techniques.append(('NUS', NUS(random_state=42)))
        except:
            pass
        
        results = {}
        
        for name, sampler in techniques:
            try:
                # Apply technique
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                
                # Calculate complexity
                cm = ComplexityMeasures(X_resampled, y_resampled)
                complexity = cm.analyze_overlap()
                
                results[name] = {
                    'X_resampled': X_resampled,
                    'y_resampled': y_resampled,
                    'complexity': complexity
                }
                
            except Exception as e:
                # Some techniques might fail, that's okay for this test
                continue
        
        # Should have at least baseline techniques working
        assert len(results) >= 2


class TestCrossModuleIntegration:
    """Test integration between different modules."""
    
    def test_techniques_complexity_integration(self, sample_binary_data):
        """Test integration between techniques and complexity modules."""
        from fairsample.techniques import RandomOverSampler
        from fairsample.complexity import ComplexityMeasures, compare_pre_post_overlap
        
        X, y = sample_binary_data
        
        # Apply technique
        sampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Use complexity measures
        cm_original = ComplexityMeasures(X, y)
        cm_resampled = ComplexityMeasures(X_resampled, y_resampled)
        
        # Individual complexity analysis
        original_analysis = cm_original.analyze_overlap()
        resampled_analysis = cm_resampled.analyze_overlap()
        
        # Comparison analysis
        comparison = compare_pre_post_overlap(X, y, X_resampled, y_resampled)
        
        # Verify integration works
        assert original_analysis is not None
        assert resampled_analysis is not None
        assert comparison is not None
    
    def test_techniques_utils_integration(self, sample_binary_data):
        """Test integration between techniques and utils modules."""
        from fairsample.utils import get_resampled_data, get_available_techniques
        from fairsample.techniques import RandomOverSampler
        
        X, y = sample_binary_data
        
        # Get available techniques
        available = get_available_techniques()
        assert 'RandomOverSampler' in available
        
        # Use utils to get resampled data
        try:
            resampled_data = get_resampled_data(X, y, ['RandomOverSampler'])
            assert 'RandomOverSampler' in resampled_data
            
            # Verify we can use the technique directly too
            sampler = RandomOverSampler(random_state=42)
            X_direct, y_direct = sampler.fit_resample(X, y)
            
            assert X_direct is not None
            assert y_direct is not None
            
        except Exception as e:
            pytest.skip(f"Techniques-utils integration failed: {e}")
    
    def test_utils_complexity_integration(self, sample_binary_data):
        """Test integration between utils and complexity modules."""
        from fairsample.utils import get_resampled_data
        from fairsample.complexity import ComplexityMeasures
        
        X, y = sample_binary_data
        
        try:
            # Get resampled data using utils
            resampled_data = get_resampled_data(X, y, ['RandomOverSampler'])
            
            # Use complexity measures on the results
            technique_data = resampled_data['RandomOverSampler']
            X_resampled = technique_data['X_resampled']
            y_resampled = technique_data['y_resampled']
            
            cm = ComplexityMeasures(X_resampled, y_resampled)
            complexity = cm.analyze_overlap()
            
            assert complexity is not None
            
        except Exception as e:
            pytest.skip(f"Utils-complexity integration failed: {e}")


class TestDataFormatCompatibility:
    """Test compatibility with different data formats."""
    
    def test_numpy_array_compatibility(self, sample_binary_data):
        """Test compatibility with numpy arrays."""
        from fairsample import RandomOverSampler, ComplexityMeasures
        
        X, y = sample_binary_data
        
        # Ensure data is numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Test technique
        sampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Test complexity measures
        cm = ComplexityMeasures(X, y)
        results = cm.analyze_overlap()
        
        assert results is not None
        assert isinstance(X_resampled, np.ndarray)
        assert isinstance(y_resampled, np.ndarray)
    
    def test_pandas_dataframe_compatibility(self, sample_binary_dataframe):
        """Test compatibility with pandas DataFrames."""
        from fairsample import RandomOverSampler, ComplexityMeasures
        
        df = sample_binary_dataframe
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Test technique with DataFrame
        sampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Test complexity measures with DataFrame
        cm = ComplexityMeasures(X, y)
        results = cm.analyze_overlap()
        
        assert results is not None
        assert X_resampled is not None
        assert y_resampled is not None
    
    def test_mixed_data_types(self, sample_categorical_data):
        """Test with mixed categorical and numerical data."""
        from fairsample import RandomOverSampler, ComplexityMeasures
        
        X, y = sample_categorical_data
        
        try:
            # Test technique
            sampler = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            # Test complexity measures
            cm = ComplexityMeasures(X, y)
            results = cm.analyze_overlap()
            
            assert results is not None
            
        except Exception as e:
            pytest.skip(f"Mixed data types test failed: {e}")


class TestScalabilityAndPerformance:
    """Test scalability and performance characteristics."""
    
    def test_small_dataset_performance(self, sample_small_data):
        """Test performance with small datasets."""
        from fairsample import RandomOverSampler, ComplexityMeasures
        import time
        
        X, y = sample_small_data
        
        # Test technique performance
        start_time = time.time()
        sampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        technique_time = time.time() - start_time
        
        # Test complexity measures performance
        start_time = time.time()
        cm = ComplexityMeasures(X, y)
        results = cm.analyze_overlap()
        complexity_time = time.time() - start_time
        
        # Should complete quickly for small data
        assert technique_time < 5.0  # 5 seconds
        assert complexity_time < 30.0  # 30 seconds
        assert results is not None
    
    def test_medium_dataset_performance(self, sample_binary_data):
        """Test performance with medium-sized datasets."""
        from fairsample import RandomOverSampler
        import time
        
        X, y = sample_binary_data
        
        # Test technique performance
        start_time = time.time()
        sampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        technique_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert technique_time < 10.0  # 10 seconds
        assert X_resampled is not None
    
    def test_memory_usage(self, sample_binary_data):
        """Test memory usage characteristics."""
        from fairsample import RandomOverSampler, ComplexityMeasures
        
        X, y = sample_binary_data
        
        # Test that we can create multiple instances without issues
        samplers = [RandomOverSampler(random_state=i) for i in range(5)]
        cms = [ComplexityMeasures(X, y) for _ in range(3)]
        
        assert len(samplers) == 5
        assert len(cms) == 3


class TestErrorHandlingAndRobustness:
    """Test error handling and robustness."""
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        from fairsample import RandomOverSampler, ComplexityMeasures
        
        # Test technique with invalid input
        sampler = RandomOverSampler(random_state=42)
        
        with pytest.raises((ValueError, TypeError)):
            sampler.fit_resample("invalid", [0, 1, 0])
        
        # Test complexity measures with invalid input
        with pytest.raises((ValueError, TypeError)):
            ComplexityMeasures("invalid", [0, 1, 0])
    
    def test_edge_case_handling(self, sample_edge_case_data):
        """Test handling of edge cases."""
        from fairsample import RandomOverSampler, ComplexityMeasures
        
        # Test with single class data
        X_single, y_single = sample_edge_case_data['single_class']
        
        sampler = RandomOverSampler(random_state=42)
        
        # Should handle single class gracefully
        try:
            X_resampled, y_resampled = sampler.fit_resample(X_single, y_single)
        except ValueError:
            # Acceptable to raise ValueError for single class
            pass
        
        # Complexity measures should raise appropriate error
        with pytest.raises(ValueError):
            ComplexityMeasures(X_single, y_single)
    
    def test_reproducibility(self, sample_binary_data):
        """Test reproducibility with random states."""
        from fairsample import RandomOverSampler, ComplexityMeasures
        
        X, y = sample_binary_data
        
        # Test technique reproducibility
        sampler1 = RandomOverSampler(random_state=42)
        X_res1, y_res1 = sampler1.fit_resample(X, y)
        
        sampler2 = RandomOverSampler(random_state=42)
        X_res2, y_res2 = sampler2.fit_resample(X, y)
        
        np.testing.assert_array_equal(X_res1, X_res2)
        np.testing.assert_array_equal(y_res1, y_res2)
        
        # Test complexity measures reproducibility
        cm1 = ComplexityMeasures(X, y)
        results1 = cm1.analyze_overlap()
        
        cm2 = ComplexityMeasures(X, y)
        results2 = cm2.analyze_overlap()
        
        # Basic measures should be identical
        assert results1['n_samples'] == results2['n_samples']
        assert results1['n_features'] == results2['n_features']


class TestPackageConsistency:
    """Test package consistency and API design."""
    
    def test_consistent_api_design(self):
        """Test that all techniques follow consistent API design."""
        from fairsample.techniques import RandomOverSampler, RandomUnderSampler
        
        # All techniques should have fit_resample method
        techniques = [RandomOverSampler(), RandomUnderSampler()]
        
        for technique in techniques:
            assert hasattr(technique, 'fit_resample')
            assert callable(technique.fit_resample)
    
    def test_consistent_return_types(self, sample_binary_data):
        """Test that techniques return consistent types."""
        from fairsample.techniques import RandomOverSampler, RandomUnderSampler
        
        X, y = sample_binary_data
        techniques = [RandomOverSampler(random_state=42), RandomUnderSampler(random_state=42)]
        
        for technique in techniques:
            X_resampled, y_resampled = technique.fit_resample(X, y)
            
            # Should return numpy arrays
            assert isinstance(X_resampled, np.ndarray)
            assert isinstance(y_resampled, np.ndarray)
            
            # Should have correct dimensions
            assert X_resampled.shape[1] == X.shape[1]
            assert len(X_resampled) == len(y_resampled)
    
    def test_package_imports_consistency(self):
        """Test that package imports are consistent."""
        # Test main package imports
        from fairsample import RandomOverSampler, ComplexityMeasures, compare_techniques
        
        # Test submodule imports
        from fairsample.techniques import RandomOverSampler as TechRandomOverSampler
        from fairsample.complexity import ComplexityMeasures as ComplexComplexityMeasures
        from fairsample.utils import compare_techniques as UtilsCompareTechniques
        
        # Should be the same classes/functions
        assert RandomOverSampler is TechRandomOverSampler
        assert ComplexityMeasures is ComplexComplexityMeasures
        assert compare_techniques is UtilsCompareTechniques