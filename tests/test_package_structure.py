"""
Test package structure and imports.
"""

import pytest
import importlib
import sys
import os


class TestPackageStructure:
    """Test package structure and basic imports."""
    
    def test_package_importable(self):
        """Test that the main package can be imported."""
        try:
            import fairsample
            assert hasattr(fairsample, '__version__')
        except ImportError as e:
            pytest.fail(f"Cannot import fairsample: {e}")
    
    def test_version_info(self):
        """Test that version information is available."""
        import fairsample
        
        # Check version attributes exist
        assert hasattr(fairsample, '__version__')
        assert hasattr(fairsample, '__author__')
        assert hasattr(fairsample, '__description__')
        
        # Check version format
        version = fairsample.__version__
        assert isinstance(version, str)
        assert len(version.split('.')) >= 2  # At least major.minor
    
    def test_core_modules_exist(self):
        """Test that core modules exist."""
        core_modules = [
            'fairsample.techniques',
            'fairsample.complexity',
            'fairsample.utils'
        ]
        
        for module_name in core_modules:
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                pytest.fail(f"Cannot import core module {module_name}: {e}")
    
    def test_techniques_module_structure(self):
        """Test techniques module structure."""
        from fairsample import techniques
        
        # Check __all__ exists and is not empty
        assert hasattr(techniques, '__all__')
        assert len(techniques.__all__) > 0
        
        # Check BaseSampler exists
        assert hasattr(techniques, 'BaseSampler')
        
        # Check some key techniques exist
        key_techniques = ['RFCL', 'NUS', 'URNS', 'OSM', 'RandomOverSampler']
        for technique in key_techniques:
            assert hasattr(techniques, technique), f"Missing technique: {technique}"
    
    def test_complexity_module_structure(self):
        """Test complexity module structure."""
        from fairsample import complexity
        
        # Check ComplexityMeasures exists
        assert hasattr(complexity, 'ComplexityMeasures')
        assert hasattr(complexity, 'compare_pre_post_overlap')
    
    def test_utils_module_structure(self):
        """Test utils module structure."""
        from fairsample import utils
        
        # Check key utility functions exist
        key_functions = [
            'compare_techniques',
            'auto_select_technique',
            'get_resampled_data'
        ]
        
        for func in key_functions:
            assert hasattr(utils, func), f"Missing utility function: {func}"
    
    def test_main_package_exports(self):
        """Test that main package exports key components."""
        import fairsample
        
        # Check key techniques are available at package level
        key_exports = [
            'RFCL', 'NUS', 'URNS', 'OSM',
            'ComplexityMeasures',
            'compare_techniques'
        ]
        
        for export in key_exports:
            assert hasattr(fairsample, export), f"Missing export: {export}"
    
    def test_no_test_modules_in_main_package(self):
        """Test that testing modules are not in main package."""
        import fairsample
        
        # These should NOT be in main package after restructuring
        test_modules = ['evaluation', 'visualization', 'data']
        
        for module in test_modules:
            assert not hasattr(fairsample, module), \
                f"Test module {module} should not be in main package"
    
    def test_package_directory_structure(self):
        """Test that package directory structure is correct."""
        import fairsample
        package_dir = os.path.dirname(fairsample.__file__)
        
        # Required directories
        required_dirs = ['techniques', 'complexity', 'utils']
        for dir_name in required_dirs:
            dir_path = os.path.join(package_dir, dir_name)
            assert os.path.isdir(dir_path), f"Missing directory: {dir_name}"
        
        # Should not have these directories (moved to test/)
        excluded_dirs = ['evaluation', 'visualization', 'data']
        for dir_name in excluded_dirs:
            dir_path = os.path.join(package_dir, dir_name)
            assert not os.path.isdir(dir_path), f"Directory {dir_name} should be moved to test/"
    
    def test_test_directory_exists(self):
        """Test that test directory exists with moved modules."""
        # Check if test directory exists
        test_dir = 'test'
        if os.path.exists(test_dir):
            # Test modules should be in test directory
            test_modules = ['evaluation', 'visualization', 'data']
            for module in test_modules:
                module_path = os.path.join(test_dir, module)
                assert os.path.isdir(module_path), f"Test module {module} not found in test/"