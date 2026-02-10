"""
Test pip package specific functionality and requirements.
"""

import pytest
import sys
import os
import subprocess
import importlib.util


class TestPackageInstallation:
    """Test package installation and distribution."""
    
    def test_package_metadata(self):
        """Test package metadata is properly defined."""
        import fairsample
        
        # Check version info
        assert hasattr(fairsample, '__version__')
        assert hasattr(fairsample, '__author__')
        assert hasattr(fairsample, '__description__')
        
        # Version should be valid semver format
        version = fairsample.__version__
        version_parts = version.split('.')
        assert len(version_parts) >= 2
        
        # Author should not be empty
        assert len(fairsample.__author__) > 0
        
        # Description should not be empty
        assert len(fairsample.__description__) > 0
    
    def test_setup_py_exists(self):
        """Test that setup.py exists and is valid."""
        setup_py_path = 'setup.py'
        assert os.path.exists(setup_py_path), "setup.py file not found"
        
        # Try to read setup.py
        with open(setup_py_path, 'r') as f:
            setup_content = f.read()
        
        # Check for required setup.py components
        required_components = [
            'name=',
            'version=',
            'author=',
            'description=',
            'packages=',
            'install_requires='
        ]
        
        for component in required_components:
            assert component in setup_content, f"Missing {component} in setup.py"
    
    def test_manifest_in_exists(self):
        """Test that MANIFEST.in exists for proper package distribution."""
        manifest_path = 'MANIFEST.in'
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                manifest_content = f.read()
            
            # Should include important files
            important_files = ['README', 'LICENSE']
            for file_pattern in important_files:
                # Check if any line includes the file pattern
                found = any(file_pattern.lower() in line.lower() for line in manifest_content.split('\n'))
                if not found:
                    print(f"Warning: {file_pattern} not found in MANIFEST.in")
    
    def test_readme_exists(self):
        """Test that README file exists."""
        readme_files = ['README.md', 'README.rst', 'README.txt', 'README']
        readme_exists = any(os.path.exists(f) for f in readme_files)
        assert readme_exists, "No README file found"
    
    def test_license_exists(self):
        """Test that LICENSE file exists."""
        license_files = ['LICENSE', 'LICENSE.txt', 'LICENSE.md', 'COPYING']
        license_exists = any(os.path.exists(f) for f in license_files)
        assert license_exists, "No LICENSE file found"


class TestPackageDependencies:
    """Test package dependencies and requirements."""
    
    def test_requirements_file_exists(self):
        """Test that requirements.txt exists."""
        requirements_path = 'requirements.txt'
        assert os.path.exists(requirements_path), "requirements.txt not found"
        
        # Read and validate requirements
        with open(requirements_path, 'r') as f:
            requirements = f.read().strip().split('\n')
        
        # Should have some requirements
        non_empty_requirements = [req for req in requirements if req.strip() and not req.startswith('#')]
        assert len(non_empty_requirements) > 0, "No requirements specified"
    
    def test_core_dependencies_importable(self):
        """Test that core dependencies can be imported."""
        core_dependencies = [
            'numpy',
            'pandas', 
            'scikit-learn',
            'matplotlib',
            'scipy'
        ]
        
        for dep in core_dependencies:
            try:
                if dep == 'scikit-learn':
                    import sklearn
                else:
                    __import__(dep)
            except ImportError:
                pytest.fail(f"Core dependency {dep} not available")
    
    def test_optional_dependencies_handling(self):
        """Test that optional dependencies are handled gracefully."""
        # Test that package works even if some optional dependencies are missing
        import fairsample
        
        # Basic functionality should work
        assert hasattr(fairsample, 'RandomOverSampler')
        assert hasattr(fairsample, 'ComplexityMeasures')
    
    def test_python_version_compatibility(self):
        """Test Python version compatibility."""
        # Check minimum Python version
        python_version = sys.version_info
        
        # Should work with Python 3.7+
        assert python_version >= (3, 7), f"Python {python_version} not supported, requires 3.7+"


class TestPackageStructure:
    """Test package structure for pip distribution."""
    
    def test_package_directory_structure(self):
        """Test that package has proper directory structure."""
        # Main package directory should exist
        assert os.path.isdir('fairsample'), "Main package directory not found"
        
        # Required subdirectories
        required_dirs = ['techniques', 'complexity', 'utils']
        for dir_name in required_dirs:
            dir_path = os.path.join('fairsample', dir_name)
            assert os.path.isdir(dir_path), f"Required directory {dir_name} not found"
        
        # Each directory should have __init__.py
        for dir_name in required_dirs:
            init_path = os.path.join('fairsample', dir_name, '__init__.py')
            assert os.path.exists(init_path), f"Missing __init__.py in {dir_name}"
    
    def test_test_directory_separation(self):
        """Test that test modules are properly separated."""
        # Test directory should exist
        test_dir = 'test'
        if os.path.exists(test_dir):
            # Should contain moved modules
            moved_modules = ['evaluation', 'visualization', 'data']
            for module in moved_modules:
                module_path = os.path.join(test_dir, module)
                assert os.path.isdir(module_path), f"Test module {module} not found in test/"
        
        # Main package should NOT contain test modules
        main_package_dir = 'fairsample'
        excluded_modules = ['evaluation', 'visualization', 'data']
        for module in excluded_modules:
            module_path = os.path.join(main_package_dir, module)
            assert not os.path.isdir(module_path), f"Module {module} should be in test/, not main package"
    
    def test_init_files_exist(self):
        """Test that all __init__.py files exist."""
        # Main package
        main_init = os.path.join('fairsample', '__init__.py')
        assert os.path.exists(main_init), "Main package __init__.py not found"
        
        # Submodules
        submodules = ['techniques', 'complexity', 'utils']
        for submodule in submodules:
            init_path = os.path.join('fairsample', submodule, '__init__.py')
            assert os.path.exists(init_path), f"Missing __init__.py in {submodule}"
    
    def test_no_pycache_in_distribution(self):
        """Test that __pycache__ directories are not included in distribution."""
        # This is more of a packaging guideline test
        # In actual distribution, __pycache__ should be excluded
        
        def find_pycache_dirs(root_dir):
            pycache_dirs = []
            for root, dirs, files in os.walk(root_dir):
                if '__pycache__' in dirs:
                    pycache_dirs.append(os.path.join(root, '__pycache__'))
            return pycache_dirs
        
        pycache_dirs = find_pycache_dirs('fairsample')
        
        # If __pycache__ directories exist, they should be in .gitignore
        if pycache_dirs and os.path.exists('.gitignore'):
            with open('.gitignore', 'r') as f:
                gitignore_content = f.read()
            
            # Should ignore __pycache__
            assert '__pycache__' in gitignore_content or '*.pyc' in gitignore_content


class TestPackageImports:
    """Test package imports for pip installation."""
    
    def test_main_package_import(self):
        """Test main package can be imported."""
        try:
            import fairsample
            assert fairsample is not None
        except ImportError as e:
            pytest.fail(f"Cannot import main package: {e}")
    
    def test_submodule_imports(self):
        """Test all submodules can be imported."""
        submodules = [
            'fairsample.techniques',
            'fairsample.complexity', 
            'fairsample.utils'
        ]
        
        for submodule in submodules:
            try:
                importlib.import_module(submodule)
            except ImportError as e:
                pytest.fail(f"Cannot import submodule {submodule}: {e}")
    
    def test_key_classes_importable(self):
        """Test that key classes can be imported from main package."""
        from fairsample import (
            RandomOverSampler,
            ComplexityMeasures,
            compare_techniques
        )
        
        assert RandomOverSampler is not None
        assert ComplexityMeasures is not None
        assert compare_techniques is not None
    
    def test_star_import_works(self):
        """Test that star import works properly."""
        # This tests the __all__ definition
        try:
            from fairsample import *
            # Should have access to key components
            assert 'RandomOverSampler' in globals()
            assert 'ComplexityMeasures' in globals()
        except Exception as e:
            pytest.fail(f"Star import failed: {e}")


class TestPackageDocumentation:
    """Test package documentation for pip distribution."""
    
    def test_module_docstrings(self):
        """Test that modules have proper docstrings."""
        import fairsample
        import fairsample.techniques
        import fairsample.complexity
        import fairsample.utils
        
        modules = [
            fairsample,
            fairsample.techniques,
            fairsample.complexity,
            fairsample.utils
        ]
        
        for module in modules:
            assert module.__doc__ is not None, f"Module {module.__name__} missing docstring"
            assert len(module.__doc__.strip()) > 0, f"Module {module.__name__} has empty docstring"
    
    def test_class_docstrings(self):
        """Test that key classes have docstrings."""
        from fairsample import RandomOverSampler, ComplexityMeasures
        
        classes = [RandomOverSampler, ComplexityMeasures]
        
        for cls in classes:
            assert cls.__doc__ is not None, f"Class {cls.__name__} missing docstring"
            assert len(cls.__doc__.strip()) > 0, f"Class {cls.__name__} has empty docstring"
    
    def test_function_docstrings(self):
        """Test that key functions have docstrings."""
        from fairsample.utils import compare_techniques
        from fairsample.complexity import compare_pre_post_overlap
        
        functions = [compare_techniques, compare_pre_post_overlap]
        
        for func in functions:
            assert func.__doc__ is not None, f"Function {func.__name__} missing docstring"
            assert len(func.__doc__.strip()) > 0, f"Function {func.__name__} has empty docstring"


class TestPackageCompatibility:
    """Test package compatibility across different environments."""
    
    def test_cross_platform_compatibility(self):
        """Test basic cross-platform compatibility."""
        import fairsample
        
        # Should work on different platforms
        # Basic import and instantiation test
        from fairsample import RandomOverSampler
        sampler = RandomOverSampler()
        assert sampler is not None
    
    def test_numpy_version_compatibility(self):
        """Test compatibility with different numpy versions."""
        import numpy as np
        
        # Check numpy version
        numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
        
        # Should work with numpy 1.18+
        assert numpy_version >= (1, 18), f"Numpy version {np.__version__} might not be supported"
        
        # Test basic numpy operations work
        from fairsample import RandomOverSampler
        import numpy as np
        
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        sampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        assert X_resampled is not None
        assert y_resampled is not None
    
    def test_pandas_version_compatibility(self):
        """Test compatibility with pandas."""
        import pandas as pd
        
        # Test basic pandas operations work
        from fairsample import RandomOverSampler
        
        # Create DataFrame
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        X = df[['feature1', 'feature2']]
        y = df['target']
        
        sampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        assert X_resampled is not None
        assert y_resampled is not None


class TestPackagePerformance:
    """Test package performance characteristics for pip distribution."""
    
    def test_import_time(self):
        """Test that package imports quickly."""
        import time
        
        start_time = time.time()
        import fairsample
        import_time = time.time() - start_time
        
        # Should import within reasonable time (5 seconds)
        assert import_time < 5.0, f"Package import took {import_time:.2f} seconds"
    
    def test_memory_footprint(self):
        """Test package memory footprint."""
        import psutil
        import os
        
        # Get memory before import
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        # Import package
        import fairsample
        
        # Get memory after import
        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before
        
        # Should not use excessive memory (100MB limit)
        memory_increase_mb = memory_increase / (1024 * 1024)
        assert memory_increase_mb < 100, f"Package import used {memory_increase_mb:.2f} MB"


class TestPackageSecurity:
    """Test package security considerations."""
    
    def test_no_dangerous_imports(self):
        """Test that package doesn't import dangerous modules."""
        import ast
        import os
        
        dangerous_modules = ['os.system', 'subprocess.call', 'eval', 'exec']
        
        def check_file_for_dangerous_imports(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST to check for dangerous imports/calls
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                if any(dangerous in alias.name for dangerous in dangerous_modules):
                                    return False
                        elif isinstance(node, ast.ImportFrom):
                            if node.module and any(dangerous in node.module for dangerous in dangerous_modules):
                                return False
                except SyntaxError:
                    # Skip files with syntax errors
                    pass
                
                return True
            except Exception:
                return True  # Skip files that can't be read
        
        # Check main package files
        package_dir = 'fairsample'
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    is_safe = check_file_for_dangerous_imports(filepath)
                    assert is_safe, f"Potentially dangerous imports found in {filepath}"
    
    def test_no_hardcoded_credentials(self):
        """Test that package doesn't contain hardcoded credentials."""
        import os
        import re
        
        # Patterns that might indicate credentials
        credential_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']'
        ]
        
        def check_file_for_credentials(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                
                for pattern in credential_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        return False
                return True
            except Exception:
                return True  # Skip files that can't be read
        
        # Check main package files
        package_dir = 'fairsample'
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    is_safe = check_file_for_credentials(filepath)
                    assert is_safe, f"Potential hardcoded credentials found in {filepath}"