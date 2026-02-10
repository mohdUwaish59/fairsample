#!/usr/bin/env python3
"""
Comprehensive test runner for fairsample package.

This script runs all tests and generates a detailed report of the package's
readiness for pip distribution and production use.
"""

import pytest
import sys
import os
import time
import subprocess
from pathlib import Path


def run_test_suite():
    """Run the complete test suite and generate report."""
    
    print("=" * 80)
    print(" FairSample - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()
    
    # Test configuration
    test_files = [
        'test_package_structure.py',
        'test_techniques.py', 
        'test_complexity.py',
        'test_utils.py',
        'test_integration.py',
        'test_pip_package.py'
    ]
    
    # Results tracking
    results = {}
    total_start_time = time.time()
    
    # Run each test file
    for test_file in test_files:
        print(f"Running {test_file}...")
        print("-" * 60)
        
        start_time = time.time()
        
        # Run pytest for this file
        result = pytest.main([
            test_file,
            '-v',
            '--tb=short',
            '--disable-warnings'
        ])
        
        execution_time = time.time() - start_time
        
        results[test_file] = {
            'result': result,
            'time': execution_time,
            'status': 'PASSED' if result == 0 else 'FAILED'
        }
        
        print(f"Status: {results[test_file]['status']}")
        print(f"Time: {execution_time:.2f}s")
        print()
    
    total_time = time.time() - total_start_time
    
    # Generate summary report
    print("=" * 80)
    print(" TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed_tests = 0
    failed_tests = 0
    
    for test_file, result in results.items():
        status_symbol = "✓" if result['status'] == 'PASSED' else "✗"
        print(f"{status_symbol} {test_file:<35} {result['status']:<8} ({result['time']:.2f}s)")
        
        if result['status'] == 'PASSED':
            passed_tests += 1
        else:
            failed_tests += 1
    
    print()
    print(f"Total Tests: {len(test_files)}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/len(test_files)*100):.1f}%")
    print(f"Total Time: {total_time:.2f}s")
    
    # Package readiness assessment
    print()
    print("=" * 80)
    print(" PACKAGE READINESS ASSESSMENT")
    print("=" * 80)
    
    critical_tests = [
        'test_package_structure.py',
        'test_techniques.py',
        'test_complexity.py',
        'test_pip_package.py'
    ]
    
    critical_passed = all(
        results[test]['status'] == 'PASSED' 
        for test in critical_tests 
        if test in results
    )
    
    if critical_passed and failed_tests == 0:
        print("🎉 PACKAGE READY FOR PRODUCTION!")
        print("   All tests passed. Package is ready for pip distribution.")
        readiness_status = "READY"
    elif critical_passed:
        print("⚠️  PACKAGE MOSTLY READY")
        print("   Critical tests passed, but some optional tests failed.")
        print("   Package can be distributed but improvements recommended.")
        readiness_status = "MOSTLY_READY"
    else:
        print("❌ PACKAGE NOT READY")
        print("   Critical tests failed. Package needs fixes before distribution.")
        readiness_status = "NOT_READY"
    
    # Detailed recommendations
    print()
    print("RECOMMENDATIONS:")
    
    if results.get('test_package_structure.py', {}).get('status') == 'FAILED':
        print("- Fix package structure issues before proceeding")
    
    if results.get('test_techniques.py', {}).get('status') == 'FAILED':
        print("- Resolve technique implementation issues")
    
    if results.get('test_complexity.py', {}).get('status') == 'FAILED':
        print("- Fix complexity measures implementation")
    
    if results.get('test_utils.py', {}).get('status') == 'FAILED':
        print("- Address utility function issues")
    
    if results.get('test_integration.py', {}).get('status') == 'FAILED':
        print("- Resolve integration issues between modules")
    
    if results.get('test_pip_package.py', {}).get('status') == 'FAILED':
        print("- Fix pip package distribution issues")
    
    if failed_tests == 0:
        print("- All tests passed! Consider adding more edge case tests")
        print("- Package is ready for restructuring and pip distribution")
    
    return readiness_status, results


def check_test_environment():
    """Check if test environment is properly set up."""
    
    print("Checking test environment...")
    
    # Check if pytest is available
    try:
        import pytest
        print(f"✓ pytest available (version {pytest.__version__})")
    except ImportError:
        print("✗ pytest not available - install with: pip install pytest")
        return False
    
    # Check if package is importable
    try:
        import fairsample
        print(f"✓ fairsample importable (version {fairsample.__version__})")
    except ImportError as e:
        print(f"✗ fairsample not importable: {e}")
        return False
    
    # Check required dependencies
    required_deps = ['numpy', 'pandas', 'sklearn', 'matplotlib', 'scipy']
    missing_deps = []
    
    for dep in required_deps:
        try:
            if dep == 'sklearn':
                import sklearn
            else:
                __import__(dep)
            print(f"✓ {dep} available")
        except ImportError:
            print(f"✗ {dep} not available")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\nMissing dependencies: {missing_deps}")
        print("Install with: pip install " + " ".join(missing_deps))
        return False
    
    print("✓ Test environment ready")
    return True


def generate_test_report(results):
    """Generate detailed test report file."""
    
    report_content = f"""
# FairSample Test Report

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Test Results Summary

| Test File | Status | Time (s) |
|-----------|--------|----------|
"""
    
    for test_file, result in results.items():
        status_symbol = "✅" if result['status'] == 'PASSED' else "❌"
        report_content += f"| {test_file} | {status_symbol} {result['status']} | {result['time']:.2f} |\n"
    
    passed_count = sum(1 for r in results.values() if r['status'] == 'PASSED')
    total_count = len(results)
    
    report_content += f"""
## Summary Statistics

- **Total Tests**: {total_count}
- **Passed**: {passed_count}
- **Failed**: {total_count - passed_count}
- **Success Rate**: {(passed_count/total_count*100):.1f}%

## Test Categories

### 1. Package Structure Tests
- Validates package organization and imports
- Ensures proper module separation
- Checks for required files and directories

### 2. Techniques Tests  
- Tests all resampling techniques
- Validates API consistency
- Checks robustness and edge cases

### 3. Complexity Measures Tests
- Tests all complexity measures
- Validates mathematical correctness
- Checks performance and scalability

### 4. Utils Tests
- Tests utility functions
- Validates integration capabilities
- Checks helper function reliability

### 5. Integration Tests
- Tests end-to-end workflows
- Validates cross-module compatibility
- Checks data format support

### 6. Pip Package Tests
- Tests pip distribution readiness
- Validates package metadata
- Checks installation requirements

## Recommendations

"""
    
    if passed_count == total_count:
        report_content += "🎉 **All tests passed!** Package is ready for production use and pip distribution.\n"
    else:
        report_content += f"⚠️ **{total_count - passed_count} test(s) failed.** Review failed tests before distribution.\n"
    
    # Write report to file
    with open('test_report.md', 'w') as f:
        f.write(report_content)
    
    print(f"\n📄 Detailed test report saved to: test_report.md")


def main():
    """Main test runner function."""
    
    # Change to tests directory
    test_dir = Path(__file__).parent
    os.chdir(test_dir)
    
    # Check environment
    if not check_test_environment():
        print("\n❌ Test environment not ready. Please fix issues and try again.")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print(" STARTING COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    # Run tests
    readiness_status, results = run_test_suite()
    
    # Generate report
    generate_test_report(results)
    
    # Exit with appropriate code
    if readiness_status == "READY":
        print("\n🎉 ALL TESTS PASSED - PACKAGE READY FOR RESTRUCTURING!")
        sys.exit(0)
    elif readiness_status == "MOSTLY_READY":
        print("\n⚠️  MOSTLY READY - MINOR ISSUES TO ADDRESS")
        sys.exit(1)
    else:
        print("\n❌ PACKAGE NOT READY - CRITICAL ISSUES TO FIX")
        sys.exit(2)


if __name__ == "__main__":
    main()