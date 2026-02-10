# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2024

### Added
- 14+ resampling techniques for imbalanced datasets
  - Overlap-based undersampling (RFCL, URNS, NUS, DeviOCSVM, FCMBoostOBU)
  - Hybrid methods (SVDDWSMOTE, ODBOT, EHSO)
  - Clustering-based (NBUS, KMeansUndersampling variants)
  - Comprehensive (OSM)
  - Baseline methods (RandomOverSampler, RandomUnderSampler)

- 40+ complexity measures
  - Feature overlap measures (F1, F1v, F2, F3, F4, Input Noise)
  - Instance overlap measures (N3, N4, kDN, CM, R-value, D3, SI, etc.)
  - Structural overlap measures (N1, N2, T1, DBC, LSC, Clust, etc.)
  - Multiresolution measures (Purity, MRCA, C1, C2, etc.)

- Utility functions
  - `compare_techniques()` - Compare techniques by complexity scores
  - `get_resampled_data()` - Get resampled data for custom workflows
  - `get_available_techniques()` - List all available techniques
  - `validate_input_data()` - Input validation

- Complexity analysis
  - `ComplexityMeasures` class with flexible API
  - `get_all_complexity_measures()` - Get all or specific measures
  - `analyze_overlap()` - Quick complexity analysis
  - `compare_pre_post_overlap()` - Compare before/after resampling

### Features
- Pandas DataFrame support
- Scikit-learn compatible API
- Flexible measure selection (all/category/specific)
- No forced evaluation or visualization
- Clean data output for user workflows

### Documentation
- Comprehensive README
- Usage examples
- API documentation
- Contributing guidelines

## [Unreleased]

### Planned
- Additional resampling techniques
- GPU acceleration support
- More complexity measures
- Performance optimizations
