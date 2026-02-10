# FairSample

Fair sampling for imbalanced datasets with 14+ resampling techniques and 40+ complexity measures.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why FairSample?

Most imbalanced learning packages only provide resampling techniques. FairSample adds **complexity measures** to help you understand *why* your dataset is difficult and *which* technique works best.

## Installation

```bash
pip install fairsample
```

## Quick Start

```python
from fairsample import RFCL
from fairsample.complexity import ComplexityMeasures
import pandas as pd

# Load data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Check complexity
cm = ComplexityMeasures(X, y)
complexity = cm.analyze_overlap()
print(f"Overlap (N3): {complexity['N3']:.4f}")

# Apply resampling
sampler = RFCL(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X, y)

# Use resampled data
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_resampled, y_resampled)
```

## Features

**14+ Resampling Techniques:**
- RFCL, NUS, URNS - Overlap-based undersampling
- SVDDWSMOTE, ODBOT, EHSO - Hybrid methods
- NBUS, KMeansUndersampling - Clustering-based (multiple variants)
- OSM - Comprehensive overlap handling
- RandomOverSampler, RandomUnderSampler - Baselines

**40+ Complexity Measures:**
- Feature Overlap: F1, F1v, F2, F3, F4, Input Noise
- Instance Overlap: N3, N4, kDN, CM, R-value, D3, SI, Borderline, Degree of Overlap
- Structural: N1, N2, T1, DBC, LSC, Clust, NSG, ICSV, ONB
- Multiresolution: Purity, Neighbourhood Separability, MRCA, C1, C2

## Usage

### Compare Multiple Techniques

```python
from fairsample.utils import compare_techniques

results = compare_techniques(
    X, y,
    techniques=['RFCL', 'NUS', 'URNS'],
    complexity_measures='basic'
)
print(results.sort_values('N3'))  # Lower N3 = less overlap
```

### Get All Complexity Measures

```python
# All measures
all_measures = cm.get_all_complexity_measures(measures='all')

# By category
feature_measures = cm.get_all_complexity_measures(measures='feature')

# Specific measures
selected = cm.get_all_complexity_measures(measures=['N3', 'F1', 'N1'])
```

### Compare Before/After

```python
from fairsample.complexity import compare_pre_post_overlap

X_resampled, y_resampled = sampler.fit_resample(X, y)
comparison = compare_pre_post_overlap(X, y, X_resampled, y_resampled)
print(comparison['improvements'])
```

## API

All techniques follow scikit-learn's API:

```python
sampler = RFCL(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X, y)
```

## Requirements

Python 3.8+ with numpy, scikit-learn, scipy, pandas, matplotlib, seaborn

## Contributing

Contributions welcome! Submit a PR or open an issue.

## License

MIT License

## Citation

```bibtex
@software{fairsample,
  author = {Mohd Uwaish},
  title = {FairSample: Fair Sampling for Imbalanced Datasets},
  year = {2024},
  url = {https://github.com/yourusername/fairsample}
}
```
