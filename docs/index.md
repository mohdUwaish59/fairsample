# FairSample

A Python toolkit for handling imbalanced datasets with **14+ resampling techniques** and **40+ complexity measures**.

## Why This Package?

Most imbalanced learning packages only provide resampling techniques. This toolkit adds **complexity measures** to help you:

- Understand *why* your dataset is difficult
- Identify class overlap and boundary issues
- Choose the *best* technique for your data
- Measure improvement after resampling

## Key Features

- **14+ Resampling Techniques** - Overlap-based, hybrid, and clustering methods
- **40+ Complexity Measures** - Feature, instance, structural, and multiresolution metrics
- **Scikit-learn Compatible** - Standard `fit_resample()` API
- **Pandas Support** - Works seamlessly with DataFrames
- **No Forced Workflow** - You control training and evaluation

## Quick Example

```python
from fairsample import RFCL
from fairsample.complexity import ComplexityMeasures

# Check complexity
cm = ComplexityMeasures(X, y)
print(f"Overlap: {cm.analyze_overlap()['N3']:.4f}")

# Apply resampling
sampler = RFCL(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X, y)
```

## What's Included

### Resampling Techniques
- **Overlap-Based**: RFCL, NUS, URNS, DeviOCSVM, FCMBoostOBU
- **Hybrid**: SVDDWSMOTE, ODBOT, EHSO
- **Clustering**: NBUS (4 variants), KMeansUndersampling (4 variants)
- **Comprehensive**: OSM
- **Baselines**: RandomOverSampler, RandomUnderSampler

### Complexity Measures
- **Feature Overlap** (6): F1, F1v, F2, F3, F4, Input Noise
- **Instance Overlap** (9): N3, N4, kDN, CM, R-value, D3, SI, Borderline, Degree of Overlap
- **Structural** (9): N1, N2, T1, DBC, LSC, Clust, NSG, ICSV, ONB
- **Multiresolution** (5): Purity, Neighbourhood Separability, MRCA, C1, C2

## Get Started

[Installation](getting-started/installation.md){ .md-button .md-button--primary }
[Quick Start](getting-started/quick-start.md){ .md-button }
[Examples](examples/basic-usage.md){ .md-button }

## License

MIT License - Free for commercial and personal use.
