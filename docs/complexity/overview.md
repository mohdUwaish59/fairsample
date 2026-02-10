# Complexity Measures Overview

Complexity measures quantify dataset difficulty and class overlap. Use them to understand *why* your dataset is challenging and *which* technique works best.

## Why Complexity Measures?

Traditional metrics (accuracy, F1-score) tell you *how well* a model performs, but not *why* it struggles. Complexity measures reveal:

- **Class overlap** - How much classes overlap in feature space
- **Boundary complexity** - How irregular the decision boundary is
- **Feature discriminability** - Which features separate classes well
- **Structural issues** - Clusters, outliers, and noise

## Categories

### Feature Overlap (6 measures)
Analyze how well individual features separate classes.

- F1, F1v, F2, F3, F4, Input Noise

[Learn more →](feature-overlap.md)

### Instance Overlap (9 measures)
Measure overlap at the instance level.

- N3, N4, kDN, CM, R-value, D3, SI, Borderline, Degree of Overlap

[Learn more →](instance-overlap.md)

### Structural (9 measures)
Analyze dataset structure and topology.

- N1, N2, T1, DBC, LSC, Clust, NSG, ICSV, ONB

[Learn more →](structural.md)

### Multiresolution (5 measures)
Examine complexity at multiple scales.

- Purity, Neighbourhood Separability, MRCA, C1, C2

[Learn more →](multiresolution.md)

## Basic Usage

```python
from fairsample.complexity import ComplexityMeasures

# Create analyzer
cm = ComplexityMeasures(X, y)

# Get basic overlap measures
basic = cm.analyze_overlap()
print(f"N3: {basic['N3']:.4f}")
print(f"F1: {basic['F1']:.4f}")

# Get all measures
all_measures = cm.get_all_complexity_measures(measures='all')
print(all_measures)
```

## Get Specific Categories

```python
# Feature overlap only
feature = cm.get_all_complexity_measures(measures='feature')

# Instance overlap only
instance = cm.get_all_complexity_measures(measures='instance')

# Structural only
structural = cm.get_all_complexity_measures(measures='structural')

# Multiresolution only
multi = cm.get_all_complexity_measures(measures='multiresolution')
```

## Get Specific Measures

```python
# Select specific measures
selected = cm.get_all_complexity_measures(
    measures=['N3', 'F1', 'N1', 'T1']
)
```

## Compare Before/After Resampling

```python
from fairsample import RFCL
from fairsample.complexity import compare_pre_post_overlap

# Apply resampling
sampler = RFCL(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X, y)

# Compare complexity
comparison = compare_pre_post_overlap(X, y, X_resampled, y_resampled)

print("Improvements:")
for measure, improvement in comparison['improvements'].items():
    print(f"{measure}: {improvement:.2%}")
```

## Interpreting Results

### Lower is Better
Most measures: lower values indicate less complexity/overlap.

- **N3 < 0.1**: Low overlap, easy dataset
- **N3 0.1-0.3**: Moderate overlap
- **N3 > 0.3**: High overlap, difficult dataset

### Higher is Better
Some measures (like F1): higher values indicate better separability.

## Common Patterns

| Pattern | Likely Issue | Recommended Action |
|---------|--------------|-------------------|
| High N3, High F1 | Instance overlap | Use RFCL, NUS, URNS |
| High N3, Low F1 | Feature overlap | Feature engineering |
| High N1, Low N3 | Boundary complexity | Use SVDDWSMOTE, OSM |
| High Clust | Multiple clusters | Use NBUS, KMeans |

## Next Steps

- [Feature Overlap Measures](feature-overlap.md)
- [Instance Overlap Measures](instance-overlap.md)
- [Structural Measures](structural.md)
- [Multiresolution Measures](multiresolution.md)
