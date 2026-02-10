# Resampling Techniques Overview

This toolkit provides 14+ state-of-the-art resampling techniques for handling imbalanced datasets with class overlap.

## Categories

### Overlap-Based Undersampling
Focus on removing overlapping instances from the majority class.

- **RFCL** - Random Forest Cleaning Rule
- **NUS** - Neighbourhood-based Under-Sampling  
- **URNS** - Undersampling based on Recursive Neighbourhood Search
- **DeviOCSVM** - One-Class SVM method
- **FCMBoostOBU** - Fuzzy C-Means Boosted Overlap-Based Undersampling

[Learn more →](overlap-based.md)

### Hybrid Methods
Combine undersampling and oversampling strategies.

- **SVDDWSMOTE** - SVDD-based overlap handler with SMOTE
- **ODBOT** - Outlier Detection-Based Oversampling Technique
- **EHSO** - Evolutionary Hybrid Sampling

[Learn more →](hybrid.md)

### Clustering-Based
Use clustering to identify and handle overlapping regions.

- **NBUS** - 4 variants (Centroid, Median, Nearest, Farthest)
- **KMeansUndersampling** - 4 variants (Centroid, Median, Nearest, Farthest)

[Learn more →](clustering.md)

### Comprehensive
- **OSM** - Overlap-Separating Model (handles multiple overlap types)

### Baselines
- **RandomOverSampler** - Random oversampling
- **RandomUnderSampler** - Random undersampling

## Common API

All techniques follow scikit-learn's API:

```python
from fairsample import RFCL

sampler = RFCL(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X, y)
```

## Choosing a Technique

Use complexity measures to guide your choice:

```python
from fairsample.utils import compare_techniques

results = compare_techniques(
    X, y,
    techniques=['RFCL', 'NUS', 'URNS', 'SVDDWSMOTE'],
    complexity_measures='basic'
)

# Lower N3 = less overlap
best = results.sort_values('N3').iloc[0]
print(f"Best technique: {best['technique']}")
```

## Performance Considerations

| Technique | Speed | Memory | Best For |
|-----------|-------|--------|----------|
| RFCL | Fast | Low | General overlap |
| NUS | Medium | Low | Local overlap |
| URNS | Slow | Medium | Complex overlap |
| SVDDWSMOTE | Slow | High | Small datasets |
| NBUS | Fast | Low | Large datasets |
| KMeansUndersampling | Fast | Low | Large datasets |

## Next Steps

- [Overlap-Based Techniques](overlap-based.md)
- [Hybrid Methods](hybrid.md)
- [Clustering-Based Methods](clustering.md)
