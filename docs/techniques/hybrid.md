# Hybrid Methods

Hybrid methods combine undersampling and oversampling strategies.

## SVDDWSMOTE

Combines Support Vector Data Description with SMOTE.

```python
from fairsample import SVDDWSMOTE

sampler = SVDDWSMOTE(
    nu=0.5,
    kernel='rbf',
    gamma='scale',
    k_neighbors=5,
    random_state=42
)
X_resampled, y_resampled = sampler.fit_resample(X, y)
```

**Parameters:**
- `nu`: SVDD parameter (default: 0.5)
- `kernel`: Kernel type (default: 'rbf')
- `gamma`: Kernel coefficient (default: 'scale')
- `k_neighbors`: SMOTE neighbors (default: 5)

**Best for:** Small datasets with non-linear boundaries

## ODBOT - Outlier Detection-Based Oversampling Technique

Uses outlier detection to guide oversampling.

```python
from fairsample import ODBOT

sampler = ODBOT(
    contamination=0.1,
    n_neighbors=5,
    random_state=42
)
X_resampled, y_resampled = sampler.fit_resample(X, y)
```

**Parameters:**
- `contamination`: Expected outlier proportion (default: 0.1)
- `n_neighbors`: Number of neighbors (default: 5)

**Best for:** Datasets with outliers

## EHSO - Evolutionary Hybrid Sampling

Uses evolutionary algorithms for hybrid sampling.

```python
from fairsample import EHSO

sampler = EHSO(
    population_size=50,
    generations=100,
    mutation_rate=0.1,
    random_state=42
)
X_resampled, y_resampled = sampler.fit_resample(X, y)
```

**Parameters:**
- `population_size`: EA population size (default: 50)
- `generations`: Number of generations (default: 100)
- `mutation_rate`: Mutation probability (default: 0.1)

**Best for:** Complex datasets, willing to trade speed for quality

## Comparison

```python
from fairsample.utils import compare_techniques

results = compare_techniques(
    X, y,
    techniques=['SVDDWSMOTE', 'ODBOT', 'EHSO'],
    complexity_measures='basic'
)

print(results[['technique', 'N3', 'sample_size']])
```

## Next Steps

- [Clustering-Based Methods](clustering.md)
- [Overlap-Based Methods](overlap-based.md)
- [API Reference](../api/techniques.md)
