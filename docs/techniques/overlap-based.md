# Overlap-Based Undersampling

These techniques identify and remove overlapping instances from the majority class to reduce class overlap.

## RFCL - Random Forest Cleaning Rule

Uses Random Forest to identify and remove noisy majority class instances.

```python
from fairsample import RFCL

sampler = RFCL(
    n_estimators=100,
    max_depth=None,
    random_state=42
)
X_resampled, y_resampled = sampler.fit_resample(X, y)
```

**Parameters:**
- `n_estimators`: Number of trees (default: 100)
- `max_depth`: Maximum tree depth (default: None)
- `random_state`: Random seed

**Best for:** General-purpose overlap reduction, fast execution

## NUS - Neighbourhood-based Under-Sampling

Removes majority instances based on local neighborhood analysis.

```python
from fairsample import NUS

sampler = NUS(
    n_neighbors=5,
    threshold=0.5
)
X_resampled, y_resampled = sampler.fit_resample(X, y)
```

**Parameters:**
- `n_neighbors`: Number of neighbors to consider (default: 5)
- `threshold`: Decision threshold (default: 0.5)

**Best for:** Datasets with localized overlap regions

## URNS - Undersampling based on Recursive Neighbourhood Search

Recursively identifies and removes overlapping instances.

```python
from fairsample import URNS

sampler = URNS(
    n_neighbors=5,
    max_iterations=10
)
X_resampled, y_resampled = sampler.fit_resample(X, y)
```

**Parameters:**
- `n_neighbors`: Number of neighbors (default: 5)
- `max_iterations`: Maximum recursion depth (default: 10)

**Best for:** Complex overlap patterns, willing to trade speed for quality

## DeviOCSVM - One-Class SVM Method

Uses One-Class SVM to identify majority class outliers near the boundary.

```python
from fairsample import DeviOCSVM

sampler = DeviOCSVM(
    nu=0.5,
    kernel='rbf',
    gamma='scale'
)
X_resampled, y_resampled = sampler.fit_resample(X, y)
```

**Parameters:**
- `nu`: Upper bound on fraction of outliers (default: 0.5)
- `kernel`: Kernel type (default: 'rbf')
- `gamma`: Kernel coefficient (default: 'scale')

**Best for:** Non-linear decision boundaries

## FCMBoostOBU - Fuzzy C-Means Boosted Overlap-Based Undersampling

Uses fuzzy clustering to identify and remove overlapping instances.

```python
from fairsample import FCMBoostOBU

sampler = FCMBoostOBU(
    n_clusters=3,
    m=2.0,
    max_iter=100
)
X_resampled, y_resampled = sampler.fit_resample(X, y)
```

**Parameters:**
- `n_clusters`: Number of fuzzy clusters (default: 3)
- `m`: Fuzziness parameter (default: 2.0)
- `max_iter`: Maximum iterations (default: 100)

**Best for:** Datasets with fuzzy boundaries

## Comparison

```python
from fairsample.utils import compare_techniques

results = compare_techniques(
    X, y,
    techniques=['RFCL', 'NUS', 'URNS', 'DeviOCSVM', 'FCMBoostOBU'],
    complexity_measures='basic'
)

print(results[['technique', 'N3', 'F1', 'sample_size']])
```

## Next Steps

- [Hybrid Methods](hybrid.md)
- [Clustering-Based Methods](clustering.md)
- [API Reference](../api/techniques.md)
