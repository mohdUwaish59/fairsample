# Clustering-Based Methods

Clustering-based methods use clustering to identify and handle overlapping regions.

## NBUS - Neighbourhood-Based Undersampling

Uses clustering with 4 different selection strategies.

### Variants

```python
from fairsample import NBUS

# Centroid: Select cluster centroids
sampler = NBUS(variant='centroid', n_clusters=5, random_state=42)

# Median: Select median points
sampler = NBUS(variant='median', n_clusters=5, random_state=42)

# Nearest: Select nearest to centroid
sampler = NBUS(variant='nearest', n_clusters=5, random_state=42)

# Farthest: Select farthest from centroid
sampler = NBUS(variant='farthest', n_clusters=5, random_state=42)

X_resampled, y_resampled = sampler.fit_resample(X, y)
```

**Parameters:**
- `variant`: Selection strategy ('centroid', 'median', 'nearest', 'farthest')
- `n_clusters`: Number of clusters (default: 5)
- `random_state`: Random seed

**Best for:** Large datasets, fast execution needed

## KMeansUndersampling

Uses K-Means clustering with 4 selection strategies.

### Variants

```python
from fairsample import KMeansUndersampling

# Centroid: Select cluster centroids
sampler = KMeansUndersampling(variant='centroid', n_clusters=5, random_state=42)

# Median: Select median points
sampler = KMeansUndersampling(variant='median', n_clusters=5, random_state=42)

# Nearest: Select nearest to centroid
sampler = KMeansUndersampling(variant='nearest', n_clusters=5, random_state=42)

# Farthest: Select farthest from centroid
sampler = KMeansUndersampling(variant='farthest', n_clusters=5, random_state=42)

X_resampled, y_resampled = sampler.fit_resample(X, y)
```

**Parameters:**
- `variant`: Selection strategy ('centroid', 'median', 'nearest', 'farthest')
- `n_clusters`: Number of clusters (default: 5)
- `random_state`: Random seed

**Best for:** Large datasets, well-defined clusters

## Choosing a Variant

- **Centroid**: Balanced representation
- **Median**: Robust to outliers
- **Nearest**: Preserve cluster core
- **Farthest**: Preserve cluster boundary

## Comparison

```python
from fairsample.utils import compare_techniques

results = compare_techniques(
    X, y,
    techniques=[
        'NBUS_centroid', 'NBUS_median',
        'KMeansUndersampling_centroid', 'KMeansUndersampling_median'
    ],
    complexity_measures='basic'
)

print(results[['technique', 'N3', 'sample_size']])
```

## Example: Find Best Variant

```python
from fairsample import NBUS
from fairsample.complexity import ComplexityMeasures

variants = ['centroid', 'median', 'nearest', 'farthest']
results = []

for variant in variants:
    sampler = NBUS(variant=variant, random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)
    
    cm = ComplexityMeasures(X_res, y_res)
    n3 = cm.calculate_N3()
    
    results.append({
        'variant': variant,
        'N3': n3,
        'samples': len(X_res)
    })

# Print results
import pandas as pd
df = pd.DataFrame(results)
print(df.sort_values('N3'))
```

## Next Steps

- [Overlap-Based Methods](overlap-based.md)
- [Hybrid Methods](hybrid.md)
- [API Reference](../api/techniques.md)
