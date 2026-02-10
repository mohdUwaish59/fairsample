# Utils API Reference

Utility functions for comparing techniques and getting resampled data.

## compare_techniques

Compare multiple resampling techniques based on complexity measures.

```python
from fairsample.utils import compare_techniques

results = compare_techniques(
    X, y,
    techniques=['RFCL', 'NUS', 'URNS'],
    complexity_measures='basic'
)
```

**Parameters:**

- `X`: array-like, shape (n_samples, n_features)
    - Feature matrix
- `y`: array-like, shape (n_samples,)
    - Target vector
- `techniques`: list of str
    - List of technique names to compare
- `complexity_measures`: str or list, default='basic'
    - Which measures to calculate:
        - `'basic'`: Quick subset (N3, F1, N1, T1, imbalance_ratio)
        - `'all'`: All 40+ measures
        - `'feature'`: Feature overlap measures
        - `'instance'`: Instance overlap measures
        - `'structural'`: Structural measures
        - `'multiresolution'`: Multiresolution measures
        - List of specific measures: `['N3', 'F1', 'N1']`

**Returns:**

- `results`: pandas.DataFrame
    - DataFrame with columns: technique, complexity measures, sample_size

**Example:**

```python
results = compare_techniques(
    X, y,
    techniques=['RFCL', 'NUS', 'URNS'],
    complexity_measures=['N3', 'F1', 'N1']
)

# Sort by N3 (lower is better)
print(results.sort_values('N3'))
```

## get_resampled_data

Get resampled data for multiple techniques.

```python
from fairsample.utils import get_resampled_data

data = get_resampled_data(
    X, y,
    techniques=['RFCL', 'NUS', 'URNS']
)
```

**Parameters:**

- `X`: array-like, shape (n_samples, n_features)
    - Feature matrix
- `y`: array-like, shape (n_samples,)
    - Target vector
- `techniques`: list of str
    - List of technique names

**Returns:**

- `data`: dict
    - Dictionary mapping technique names to dicts with keys:
        - `'X'`: Resampled features
        - `'y'`: Resampled targets

**Example:**

```python
data = get_resampled_data(X, y, ['RFCL', 'NUS'])

# Access resampled data
X_rfcl = data['RFCL']['X']
y_rfcl = data['RFCL']['y']

# Save to CSV
import pandas as pd
for technique, info in data.items():
    df = pd.DataFrame(info['X'])
    df['target'] = info['y']
    df.to_csv(f'{technique}_data.csv', index=False)
```

## Available Technique Names

Use these strings in `techniques` parameter:

**Overlap-Based:**
- `'RFCL'`
- `'NUS'`
- `'URNS'`
- `'DeviOCSVM'`
- `'FCMBoostOBU'`

**Hybrid:**
- `'SVDDWSMOTE'`
- `'ODBOT'`
- `'EHSO'`

**Clustering-Based:**
- `'NBUS_centroid'`
- `'NBUS_median'`
- `'NBUS_nearest'`
- `'NBUS_farthest'`
- `'KMeansUndersampling_centroid'`
- `'KMeansUndersampling_median'`
- `'KMeansUndersampling_nearest'`
- `'KMeansUndersampling_farthest'`

**Comprehensive:**
- `'OSM'`

**Baselines:**
- `'RandomOverSampler'`
- `'RandomUnderSampler'`
