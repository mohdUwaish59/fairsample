# Comparing Techniques

Learn how to compare multiple resampling techniques to find the best one for your data.

## Quick Comparison

```python
from fairsample.utils import compare_techniques

results = compare_techniques(
    X, y,
    techniques=['RFCL', 'NUS', 'URNS', 'SVDDWSMOTE'],
    complexity_measures='basic'
)

# View results
print(results)

# Sort by N3 (lower is better)
print(results.sort_values('N3'))
```

## Compare with All Measures

```python
results = compare_techniques(
    X, y,
    techniques=['RFCL', 'NUS', 'URNS'],
    complexity_measures='all'
)

# View specific columns
print(results[['technique', 'N3', 'F1', 'N1', 'sample_size']])
```

## Compare Specific Measures

```python
results = compare_techniques(
    X, y,
    techniques=['RFCL', 'NUS', 'URNS'],
    complexity_measures=['N3', 'F1', 'N1', 'T1']
)

print(results)
```

## Visualize Comparison

```python
import matplotlib.pyplot as plt
import seaborn as sns

results = compare_techniques(
    X, y,
    techniques=['RFCL', 'NUS', 'URNS', 'SVDDWSMOTE', 'NBUS_centroid'],
    complexity_measures='basic'
)

# Plot N3 comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=results, x='technique', y='N3')
plt.xticks(rotation=45)
plt.title('N3 Overlap Comparison')
plt.ylabel('N3 (lower is better)')
plt.tight_layout()
plt.savefig('n3_comparison.png')
plt.show()
```

## Compare Multiple Measures

```python
import matplotlib.pyplot as plt

results = compare_techniques(
    X, y,
    techniques=['RFCL', 'NUS', 'URNS'],
    complexity_measures=['N3', 'F1', 'N1']
)

# Plot multiple measures
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, measure in enumerate(['N3', 'F1', 'N1']):
    axes[i].bar(results['technique'], results[measure])
    axes[i].set_title(f'{measure} Comparison')
    axes[i].set_xlabel('Technique')
    axes[i].set_ylabel(measure)
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('multi_measure_comparison.png')
plt.show()
```

## Compare with Original Data

```python
from fairsample.complexity import ComplexityMeasures

# Original complexity
cm_original = ComplexityMeasures(X, y)
original_n3 = cm_original.calculate_N3()

# Compare techniques
results = compare_techniques(
    X, y,
    techniques=['RFCL', 'NUS', 'URNS'],
    complexity_measures='basic'
)

# Add original as baseline
results.loc[len(results)] = {
    'technique': 'Original',
    'N3': original_n3,
    'sample_size': len(X)
}

print(results.sort_values('N3'))
```

## Statistical Comparison

```python
from fairsample.utils import compare_techniques
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
import numpy as np

techniques = ['RFCL', 'NUS', 'URNS']
results = []

for technique in techniques:
    # Create pipeline
    pipeline = Pipeline([
        ('sampler', eval(technique)(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Cross-validation
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1_macro')
    
    results.append({
        'technique': technique,
        'mean_f1': scores.mean(),
        'std_f1': scores.std()
    })

# Print results
import pandas as pd
df_results = pd.DataFrame(results)
print(df_results.sort_values('mean_f1', ascending=False))
```

## Compare Execution Time

```python
import time
from fairsample.utils import get_resampled_data

techniques = ['RFCL', 'NUS', 'URNS', 'NBUS_centroid']
timing_results = []

for technique in techniques:
    start = time.time()
    data = get_resampled_data(X, y, [technique])
    elapsed = time.time() - start
    
    timing_results.append({
        'technique': technique,
        'time_seconds': elapsed,
        'samples': len(data[technique]['X'])
    })

# Print results
import pandas as pd
df_timing = pd.DataFrame(timing_results)
print(df_timing.sort_values('time_seconds'))
```

## Find Best Technique

```python
from fairsample.utils import compare_techniques

def find_best_technique(X, y, techniques, metric='N3'):
    """Find the best technique based on a complexity metric."""
    results = compare_techniques(
        X, y,
        techniques=techniques,
        complexity_measures='basic'
    )
    
    # Lower is better for most metrics
    best = results.sort_values(metric).iloc[0]
    
    return best['technique'], best[metric]

# Find best
techniques = ['RFCL', 'NUS', 'URNS', 'SVDDWSMOTE', 'NBUS_centroid']
best_technique, best_score = find_best_technique(X, y, techniques)

print(f"Best technique: {best_technique}")
print(f"N3 score: {best_score:.4f}")
```

## Next Steps

- [Complete Workflows](workflows.md)
- [Basic Usage](basic-usage.md)
- [API Reference](../api/utils.md)
