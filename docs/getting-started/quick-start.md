# Quick Start

Get up and running with FairSample in minutes.

## Basic Resampling

```python
from fairsample import RFCL
import pandas as pd

# Load your data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Apply resampling
sampler = RFCL(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X, y)

# Train your model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_resampled, y_resampled)
```

## Check Complexity

```python
from fairsample.complexity import ComplexityMeasures

# Analyze your data
cm = ComplexityMeasures(X, y)
complexity = cm.analyze_overlap()

print(f"N3 (overlap): {complexity['N3']:.4f}")
print(f"Imbalance ratio: {complexity['imbalance_ratio']:.2f}")
```

## Compare Techniques

```python
from fairsample.utils import compare_techniques

# Compare multiple techniques
results = compare_techniques(
    X, y,
    techniques=['RFCL', 'NUS', 'URNS'],
    complexity_measures='basic'
)

# View results
print(results.sort_values('N3'))
```

## Complete Workflow

```python
from fairsample import RFCL
from fairsample.complexity import ComplexityMeasures, compare_pre_post_overlap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# 1. Check original complexity
cm = ComplexityMeasures(X, y)
original = cm.analyze_overlap()
print(f"Original N3: {original['N3']:.4f}")

# 2. Apply resampling
sampler = RFCL(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X, y)

# 3. Compare complexity
comparison = compare_pre_post_overlap(X, y, X_resampled, y_resampled)
print(f"Improvement: {comparison['improvements']}")

# 4. Train and evaluate
clf = RandomForestClassifier(random_state=42)
scores = cross_val_score(clf, X_resampled, y_resampled, cv=5)
print(f"Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

## Next Steps

- [Learn about all techniques](../techniques/overview.md)
- [Explore complexity measures](../complexity/overview.md)
- [See more examples](../examples/basic-usage.md)
