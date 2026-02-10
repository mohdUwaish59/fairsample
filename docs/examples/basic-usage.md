# Basic Usage Examples

Simple examples to get you started with the toolkit.

## Example 1: Single Technique

```python
from fairsample import RFCL
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

# Load data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Apply RFCL
sampler = RFCL(random_state=42)
X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_resampled, y_train_resampled)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Example 2: Check Complexity First

```python
from fairsample.complexity import ComplexityMeasures

# Analyze complexity
cm = ComplexityMeasures(X_train, y_train)
complexity = cm.analyze_overlap()

print(f"N3 (overlap): {complexity['N3']:.4f}")
print(f"F1 (feature overlap): {complexity['F1']:.4f}")
print(f"Imbalance ratio: {complexity['imbalance_ratio']:.2f}")

# Decide based on complexity
if complexity['N3'] > 0.3:
    print("High overlap detected - using RFCL")
    sampler = RFCL(random_state=42)
else:
    print("Low overlap - using random undersampling")
    from imblearn.under_sampling import RandomUnderSampler
    sampler = RandomUnderSampler(random_state=42)

X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
```

## Example 3: Save Resampled Data

```python
from fairsample import RFCL
import pandas as pd

# Apply resampling
sampler = RFCL(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X, y)

# Convert to DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['target'] = y_resampled

# Save to CSV
df_resampled.to_csv('resampled_data.csv', index=False)
print(f"Saved {len(df_resampled)} samples")
```

## Example 4: Multiple Datasets

```python
from fairsample.utils import get_resampled_data

# Get resampled data for multiple techniques
data = get_resampled_data(
    X, y,
    techniques=['RFCL', 'NUS', 'URNS']
)

# Save each to CSV
for technique, info in data.items():
    df = pd.DataFrame(info['X'])
    df['target'] = info['y']
    df.to_csv(f'{technique}_data.csv', index=False)
    print(f"{technique}: {len(df)} samples")
```

## Example 5: Track Improvement

```python
from fairsample import RFCL
from fairsample.complexity import compare_pre_post_overlap

# Apply resampling
sampler = RFCL(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X, y)

# Compare complexity
comparison = compare_pre_post_overlap(X, y, X_resampled, y_resampled)

print("Before resampling:")
print(comparison['before'])

print("\nAfter resampling:")
print(comparison['after'])

print("\nImprovements:")
for measure, improvement in comparison['improvements'].items():
    print(f"{measure}: {improvement:+.2%}")
```

## Example 6: Cross-Validation

```python
from fairsample import RFCL
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from imblearn.pipeline import Pipeline

# Create pipeline
pipeline = Pipeline([
    ('sampler', RFCL(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Cross-validation
scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1_macro')
print(f"F1-Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

## Example 7: Get All Complexity Measures

```python
from fairsample.complexity import ComplexityMeasures

cm = ComplexityMeasures(X, y)

# Get all measures
all_measures = cm.get_all_complexity_measures(measures='all')

# Print sorted by value
print("Complexity Measures (sorted):")
for measure, value in sorted(all_measures.items(), key=lambda x: x[1]):
    print(f"{measure:30s}: {value:.4f}")
```

## Example 8: Category-Specific Measures

```python
from fairsample.complexity import ComplexityMeasures

cm = ComplexityMeasures(X, y)

# Get feature overlap measures
feature = cm.get_all_complexity_measures(measures='feature')
print("Feature Overlap:", feature)

# Get instance overlap measures
instance = cm.get_all_complexity_measures(measures='instance')
print("Instance Overlap:", instance)

# Get structural measures
structural = cm.get_all_complexity_measures(measures='structural')
print("Structural:", structural)
```

## Next Steps

- [Comparing Techniques](comparison.md)
- [Complete Workflows](workflows.md)
- [API Reference](../api/techniques.md)
