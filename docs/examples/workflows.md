# Complete Workflows

End-to-end workflows for common use cases.

## Workflow 1: Quick Single Technique

```python
from fairsample import RFCL
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load and split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Resample
sampler = RFCL(random_state=42)
X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)

# Train and evaluate
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_res, y_train_res)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Workflow 2: Complexity-Guided Selection

```python
from fairsample.complexity import ComplexityMeasures
from fairsample import RFCL, NUS
from imblearn.under_sampling import RandomUnderSampler

# Analyze complexity
cm = ComplexityMeasures(X_train, y_train)
complexity = cm.analyze_overlap()

# Choose technique based on complexity
if complexity['N3'] > 0.3:
    print("High overlap - using RFCL")
    sampler = RFCL(random_state=42)
elif complexity['N3'] > 0.1:
    print("Moderate overlap - using NUS")
    sampler = NUS(random_state=42)
else:
    print("Low overlap - using random undersampling")
    sampler = RandomUnderSampler(random_state=42)

# Apply and train
X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_res, y_train_res)
```

## Workflow 3: Compare and Select Best

```python
from fairsample.utils import compare_techniques
from fairsample import RFCL

# Compare techniques
results = compare_techniques(
    X_train, y_train,
    techniques=['RFCL', 'NUS', 'URNS', 'NBUS_centroid'],
    complexity_measures='basic'
)

# Select best (lowest N3)
best_technique = results.sort_values('N3').iloc[0]['technique']
print(f"Best technique: {best_technique}")

# Apply best technique
sampler = eval(best_technique)(random_state=42)
X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)

# Train
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_res, y_train_res)
```

## Workflow 4: Track Improvement

```python
from fairsample import RFCL
from fairsample.complexity import compare_pre_post_overlap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Before resampling
clf_before = RandomForestClassifier(random_state=42)
clf_before.fit(X_train, y_train)
f1_before = f1_score(y_test, clf_before.predict(X_test), average='macro')

# Apply resampling
sampler = RFCL(random_state=42)
X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)

# After resampling
clf_after = RandomForestClassifier(random_state=42)
clf_after.fit(X_train_res, y_train_res)
f1_after = f1_score(y_test, clf_after.predict(X_test), average='macro')

# Compare complexity
comparison = compare_pre_post_overlap(X_train, y_train, X_train_res, y_train_res)

print(f"F1 Before: {f1_before:.4f}")
print(f"F1 After: {f1_after:.4f}")
print(f"F1 Improvement: {(f1_after - f1_before):.4f}")
print(f"\nComplexity Improvements:")
for measure, improvement in comparison['improvements'].items():
    print(f"{measure}: {improvement:+.2%}")
```

## Workflow 5: Export Multiple Datasets

```python
from fairsample.utils import get_resampled_data
import pandas as pd

# Get resampled data for multiple techniques
data = get_resampled_data(
    X, y,
    techniques=['RFCL', 'NUS', 'URNS', 'NBUS_centroid']
)

# Save each to CSV
for technique, info in data.items():
    df = pd.DataFrame(info['X'], columns=X.columns)
    df['target'] = info['y']
    df.to_csv(f'data_{technique}.csv', index=False)
    print(f"Saved {technique}: {len(df)} samples")
```

## Workflow 6: Cross-Validation Pipeline

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

## Workflow 7: Hyperparameter Tuning

```python
from fairsample import RFCL
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline

# Create pipeline
pipeline = Pipeline([
    ('sampler', RFCL(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Parameter grid
param_grid = {
    'sampler__n_estimators': [50, 100, 200],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20]
}

# Grid search
grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.4f}")
```

## Workflow 8: Production Pipeline

```python
from fairsample import RFCL
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
import joblib

# Create full pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('sampler', RFCL(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Save
joblib.dump(pipeline, 'model_pipeline.pkl')

# Load and predict
loaded_pipeline = joblib.load('model_pipeline.pkl')
predictions = loaded_pipeline.predict(X_new)
```

## Next Steps

- [Basic Usage](basic-usage.md)
- [Comparing Techniques](comparison.md)
- [API Reference](../api/techniques.md)
