# Feature Overlap Measures

Feature-level measures analyze how well individual features separate classes.

## F1 - Maximum Fisher's Discriminant Ratio

Measures the maximum discriminative power across all features.

```python
cm = ComplexityMeasures(X, y)
f1 = cm.calculate_F1()
print(f"F1: {f1:.4f}")
```

**Interpretation:**
- Higher values indicate better feature separability
- Low F1 suggests poor feature discrimination

## F1v - Directional-Vector Maximum Fisher's Discriminant Ratio

Extends F1 by considering feature combinations.

```python
f1v = cm.calculate_F1v()
```

**Interpretation:**
- Higher values indicate better separability in feature space

## F2 - Volume of Overlapping Region

Measures the volume of the region where classes overlap.

```python
f2 = cm.calculate_F2()
```

**Interpretation:**
- Lower values indicate less overlap
- 0.0: No overlap
- 1.0: Complete overlap

## F3 - Maximum Individual Feature Efficiency

Measures the efficiency of the best single feature.

```python
f3 = cm.calculate_F3()
```

**Interpretation:**
- Higher values indicate at least one feature separates classes well

## F4 - Collective Feature Efficiency

Measures how well features collectively separate classes.

```python
f4 = cm.calculate_F4()
```

**Interpretation:**
- Higher values indicate good collective discrimination

## Input Noise

Estimates the amount of noise in input features.

```python
noise = cm.calculate_input_noise()
```

**Interpretation:**
- Lower values indicate cleaner data
- Higher values suggest noisy features

## Example: Analyze All Feature Measures

```python
from fairsample.complexity import ComplexityMeasures

cm = ComplexityMeasures(X, y)
feature_measures = cm.get_all_complexity_measures(measures='feature')

for measure, value in feature_measures.items():
    print(f"{measure}: {value:.4f}")
```

## Next Steps

- [Instance Overlap Measures](instance-overlap.md)
- [Structural Measures](structural.md)
