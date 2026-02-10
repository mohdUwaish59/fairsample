# Instance Overlap Measures

Instance-level measures quantify overlap between classes at the data point level.

## N3 - Error Rate of Nearest Neighbor

Measures how often nearest neighbors have different class labels.

```python
cm = ComplexityMeasures(X, y)
n3 = cm.calculate_N3()
print(f"N3: {n3:.4f}")
```

**Interpretation:**
- 0.0: No overlap, perfect separation
- 0.1-0.3: Moderate overlap
- > 0.3: High overlap

**Best for:** Quick assessment of overall overlap

## N4 - Non-linearity of Nearest Neighbor

Measures non-linearity by interpolating between nearest neighbors.

```python
n4 = cm.calculate_N4()
```

**Interpretation:**
- Low: Linear boundary
- High: Non-linear, complex boundary

## kDN - k-Disagreeing Neighbors

Fraction of instances with disagreeing neighbors.

```python
kdn = cm.calculate_kDN(k=5)
```

**Parameters:**
- `k`: Number of neighbors (default: 5)

**Interpretation:**
- Low: Clear class regions
- High: Mixed neighborhoods

## CM - Class Imbalance Metric

Measures class imbalance combined with overlap.

```python
cm_score = cm.calculate_CM()
```

**Interpretation:**
- Higher values indicate more severe imbalance with overlap

## R-value - Overlap Region Size

Estimates the size of the overlap region.

```python
r_value = cm.calculate_R_value()
```

**Interpretation:**
- 0.0: No overlap
- 1.0: Complete overlap

## D3 - Disjunct Class Percentage

Percentage of instances in disjunct regions.

```python
d3 = cm.calculate_D3()
```

**Interpretation:**
- High: Many isolated instances
- Low: Cohesive class regions

## SI - Silhouette Index

Measures how well instances fit their class cluster.

```python
si = cm.calculate_SI()
```

**Interpretation:**
- 1.0: Perfect clustering
- 0.0: Overlapping clusters
- -1.0: Misclassified instances

## Borderline - Borderline Instance Ratio

Fraction of instances near the class boundary.

```python
borderline = cm.calculate_borderline()
```

**Interpretation:**
- High: Many boundary instances (difficult)
- Low: Clear separation

## Degree of Overlap

Overall degree of class overlap.

```python
overlap = cm.calculate_degree_of_overlap()
```

**Interpretation:**
- 0.0: No overlap
- 1.0: Complete overlap

## Example: Analyze All Instance Measures

```python
from fairsample.complexity import ComplexityMeasures

cm = ComplexityMeasures(X, y)

# Get all instance measures
instance_measures = cm.get_all_complexity_measures(measures='instance')

# Print sorted by value
for measure, value in sorted(instance_measures.items(), key=lambda x: x[1]):
    print(f"{measure}: {value:.4f}")
```

## Example: Track Improvement

```python
from fairsample import RFCL

# Before resampling
cm_before = ComplexityMeasures(X, y)
n3_before = cm_before.calculate_N3()

# Apply resampling
sampler = RFCL(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X, y)

# After resampling
cm_after = ComplexityMeasures(X_resampled, y_resampled)
n3_after = cm_after.calculate_N3()

# Calculate improvement
improvement = (n3_before - n3_after) / n3_before * 100
print(f"N3 improved by {improvement:.1f}%")
```

## Next Steps

- [Feature Overlap Measures](feature-overlap.md)
- [Structural Measures](structural.md)
- [Examples](../examples/basic-usage.md)
