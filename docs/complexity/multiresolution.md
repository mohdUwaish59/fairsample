# Multiresolution Measures

Multiresolution measures analyze complexity at multiple scales.

## Purity

Measures class purity in local neighborhoods.

```python
cm = ComplexityMeasures(X, y)
purity = cm.calculate_purity()
```

**Interpretation:**
- 1.0: Perfect purity (no mixing)
- 0.0: Complete mixing

## Neighbourhood Separability

Measures how well neighborhoods separate classes.

```python
ns = cm.calculate_neighbourhood_separability()
```

**Interpretation:**
- Higher values indicate better separation

## MRCA - Multiresolution Complexity Analysis

Analyzes complexity across multiple resolutions.

```python
mrca = cm.calculate_MRCA()
```

**Interpretation:**
- Captures complexity at different scales

## C1 - Entropy of Class Proportions

Measures entropy of class distribution.

```python
c1 = cm.calculate_C1()
```

**Interpretation:**
- Higher values indicate more balanced classes

## C2 - Imbalance Ratio

Measures the degree of class imbalance.

```python
c2 = cm.calculate_C2()
```

**Interpretation:**
- 1.0: Perfectly balanced
- Higher values indicate more imbalance

## Example: Analyze All Multiresolution Measures

```python
cm = ComplexityMeasures(X, y)
multi = cm.get_all_complexity_measures(measures='multiresolution')

for measure, value in multi.items():
    print(f"{measure}: {value:.4f}")
```

## Next Steps

- [Feature Overlap](feature-overlap.md)
- [Instance Overlap](instance-overlap.md)
- [Examples](../examples/basic-usage.md)
