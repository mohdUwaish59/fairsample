# Complexity API Reference

Complete API reference for complexity measures.

## ComplexityMeasures

Main class for calculating complexity measures.

::: fairsample.complexity.ComplexityMeasures

## Usage

```python
from fairsample.complexity import ComplexityMeasures

# Create analyzer
cm = ComplexityMeasures(X, y)

# Get basic overlap measures
basic = cm.analyze_overlap()

# Get all measures
all_measures = cm.get_all_complexity_measures(measures='all')

# Get specific category
feature = cm.get_all_complexity_measures(measures='feature')

# Get specific measures
selected = cm.get_all_complexity_measures(measures=['N3', 'F1'])
```

## Individual Measures

### Feature Overlap

```python
# F1 - Maximum Fisher's Discriminant Ratio
f1 = cm.calculate_F1()

# F1v - Directional-vector maximum Fisher's discriminant ratio
f1v = cm.calculate_F1v()

# F2 - Volume of overlapping region
f2 = cm.calculate_F2()

# F3 - Maximum individual feature efficiency
f3 = cm.calculate_F3()

# F4 - Collective feature efficiency
f4 = cm.calculate_F4()

# Input Noise
noise = cm.calculate_input_noise()
```

### Instance Overlap

```python
# N3 - Error rate of nearest neighbor
n3 = cm.calculate_N3()

# N4 - Non-linearity of nearest neighbor
n4 = cm.calculate_N4()

# kDN - k-Disagreeing neighbors
kdn = cm.calculate_kDN(k=5)

# CM - Class imbalance metric
cm_score = cm.calculate_CM()

# R-value - Overlap region size
r_value = cm.calculate_R_value()

# D3 - Disjunct class percentage
d3 = cm.calculate_D3()

# SI - Silhouette index
si = cm.calculate_SI()

# Borderline - Borderline instance ratio
borderline = cm.calculate_borderline()

# Degree of overlap
overlap = cm.calculate_degree_of_overlap()
```

### Structural

```python
# N1 - Fraction of borderline points
n1 = cm.calculate_N1()

# N2 - Ratio of intra/extra class nearest neighbor distance
n2 = cm.calculate_N2()

# T1 - Fraction of hyperspheres covering data
t1 = cm.calculate_T1()

# DBC - Distance-based complexity
dbc = cm.calculate_DBC()

# LSC - Local set cardinality
lsc = cm.calculate_LSC()

# Clust - Clustering measure
clust = cm.calculate_Clust()

# NSG - Number of spanning graphs
nsg = cm.calculate_NSG()

# ICSV - Inter-class to intra-class similarity variance
icsv = cm.calculate_ICSV()

# ONB - Overlap of neighborhoods between classes
onb = cm.calculate_ONB()
```

### Multiresolution

```python
# Purity
purity = cm.calculate_purity()

# Neighbourhood Separability
ns = cm.calculate_neighbourhood_separability()

# MRCA - Multiresolution complexity analysis
mrca = cm.calculate_MRCA()

# C1 - Entropy of class proportions
c1 = cm.calculate_C1()

# C2 - Imbalance ratio
c2 = cm.calculate_C2()
```

## Utility Functions

### compare_pre_post_overlap

Compare complexity before and after resampling.

```python
from fairsample.complexity import compare_pre_post_overlap

comparison = compare_pre_post_overlap(
    X_before, y_before,
    X_after, y_after,
    measures='basic'
)

print(comparison['before'])
print(comparison['after'])
print(comparison['improvements'])
```

## Measure Categories

Use these strings with `get_all_complexity_measures()`:

- `'all'` - All 40+ measures
- `'basic'` - Quick subset (N3, F1, N1, T1, imbalance_ratio)
- `'feature'` - Feature overlap measures
- `'instance'` - Instance overlap measures
- `'structural'` - Structural measures
- `'multiresolution'` - Multiresolution measures
- `['N3', 'F1', ...]` - List of specific measures
