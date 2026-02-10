# Structural Measures

Structural measures analyze the topology and geometry of the dataset.

## N1 - Fraction of Borderline Points

Fraction of points on the class boundary using MST.

```python
cm = ComplexityMeasures(X, y)
n1 = cm.calculate_N1()
```

**Interpretation:**
- Higher values indicate more boundary complexity

## N2 - Ratio of Intra/Extra Class Nearest Neighbor Distance

Compares distances within and between classes.

```python
n2 = cm.calculate_N2()
```

**Interpretation:**
- Lower values indicate better separation

## T1 - Fraction of Hyperspheres Covering Data

Measures how many hyperspheres are needed to cover the data.

```python
t1 = cm.calculate_T1()
```

**Interpretation:**
- Higher values indicate more complex structure

## DBC - Distance-Based Complexity

Measures complexity based on distance distributions.

```python
dbc = cm.calculate_DBC()
```

## LSC - Local Set Cardinality

Measures the average size of local neighborhoods.

```python
lsc = cm.calculate_LSC()
```

## Clust - Clustering Measure

Measures how well data forms clusters.

```python
clust = cm.calculate_Clust()
```

**Interpretation:**
- Higher values indicate more distinct clusters

## NSG - Number of Spanning Graphs

Counts the number of connected components.

```python
nsg = cm.calculate_NSG()
```

## ICSV - Inter-Class to Intra-Class Similarity Variance

Compares inter-class and intra-class variance.

```python
icsv = cm.calculate_ICSV()
```

## ONB - Overlap of Neighborhoods Between Classes

Measures neighborhood overlap between classes.

```python
onb = cm.calculate_ONB()
```

## Example: Analyze All Structural Measures

```python
cm = ComplexityMeasures(X, y)
structural = cm.get_all_complexity_measures(measures='structural')

for measure, value in structural.items():
    print(f"{measure}: {value:.4f}")
```

## Next Steps

- [Multiresolution Measures](multiresolution.md)
- [Examples](../examples/basic-usage.md)
