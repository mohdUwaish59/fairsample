# Techniques API Reference

Complete API reference for all resampling techniques.

## Base API

All techniques follow scikit-learn's API:

```python
sampler = Technique(parameters)
X_resampled, y_resampled = sampler.fit_resample(X, y)
```

## Overlap-Based Undersampling

### RFCL

::: fairsample.techniques.RFCL

### NUS

::: fairsample.techniques.NUS

### URNS

::: fairsample.techniques.URNS

### DeviOCSVM

::: fairsample.techniques.DeviOCSVM

### FCMBoostOBU

::: fairsample.techniques.FCMBoostOBU

## Hybrid Methods

### SVDDWSMOTE

::: fairsample.techniques.SVDDWSMOTE

### ODBOT

::: fairsample.techniques.ODBOT

### EHSO

::: fairsample.techniques.EHSO

## Clustering-Based

### NBUS

::: fairsample.techniques.NBUS

### KMeansUndersampling

::: fairsample.techniques.KMeansUndersampling

## Comprehensive

### OSM

::: fairsample.techniques.OSM

## Baselines

### RandomOverSampler

From imbalanced-learn:

```python
from imblearn.over_sampling import RandomOverSampler

sampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X, y)
```

### RandomUnderSampler

From imbalanced-learn:

```python
from imblearn.under_sampling import RandomUnderSampler

sampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X, y)
```
