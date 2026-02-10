# Installation

## Requirements

- Python 3.8 or higher
- pip package manager

## Install from PyPI

```bash
pip install fairsample
```

## Install from Source

```bash
git clone https://github.com/yourusername/fairsample.git
cd fairsample
pip install -e .
```

## Dependencies

The package will automatically install:

- numpy >= 1.20.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- imbalanced-learn >= 0.9.0

## Optional Dependencies

For additional features:

```bash
# Fuzzy logic support
pip install scikit-fuzzy>=0.4.2

# Optimization methods
pip install cvxopt>=1.2.0

# Development tools
pip install fairsample[dev]
```

## Verify Installation

```python
import fairsample
print(fairsample.__version__)
```

## Troubleshooting

### Import Errors

If you get import errors, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Windows Users

Some packages may require Visual C++ build tools. Download from [Microsoft](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

### Mac Users with M1/M2

Use conda for better compatibility:

```bash
conda install numpy scipy scikit-learn
pip install fairsample
```
