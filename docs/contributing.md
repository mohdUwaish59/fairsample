# Contributing

Thank you for considering contributing to FairSample!

## Ways to Contribute

- Report bugs and issues
- Suggest new features or techniques
- Improve documentation
- Submit bug fixes
- Add new resampling techniques
- Add new complexity measures

## Development Setup

1. Fork and clone the repository:

```bash
git clone https://github.com/yourusername/fairsample.git
cd fairsample
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:

```bash
pip install -e ".[dev]"
```

4. Run tests:

```bash
pytest tests/
```

## Code Style

- Follow PEP 8
- Use type hints where possible
- Write docstrings for all public functions
- Keep functions focused and small

## Adding a New Technique

1. Create a new file in `fairsample/techniques/`
2. Implement the technique following scikit-learn's API:

```python
from imblearn.base import BaseSampler

class MyTechnique(BaseSampler):
    def __init__(self, param1=default1, random_state=None):
        self.param1 = param1
        self.random_state = random_state
    
    def fit_resample(self, X, y):
        # Your implementation
        return X_resampled, y_resampled
```

3. Add to `fairsample/techniques/__init__.py`
4. Write tests in `tests/`
5. Update documentation

## Adding a Complexity Measure

1. Add method to `ComplexityMeasures` class in `fairsample/complexity/measures.py`:

```python
def calculate_my_measure(self):
    """
    Calculate my complexity measure.
    
    Returns
    -------
    float
        The complexity score.
    """
    # Your implementation
    return score
```

2. Add to appropriate category in `get_all_complexity_measures()`
3. Write tests
4. Update documentation

## Testing

Write tests for all new code:

```python
def test_my_technique():
    from fairsample import MyTechnique
    
    X, y = make_classification(n_samples=100, n_classes=2, weights=[0.9, 0.1])
    sampler = MyTechnique()
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    assert len(X_resampled) > 0
    assert len(X_resampled) == len(y_resampled)
```

Run tests:

```bash
pytest tests/ -v
```

## Documentation

Update documentation for any changes:

1. Update docstrings in code
2. Update relevant markdown files in `docs/`
3. Preview documentation:

```bash
mkdocs serve
```

4. Build documentation:

```bash
mkdocs build
```

## Pull Request Process

1. Create a new branch:

```bash
git checkout -b feature/my-feature
```

2. Make your changes and commit:

```bash
git add .
git commit -m "Add my feature"
```

3. Push to your fork:

```bash
git push origin feature/my-feature
```

4. Open a Pull Request on GitHub

5. Ensure all tests pass and documentation is updated

## Code Review

All submissions require review. We'll provide feedback and may request changes.

## Questions?

Open an issue or start a discussion on GitHub.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
