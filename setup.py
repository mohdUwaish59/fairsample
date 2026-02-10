#!/usr/bin/env python
"""
Setup script for fairsample package.
"""

import os
from setuptools import setup, find_packages

# Read version from __version__.py
version = {}
version_file = os.path.join('fairsample', '__version__.py')
with open(version_file) as f:
    exec(f.read(), version)

# Read README for long description
readme_file = 'README.md'
if os.path.exists(readme_file):
    with open(readme_file, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = version.get('__description__', 'Fair sampling for imbalanced datasets')

# Core dependencies
install_requires = [
    'numpy>=1.20.0',
    'scikit-learn>=1.0.0',
    'scipy>=1.7.0',
    'pandas>=1.3.0',
    'matplotlib>=3.4.0',
    'seaborn>=0.11.0',
    'imbalanced-learn>=0.9.0',
]

# Optional dependencies
extras_require = {
    'fuzzy': ['scikit-fuzzy>=0.4.2'],
    'optimization': ['cvxopt>=1.2.0'],
    'dev': [
        'pytest>=7.0.0',
        'pytest-cov>=4.0.0',
        'black>=22.0.0',
        'flake8>=4.0.0',
    ],
}

# Add 'all' option
extras_require['all'] = list(set(
    dep for deps in extras_require.values() for dep in deps
))

setup(
    name='fairsample',
    version=version.get('__version__', '1.0.0'),
    author=version.get('__author__', 'Mohd Uwaish'),
    author_email=version.get('__email__', 'your.email@example.com'),
    description=version.get('__description__', 'Fair sampling for imbalanced datasets'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=version.get('__url__', 'https://github.com/yourusername/fairsample'),
    packages=find_packages(exclude=['tests*', 'examples*', 'docs*']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=[
        'machine learning',
        'imbalanced data',
        'class overlap',
        'resampling',
        'undersampling',
        'oversampling',
        'fair sampling',
        'fairness',
        'complexity measures',
    ],
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    zip_safe=False,
)
