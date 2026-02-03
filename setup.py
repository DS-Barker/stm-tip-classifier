"""
STM Tip Classifier
==================

Automated classification of scanning tunneling microscope (STM) probe 
tip quality using deep learning and deterministic methods.

Author: Dylan S. Barker
Institution: University of Leeds
Year: 2024
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt') as f:
    for line in f:
        line = line.strip()
        # Skip comments and optional dependencies
        if line and not line.startswith('#') and not line.startswith('jupyter'):
            # Remove version constraints for setup.py
            package = line.split('>=')[0].split('==')[0].split('<')[0]
            requirements.append(line)

# Development dependencies
dev_requirements = [
    'pytest>=7.0.0',
    'pytest-cov>=3.0.0',
    'pytest-mock>=3.6.0',
    'black>=22.0.0',
    'flake8>=4.0.0',
    'mypy>=0.950',
    'isort>=5.10.0',
    'pylint>=2.13.0',
]

# Documentation dependencies
docs_requirements = [
    'sphinx>=4.5.0',
    'sphinx-rtd-theme>=1.0.0',
    'sphinx-autodoc-typehints>=1.18.0',
]

# Jupyter/notebook dependencies
notebook_requirements = [
    'jupyter>=1.0.0',
    'ipykernel>=6.0.0',
    'notebook>=6.4.0',
    'ipywidgets>=7.7.0',
]

setup(
    # Package Metadata
    name='stm-tip-classifier',
    version='1.0.0',
    author='Dylan S. Barker',
    author_email='dylan.barker01@gmail.com',
    description='Automating Scanning Tunnelling Microscopy: A comparative Study of Machine Learning and Deterministic Methods',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/stm-tip-classifier',
    
    # Package Configuration
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    
    # Python Version
    python_requires='>=3.9.0,<3.13',
    
    # Dependencies
    install_requires=requirements,
    
    # Optional Dependencies
    extras_require={
        'dev': dev_requirements,
        'docs': docs_requirements,
        'notebooks': notebook_requirements,
        'all': dev_requirements + docs_requirements + notebook_requirements,
    },
    
    # Entry Points (CLI commands)
    entry_points={
        'console_scripts': [
            'stm-classify=src.labview_interface.classify_tip:main',
            'stm-train=src.cnn_classifier.train:main',
            'stm-evaluate=src.cnn_classifier.evaluate:main',
        ],
    },
    
    # Package Data
    package_data={
        'src': [
            'config/*.yaml',
            'config/*.json',
        ],
    },
    
    # Classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    
    # Keywords
    keywords=[
        'scanning-tunneling-microscopy',
        'stm',
        'nanotechnology',
        'machine-learning',
        'computer-vision',
        'image-classification',
        'tensorflow',
        'automation',
        'scientific-computing',
    ],
    
    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/stm-tip-classifier/issues',
        'Source': 'https://github.com/yourusername/stm-tip-classifier',
        'Documentation': 'https://github.com/yourusername/stm-tip-classifier/tree/main/docs',
    },
    
    # Zip Safe
    zip_safe=False,
)

"""
Installation Instructions
=========================

Basic Installation:
    pip install -e .

Development Installation (with dev tools):
    pip install -e ".[dev]"

Full Installation (all optional dependencies):
    pip install -e ".[all]"

Specific extras:
    pip install -e ".[notebooks]"  # For Jupyter support
    pip install -e ".[docs]"       # For documentation building

Usage After Installation:
    
    # Command-line tools
    stm-classify --image path/to/image.png --method CNN
    stm-train --config config.yaml --data data/si111_7x7/
    stm-evaluate --model models/my_model.h5 --test_data data/test/
    
    # In Python
    from cnn_classifier import CNNModel
    from deterministic_classifier import CrossCorrelation

Uninstallation:
    pip uninstall stm-tip-classifier
"""