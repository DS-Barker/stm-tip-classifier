# STM Tip Classifier

**Automated classification of scanning tunneling microscope (STM) probe tip quality using deep learning and deterministic methods.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Research Code](https://img.shields.io/badge/status-research%20code-yellow)](https://github.com/yourusername/stm-tip-classifier)

> **⚠️ Note**: This repository represents a port of research code from my PhD thesis to a public GitHub repository. While the core algorithms have been validated through peer-reviewed research, the codebase is actively being refined for broader use. Users may encounter issues during installation or execution as the code is adapted from a research environment. Bug reports and contributions are welcome and appreciated.
---

## Overview

This repository contains the implementation of automated STM tip state classification methods developed during my PhD research at the University of Leeds. The system achieves **96% accuracy**, matching human expert performance, while **reducing manual intervention time by 90%** during nanoscience experiments.

**Key Innovation**: Demonstrates when simpler deterministic methods (cross-correlation) can match deep learning performance, avoiding the need for large labeled datasets.

---

## Key Features

- ** CNN Classifier**: TensorFlow-based convolutional neural network achieving 96% accuracy on Si(111)-7×7 surfaces
- ** Deterministic Methods**: Cross-correlation (97% precision) and circularity analysis for rapid classification
- ** LabVIEW Integration**: Real-time tip quality assessment during experiments via Nanonis SPM interface
- ** Automated Tip Preparation**: Autonomous tip shaping loop that eliminates operator presence requirement
- ** Comprehensive Evaluation**: ROC curves, confusion matrices, and method comparison tools

---

## Performance Results

Comparison across three classification methods on silicon surfaces:

| Method | Accuracy | Precision | Dataset Requirements |
|--------|----------|-----------|----------------------|
| **CNN** | 96% | 92% | 1000+ labeled images |
| **Cross-Correlation** | 90% | 97% | 1 reference image |
| **Human Expert** | 92% | 93% | N/A |

### Surface Coverage
- ✅ Si(111)-7×7 reconstruction
- ✅ B-doped Si(111) surface  
- ✅ Cu(111) with C₆₀ molecules
- ✅ Cu(111) with CO molecules and adatoms

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stm-tip-classifier.git
cd stm-tip-classifier

# Create conda environment
conda env create -f environment.yml
conda activate stm-classifier

# Or use pip
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Basic Usage

#### 1. Classify a single image with CNN
```python
from src.cnn_classifier import CNNModel
import numpy as np
from PIL import Image

# Load pre-trained model
model = CNNModel.load('models/si111_7x7_cnn.h5')

# Load and classify image
image = np.array(Image.open('path/to/stm_image.png'))
result = model.predict(image)

print(f"Classification: {result['label']}")  # "good" or "bad"
print(f"Confidence: {result['confidence']:.2%}")
```

#### 2. Classify with cross-correlation (deterministic)
```python
from src.deterministic_classifier import CrossCorrelation

# Load reference image (example of "good" tip)
reference = np.array(Image.open('data/reference_images/si111_good.png'))

# Create classifier
ccr = CrossCorrelation(reference, threshold=0.99)

# Classify test image
result = ccr.classify(test_image)
print(f"CCR Value: {result['ccr_value']:.3f}")
print(f"Classification: {result['label']}")
```

#### 3. Compare both methods
See [examples/03_compare_methods.ipynb](examples/03_compare_methods.ipynb) for a complete Jupyter notebook demonstration.

---

## Repository Structure

```
stm-tip-classifier/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment
├── setup.py                     # Package installation
│
├── src/                         # Source code
│   ├── cnn_classifier/          # Deep learning models
│   ├── deterministic_classifier/# Cross-correlation & circularity
│   ├── data_processing/         # Image loading & preprocessing
│   └── labview_interface/       # LabVIEW integration scripts
│
├── labview/                     # LabVIEW automation VIs
│   ├── main_control.vi          # Main automation loop
│   ├── tip_preparation.vi       # Tip shaping routine
│   └── python_caller.vi         # Python integration
│
├── models/                      # Pre-trained models
│   ├── si111_7x7_cnn.h5        # Silicon 7×7 model
│   └── bsi_cnn.h5              # B:Si model
│
├── results/                     # Performance metrics
│   ├── roc_curves/              # ROC curve visualizations
│   └── performance_comparison.csv
│
├── examples/                    # Usage examples
│   ├── 01_train_cnn_model.py
│   ├── 02_run_ccr_classifier.py
│   └── 03_compare_methods.ipynb
│
├── tests/                       # Unit tests
└── docs/                        # Detailed documentation
    ├── architecture.md          # System design
    ├── labview_integration.md   # LabVIEW setup guide
    └── usage_examples.md        # Detailed examples
```

---

## Datasets

Due to size constraints, full datasets are not included in this repository. Sample images are provided in `examples/sample_images/`.

**Full datasets available upon request** or can be regenerated using the automated data collection scripts in `scripts/`.

### Dataset Structure
```
data/
├── si111_7x7/
│   ├── train/ (1000 images: 500 good, 500 bad)
│   ├── validation/ (200 images)
│   └── test/ (200 images)
├── bsi/
│   └── [similar structure]
└── reference_images/
    ├── si111_good.png
    └── bsi_good.png
```

---

## LabVIEW Automation

The system integrates with Nanonis SPM software for real-time tip quality control:

1. **Automated scanning**: Acquires STM images at specified intervals
2. **Classification**: Python classifier determines tip quality
3. **Decision logic**: If "bad" → trigger tip preparation routine
4. **Tip shaping**: Controlled bias pulses and surface indentation
5. **Validation**: Repeat until "good" tip achieved

**Average time to achieve good tip**: 9.3 minutes (~12 reshaping attempts)

See [docs/labview_integration.md](docs/labview_integration.md) for detailed setup instructions.

---

## Documentation

- **[Architecture Overview](docs/architecture.md)**: System design and data flow
- **[LabVIEW Integration](docs/labview_integration.md)**: Hardware interface setup
- **[Usage Examples](docs/usage_examples.md)**: Detailed code examples and workflows

---

## Training Your Own Model

### Prerequisites
- Labeled dataset of STM images (minimum ~500 images, balanced classes)
- Images organized in `data/your_surface/train/good/` and `train/bad/` directories

### Training Script
```bash
# Hyperparameter search
python scripts/run_hyperparameter_search.py \
  --data_dir data/your_surface/ \
  --output_dir models/your_model/

# Train final model with best parameters
python src/cnn_classifier/train.py \
  --config models/your_model/best_config.json \
  --epochs 100 \
  --output models/your_model.h5
```

See [examples/01_train_cnn_model.py](examples/01_train_cnn_model.py) for detailed training workflow.

---

## 🔧 Troubleshooting

As this codebase is being actively ported from a research environment, you may encounter issues. Here are some common problems and solutions:

### Installation Issues

**Problem**: Package conflicts during installation  
**Solution**: Try creating a fresh conda environment with Python 3.9:
```bash
conda create -n stm-fresh python=3.9
conda activate stm-fresh
pip install -r requirements.txt
```

**Problem**: SPIEPy installation fails  
**Solution**: This package may need to be installed from source. Please open an issue with your error message.

### Import Errors

**Problem**: `ModuleNotFoundError` when importing  
**Solution**: Ensure you've installed the package in development mode:
```bash
pip install -e .
```

### Data Format Issues

**Problem**: Cannot load .sxm files  
**Solution**: The code was developed with Nanonis .sxm format. Other SPM file formats may require adaptation. Please open an issue with your file type.

### Found a Bug?

Please [open an issue](https://github.com/yourusername/stm-tip-classifier/issues) with:
1. Your Python version (`python --version`)
2. Operating system
3. Full error message
4. Minimal code to reproduce the problem

I aim to respond to issues within a few days and appreciate your patience as the code is refined for broader use.

---

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_cnn_classifier.py

# Run with coverage
pytest --cov=src tests/
```

---

## Citation

This work is from my PhD thesis:

**Dylan S. Barker**  
*"Automating Scanning Tunnelling Microscopy: A comparative Study of Machine Learning and Deterministic Methods"*  
University of Leeds, 2024

If you use this code in your research, please cite:
```bibtex
@phdthesis{barker2024automation,
  title={Automating Scanning Tunnelling Microscopy: A comparative Study of Machine Learning and Deterministic Methods},
  author={Barker, Dylan S.},
  year={2024},
  school={University of Leeds}
}
```

---

## Contributing

Contributions are welcome! This codebase is being actively developed and improved as it transitions from a research environment to a public repository.

**Ways to contribute:**
- Report bugs or installation issues via [GitHub Issues](https://github.com/yourusername/stm-tip-classifier/issues)
- Suggest features or improvements
- Improve documentation
- Submit pull requests with fixes or enhancements

**Current priorities:**
- Testing across different Python environments
- Improving installation process
- Expanding documentation with more examples
- Adding compatibility for additional STM file formats

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run code formatting
black src/ tests/
flake8 src/ tests/

# Run type checking
mypy src/
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Key Insights

**When to use each method:**

- ✅ **Use CNN** when:
  - Large labeled datasets available (1000+ images)
  - Complex features across multiple surfaces
  - High accuracy priority

- ✅ **Use Cross-Correlation** when:
  - Limited data available
  - Clear reference features exist
  - Interpretability important
  - Fast prototyping needed

**Main Contribution**: Demonstrating that traditional computer vision methods can achieve comparable performance to deep learning for domain-specific scientific applications, with significantly less overhead.

---

## Contact

**Dylan S. Barker**  
- Email: dylan.barker01@gmail.com
- LinkedIn: [linkedin.com/in/dylan-s-barker](https://www.linkedin.com/in/dylan-s-barker)
- Institution: University of Leeds

---

## Acknowledgments

- Supervisor: Dr. Adam Sweetman
- Molecular Nanoscale Physics (MNP) Group, University of Leeds
- Nanonis SPM software and LabVIEW integration support
- SPIEPy library developers

---

## Roadmap

Future enhancements:
- [ ] Implement online learning for tip adaptation
- [ ] Add uncertainty quantification
- [ ] Develop transfer learning pipeline for new surfaces

---

**If you find this work useful, please consider starring the repository!**