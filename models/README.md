# Trained Models
## STM Tip Classifier Model Repository

This directory contains trained CNN models, training histories, and performance visualisations from hyperparameter searches and final model training.

---

## Directory Structure

```
models/
├── README.md                          # This file
├── model_configs/                     # Model configuration files
│   ├── si111_7x7_config.json
│   └── bsi_config.json
├── thesis_models/                     # Final models from thesis (committed)
│   ├── si111_7x7_final.h5           # 96% accuracy - Si(111)-7×7
│   ├── bsi_final.h5                  # 90% accuracy - B:Si
│   └── training_histories/
│       ├── si111_7x7_history.csv
│       └── bsi_history.csv
```

---

## Thesis Models (Included in Repository)

These are the final, validated models from the PhD thesis achieving human-expert level performance.

### Si(111)-7×7 Surface Classifier
**File**: `thesis_models/si111_7x7_final.h5`

| Metric | Value |
|--------|-------|
| **Accuracy** | 96% |
| **Precision** | 92% |
| **Architecture** | 5 conv layers [20, 40, 80, 160, 320] + 3 dense [32, 64, 128] |
| **Input Size** | 700×700 px |
| **Training Epochs** | 100 |
| **Dataset** | 1000 images (500 good, 500 bad) |

**Usage**:
```python
from tensorflow import keras

model = keras.models.load_model('models/thesis_models/si111_7x7_final.h5')
```

### B:Si Surface Classifier
**File**: `thesis_models/bsi_final.h5`

| Metric | Value |
|--------|-------|
| **Accuracy** | 90% |
| **Precision** | 97% |
| **Architecture** | Same as Si(111)-7×7 |
| **Input Size** | 700×700 px |
| **Training Epochs** | 100 |
| **Dataset** | 800 images (400 good, 400 bad) |

---

## Performance Histories

Training and performance visualisations are not included here, however full histories for each plot whilst training are. Metrics can be extracted if needed. 

```
thesis_models/
├── thesis_models/
│   ├── training_histories
│     ├── bsi_history.csv
│     └── si111_7x7_history.csv
```

---

## Hyperparameter Search Results

Results from systematic architecture searches are **not committed to git** due to size.

### Search Parameters Tested

**Convolutional Layers**:
- Number of layers: 3, 4, 5
- Starting filters: 16, 20, 32
- Filter progression: doubling (16→32→64→128)

**Dense Layers**:
- Number of layers: 1, 2, 3
- Starting neurons: 32, 64, 128, 256
- Neuron progression: doubling

**Total models trained**: 108 per surface (3×3×3×4 combinations)

### Accessing Search Results

Hyperparameter search models are large (50-200MB each) and not included. To reproduce:

**Run your own search**:
   ```bash
   python scripts/hyperparameter_search.py --surface si111_7x7 --epochs 30
   ```

---

## Model Naming Convention

Models follow this naming scheme:
```
{epochs}epochs_conv{n}x{filters}_dense{n}x{neurons}.h5
```

**Examples**:
- `30epochs_conv5x20_dense3x32.h5` - 30 epochs, 5 conv layers starting at 20 filters, 3 dense layers starting at 32 neurons
- `100epochs_conv4x16_dense2x64.h5` - 100 epochs, 4 conv layers starting at 16 filters, 2 dense layers starting at 64 neurons

---

## Model Configuration Files

Configuration files store model architectures and training parameters in JSON format:


**Usage**:
```python
import json
from src.cnn_classifier.model import build_model

# Load config
with open('models/model_configs/si111_7x7_config.json') as f:
    config = json.load(f)

# Build model from config
model = build_model(**config['architecture'])
```

## Loading Models

### Python (TensorFlow/Keras)
```python
import tensorflow as tf
from pathlib import Path

# Load model
model_path = Path('models/thesis_models/si111_7x7_final.h5')
model = tf.keras.models.load_model(model_path)

# Check architecture
model.summary()

# Make predictions
predictions = model.predict(test_images)
```

### Using Helper Functions
```python
from src.cnn_classifier.model import get_thesis_architecture

# Load pre-configured thesis model architecture
model = get_thesis_architecture('si111_7x7')
model.load_weights('models/thesis_models/si111_7x7_final.h5')
```

---

## Regenerating Models

If you need to retrain models from scratch:

### Recreate Thesis Models
```bash
# Train Si(111)-7×7 model
python scripts/train_final_model.py \
  --surface si111_7x7 \
  --config models/model_configs/si111_7x7_config.json \
  --epochs 100 \
  --output models/thesis_models/si111_7x7_retrained.h5
```

### Run Hyperparameter Search
```bash
# Full search (takes hours!)
python scripts/hyperparameter_search.py \
  --surface si111_7x7 \
  --conv_layers 3 4 5 \
  --conv_filters 16 20 32 \
  --dense_neurons 32 64 128 256 \
  --dense_layers 1 2 3 \
  --epochs 30
```

---

## Citation

If you use these models in your research, please cite:

```bibtex
@phdthesis{barker2024stm,
  title={Automating Scanning Tunnelling Microscopy: A comparative Study of Machine Learning and Deterministic Methods},
  author={Barker, Dylan S.},
  year={2024},
  school={University of Leeds}
}
```

---

## Related Resources

- **Training scripts**: `scripts/`
- **Model architectures**: `src/cnn_classifier/model.py`
- **Evaluation tools**: `src/cnn_classifier/evaluate.py`
- **Results & analysis**: `results/`
- **Documentation**: `docs/`

---

## Important Notes

1. **Model compatibility**: Models trained with TensorFlow 2.10.0. May not work with other versions.
2. **Input requirements**: Images must be grayscale, normalized to [0, 1], shape (700, 700, 1) for thesis models.
3. **Class encoding**: 0 = Good tip, 1 = Bad tip (or vice versa depending on data generator)
4. **GPU recommended**: Training on CPU takes 10-20x longer.

---

## Questions?

For questions about models or to request specific model files:
- **Author**: Dylan S. Barker
- **Email**: dylan.barker01@gmail.com
- **Institution**: University of Leeds

---

**Last Updated**: February 2026