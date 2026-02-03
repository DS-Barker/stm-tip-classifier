# Usage Examples

## Table of Contents
1. [Training a New CNN Model](#training-a-new-cnn-model)
2. [Running Cross-Correlation Classifier](#running-cross-correlation-classifier)
3. [Comparing Methods on Your Data](#comparing-methods-on-your-data)
4. [Generating ROC Curves](#generating-roc-curves)
5. [Using the LabVIEW Automation Tool](#using-the-labview-automation-tool)

---

## Training a New CNN Model

### Prerequisites
- Labeled dataset in `data/your_dataset/`
- Images organized: `data/your_dataset/good/` and `data/your_dataset/bad/`

### Step 1: Prepare Data
```python
from src.data_processing import ImageLoader, Augmentation

# Load images
loader = ImageLoader(data_dir='data/si111_7x7/')
X_train, y_train = loader.load_split('train')
X_val, y_val = loader.load_split('validation')

# Augment training data
aug = Augmentation(flip=True, rotate=True)
X_train_aug, y_train_aug = aug.apply(X_train, y_train)
```

### Step 2: Train Model
```python
from src.cnn_classifier import CNNModel

# Create model
model = CNNModel(
    input_shape=(700, 700, 1),
    conv_layers=5,
    filters=[20, 40, 60, 80, 100],
    dense_layers=[32, 64, 128]
)

# Train
history = model.train(
    X_train_aug, y_train_aug,
    X_val, y_val,
    epochs=100,
    batch_size=32
)

# Save
model.save('models/my_custom_model.h5')
```

### Step 3: Evaluate
```python
from src.cnn_classifier import Evaluator

evaluator = Evaluator(model)
accuracy, precision = evaluator.evaluate(X_test, y_test)
evaluator.plot_roc_curve(save_path='results/roc_curves/my_model.png')

print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
```

**Expected output:**
```
Accuracy: 94%
Precision: 91%
AUROC: 0.98
```

---

## Running Cross-Correlation Classifier

### Single Image Classification
```python
from src.deterministic_classifier import CrossCorrelation
import numpy as np
from PIL import Image

# Load reference image (good tip example)
reference = np.array(Image.open('data/reference_images/si111_good.png'))

# Load test image
test_image = np.array(Image.open('data/test/unknown_tip.png'))

# Classify
ccr = CrossCorrelation(reference, threshold=0.99)
result = ccr.classify(test_image)

print(f"Classification: {result['label']}")  # "good" or "bad"
print(f"CCR Value: {result['ccr_value']:.3f}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Batch Classification
```python
from pathlib import Path

# Classify all images in directory
test_dir = Path('data/test_dataset/')
results = []

for img_path in test_dir.glob('*.png'):
    img = np.array(Image.open(img_path))
    result = ccr.classify(img)
    results.append({
        'filename': img_path.name,
        'classification': result['label'],
        'ccr': result['ccr_value']
    })

# Save results
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('results/batch_classification.csv', index=False)
```

---

## Comparing Methods on Your Data

### Run Both Classifiers
```python
from src.cnn_classifier import CNNModel
from src.deterministic_classifier import CrossCorrelation
from sklearn.metrics import classification_report

# Load models
cnn = CNNModel.load('models/si111_7x7_cnn.h5')
ccr = CrossCorrelation(reference_image, threshold=0.99)

# Test on same dataset
cnn_predictions = cnn.predict(X_test)
ccr_predictions = [ccr.classify(img)['label'] for img in X_test]

# Compare
print("CNN Performance:")
print(classification_report(y_true, cnn_predictions))

print("\nCCR Performance:")
print(classification_report(y_true, ccr_predictions))
```

### Visualize Comparison
```python
import matplotlib.pyplot as plt

methods = ['CNN', 'CCR', 'Human']
accuracies = [0.96, 0.90, 0.92]
precisions = [0.92, 0.97, 0.93]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].bar(methods, accuracies)
ax[0].set_ylabel('Accuracy')
ax[1].bar(methods, precisions)
ax[1].set_ylabel('Precision')
plt.savefig('results/method_comparison.png')
```

---

## Generating ROC Curves
```python
from src.cnn_classifier import Evaluator
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get predictions with confidence scores
y_scores = model.predict_proba(X_test)[:, 1]

# Calculate ROC
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'CNN (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Si(111)-7×7 Dataset')
plt.legend()
plt.savefig('results/roc_curves/si111_roc.png', dpi=300)
```

---

## Using the LabVIEW Automation Tool

### From Command Line (Testing)
```bash
# Classify single image
python src/labview_interface/classify_tip.py \
  --image data/test/tip_001.sxm \
  --method CCR \
  --threshold 0.99

# Output:
# {"classification": "good", "confidence": 0.95, "ccr_value": 0.987}
```

### From LabVIEW
1. Open `labview/main_control.vi`
2. Set parameters in front panel:
   - Scan size: 20 nm
   - Classification method: CCR
   - CCR threshold: 0.99
   - Max iterations: 50
3. Click "Run"
4. Monitor progress in output log
5. Tool stops when "good" tip achieved

### Customizing Tip Shaping
Edit parameters in `labview/tip_preparation.vi`:
```
Bias voltage: 0.5-1.0 V (higher = more aggressive)
Indent depth: 600-1000 pm (deeper = more change)
Dwell time: 100-500 ms
```

**Tip**: Start conservative (0.5 V, 600 pm) and increase if changes are too subtle.

---

## Common Workflows

### Workflow 1: Evaluate on New Surface
```python
# 1. Collect small dataset (50-100 images)
# 2. Try CCR first (faster to set up)
reference = load_good_example()
ccr = CrossCorrelation(reference, threshold=0.95)
results = ccr.classify_batch(new_dataset)

# 3. If CCR precision < 90%, train CNN
if precision < 0.90:
    model = train_cnn(new_dataset_labeled)
```

### Workflow 2: Reproduce Thesis Results
```bash
# Run all experiments from thesis
python scripts/reproduce_thesis_results.py

# Outputs:
# - results/roc_curves/si111_roc.png
# - results/roc_curves/bsi_roc.png
# - results/performance_comparison.csv
```

---

## Tips & Best Practices

1. **Reference Image Selection (CCR)**:
   - Use high-quality scan (atomic resolution clear)
   - Crop to single feature (adatom, molecule)
   - Size: 50-100 pixels typical

2. **CNN Training**:
   - Balance dataset (equal good/bad examples)
   - Use validation set to prevent overfitting
   - Data augmentation crucial for small datasets

3. **Threshold Tuning**:
   - CCR threshold 0.99 works for Si surfaces
   - May need 0.95-0.97 for noisier data
   - Plot CCR distribution to choose optimal value

4. **Performance Benchmarks** (Intel i7, GTX 1060):
   - CCR classification: 0.5 seconds/image
   - CNN classification: 2 seconds/image
   - CNN training (100 epochs): 30 minutes