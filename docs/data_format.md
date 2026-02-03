# Data Format Specification

## STM Image Files (.sxm)

Nanonis SXM format contains:
- Header: Scan parameters (bias, current, size)
- Data: 2D array of height values

### Loading SXM Files
```python
from SPIEPy import load_sxm

data, metadata = load_sxm('scan_001.sxm')
# data: numpy array (700, 700)
# metadata: dict with scan parameters
```

## Dataset Organization

### Training Data Structure
```
data/
├── si111_7x7/
│   ├── train/
│   │   ├── good/  (500 images)
│   │   └── bad/   (500 images)
│   ├── validation/
│   │   ├── good/  (100 images)
│   │   └── bad/   (100 images)
│   └── test/
│       ├── good/  (100 images)
│       └── bad/   (100 images)
```

### Labels
Binary classification:
- **"good"**: Atomic resolution clear, no double tips
- **"bad"**: Blurry, double tip, or other artifacts

## Preprocessing Pipeline
1. Load raw .sxm file
2. Plane subtraction (remove sample tilt)
3. Normalize to [0, 1]
4. Resize to 700×700 (if needed)
5. Convert to grayscale numpy array