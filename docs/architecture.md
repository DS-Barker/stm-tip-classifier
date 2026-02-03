# System Architecture

## Overview
This project implements automated STM tip classification through three main components:
1. CNN-based classifier (TensorFlow)
2. Deterministic classifier (cross-correlation + circularity)
3. LabVIEW automation interface

## Component Diagram
```
┌─────────────────────────────────────────┐
│         LabVIEW Control System          │
│  (Nanonis SPM, tip preparation tool)    │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│      Python Classification Layer         │
│  ┌─────────────┐    ┌─────────────────┐ │
│  │ CNN Model   │    │ Deterministic   │ │
│  │ (TensorFlow)│    │ (CCR/Circularity)│ │
│  └─────────────┘    └─────────────────┘ │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│         Data Processing Pipeline         │
│  (Image loading, preprocessing, etc.)    │
└─────────────────────────────────────────┘
```

## Data Flow

### Real-time Classification (LabVIEW → Python)
1. LabVIEW acquires STM image (700×700 pixels)
2. Image saved to temp directory
3. Python script called with image path
4. Classification performed (CNN or CCR)
5. Result returned to LabVIEW
6. Decision: accept tip or reshape

### Training Pipeline (Offline)
1. Collect dataset using automated script
2. Manual labeling of "good" vs "bad" tips
3. Data augmentation (flips, rotations)
4. CNN training with hyperparameter search
5. Model evaluation (ROC curves, confusion matrices)
6. Best model saved for deployment

## File Organization
- `src/cnn_classifier/`: Neural network implementation
- `src/deterministic_classifier/`: Cross-correlation methods
- `src/labview_interface/`: Python entry points for LabVIEW
- `labview/`: LabVIEW VIs for experiment control
- `models/`: Trained model weights (.h5 files)

## Key Design Decisions

### Why Two Classification Methods?
- **CNN**: High accuracy (96%), but requires large labeled dataset
- **CCR**: Comparable precision (97%), minimal data needed, transparent
- Thesis contribution: showing when simpler methods suffice

### Image Preprocessing
All images undergo:
1. Plane subtraction (remove tilt)
2. Normalization (0-1 range)
3. Resizing to 700×700 if needed
4. Optional: data augmentation for training

## Dependencies
- TensorFlow 2.10.0 (GPU support optional)
- LabVIEW 2021 with Nanonis interface
- Python 3.9+