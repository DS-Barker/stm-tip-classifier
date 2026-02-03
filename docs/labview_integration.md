# LabVIEW Integration Guide

## Overview
The LabVIEW automation tool provides real-time tip classification during STM experiments. It interfaces with Nanonis SPM software and calls Python classifiers.

## Requirements
- LabVIEW 2021 or later
- Nanonis SPM software
- Nanonis Programming Interface V5 (5.12.203.120 or later)
- Python 3.9+ with packages installed
- Access to experimental STM system

## LabVIEW File Structure

### Main VIs
- **`main_control.vi`**: Top-level automation loop
  - Acquires image
  - Calls Python classifier
  - Makes reshaping decision
  - Logs results

- **`tip_preparation.vi`**: Tip shaping subroutine
  - Applies bias pulse
  - Indents tip into surface
  - Configurable parameters

- **`data_acquisition.vi`**: Image capture wrapper
  - Interfaces with Nanonis
  - Saves image to disk
  - Returns file path

## Setting Up the Automation Tool

### 1. Configure Python Path
In `python_caller.vi`, set the Python executable path:
```
Default: C:\Users\YourName\anaconda3\envs\stm\python.exe
```

### 2. Set Classification Parameters
In `main_control.vi` front panel:
- **Classifier Type**: "CNN" or "CCR"
- **CCR Threshold**: 0.99 (default)
- **Tip Shaping Voltage**: 0.5-1.0 V
- **Indent Depth**: 600-1000 pm
- **Max Iterations**: 50 (stop after 50 failed attempts)

### 3. Prepare Reference Image (CCR only)
- Acquire clean image with good tip
- Crop representative feature (e.g., Si adatom, C60 molecule)
- Save as `reference.png` in `data/reference_images/`
- Path set in LabVIEW config

## Python Interface

### Command-Line Call Format
LabVIEW calls Python scripts via:
```bash
python src/labview_interface/classify_tip.py \
  --image /path/to/image.sxm \
  --method CCR \
  --threshold 0.99 \
  --output /path/to/result.json
```

### Output Format
Python returns JSON:
```json
{
  "classification": "good",
  "confidence": 0.95,
  "method": "CCR",
  "ccr_value": 0.987,
  "timestamp": "2024-02-03T14:23:01"
}
```

LabVIEW parses this to make decisions.

## Workflow Example

### Automated Tip Preparation Session
1. User starts `main_control.vi`
2. Initial scan acquired
3. Python classifies → "bad" (CCR = 0.82)
4. LabVIEW triggers `tip_preparation.vi`
   - Pulls back tip
   - Moves to clean surface area
   - Indent: 800 pm, 0.7 V
   - Returns to scan position
5. New scan acquired
6. Python classifies → "good" (CCR = 0.99)
7. **Success!** Experiment can proceed

Average time: 9.3 minutes (12 shaping attempts)

## Troubleshooting

### Python Script Not Found
- Check Python path in `python_caller.vi`
- Verify `src/labview_interface/classify_tip.py` exists
- Test in terminal: `python classify_tip.py --help`

### Classification Always Returns "bad"
- CCR threshold too high → lower to 0.95
- Reference image mismatch → recapture reference
- Check image preprocessing (plane subtraction)

### LabVIEW Crashes During Tip Shaping
- Voltage/indent too aggressive → reduce parameters
- Surface contaminated → move to clean terrace
- Check Nanonis connection

## Advanced: Custom Classifiers

To add a new classification method:
1. Implement in `src/deterministic_classifier/your_method.py`
2. Add entry point in `src/labview_interface/classify_tip.py`
3. Update `main_control.vi` dropdown menu
4. Test standalone before integrating

## Performance Notes
- Image acquisition: 30-60 seconds
- Python classification: 0.5-2 seconds (CCR faster than CNN)
- Tip shaping: 10-20 seconds
- Total cycle: ~1 minute per iteration