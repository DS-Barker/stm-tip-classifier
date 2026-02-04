"""
Deterministic Classifier Evaluation
====================================

Evaluates cross-correlation (CCR) based tip classification on a test dataset.

Usage:
    from src.deterministic_classifier.evaluate import evaluate_deterministic
    
    # Evaluate CCR classifier
    results = evaluate_deterministic(
        test_data_dir='data/bsi/processed/test',
        sxm_dir='data/bsi/raw/sxm',
        ref_filename='Si-B_new.npy',
        ccr_threshold=0.92
    )
    
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Precision: {results['precision']:.2%}")

Author: Dylan S. Barker, University of Leeds, 2024
"""

import numpy as np
import pandas as pd

from statistics import mean
from pathlib import Path
import time
import os
import site

from ..data_processing.utils import auto_unit
from ..data_processing.nanonis_utils import get_image_data, Get_ranges
from .cross_correlation import load_ref, ccr_topn


def convert_to_preferred_format(sec):
    """Convert seconds to HH:MM:SS format."""
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60
    return "%02d:%02d:%02d" % (hour, min, sec)


def evaluate_deterministic(test_data_dir, sxm_dir, ref_filename, ccr_threshold=0.92):
    """
    Evaluate deterministic CCR-based classifier on test data.
    
    Classifies STM images using cross-correlation ratio (CCR) and compares
    against ground truth labels to calculate accuracy and precision.
    
    Args:
        test_data_dir (str or Path): Directory containing test data with Good/Bad subdirectories
        sxm_dir (str or Path): Directory containing original .sxm files
        ref_filename (str): Filename of reference image (.npy file)
        ccr_threshold (float): CCR threshold for classification. 
            Values >= threshold are classified as "Good". Default: 0.92
    
    Returns:
        dict: Evaluation results containing:
            - 'accuracy': Overall classification accuracy
            - 'precision': True positive precision (Good predictions that are correct)
            - 'total_samples': Total number of images evaluated
            - 'true_positives': Number of correct "Good" predictions
            - 'total_positives': Total "Good" predictions made
            - 'evaluation_time': Time taken for evaluation
            - 'ccr_threshold': Threshold used
    
    Example:
        >>> results = evaluate_deterministic(
        ...     test_data_dir='data/bsi/processed/test',
        ...     sxm_dir='data/bsi/raw/sxm',
        ...     ref_filename='Si-B_new.npy',
        ...     ccr_threshold=0.92
        ... )
        >>> print(f"Accuracy: {results['accuracy']:.2%}")
        >>> print(f"Precision: {results['precision']:.2%}")
    """
    # Find the start time
    initial_t = time.perf_counter()
    
    test_data_dir = Path(test_data_dir)
    sxm_dir = Path(sxm_dir)
    
    # Import the reference image
    ref = load_ref(ref_filename)
    
    count_acc = 0
    count_tp_prec = 0
    total_positive_preds = 0
    total = 0
    counter = 0
    
    # Count total images
    for i in os.listdir(test_data_dir):
        total += len(os.listdir(test_data_dir.joinpath(i)))
    
    # Evaluate each image
    for folder in os.listdir(test_data_dir):
        CURRENT_FOLDER = test_data_dir.joinpath(folder)
        for file in os.listdir(CURRENT_FOLDER):
            counter += 1
            
            print('', end=f'\rWorking on image {counter}/{total}', flush=True)
            
            CURRENT_FILE = sxm_dir.joinpath(file).with_suffix('.sxm')
    
            # Get the image data
            img = get_image_data(CURRENT_FILE, creepcut=True, thresh_type='median_thresh')
    
            Scan_pixels, Scan_range = Get_ranges(CURRENT_FILE)
    
            dist_per_pixel = Scan_range[0]/Scan_pixels[0]
            prefix, factor = auto_unit(dist_per_pixel, 'm')
    
            # Calculate the CCR 
            maxima_vals = ccr_topn(ref, img, top_n=5, with_ims=False)
            CCR = round(mean(maxima_vals), 3)

            # # Calculate the FRC - deprecated
            # final_FRC = PF.get_frc_res(img, pixel_width= dist_per_pixel*factor)      
            # FRC = final_FRC['True Res']
    
            # Classify based on CCR threshold
            if CCR >= ccr_threshold:
                det_label = 'Good'
                total_positive_preds += 1
            else: 
                det_label = 'Bad'
            
            # Check if prediction is correct
            if det_label == folder:
                count_acc += 1
    
            # Check if true positive
            if det_label == 'Good' and folder == 'Good':
                count_tp_prec += 1
    
            print(f'CCR = {CCR} Label = {det_label} Actual Label = {folder}')
    
    # Calculate metrics
    accuracy = count_acc / total
    precision = count_tp_prec / total_positive_preds if total_positive_preds > 0 else 0.0
    
    print(f'\nTotal accuracy: {accuracy:.2%}')
    print(f'Total true positive precision: {precision:.2%}')
    
    # Find the final time
    final_t = time.perf_counter()
    total_time = final_t - initial_t
    
    print('*'*20)
    print(f'Total time taken for evaluation was {convert_to_preferred_format(total_time)}')
    print('*'*20)
    
    # Return results as dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'total_samples': total,
        'true_positives': count_tp_prec,
        'total_positives': total_positive_preds,
        'evaluation_time': total_time,
        'ccr_threshold': ccr_threshold
    }


# If running as script (not imported)
if __name__ == "__main__":
    # Change these later to fit with Git structure
    DATA_DIR = Path("INSERT DATA DIRECTORY HERE")
    SXM_DIR = Path("INSERT RAW SXM IMAGE DIRECTORY HERE")
    
    REF_FILENAME = Path(__file__).parent.joinpath(r'refs/Si(111)-corerhole.npy')
    
    # Set the thresholds for CCR (and FRC?) which defines a good tip. 
    CCR_thresh = 0.92
    # FRC_thresh = 125
    
    results = evaluate_deterministic(
        test_data_dir=DATA_DIR,
        sxm_dir=SXM_DIR,
        ref_filename=REF_FILENAME,
        ccr_threshold=CCR_thresh
    )
    
    print(f"\nFinal Results:")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Precision: {results['precision']:.2%}")
    print(f"True Positives: {results['true_positives']}/{results['total_positives']}")
    print(f"Total Samples: {results['total_samples']}")