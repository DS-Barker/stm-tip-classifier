"""
Circularity Measurement for STM Images
======================================

Measures circularity and roundness of features in STM images using multiple
thresholding values to assess tip quality.

This is a deterministic alternative to CNN classification, particularly useful
for identifying double tips which appear as non-circular features.

Usage:
    from src.deterministic_classifier.circularity import measure_circularity
    
    # Measure single image
    result = measure_circularity('path/to/image.png')
    print(f"Circularity: {result['circularity']:.3f}")
    print(f"Roundness: {result['roundness']:.3f}")
    
    # Batch process
    results = measure_circularity_batch('data/test_images/')

Author: Dylan S. Barker, University of Leeds, 2024
"""

import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import diplib as dip
import pandas as pd


def measure_circularity(image_path, thresholds=None):
    """
    Measure circularity and roundness of features in an STM image.
    
    Uses multiple threshold values to identify features and calculates their
    circularity (how close to a perfect circle) and roundness metrics. This
    helps identify double tips, which appear as elongated or irregular features.
    
    Args:
        image_path (str or Path): Path to image file
        thresholds (list of float, optional): Threshold values between 0-1.
            Multiple thresholds are averaged for robust measurement.
            Default: [0.4, 0.5, 0.6, 0.7]
    
    Returns:
        dict: Measurement results containing:
            - 'circularity': Average circularity score (0-1, 1 = perfect circle)
            - 'roundness': Average roundness score (0-1, 1 = perfect circle)
            - 'filename': Name of the analyzed file
            - 'valid': Whether measurement was successful
    
    Example:
        >>> result = measure_circularity('data/test/image001.png')
        >>> print(f"Circularity: {result['circularity']:.3f}")
        >>> if result['circularity'] > 0.8:
        ...     print("Likely a good tip (circular feature)")
        ... else:
        ...     print("Likely a bad tip (non-circular)")
    
    Notes:
        - Single pixels are ignored (not counted as circles)
        - Largest connected feature is used for measurement
        - Image is converted to grayscale automatically
    """
    if thresholds is None:
        thresholds = [0.4, 0.5, 0.6, 0.7]
    
    image_path = Path(image_path)
    
    # Import the image
    img = Image.open(image_path).convert('L')
    
    circularity = 0
    roundness = 0
    valid = True
    
    # Convert thresholds to 0-255 range
    threshs = [255 * t for t in thresholds]
    
    try:
        for thresh in threshs:
            # Apply threshold
            tmp_img = img.point(lambda p: 255 if p > thresh else 0)
            tmp_img = np.array(tmp_img.convert('L')) / 255
            tmp_img = np.pad(tmp_img, pad_width=1, mode='constant', constant_values=0)
            
            # Label connected components
            labels = dip.Label(tmp_img > 0)
            msr = dip.MeasurementTool.Measure(labels, features=["Roundness", "Circularity"])
    
            # Find largest feature (ignore single pixels)
            # This stops the script counting spurious pixels as perfect circles
            areas = []
            for obj in np.asarray(msr['SolidArea']):
                areas.append(obj[0])
    
            circle_index = np.array(areas).argmax() + 1
            
            circularity += msr[circle_index]["Circularity"][0]
            roundness += msr[circle_index]["Roundness"][0]
        
        # Average over all thresholds
        circularity /= len(threshs)
        roundness /= len(threshs)
        
    except Exception as e:
        print(f"Warning: Could not measure {image_path.name}: {e}")
        valid = False
        circularity = 0.0
        roundness = 0.0
    
    return {
        'circularity': circularity,
        'roundness': roundness,
        'filename': image_path.name,
        'valid': valid
    }


def measure_circularity_batch(data_dir, thresholds=None, output_csv=None):
    """
    Measure circularity for all images in a directory.
    
    Args:
        data_dir (str or Path): Directory containing images
        thresholds (list of float, optional): Threshold values. 
            Default: [0.4, 0.5, 0.6, 0.7]
        output_csv (str or Path, optional): Path to save results CSV.
            If None, results not saved to file.
    
    Returns:
        pd.DataFrame: Results with columns:
            - 'filename': Image filename
            - 'circularity': Circularity measurement
            - 'roundness': Roundness measurement
            - 'valid': Whether measurement succeeded
    
    Example:
        >>> results_df = measure_circularity_batch('data/test_images/')
        >>> print(results_df.describe())
        >>> 
        >>> # Find images with poor circularity
        >>> bad_tips = results_df[results_df['circularity'] < 0.5]
        >>> print(f"Found {len(bad_tips)} likely bad tips")
    """
    data_dir = Path(data_dir)
    
    if thresholds is None:
        thresholds = [0.4, 0.5, 0.6, 0.7]
    
    results = []
    
    # Get all image files
    image_files = [f for f in data_dir.iterdir() 
                   if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']]
    
    total = len(image_files)
    
    for count, image_path in enumerate(image_files, 1):
        print('', end=f'\rProcessing image {count}/{total}', flush=True)
        
        result = measure_circularity(image_path, thresholds=thresholds)
        results.append(result)
    
    print()  # New line after progress
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save if output path provided
    if output_csv is not None:
        output_csv = Path(output_csv)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to: {output_csv}")
    
    return df


# If running as script
if __name__ == "__main__":

    '''
    Used for evaluation of Good and Bad images, needed to find a threshold at which a feature is considered circular enough to be taken with an ideal tip. 
    '''

    OUTPUT_DIR = Path("INSERT OUTPUT DIRECTORY HERE FOR HISTOGRAM AND CSV")
    DATA_DIR = Path("INSERT DATA DIRECTORY HERE WITH IMAGES SPLIT INTO GOOD AND BAD SUB-DIRS")     
    
    OUTPUT_NAME = 'Roundness-Circularity-meanthresh-0.4-0.7'
    
    GOOD_DIR = DATA_DIR.joinpath('Good')
    BAD_DIR = DATA_DIR.joinpath('Bad')
    
    threshs_x = [0.4, 0.5, 0.6, 0.7]
    
    # Process good images
    print("Processing good images...")
    good_df = measure_circularity_batch(
        GOOD_DIR, 
        thresholds=threshs_x,
        output_csv=OUTPUT_DIR / f'{OUTPUT_NAME}-Goods.csv'
    )
    
    # Process bad images
    print("Processing bad images...")
    bad_df = measure_circularity_batch(
        BAD_DIR,
        thresholds=threshs_x,
        output_csv=OUTPUT_DIR / f'{OUTPUT_NAME}-Bads.csv'
    )
    
    # Plot histogram
    colours = ['limegreen', 'indianred']
    labels = ['Good Images', 'Bad Images']
    
    data_list = [good_df['circularity'], bad_df['circularity']]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.hist(data_list, 100, density=False, histtype='bar', 
            stacked=True, color=colours, label=labels)
    ax.legend()
    ax.set_xlabel('Circularity')
    ax.set_xlim(0, 0.5)
    ax.set_xticks(np.arange(0, 0.5, 0.05))
    ax.set_ylabel('Counts')
    ax.grid(alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Hist.png')
    plt.close()
    
    print(f"\nResults:")
    print(f"Good images - Mean circularity: {good_df['circularity'].mean():.3f}")
    print(f"Bad images - Mean circularity: {bad_df['circularity'].mean():.3f}")