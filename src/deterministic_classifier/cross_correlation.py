"""
Cross-Correlation Methods for STM Tip Classification
====================================================

Provides cross-correlation based classification methods for STM images,
including normalized cross-correlation, reference image loading, and
rotation-invariant feature detection.

This module contains deterministic alternatives to CNN classification,
achieving comparable precision (97%) without requiring large labeled datasets.

Key Functions:
- normxcorr2: Normalized 2D cross-correlation (MATLAB port)
- ccr_topn: Find top N cross-correlation matches
- load_ref: Load reference images for comparison
- find_max_CCR: Rotation-invariant CCR with automatic angle correction
- auto_downscaler: Automatically match image resolutions
- find_molecules: Batch feature detection

Usage:
    from src.deterministic_classifier.cross_correlation import ccr_topn, load_ref
    
    # Load reference
    ref = load_ref('Si-B_new.npy')
    
    # Calculate CCR
    ccr_values = ccr_topn(ref, test_image, top_n=5)
    avg_ccr = np.mean(ccr_values)
    
    if avg_ccr >= 0.92:
        print("Good tip quality")

Author: Dylan S. Barker, University of Leeds, 2024
"""

from pathlib import Path
from skimage.transform import warp_polar, rotate
from skimage.registration import phase_cross_correlation
import numpy as np
import math
from scipy.signal import fftconvolve
import cv2
import os


def normxcorr2(template, image, mode="full"):
    """
    Normalized 2D cross-correlation (MATLAB normxcorr2 port).
    
    Python implementation using scipy's fftconvolve and numpy.
    Credit: Sabrewarrior (https://github.com/Sabrewarrior)
    
    Args:
        template (np.ndarray): N-D array template for cross-correlation.
            Must be less than or equal dimensions to image.
        image (np.ndarray): N-D array of input image being compared to template
        mode (str): Convolution mode. Options:
            - 'full' (default): Full discrete linear convolution
            - 'valid': Only elements not relying on zero-padding
            - 'same': Output same size as image
    
    Returns:
        np.ndarray: Cross-correlation map. Size depends on mode parameter.
    
    Note:
        Input arrays should be floating point numbers.
    
    Example:
        >>> ref = np.random.rand(50, 50)
        >>> img = np.random.rand(100, 100)
        >>> ccr_map = normxcorr2(ref, img)
    """
    # Warn if arguments might be swapped
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    # Mean-center inputs
    template = template - np.mean(template)
    image = image - np.mean(image)

    # Perform normalized cross-correlation
    a1 = np.ones(template.shape)
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)
    
    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0
    
    return out

def load_ref(ref_name):
    """
    Load a reference image from a numpy file.
    
    Args:
        ref_name (str or Path): Name of reference image file (.npy).
            If string: looks in 'stm-tip-classifier\src\deterministic_classifier\refs'
            If Path: uses path directly
    
    Returns:
        np.ndarray: Reference image as numpy array
    
    Example:
        >>> ref = load_ref('Si111-cornerhole.npy')
        >>> print(ref.shape)
        (700, 700)
    """
    if type(ref_name) == str:
        ref_path = Path(os.getcwd()).joinpath(r'src\deterministic_classifier\refs').joinpath(ref_name)
    else:
        ref_path = ref_name
        
    return np.load(ref_path)

def ccr_topn(ref, img, top_n=20, min_peak_sep=[20, 20], with_ims=False, 
             with_positions_plus_FM=False, zero_border=True):
    """
    Find top N cross-correlation matches between reference and input image.
    
    Calculates normalized cross-correlation and finds the highest scoring
    local maxima, useful for identifying recurring features or patterns.
    
    Args:
        ref (np.ndarray): Reference image/template
        img (np.ndarray): Input image to search within
        top_n (int): Number of top matches to find. Default: 20
        min_peak_sep (list): [x, y] minimum separation between peaks in pixels.
            Default: [20, 20]
        with_ims (bool): If True, return cropped images of matches. Default: False
        with_positions_plus_FM (bool): If True, return positions and feature map.
            Default: False
        zero_border (bool): If True, zero out image borders to avoid edge effects.
            Default: True
    
    Returns:
        list or tuple: Depending on flags:
            - Default: List of CCR values for top N matches
            - with_ims=True: (ccr_values, cropped_images)
            - with_positions_plus_FM=True: (ccr_values, positions, feature_map)
            - Both flags: (ccr_values, cropped_images, positions, feature_map)
    
    Example:
        >>> ref = load_ref('reference.npy')
        >>> img = np.load('test_image.npy')
        >>> ccr_vals = ccr_topn(ref, img, top_n=5)
        >>> avg_ccr = np.mean(ccr_vals)
        >>> print(f"Average CCR: {avg_ccr:.3f}")
    """
    # Extract peak separation parameters
    min_peak_sep_x = min_peak_sep[0]
    min_peak_sep_y = min_peak_sep[1]

    # Calculate cross-correlation
    ref_sd = ref.shape
    maxima_positions = []
    maxima_vals = []

    ccr_fm = normxcorr2(ref, img)
    ccr_fm = ccr_fm[math.ceil(ref_sd[0]/2):-math.ceil(ref_sd[0]/2), 
                    math.ceil(ref_sd[1]/2):-math.ceil(ref_sd[1]/2)]
    ccr_copy = np.copy(ccr_fm)

    # Zero out border to avoid edge effects
    if zero_border:
        ccr_copy[:2*min_peak_sep_x, :] = 0
        ccr_copy[-(2*min_peak_sep_x):, :] = 0
        ccr_copy[:, :2*min_peak_sep_x] = 0
        ccr_copy[:, -(2*min_peak_sep_x):] = 0

    ccr_sd = ccr_fm.shape

    # Find local maxima
    count = 0
    while len(maxima_positions) < top_n:
        max_pos = np.unravel_index(ccr_copy.argmax(), ccr_fm.shape)
        
        # Check if maximum is not too close to edge
        if (ref_sd[0]/2 <= max_pos[0] <= ccr_sd[0]-ref_sd[0]/2 and  
            ref_sd[1]/2 <= max_pos[1] <= ccr_sd[1]-ref_sd[1]/2):
            maxima_positions.append(max_pos)
            maxima_vals.append(ccr_fm[max_pos])

        # Zero out region around found maximum
        ccr_copy[max_pos[0] - min_peak_sep_x: max_pos[0] + min_peak_sep_x, 
                 max_pos[1] - min_peak_sep_y: max_pos[1] + min_peak_sep_y] = 0

        # Failsafe: stop if no maxima found after 3 attempts
        if len(maxima_positions) == 0:
            count += 1
        if count == 3:
            return [0]
        
    # Return based on requested outputs
    if with_ims and with_positions_plus_FM:
        maxima_ims = []
        for max_pos in maxima_positions:
            maxima_im = img[(max_pos[0]-int(ref_sd[0]/2)):(max_pos[0]+int(ref_sd[0]/2)), 
                           (max_pos[1]-int(ref_sd[1]/2)):(max_pos[1]+int(ref_sd[1]/2))]
            maxima_ims.append(maxima_im)
        return maxima_vals, maxima_ims, maxima_positions, ccr_fm

    if with_ims:
        maxima_ims = []
        for max_pos in maxima_positions:
            maxima_im = img[(max_pos[0]-int(ref_sd[0]/2)):(max_pos[0]+int(ref_sd[0]/2)), 
                           (max_pos[1]-int(ref_sd[1]/2)):(max_pos[1]+int(ref_sd[1]/2))]
            maxima_ims.append(maxima_im)
        return maxima_vals, maxima_ims

    if with_positions_plus_FM:
        return maxima_vals, maxima_positions, ccr_fm

    return maxima_vals

def find_max_CCR(img, refs, refs_angles, input_info, ref_info):
    """
    Find optimal rotation angle for maximum cross-correlation.
    
    Tests multiple reference rotations and uses both iterative angle refinement
    and spherical coordinate transformation to find the best match angle.
    
    Args:
        img (np.ndarray): Input image
        refs (list): List of pre-rotated reference images
        refs_angles (list): Angles corresponding to each reference image
        input_info (list): [scan_size, num_pixels] for input image
        ref_info (list): [scan_size, num_pixels] for reference image
        
    Returns:
        tuple: (max_ccr, optimization_method, optimal_angle, feature_image, feature_location)
            - max_ccr (float): Maximum CCR value found
            - optimization_method (str): 'Iterative', 'Spherical', or 'Same Answer'
            - optimal_angle (float): Best rotation angle in degrees
            - feature_image (np.ndarray): Cropped image of matched feature
            - feature_location (tuple): (y, x) pixel coordinates of feature
    
    Example:
        >>> refs = [rotate(ref, angle) for angle in range(0, 360, 30)]
        >>> angles = list(range(0, 360, 30))
        >>> ccr, method, angle, feature, loc = find_max_CCR(
        ...     img, refs, angles, [10.0, 1000], [10.0, 1000]
        ... )
        >>> print(f"Best CCR: {ccr:.3f} at {angle}° using {method}")
    """
    best_ccr = [0, 'index', 'Position']    
    refs_ds = list(range(len(refs)))
    
    # Find best reference rotation via coarse search
    for i, ref in enumerate(refs):
        # Skip special 'BigRef' marker
        if refs_angles[i] == 'BigRef':
            continue
        
        # Match resolutions
        downscaled_img, downscaled_ref, _ = auto_downscaler(img, input_info, ref, ref_info)
        refs_ds[i] = downscaled_ref
        
        # Calculate CCR
        ccr, maxima_positions, _ = ccr_topn(downscaled_ref, downscaled_img, top_n=1, 
                                            with_positions_plus_FM=True)
        ccr = round(ccr[0], 4)
        
        if ccr > best_ccr[0]:
            best_ccr = [ccr, i, maxima_positions[0]] 

    # Extract feature region
    ref_height, ref_width = refs_ds[best_ccr[1]].shape
    img_small = np.copy(downscaled_img[int(best_ccr[2][0]-ref_height/2):int(best_ccr[2][0]+ref_height/2), 
                                       int(best_ccr[2][1]-ref_width/2):int(best_ccr[2][1]+ref_width/2)])
    
    # Method 1: Spherical rotation correction
    radius = downscaled_ref.shape[0]/2
    img_polar = warp_polar(img_small, radius=radius)
    ref_polar = warp_polar(refs_ds[best_ccr[1]], radius=radius)
    
    shifts, _, _ = phase_cross_correlation(img_polar, ref_polar)
    ref_sphere_rot = rotate(refs_ds[best_ccr[1]], shifts[0])
    spherical_ccr = ccr_topn(ref_sphere_rot, downscaled_img, top_n=1, 
                            with_positions_plus_FM=False)  
    spherical_ccr = round(spherical_ccr[0], 4) 
    
    # Method 2: Iterative rotation refinement
    best_angle = refs_angles[best_ccr[1]]
    current_best_ccr = best_ccr[0]
    
    # Level 1: Coarse search (-30° to +30° in 10° steps)
    level1_angle = 0
    angles_to_check_lv1 = range(-30, 31, 10)
    
    for angle in angles_to_check_lv1:
        rotated_ref = rotate(refs_ds[best_ccr[1]], angle)
        ccr = ccr_topn(rotated_ref, downscaled_img, top_n=1, with_positions_plus_FM=False)
        ccr = round(ccr[0], 4)
        
        if ccr > current_best_ccr:
            current_best_ccr = ccr
            level1_angle = angle

    best_angle += level1_angle
    
    # Level 2: Fine search (-5° to +5° in 1° steps)
    level2_angle = 0
    angles_to_check_lv2 = range(-5, 6, 1)
    
    for angle in angles_to_check_lv2:
        rotated_ref = rotate(refs_ds[best_ccr[1]], level1_angle + angle)
        ccr, maxima_positions, _ = ccr_topn(rotated_ref, downscaled_img, top_n=1, 
                                           with_positions_plus_FM=True)
        ccr = round(ccr[0], 4)
        
        if ccr > current_best_ccr:
            current_best_ccr = ccr
            level2_angle = angle
            
    best_angle += level2_angle
    
    # Return best method
    if current_best_ccr > spherical_ccr:
        return current_best_ccr, 'Iterative', best_angle, img_small, best_ccr[2]
    elif spherical_ccr > current_best_ccr:
        return spherical_ccr, 'Spherical', shifts[0] + refs_angles[best_ccr[1]], img_small, best_ccr[2]
    else:
        return spherical_ccr, 'Same Answer', best_angle, img_small, best_ccr[2]

def auto_downscaler(input_img, input_info, ref, ref_info):
    """
    Automatically match image resolutions by downscaling the higher-resolution image.
    
    Ensures reference and input images have the same pixel-per-distance ratio
    before cross-correlation by downscaling whichever has higher resolution.
    
    Args:
        input_img (np.ndarray): Input image
        input_info (list): [scan_size, num_pixels] for input (either x or y axis)
        ref (np.ndarray): Reference image
        ref_info (list): [scan_size, num_pixels] for reference (either x or y axis)

    Returns:
        tuple: (input_out, ref_out, which_downscaled)
            - input_out (np.ndarray): Potentially downscaled input image
            - ref_out (np.ndarray): Potentially downscaled reference image
            - which_downscaled (str): 'Input', 'ref', or 'Neither'
    
    Example:
        >>> input_img = np.random.rand(1000, 1000)  # 10nm at 1000px
        >>> ref = np.random.rand(500, 500)          # 10nm at 500px
        >>> matched_input, matched_ref, which = auto_downscaler(
        ...     input_img, [10.0, 1000], ref, [10.0, 500]
        ... )
        >>> print(f"Downscaled: {which}")
    """
    which_downscale = "Neither"
    
    input_dist_per_pix = input_info[0] / input_info[1]
    ref_dist_per_pix = ref_info[0] / ref_info[1]
    
    if input_dist_per_pix == ref_dist_per_pix:
        return input_img, ref, which_downscale
    
    elif input_dist_per_pix > ref_dist_per_pix:
        # Input has lower resolution, downscale reference
        which_downscale = "ref"
        scale_factor = ref_dist_per_pix / input_dist_per_pix
        ref_out = cv2.resize(ref, (0, 0), fx=scale_factor, fy=scale_factor)
        return input_img, ref_out, which_downscale
       
    elif input_dist_per_pix < ref_dist_per_pix:
        # Reference has lower resolution, downscale input
        which_downscale = "Input"        
        scale_factor = input_dist_per_pix / ref_dist_per_pix 
        input_out = cv2.resize(input_img, (0, 0), fx=scale_factor, fy=scale_factor)
        return input_out, ref, which_downscale

def find_molecules(input_im, refs, refs_angles, input_info, ref_info, threshold, max_output=5):
    """
    Batch feature detection using iterative CCR with blanking.
    
    Repeatedly applies find_max_CCR to find multiple instances of a feature,
    blanking out each found feature before searching for the next.
    
    Args:
        input_im (np.ndarray): Input image to search
        refs (list): List of pre-rotated reference images
        refs_angles (list): Angles corresponding to reference list
        input_info (list): [scan_size, num_pixels] for input image
        ref_info (list): [scan_size, num_pixels] for reference image
        threshold (float): Minimum CCR value to accept a match
        max_output (int): Maximum number of features to find. Default: 5
        
    Returns:
        tuple: (feature_positions, ccr_values, features)
            - feature_positions (list): [(x, y), ...] pixel coordinates
            - ccr_values (list): CCR value for each feature
            - features (list): Cropped images of each feature
    
    Example:
        >>> positions, ccrs, images = find_molecules(
        ...     img, refs, angles, [10, 1000], [10, 1000],
        ...     threshold=0.85, max_output=10
        ... )
        >>> print(f"Found {len(positions)} molecules above threshold")
    """
    current_ccr = 1.0
    counter = 0
    feature_positions = []
    ccr_values = []
    features = []
    
    mean_val = input_im.mean()
    
    while counter <= max_output:
        counter += 1
    
        ccr, _, _, feature, feature_loc = find_max_CCR(input_im, refs, refs_angles, 
                                                        input_info, ref_info)
        current_ccr = ccr
        
        # Stop if below threshold
        if current_ccr < threshold:
            break
        
        # Store results (swap coordinates to [x, y] format)
        feature_coord = [feature_loc[1], feature_loc[0]]
        ccr_values.append(ccr)
        feature_positions.append(feature_coord)
        features.append(feature)
        
        # Blank out found feature
        shaveoff = 40
        input_im[feature_loc[0] - shaveoff: feature_loc[0] + shaveoff, 
                 feature_loc[1] - shaveoff: feature_loc[1] + shaveoff] = mean_val
    
    return feature_positions, ccr_values, features


# If running as script
if __name__ == "__main__":
    """
    Evaluation mode: Find optimal CCR threshold for good vs bad tip classification.
    
    Processes good and bad tip images, calculates CCR distributions, and generates
    histogram to visualize separation between classes.
    """
    
    OUTPUT_DIR = Path("INSERT OUTPUT DIRECTORY HERE FOR HISTOGRAM AND CSV")
    DATA_DIR = Path("INSERT DATA DIRECTORY HERE WITH IMAGES SPLIT INTO GOOD AND BAD SUB-DIRS")
    REF_FILENAME = 'INSERT REFERENCE IMAGE FILENAME HERE (e.g., Si-B_new.npy)'
    
    OUTPUT_NAME = 'CCR-analysis'
    
    GOOD_DIR = DATA_DIR.joinpath('Good')
    BAD_DIR = DATA_DIR.joinpath('Bad')
    
    # Load reference image
    print("Loading reference image...")
    ref = load_ref(REF_FILENAME)
    
    # Process good images
    print("\nProcessing good images...")
    good_ccrs = []
    good_files = []
    
    for count, img_file in enumerate(GOOD_DIR.iterdir(), 1):
        if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.npy']:
            print(f'\rGood: {count}/{len(list(GOOD_DIR.iterdir()))}', end='', flush=True)
            
            # Load image
            if img_file.suffix == '.npy':
                img = np.load(img_file)
            else:
                from PIL import Image
                img = np.array(Image.open(img_file).convert('L'))
            
            # Calculate CCR (average of top 5 matches)
            ccr_vals = ccr_topn(ref, img, top_n=5, with_ims=False)
            avg_ccr = np.mean(ccr_vals)
            
            good_ccrs.append(avg_ccr)
            good_files.append(img_file.name)
    
    print()
    
    # Process bad images
    print("\nProcessing bad images...")
    bad_ccrs = []
    bad_files = []
    
    for count, img_file in enumerate(BAD_DIR.iterdir(), 1):
        if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.npy']:
            print(f'\rBad: {count}/{len(list(BAD_DIR.iterdir()))}', end='', flush=True)
            
            # Load image
            if img_file.suffix == '.npy':
                img = np.load(img_file)
            else:
                from PIL import Image
                img = np.array(Image.open(img_file).convert('L'))
            
            # Calculate CCR
            ccr_vals = ccr_topn(ref, img, top_n=5, with_ims=False)
            avg_ccr = np.mean(ccr_vals)
            
            bad_ccrs.append(avg_ccr)
            bad_files.append(img_file.name)
    
    print()
    
    # Create DataFrames
    import pandas as pd
    good_df = pd.DataFrame({'filename': good_files, 'ccr': good_ccrs})
    bad_df = pd.DataFrame({'filename': bad_files, 'ccr': bad_ccrs})
    
    # Save CSVs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    good_df.to_csv(OUTPUT_DIR / f'{OUTPUT_NAME}-Goods.csv', index=False)
    bad_df.to_csv(OUTPUT_DIR / f'{OUTPUT_NAME}-Bads.csv', index=False)
    print(f"\nSaved CSVs to {OUTPUT_DIR}")
    
    # Plot histogram
    import matplotlib.pyplot as plt
    
    colours = ['limegreen', 'indianred']
    labels = ['Good Images', 'Bad Images']
    
    data_list = [good_df['ccr'], bad_df['ccr']]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.hist(data_list, 50, density=False, histtype='bar', 
            stacked=True, color=colours, label=labels, alpha=0.7)
    ax.legend()
    ax.set_xlabel('CCR Value')
    ax.set_xlim(0, 1.0)
    ax.set_xticks(np.arange(0, 1.0, 0.1))
    ax.set_ylabel('Counts')
    ax.set_title('CCR Distribution: Good vs Bad Tips')
    ax.grid(alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{OUTPUT_NAME}_histogram.png', dpi=150)
    plt.close()
    print(f"Saved histogram to {OUTPUT_DIR / f'{OUTPUT_NAME}_histogram.png'}")
    
    # Print statistics
    print(f"\nResults:")
    print(f"Good images - Mean CCR: {good_df['ccr'].mean():.3f}, Std: {good_df['ccr'].std():.3f}")
    print(f"Bad images - Mean CCR: {bad_df['ccr'].mean():.3f}, Std: {bad_df['ccr'].std():.3f}")
    print(f"\nSuggested threshold (between means): {(good_df['ccr'].mean() + bad_df['ccr'].mean()) / 2:.3f}")