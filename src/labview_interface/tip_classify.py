import cv2
import numpy as np

from pathlib import Path
from statistics import mean
from skimage.transform import warp_polar, rotate
from skimage.registration import phase_cross_correlation
import sys

from ..data_processing.nanonis_utils import get_image_data, get_ranges, detect_tip_change, flatten_by_line
from ..deterministic_classifier.frc_deprecated import get_frc_res 
from ..deterministic_classifier.cross_correlation import load_ref, ccr_topn

''' Script to run through a input folder rull of sxm scans split images into two folders depending on thresholds. 
'''

file = Path(sys.argv[1])

which_downscale = sys.argv[2]
downscale_factor = float(sys.argv[3])

ref_filename = sys.argv[4]

# Define the thresholds
ccr_threshold = 0.88
FRC_threshold = 85

ref = load_ref(ref_filename)

# Import the image
if file.suffix == '.sxm':
    img = get_image_data(file)

    scan_pixels, scan_range, _ = get_ranges(file)
    pixel_width = scan_range[0]/scan_pixels[0]

    # Downscale the refernce image if it is higher resolution than the input image
    if which_downscale == 'Ref':
        ref = cv2.resize(ref, (0, 0), fx=downscale_factor, fy=downscale_factor)

    # Downscale input image if it is higher resoluttion than the reference image
    if which_downscale == 'Input':
        img = cv2.resize(img, (0,0), fx=downscale_factor, fy=downscale_factor )

    # Check for tip changes
    tip_change_detected = detect_tip_change(img) 

    irows, icols = img.shape
    bottom_cutoff = int(irows / 36)               # Cuts off the equvalent of first 20 rows in a 720px scan 

    img, _ = flatten_by_line(img)
    img = img[bottom_cutoff:, bottom_cutoff:]                         # Currently the first 20 lines on the bottom are sliced off due to a small amount of creep, however this may not be essential
    img = img - img.mean()

    median_height = np.mean(img, axis=(0, 1))
    max_height = np.max(img, axis=(0, 1))
    min_height = np.min(img, axis=(0, 1))

    vmax = median_height + ((max_height - median_height) * .65)
    vmin = median_height - ((median_height - min_height) * .95)

    img = np.clip(img, vmin, vmax)

    # Find Rotation
    maxima_vals, maxima_ims = ccr_topn(ref, img, top_n=1, with_ims=True)

    if which_downscale == 'Ref':
        radius = maxima_ims[0].shape[0]/2
    elif which_downscale == 'Input':
        radius = ref.shape[0]/2
    else:
        radius = ref.shape[0]/2

    img_polar = warp_polar(maxima_ims[0], radius = radius)
    ref_polar = warp_polar(ref, radius= radius)

    shifts, error, phasediff = phase_cross_correlation(img_polar, ref_polar)

    # DO I NEED A BOUNDARY CONDITION HERE?
    img = rotate(img, shifts[0])

if not tip_change_detected:                                 # There's no point in doing everything else if the image contains a tip change

    # Do the ccr thing
    maxima_vals = ccr_topn(ref, img, top_n=5, with_ims=False)
    ccr = round(mean(maxima_vals), 2)

    # Do the FRC thing
    final_FRC = get_frc_res(img, pixel_width)      # Would be better to calculate the pixel width for this but for our images it doens't matter much
    FRC = final_FRC['True Res']


    if FRC <= FRC_threshold and FRC != 0 and ccr >= ccr_threshold:
        Tip_good = True
    else:
        Tip_good = False

    print(f'{tip_change_detected} {Tip_good} {FRC} {ccr}')
else: 
    Tip_good = False
    print(f'{tip_change_detected} {Tip_good} {None} {None}')



        
