from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import access2thematrix
import pySPM
import spiepy
import cv2
import statistics as st

def Get_ranges(filename):

    ''' Function which returns the scan range and scan pixel values for an sxm file. 

    Args: 
        filename::str/path
            Input path for sxm file 

    Returns:
        Scan_pixels::[int,int]
            Number of pixels per each row/column in a scan.
        Scan_range:[float,float]
            Physical distance in the scan window. 
        Scan_offset::[float, float]
            Physical offset of the scan window (top left to be consistent with matplotlib) 
            
    Edits: 
        19/04/2024 - Adding in the scan offset being output as well to get out absolute coordinates 
    '''

    file_path = Path(filename)

    Scan_pixels = [None, None]
    Scan_range = [None, None]
    Scan_offset = [None, None]
        
    if file_path.exists:
        with open(file_path, 'rb') as fid:
            fid = fid.readlines()
    else:
        print ("File does not exist")
        return Scan_pixels, Scan_range, Scan_offset
    
    Stop = fid.index(b':SCANIT_END:\n')
    Endline = Stop + 2
    Gap = Stop - 1

    for i in range(0, Endline):
        if i == Gap:
            continue
        s1 = fid[i]
        s1 = s1.decode('utf-8')
        if s1.endswith(':\n') == True:
            
            if s1 == ':SCAN_PIXELS:\n':
                Scan_pixels = fid[i+1].decode('utf-8').split()
                Scan_pixels = [int(x) for x in Scan_pixels]
            elif s1 == ':SCAN_RANGE:\n':
                Scan_range = fid[i+1].decode('utf-8').split()
                Scan_range = [float(x) for x in Scan_range]
            elif s1 == ':SCAN_OFFSET:\n':
                Scan_offset = fid[i+1].decode('utf-8').split()
                Scan_offset = [float(x) for x in Scan_offset]
                
                Scan_offset[0] = Scan_offset[0] - Scan_range[0]/2
                Scan_offset[1] = Scan_offset[1] - Scan_range[1]/2
                
    return Scan_pixels, Scan_range, Scan_offset

def get_image_data(filepath, channel = 'Z', thresh_type='raw_flat', sig = 2, creepcut=False, phys_thresh_vals = None):

    ''' Function to take in sxm or mtrx files and output the image information as well as a few different flattening versions. 
    
    Args: 
        filepath::str/path 
            path to image file
        channel::str
            Imaging channel which we want to extract. Any of the channels included in the sxm files should work, main one's being 'Z' and 'OC_M1_Freq._Shift'
        thresh_type::str/[str]
            Option of which type of thresholding the returned image should take. Options are raw, raw_flat, mean_thresh, median_thresh, phys_thresh. Can also use a list of strings to output multiple images in the order specified. 
        sig::float
            Used for mean and median thresholding: multiple of standard deviations around the mean/median used as upper and lower bounds of the threshold 
        creepcut::bool
            True to cut off the first 2.7% of the image (equating to 20 lines from a 720px image) 
        phys_thresh_vals:: [float, float]
            Used for physical thresholding: physical height above and below the mean height used as bounds of the threshold. Upper can be set to 'max' for max height and lower can be set to 'min' for min. 
    
    Returns:
        Image::arr/[arr]
            Array containing the image values or list of arrays if thresh_type was given as a list. 

    edited 15/09/2022 to change thresholding to median centered sigma thresholding. 

    edited 21/10/2022 to add physical thresholding

    edited 31/10/2022 to output only the desired images by adding thresh_type argument. 

    edited 09/12/2022 to include max and min in the physical thresholding

    edited 09/06/2023 change to creepcut to be a percentage of the image rather than set number of lines. 

    '''
    
    filepath = Path(filepath)

    if channel == 'Z':

        # Only need to bother with this is we're looking at the Z, but even so probably a worthless bit of code. 
        
        if filepath.suffix == '.Z_mtrx':
            mtrx_data = access2thematrix.MtrxData()
            data_file = r'{}'.format(filepath)
            traces, _ = mtrx_data.open(data_file)
            im, _ = mtrx_data.select_image(traces[0])
            img = im.data


    if filepath.suffix == '.sxm':
        S = pySPM.SXM(filepath)
        Z_data = S.get_channel(channel).show()   #  type: ignore
        rawZ= Z_data._A                      #  Takes the raw data out of the SPM defined data structure into a simple array
        img = np.array(rawZ)
        # img = np.flip(img, 0)

        # Find NaN value rows and remove
        img = img[~np.isnan(img).any(axis=1)]

    else:
        raise Exception("Image is not of the correct type.")

    plt.close()

    if creepcut:
        cut_amount = int(img.shape[0]*0.027)
        img = img[cut_amount:, cut_amount:]
        # img=img[20:,20:]

    if type(thresh_type) == list:

        output_ims = []

        for thresh in thresh_type:

            if thresh == 'raw':
                output_ims.append(img)
                continue

            if thresh == 'raw_norm':
                img_norm = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX) # type: ignore
                output_ims.append(img_norm)
                continue

            if thresh == 'raw_plane_flat':
                img_flat, _ = spiepy.flatten_xy(img)
                # mask, _ = spiepy.mask_by_troughs_and_peaks(img_flat)
                # img_flat, _ = spiepy.flatten_xy(img, mask)
                img_flat_norm = cv2.normalize(img_flat, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX) # type: ignore
                output_ims.append(img_flat_norm)

            img_flat, img_diff = flatten_by_line(img)
            img_flat = img_flat - img_flat.mean()

            if thresh == 'raw_flat':
                img_flat_norm = cv2.normalize(img_flat, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX) # type: ignore
                output_ims.append(img_flat_norm)
                continue

            if thresh == 'raw_flat_nonorm':
                output_ims.append(img_flat)
                continue
                
            median_height = np.median(img_flat, axis=(0, 1))
            mean_height = np.mean(img_flat, axis=(0, 1))
            max_height = np.max(img_flat, axis=(0, 1))
            min_height = np.min(img_flat, axis=(0, 1))
            sigma = np.std(img_flat, axis=(0,1))

            if thresh == 'mean_thresh':
                vmax_mean = mean_height + (sigma * sig)
                vmin_mean = mean_height - (sigma * sig)
                img_flat_mean = np.clip(img_flat, vmin_mean, vmax_mean)
                mean_thresh = cv2.normalize(img_flat_mean, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX) # type: ignore

                output_ims.append(mean_thresh)
                continue

            if thresh == 'median_thresh':
                vmax_med = median_height + (sigma * sig)
                vmin_med = median_height - (sigma * sig)    
                img_flat_median = np.clip(img_flat, vmin_med, vmax_med)
                median_thresh = cv2.normalize(img_flat_median, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)  # type: ignore

                output_ims.append(median_thresh)
                continue

            if thresh == 'phys_thresh':

                if not phys_thresh_vals is None:

                    if phys_thresh_vals[0] == 'max':
                        physical_vmax = max_height
                    else:
                        physical_vmax = mean_height + phys_thresh_vals[0]
                    
                    if phys_thresh_vals[1] == 'min':
                        physical_vmin = min_height
                    else:
                        physical_vmin = mean_height - phys_thresh_vals[1]
                    
                    img_flat_physical = np.clip(img_flat, physical_vmin, physical_vmax)
                    phys_thresh = cv2.normalize(img_flat_physical, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)  # type: ignore

                    output_ims.append(phys_thresh)
                    continue

                else:
                    raise Exception('No phys_thresh_vals given. Please try again.')
                
        return output_ims


    if thresh_type == 'raw':
        return img

    if thresh_type == 'raw_norm':
        return cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX) # type: ignore

    if thresh_type == 'raw_plane_flat':         #Neeeds fixing. 
        img_flat, _ = spiepy.flatten_xy(img)
        # mask, _ = spiepy.mask_by_troughs_and_peaks(img_flat)
        # img_flat, _ = spiepy.flatten_xy(img, mask)
        
        return cv2.normalize(img_flat, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX) # type: ignore
    
    img_flat, _ = flatten_by_line(img)
    img_flat = img_flat - img_flat.mean()

    if thresh_type == 'raw_flat':
        return cv2.normalize(img_flat, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX) # type: ignore
    
    if thresh_type == 'raw_flat_nonorm':
        return img_flat
        

    median_height = np.median(img_flat, axis=(0, 1))
    mean_height = np.mean(img_flat, axis=(0, 1))
    max_height = np.max(img_flat, axis=(0, 1))
    min_height = np.min(img_flat, axis=(0, 1))
    sigma = np.std(img_flat, axis=(0,1))

    if thresh_type == 'mean_thresh':
        vmax_mean = mean_height + (sigma * sig)
        vmin_mean = mean_height - (sigma * sig)
        img_flat_mean = np.clip(img_flat, vmin_mean, vmax_mean)
        mean_thresh = cv2.normalize(img_flat_mean, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX) # type: ignore

        return mean_thresh
    
    if thresh_type == 'median_thresh':
        vmax_med = median_height + (sigma * sig)
        vmin_med = median_height - (sigma * sig)    
        img_flat_median = np.clip(img_flat, vmin_med, vmax_med)
        median_thresh = cv2.normalize(img_flat_median, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)  # type: ignore

        return median_thresh

    if thresh_type == 'phys_thresh':

        if not phys_thresh_vals is None:

            if phys_thresh_vals[0] == 'max':
                physical_vmax = max_height
            else:
                physical_vmax = mean_height + phys_thresh_vals[0]
            
            if phys_thresh_vals[1] == 'min':
                physical_vmin = min_height
            else:
                physical_vmin = mean_height - phys_thresh_vals[1]

            img_flat_physical = np.clip(img_flat, physical_vmin, physical_vmax)
            phys_thresh = cv2.normalize(img_flat_physical, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)  # type: ignore

            return phys_thresh

        else:
            print('No phys_thresh_vals given. Please try again.')
            return

def flatten_by_line(im):
    
    ''' Function to flatten an image via median subtraction

    Args: 
        im::arr
            Input image before flattening.
    
    Returns:
        im_flat:arr
            Flattened image.
        im_diff::arr
            Image containing the median of each line used for subtraction.
    '''
    
    rows, cols = im.shape
    #corrected_row = [0 for value in range(0, cols)]
    flattened_im = [[0 for x in range(0,cols)] for y in range(0, rows)]
    differece_im = [[0 for x in range(0,cols)] for y in range(0, rows)]
    
    for row in range(0, rows):
        median = st.median(im[row])
        for col in range(0, cols):
            flattened_im[row][col] = im[row][col] - median
            differece_im[row][col] = median

    return np.array(flattened_im), np.array(differece_im)
