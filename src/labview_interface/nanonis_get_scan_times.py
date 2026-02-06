import sys
from datetime import datetime
import numpy as np
from pathlib import Path

from skimage.registration import phase_cross_correlation

from ..data_processing.nanonis_utils import get_image_data, get_times, get_ranges


def Find_Drift(filenames):

    ''' Function which applied phase cross correlation between two images to calculate the drift between the two. Returns the drift in pixels as shift and the error in overlap between the two images in error.

    filenames - list of two filenames which should be the same but with some drift between. 
    '''

    ims = {}

    for filename in filenames:
        #Import image data
        ims[f'{filename.name[:-4]}'] = get_image_data(filename, thresh_type= 'raw_flat')
        

    image = ims[filenames[0].name[:-4]]
    offset_image = ims[filenames[1].name[:-4]]
    
    #calculate x and y pixel drift
    shift, error, _ = phase_cross_correlation(image, offset_image)
    shift = -shift          #For some reason it results in the opposite shift?? This might be something which I have two change when not using the simulator. 

    #calculate z drift
    initial_avg_z = np.average(ims[filenames[0].name[:-4]])
    final_avg_z = np.average(ims[filenames[1].name[:-4]])

    z_shift = final_avg_z -initial_avg_z
    
    return shift, z_shift, error

FMT = '%H:%M:%S'

##################### SET UP INPUTS #####################

input_1 = Path(sys.argv[1])
input_2 = Path(sys.argv[2])


filenames = [input_1, input_2]

First_start, First_time = get_times(filenames[0])
Second_start, Second_time = get_times(filenames[1])

First_start = First_start.split()
First_start = First_start[0]

Second_start = Second_start.split()
Second_start = Second_start[0]

First_time = float(First_time)
Second_time = float(Second_time)

tdelta = datetime.strptime(Second_start, FMT) - datetime.strptime(First_start, FMT)
Start_time_diff = tdelta.total_seconds()

Total_diff = Start_time_diff

################## GET SCAN RANGE AND PIXELS ##################

First_pixels, First_range, _ = get_ranges(filenames[0])

dist_per_pixel = float(First_range[0]) / float(First_pixels[0])           #Requires that both images are of same range and number of pixels

################## NOW DO CROSS CORRELATION ##################

shift, z_shift, error = Find_Drift(filenames)

shift = shift * dist_per_pixel

with open("Shift_and_times.txt", "w") as f:
    f.write('{}\n{}\n{}\n{}\n{}'.format(Total_diff, shift[0], shift[1], z_shift, error))

print(shift, error)
