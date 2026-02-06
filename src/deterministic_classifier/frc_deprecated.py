from cmath import inf
import numpy as np
import sys
from statistics import mean
from csaps import csaps
import scipy
import numpy.fft as fft
import itertools
import matplotlib.pyplot as plt 

def get_frc_res(img, pixel_width):

    '''
    Main function to output the resolution of an input image using FRC. 

    '''

    FRC_vals = {}

    # Splitting images using a diagonal splitting function
    f1, f2, f3, f4 = diagonal_split(img)
    

    # Normalising the image splits via hanning method
    f1 = apply_hanning_2d(normalize_data_ab(0, 1, f1))
    f2 = apply_hanning_2d(normalize_data_ab(0, 1, f2))
    f3 = apply_hanning_2d(normalize_data_ab(0, 1, f3))
    f4 = apply_hanning_2d(normalize_data_ab(0, 1, f4))

    # Calculate the FRC correlation values
    xc, corr1, xt, thres_val = FRC(f1, f2)
    _, corr2, _, _           = FRC(f3, f4)

    x_vals = xc[:-1]/2
    
    # Setting up threshold - here using 1/7 threshold
    threshold_line = np.ones(len(x_vals)) * (1/7)

    corr_all = [corr1, corr2]

    for i, corr in enumerate(corr_all):

        spline_obj = smoothSpline(x_vals, corr[:-1], smooth = 0.996)
        spline = splineEval(x_vals, spline_obj)
        
        idx = np.argwhere(np.diff(np.sign(threshold_line - spline))).flatten()
        try:
            FRC_vals[f'frc{i+1}'] = x_vals[idx[0]]
        except IndexError:
            FRC_vals[f'frc{i+1}'] = inf

    FRC_vals['FRC Avg'] = mean([FRC_vals['frc1'], FRC_vals['frc2']])
    FRC_vals['Pixel Res'] = 1 / FRC_vals['FRC Avg']
    FRC_vals['True Res'] = pixel_width * FRC_vals['Pixel Res']

    return FRC_vals

def FRC( i1, i2, thresholding='half-bit', inscribed_rings=True, analytical_arc_based=True, info_split=True, ring_size= 1):
    
    ''' Check whether the dimensions of input image is 
    square or not
    '''
    if ( np.shape(i1) != np.shape(i2) ) :
        print('\n [!] input images must have the same dimensions\n')
        import sys
        sys.exit()
    if ( np.shape(i1)[0] != np.shape(i1)[1]) :
        print('\n [!] input images must be squares\n')
        import sys
        sys.exit()

    ''' Performing the fourier transform of input
    images to determine the FRC
    '''
    # FFT of input images
    I1 = fft.fftshift(fft.fft2(i1))
    I2 = fft.fftshift(fft.fft2(i2))

    # This is effectively computing the equation for FRC - However I don't really understand what spinavej does...
    # It must be summing the input over all of the rings, however I can't see how to change the ring size anywhere
    C  = spinavej(I1*np.conjugate(I2), inscribed_rings=inscribed_rings, ring_size= ring_size)
    C = np.real(C)
    C1 = spinavej(np.abs(I1)**2, inscribed_rings=inscribed_rings, ring_size= ring_size)
    C2 = spinavej(np.abs(I2)**2, inscribed_rings=inscribed_rings, ring_size= ring_size)
    C  = C.astype(np.float32)
    C1 = np.real(C1).astype(np.float32)
    C2 = np.real(C2).astype(np.float32)
    FSC    = abs(C)/np.sqrt(C1*C2)
    x_fsc  = np.arange(np.shape(C)[0]) / (np.shape(i1)[0]/2)
    
    ring_plots=False
    if(inscribed_rings==True):
      ''' for rings with max radius 
      as L/2
      '''
      if (analytical_arc_based == True):
        ''' perimeter of circle based calculation to
        determine n in each ring
        '''
        r      = np.arange(np.shape(i1)[0]/2) # array (0:1:L/2-1)
        n      = 2*np.pi*r # perimeter of r's from above
        n[0]   = 1
        eps    = np.finfo(float).eps
        #t1 = np.divide(np.ones(np.shape(n)),n+eps)
        inv_sqrt_n = np.divide(np.ones(np.shape(n)),np.sqrt(n)) # 1/sqrt(n)
        x_T    = r/(np.shape(i1)[0]/2)
      else:
        ''' no. of pixels along the border of each circle 
        is used to determine n in each ring
        '''
        indices = ring_indices( i1, inscribed_rings=True, plot=ring_plots)
        N_ind = len(indices)  
        n = np.zeros(N_ind) 
        for i in range(N_ind):
          n[i] = len(indices[i][0])
        inv_sqrt_n = np.divide(np.ones(np.shape(n)),np.sqrt(n)) # 1/sqrt(n)
        x_T = np.arange(N_ind)/(np.shape(i1)[0]/2)

    else:
      ''' for rings with max radius as distance
      between origin and corner of image
      '''
      if (analytical_arc_based == True):
        r      = np.arange(len(C)) # array (0:1:sqrt(rx*rx + ry*ry)) where rx=ry=L/2
        n      = 2*np.pi*r # perimeter of r's from above
        n[0]   = 1
        eps    = np.finfo(float).eps
        #t1 = np.divide(np.ones(np.shape(n)),n+eps)
        inv_sqrt_n = np.divide(np.ones(np.shape(n)),np.sqrt(n)) # 1/sqrt(n)
        x_T    = r/(np.shape(i1)[0]/2)
      else:
        indices = ring_indices( i1, inscribed_rings=False, plot=ring_plots)
        N_ind = len(indices)  
        n = np.zeros(N_ind) 
        for i in range(N_ind):
          n[i] = len(indices[i][0])
        inv_sqrt_n = np.divide(np.ones(np.shape(n)),np.sqrt(n)) # 1/sqrt(n)
        x_T = np.arange(N_ind)/(np.shape(i1)[0]/2)


    if info_split:
      ''' Thresholding based on the fact that 
      SNR is split as the data is divided into
      two half datasets
      '''
      if (thresholding  == 'one-bit'):  T = (0.5+2.4142*inv_sqrt_n)/(1.5+1.4142*inv_sqrt_n) #information split
      elif(thresholding == 'half-bit'): T = (0.4142+2.287*inv_sqrt_n)/ (1.4142+1.287*inv_sqrt_n) # diagonal split 
      elif(thresholding == '0.5'):      T = 0.5*np.ones(np.shape(n))
      elif(thresholding =='em'):        T = (1/7)*np.ones(np.shape(n))
      else:
        t1 = (0.5+2.4142*inv_sqrt_n)/(1.5+1.4142*inv_sqrt_n)
        t2 = (0.2071+1.9102*inv_sqrt_n)/(1.2071+0.9102*inv_sqrt_n) # information split twice 
        t3 = 0.5*np.ones(np.shape(n))
        t4 = (1/7)*np.ones(np.shape(n))
        T = [t1, t2, t3, t4]
    else:  
      if (thresholding == 'one-bit'):  T = (1+3*inv_sqrt_n)/(2+2*inv_sqrt_n) # pixel split
      elif(thresholding == 'half-bit'):T = (0.4142+2.287*inv_sqrt_n)/ (1.4142+1.287*inv_sqrt_n) # diagonal split 
      elif(thresholding == '0.5'):     T = 0.5*np.ones(np.shape(n))
      elif(thresholding=='em'):        T = (1/7)*np.ones(np.shape(n))
      else:
          t1 = (1+3*inv_sqrt_n)/(2+2*inv_sqrt_n)
          t2 = (0.4142+2.287*inv_sqrt_n)/ (1.4142+1.287*inv_sqrt_n) 
          t3 = 0.5*np.ones(np.shape(n))
          t5 = (1/7)*np.ones(np.shape(n))
          T = [t1, t2, t3, t4]

    return (x_fsc, FSC, x_T, T)

def diagonal_split(x):

  ''' pre-processing steps interms of
  cropping to enable the diagonal 
  splitting of the input image
  '''

  h, w = x.shape
  cp_x = x 
  ''' cropping the rows '''
  if  (np.mod(h, 4)==1):
    cp_x = cp_x[:-1]
  elif(np.mod(h, 4)==2):
    cp_x = cp_x[1:-1]
  elif(np.mod(h, 4)==3):
    cp_x = cp_x[1:-2]
    
  ''' cropping the columns'''
  if  (np.mod(w, 4)==1):
    cp_x = cp_x[:, :-1]
  elif(np.mod(w, 4)==2):
    cp_x = cp_x[:,1:-1]
  elif(np.mod(w, 4)==3):
    cp_x = cp_x[:, 1:-2]


  x = cp_x
  h, w = x.shape
  if((np.mod(h, 4)!=0) or (np.mod(w, 4)!=0)):
    print('[!] diagonal splitting not possible due to cropping issue')
    print('[!] re-check the cropping portion')


  row_indices = np.arange(0, h)
  col_indices = np.arange(0, w)

  row_split_u = row_indices[::2]
  row_split_d  = np.asanyarray(list(set(row_indices)-set(row_split_u)))

  col_split_l = col_indices[::2]
  col_split_r = np.asanyarray(list(set(col_indices)-set(col_split_l)))

  ''' ordered pair of pre-processing
  of the diagonal elements 
  and sub-sequent splits of the image
  '''
  op1  = list(itertools.product(row_split_u, col_split_l))
  ind  = [np.asanyarray([fo for fo, _ in op1]), np.asanyarray([so for _, so in op1])]
  s_a1 = x[tuple(ind)]
  s_a1 = s_a1.reshape((len(row_split_u), len(col_split_l)))
  
  op2  = list(itertools.product(row_split_d, col_split_r))
  ind  = [np.asanyarray([fo for fo, _ in op2]), np.asanyarray([so for _, so in op2])]
  s_a2 = x[tuple(ind)]
  s_a2 = s_a2.reshape((len(row_split_d), len(col_split_r)))
  
  op3  = list(itertools.product(row_split_d, col_split_l))
  ind  = [np.asanyarray([fo for fo, _ in op3]), np.asanyarray([so for _, so in op3])]
  s_b1 = x[tuple(ind)]
  s_b1 = s_b1.reshape((len(row_split_d), len(col_split_l)))

  op4  = list(itertools.product(row_split_u, col_split_r))
  ind  = [np.asanyarray([fo for fo, _ in op4]), np.asanyarray([so for _, so in op4])]
  s_b2 = x[tuple(ind)]
  s_b2 = s_b2.reshape((len(row_split_u), len(col_split_r)))

  return(s_a1, s_a2, s_b1, s_b2)

def spinavej(x, inscribed_rings=True, ring_size= 1):
    ''' modification of code by sajid an
    Based on the MATLAB code by Michael Wojcik
    '''
    shape = np.shape(x)     
    dim = np.size(shape)
    ''' Depending on the dimension of the image 2D/3D, create an array of integers 
    which increase with distance from the center of the array
    '''
    if dim == 2 :
        # Finding number of indecies and centreing about zero
        nr,nc = shape
        nrdc = np.floor(nr/2)
        ncdc = np.floor(nc/2)
        r = np.arange(nr)-nrdc 
        c = np.arange(nc)-ncdc  

        [R,C] = np.meshgrid(r,c)
        index = np.round(np.sqrt(R**2+C**2))
        indexf = np.floor(np.sqrt(R**2+C**2))
        indexC = np.ceil(np.sqrt(R**2+C**2))
    
    else :
        print('input is not a 2d array')
    '''
    The index array has integers from 1 to maxindex arranged according to distance
    from the center
    '''

    if (inscribed_rings == True):
        maxindex = nr/2
    else:
        maxindex = np.max(index)
    output = np.zeros(int(maxindex/ring_size),dtype = complex)
    
    ''' In the next step output is generated. The output is an array of length
    maxindex. The elements in this array corresponds to the sum of all the elements
    in the original array correponding to the integer position of the output array 
    divided by the number of elements in the index array with the same value as the
    integer position. 
    
    Depending on the size of the input array, use either the pixel or index method.
    By-pixel method for large arrays and by-index method for smaller ones.
    '''
    #print('performed by index method')
    indices = []
    indicesf, indicesC = [], []
    for i in np.arange(int(maxindex), step = ring_size):
        #indices.append(np.where(index == i+1))
        indicesf.append(np.where(indexf == i))
        indicesC.append(np.where(indexC == i))

    for i in np.arange(int(maxindex), step = ring_size):
      #if i < len(np.arange(int(maxindex), step = ring_size)):
        #output[i] = sum(x[indices[i]])/len(indices[i][0])
        # print(i)
        # print(x[indicesC[i]])
        output[i] = (sum(x[indicesf[i]])+sum(x[indicesC[i]]))/2
    return output

def ring_indices(x, inscribed_rings=True, plot=False):
    print("ring plots is:", plot)
    
    #read the shape and dimensions of the input image
    shape = np.shape(x)     
    dim = np.size(shape)
    
    '''Depending on the dimension of the image 2D/3D, 
    create an array of integers  which increase with 
    distance from the center of the array
    '''
    if dim == 2 :
        nr,nc = shape
        nrdc = np.floor(nr/2)
        ncdc = np.floor(nc/2)
        r = np.arange(nr)-nrdc 
        c = np.arange(nc)-ncdc 
        [R,C] = np.meshgrid(r,c)
        index = np.round(np.sqrt(R**2+C**2))    
    
    elif dim == 3 :
        nr,nc,nz = shape
        nrdc = np.floor(nr/2)+1
        ncdc = np.floor(nc/2)+1
        nzdc = np.floor(nz/2)+1
        r = np.arange(nr)-nrdc + 1
        c = np.arange(nc)-ncdc + 1 
        z = np.arange(nc)-nzdc + 1 
        [R,C,Z] = np.meshgrid(r,c,z)
        index = np.round(np.sqrt(R**2+C**2+Z**2))+1    
    else :
        print('input is neither a 2d or 3d array')
   
    ''' if inscribed_rings is True then the outmost
    ring use to evaluate the FRC will be the circle
    inscribed in the square input image of size L. 
    (i.e. FRC_r <= L/2). Else the arcs of the rings 
    beyond the inscribed circle will also be
    considered while determining FRC 
    (i.e. FRC_r<=sqrt((L/2)^2 + (L/2)^2))
    '''
    if (inscribed_rings == True):
        maxindex = nr/2
    else:
        maxindex = np.max(index)
    #output = np.zeros(int(maxindex),dtype = complex)

    ''' In the next step the output is generated. The output is an array of length
    maxindex. The elements in this array corresponds to the sum of all the elements
    in the original array correponding to the integer position of the output array 
    divided by the number of elements in the index array with the same value as the
    integer position. 
    
    Depening on the size of the input array, use either the pixel or index method.
    By-pixel method for large arrays and by-index method for smaller ones.
    '''
    print('performed by index method')
    indices = []
    for i in np.arange(int(maxindex)):
        indices.append(np.where(index == i))

    if plot is True:
        img_plane = np.zeros((nr, nc))
        for i in range(int(maxindex)):
            if ((i%20)==0):
                img_plane[indices[i]]=1.0
            
        plt.imshow(img_plane, cmap='copper_r')
        if inscribed_rings is True:
            plt.title('   FRC rings with the max radius as that\
            \n of the inscribed circle in the image (spacing of 20 [px] between rings)')
        else:
            plt.title('   FRC rings extending beyond the radius of\
            \n the inscribed circle in the image (spacing of 20 [px] between rings)')
    return(indices)

def apply_hanning_2d(img):
  ''' used for experimental images to minimize 
  boundry effects
  '''
  hann_filt = np.hanning(img.shape[0])
  hann_filt = hann_filt.reshape(img.shape[0], 1)
  #hann_filt = np.power(hann_filt, 2)
  hann_img = img*hann_filt
  hann_img = hann_img*np.transpose(hann_img)
  return(hann_img)

def normalize_data_ab(a, b, data):
    min_x = min(data.ravel())
    max_x = max(data.ravel())  
    range_x = max_x - min_x 
    return((b-a)*((data-min_x)/range_x)+a)

def smoothSpline(x, y, smooth=None, Normalise=True):
    """
    This is a function that generates a spline object interpretable by scipy using the separate csaps library.
    csaps library documentation found here: https://pypi.org/project/csaps/

    Parameters
    ----------
    x : 1d numpy array
        X values of the data.
    y : 1d numpy array
        Y values of the data.
    smooth : Int, optional
        Smoothing parameter for spline fitting. If not defined, cspas will automatically compute a smoothing parameter
    Normalise : Bool, optional
        If true, the data will be scaled and centred using the z-score method before fitting the spline. The default is True.

    Returns
    -------
    splineObj : List of form [spline, Normalise, smooth]
        Spline is the fitted spline in the Ppoly format of the scipy.interpolate module.
        Normalise is the boolean for whether the data was normalised before the spline was fitted (important for differentiation)
        smooth is the smoothing parameter that was used for the spline
    """
    
    #Check that x and y have the same length
    if len(x) != len(y):
        sys.exit('smoothSpline error, X and Y are different lengths')
       
    #Normalise data input if selected - Uses the Z score method
    if Normalise:      
        #Standard deviations
        xstdev = x.std()
        ystdev = y.std()
        #Means
        xmean = x.mean()
        ymean = y.mean()
        #Normalised data
        xfit = (x-xmean)/xstdev
        yfit = (y-ymean)/ystdev        
    else:
        xfit = x
        yfit = y
        xstdev = None
        xmean  = None                    
    
    #Fit splines
    #Check if smooth parameter was defined
    if smooth == None:
        cfit = csaps(xfit, yfit) # Fit the spline with computed smoothParam (csaps default if none defined)        
        #Extract the spline and smoothParam from cfit
        cspline = cfit
        smooth = cfit.smooth
        
    else:
        cspline = csaps(xfit, yfit, smooth=smooth) #Fits spline with the defined smooth parameter
    
    coeffs  = cspline.spline.coeffs
    order   = cspline.spline.order
    pieces  = cspline.spline.pieces
    breaks  = cspline.spline.breaks
    #Coefficients for scipy.PPoly format
    c = np.reshape(coeffs, [order, pieces])
    #Undo normalisation of coefficients
    if Normalise:
        c = (c*ystdev) #Rescale the coefficients by ystdev
        c[-1,:] = c[-1,:] + ymean #Offset each polynomial by ymean  
        
    spline = scipy.interpolate.PPoly(c, breaks)
    #Create a list containing the parameters needed to differentiate the spline later
    splineObj = [spline, Normalise , smooth, xstdev, xmean]
    
    return splineObj

def splineEval(x, splineObj, nu=0):
    """
    This function is to be used to evaluate splines generated using smoothSpline function

    Parameters
    ----------
    x : 1d numpy array
        The x values to evaluate the spline over.
    splineObj : splineObj list
        List containing a PPoly spline, as output by the smoothSpline function in functions.py.
    nu : Int, optional
        Order of derivative to evaluate the spline with. The default is 0 (i.e. no differentiation)

    Returns
    -------
    y : 1d numpy array
        The spline.

    """
    #Find the degree of the spline polynomial
    spline_deg = splineObj[0].c.shape[0] - 1
    #Check to see how the degree of the spline polynomial compares to the order of differentiation requested
    if spline_deg == nu:
        print('WARNING: Degree of spline polynomials = order of differentiation. Differentiated spline will have no dependence on x between break points')
    if spline_deg < nu:
        sys.exit('Error in splineEval - Order of differentiation requested is larger than the degree of the spline polynomials')
    #Check to see if data was normalised when fitting the spline
    if splineObj[1]:
        #Finds the standard deviation of the x values used to fit the spline
        ###xstdev = splineObj[0].x.std()
        #Evaluates the spline to the specified orfer of derivative, dividing by xstdev ^ nu to deal with the normalisation
        ####y  = (splineObj[0].__call__(x, nu=nu)) / (xstdev**nu)
        x_norm = ((x - splineObj[4])/ splineObj[3])
        y = splineObj[0].__call__(x_norm, nu=nu) / (splineObj[3]**nu)
 
    else:
        #Evaluates the spline to the specified order of derivative
        y = (splineObj[0].__call__(x, nu=nu))

    return y

