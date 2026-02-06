from matplotlib.colors import ListedColormap
import numpy as np 
from pathlib import Path

from skimage.registration import phase_cross_correlation

def auto_unit(num, siUnit=''):
    """
    A Phil Blowey function to convert numbers and units from SI units into units with prefixes. Made by Phil Blowey.

    Parameters
    ----------
    num : float
        A value in an SI base unit to convert into a more convenient unit.
    siUnit : String, optional
        The SI unit associated with the given value. The default is ''.

    Returns
    -------
    list
        Returns a list containing the new unit (prefix+original unit) and the conversion factor needed to convert the supplied number to the new unit.

    """
    _prefix = {'y': 1e-24,  # yocto
               'z': 1e-21,  # zepto
               'a': 1e-18,  # atto
               'f': 1e-15,  # femto
               'p': 1e-12,  # pico
               'n': 1e-9,   # nano
               'u': 1e-6,   # micro
               'm': 1e-3,   # mili
               'c': 1e-2,   # centi
               'd': 1e-1,   # deci
               'k': 1e3,    # kilo
               'M': 1e6,    # mega
               'G': 1e9,    # giga
               'T': 1e12,   # tera
               'P': 1e15,   # peta
               'E': 1e18,   # exa
               'Z': 1e21,   # zetta
               'Y': 1e24,   # yotta
               }
    #Get the exponent of scientific notation for num
    exp = np.floor(np.log10(abs(num)))
    #Get the exponent to the nearest 3 below
    exp3 = (exp//3)*3
    #expVal Convert the new exponent back into scientific form
    expVal = 1*10**exp3
    #Get the key from the dictionary for this value
    preUnit = list(_prefix.keys())[list(_prefix.values()).index(expVal)]
    conFactor = 1*10**(-exp3)
    return [preUnit+siUnit, conFactor] 

def load_gwyddion_cmap(channel='Z'):
    ''' Function to import the gwyddion colourmap for use in pyplot cmaps.

    Args: 
        channel::str
            Channel determining which cmap is best to use. Options are 'Z', 
            'Current' and 'OC_M1_Freq._Shift', matching the sxm channels. 

    Returns:
        cmap::list
            Colour map for use in cmap argument in pyplot functions. 
    '''
    
    # Get the directory where THIS script is located
    script_dir = Path(__file__).parent
    
    # cmaps folder is in the same directory as this script
    cmaps_dir = script_dir / 'cmaps'
    
    if channel == 'Z':
        gwyd_path = cmaps_dir / 'Gwyddionnet.pymap'
    elif channel == 'Current':
        gwyd_path = cmaps_dir / 'Gwyddioncurrent.pymap'
    elif channel == 'OC_M1_Freq._Shift':
        gwyd_path = cmaps_dir / 'Grey.pymap'
    else:
        raise ValueError(f"Unknown channel: {channel}. Valid options: 'Z', 'Current', 'OC_M1_Freq._Shift'")
    
    # Check file exists
    if not gwyd_path.exists():
        raise FileNotFoundError(f"Colormap file not found: {gwyd_path}")
    
    RawRGB = np.genfromtxt(gwyd_path, skip_header=1)
            
    return ListedColormap(RawRGB)  # type: ignore

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
