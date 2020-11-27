# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from sklearn.preprocessing import minmax_scale

# --------------------------------------------------------------------------- #
# Image Kernels Function
# --------------------------------------------------------------------------- #
def KernelsHist(frame):
    '''
    Function that evaluates the kernels of interest for the input frame and
    outputs the histogram of each colour channel that remains.
    
    Kernels size: 3x3
    
    Parameters
    ----------
    frame : uint8 array
        Input frame of any dimensions.
        
    Returns
    -------
    hist : unt32 array of size 256x1 or 256x3
        Array contains the evaluate histograms of each colour channel's 
        kernels of interest.
    
    '''
    # def lcd(row, colm):
    #     '''
    #     Function that evaluates the lowest common divisor for the input frame's
    #     dimnesions. Will return an odd number.
        
    #     Parameters
    #     ----------
    #     row : int
    #         Number of rows of elements.
    #     colm : int
    #         Number of columns of elements.
            
    #     Returns
    #     -------
    #     kernSize : int
    #         Dynamic kernel size; lowest common divisor.
            
    #     '''
    #     # Initialize kernel size
    #     kernSize = 3
        
    #     # Search for optimal lowest value
    #     while row%kernSize !=0 or colm%kernSize !=0:
    #         # BUG: will enter inf loop if not adjust kernSize val.
    #         kernSize += 2
    #     # Return kernel size
    #     return kernSize
    
    # NaN expansion for kernel column & row
    expd_vals = np.full(3, np.nan)
    
    # Kernel size
    kern_size = 3
    # Assert image type (RGB/Grayscale)
    try:
        # Get input frame dimmentions
        row, colm, col = frame.shape
    # Frame is Grayscale/Binary
    except ValueError:
        # Get input grayscale frame dimmentions
        row, colm = frame.shape
        
        # Initialize list to store kernels
        kerns = []
        
        for i in range(1, row, kern_size):
            for k in range(1, colm, kern_size):
                # Kernel contains at least one non-zero element (useful info)
                if frame[i-1:i+2, k-1:k+2].any() > 0:
                    # Safeguard for edges of frame; Row
                    if frame[i-1:i+2, k-1:k+2].shape[0] == 2:
                        # Expand kernel row & append NaN values
                        kerns.append(np.vstack((frame[i-1:i+2, k-1:k+2], expd_vals)))
                        
                    # Expand Column
                    elif frame[i-1:i+2, k-1:k+2].shape[1] == 2:
                        # Expand kernel column & append NaN values
                        kerns.append(np.column_stack((frame[i-1:i+2, k-1:k+2], expd_vals)))
                        
                    # Expand both column & row
                    elif frame[i-1:i+2, k-1:k+2].shape[1] == 2 and \
                        frame[i-1:i+2, k-1:k+2].shape[0] == 2:
                            # Expand & append kernel
                            kerns.append(np.pad(frame[i-1:i+2, k-1:k+2], (0, 1), 
                                                'constant', constant_values=(np.nan)))
                    else:
                        # Append kernel
                        kerns.append(frame[i-1:i+2, k-1:k+2])
        # BUG: list at index [1744] appends 2x2 kernel
        # Must implement an assistive function for dynamic kernel size or pad with NaN (?)
        # Convert list to array
        # kern_arr = np.array(kerns, dtype=np.uint8)
        
        # # Clear list
        # kerns.clear()
        
        # # Initialize histogram array
        # hist = np.zeros(256, dtype=np.uint64)
        
        # Evaluate mask for valid values
        mask = ~np.isnan(kerns)
        
        # Evaluate histogram
        hist = np.bincount(np.uint8(kerns)[mask].flatten(), minlength=256)
        
        # Clear list
        kerns.clear()
        
        # Return histogram
        return hist
    # Frame is RGB
    else:
        # Initialize lists for each colour channel
        B_kern = []
        G_kern = []
        R_kern = []
        
        # Median image kernels
        for c in range(col):
            for i in range(1, row, kern_size):
                for k in range(1, colm, kern_size):
                    # Kernel contains at least one non-zero element (useful info)
                    if frame[i-1:i+2, k-1:k+2, c].any() > 0:
                        # Blue channel
                        if c == 0:
                            # Safeguard for edges of frame; Row
                            if frame[i-1:i+2, k-1:k+2, 0].shape[0] == 2:
                                # Expand kernel row & append NaN values
                                B_kern.append(np.vstack((frame[i-1:i+2, k-1:k+2, 0], expd_vals)))
                                
                            # Expand Column
                            elif frame[i-1:i+2, k-1:k+2, 0].shape[1] == 2:
                                # Expand kernel column & append NaN values
                                B_kern.append(np.column_stack((frame[i-1:i+2, k-1:k+2, 0], expd_vals)))
                            # Expand both column & row
                            elif frame[i-1:i+2, k-1:k+2, 0].shape[1] == 2 and \
                                frame[i-1:i+2, k-1:k+2, 0].shape[0] == 2:
                                    # Expand & append kernel
                                    B_kern.append(np.pad(frame[i-1:i+2, k-1:k+2, 0], (0, 1), 
                                                'constant', constant_values=(np.nan)))
                            else:
                                B_kern.append(frame[i-1:i+2, k-1:k+2, 0])
                        # Green channel
                        elif c == 1:
                            # Safeguard for edges of frame; Row
                            if frame[i-1:i+2, k-1:k+2, 1].shape[0] == 2:
                                # Expand kernel row & append NaN values
                                G_kern.append(np.vstack((frame[i-1:i+2, k-1:k+2, 1], expd_vals)))
                                
                            # Expand Column
                            elif frame[i-1:i+2, k-1:k+2, 1].shape[1] == 2:
                                # Expand kernel column & append NaN values
                                G_kern.append(np.column_stack((frame[i-1:i+2, k-1:k+2, 1], expd_vals)))
                            # Expand both column & row
                            elif frame[i-1:i+2, k-1:k+2, 1].shape[1] == 2 and \
                                frame[i-1:i+2, k-1:k+2, 1].shape[0] == 2:
                                    # Expand & append kernel
                                    B_kern.append(np.pad(frame[i-1:i+2, k-1:k+2, 1], (0, 1), 
                                                'constant', constant_values=(np.nan)))
                            else:
                                G_kern.append(frame[i-1:i+2, k-1:k+2, 1])
                        # Red channel
                        else:
                            # Safeguard for edges of frame; Row
                            if frame[i-1:i+2, k-1:k+2, 2].shape[0] == 2:
                                # Expand kernel row & append NaN values
                                R_kern.append(np.vstack((frame[i-1:i+2, k-1:k+2, 2], expd_vals)))
                                
                            # Expand Column
                            elif frame[i-1:i+2, k-1:k+2, 2].shape[1] == 2:
                                # Expand kernel column & append NaN values
                                R_kern.append(np.column_stack((frame[i-1:i+2, k-1:k+2, 2], expd_vals)))
                            # Expand both column & row
                            elif frame[i-1:i+2, k-1:k+2, 2].shape[1] == 2 and \
                                frame[i-1:i+2, k-1:k+2, 2].shape[0] == 2:
                                    # Expand & append kernel
                                    B_kern.append(np.pad(frame[i-1:i+2, k-1:k+2, 2], (0, 1), 
                                                'constant', constant_values=(np.nan)))
                            else:
                                R_kern.append(frame[i-1:i+2, k-1:k+2, 2])
        # Evaluate mask for each colour channel
        B_mask = ~np.isnan(B_kern)
        G_mask = ~np.isnan(G_kern)
        R_mask = ~np.isnan(R_kern)
        
        # Convert lists into array
        # kerns = np.array((B_kern, G_kern, R_kern), dtype=np.uint8)
        
        # Initialize histogram array
        hist = np.zeros((256, 3), dtype=np.uint64)
        
        # Evaluate histogram of kernels
        # Blue channel
        hist[:, 0] = np.bincount(np.uint8(B_kern)[B_mask].flatten(),minlength=256)
        
        # Green channel
        hist[:, 1] = np.bincount(np.uint8(G_kern)[G_mask].flatten(),minlength=256)
        
        # Red channel
        hist[:, 2] = np.bincount(np.uint8(R_kern)[R_mask].flatten(),minlength=256)
        
        # Return resulting array
        return hist


# --------------------------------------------------------------------------- #
# Filter Histograms Function
# --------------------------------------------------------------------------- #
def HistBpf(hist):
    '''
    Function that implements a Band Pass FIR filter in order to suppress the 
    potential effects of the number of zero pixels has on the histograms.
    
    Since background subtraction/foreground detection occurs, it is expected that
    the most dominant pixel value is zero, therefore it has the potential to 
    'overshadow' potential useful information.
    
    Parameters
    ----------
    hist : uint32 array (either 256x3 or 256x1)
        Array containing the evaluate histograms of each colour channel present.
    
    Returns
    -------
    hist/ filt_hist: uint32 array/float32 array
        If no useful info present, returns the input histogram (uint32).
        If useful info present, returns filtered histogram(s) (float32).
    
    '''
    
    try:
        # Get histogram dimentions
        row, colm = hist.shape
        
    # Image is grayscale
    except ValueError:
        
        # Initialize array to strore results
        filt_hist = np.zeros(256, dtype=np.float32)
        
        # Find max value of useful info
        info_hist_max = np.max(hist[3:])
        
        # Check if no objects present; if no info present, return input hist
        if info_hist_max == 0:
            return hist
        # Useful info present; filtering
        else:
            # Evaluate Low Pass FIR coefficient
            lp_fir_coeff = info_hist_max / np.max(hist)
            
            # Evaluate High Pass FIR threshold
            hp_thresh = info_hist_max * 0.2
            
            # Loop through each histogram's value & implement Band Pass FIR Filter
            for k in range(256):
                # If current value greater than max value of useful info (Low Pass)
                if hist[k] > info_hist_max:
                    # Multiply current value with the corresponding coefficient
                    filt_hist[k] = np.uint32(hist[k] * lp_fir_coeff)
                # If current value greater than High Pass threshold & less than
                # or equal to max value of useful info (High Pass)
                elif hist[k] > hp_thresh and hist[k] <= info_hist_max:
                    filt_hist[k] = hist[k]
                # Else set current value to zero
                else:
                    filt_hist[k] = np.uint32(0)
    # Image is RGB
    else:
        
        # Initialize array to strore results
        filt_hist = np.zeros((row, colm), dtype=np.float32)
        # Loop through every colour channel
        for i in range(colm):
            # Find max value of useful info
            info_hist_max = np.max(hist[3::, i])
            
            # Check if no objects present; continue to next colour channel
            if info_hist_max == 0:
                continue
            else:
                # Evaluate Low Pass FIR coefficient
                lp_fir_coeff = info_hist_max / np.max(hist[::, i])
                
                # Evaluate High Pass threshold
                hp_thresh = info_hist_max * 0.2
                
                # Loop through each histogram's value & implement Band Pass FIR Filter
                for k in range(row):
                    # If current value greater than max value of useful info (Low Pass)
                    if hist[k, i] > info_hist_max:
                        # Multiply current value with the corresponding coefficient
                        filt_hist[k, i] = np.uint32(hist[k, i] * lp_fir_coeff)
                    # If current value greater than High Pass threshold & less than
                    # or equal to max value of useful info (High Pass)
                    elif hist[k, i] > hp_thresh and hist[k, i] <= info_hist_max:
                        filt_hist[k, i] = hist[k, i]
                    # Else set current value to zero
                    else:
                        filt_hist[k, i] = np.uint32(0)
            
    # Return filtered histogram(s)
    return filt_hist


# -------------------------------------------------------------------------- #
# Normalize Histograms Functions
# -------------------------------------------------------------------------- #
def HistNorm(filt_hist):
    '''
    Histogram normalization function. Utilizes the mathematical normalization
    of ( histogram - min value / max value - min value)
    
    Parameters
    ----------
    filt_hist : float32/uint32 array (256x3 or 256x1)
        Array containing the evaluated & filtered histograms of each colour
        channel.
    
    Returns
    -------
    hist_norm/filt_hist: float32/uint32 array (256x3 or 256x1)
        Array of same size as input array, containing the normalized innput 
        histograms.
    
    '''
    
    try:
        # Evaluate input histogram dimensions
        row, colm = filt_hist.shape
        
    # Image is Grayscale
    except ValueError:
        # Set default values to return
        hist_norm = np.zeros(256, dtype=np.float32)
        # Check if current colour channel's histogram is empty
        if np.any(filt_hist > 0) == True:
            # Normalize current histogram
            hist_norm = minmax_scale(filt_hist)
        # Else continue to next colour channel
        else:
            return filt_hist
    # Image is RGB
    else:
        # Preallocate array for normalized histograms
        hist_norm = np.zeros((row, colm), dtype=np.float32)
        # Loop through each colour channel
        for i in range(0, colm):
            # Check if current colour channel's histogram is empty
            if np.any(filt_hist[:, i] > 0) == True:
                # Normalize current histogram
                hist_norm[:, i] = minmax_scale(filt_hist[:, i])
            # Else continue to next colour channel
            else:
                continue
    # Return normalized histogram(s)
    return hist_norm


# ------------------------------------------------------------------------- #
# Implement windows for each Colour Channel Function
# ------------------------------------------------------------------------- #
def HistWindows(hist_norm):
    '''
    Function that implements plain & overlapping windows containing values
    of each colour channel's histograms.
    
    Input:         hist_norm: An array of size 256x3 that contains the histograms
                              of each colour channel. In the current implementation
                              it is optimal for the histogram values to be normalized.
    
    Outputs:       win_vals: An array of size (8, 32, 3) that contains the values
                             of the input histograms.
                   
                   win_binloc: An array of size (8, 32, 3) that contains the bin
                               locations of the correspondig plain values.
                               
                   over_vals: An array of size (8, 32, 3) that contains the values
                              of the input histograms in an overlapping manner,
                              starting from the 15th element.
                   
                   over_binloc: An array of size(8, 32, 3) that contains the bin
                                locations of the corresponding overlapping values
    '''
    # Enumerate histogram bins
    bin_c = np.arange(0, 256, dtype=np.uint8)
    
    # Window index
    idx = 0
    
    try:
        # Evaluate histogram dimensions
        row, colm = hist_norm.shape
    # Frame is grayscale
    except ValueError:
        # Initialize windows values array
        win_vals = np.zeros((15, 32), dtype=np.float32)
        
        # Initialize windows locations array
        win_binloc = np.zeros((15, 32), dtype=np.uint8)
        
        # Loop through every window
        for i in range(32, 256, 32):
            # Plain window values
            win_vals[idx, :] = hist_norm[i-32:i]
            
            # Plain window locations
            win_binloc[idx, :] = bin_c[i-32:i]
            
            # Overlapping window values
            win_vals[idx+1, :] = hist_norm[i-16:i+16]
            
            # Overlapping window locations
            win_binloc[idx+1, :] = bin_c[i-16:i+16]
            
            # Update array index
            idx += 2
            
    # Frame is RGB
    else:
        # Initialize windows values array
        win_vals = np.zeros((15, 32, 3), dtype=np.float32)
        
        # Initialize windows locations
        win_binloc = np.zeros((15, 32, 3), dtype=np.uint8)
        
        # Iterate over each colour channel
        for c in range(colm):
            # Loop through every window
            for i in range(32, row, 32):
                # Plain window values
                win_vals[idx, :, c] = hist_norm[i-32:i, c]
                
                # Plain window locations
                win_binloc[idx, :, c] = bin_c[i-32:i]
                
                # Overlapping window values
                win_vals[idx+1, :, c] = hist_norm[i-16:i+16, c]
                
                # Overlapping window locations
                win_binloc[idx+1, :, c] = bin_c[i-16:i+16]
                
                # Update array index
                idx += 2
                
            # Reset array index
            idx = 0
    
    # Return resulting arrays
    return win_vals, win_binloc