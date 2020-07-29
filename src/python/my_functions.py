#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:12:19 2019

@author: iason
"""
import cv2 as cv
import numpy as np
from scipy.stats import norm, entropy, kstest
from sklearn.cluster import KMeans
#import ICP

# --------------------------------------------------------------------------- #
# Image Kernels Function
# --------------------------------------------------------------------------- #
def kernels_hist(frame):
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
    # Frame is BGR/RGB
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
# Determine  available Video (Camera sensors) sources
# --------------------------------------------------------------------------- #
def user_inpt():
    '''
    Function that prompts user to choose camera.
    In its current implementation, it is limited to returing only two possible
    options.
    
    Input: available cameras list
    Output: Valid user choise
    '''
    def detectSources():
        '''
        Function that detects any available camera.
        Input: max value of index integer number
        Output: list of available cameras currently connected.
        '''
        # Initialize list to store available sensors indices
        avail_cams = []
        
        # Initialize indices flag
        i = 0
        
        # Check for any possible input sources
        while True:
            # Test input source of current index
            cap = cv.VideoCapture(i)
            
            # If current index source is unavaiable, break
            if cap is None or not cap.isOpened():
                break
            
            # Else, append current source's index
            else:
                avail_cams.append(i)
            i += 1
            
        # Return list of available sources
        return avail_cams
    
    # Get available sources
    avail_source = detectSources()
    
    # If only one source has been detected, return its index
    if len(avail_source) == 1:
      return avail_source[0]
      
    # If two sources have been detected, return the index of the second source
    elif len(avail_source) == 2:
        return avail_source[1]

# --------------------------------------------------------------------------- #
# Filter Histograms Function
# --------------------------------------------------------------------------- #
def hist_bpf(hist):
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
def hist_norm(filt_hist):
    '''
    
    
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
        # Preallocate array for normalized histograms
        hist_norm = np.zeros(256, dtype=np.float32)
        # Check if current colour channel's histogram is empty
        if np.any(filt_hist > 0) == True:
            # Evaluate min value of current histogram
            hist_min = np.min(filt_hist)
            
            # Evaluate max value of current histogram
            hist_max = np.max(filt_hist)
            
            # Normalize current histogram
            hist_norm = (filt_hist - hist_min) / (hist_max - hist_min)
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
                # Evaluate min value of current histogram
                hist_min = np.min(filt_hist[::, i])
                
                # Evaluate max value of current histogram
                hist_max = np.max(filt_hist[::, i])
                
                # Normalize current histogram
                hist_norm[::, i] = (filt_hist[::, i] - hist_min) / (hist_max - hist_min)
            # Else continue to next colour channel
            else:
                continue
    # Return normalized histogram(s)
    return hist_norm


# ------------------------------------------------------------------------- #
# Implement windows for each Colour Channel Function
# ------------------------------------------------------------------------- #
def windows(hist_norm):
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


# --------------------------------------------------------------------------- #
# Implement Rule Based Tree on RGB Windows Function
# --------------------------------------------------------------------------- #
def rule_tree(win_vals, over_vals, win_binloc, over_binloc):
    '''
    Implementation of a rule based decision tree, in order to find regions of
    interest, and determine number of clusters. In its current implementation
    the algorithm assumes that the input arrays are of size (8x32x3). The only
    restrictions are the number of rows that must be taken into account & the
    arrays dimmentions must be the same.
    
    Inputs: hist_norm - Normalized & filtered histogram of current frame.
            bin_c - Locations of histogram values (bin locations)
    
    Outputs: final_win_loc - Bin location for windows of interest.
            final_win_vals - Values of windows of interest.
            final_win_class - Max value of windows of interest.
    '''
    
    # Initialize array of max values of windows
    max_win_val = np.zeros((8, 1, 3), dtype=np.float32)
    
    # Initialize array of max values of overlapping windows
    max_over_val = np.zeros((8, 1, 3), dtype=np.float32)
    
    # Initialize arrays for determining if usefull info is present
    win_use_info_cl = np.zeros((8, 4, 3), dtype=np.bool)
    over_use_info_cl = np.zeros((8, 4, 3), dtype=np.bool)
    
    # Initialize arrays for determining max value of current window
    max_val_logic_over = np.zeros((8, 1, 3), dtype=np.bool)
    max_val_logic_win = np.zeros((8, 1, 3), dtype=np.bool)
    
    # Initialize arrays for value percentage
    max_val_over_perc = np.zeros((8, 1, 3), dtype=np.bool)
    max_val_win_perc = np.zeros((8, 1, 3), dtype=np.bool)
    
    # Initialize arrays that store desired window locations, values & max vals
    final_win_loc = np.zeros((8, 32, 3), dtype=np.uint8)
    final_win_vals = np.zeros((8, 32, 3), dtype=np.float32)
    
    for c in range(3):
        # Iterate over each row of windows(plain & overlapping)
        for i in range(8):
            # Case for determining if black object is present
            if i == 0:
                # Evaluate max value of current plain & overlapping window
                max_win_val[i, :, c] = np.max(win_vals[i, 2:, c])
                max_over_val[i, :, c] = np.max(over_vals[i, :, c])
                
                # Initialize array index
                c_idx = 0
                # Check if max value is near the ends of plain & overlapping window
                # And if window contains info
                for val in range(-1, -5, -1):
                    # Overlapping window
                    over_use_info_cl[i, c_idx, c] = np.logical_and.reduce((over_vals
                    [i, val, c]< max_over_val[i, :, c], over_vals[i, c_idx, c] 
                    < max_over_val[i, :, c], np.mean(over_vals[i, :, c]) > 0))
                    
                    # Plain window
                    win_use_info_cl[i, c_idx, c] = np.logical_and.reduce((
                    win_vals[i, val,c] < max_win_val[i, :, c],
                    np.mean(win_vals[i, :, c]) > 0,
                    win_vals[i, c_idx, c] < max_win_val[i, :, c]))
                    
                    # Update array index
                    c_idx += 1
            else:
                # Evaluate max value of plain & overlapping current window
                max_win_val[i, :, c] = np.max(win_vals[i, :, c])
                max_over_val[i, :, c] = np.max(over_vals[i, :, c])
                # Initialize index for useful info classification
                c_idx = 0
                # Check past and future values around max values if they contain
                # useful info
                for val in range(-1, -5, -1):
                    # Store bool results for plain window criteria
                    win_use_info_cl[i, c_idx, c] = np.logical_and.reduce((win_vals[i, val, c] 
                    < max_win_val[i, :, c], win_vals[i, c_idx, c] < max_win_val[i, :, c],
                    np.mean(win_vals[i, :, c]) > 0))
                    # Store bool results for overlapping window criteria
                    over_use_info_cl[i, c_idx, c] = np.logical_and.reduce((over_vals[i, val, c]
                    < max_over_val[i, :, c], over_vals[i, c_idx, c] < max_over_val[i, :, c],
                    np.mean(over_vals[i, :, c]) > 0))
                    # Update index for checking future values
                    c_idx += 1
            
            # Check if max overlapping value is greater than max plain value
            max_val_logic_over[i, :, c] = max_over_val[i, :, c] > max_win_val[i, :, c]
            
            # Check if max overlapping value is greater than 0.4
            max_val_over_perc[i, :, c] = max_over_val[i, :, c] > 0.4
            
            # Check if max plain value is greater than max overlapping value
            max_val_logic_win[i, :, c] = max_win_val[i, :, c] > max_over_val[i, :, c]
            
            # Check if max plain value is greater than 0.4
            max_val_win_perc[i, :, c] = max_win_val[i, :, c] > 0.4
            
            # If max plain value is greater than max overlapping
            if max_val_logic_win[i, :, c] == True:
                
                # If plain contains useful info & over 0.4
                if win_use_info_cl[i, :, c].all() and max_val_win_perc[i, :, c]:
                    # Get plain window location
                    final_win_loc[i, :, c] = win_binloc[i, :, c]
                    
                    # Get Plain window values
                    final_win_vals[i, :, c] = win_vals[i, :, c]
                    
                    
                # If overlapping contains useful info & over 0.4
                elif over_use_info_cl[i, :, c].all() and max_val_over_perc[i, :, c]:
                    # Get current overlapping window bin locations
                    final_win_loc[i, :, c] = over_binloc[i, :, c]
                    
                    # Get current overlapping window values
                    final_win_vals[i, :, c] = over_vals[i, :, c]
                    
                # If both windows do not fill criteria, continue to next windows
                else:
                    continue
                
            # If overlapping max value is greater than max plain
            elif max_val_logic_over[i, :, c] == True:
                
                # If overlapping contains useful info & over 0.4
                if over_use_info_cl[i, :, c].all() and max_val_over_perc[i, :, c]:
                    # Get current overlapping window's binlocation
                    final_win_loc[i, :, c] = over_binloc[i, :, c]
                    
                    # Get current overlapping window's values
                    final_win_vals[i, :, c] = over_vals[i, :, c]
                    
                # If plain contains useful info & over 0.4
                elif win_use_info_cl[i, :, c].all() and max_val_win_perc[i, :, c]:
                    # Get plain window bin locations
                    final_win_loc[i, :, c] = win_binloc[i, :, c]
                    
                    # Get plain window values
                    final_win_vals[i, :, c] = win_vals[i, :, c]
                    
                # If both windows do not fill criteria, continue to next windows
                else:
                    continue
                    
            # If both max values are equal
            elif max_val_logic_over[i, :, c] == max_val_logic_win[i, :, c]:
                # If plain contains useful info & over 0.4
                if win_use_info_cl[i, :, c].all() and max_val_win_perc[i, :, c]:
                    # Get plain window bin location
                    final_win_loc[i, :, c] = win_binloc[i, :, c]
                    
                    # Get plain window values
                    final_win_vals[i, :, c] = win_vals[i, :, c]
                    
                
                # If overlapping contains useful info & over 0.4
                elif over_use_info_cl[i, :, c].all() and max_val_over_perc[i, :, c]:
                    # Get overlapping window's binlocation
                    final_win_loc[i, :, c] = over_binloc[i, :, c]
                    
                    # Get overlapping window's values
                    final_win_vals[i, :, c] = over_vals[i, :, c]
                    
                # Case when both windows fulfill criteria - Entropy(?)
                elif np.logical_and.reduce((over_use_info_cl[i,:,c].all() == True,
                                        max_val_over_perc[i,:,c] == True,
                                        win_use_info_cl[i,:,c].all() == True,
                                        max_val_win_perc[i,:,c] == True)):
                    
                    # Evaluate mean value of plain & overlapping windows
                    plain_mean = np.mean(win_vals[i,:,c], dtype=np.float64)
                    over_mean = np.mean(over_vals[i,:,c], dtype=np.float64)
                    
                    # Evaluate if plain mean greater than overlapping
                    if plain_mean > over_mean:
                        # Store plain windows values & bin locations
                        final_win_loc[i,:,c] = win_binloc[i,:,c]
                        
                        final_win_vals[i,:,c] = win_vals[i,:,c]
                    
                    # Else overlapping window contains useful info
                    else:
                        # Store overlapping windows values & bin locations
                        final_win_loc[i,:,c] = over_binloc[i,:,c]
                        
                        final_win_vals[i,:,c] = over_vals[i,:,c]
                
                # If both windows do not fill criteria, continue to next windows
                else:
                    continue
                
            # Continue to next set of windows
            else:
                continue
            
#            # Prune tree to remove greedy results
#            # If first window, continue
#            if i == 0:
#                continue
#            
#            # Check if bin locations of previous windows, similar to current
#            else:
#                if final_win_loc[i-1,:,c].any() == final_win_loc[i,:,c].any():
#                    # Determine which window contains the most info
#                    curr_win_valn = np.count_nonzero(final_win_vals[i,:,c])
#                    prev_win_valn = np.count_nonzero(final_win_vals[i-1,:,c])
#                    
#                    # Evaluate mean of current & past windows
#                    curr_mean = np.mean(final_win_vals[i,:,c])
#                    prev_mean = np.mean(final_win_vals[i,:,c])
#                    
#                    
        
    # Get resulting arrays as a Pandas series
    out = pd.Series((final_win_vals[:, :, :], final_win_loc[:, :, :]))
    
    # Return resulting arrays
    return(out)


# --------------------------------------------------------------------------- #
# K-Means Implementation Function - OpenCV
# --------------------------------------------------------------------------- #
def kmeans_cv(frame, n_clusters):
    '''
    K-Means function that utilizes the already built-in function found in the
    opencv library. In the current implementation, the k-means clustering alg-
    orithm is utilized in order to separate potential present objects in the 
    current frame. The algorithm utilizes the Kmeans++ initialization.
    The criteria for the K-Means are defined as, max number of iterations set to
    300, and the acceptable error rate is set to 1e-4.
    
    Inputs:         frame: Current frame; can be any size or colour type
                    
                    n_clusters: Number of clusters for the algorithm to evaluate
                    
    Outputs:         res_frame: Resulting frame from the k-means algorithm
    '''
    # Define criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 300, 1e-4)
    
    # Flatten input frame
    inpt_cv = np.float32(frame.reshape((-1, 3)))
    
    # Fit current frame to the k-means algorithm
    ret,label,center = cv.kmeans(inpt_cv, n_clusters, None, criteria,
                               10, cv.KMEANS_PP_CENTERS)
    
    # Obtain labels
    center = np.uint8(center)
    
    # Evaluate new frame based on resulting labels
    res_frame = center[label.flatten()]
    
    # Reshape frame to its origianl dimmentions
    res_frame = res_frame.reshape((frame.shape))
    
    # Return resulting array
    return(res_frame)


# --------------------------------------------------------------------------- #
# K-Means Implementation Function - Scikit-Learn
# --------------------------------------------------------------------------- #
def kmeans_sk(frame, n_clusters):
    '''
    K-Means function that utilizes the already built-in function found in the
    scikit-learn library. In the current implementation, the k-means clustering
    algorithm is utilized in order to separate potential present objects in the
    current frame. This function utilizes the Kmeans++ initialization, as well
    as 2 CPU cores for faster processing. The criteria for the k-means are
    kept to their default values, which are 300 max iterations and acceptable
    error rate of 1e-4.
    
    Inputs:         frame: Current frame; can be any size or colour type
                    
                    n_clusters: Number of clusters for the algorithm to evaluate
                    
    Outputs:         res_frame: Resulting frame from the k-means algorithm
    '''
    # Get image dimmentions
    heigh, width, _ =  frame.shape
    
    # Flatten image values for the k-means algorithm
    inpt = np.reshape(frame, (width * heigh, 3))
    
    # Initialize the k-means model
    kmeans = KMeans(n_clusters, init='k-means++', n_jobs=2)
    
    # Fit the input image into the model
    kmeans.fit(inpt)
    
    # Predict the closest cluster each sample in input image belongs to
    labels = kmeans.predict(inpt)
    
    # Output separated objects into image
    res_frame = np.zeros((heigh, width, 3), dtype=np.uint8)
    
    # Initialize label index
    label_idx = 0
    
    # Loop through image dimentions
    for i in range(heigh):
        for k in range(width):
            # At each iteration, select the corresponding cluster center of each label
            res_frame[i, k] = kmeans.cluster_centers_[labels[label_idx]]
            # Update label index
            label_idx += 1
    
    # Return resulting frame
    return(res_frame)

# --------------------------------------------------------------------------- #
# Kolmogorov-Sminrov Statistical Test to converge MOG2
# --------------------------------------------------------------------------- #
def stop_mog2(curr_hist, prev_hist):
    '''
    Kolmogorov-Smirnov Statistical test for MOG2 to converge.
    The current function evaluates the Cumulative Densities Functions
    of current & previous frame's histograms. Depending upon the results of
    the function, the MOG2 algorithm will either continue, or stop(e.g. converged).
    
    Alternative hypothesis (H1) is defined as histograms do not overlap significantly,
    therefore the MOG2 algorithm has not converged yet.
    
    Null hypothesis (H0) is defined as histograms do overlap significantly,
    therefore the MOG2 algorithm has converged, so it must stop.
    
    Inputs:         curr_hist: Histogram of current frame
                    prev_hist: Histogram of previous frame
    
    Outputs:        Tks_b : Difference current current and previous histogram 
                            for the Blue colour channel.
                    
                    Tks_g : Difference between current and previous histogram 
                            for the Green colour channel.
                    
                    Tks_r : Difference between current and previous histogram
                            for the Red colour channel
    '''
    # Evaluate CDF of each colour channel for current frame
    curr_cdf_b = curr_hist[:, 0] / 256
    curr_cdf_g = curr_hist[:, 1] / 256
    curr_cdf_r = curr_hist[:, 2] / 256
    
    # Evaluate CDF of each colour channel for previous frame
    prev_cdf_b = prev_hist[:, 0] / 256
    prev_cdf_g = prev_hist[:, 1] / 256
    prev_cdf_r = prev_hist[:, 2] / 256
    
    # Evaluate difference for Blue colour channel
    Tks_b = np.max(np.abs(curr_cdf_b - prev_cdf_b))
    
    # Evaluate difference for Green colour channel
    Tks_g = np.max(np.abs(curr_cdf_g - prev_cdf_g))
    
    # Evaluate difference for Red colour channel
    Tks_r = np.max(np.abs(curr_cdf_r - prev_cdf_r))
    
    # If Null Hypothesis true
    if np.logical_and.reduce((Tks_b < 0.01,
                             Tks_g < 0.01,
                             Tks_r < 0.01)):
        return(True)
    # If Alternative Hypothesis true
    else:
        return(False)

# -------------------------------------------------------------------------- #
# Histogram Standard Deviation to converge MOG2
# -------------------------------------------------------------------------- #
def hist_deviation(out_win_vals):
    '''
    Function that evaluates the deviation of each resulting adaptive window,
    in order to evaluate if the MOG2 algorithm needs to converge or to continue.
    The function identifies the useful windows and evaluates its deviation.
    Useful windows are defined as the windows that contain values.
    
    Inputs:           out_win_vals: A 3d array of unkown size, containing the 
                                    values of the resulting adaptive windows.
    
    Outputs:          True if the majority of the deviations are greater than 
                      the threshold value, defined as 0.21.
                      
                      False if the majority of the deviations are less than the
                      threshold value.
    '''
    try:
        # Obtain dimmentions of input array
        h, _, c = out_win_vals.shape
    except:
        # If no windows of interest, return false
        return(False)
    else:
        
        # Initialize arrays to store results
        res_var = np.zeros((h, 1, 3), dtype=np.float64)
        res_logic = np.zeros((h, 1, 3), dtype=np.bool)
        
        # Initialize indices to store the number of windows that its deviation is 
        # greater than or equal to the theshold value & the number of windows that are empty
        n_valid = 0
        n_reject = 0
        
        # Loop through each colour channel
        for c in range(c):
            # Loop through each window
            for i in range(h):
                # Evaluate deviation of current window
                res_var[i, 0, c] = np.std(out_win_vals[i, :, c], dtype=np.float64)
                
                # Evaluate if current window is non empty
                if np.any(out_win_vals[i, :, c]) > 0:
                    # If deviation of current window greater than or equal to thresh
                    if res_var[i, 0, c] >= 0.21:
                        # Increase the number of valid windows
                        n_valid += 1
                        # Store boolean result for debuggind purposes
                        res_logic[i, 0, c] = True
                    # If deviation of current window less than thresh
                    else:
                        # Increase the number of rejected windows
                        n_reject += 1
                        # Store boolean result for debuggind purposes
                        res_logic[i, 0, c] = False
                # If current window empty, increase the number of rejected windows
                else:
                    n_reject += 1
        
        # Evaluate the number of non empty windows
        n_actual = (h * 3) - n_reject
        
        # Evaluate if number of valid windows greather than or equal to the mean of
        # windows that contain useful info
        if n_valid >= np.ceil(n_actual / 2):
            return(True)
        else:
            return(False)


# --------------------------------------------------------------------------- #
# Image Processing Function - Morphological Operations & Contour capture
# --------------------------------------------------------------------------- #
def frame_proc(frame, fgmask, kernel, contour_size):
    '''
    Function that implements a number of morphological operations in order to 
    capture the contours of the detected objects (i.e. contours of interest)
    
    The function first perfoms an bitwise self addition of the input frame,
    utilizing the evaluated mask from the MOG2 algorithm.
    
    Afterwards, a morphological diation is performed on the frame in order to 
    close potential empty regions inside the contous of interest.
    
    The morphological closing is performed on the frame in order to capture the
    now filled contours of interest.
    
    The entire silhouette of each contour is evaluated and captured, in order
    to draw the detected contours; the contours that are below of a threshold
    value are deleted.
    
    Inputs:             frame: Input frame
    
                        fgmask: Evaluated mask for the current input frame
                        
                        kernel: Kernel of size 9x9
                        
                        contour_size: Threshold of accepted contours size,
                                      defined as 60 pixels.
                                      
    Outputs:            res2: Output frame, masked with the morphological closing
                              of input frame
                              
                        res: Frame that the contours are to be drawn.
                        
                        contours: The detected contous of current frame
    '''
    # Self bitwise operation on current frame
    res = cv.bitwise_and(frame,frame, mask= fgmask)
    
    # Morphologically dilate current frame
    e_im = cv.dilate(fgmask, kernel, iterations = 1)
    
    # Morphologically close current frame
    e_im = cv.morphologyEx(e_im, cv.MORPH_CLOSE, kernel)
    
    # Evaluate & capture each entire silhouettes
    contours, hierarchy = cv.findContours(e_im, cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_SIMPLE)
    
    # Remove contours that are lower than threshold's value
    temp = []
    num = 0
    # Loop through each detected contours
    for i in range(len(contours)):
        # If current contour size less than threshold's value, store contour
        if len(contours[i]) < contour_size:
            temp.append(i)
    # Loop through each contour that is less than threshold's value
    for i in temp:
        # Delete the contours that are less than threshold's value
        del contours[i - num]
        num = num + 1
    
    # Perform bitwise and operation using the morphological processed frame as
    # a mask
    res2 = cv.bitwise_and(frame,frame, mask = e_im)
    
    # Implement outputs as a pandas Series object
    out = pd.Series((res2, res, contours))
    
    # Return resulting frames and detected contours
    return(out)


# --------------------------------------------------------------------------- #
# Implement adaptive windows of interest by means of KS statistical test
# --------------------------------------------------------------------------- #
def adapt_win(final_win_vals, final_win_loc, hist_norm,
              win_vals, win_binloc, over_vals, over_binloc, deb_flg):
    '''
    Function that utilizes the Kolmogorov-Smirnov statistical test in order to
    implement adaptive windows that contain the resulting histograms of each
    colour channel. In its current implementation, the windows of interest,
    resulting from the Rule Based Tree (rule_tree) function are utilized.
    
    The Null Hypothesis (H0) is defined as the shape of plain & overlapping 
    windows histograms overlap, therefore the current window expands accordingly
    untill they no longer overlap significantly.
    
    The Alternative Hypothesis is defined as the shape of plain & overlapping
    windows do not overlap significantly, therefore no need to expand current 
    window, keep current window as is
    
    In its current implementation, it is assumed that the input arrays have
    dimmentions of (8x32x3); however the only restriction is the number of rows
    that the algorithm takes into consideration.
    
    
    Inputs:            final_win_vals: The values of the windows of interest 
                                       that the rule based tree has deemed as
                                       appropriate.
                       
                       final_win_loc: The bin locations of the corrensponding
                                      windows of interest.
                                     
                       hist_norm: The normalized histogram of each colour 
                                  channel; it is utilized for the expansion of 
                                  the plain & overlapping windows that the 
                                  KS test deems as of significanlty similar
                                  shape.
                                  
                       win_vals: The resulting values of plain windows. The 
                                 resulting array is passed into the 'ks_test'
                                 function.
                                 
                       win_binloc: The resulting bin locations of plain windows.
                                   The resulting array is passed into the 'ks_test'
                                   function.
                                   
                       over_vals: The resulting values of overlapping values.
                                  The resulting array is passed into the 'ks_test'
                                  function.
                                  
                       over_binloc: The resulting bin locations of overlapping
                                    windows. The resulting array is passed into
                                    the 'ks_test' function.
                                  
    Outputs:           out_binlocs: The resulting bin locations of the adaptive
                                    windows. In its current implementation, the
                                    bin locations are utilized to determine the
                                    number of clusters for the k-means algorithm
                                    
                       out_win_vals: The resulting values of the adaptive wind-
                                     ows. In its current implementation, the 
                                     values are utilized as a proof of concept
                                     and to enumerate the resulting number of
                                     clusters for the k-means algorithm.
                                     
                       n_clusters: The number of clusters for the k-means algo-
                                   rithm. The following assumptions are made
                                   in order to obtain the number of clusters:
                                   a) Windows of different colour channel but
                                      with same bin location, count as 1 cluster
                                   b) Windows of same or different colour chan-
                                      nel but different bin location constitute
                                      as one cluster each.
    '''
# --------------------------------------------------------------------------- #
# Kolmogorov-Smirnov Statistical Hypothesis test Function
# --------------------------------------------------------------------------- #
    def ks_test_tree(win_vals, win_binloc, over_vals, over_binloc, hist_norm,
                final_win_vals, final_win_loc, bin_c):
        '''
        Kolmogorov-Smirnov statistical test function. The statistical test evaluates
        the similarity between the plain and overlapping windows.
        
        Null Hypothesis is defined as the shape of plain & overlapping windows is
        significantly similar, therefore further expansion of current window is needed.
        
        Alternative Hypothesis is defined as the shape of plain & overlapping windows
        is not significantly similar, therefore current window does not require 
        further expansion.
        
        Inputs:             win_vals: Array of size (1x32x3) that contains the plain
                                      windows values.
                                      
                            win_binloc: Array of size(1x32x3) that contains the
                                      bin locations of plain windows.
                                      
                            over_vals: Array of size (1x32x3) that contains the 
                                      overlapping windows values
                                      
                            over_binloc: Array of size (1x32x3) that contains the
                                      bin locations of overlapping windows
                                      
                            hist_norm: Array of size (256x3) that contains the 
                                      normalized histograms of each colour channels
                                      
                            final_win_vals: Array of size(1x32x3) that contains the
                                            values of the windows that the Rule Based
                                            Tree deemed as of interest.
                                        
                            final_win_loc: Array of size (1x32x3) that contains the
                                        bin locations of the windows that the
                                        Rule Based Tree deemed as of interest.
                                        
                            bin_c: Array of size(256) that contains the total bin
                                        locations of each histogram.(0-255)
                                        
        
        Outputs:            out_binlocs: Array of size (32x3) that contains the
                                            adaptive windows bin locations.
                                            
                            out_win_vals: Array of size (32x3) that contains the 
                                            adaptive windows values.
        '''
# --------------------------------------------------------------------------- #
# Alternative Hypothesis (H1) Function
# --------------------------------------------------------------------------- #
        def alt_hyp(final_win_loc, final_win_vals):
            '''
            Alternative Hypothesis (H1) of Kolmogorov-Smirnov statistical test.
            H1 is defined as the shape of input histograms is not significantly the 
            same.
            
            Inputs:         final_win_loc: Input array that contains bin locations of 
                                       histograms of interest, as deemed by 
                                       'rule_based_tree' function.
                                       
                            final_win_vals: Values of input array that contains the
                                       windows of interest, as deemed by
                                       'rule_based_tree' function.
            
            Outputs:        out_binlocs: The resulting bin locations of output windows
            
                            out_win_vals: Array containing the resulting windows values.
            '''
            # Check if all colour channels are present
            if np.logical_and.reduce((np.any(final_win_loc[:, 0]) != 0,
                                      np.any(final_win_loc[:, 1]) != 0,
                                      np.any(final_win_loc[:, 2]) != 0)):
                
                # Initialize array values
                out_binlocs = np.zeros((32, 3), dtype=np.uint8)
                out_win_vals = np.zeros((32, 3), dtype=np.float32)
                
                # Evaluate bin locations
                out_binlocs[:, 0] = final_win_loc[:, 0]  # Blue Channel
                out_binlocs[:, 1] = final_win_loc[:, 1]  # Green Channel
                out_binlocs[:, 2] = final_win_loc[:, 2]  # Red Channel
                
                # Evaluate windows values
                out_win_vals[:, 0] = final_win_vals[:, 0]  # Blue Channel
                out_win_vals[:, 1] = final_win_vals[:, 1]  # Green Channel
                out_win_vals[:, 2] = final_win_vals[:, 2]  # Red Channel
                
                # Return resulting arrays as a pandas Series object
                out = pd.Series((out_binlocs, out_win_vals))
                
                # Return output
                return(out)
            # Check if Blue channel present
            elif np.any(final_win_loc[:, 0] != 0):
                # If Green & Blue Channels present
                if np.any(final_win_loc[i, 1]) != 0:
                    # Initialize array values
                    out_binlocs = np.zeros((32, 3), dtype=np.uint8)
                    out_win_vals = np.zeros((32, 3), dtype=np.float32)
                    
                    # Evaluate bin locations
                    out_binlocs[:, 0] = final_win_loc[:, 0]  # Blue Channel
                    out_binlocs[:, 1] = final_win_loc[:, 1]  # Green Channel
                    
                    # Evaluate windows values
                    out_win_vals[:, 0] = final_win_vals[:, 0]  # Blue Channel
                    out_win_vals[:, 1] = final_win_vals[:, 1]  # Green Channel
                    
                    # Return resulting arrays as a pandas Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return output
                    return(out)
                # Check if Red & Blue Channels present
                elif np.any(final_win_loc[:, 2]) != 0:
                    # Initialize array values
                    out_binlocs = np.zeros((32, 3), dtype=np.uint8)
                    out_win_vals = np.zeros((32, 3), dtype=np.float32)
                    
                    # Evaluate bin locations
                    out_binlocs[:, 0] = final_win_loc[:, 0]  # Blue Channel
                    out_binlocs[:, 2] = final_win_loc[:, 2]  # Red Channel
                    
                    # Evaluate windows values
                    out_win_vals[:, 0] = final_win_vals[:, 0]  # Blue Channel
                    out_win_vals[:, 2] = final_win_vals[:, 2]  # Red Channel
                    
                    # Return resulting arrays as a pandas Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return output
                    return(out)
                # If only Blue Channel present
                else:
                    # Initialize array values
                    out_binlocs = np.zeros((32, 3), dtype=np.uint8)
                    out_win_vals = np.zeros((32, 3), dtype=np.float32)
                    
                    # Evaluate bin locations
                    out_binlocs[:, 0] = final_win_loc[:, 0]  # Blue Channel
                    
                    # Evaluate windows values
                    out_win_vals[:, 0] = final_win_vals[:, 0]  # Blue Channel
                    
                    # Return resulting arrays as a pandas Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return output
                    return(out)
            # If Green Channel present
            elif np.any(final_win_loc[:, 1]) != 0:
                # If Green & Blue present
                if np.any(final_win_loc[:, 0]) != 0:
                    # Initialize array values
                    out_binlocs = np.zeros((32, 3), dtype=np.uint8)
                    out_win_vals = np.zeros((32, 3), dtype=np.float32)
                    
                    # Evaluate bin locations
                    out_binlocs[:, 0] = final_win_loc[:, 0]  # Blue Channel
                    out_binlocs[:, 1] = final_win_loc[:, 1]  # Green Channel
                    
                    # Evaluate windows values
                    out_win_vals[:, 0] = final_win_vals[:, 0]  # Blue Channel
                    out_win_vals[:, 1] = final_win_vals[:, 1]  # Green Channel
                    
                    # Return resulting arrays as a pandas Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return output
                    return(out)
                # If Red & Green Channels present
                elif np.any(final_win_loc[:, 2]) != 0:
                    # Initialize array values
                    out_binlocs = np.zeros((32, 3), dtype=np.uint8)
                    out_win_vals = np.zeros((32, 3), dtype=np.float32)
                    
                    # Evaluate bin locations
                    out_binlocs[:, 1] = final_win_loc[:, 1]  # Green Channel
                    out_binlocs[:, 2] = final_win_loc[:, 2]  # Red Channel
                    
                    # Evaluate windows values
                    out_win_vals[:, 1] = final_win_vals[:, 1]  # Green Channel
                    out_win_vals[:, 2] = final_win_vals[:, 2]  # Red Channel
                    
                    # Return resulting arrays as a pandas Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return output
                    return(out)
                # If Green Channel  present only
                else:
                    # Initialize array values
                    out_binlocs = np.zeros((32, 3), dtype=np.uint8)
                    out_win_vals = np.zeros((32, 3), dtype=np.float32)
                    
                    # Evaluate bin locations
                    out_binlocs[:, 1] = final_win_loc[:, 1]  # Green Channel
                    
                    # Evaluate windows values
                    out_win_vals[:, 1] = final_win_vals[:, 1]  # Green Channel
                    
                    # Return resulting arrays as a pandas Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return output
                    return(out)
            # If Red channel present
            elif np.any(final_win_loc[:, 2]) != 0:
                # If Red & Blue present
                if np.any(final_win_loc[:, 0]) != 0:
                    # Initialize array values
                    out_binlocs = np.zeros((32, 3), dtype=np.uint8)
                    out_win_vals = np.zeros((32, 3), dtype=np.float32)
                    
                    # Evaluate bin locations
                    out_binlocs[:, 0] = final_win_loc[:, 0]  # Blue Channel
                    out_binlocs[:, 2] = final_win_loc[:, 2]  # Red Channel
                    
                    # Evaluate windows values
                    out_win_vals[:, 0] = final_win_vals[:, 0]  # Blue Channel
                    out_win_vals[:, 2] = final_win_vals[:, 2]  # Red Channel
                    
                    # Return resulting arrays as a pandas Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return output
                    return(out)
                # If Red & Green present
                elif np.any(final_win_loc[:, 1]) != 0:
                    
                    # Initialize array values
                    out_binlocs = np.zeros((32, 3), dtype=np.uint8)
                    out_win_vals = np.zeros((32, 3), dtype=np.float32)
                    
                    # Evaluate bin locations
                    out_binlocs[:, 1] = final_win_loc[:, 1]  # Green Channel
                    out_binlocs[:, 2] = final_win_loc[:, 2]  # Red Channel
                    
                    # Evaluate windows values
                    out_win_vals[:, 1] = final_win_vals[:, 1]  # Green Channel
                    out_win_vals[:, 2] = final_win_vals[:, 2]  # Red Channel
                    
                    # Return resulting arrays as a pandas Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return output
                    return(out)
                # If only Red Channel present
                else:
                    # Initialize array values
                    out_binlocs = np.zeros((32, 3), dtype=np.uint8)
                    out_win_vals = np.zeros((32, 3), dtype=np.float32)
                    
                    # Evaluate bin locations
                    out_binlocs[:, 2] = final_win_loc[:, 2]  # Red Channel
                    
                    # Evaluate windows values
                    out_win_vals[:, 2] = final_win_vals[:, 2]  # Red Channel
                    
                    # Return resulting arrays as a pandas Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return output
                    return(out)
            # If no colour channel present
            else:
                # Return zero filled arrays
                out_binlocs = np.zeros((32, 3), dtype=np.uint8)
                out_win_vals = np.zeros((32, 3), dtype=np.uint8)
                
                # Return resulting arrays as a pandas Series object
                out = pd.Series((out_binlocs, out_win_vals))
                
                # Return output
                return(out)
# --------------------------------------------------------------------------- #
# End of Alternative Hypothesis (H1) Function
# --------------------------------------------------------------------------- #
        # If all colour channels present
        if np.logical_and.reduce((np.any(final_win_loc[:, 0]) != 0,
                                  np.any(final_win_loc[:, 1]) != 0,
                                  np.any(final_win_loc[:, 2]) != 0)):
            # Evaluate Red channel's CDF
            ur = win_vals[:, 2] / 32
            vr = over_vals[:, 2] / 32
            
            # Evaluate difference of plain & overlapping CDF's
            Tks_r = np.max(np.abs(ur - vr))
            
            # Evaluate Green channel's CDF
            ug = win_vals[:, 1] / 32
            vg = over_vals[:, 1] / 32
            
            # Evaluate difference of plain & overlapping CDF's
            Tks_g = np.max(np.abs(ug - vg))
            
            # Evaluate Blue Channel's CDF
            ub = win_vals[:, 0] / 32
            vb = over_vals[:, 0] / 32
            
            # Evaluate difference of plain & overlapping CDF's
            Tks_b = np.max(np.abs(ub - vb))
            
            # If Alternative Hypothesis (H1) true
            if Tks_r >= 0.5 and Tks_g >= 0.5 and Tks_b >= 0.5:
                
                # Utilize alternative hypothesis function
                out_binlocs, out_win_vals = alt_hyp(final_win_loc[:, :]
                                                  , final_win_vals[:, :])
                
                # Return resulting arrays as a pandas Series object
                out = pd.Series((out_binlocs, out_win_vals))
                
                # Return output
                return(out)
                
            # If Null Hypothesis (H0) true
            else:
                # Initialize index for Colour Channels
                idx = 1
                cdx = 1
                while Tks_r < 0.5 or Tks_g < 0.5 or Tks_b < 0.5:
                    
                    # Initialize arrays to expand Channel values
                    temp_win = np.zeros((32 + idx, 3), dtype=np.float32)
                    temp_over = np.zeros((32 + idx, 3), dtype=np.float32)
                    
                    # If at begining of histogram values(first window)
                    if np.logical_or.reduce((final_win_loc[0, 2] == 0 ,
                                             final_win_loc[0, 1] == 0,
                                             final_win_loc[0, 0] == 0)):
                        if idx <= 15:
                            # Red Channel Windows
                            temp_win[:, 2] = hist_norm[0:32+idx, 2]
                            temp_over[:, 2] = hist_norm[np.abs(15 - idx):47, 2]
                            
                            # Green channel Windows
                            temp_win[:, 1] = hist_norm[0:32+idx, 1]
                            temp_over[:, 1] = hist_norm[np.abs(15 - idx):47, 1]
                            
                            # Blue Channel WIndows
                            temp_win[:, 0] = hist_norm[0:32+idx, 0]
                            temp_over[:, 0] = hist_norm[np.abs(15 - idx):47, 0]
                        else:
                            # Red Channel Windows
                            temp_win[:, 2] = hist_norm[0:32+idx, 2]
                            temp_over[:, 2] = hist_norm[0:47+cdx, 2]
                            
                            # Green channel Windows
                            temp_win[:, 1] = hist_norm[0:32+idx, 1]
                            temp_over[:, 1] = hist_norm[0:47+cdx, 1]
                            
                            # Blue Channel WIndows
                            temp_win[:, 0] = hist_norm[0:32+idx, 0]
                            temp_over[:, 0] = hist_norm[0:47+cdx, 0]
                            # Increment index
                            cdx += 1
                        
                    # If at the end of histogram values (last window)
                    elif np.logical_or.reduce((final_win_loc[0, 2] == 224 ,
                                               final_win_loc[0, 1] == 224,
                                               final_win_loc[0, 0] == 224,
                                               final_win_loc[0, 0] == 239,
                                               final_win_loc[0, 1] == 239,
                                               final_win_loc[0, 2] == 239)):
                        # Red Channel Windows
                        temp_win[:, 2] = hist_norm[np.int(win_binloc[0, 2]) 
                                          -idx: np.int(win_binloc[-1, 2])+1, 2]
                        
                        # Pad zeros to the end of the window
                        temp_over[:, 2] = np.pad(hist_norm[over_binloc[0,2] 
                        - 1: over_binloc[16,2]+1, 2], 
                        (0, (len(temp_over) - (len(hist_norm[over_binloc[0,2]
                        - 1: over_binloc[16,2]+1, 2])))), 'constant')
                        
                        
                        # Green Channel Windows
                        temp_win[:, 1] = hist_norm[np.int(win_binloc[0, 1])
                                          -idx: np.int(win_binloc[-1, 1])+1, 1]
                        
                        # Pad zeros to the end of the window
                        temp_over[:, 1] = np.pad(hist_norm[np.int(over_binloc[ 0, 1])
                                           -idx : np.int(over_binloc[16, 1])+1, 1],
                        (0, len(temp_over) - len(hist_norm[np.int(over_binloc[ 0, 1])
                                           -idx : np.int(over_binloc[16, 1])+1, 1])),
                        'constant')
                        
                        # Blue Channel Windows
                        temp_win[:, 0] = hist_norm[np.int(win_binloc[0, 0])
                                          -idx: np.int(win_binloc[-1, 0])+1, 0]
                        
                        # Pad zeros to the end of the window
                        temp_over[:, 0] = np.pad(hist_norm[np.int(over_binloc[ 0, 0])
                                           -idx : np.int(over_binloc[16, 0])+1, 0],
                        (0, len(temp_over) - len(hist_norm[np.int(over_binloc[ 0, 0])
                                           -idx : np.int(over_binloc[16, 0])+1, 0])),
                        'constant')
                    else:
                        # Red Channel Windows
                        temp_win[:, 2] = hist_norm[np.int(win_binloc[0, 2])
                                   : np.int(win_binloc[-1, 2]) + idx, 2]
                        
                        temp_over[:, 2] = hist_norm[np.int(over_binloc[i, 0, 2])
                                       -idx : np.int(over_binloc[-1, 2]), 2]
                        
                        # Green Channel Windows
                        temp_win[:, 1] = hist_norm[np.int(win_binloc[0, 1])
                                   : np.int(win_binloc[-1, 1]) + idx, 1]
                        
                        temp_over[:, 1] = hist_norm[np.int(over_binloc[0, 1])
                                    -idx : np.int(over_binloc[-1, 1]), 1]
                        
                        # Blue Channel Windows
                        temp_win[:, 0] = hist_norm[np.int(win_binloc[0, 0])
                                   : np.int(win_binloc[-1, 0]) + idx, 0]
                        
                        temp_over[:, 0] = hist_norm[np.int(over_binloc[0, 0])
                                    -idx : np.int(over_binloc[-1, 0]), 0]
                        
                    # Re Evaluate Red channel
                    ur = temp_win[:, 2] / len(temp_win)
                    vr = temp_over[:, 2] / len(temp_over)
                    Tks_r = np.max(np.abs(ur - vr))
                    
                    # Re Evaluate Green channel
                    ug = temp_win[:, 1] / len(temp_win)
                    vg = temp_over[:, 1] / len(temp_over)
                    Tks_g = np.max(np.abs(ug - vg))
                    
                    # Re Evaluate Blue Channel
                    ub = temp_win[:, 0] / len(temp_win)
                    vb = temp_over[:, 0] / len(temp_over)
                    Tks_b = np.max(np.abs(ub - vb))
                    
                    # Increase index value for next iteration
                    idx += 1
                    
                # Initialize output arrays
                out_binlocs = np.zeros((32 + idx, 3), dtype=np.uint8)
                out_win_vals = np.zeros((32 + idx, 3), dtype=np.float32)
                
                # If current window is first window
                if final_win_loc[0, 0] == 0:
                    # Blue Channel values
                    out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0]):
                                         np.int(final_win_loc[-1, 0]) + idx, 0]
                    
                    out_binlocs[:, 0] = bin_c[np.int(final_win_loc[0,0]):
                        np.int(final_win_loc[-1,0]) + idx]
                    # Green Channel values
                    out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]):
                                         np.int(final_win_loc[-1, 1] + idx), 1]
                    
                    out_binlocs[:, 1] = bin_c[np.int(final_win_loc[0, 1]):
                                         np.int(final_win_loc[-1, 1] + idx)]
                    
                    # Red Channel values
                    out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]):
                                         np.int(final_win_loc[-1, 2] + idx), 2]
                    
                    out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]):
                                         np.int(final_win_loc[-1, 2] + idx)]
                    
                # If current window is last window
                elif np.logical_or.reduce((final_win_loc[0, 2] == 224 ,
                                           final_win_loc[0, 1] == 224,
                                           final_win_loc[0, 0] == 224,
                                           final_win_loc[0, 0] == 239,
                                           final_win_loc[0, 1] == 239,
                                           final_win_loc[0, 2] == 239)):
                    # Blue Channel values
                    out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0]
                                      - idx):np.int(final_win_loc[-1, 0]), 0]
                    
                    out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0]
                                      - idx):np.int(final_win_loc[-1, 0])]
                    
                    # Green Channel values
                    out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]
                                      - idx):np.int(final_win_loc[-1, 1]), 1]
                    
                    out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]
                                      - idx):np.int(final_win_loc[-1, 1])]
                    
                    # Red Channel values
                    out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]
                                      - idx):np.int(final_win_loc[-1, 2]), 2]
                    
                    out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]
                                      - idx):np.int(final_win_loc[-1, 2])]
                    
                # If current window any window except first or last
                else:
                    
                    # Check if index reaches before beginig of values
                    try:
                        bin_c[np.int(final_win_loc[0, 0] - idx)] >= 0
                    # If Error raised
                    except:
                        # Blue Channel values
                        out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0]): 
                                            np.int(final_win_loc[-1, 0] + idx), 0]
                        
                        out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0]): 
                                            np.int(final_win_loc[-1, 0] + idx)]
                        
                        # Green Channel values
                        out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]): 
                                             np.int(final_win_loc[-1, 1] + idx), 1]
                        
                        out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]): 
                                             np.int(final_win_loc[-1, 1] + idx)]
                        
                        # Red Channel values
                        out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]): 
                                             np.int(final_win_loc[-1, 2] + idx)]
                        
                        out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]): 
                                             np.int(final_win_loc[-1, 2] + idx)]
                        
                    # If no errors occured (not in the begining of values)
                    # check index reaches beyond the end of values
                    else:
                        try:
                            bin_c[np.int(final_win_loc[-1, 0] + idx)] >= 255
                            # If Error raised (beyond end of values)
                        except:
                            # Blue Channel values
                            out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0] 
                                               -idx):np.int(final_win_loc[-1, 0]), 0]
                            
                            out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0] 
                                               -idx):np.int(final_win_loc[-1, 0])]
                            
                            # Green Channel values
                            out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]
                                              - idx):np.int(final_win_loc[-1, 1]), 1]
                            
                            out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]
                                              - idx):np.int(final_win_loc[-1, 1])]
                            
                            # Red Channel values
                            out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]
                                              - idx):np.int(final_win_loc[-1, 2]), 2]
                            
                            out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]
                                              - idx):np.int(final_win_loc[-1, 2])]
                            
                        # If no error raised (not beyond or below range of values)
                        else:
                            # Blue Channel values
                            out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0]
                                        - idx):np.int(final_win_loc[-1, 0] + idx), 0]
                            
                            out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0]
                                        - idx):np.int(final_win_loc[-1, 0] + idx)]
                            
                            # Green Channel values
                            out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]
                                        - idx):np.int(final_win_loc[-1, 1] + idx), 1]
                            
                            out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]
                                        - idx):np.int(final_win_loc[-1, 1] + idx)]
                            
                            # Red Channel values
                            out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]
                                        - idx):np.int(final_win_loc[-1, 2] + idx), 2]
                            
                            out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]
                                        - idx):np.int(final_win_loc[-1, 2] + idx)]
                            
                # Store resulting arrays as a pd Series object
                out = pd.Series((out_binlocs, out_win_vals))
                
                # Return resulting arrays
                return(out)
            
        # Determine if Blue channel is present
        if np.any(final_win_loc[:, 0]) != 0:
            # Check if Red & Blue channels present only
            if np.any(final_win_loc[:, 2]) != 0:
                # Evaluate Red channel's CDF
                ur = win_vals[:, 2] / 32
                vr = over_vals[:, 2] / 32
                
                # Evaluate difference of plain & overlapping CDF's
                Tks_r = np.max(np.abs(ur - vr))
                
                # Evaluate Blue Channel's CDF
                ub = win_vals[:, 0] / 32
                vb = over_vals[:, 0] / 32
                
                # Evaluate difference of plain & overlapping CDF's
                Tks_b = np.max(np.abs(ub - vb))
                
                # If Alternative Hypothesis (H1) true
                if Tks_r >= 0.5 and Tks_b >= 0.5:
                    
                    # Utilize alternative hypothesis function
                    (out_binlocs, out_win_vals) = alt_hyp(final_win_loc[:, :]
                                                      , final_win_vals[:, :])
                    
                    # Return resulting arrays as a pandas Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return output
                    return(out)
                    
                # If Null Hypothesis (H0) true
                else:
                    # Initialize index for Colour Channels
                    idx = 1
                    cdx = 1
                    while Tks_r < 0.5 or Tks_b < 0.5:
                        if np.logical_or.reduce((bin_c[0:32+idx] == 255,
                                    bin_c[np.int(win_binloc[0, 2]) 
                                    -idx: np.int(win_binloc[-1, 2])+1, 2]== 0,
                                    bin_c[np.int(win_binloc[0, 2])
                                       : np.int(win_binloc[-1, 2]) + idx, 2],
                                    bin_c[np.int(over_binloc[i, 0, 2])
                                           -idx : np.int(over_binloc[-1, 2]), 2])):
                            break
                        
                        # Initialize arrays to expand Red & Blue Channels values
                        temp_win = np.zeros((32 + idx, 3), dtype=np.float32)
                        temp_over = np.zeros((32 + idx, 3), dtype=np.float32)
                        
                        
                        # If at begining of histogram values(first window)
                        if np.logical_or(final_win_loc[0, 2] == 0 ,
                                         final_win_loc[0, 0] == 0):
                            
                            if idx <= 15:
                                # Red Channel Windows
                                temp_win[:, 2] = hist_norm[0:32+idx, 2]
                                temp_over[:, 2] = hist_norm[np.abs(15 - idx):47, 2]
                                
                                # Blue Channel WIndows
                                temp_win[:, 0] = hist_norm[0:32+idx, 0]
                                temp_over[:, 0] = hist_norm[np.abs(15 - idx):47, 0]
                            else:
                                # Red Channel Windows
                                temp_win[:, 2] = hist_norm[0:32+idx, 2]
                                temp_over[:, 2] = hist_norm[0:47+cdx, 2]
                                
                                # Blue Channel WIndows
                                temp_win[:, 0] = hist_norm[0:32+idx, 0]
                                temp_over[:, 0] = hist_norm[0:47+cdx, 0]
                                # Increment index
                                cdx += 1
                            
                        # If at the end of histogram values (last window)
                        elif np.logical_or.reduce((final_win_loc[0, 2] == 224 ,
                                           final_win_loc[0, 0] == 224,
                                           final_win_loc[0, 0] == 239,
                                           final_win_loc[0, 2] == 239)):
                            
                            # Red Channel Windows
                            temp_win[:, 2] = hist_norm[np.int(win_binloc[0, 2]) 
                                              -idx: np.int(win_binloc[-1, 2])+1, 2]
                            
                            temp_over[:, 2] = np.pad(hist_norm[np.int(over_binloc[0, 2])
                            -idx: np.int(over_binloc[16, 2]), 2],
                            (0, len(temp_over) - len(hist_norm[np.int(over_binloc[0, 2])
                            -idx: np.int(over_binloc[16, 2]), 2])), 'constant')
                            
                                
                            # Blue Channel Windows
                            temp_win[:, 0] = hist_norm[np.int(win_binloc[0, 0])
                                              -idx: np.int(win_binloc[-1, 0])+1, 0]
                            
                            temp_over[:, 0] = np.pad(hist_norm[np.int(over_binloc[ 0, 0])
                                               -idx : np.int(over_binloc[16, 0]), 0],
                            (0, len(temp_over) - len(hist_norm[np.int(over_binloc[ 0, 0])
                                               -idx : np.int(over_binloc[16, 0]), 0])),
                            'constant')
                        
                        else:
                            
                            # Red Channel Windows
                            temp_win[:, 2] = hist_norm[np.int(win_binloc[0, 2])
                                       : np.int(win_binloc[-1, 2]) + idx, 2]
                            
                            temp_over[:, 2] = hist_norm[np.int(over_binloc[i, 0, 2])
                                           -idx : np.int(over_binloc[-1, 2]), 2]
                            
                            # Blue Channel Windows
                            temp_win[:, 0] = hist_norm[np.int(win_binloc[0, 0])
                                       : np.int(win_binloc[-1, 0]) + idx, 0]
                            
                            temp_over[:, 0] = hist_norm[np.int(over_binloc[0, 0])
                                        -idx : np.int(over_binloc[-1, 0]), 0]
                            
                        # Re Evaluate Red channel
                        ur = temp_win[:, 2] / len(temp_win)
                        vr = temp_over[:, 2] / len(temp_over)
                        Tks_r = np.max(np.abs(ur - vr))
                        
                        # Re Evaluate Blue Channel
                        ub = temp_win[:, 0] / len(temp_win)
                        vb = temp_over[:, 0] / len(temp_over)
                        Tks_b = np.max(np.abs(ub - vb))
                        
                        # Increase index value for next iteration
                        idx += 1
                        
                    # Initialize output arrays
                    out_binlocs = np.zeros((32 + idx, 3), dtype=np.uint8)
                    out_win_vals = np.zeros((32 + idx, 3), dtype=np.float32)
                    
                    # If current window is first window
                    if final_win_loc[0, 0] == 0:
                        # Blue Channel values
                        out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0]):
                            np.int(final_win_loc[-1, 0] + idx), 0]
                        
                        out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0]):
                            np.int(final_win_loc[-1, 0] + idx)]
                        
                        # Red Channel values
                        out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]):
                                             np.int(final_win_loc[-1, 2] + idx), 2]
                        
                        out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]):
                                             np.int(final_win_loc[-1, 2] + idx)]
                    
                    # If current window is last window
                    elif np.logical_or.reduce((final_win_loc[0, 2] == 224 ,
                                           final_win_loc[0, 0] == 224,
                                           final_win_loc[0, 0] == 239,
                                           final_win_loc[0, 2] == 239)):
                        # Blue Channel values
                        out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0]
                                          - idx):np.int(final_win_loc[-1, 0]), 0]
                        
                        out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0]
                                          - idx):np.int(final_win_loc[-1, 0])]
                        
                        # Red Channel values
                        out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]
                                          - idx):np.int(final_win_loc[-1, 2]), 2]
                        
                        out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]
                                          - idx):np.int(final_win_loc[-1, 2])]
                        
                    # If current window any window except first or last
                    else:
                        
                        # Check if index reaches before beginig of values
                        try:
                            bin_c[np.int(final_win_loc[0, 0] - idx)] >= 0
                        # If Error raised
                        except:
                            # Blue Channel values
                            out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0]): 
                                                 np.int(final_win_loc[-1, 0] + idx), 0]
                            
                            out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0]): 
                                                 np.int(final_win_loc[-1, 0] + idx)]
                            
                            # Red Channel values
                            out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]): 
                                                np.int(final_win_loc[-1, 2] + idx), 2]
                            
                            out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]): 
                                                np.int(final_win_loc[-1, 2] + idx)]
                            
                        # If no errors occured (not in the begining of values)
                        # check index reaches beyond the end of values
                        else:
                            try:
                                bin_c[np.int(final_win_loc[-1, 0] + idx)] >= 255
                            # If Error raised (beyond end of values)
                            except:
                                # Blue Channel values
                                out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0] 
                                               -idx):np.int(final_win_loc[-1, 0]), 0]
                                
                                out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0] 
                                               -idx):np.int(final_win_loc[-1, 0])]
                                
                                # Red Channel values
                                out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]
                                                 - idx):np.int(final_win_loc[-1, 2]), 2]
                                
                                out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]
                                                 - idx):np.int(final_win_loc[-1, 2])]
                                
                            # If no error raised (not beyond or below range of values)
                            else:
                                # Blue Channel values
                                out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0]
                                          - idx):np.int(final_win_loc[-1, 0] + idx), 0]
                                
                                out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0]
                                          - idx):np.int(final_win_loc[-1, 0] + idx)]
                                
                                # Red Channel values
                                out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]
                                            - idx):np.int(final_win_loc[-1, 2] + idx), 2]
                                
                                out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]
                                            - idx):np.int(final_win_loc[-1, 2] + idx)]
                                
                    # Store resulting arrays as a pd Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return resulting arrays
                    return(out)
            
            # Check if Blue & Green channels are present
            elif np.any(final_win_loc[:, 1]) != 0:
                # Evaluate Green channel's CDF
                ug = win_vals[:, 1] / 32
                vg = over_vals[:, 1] / 32
                
                # Evaluate difference of plain & overlapping CDF's
                Tks_g = np.max(np.abs(ug - vg))
                
                # Evaluate Blue Channel's CDF
                ub = win_vals[:, 0] / 32
                vb = over_vals[:, 0] / 32
                
                # Evaluate difference of plain & overlapping CDF's
                Tks_b = np.max(np.abs(ub - vb))
                
                # If Alternative Hypothesis (H1) true
                if Tks_g >= 0.5 and Tks_b >= 0.5:
                    
                    # Utilize alternative hypothesis function
                    (out_binlocs, out_win_vals) = alt_hyp(final_win_loc[:, :]
                                                      , final_win_vals[:, :])
                    
                    # Return resulting arrays as a pandas Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return output
                    return(out)
                    
                # If Null Hypothesis (H0) true
                else:
                    # Initialize index for Colour Channels
                    idx = 1
                    cdx = 1
                    while Tks_g < 0.5 or Tks_b < 0.5:
                        # Stop for the most common rule (i.e. value expansion inv)
                        if bin_c[0,:32+idx,1] == 255:
                            break
                        
                        # Initialize arrays to expand Green & Blue Channels window
                        temp_win = np.zeros((32 + idx, 3), dtype=np.float32)
                        temp_over = np.zeros((32 + idx, 3), dtype=np.float32)
                        
                        # If at begining of histogram values(first window)
                        if np.logical_or(final_win_loc[0, 1] == 0,
                                         final_win_loc[0, 0] == 0):
                            if idx <= 15:
                                # Green channel Windows
                                temp_win[:, 1] = hist_norm[0:32+idx, 1]
                                temp_over[:, 1] = hist_norm[np.abs(15 - idx):47, 1]
                                
                                # Blue Channel WIndows
                                temp_win[:, 0] = hist_norm[0:32+idx, 0]
                                temp_over[:, 0] = hist_norm[np.abs(15 - idx):47, 0]
                            else:
                                # Green channel Windows
                                temp_win[:, 1] = hist_norm[0:32+idx, 1]
                                temp_over[:, 1] = hist_norm[0:47+cdx, 1]
                                
                                # Blue Channel WIndows
                                temp_win[:, 0] = hist_norm[0:32+idx, 0]
                                temp_over[:, 0] = hist_norm[0:47+cdx, 0]
                                # Increment index
                                cdx += 1
                        
                        # If at the end of histogram values (last window)
                        elif np.logical_or.reduce((final_win_loc[0, 1] == 224,
                                           final_win_loc[0, 0] == 224,
                                           final_win_loc[0, 0] == 239,
                                           final_win_loc[0, 1] == 239)):
                            
                            # Green Channel Windows
                            temp_over[:, 1] = np.pad(hist_norm[np.int(over_binloc[0, 1])
                                              -idx: np.int(over_binloc[-1, 1])+1, 1],
                            (0, len(temp_over) - len(hist_norm[np.int(over_binloc[0, 1])
                                              -idx: np.int(over_binloc[16, 1]), 1])),
                            'constant')
                            
                            temp_win[:, 1] = hist_norm[np.int(win_binloc[ 0, 1])
                                               -idx : np.int(win_binloc[16, 1])+1, 1]
                            
                            # Blue Channel Windows
                            temp_win[:, 0] = hist_norm[np.int(win_binloc[0, 0])
                                              -idx: np.int(win_binloc[-1, 0])+1, 0]
                            
                            temp_over[:, 0] = np.pad(hist_norm[np.int(over_binloc[ 0, 0])
                                               -idx : np.int(over_binloc[16, 0]), 0],
                            (0, len(temp_over) - len(hist_norm[np.int(over_binloc[ 0, 0])
                                               -idx : np.int(over_binloc[16, 0]), 0])),
                            'constant')
                            
                        else:
                            
                            # Green Channel Windows
                            temp_win[:, 1] = hist_norm[np.int(win_binloc[0, 1])
                                       : np.int(win_binloc[-1, 1]) + idx, 1]
                            
                            temp_over[:, 1] = hist_norm[np.int(over_binloc[0, 1])
                                        -idx : np.int(over_binloc[-1, 1]), 1]
                            
                            # Blue Channel Windows
                            temp_win[:, 0] = hist_norm[np.int(win_binloc[0, 0])
                                       : np.int(win_binloc[-1, 0]) + idx, 0]
                            
                            temp_over[:, 0] = hist_norm[np.int(over_binloc[0, 0])
                                        -idx : np.int(over_binloc[-1, 0]), 0]
                            
                        # Re Evaluate Green channel
                        ug = temp_win[:, 1] / len(temp_win)
                        vg = temp_over[:, 1] / len(temp_over)
                        Tks_g = np.max(np.abs(ug - vg))
                        
                        # Re Evaluate Blue Channel
                        ub = temp_win[:, 0] / len(temp_win)
                        vb = temp_over[:, 0] / len(temp_over)
                        Tks_b = np.max(np.abs(ub - vb))
                        
                        # Increase index value for next iteration
                        idx += 1
                        
                    # Initialize output arrays
                    out_binlocs = np.zeros((32 + idx, 3), dtype=np.uint8)
                    out_win_vals = np.zeros((32 + idx, 3), dtype=np.float32)
                    
                    # If current window is first window
                    if final_win_loc[0, 0] == 0:
                        # Blue Channel values
                        out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0]):
                                             np.int(final_win_loc[-1, 0] + idx), 0]
                        
                        out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0]):
                                             np.int(final_win_loc[-1, 0] + idx)]
                        
                        # Green Channel values
                        out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]):
                                             np.int(final_win_loc[-1, 1] + idx), 1]
                        
                        out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]):
                                             np.int(final_win_loc[-1, 1] + idx)]
                        
                    # If current window is last window
                    elif np.logical_or.reduce((final_win_loc[0, 1] == 224,
                                           final_win_loc[0, 0] == 224,
                                           final_win_loc[0, 0] == 239,
                                           final_win_loc[0, 1] == 239)):
                        # Blue Channel values
                        out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0]
                                          - idx):np.int(final_win_loc[-1, 0]), 0]
                        
                        out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0]
                                          - idx):np.int(final_win_loc[-1, 0])]
                        
                        # Green Channel values
                        out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]
                                          - idx):np.int(final_win_loc[-1, 1]), 1]
                        
                        out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]
                                          - idx):np.int(final_win_loc[-1, 1])]
                        
                    # If current window any window except first or last
                    else:
                        
                        # Check if index reaches before beginig of values
                        try:
                            bin_c[np.int(final_win_loc[0, 0] - idx)] >= 0
                        # If Error raised
                        except:
                            # Blue Channel values
                            out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0]): 
                                                    np.int(final_win_loc[-1, 0] + idx), 0]
                            
                            out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0]): 
                                                    np.int(final_win_loc[-1, 0] + idx)]
                            
                            # Green Channel values
                            out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]): 
                                                    np.int(final_win_loc[-1, 1] + idx), 1]
                            
                            out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]): 
                                                    np.int(final_win_loc[-1, 1] + idx)]
                            
                        # If no errors occured (not in the begining of values)
                        # check index reaches beyond the end of values
                        else:
                            try:
                                bin_c[np.int(final_win_loc[-1, 0] + idx)] >= 255
                            # If Error raised (beyond end of values)
                            except:
                                # Blue Channel values
                                out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0] 
                                                    -idx):np.int(final_win_loc[-1, 0]), 0]
                                
                                out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0] 
                                                    -idx):np.int(final_win_loc[-1, 0])]
                                
                                # Green Channel values
                                out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]
                                                 - idx):np.int(final_win_loc[-1, 1]), 1]
                                
                                out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]
                                                 - idx):np.int(final_win_loc[-1, 1])]
                                
                            # If no error raised (not beyond or below range of values)
                            else:
                                # Blue Channel values
                                out_win_vals[:, 0] = hist_norm[np.int(
                                            final_win_loc[0, 0] - idx):np.int(
                                            final_win_loc[-1, 0] + idx), 0]
                                
                                out_binlocs[:,0] = bin_c[np.int(
                                            final_win_loc[0, 0] - idx):np.int(
                                            final_win_loc[-1, 0] + idx)]
                                
                                # Green Channel values
                                out_win_vals[:, 1] = hist_norm[np.int(
                                            final_win_loc[0, 1] - idx):np.int(
                                            final_win_loc[-1, 1] + idx), 1]
                                
                                out_binlocs[:,1] = bin_c[np.int(
                                            final_win_loc[0, 1] - idx):np.int(
                                            final_win_loc[-1, 1] + idx)]
                                
                    # Store resulting arrays as a pd Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return resulting arrays
                    return(out)
            
            # If only Blue Channel present
            else:
                # Evaluate difference of plain & overlapping CDF's
                Tks_g = np.max(np.abs(ug - vg))
                
                # Evaluate Blue Channel's CDF
                ub = win_vals[:, 0] / 32
                vb = over_vals[:, 0] / 32
                
                # Evaluate difference of plain & overlapping CDF's
                Tks_b = np.max(np.abs(ub - vb))
                
                # If Alternative Hypothesis (H1) true
                if Tks_b >= 0.5:
                    
                    # Utilize alternative hypothesis function
                    (out_binlocs, out_win_vals) = alt_hyp(final_win_loc[:, :]
                                                      , final_win_vals[:, :])
                    
                    # Return resulting arrays as a pandas Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return output
                    return(out)
                    
                # If Null Hypothesis (H0) true
                else:
                    # Initialize index for Colour Channels
                    idx = 1
                    cdx = 1
                    while Tks_b < 0.5:
                        # Initialize arrays to expand Blue Channel Window
                        temp_win = np.zeros((32 + idx, 3), dtype=np.float32)
                        temp_over = np.zeros((32 + idx, 3), dtype=np.float32)
                        
                        # If at begining of histogram values(first window)
                        if final_win_loc[0, 0] == 0:
                            if idx <= 15:
                                # Blue Channel WIndows
                                temp_win[:, 0] = hist_norm[0:32+idx, 0]
                                temp_over[:, 0] = hist_norm[np.abs(15 - idx):47, 0]
                            else:
                                # Blue Channel WIndows
                                temp_win[:, 0] = hist_norm[0:32+idx, 0]
                                temp_over[:, 0] = hist_norm[0:47+cdx, 0]
                                # Incremend index
                                cdx += 1
                        
                        # If at the end of histogram values (last window)
                        elif np.logical_or(final_win_loc[0, 0] == 224,
                                           final_win_loc[0, 0] == 239):
                            
                            # Blue Channel Windows
                            temp_win[:, 0] = hist_norm[np.int(win_binloc[0, 0])
                                              -idx: np.int(win_binloc[-1, 0])+1, 0]
                            
                            temp_over[:, 0] = np.pad(hist_norm[np.int(over_binloc[ 0, 0])
                                               -idx : np.int(over_binloc[16, 0]), 0],
                            (0, len(temp_over) - len(hist_norm[np.int(over_binloc[ 0, 0])
                                               -idx : np.int(over_binloc[16, 0]), 0])),
                            'constant')
                            
                        else:
                            
                            # Blue Channel Windows
                            temp_win[:, 0] = hist_norm[np.int(win_binloc[0, 0])
                                       : np.int(win_binloc[-1, 0]) + idx, 0]
                            
                            temp_over[:, 0] = hist_norm[np.int(over_binloc[0, 0])
                                        -idx : np.int(over_binloc[-1, 0]), 0]
                            
                        # Re Evaluate Blue Channel
                        ub = temp_win[:, 0] / len(temp_win)
                        vb = temp_over[:, 0] / len(temp_over)
                        Tks_b = np.max(np.abs(ub - vb))
                        
                        # Increase index value for next iteration
                        idx += 1
                        
                    # Initialize output arrays
                    out_binlocs = np.zeros((32 + idx, 3), dtype=np.uint8)
                    out_win_vals = np.zeros((32 + idx, 3), dtype=np.float32)
                    
                    # If current window is first window
                    if final_win_loc[0, 0] == 0:
                        # Blue Channel values
                        out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0]):
                                             np.int(final_win_loc[-1, 0] + idx), 0]
                        
                        out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0]):
                                             np.int(final_win_loc[-1, 0] + idx)]
                    
                    # If current window is last window
                    elif final_win_loc[0, 0] == 224 or final_win_loc[0,0] == 239:
                        # Blue Channel values
                        out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0]
                                          - idx):np.int(final_win_loc[-1, 0]), 0]
                        
                        out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0]
                                          - idx):np.int(final_win_loc[-1, 0])]
                        
                    # If current window any window except first or last
                    else:
                        
                        # Check if index reaches before beginig of values
                        try:
                            bin_c[np.int(final_win_loc[0, 0] - idx)] >= 0
                        # If Error raised
                        except:
                            # Blue Channel values
                            out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0]): 
                                                    np.int(final_win_loc[-1, 0] + idx), 0]
                            
                            out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0]): 
                                                    np.int(final_win_loc[-1, 0] + idx)]
                            
                        # If no errors occured (not in the begining of values)
                        # check index reaches beyond the end of values
                        else:
                            try:
                                bin_c[np.int(final_win_loc[-1, 0] + idx)] >= 255
                            # If Error raised (beyond end of values)
                            except:
                                # Blue Channel values
                                out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0] 
                                                    -idx):np.int(final_win_loc[-1, 0]), 0]
                                
                                out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0] 
                                                    -idx):np.int(final_win_loc[-1, 0])]
                                
                            # If no error raised (not beyond or below range of values)
                            else:
                                # Blue Channel values
                                out_win_vals[:, 0] = hist_norm[np.int(
                                            final_win_loc[0, 0] - idx):np.int(
                                            final_win_loc[-1, 0] + idx), 0]
                                
                                out_binlocs[:,0] = bin_c[np.int(
                                            final_win_loc[0, 0] - idx):np.int(
                                            final_win_loc[-1, 0] + idx)]
                                
                    # Store resulting arrays as a pd Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return resulting arrays
                    return(out)
        # If Green Channel present
        elif np.any(final_win_loc[:, 1]) != 0:
            # If Green & Blue channels present
            if np.any(final_win_loc[:, 0]) != 0:
                # Evaluate Green channel's CDF
                ug = win_vals[:, 1] / 32
                vg = over_vals[:, 1] / 32
                
                # Evaluate difference of plain & overlapping CDF's
                Tks_g = np.max(np.abs(ug - vg))
                
                # Evaluate Blue Channel's CDF
                ub = win_vals[:, 0] / 32
                vb = over_vals[:, 0] / 32
                
                # Evaluate difference of plain & overlapping CDF's
                Tks_b = np.max(np.abs(ub - vb))
                
                # If Alternative Hypothesis (H1) true
                if Tks_g >= 0.5 and Tks_b >= 0.5:
                    
                    # Utilize alternative hypothesis function
                    (out_binlocs, out_win_vals) = alt_hyp(final_win_loc[:, :]
                                                      , final_win_vals[:, :])
                    
                    # Return resulting arrays as a pandas Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return output
                    return(out)
                    
                # If Null Hypothesis (H0) true
                else:
                    # Initialize index for Colour Channels
                    idx = 1
                    cdx = 1
                    while Tks_g < 0.5 or Tks_b < 0.5:
                        # Initialize arrays to expand Green & Blue Channels window
                        temp_win = np.zeros((32 + idx, 3), dtype=np.float32)
                        temp_over = np.zeros((32 + idx, 3), dtype=np.float32)
                        
                        # If at begining of histogram values(first window)
                        if np.logical_or(final_win_loc[0, 1] == 0,
                                         final_win_loc[0, 0] == 0):
                            if idx <= 15:
                                # Green channel Windows
                                temp_win[:, 1] = hist_norm[0:32+idx, 1]
                                temp_over[:, 1] = hist_norm[np.abs(15 - idx):47, 1]
                                
                                # Blue Channel WIndows
                                temp_win[:, 0] = hist_norm[0:32+idx, 0]
                                temp_over[:, 0] = hist_norm[np.abs(15 - idx):47, 0]
                            else:
                                # Green channel Windows
                                temp_win[:, 1] = hist_norm[0:32+idx, 1]
                                temp_over[:, 1] = hist_norm[0:47+cdx, 1]
                                
                                # Blue Channel WIndows
                                temp_win[:, 0] = hist_norm[0:32+idx, 0]
                                temp_over[:, 0] = hist_norm[0:47+cdx, 0]
                                # Increment index
                                cdx += 1
                        
                        # If at the end of histogram values (last window)
                        elif np.logical_or.reduce((final_win_loc[0, 1] == 224,
                                           final_win_loc[0, 0] == 224,
                                           final_win_loc[0, 0] == 239,
                                           final_win_loc[0, 1] == 239)):
                            
                            # Green Channel Windows
                            temp_win[:, 1] = hist_norm[np.int(win_binloc[0, 1])
                                              -idx: np.int(win_binloc[-1, 1])+1, 1]
                            
                            temp_over[:, 1] = np.pad(hist_norm[np.int(over_binloc[ 0, 1])
                                               -idx : np.int(over_binloc[16, 1]), 1],
                            (0, len(temp_over) - len(hist_norm[np.int(over_binloc[ 0, 1])
                                               -idx : np.int(over_binloc[16, 1]), 1])),
                            'constant')
                            
                            # Blue Channel Windows
                            temp_win[:, 0] = hist_norm[np.int(win_binloc[0, 0])
                                              -idx: np.int(win_binloc[-1, 0])+1, 0]
                            
                            temp_over[:, 0] = np.pad(hist_norm[np.int(over_binloc[ 0, 0])
                                               -idx : np.int(over_binloc[16, 0]), 0],
                            (0, len(temp_over) - len(hist_norm[np.int(over_binloc[ 0, 0])
                                               -idx : np.int(over_binloc[16, 0]), 0])),
                            'constant')
                            
                        else:
                            
                            # Green Channel Windows
                            temp_win[:, 1] = hist_norm[np.int(win_binloc[0, 1])
                                       : np.int(win_binloc[-1, 1]) + idx, 1]
                            
                            temp_over[:, 1] = hist_norm[np.int(over_binloc[0, 1])
                                        -idx : np.int(over_binloc[-1, 1]), 1]
                            
                            # Blue Channel Windows
                            temp_win[:, 0] = hist_norm[np.int(win_binloc[0, 0])
                                        : np.int(win_binloc[-1, 0]) + idx, 0]
                            
                            temp_over[:, 0] = hist_norm[np.int(over_binloc[0, 0])
                                        -idx : np.int(over_binloc[-1, 0]), 0]
                            
                        # Re Evaluate Green channel
                        ug = temp_win[:, 1] / len(temp_win)
                        vg = temp_over[:, 1] / len(temp_over)
                        Tks_g = np.max(np.abs(ug - vg))
                        
                        # Re Evaluate Blue Channel
                        ub = temp_win[:, 0] / len(temp_win)
                        vb = temp_over[:, 0] / len(temp_over)
                        Tks_b = np.max(np.abs(ub - vb))
                        
                        # Increase index value for next iteration
                        idx += 1
                        
                    # Initialize output arrays
                    out_binlocs = np.zeros((32 + idx, 3), dtype=np.uint8)
                    out_win_vals = np.zeros((32 + idx, 3), dtype=np.float32)
                    
                    # If current window is first window
                    if final_win_loc[0, 0] == 0:
                        # Blue Channel values
                        out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0]):
                                             np.int(final_win_loc[-1, 0] + idx), 0]
                        
                        out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0]):
                                             np.int(final_win_loc[-1, 0] + idx)]
                        
                        # Green Channel values
                        out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]):
                                             np.int(final_win_loc[-1, 1] + idx), 1]
                        
                        out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]):
                                             np.int(final_win_loc[-1, 1] + idx)]
                    
                    # If current window is last window
                    elif np.logical_or.reduce((final_win_loc[0, 1] == 224,
                                           final_win_loc[0, 0] == 224,
                                           final_win_loc[0, 0] == 239,
                                           final_win_loc[0, 1] == 239)):
                        # Blue Channel values
                        out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0]
                                          - idx):np.int(final_win_loc[-1, 0]), 0]
                        
                        out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0]
                                          - idx):np.int(final_win_loc[-1, 0])]
                        
                        # Green Channel values
                        out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]
                                          - idx):np.int(final_win_loc[-1, 1]), 1]
                        
                        out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]
                                          - idx):np.int(final_win_loc[-1, 1])]
                        
                    # If current window any window except first or last
                    else:
                        
                        # Check if index reaches before beginig of values
                        try:
                            bin_c[np.int(final_win_loc[0, 0] - idx)] >= 0
                        # If Error raised
                        except:
                            # Blue Channel values
                            out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0]): 
                                                    np.int(final_win_loc[-1, 0] + idx), 0]
                            
                            out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0]): 
                                                    np.int(final_win_loc[-1, 0] + idx)]
                            
                            # Green Channel values
                            out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]): 
                                                    np.int(final_win_loc[-1, 1] + idx), 1]
                            
                            out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]): 
                                                    np.int(final_win_loc[-1, 1] + idx)]
                            
                        # If no errors occured (not in the begining of values)
                        # check index reaches beyond the end of values
                        else:
                            try:
                                bin_c[np.int(final_win_loc[-1, 0] + idx)] >= 255
                                # If Error raised (beyond end of values)
                            except:
                                # Blue Channel values
                                out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0] 
                                                    -idx):np.int(final_win_loc[-1, 0]), 0]
                                
                                out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0] 
                                                    -idx):np.int(final_win_loc[-1, 0])]
                                
                                # Green Channel values
                                out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]
                                                 - idx):np.int(final_win_loc[-1, 1]), 1]
                                
                                out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]
                                                 - idx):np.int(final_win_loc[-1, 1])]
                                
                            # If no error raised (not beyond or below range of values)
                            else:
                                # Blue Channel values
                                out_win_vals[:, 0] = hist_norm[np.int(
                                            final_win_loc[0, 0] - idx):np.int(
                                            final_win_loc[-1, 0] + idx), 0]
                                
                                out_binlocs[:,0] = bin_c[np.int(
                                            final_win_loc[0, 0] - idx):np.int(
                                            final_win_loc[-1, 0] + idx)]
                                
                                # Green Channel values
                                out_win_vals[:, 1] = hist_norm[np.int(
                                            final_win_loc[0, 1] - idx):np.int(
                                            final_win_loc[-1, 1] + idx), 1]
                                
                                out_binlocs[:,1] = bin_c[np.int(
                                            final_win_loc[0, 1] - idx):np.int(
                                            final_win_loc[-1, 1] + idx)]
                                
                    # Store resulting arrays as a pd Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return resulting arrays
                    return(out)
            # If Green & Red channels present
            elif np.any(final_win_loc[:, 2]) != 0:
                # Evaluate Red channel's CDF
                ur = win_vals[:, 2] / 32
                vr = over_vals[:, 2] / 32
                
                # Evaluate difference of plain & overlapping CDF's
                Tks_r = np.max(np.abs(ur - vr))
                
                # Evaluate Green channel's CDF
                ug = win_vals[:, 1] / 32
                vg = over_vals[:, 1] / 32
                
                # Evaluate difference of plain & overlapping CDF's
                Tks_g = np.max(np.abs(ug - vg))
                
                # If Alternative Hypothesis (H1) true
                if Tks_r >= 0.5 and Tks_g >= 0.5:
                    
                    # Utilize alternative hypothesis function
                    (out_binlocs, out_win_vals) = alt_hyp(final_win_loc[:, :]
                                                      , final_win_vals[:, :])
                    
                    # Return resulting arrays as a pandas Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return output
                    return(out)
                    
                # If Null Hypothesis (H0) true
                else:
                    # Initialize index for Colour Channels
                    idx = 1
                    cdx = 1
                    while Tks_r < 0.5 or Tks_g < 0.5:
                        # Initialize arrays to expand Red & Green Channels values
                        temp_win = np.zeros((32 + idx, 3), dtype=np.float32)
                        temp_over = np.zeros((32 + idx, 3), dtype=np.float32)
                        
                        # If at begining of histogram values(first window)
                        if np.logical_or(final_win_loc[0, 2] == 0,
                                         final_win_loc[0, 1] == 0):
                            if idx <= 15:
                                # Red Channel Windows
                                temp_win[:, 2] = hist_norm[0:32+idx, 2]
                                temp_over[:, 2] = hist_norm[np.abs(15 - idx):47, 2]
                                
                                # Green channel Windows
                                temp_win[:, 1] = hist_norm[0:32+idx, 1]
                                temp_over[:, 1] = hist_norm[np.abs(15 - idx):47, 1]
                            else:
                                # Red Channel Windows
                                temp_win[:, 2] = hist_norm[0:32+idx, 2]
                                temp_over[:, 2] = hist_norm[0:47+cdx, 2]
                                
                                # Green channel Windows
                                temp_win[:, 1] = hist_norm[0:32+idx, 1]
                                temp_over[:, 1] = hist_norm[0:47+cdx, 1]
                                # Increment index
                                cdx = 1
                        
                        # If at the end of histogram values (last window)
                        elif np.logical_or.reduce((final_win_loc[0, 2] == 224 ,
                                           final_win_loc[0, 1] == 224,
                                           final_win_loc[0, 1] == 239,
                                           final_win_loc[0, 2] == 239)):
                            
                            # Red Channel Windows
                            temp_win[:, 2] = hist_norm[np.int(win_binloc[0, 2]) 
                                              -idx: np.int(win_binloc[-1, 2])+1, 2]
                            
                            temp_over[:, 2] = np.pad(hist_norm[np.int(over_binloc[0, 2])
                            -idx: np.int(over_binloc[16, 2]), 2],
                            (0, len(temp_over) - len(hist_norm[np.int(over_binloc[0, 2])
                            -idx: np.int(over_binloc[16, 2]), 2])), 'constant')
                                
                            # Green Channel Windows
                            temp_win[:, 1] = hist_norm[np.int(win_binloc[0, 1])
                                              -idx: np.int(win_binloc[-1, 1])+1, 1]
                            
                            temp_over[:, 1] = np.pad(hist_norm[np.int(over_binloc[ 0, 1])
                                               -idx : np.int(over_binloc[16, 1]), 1],
                            (0, len(temp_over) - len(hist_norm[np.int(over_binloc[ 0, 1])
                                               -idx : np.int(over_binloc[16, 1]), 1])),
                            'constant')
                            
                        else:
                            
                            # Red Channel Windows
                            temp_win[:, 2] = hist_norm[np.int(win_binloc[0, 2])
                                        : np.int(win_binloc[-1, 2]) + idx, 2]
                            
                            temp_over[:, 2] = hist_norm[np.int(over_binloc[i, 0, 2])
                                           -idx : np.int(over_binloc[-1, 2]), 2]
                            
                            # Green Channel Windows
                            temp_win[:, 1] = hist_norm[np.int(win_binloc[0, 1])
                                        : np.int(win_binloc[-1, 1]) + idx, 1]
                            
                            temp_over[:, 1] = hist_norm[np.int(over_binloc[0, 1])
                                        -idx : np.int(over_binloc[-1, 1]), 1]
                            
                            
                        # Re Evaluate Red channel
                        ur = temp_win[:, 2] / len(temp_win)
                        vr = temp_over[:, 2] / len(temp_over)
                        Tks_r = np.max(np.abs(ur - vr))
                        
                        # Re Evaluate Green channel
                        ug = temp_win[:, 1] / len(temp_win)
                        vg = temp_over[:, 1] / len(temp_over)
                        Tks_g = np.max(np.abs(ug - vg))
                        
                        
                        # Increase index value for next iteration
                        idx += 1
                        
                    # Initialize output arrays
                    out_binlocs = np.zeros((32 + idx, 3), dtype=np.uint8)
                    out_win_vals = np.zeros((32 + idx, 3), dtype=np.float32)
                    
                    # If current window is first window
                    if final_win_loc[0, 1] == 0:
                        # Green Channel values
                        out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]):
                                             np.int(final_win_loc[-1, 1] + idx), 1]
                        
                        out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]):
                                             np.int(final_win_loc[-1, 1] + idx)]
                        
                        # Red Channel values
                        out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]):
                                             np.int(final_win_loc[-1, 2] + idx), 2]
                        
                        out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]):
                                             np.int(final_win_loc[-1, 2] + idx)]
                    
                    # If current window is last window
                    elif np.logical_or.reduce((final_win_loc[0, 2] == 224 ,
                                           final_win_loc[0, 1] == 224,
                                           final_win_loc[0, 1] == 239,
                                           final_win_loc[0, 2] == 239)):
                        # Green Channel values
                        out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]
                                          - idx):np.int(final_win_loc[-1, 1]), 1]
                        
                        out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]
                                          - idx):np.int(final_win_loc[-1, 1])]
                        
                        # Red Channel values
                        out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]
                                          - idx):np.int(final_win_loc[-1, 2]), 2]
                        
                        out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]
                                          - idx):np.int(final_win_loc[-1, 2])]
                        
                    # If current window any window except first or last
                    else:
                        
                        # Check if index reaches before beginig of values
                        try:
                            bin_c[np.int(final_win_loc[0, 1] - idx)] >= 0
                        # If Error raised
                        except:
                            # Green Channel values
                            out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]): 
                                                    np.int(final_win_loc[-1, 1] + idx), 1]
                            
                            out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]): 
                                                    np.int(final_win_loc[-1, 1] + idx)]
                            
                            # Red Channel values
                            out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]): 
                                                    np.int(final_win_loc[-1, 2] + idx), 2]
                            
                            out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]): 
                                                    np.int(final_win_loc[-1, 2] + idx)]
                            
                        # If no errors occured (not in the begining of values)
                        # check index reaches beyond the end of values
                        else:
                            try:
                                bin_c[np.int(final_win_loc[-1, 1] + idx)] >= 255
                            # If Error raised (beyond end of values)
                            except:
                                # Green Channel values
                                out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]
                                                 - idx):np.int(final_win_loc[-1, 1]), 1]
                                
                                out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]
                                                 - idx):np.int(final_win_loc[-1, 1])]
                                
                                # Red Channel values
                                out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]
                                                  - idx):np.int(final_win_loc[-1, 2]), 2]
                                
                                out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]
                                                  - idx):np.int(final_win_loc[-1, 2])]
                                
                            # If no error raised (not beyond or below range of values)
                            else:
                                # Green Channel values
                                out_win_vals[:, 1] = hist_norm[np.int(
                                            final_win_loc[0, 1] - idx):np.int(
                                            final_win_loc[-1, 1] + idx), 1]
                                
                                out_binlocs[:,1] = bin_c[np.int(
                                            final_win_loc[0, 1] - idx):np.int(
                                            final_win_loc[-1, 1] + idx)]
                                
                                # Red Channel values
                                out_win_vals[:, 2] = hist_norm[np.int(
                                            final_win_loc[0, 2] - idx):np.int(
                                            final_win_loc[-1, 2] + idx), 2]
                                
                                out_binlocs[:,2] = bin_c[np.int(
                                            final_win_loc[0, 2] - idx):np.int(
                                            final_win_loc[-1, 2] + idx)]
                                
                    # Store resulting arrays as a pd Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return resulting arrays
                    return(out)
            # If Green Channel only
            else:
                # Evaluate Green channel's CDF
                ug = win_vals[:, 1] / 32
                vg = over_vals[:, 1] / 32
                
                # Evaluate difference of plain & overlapping CDF's
                Tks_g = np.max(np.abs(ug - vg))
                
                # If Alternative Hypothesis (H1) true
                if Tks_g >= 0.5:
                    
                    # Utilize alternative hypothesis function
                    (out_binlocs, out_win_vals) = alt_hyp(final_win_loc[:, :]
                                                      , final_win_vals[:, :])
                    
                    # Return resulting arrays as a pandas Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return output
                    return(out)
                    
                # If Null Hypothesis (H0) true
                else:
                    # Initialize index for Colour Channels
                    idx = 1
                    cdx = 1
                    while Tks_g < 0.5:
                        # Initialize arrays to expand Green Channel window
                        temp_win = np.zeros((32 + idx, 3), dtype=np.float32)
                        temp_over = np.zeros((32 + idx, 3), dtype=np.float32)
                        
                        # If at begining of histogram values(first window)
                        if final_win_loc[0, 1] == 0:
                            if idx <= 15:
                                # Green channel Windows
                                temp_win[:, 1] = hist_norm[0:32+idx, 1]
                                temp_over[:, 1] = hist_norm[np.abs(15 - idx):47, 1]
                            else:
                                # Green channel Windows
                                temp_win[:, 1] = hist_norm[0:32+idx, 1]
                                temp_over[:, 1] = hist_norm[0:47+cdx, 1]
                                # Increment index
                                cdx += 1
                        
                        # If at the end of histogram values (last window)
                        elif final_win_loc[0, 1] == 224 or final_win_loc[0,1] == 239:
                            
                            # Green Channel Windows
                            temp_win[:, 1] = hist_norm[np.int(win_binloc[0, 1])
                                              -idx: np.int(win_binloc[-1, 1])+1, 1]
                            
                            temp_over[:, 1] = np.pad(hist_norm[np.int(over_binloc[ 0, 1])
                                               -idx : np.int(over_binloc[16, 1]), 1],
                        (0, len(temp_over) - len(hist_norm[np.int(over_binloc[ 0, 1])
                                               -idx : np.int(over_binloc[16, 1]), 1])),
                        'constant')
                            
                        else:
                            
                            # Green Channel Windows
                            temp_win[:, 1] = hist_norm[np.int(win_binloc[0, 1])
                                       : np.int(win_binloc[-1, 1]) + idx, 1]
                            
                            temp_over[:, 1] = hist_norm[np.int(over_binloc[0, 1])
                                        -idx : np.int(over_binloc[-1, 1]), 1]
                            
                        # Re Evaluate Green channel
                        ug = temp_win[:, 1] / len(temp_win)
                        vg = temp_over[:, 1] / len(temp_over)
                        Tks_g = np.max(np.abs(ug - vg))
                        
                        # Increase index value for next iteration
                        idx += 1
                        
                    # Initialize output arrays
                    out_binlocs = np.zeros((32 + idx, 3), dtype=np.uint8)
                    out_win_vals = np.zeros((32 + idx, 3), dtype=np.float32)
                    
                    # If current window is first window
                    if final_win_loc[0, 1] == 0:
                        # Green Channel values
                        out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]):
                                             np.int(final_win_loc[-1, 1] + idx), 1]
                        
                        out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]):
                                             np.int(final_win_loc[-1, 1] + idx)]
                        
                    # If current window is last window
                    elif final_win_loc[0, 1] == 224 or final_win_loc[0,1] == 239:
                        # Green Channel values
                        out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]
                                          - idx):np.int(final_win_loc[-1, 1]), 1]
                        
                        out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]
                                          - idx):np.int(final_win_loc[-1, 1])]
                        
                    # If current window any window except first or last
                    else:
                        # Check if index reaches before beginig of values
                        try:
                            bin_c[np.int(final_win_loc[0, 1] - idx)] >= 0
                        # If Error raised
                        except:
                            # Green Channel values
                            out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]): 
                                                    np.int(final_win_loc[-1, 1] + idx), 1]
                            
                            out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]): 
                                                    np.int(final_win_loc[-1, 1] + idx)]
                            
                        # If no errors occured (not in the begining of values)
                        # check index reaches beyond the end of values
                        else:
                            try:
                                bin_c[np.int(final_win_loc[-1, 1] + idx)] >= 255
                            # If Error raised (beyond end of values)
                            except:
                                # Green Channel values
                                out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]
                                                 - idx):np.int(final_win_loc[-1, 1]), 1]
                                
                                out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]
                                                 - idx):np.int(final_win_loc[-1, 1])]
                                
                            # If no error raised (not beyond or below range of values)
                            else:
                                # Green Channel values
                                out_win_vals[:, 1] = hist_norm[np.int(
                                            final_win_loc[0, 1] - idx):np.int(
                                            final_win_loc[-1, 1] + idx), 1]
                                
                                out_binlocs[:,1] = bin_c[np.int(
                                            final_win_loc[0, 1] - idx):np.int(
                                            final_win_loc[-1, 1] + idx)]
                                
                    # Store resulting arrays as a pd Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return resulting arrays
                    return(out)
        # If Red Chanel present
        elif np.any(final_win_loc[:, 2]) != 0:
            # If Red & Green Channels present
            if np.any(final_win_loc[:, 1]) != 0:
                # Evaluate Red channel's CDF
                ur = win_vals[:, 2] / 32
                vr = over_vals[:, 2] / 32
                
                # Evaluate difference of plain & overlapping CDF's
                Tks_r = np.max(np.abs(ur - vr))
                
                # Evaluate Green channel's CDF
                ug = win_vals[:, 1] / 32
                vg = over_vals[:, 1] / 32
                
                # Evaluate difference of plain & overlapping CDF's
                Tks_g = np.max(np.abs(ug - vg))
                
                # If Alternative Hypothesis (H1) true
                if Tks_r >= 0.5 and Tks_g >= 0.5:
                    
                    # Utilize alternative hypothesis function
                    (out_binlocs, out_win_vals) = alt_hyp(final_win_loc[:, :]
                                                      , final_win_vals[:, :])
                    
                    # Return resulting arrays as a pandas Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return output
                    return(out)
                    
                # If Null Hypothesis (H0) true
                else:
                    # Initialize index for Colour Channels
                    idx = 1
                    cdx = 1
                    while Tks_r < 0.5 or Tks_g < 0.5:
                        # Initialize arrays to expand Red & Green Channels values
                        temp_win = np.zeros((32 + idx, 3), dtype=np.float32)
                        temp_over = np.zeros((32 + idx, 3), dtype=np.float32)
                        
                        # If at begining of histogram values(first window)
                        if np.logical_or(final_win_loc[0, 2] == 0,
                                         final_win_loc[0, 1] == 0):
                            if idx <= 15:
                                # Red Channel Windows
                                temp_win[:, 2] = hist_norm[0:32+idx, 2]
                                temp_over[:, 2] = hist_norm[np.abs(15 - idx):47, 2]
                                
                                # Green channel Windows
                                temp_win[:, 1] = hist_norm[0:32+idx, 1]
                                temp_over[:, 1] = hist_norm[np.abs(15 - idx):47, 1]
                            else:
                                # Red Channel Windows
                                temp_win[:, 2] = hist_norm[0:32+idx, 2]
                                temp_over[:, 2] = hist_norm[0:47+cdx, 2]
                                
                                # Green channel Windows
                                temp_win[:, 1] = hist_norm[0:32+idx, 1]
                                temp_over[:, 1] = hist_norm[0:47+cdx, 1]
                                # Increment index
                                cdx += 1
                        
                        # If at the end of histogram values (last window)
                        elif np.logical_or.reduce((final_win_loc[0, 2] == 224 ,
                                           final_win_loc[0, 1] == 224,
                                           final_win_loc[0, 1] == 239,
                                           final_win_loc[0, 2] == 239)):
                            # Red Channel Windows
                            temp_win[:, 2] = hist_norm[np.int(win_binloc[0, 2]) 
                                              -idx: np.int(win_binloc[-1, 2])+1, 2]
                            
                            temp_over[:, 2] = np.pad(hist_norm[np.int(over_binloc[0, 2])
                            -idx: np.int(over_binloc[16, 2]), 2], 
                            (0, len(temp_over) - len(hist_norm[np.int(over_binloc[0, 2])
                            -idx: np.int(over_binloc[16, 2]), 2])), 'constant')
                                
                            # Green Channel Windows
                            temp_win[:, 1] = hist_norm[np.int(win_binloc[0, 1])
                                              -idx: np.int(win_binloc[-1, 1])+1, 1]
                            
                            temp_over[:, 1] = np.pad(hist_norm[np.int(over_binloc[ 0, 1])
                                               -idx : np.int(over_binloc[16, 1]), 1],
                            (0, len(temp_over) - len(hist_norm[np.int(over_binloc[ 0, 1])
                                               -idx : np.int(over_binloc[16, 1]), 1])),
                            'constant')
                            
                        else:
                            
                            # Red Channel Windows
                            temp_win[:, 2] = hist_norm[np.int(win_binloc[0, 2])
                                       : np.int(win_binloc[-1, 2]) + idx, 2]
                            
                            temp_over[:, 2] = hist_norm[np.int(over_binloc[i, 0, 2])
                                           -idx : np.int(over_binloc[-1, 2]), 2]
                            
                            # Green Channel Windows
                            temp_win[:, 1] = hist_norm[np.int(win_binloc[0, 1])
                                       : np.int(win_binloc[-1, 1]) + idx, 1]
                            
                            temp_over[:, 1] = hist_norm[np.int(over_binloc[0, 1])
                                        -idx : np.int(over_binloc[-1, 1]), 1]
                            
                        # Re Evaluate Red channel
                        ur = temp_win[:, 2] / len(temp_win)
                        vr = temp_over[:, 2] / len(temp_over)
                        Tks_r = np.max(np.abs(ur - vr))
                        
                        # Re Evaluate Green channel
                        ug = temp_win[:, 1] / len(temp_win)
                        vg = temp_over[:, 1] / len(temp_over)
                        Tks_g = np.max(np.abs(ug - vg))
                        
                        # Increase index value for next iteration
                        idx += 1
                        
                    # Initialize output arrays
                    out_binlocs = np.zeros((32 + idx, 3), dtype=np.uint8)
                    out_win_vals = np.zeros((32 + idx, 3), dtype=np.float32)
                    
                    # If current window is first window
                    if final_win_loc[0, 2] == 0:
                        # Green Channel values
                        out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]):
                                             np.int(final_win_loc[-1, 1] + idx), 1]
                        
                        out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]):
                                             np.int(final_win_loc[-1, 1] + idx)]
                        
                        # Red Channel values
                        out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]):
                                             np.int(final_win_loc[-1, 2] + idx), 2]
                        
                        out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]):
                                             np.int(final_win_loc[-1, 2] + idx)]
                    
                    # If current window is last window
                    elif np.logical_or.reduce((final_win_loc[0, 2] == 224 ,
                                           final_win_loc[0, 1] == 224,
                                           final_win_loc[0, 1] == 239,
                                           final_win_loc[0, 2] == 239)):
                        # Green Channel values
                        out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]
                                          - idx):np.int(final_win_loc[-1, 1]), 1]
                        
                        out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]
                                          - idx):np.int(final_win_loc[-1, 1])]
                        
                        # Red Channel values
                        out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]
                                          - idx):np.int(final_win_loc[-1, 2]), 2]
                        
                        out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]
                                          - idx):np.int(final_win_loc[-1, 2])]
                        
                    # If current window any window except first or last
                    else:
                        
                        # Check if index reaches before beginig of values
                        try:
                            bin_c[np.int(final_win_loc[0, 2] - idx)] >= 0
                        # If Error raised
                        except:
                            # Green Channel values
                            out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]): 
                                                    np.int(final_win_loc[-1, 1] + idx), 1]
                            
                            out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]): 
                                                    np.int(final_win_loc[-1, 1] + idx)]
                            
                            # Red Channel values
                            out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]): 
                                                    np.int(final_win_loc[-1, 2] + idx), 2]
                            
                            out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]): 
                                                    np.int(final_win_loc[-1, 2] + idx)]
                            
                        # If no errors occured (not in the begining of values)
                        # check index reaches beyond the end of values
                        else:
                            try:
                                bin_c[np.int(final_win_loc[-1, 2] + idx)] >= 255
                            # If Error raised (beyond end of values)
                            except:
                                # Green Channel values
                                out_win_vals[:, 1] = hist_norm[np.int(final_win_loc[0, 1]
                                                 - idx):np.int(final_win_loc[-1, 1]), 1]
                                
                                out_binlocs[:,1] = bin_c[np.int(final_win_loc[0, 1]
                                                 - idx):np.int(final_win_loc[-1, 1])]
                                
                                # Red Channel values
                                out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]
                                                  - idx):np.int(final_win_loc[-1, 2]), 2]
                                
                                out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]
                                                  - idx):np.int(final_win_loc[-1, 2])]
                                
                            # If no error raised (not beyond or below range of values)
                            else:
                                # Green Channel values
                                out_win_vals[:, 1] = hist_norm[np.int(
                                            final_win_loc[0, 1] - idx):np.int(
                                            final_win_loc[-1, 1] + idx), 1]
                                
                                out_binlocs[:,1] = bin_c[np.int(
                                            final_win_loc[0, 1] - idx):np.int(
                                            final_win_loc[-1, 1] + idx)]
                                
                                # Red Channel values
                                out_win_vals[:, 2] = hist_norm[np.int(
                                            final_win_loc[0, 2] - idx):np.int(
                                            final_win_loc[-1, 2] + idx), 2]
                                
                                out_binlocs[:,2] = bin_c[np.int(
                                            final_win_loc[0, 2] - idx):np.int(
                                            final_win_loc[-1, 2] + idx)]
                                
                    # Store resulting arrays as a pd Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return resulting arrays
                    return(out)
            # If Red & Blue channels present
            elif np.any(final_win_loc[:, 0]) != 0:
                # Evaluate Red channel's CDF
                ur = win_vals[:, 2] / 32
                vr = over_vals[:, 2] / 32
                
                # Evaluate difference of plain & overlapping CDF's
                Tks_r = np.max(np.abs(ur - vr))
                
                # Evaluate Blue Channel's CDF
                ub = win_vals[:, 0] / 32
                vb = over_vals[:, 0] / 32
                
                # Evaluate difference of plain & overlapping CDF's
                Tks_b = np.max(np.abs(ub - vb))
                
                # If Alternative Hypothesis (H1) true
                if Tks_r >= 0.5 and Tks_b >= 0.5:
                    
                    # Utilize alternative hypothesis function
                    (out_binlocs, out_win_vals) = alt_hyp(final_win_loc[:, :]
                                                      , final_win_vals[:, :])
                    
                    # Return resulting arrays as a pandas Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return output
                    return(out)
                    
                # If Null Hypothesis (H0) true
                else:
                    # Initialize index for Colour Channels
                    idx = 1
                    cdx = 1
                    while Tks_r < 0.5 or Tks_b < 0.5:
                        # Initialize arrays to expand Red & Blue Channels values
                        temp_win = np.zeros((32 + idx, 3), dtype=np.float32)
                        temp_over = np.zeros((32 + idx, 3), dtype=np.float32)
                        
                        # If at begining of histogram values(first window)
                        if np.logical_or(final_win_loc[0, 2] == 0,
                                         final_win_loc[0, 0] == 0):
                            if idx <= 15:
                                # Red Channel Windows
                                temp_win[:, 2] = hist_norm[0:32+idx, 2]
                                temp_over[:, 2] = hist_norm[np.abs(15 - idx):47, 2]
                                
                                # Blue Channel WIndows
                                temp_win[:, 0] = hist_norm[0:32+idx, 0]
                                temp_over[:, 0] = hist_norm[np.abs(15 - idx):47, 0]
                            else:
                                # Red Channel Windows
                                temp_win[:, 2] = hist_norm[0:32+idx, 2]
                                temp_over[:, 2] = hist_norm[0:47+cdx, 2]
                                
                                # Blue Channel WIndows
                                temp_win[:, 0] = hist_norm[0:32+idx, 0]
                                temp_over[:, 0] = hist_norm[0:47+cdx, 0]
                                # Increment index
                                cdx += 1
                        
                        # If at the end of histogram values (last window)
                        elif np.logical_or.reduce((final_win_loc[0, 2] == 224 ,
                                           final_win_loc[0, 0] == 224,
                                           final_win_loc[0, 0] == 239,
                                           final_win_loc[0, 2] == 239)):
                            
                            # Red Channel Windows
                            temp_win[:, 2] = hist_norm[np.int(win_binloc[0, 2]) 
                                              -idx: np.int(win_binloc[-1, 2])+1, 2]
                            
                            temp_over[:, 2] = np.pad(hist_norm[np.int(over_binloc[0, 2])
                            -idx: np.int(over_binloc[16, 2]), 2],
                            (0, len(temp_over) - len(hist_norm[np.int(over_binloc[0, 2])
                            -idx: np.int(over_binloc[16, 2]), 2])), 'constant')
                            
                            # Blue Channel Windows
                            temp_win[:, 0] = hist_norm[np.int(win_binloc[0, 0])
                                              -idx: np.int(win_binloc[-1, 0])+1, 0]
                            
                            temp_over[:, 0] = np.pad(hist_norm[np.int(over_binloc[ 0, 0])
                                               -idx : np.int(over_binloc[16, 0]), 0],
                            (0, len(temp_over) - len(hist_norm[np.int(over_binloc[ 0, 0])
                                               -idx : np.int(over_binloc[16, 0]), 0])),
                            'constant')
                            
                        else:
                            
                            # Red Channel Windows
                            temp_win[:, 2] = hist_norm[np.int(win_binloc[0, 2])
                                       : np.int(win_binloc[-1, 2]) + idx, 2]
                            
                            temp_over[:, 2] = hist_norm[np.int(over_binloc[i, 0, 2])
                                           -idx : np.int(over_binloc[-1, 2]) , 2]
                            
                            # Blue Channel Windows
                            temp_win[:, 0] = hist_norm[np.int(win_binloc[0, 0])
                                        : np.int(win_binloc[-1, 0]) + idx, 0]
                            
                            temp_over[:, 0] = hist_norm[np.int(over_binloc[0, 0])
                                        -idx : np.int(over_binloc[-1, 0]), 0]
                            
                        # Re Evaluate Red channel
                        ur = temp_win[:, 2] / len(temp_win)
                        vr = temp_over[:, 2] / len(temp_over)
                        Tks_r = np.max(np.abs(ur - vr))
                        
                        # Re Evaluate Blue Channel
                        ub = temp_win[:, 0] / len(temp_win)
                        vb = temp_over[:, 0] / len(temp_over)
                        Tks_b = np.max(np.abs(ub - vb))
                        
                        # Increase index value for next iteration
                        idx += 1
                        
                    # Initialize output arrays
                    out_binlocs = np.zeros((32 + idx, 3), dtype=np.uint8)
                    out_win_vals = np.zeros((32 + idx, 3), dtype=np.float32)
                    
                    # If current window is first window
                    if final_win_loc[0, 0] == 0:
                        # Blue Channel values
                        out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0]):
                                             np.int(final_win_loc[-1, 0] + idx), 0]
                        
                        out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0]):
                                             np.int(final_win_loc[-1, 0] + idx)]
                        
                        # Red Channel values
                        out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]):
                                             np.int(final_win_loc[-1, 2] + idx), 2]
                        
                        out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]):
                                             np.int(final_win_loc[-1, 2] + idx)]
                    
                    # If current window is last window
                    elif np.logical_or.reduce((final_win_loc[0, 2] == 224 ,
                                           final_win_loc[0, 0] == 224,
                                           final_win_loc[0, 0] == 239,
                                           final_win_loc[0, 2] == 239)):
                        # Blue Channel values
                        out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0]
                                          - idx):np.int(final_win_loc[-1, 0]), 0]
                        
                        out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0]
                                          - idx):np.int(final_win_loc[-1, 0])]
                        
                        # Red Channel values
                        out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]
                                          - idx):np.int(final_win_loc[-1, 2]), 2]
                        
                        out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]
                                          - idx):np.int(final_win_loc[-1, 2])]
                        
                    # If current window any window except first or last
                    else:
                        
                        # Check if index reaches before beginig of values
                        try:
                            bin_c[np.int(final_win_loc[0, 0] - idx)] >= 0
                        # If Error raised
                        except:
                            # Blue Channel values
                            out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0]): 
                                                    np.int(final_win_loc[-1, 0] + idx), 0]
                            
                            out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0]): 
                                                    np.int(final_win_loc[-1, 0] + idx)]
                            
                            # Red Channel values
                            out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]): 
                                                    np.int(final_win_loc[-1, 2] + idx), 2]
                            
                            out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]): 
                                                    np.int(final_win_loc[-1, 2] + idx)]
                            
                        # If no errors occured (not in the begining of values)
                        # check index reaches beyond the end of values
                        else:
                            try:
                                bin_c[np.int(final_win_loc[-1, 0] + idx)] >= 255
                            # If Error raised (beyond end of values)
                            except:
                                # Blue Channel values
                                out_win_vals[:, 0] = hist_norm[np.int(final_win_loc[0, 0] 
                                                    -idx):np.int(final_win_loc[-1, 0]), 0]
                                
                                out_binlocs[:,0] = bin_c[np.int(final_win_loc[0, 0] 
                                                    -idx):np.int(final_win_loc[-1, 0])]
                                
                                # Red Channel values
                                out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]
                                                  - idx):np.int(final_win_loc[-1, 2]), 2]
                                
                                out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]
                                                  - idx):np.int(final_win_loc[-1, 2])]
                                
                            # If no error raised (not beyond or below range of values)
                            else:
                                # Blue Channel values
                                out_win_vals[:, 0] = hist_norm[np.int(
                                            final_win_loc[0, 0] - idx):np.int(
                                            final_win_loc[-1, 0] + idx), 0]
                                
                                out_binlocs[:,0] = bin_c[np.int(
                                            final_win_loc[0, 0] - idx):np.int(
                                            final_win_loc[-1, 0] + idx)]
                                
                                # Red Channel values
                                out_win_vals[:, 2] = hist_norm[np.int(
                                            final_win_loc[0, 2] - idx):np.int(
                                            final_win_loc[-1, 2] + idx), 2]
                                
                                out_binlocs[:,2] = bin_c[np.int(
                                            final_win_loc[0, 2] - idx):np.int(
                                            final_win_loc[-1, 2] + idx)]
                                
                    # Store resulting arrays as a pd Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return resulting arrays
                    return(out)
            # If only Red channel present
            else:
                # Evaluate Red channel's CDF
                ur = win_vals[:, 2] / 32
                vr = over_vals[:, 2] / 32
                
                # Evaluate difference of plain & overlapping CDF's
                Tks_r = np.max(np.abs(ur - vr))
                
                # If Alternative Hypothesis (H1) true
                if Tks_r >= 0.5:
                    
                    # Utilize alternative hypothesis function
                    (out_binlocs, out_win_vals) = alt_hyp(final_win_loc[:, :]
                                                      , final_win_vals[:, :])
                    
                    # Return resulting arrays as a pandas Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return output
                    return(out)
                    
                # If Null Hypothesis (H0) true
                else:
                    while Tks_r < 0.5:
                        # Initialize arrays to expand Red Channel values
                        temp_win = np.zeros((32 + idx, 3), dtype=np.float32)
                        temp_over = np.zeros((32 + idx, 3), dtype=np.float32)
                        
                        # If at begining of histogram values(first window)
                        if final_win_loc[0, 2] == 0:
                            if idx <= 15:
                                # Red Channel Windows
                                temp_win[:, 2] = hist_norm[0:32+idx, 2]
                                temp_over[:, 2] = hist_norm[np.abs(15 - idx):47, 2]
                            else:
                                # Red Channel Windows
                                temp_win[:, 2] = hist_norm[0:32+idx, 2]
                                temp_over[:, 2] = hist_norm[0:47+cdx, 2]
                                # Increment index
                                cdx += 1
                        
                        # If at the end of histogram values (last window)
                        elif final_win_loc[0, 2] == 224 or final_win_loc[0,2] == 239:
                            
                            # Red Channel Windows
                            temp_win[:, 2] = hist_norm[np.int(win_binloc[0, 2]) 
                                              -idx: np.int(win_binloc[-1, 2])+1, 2]
                            
                            temp_over[:, 2] = np.pad(hist_norm[np.int(over_binloc[0, 2])
                            -idx: np.int(over_binloc[16, 2]), 2], 
                        (0, len(temp_over) - len(hist_norm[np.int(over_binloc[0, 2])
                            -idx: np.int(over_binloc[16, 2]), 2])), 'constant')
                                
                        else:
                            
                            # Red Channel Windows
                            temp_win[:, 2] = hist_norm[np.int(win_binloc[0, 2])
                                        : np.int(win_binloc[-1, 2]) + idx, 2]
                            
                            temp_over[:, 2] = hist_norm[np.int(over_binloc[i, 0, 2])
                                           -idx : np.int(over_binloc[-1, 2]), 2]
                            
                            
                        # Re Evaluate Red channel
                        ur = temp_win[:, 2] / len(temp_win)
                        vr = temp_over[:, 2] / len(temp_over)
                        Tks_r = np.max(np.abs(ur - vr))
                        
                        # Increase index value for next iteration
                        idx += 1
                        
                    # Initialize output arrays
                    out_binlocs = np.zeros((32 + idx, 3), dtype=np.uint8)
                    out_win_vals = np.zeros((32 + idx, 3), dtype=np.float32)
                    
                    # If current window is first window
                    if final_win_loc[0, 2] == 0:
                        # Red Channel values
                        out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]):
                                             np.int(final_win_loc[-1, 2] + idx), 2]
                        
                        out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]):
                                             np.int(final_win_loc[-1, 2] + idx)]
                    
                    # If current window is last window
                    elif final_win_loc[0, 2] == 224 or final_win_loc[0,2] == 239:
                        # Red Channel values
                        out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]
                                          - idx):np.int(final_win_loc[-1, 2]), 2]
                        
                        out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]
                                          - idx):np.int(final_win_loc[-1, 2])]
                        
                    # If current window any window except first or last
                    else:
                        
                        # Check if index reaches before beginig of values
                        try:
                            bin_c[np.int(final_win_loc[0, 2] - idx)] >= 0
                        # If Error raised
                        except:
                            # Red Channel values
                            out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]): 
                                                 np.int(final_win_loc[-1, 2] + idx), 2]
                            
                            out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]): 
                                                 np.int(final_win_loc[-1, 2] + idx)]
                            
                        # If no errors occured (not in the begining of values)
                        # check index reaches beyond the end of values
                        else:
                            try:
                                bin_c[np.int(final_win_loc[-1, 2] + idx)] >= 255
                            # If Error raised (beyond end of values)
                            except:
                                # Red Channel values
                                out_win_vals[:, 2] = hist_norm[np.int(final_win_loc[0, 2]
                                                 - idx):np.int(final_win_loc[-1, 2]), 2]
                                
                                out_binlocs[:,2] = bin_c[np.int(final_win_loc[0, 2]
                                                 - idx):np.int(final_win_loc[-1, 2])]
                                
                            # If no error raised (not beyond or below range of values)
                            else:
                                # Red Channel values
                                out_win_vals[:, 2] = hist_norm[np.int(
                                        final_win_loc[0, 2] - idx):np.int(
                                        final_win_loc[-1, 2] + idx), 2]
                                
                                out_binlocs[:,2] = bin_c[np.int(
                                        final_win_loc[0, 2] - idx):np.int(
                                        final_win_loc[-1, 2] + idx)]
                                
                    # Store resulting arrays as a pd Series object
                    out = pd.Series((out_binlocs, out_win_vals))
                    
                    # Return resulting arrays
                    return(out)
        # If no colour channel present return zeros
        else:
            # Initialize arrays
            out_binlocs = np.zeros((32, 3), dtype=np.uint8)
            out_win_vals = np.zeros((32, 3), dtype=np.float32)
            
            # Store resulting arrays as a pd Series object
            out = pd.Series((out_binlocs, out_win_vals))
            
            # Return resulting arrays
            return(out)
    
    
    
    # Initialize lists to output results
    out_binlocs_l = []
    out_win_vals_l = []
    
    # Initialize number of clusters
    n_clusters = 2    # Starting from 2 since background is a cluster & possible
                      # artifacts
    
    # Enumerate histogram bins
    bin_c = np.arange(0, 256, dtype=np.uint8)
    
    for i in range(8):
        # If all channels present
        if np.logical_and.reduce((np.any(final_win_loc[i, :, 0]) != 0,
                                  np.any(final_win_loc[i, :, 1]) != 0,
                                  np.any(final_win_loc[i, :, 2]) != 0)):
            # Initialize lists to store values
            out_binlocs_list = []
            out_win_vals_list = []
            # Append bin locations
            out_binlocs_list.append(final_win_loc[i, :, :])
            
            # Append windows values
            out_win_vals_list.append(final_win_vals[i, :, :])
            
            # Convert resulting lists to pandas Series objects
            binlocs_pd = pd.Series(out_binlocs_list)
            win_vals_pd = pd.Series(out_win_vals_list)
            
            # Clear lists
            out_binlocs_list.clear()
            out_win_vals_list.clear()
            
            # Store results to arrays
            out_binlocs_temp = binlocs_pd[0]
            out_win_vals_temp = win_vals_pd[0]
            
            # Check if current windows have the same bin location for each
            # Colour channel
            if np.logical_and(out_binlocs_temp[:, 0].all() == out_binlocs_temp[:, 1].all(),
                              out_binlocs_temp[:, 1].all() == out_binlocs_temp[:, 2].all()):
                
                out_binlocs_l.append(out_binlocs_temp)
                out_win_vals_l.append(out_win_vals_temp)
                
                # Increment number of clusters by one
                n_clusters += 1
                
            # If the bin location is not the same for any colour channel
            else:
                # Perform KS-test
                out_binlocs, out_win_vals = ks_test_tree(win_vals[i, :, :], 
                    win_binloc[i, :, :], over_vals[i, :, :], over_binloc[i, :, :],
                    hist_norm,final_win_vals[i, :, :],final_win_loc[i, :, :], bin_c)
                
                out_binlocs_l.append(out_binlocs)
                out_win_vals_l.append(out_win_vals)
                
                # Check if bin locations remain the same
                if np.logical_and(out_binlocs[:, 0].all() == out_binlocs[:, 1].all(),
                                  out_binlocs[:, 1].all() == out_binlocs[:, 2].all()):
                    # Increment number of clusters by 1
                    n_clusters += 1
                # Check if Blue & Green bin locations same, but not Red's
                elif np.logical_and(out_binlocs[:, 0].all() == out_binlocs[:, 1].all(),
                                    out_binlocs[:, 1].all() != out_binlocs[:, 2].all()):
                    # Increment number of clusters by 2
                    n_clusters += 2
                    
                # Check if Blue & Red bin locations same, but not Green's
                elif np.logical_and(out_binlocs[:, 0].all() == out_binlocs[:, 2].all(),
                                    out_binlocs[:, 2].all() != out_binlocs[:, 1].all()):
                    # Increment number of cluster by 2
                    n_clusters += 2
                    
                # Check if Red & Green bin locations same, but not Blue's
                elif np.logical_and(out_binlocs[:, 1].all() == out_binlocs[:, 2].all(),
                                    out_binlocs[:, 2].all() != out_binlocs[:, 0].all()):
                    # Increment number of clusters by 2
                    n_clusters += 2
                    
                # If every colour's bin locations unequal
                else:
                    # Increment number of clusters by 3
                    n_clusters += 3
                    
        # If Blue channel present
        elif np.any(final_win_loc[i, :, 0]) != 0:
        # If Blue and Red channels present
            if np.any(final_win_loc[i, :, 2]) != 0:
                # Initialize lists to store values
                out_binlocs_list = []
                out_win_vals_list = []
                
                # Append bin locations
                out_binlocs_list.append(final_win_loc[i, :, :])
                
                # Append windows values
                out_win_vals_list.append(final_win_vals[i, :, :])
                
                # Convert resulting lists to pandas Series objects
                binlocs_pd = pd.Series(out_binlocs_list)
                win_vals_pd = pd.Series(out_win_vals_list)
                
                # Clear lists
                out_binlocs_list.clear()
                out_win_vals_list.clear()
                
                # Store results to arrays
                out_binlocs_temp = binlocs_pd[0]
                out_win_vals_temp = win_vals_pd[0]
                
                # Check if current windows have the same bin location for each
                # Colour channel
                if np.all(out_binlocs_temp[:, 0] == out_binlocs_temp[:, 2]):
                    
                    out_binlocs_l.append(out_binlocs_temp)
                    out_win_vals_l.append(out_win_vals_temp)
                    
                    # Increment number of clusters by one
                    n_clusters += 1
                    
                # If the bin location is not the same for any colour channel
                else:
                    # Perform KS-test
                    out_binlocs, out_win_vals = ks_test_tree(win_vals[i, :, :], 
                    win_binloc[i, :, :], over_vals[i, :, :], over_binloc[i, :, :],
                    hist_norm,final_win_vals[i, :, :],final_win_loc[i, :, :], bin_c)
                    
                    out_binlocs_l.append(out_binlocs)
                    out_win_vals_l.append(out_win_vals)
                    
                    # Check if bin locations remain the same
                    if out_binlocs[:, 0].all() == out_binlocs[:, 2].all():
                        # Increment number of clusters by 1
                        n_clusters += 1
                    # If Blue & Red colour bin locations unequal
                    else:
                        # Increment number of clusters by 2
                        n_clusters += 2
            # If Blue & Green channes present
            elif np.any(final_win_loc[i, :, 1]) != 0:
                # Initialize lists to store values
                out_binlocs_list = []
                out_win_vals_list = []
                
                # Append bin locations
                out_binlocs_list.append(final_win_loc[i, :, :])
                
                # Append windows values
                out_win_vals_list.append(final_win_vals[i, :, :])
                
                # Convert resulting lists to pandas Series objects
                binlocs_pd = pd.Series(out_binlocs_list)
                win_vals_pd = pd.Series(out_win_vals_list)
                
                # Clear lists
                out_binlocs_list.clear()
                out_win_vals_list.clear()
                
                # Store results to arrays
                out_binlocs_temp = binlocs_pd[0]
                out_win_vals_temp = win_vals_pd[0]
                
                # Check if current windows have the same bin location for each
                # Colour channel
                if np.all(out_binlocs_temp[:, 0] == out_binlocs_temp[:, 1]):
                    
                    out_binlocs_l.append(out_binlocs_temp)
                    out_win_vals_l.append(out_win_vals_temp)
                    
                    # Increment number of clusters by one
                    n_clusters += 1
                    
                # If the bin location is not the same for any colour channel
                else:
                    # Perform KS-test
                    out_binlocs, out_win_vals = ks_test_tree(win_vals[i, :, :], 
                    win_binloc[i, :, :], over_vals[i, :, :], over_binloc[i, :, :],
                    hist_norm,final_win_vals[i, :, :],final_win_loc[i, :, :], bin_c)
                    
                    out_binlocs_l.append(out_binlocs)
                    out_win_vals_l.append(out_win_vals)
                    
                    # Check if bin locations remain the same
                    if out_binlocs[:, 0].all() == out_binlocs[:, 1].all():
                        # Increment number of clusters by 1
                        n_clusters += 1
                    # If Blue & Red colour bin locations unequal
                    else:
                        # Increment number of clusters by 2
                        n_clusters += 2
            
            # If only Blue channel present
            else:
                # Initialize lists to store values
                out_binlocs_list = []
                out_win_vals_list = []
                
                # Append bin locations
                out_binlocs_list.append(final_win_loc[i, :, :])
                
                # Append windows values
                out_win_vals_list.append(final_win_vals[i, :, :])
                
                # Convert resulting lists to pandas Series objects
                binlocs_pd = pd.Series(out_binlocs_list)
                out_win_vals_pd = pd.Series(out_win_vals_list)
                
                # Clear lists
                out_binlocs_list.clear()
                out_win_vals_list.clear()
                
                # Store results to arrays
                out_binlocs_temp = binlocs_pd[0]
                out_win_vals_temp = out_win_vals_pd[0]
                
                out_binlocs_l.append(out_binlocs_temp)
                out_win_vals_l.append(out_win_vals_temp)
                
                # Increment number of clusters by 1
                n_clusters += 1
        # If Green channel present
        elif np.any(final_win_loc[i, :, 1]) != 0:
            # If Green & Blue channels present
            if np.any(final_win_loc[i, :, 0]) != 0:
                # Initialize lists to store values
                out_binlocs_list = []
                out_win_vals_list = []
                
                # Append bin locations
                out_binlocs_list.append(final_win_loc[i, :, :])
                
                # Append windows values
                out_win_vals_list.append(final_win_vals[i, :, :])
                
                # Convert resulting lists to pandas Series objects
                binlocs_pd = pd.Series(out_binlocs_list)
                out_win_vals_pd = pd.Series(out_win_vals_list)
                
                # Clear lists
                out_binlocs_list.clear()
                out_win_vals_list.clear()
                
                # Store results to arrays
                out_binlocs_temp = binlocs_pd[0]
                out_win_vals_temp = out_win_vals_pd[0]
                
                # Check if current windows have the same bin location for each
                # Colour channel
                if np.all(out_binlocs_temp[:, 0] == out_binlocs_temp[:, 1]):
                    
                    out_binlocs_l.append(out_binlocs_temp)
                    out_win_vals_l.append(out_win_vals_temp)
                    # Increment number of clusters by one
                    n_clusters += 1
                    
                # If the bin location is not the same for any colour channel
                else:
                    # Perform KS-test
                    out_binlocs, out_win_vals = ks_test_tree(win_vals[i, :, :], 
                    win_binloc[i, :, :], over_vals[i, :, :], over_binloc[i, :, :],
                    hist_norm,final_win_vals[i, :, :],final_win_loc[i, :, :], bin_c)
                    
                    out_binlocs_l.append(out_binlocs)
                    out_win_vals_l.append(out_win_vals)
                    
                    # Check if bin locations remain the same
                    if out_binlocs[:, 0].all() == out_binlocs[:, 1].all():
                        # Increment number of clusters by 1
                        n_clusters += 1
                    # If Blue & Red colour bin locations unequal
                    else:
                        # Increment number of clusters by 2
                        n_clusters += 2
            # If Green & Red channes present
            elif np.any(final_win_loc[i, :, 2]) != 0:
                # Initialize lists to store values
                out_binlocs_list = []
                out_win_vals_list = []
                
                # Append bin locations
                out_binlocs_list.append(final_win_loc[i, :, :])
                
                # Append windows values
                out_win_vals_list.append(final_win_vals[i, :, :])
                
                # Convert resulting lists to pandas Series objects
                binlocs_pd = pd.Series(out_binlocs_list)
                win_vals_pd = pd.Series(out_win_vals_list)
                
                # Clear lists
                out_binlocs_list.clear()
                out_win_vals_list.clear()
                
                # Store results to arrays
                out_binlocs_temp = binlocs_pd[0]
                out_win_vals_temp = win_vals_pd[0]
                
                # Check if current windows have the same bin location for each
                # Colour channel
                if np.all(out_binlocs_temp[:, 1] == out_binlocs_temp[:, 2]):
                    
                    out_binlocs_l.append(out_binlocs_temp)
                    out_win_vals_l.append(out_win_vals_temp)
                    
                    # Increment number of clusters by one
                    n_clusters += 1
                    
                # If the bin location is not the same for any colour channel
                else:
                    # Perform KS-test
                    out_binlocs, out_win_vals = ks_test_tree(win_vals[i, :, :], 
                    win_binloc[i, :, :], over_vals[i, :, :], over_binloc[i, :, :],
                    hist_norm,final_win_vals[i, :, :],final_win_loc[i, :, :], bin_c)
                    
                    out_binlocs_l.append(out_binlocs)
                    out_win_vals_l.append(out_win_vals)
                    
                    # Check if bin locations remain the same
                    if out_binlocs[:, 1].all() == out_binlocs[:, 2].all():
                        
                        # Increment number of clusters by 1
                        n_clusters += 1
                    
                    # If Green & Red colour bin locations unequal
                    else:
                        # Increment number of clusters by 2
                        n_clusters += 2
            
            # If only Green channel present
            else:
                # Initialize lists to store values
                out_binlocs_list = []
                out_win_vals_list = []
                
                # Append bin locations
                out_binlocs_list.append(final_win_loc[i, :, :])
                
                # Append windows values
                out_win_vals_list.append(final_win_vals[i, :, :])
                
                # Convert resulting lists to pandas Series objects
                binlocs_pd = pd.Series(out_binlocs_list)
                win_vals_pd = pd.Series(out_win_vals_list)
                
                # Clear lists
                out_binlocs_list.clear()
                out_win_vals_list.clear()
                
                # Store results to arrays
                out_binlocs_temp = binlocs_pd[0]
                out_win_vals_temp = win_vals_pd[0]
                
                out_binlocs_l.append(out_binlocs_temp)
                out_win_vals_l.append(out_win_vals_temp)
                
                # Increment number of clusters by 1
                n_clusters += 1
        # If Red channel present
        elif np.any(final_win_loc[i, :, 2]) != 0:
            # If Red & Blue present
            if np.any(final_win_loc[i, :, 0]) != 0:
                # Initialize lists to store values
                out_binlocs_list = []
                out_win_vals_list = []
                
                # Append bin locations
                out_binlocs_list.append(final_win_loc[i, :, :])
                
                # Append windows values
                out_win_vals_list.append(final_win_vals[i, :, :])
                
                # Convert resulting lists to pandas Series objects
                binlocs_pd = pd.Series(out_binlocs_list)
                win_vals_pd = pd.Series(out_win_vals_list)
                
                # Clear lists
                out_binlocs_list.clear()
                out_win_vals_list.clear()
                
                # Store results to arrays
                out_binlocs_temp = binlocs_pd[0]
                out_win_vals_temp = win_vals_pd[0]
                
                # Check if current windows have the same bin location for each
                # Colour channel
                if np.all(out_binlocs_temp[:, 0] == out_binlocs_temp[:, 2]):
                    
                    out_binlocs_l.append(out_binlocs_temp)
                    out_win_vals_l.append(out_win_vals_temp)
                    
                    # Increment number of clusters by one
                    n_clusters += 1
                    
                # If the bin location is not the same for any colour channel
                else:
                    # Perform KS-test
                    out_binlocs, out_win_vals = ks_test_tree(win_vals[i, :, :], 
                    win_binloc[i, :, :], over_vals[i, :, :], over_binloc[i, :, :],
                    hist_norm,final_win_vals[i, :, :],final_win_loc[i, :, :], bin_c)
                    
                    out_binlocs_l.append(out_binlocs)
                    out_win_vals_l.append(out_win_vals)
                    
                    # Check if bin locations remain the same
                    if out_binlocs[:, 0].all() == out_binlocs[:, 2].all():
                        # Increment number of clusters by 1
                        n_clusters += 1
                    # If Blue & Red colour bin locations unequal
                    else:
                        # Increment number of clusters by 2
                        n_clusters += 2
            # If Red & Green channes present
            elif np.any(final_win_loc[i, :, 1]) != 0:
                # Initialize lists to store values
                out_binlocs_list = []
                out_win_vals_list = []
                
                # Append bin locations
                out_binlocs_list.append(final_win_loc[i, :, :])
                
                # Append windows values
                out_win_vals_list.append(final_win_vals[i, :, :])
                
                # Convert resulting lists to pandas Series objects
                binlocs_pd = pd.Series(out_binlocs_list)
                win_vals_pd = pd.Series(out_win_vals_list)
                
                # Clear lists
                out_binlocs_list.clear()
                out_win_vals_list.clear()
                
                # Store results to arrays
                out_binlocs_temp = binlocs_pd[0]
                out_win_vals_temp = win_vals_pd[0]
                
                # Check if current windows have the same bin location for each
                # Colour channel
                if np.all(out_binlocs_temp[:, 2] == out_binlocs_temp[:, 1]):
                    
                    out_binlocs_l.append(out_binlocs_temp)
                    out_win_vals_l.append(out_win_vals_temp)
                    
                    # Increment number of clusters by one
                    n_clusters += 1
                    
                # If the bin location is not the same for any colour channel
                else:
                    # Perform KS-test
                    out_binlocs, out_win_vals = ks_test_tree(win_vals[i, :, :], 
                    win_binloc[i, :, :], over_vals[i, :, :], over_binloc[i, :, :],
                    hist_norm,final_win_vals[i, :, :],final_win_loc[i, :, :], bin_c)
                    
                    out_binlocs_l.append(out_binlocs)
                    out_win_vals_l.append(out_win_vals)
                    
                    # Check if bin locations remain the same
                    if out_binlocs[:, 2].all() == out_binlocs[:, 1].all():
                        # Increment number of clusters by 1
                        n_clusters += 1
                    # If Blue & Red colour bin locations unequal
                    else:
                        # Increment number of clusters by 2
                        n_clusters += 2
            
            # If only Red channel present
            else:
                # Initialize lists to store values
                out_binlocs_list = []
                out_win_vals_list = []
                
                # Append bin locations
                out_binlocs_list.append(final_win_loc[i, :, :])
                
                # Append windows values
                out_win_vals_list.append(final_win_vals[i, :, :])
                
                # Convert resulting lists to pandas Series objects
                binlocs_pd = pd.Series(out_binlocs_list)
                win_vals_pd = pd.Series(out_win_vals_list)
                
                # Clear lists
                out_binlocs_list.clear()
                out_win_vals_list.clear()
                
                # Store results to arrays
                out_binlocs_temp = binlocs_pd[0]
                out_win_vals_temp = win_vals_pd[0]
                
                out_binlocs_l.append(out_binlocs_temp)
                out_win_vals_l.append(out_win_vals_temp)
                
                # Increment number of clusters by 1
                n_clusters += 1
        # If no colour channel present
        else:
            # Continue to next windows
            continue
    
    # Convert output lists into pandas Series objects
    out_binlocs_pd = pd.Series(out_binlocs_l)
    
    out_win_vals_pd = pd.Series(out_win_vals_l)
    
    # Test if any results are present
    try:
        out_binlocs_pd[0]
    except:
        # If not present, set outputs to zero
        out_binlocs = 0
        out_win_vals = 0
    else:
        # Initialize output arrays
        out_binlocs = np.zeros((np.int8(len(out_binlocs_pd)),
                                np.int8(len(out_binlocs_pd[0])), 3), dtype=np.uint8)
        
        out_win_vals = np.zeros((np.int8(len(out_binlocs_pd)),
                                 np.int8(len(out_binlocs_pd[0])), 3), dtype=np.float32)
        
        # Loop & store each window
        for i in range(len(out_binlocs_pd)):
            out_binlocs[i, :, :] = out_binlocs_pd[i]
            
            out_win_vals[i, :, :] = out_win_vals_pd[i]
    
    # If flag is set to 1, output the resulting windows
    if deb_flg == 1:
        out = pd.Series((out_binlocs, out_win_vals, n_clusters))
        return(out)
    # Else, return the number of estimated clusters
    else:
        return(n_clusters)

