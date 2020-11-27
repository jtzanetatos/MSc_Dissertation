# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from scipy.stats import norm, entropy, kstest


# --------------------------------------------------------------------------- #
# Kolmogorov-Sminrov Statistical Test to converge MOG2
# --------------------------------------------------------------------------- #
def KSMoG2(curr_hist, prev_hist):
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
        return True
    # If Alternative Hypothesis true
    else:
        return False



# -------------------------------------------------------------------------- #
# Histogram Standard Deviation to converge MOG2
# -------------------------------------------------------------------------- #
def HistDeviation(out_win_vals):
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
            return True
        else:
            return False