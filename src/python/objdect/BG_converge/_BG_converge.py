# -*- coding: utf-8 -*-
"""
"""

import numpy as np

# --------------------------------------------------------------------------- #
# Kolmogorov-Sminrov Statistical Test to converge MOG2
# --------------------------------------------------------------------------- #
def KSMoG2(curr_hist, prev_hist):
    '''
    Kolmogorov-Smirnov Statistical test for MOG2 convergence.
    Evaluates the Cumulative Densities Functions (CDF) of current & previous
    frame's histograms. Depending upon the results, the MOG2 algorithm will
    either continue, or stop(e.g. converged).
    
    Alternative hypothesis (H1) is defined as histograms do not overlap significantly,
    therefore the MOG2 algorithm has not converged yet.
    
    Null hypothesis (H0) is defined as histograms overlap significantly,
    therefore the MOG2 algorithm has converged.
    
    Parameters
    ----------
    curr_hist : float 32 2D array
        Histogram of current frame.
    prev_hist : float 32 2D array
        N-10th frame's histogram.
    
    Returns
    -------
    bool
       Results of KS test; True for Null hypothesis (H0), False for Alternative
       Hypothesis (H1).
    
    '''
    # Initialize array to store KS test results for each channel
    signCrit = np.zeros(3, dtype=np.float32)
    
    # Evaluate CDF of each colour channel for current & past frames
    for i in range(3):
        curr_cdf = np.cumsum(curr_hist[:, i])
        
        prev_cdf = np.cumsum(prev_hist[:, i])
        
        # Evaluate KS test for current colour channel
        signCrit[i] = np.max(np.abs(curr_cdf - prev_cdf))
    
    # Null Hypothesis true
    if ( signCrit[0] < 1.0 and signCrit[1] < 1.0 and signCrit[2] < 1.0):
        return True
    # Alternative Hypothesis true
    else:
        return False



# -------------------------------------------------------------------------- #
# Histogram Standard Deviation to converge MOG2
# -------------------------------------------------------------------------- #
def HistDeviation(out_win_vals):
    '''
    Evaluation the deviation of each resulting adaptive window, in order to
    evaluate if the MOG2 algorithm needs to converge or to continue. 
    Identifies the useful windows and evaluates its deviation.
    Useful windows are defined as the windows that contain values.
    
    Parameters
    ----------
    out_win_vals : np.array
        Adaptive window values.
    
    Returns
    -------
    bool
        True if the majority of the deviations are greater than the 
        threshold value, defined as 0.21.
        
        False if the majority of the deviations are less than the
        threshold value.
    
    '''
    try:
        # Obtain dimmentions of input array
        h, _, c = out_win_vals.shape
    except:
        # If no windows of interest, return false
        return False
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