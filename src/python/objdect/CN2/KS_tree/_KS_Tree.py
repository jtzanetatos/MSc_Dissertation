# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from scipy.stats import norm, entropy, kstest

# --------------------------------------------------------------------------- #
# Implement adaptive windows of interest by means of KS statistical test
# --------------------------------------------------------------------------- #
def KSAdaptiveWindows(final_win_vals, final_win_loc, hist_norm,
              win_vals, win_binloc, over_vals, over_binloc):
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
    
    '''
    # If all colour channels present
    if np.logical_and.reduce((np.any(final_win_loc[:, 0]) != 0,
                              np.any(final_win_loc[:, 1]) != 0,
                              np.any(final_win_loc[:, 2]) != 0)):
        
        # TODO: KS Test from scipy here
        
        # If Alternative Hypothesis (H1) true
        if Tks_r >= 0.5 and Tks_g >= 0.5 and Tks_b >= 0.5:
            
            # TODO: Utilize alternative hypothesis function
            
            
        # If Null Hypothesis (H0) true
        else:
            
            # TODO: NULL Hypothesis call
            
    # Determine if Blue channel is present
    if np.any(final_win_loc[:, 0]) != 0:
        # Check if Red & Blue channels present only
        if np.any(final_win_loc[:, 2]) != 0:
            
            # TODO: KS Test from scipy here
            
            # If Alternative Hypothesis (H1) true
            if Tks_r >= 0.5 and Tks_b >= 0.5:
                
                # TODO: Utilize alternative hypothesis function
                
            # If Null Hypothesis (H0) true
            else:
                
                # TODO: NULL Hypothesis call
                
        
        # Check if Blue & Green channels are present
        elif np.any(final_win_loc[:, 1]) != 0:
            
            # TODO: KS Test from scipy here
            
            # If Alternative Hypothesis (H1) true
            if Tks_g >= 0.5 and Tks_b >= 0.5:
                
                # TODO: Utilize alternative hypothesis function
                
            # If Null Hypothesis (H0) true
            else:
                
                # TODO: NULL Hypothesis call
                
        
        # If only Blue Channel present
        else:
            
            # TODO: KS Test from scipy here
            
            # If Alternative Hypothesis (H1) true
            if Tks_b >= 0.5:
                
                # TODO: Utilize alternative hypothesis function
                
            # If Null Hypothesis (H0) true
            else:
                
                # TODO: NULL Hypothesis call
                
    # If Green Channel present
    elif np.any(final_win_loc[:, 1]) != 0:
        # If Green & Blue channels present
        if np.any(final_win_loc[:, 0]) != 0:
            
            # TODO: KS Test from scipy here
            
            # If Alternative Hypothesis (H1) true
            if Tks_g >= 0.5 and Tks_b >= 0.5:
                
                # TODO: Utilize alternative hypothesis function
                
            # If Null Hypothesis (H0) true
            else:
               
                # TODO: NULL Hypothesis call
                
        # If Green & Red channels present
        elif np.any(final_win_loc[:, 2]) != 0:
            
            # TODO: KS Test from scipy here
            
            # If Alternative Hypothesis (H1) true
            if Tks_r >= 0.5 and Tks_g >= 0.5:
                
                # TODO: Utilize alternative hypothesis function
                
            # If Null Hypothesis (H0) true
            else:
                
                # TODO: NULL Hypothesis call
                
        # If Green Channel only
        else:
            
            # TODO: KS Test from scipy here
            
            # If Alternative Hypothesis (H1) true
            if Tks_g >= 0.5:
                
                # TODO: Utilize alternative hypothesis function
                
            # If Null Hypothesis (H0) true
            else:
                
                # TODO: NULL Hypothesis call
                
    # If Red Chanel present
    elif np.any(final_win_loc[:, 2]) != 0:
        # If Red & Green Channels present
        if np.any(final_win_loc[:, 1]) != 0:
            
            # TODO: KS Test from scipy here
            
            # If Alternative Hypothesis (H1) true
            if Tks_r >= 0.5 and Tks_g >= 0.5:
                
                # TODO: Utilize alternative hypothesis function
                
            # If Null Hypothesis (H0) true
            else:
                
                # TODO: NULL Hypothesis call
                
        # If Red & Blue channels present
        elif np.any(final_win_loc[:, 0]) != 0:
            
            # TODO: KS Test from scipy here
            
            # If Alternative Hypothesis (H1) true
            if Tks_r >= 0.5 and Tks_b >= 0.5:
                
                # TODO: Utilize alternative hypothesis function
                
            # If Null Hypothesis (H0) true
            else:
                
                # TODO: NULL Hypothesis call
                
        # If only Red channel present
        else:
            
            # TODO: KS Test from scipy here
            
            # If Alternative Hypothesis (H1) true
            if Tks_r >= 0.5:
                
                # TODO: Utilize alternative hypothesis function
                
            # If Null Hypothesis (H0) true
            else:
                
                # TODO: NULL Hypothesis call
                
    # If no colour channel present return zeros
    else:
        # Initialize arrays
        out_binlocs = np.zeros((32, 3), dtype=np.uint8)
        out_win_vals = np.zeros((32, 3), dtype=np.float32)
    
    # Return resulting arrays
    return out_binlocs, out_win_vals
