# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from scipy.stats import norm, entropy, kstest

# TODO: Split code into other files


# --------------------------------------------------------------------------- #
# Implement adaptive windows of interest by means of KS statistical test
# --------------------------------------------------------------------------- #
def KSAdaptiveWindows(final_win_vals, final_win_loc, hist_norm,
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
