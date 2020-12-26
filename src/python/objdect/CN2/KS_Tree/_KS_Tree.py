# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from ._Null_Hypothesis import NullHypothesis
from ._KS_Significance_Test import KSTest

# --------------------------------------------------------------------------- #
# Implement adaptive windows of interest by means of KS statistical test
# --------------------------------------------------------------------------- #
def KSAdaptiveWindows(CN2winVals, CN2winLoc, histNorm, winVals, winLoc):
    '''
    Adaptive windows utilizing the Kolmogorov-Smirnov statistical test 
    containing the resulting histograms of each colour channel.
    In its current implementation, the windows of interest,
    resulting from the Rule Based Tree (CN2) function are utilized.
    
    The Null Hypothesis (H0) is defined as the shape of input & adjucent 
    windows overlap, therefore the current window expands accordingly
    untill they no longer overlap significantly.
    
    The Alternative Hypothesis is defined as the shape of input & adjucent
    windows do not overlap significantly, therefore no need to expand current 
    window, keep current window as is.
    
    Parameters
    ----------
    CN2winVals : np.array
        Resulting window values from CN2 tree.
    CN2winLoc : np.array
        Resulting window locations from CN2 tree.
    histNorm : float32 array
        Normalized histogram of input frame.
    winVals : float32 array
        Normalized slidding windows values of input frame's histogram.
    winLoc : uint8 array
        Slidding windows locations of input frame's histogram.
    
    Returns
    -------
    ksVals : np.array
        Resulting adaptive window values of each colour channel.
    ksLoc : np.array
        Resulting adaptive window locations of each colour channel.
    
    '''
    # Initialize array to store adaptive windows values
    ksVals = np.empty(len(CN2winVals), dtype=np.object)
    # Initialize array to store adaptive windows locations
    ksLoc = np.empty(len(CN2winLoc), dtype=np.object)
    
    # Iterate over colour channels
    for i in range(len(CN2winVals)):
        # Current colour channel present
        if CN2winLoc[i] is not None:
            # Initialize nested array to store arrays - window values
            outWinvals = np.zeros(CN2winLoc[i].shape[0], dtype=np.object)
            
            # Initialize nested array to store arrays - window locations
            outBinlocs = np.zeros(CN2winLoc[i].shape[0], dtype=np.object)
            
            # Iterate over KS test results
            for k in range(CN2winLoc[i].shape[0]):
                # KS Significance test
                signCrit = KSTest(CN2winLoc[i][k], winVals[:, :, i], winLoc)
                # If Alternative Hypothesis (H1) true
                if signCrit < 5.0:
                    # Return input windows
                    outWinvals[k] = CN2winVals[i][k]
                    outBinlocs[k] = CN2winLoc[i][k]
                    
                # If Null Hypothesis (H0) true
                else:
                    # Expand windows
                    outWinvals[k], outBinlocs[k] = NullHypothesis(CN2winLoc[i][k],
                                                        histNorm[:, i], winLoc, signCrit)
            # Store resulting windows for current colour channel
            ksVals[i] = outWinvals
            ksLoc[i] = outBinlocs
    # Return resulting arrays
    return ksVals, ksLoc
