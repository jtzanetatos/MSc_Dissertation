# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 19:54:29 2020

@author: iason
"""

import numpy as np

def KSTest(CN2winLoc, winVals, winLoc):
    '''
    Kolmogorov Smirnov statistical significance test between window derived
    from CN2 tree and its adjacent slidding windows.
    
    Parameters
    ----------
    CN2winLoc : uint8 array
        Resulting window locations from CN2 tree.
    winVals : float32 array
        Normalized slidding windows values of input frame's histogram.
    winLoc : uint8 array
        Slidding windows locations of input frame's histogram.
    
    Returns
    -------
    signCrit : float32
        KS significance test result.
    
    '''
    # First window comparison
    if CN2winLoc[0] == 0:
        plainCDF = np.cumsum(winVals[0, :])
        overCDF = np.cumsum(winVals[1, :])
        
        # Significance test on two windows
        signCrit = np.max(np.abs(plainCDF - overCDF))
    # Last window comparison
    elif CN2winLoc[0] == 224:
        plainCDF = np.cumsum(winVals[-1, :])
        overCDF = np.cumsum(winVals[winVals.shape[0] -2, :]) 
        
        # Significance test on two windows
        signCrit = np.max(np.abs(plainCDF - overCDF))
    # Remaining windows
    else:
        # Locate window
        idx = np.where(CN2winLoc[0] == winLoc[:, 0])[0]
        
        # Evaluate CDF of current window
        currCDF = np.cumsum(winVals[idx, :])
        
        # Evaluate CDF of previous window
        prevCDF = np.cumsum(winVals[idx-1, :])
        
        # Evaluate CDF of next window
        nxtCDF = np.cumsum(winVals[idx+1, :])
        
        # Significance test on all windows
        signCrit = np.max( ( np.abs(currCDF - prevCDF), 
                                np.abs(currCDF - nxtCDF) ) )
    # Return KS test results
    return signCrit