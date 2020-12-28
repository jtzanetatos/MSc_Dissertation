# -*- coding: utf-8 -*-
"""
"""

import numpy as np


def NullHypothesis(CN2winLoc, histNorm, winLoc, signCrit):
    '''
    Null Hypothesis of the KS significance test between input window derived
    from CN2 tree & its adjacent slidding windows.
    
    Null Hypothesis is defined as the shape of input window & its adjacent
    windows overlap, therefore the current window expands accordingly
    untill they no longer overlap significantly.
    
    Parameters
    ----------
    CN2winLoc : uint8 array
        Resulting window locations from CN2 tree.
    histNorm : float32 array
        Normalized histogram of input frame.
    winLoc : uint8 array
        Slidding windows locations of input frame's histogram.
    signCrit : float32
        KS significance test result.
    
    Returns
    -------
    out_winVals : float32 array
        Resulting expanded window values.
    out_winLoc : uint8 array
        Resulting expanded window locations.
    
    '''
    
    # Initialize index for Colour Channels
    idx = 0
    cdx = 1
    while signCrit >= 5.0:
        # Increase index value for next iteration
        idx += 1
        # Begining of histogram values(first window)
        if CN2winLoc[0] == 0:
            if idx <= 15:
                # Overlapping window expansion towards first element
                temp_winVals = histNorm[15 - idx:47]
            else:
                # Overlapping window expansion towards last element
                temp_winVals = histNorm[0:47+cdx]
                # Increment index
                cdx += 1
            # Output window expansion
            out_winVals = histNorm[0:32+idx]
            
            # Output window locations
            out_winLoc = np.arange(0, 32+idx, dtype=np.uint8)
            
            # Evaluate CDF of input window
            plainCDF = np.cumsum(out_winVals)
            
            # Evaluate CDF of overlapping window
            overCDF = np.cumsum(temp_winVals)
            
            # Significance test on two windows
            signCrit = np.max(np.abs(plainCDF - overCDF))
        # End of histogram values (last window)
        elif CN2winLoc[0] == 224:
            # Break when at limits of window
            if winLoc[-2, -1] + idx == 255:
                break
            # Input Channel Windows
            out_winVals = histNorm[np.int(winLoc[-1, 0])
                              -idx: np.int(winLoc[-1, -1])]
            
            # Overlapping window values
            temp_winVals = histNorm[np.int(winLoc[-2, 0]):
                                    np.int(winLoc[-2, -1]) + idx]
            
            # Output window locations
            out_winLoc = np.arange(winLoc[-1, 0]-idx,
                                   winLoc[-1, -1], dtype=np.uint8)
            
            # Evaluate CDF of input window
            plainCDF = np.cumsum(out_winVals)
            
            # Evaluate CDF of overlapping window
            overCDF = np.cumsum(temp_winVals)
            
            # Significance test on two windows
            signCrit = np.max(np.abs(plainCDF - overCDF))
        # Other cases
        else:
            # Locate window
            elem = np.where(CN2winLoc[0] == winLoc[:, 0])[0][0]
            # Break when at limits of window
            if ( winLoc[elem+1, -1] + idx == 255 or
                winLoc[elem-1, 0] - idx == 0 ):
                break
            
            # Check direction to expand
            if idx == 1:
                # Output window
                out_winVals = histNorm[np.int(winLoc[elem, 0])
                              -idx: np.int(winLoc[elem, -1]) + idx]
                
                # Output window locations
                out_winLoc = np.arange(winLoc[elem, 0]-idx,
                                   winLoc[elem, -1]+idx, dtype=np.uint8)
                # Evaluate CDF of current window
                currCDF = np.cumsum(out_winVals)
                
                # Safeguard expansion for first window case
                if elem-1 == 0:
                    # Evaluate CDF of previous window
                    prevCDF = np.cumsum(histNorm[winLoc[elem-1, 0]:
                                                 winLoc[elem-1, -1]+(2*idx)])
                else:
                    # Evaluate CDF of previous window
                    prevCDF = np.cumsum(histNorm[winLoc[elem-1, 0]-idx:
                                                 winLoc[elem-1, -1]+idx])
                
                # Safeguard expansion for last slidding window
                if elem+1 == winLoc.shape[0] - 1:
                    # Evaluate CDF of next window
                    nxtCDF = np.cumsum(histNorm[winLoc[elem+1, 0]-(2*idx):
                                                winLoc[elem+1, -1]])
                else:
                    # Evaluate CDF of next window
                    nxtCDF = np.cumsum(histNorm[winLoc[elem+1, 0]-idx:
                                                winLoc[elem+1, -1]+idx])
                
                # Significance test on all windows
                signCrit = np.max( ( np.abs(currCDF - prevCDF), 
                                        np.abs(currCDF - nxtCDF) ) )
                
                # Significance test results on both sets of comparison
                dirIndex = np.zeros(2, dtype=np.float32)
                
                dirIndex[0] = np.max(np.abs(currCDF - prevCDF))
                
                dirIndex[1] = np.max(np.abs(currCDF - nxtCDF))
            # Expand at appropriate direction
            else:
                # Expand at both directions
                if dirIndex[0] >= 5.0 and dirIndex[1] >= 5.0:
                    # Output window
                    out_winVals = histNorm[np.int(winLoc[elem, 0])
                                  -idx: np.int(winLoc[elem, -1]) + idx]
                    
                    # Output window locations
                    out_winLoc = np.arange(winLoc[elem, 0]-idx,
                                   winLoc[elem, -1]+idx, dtype=np.uint8)
                    
                    # Evaluate CDF of current window
                    currCDF = np.cumsum(out_winVals)
                    
                    # Safeguard expansion for first window case
                    if elem-1 == 0:
                        # Evaluate CDF of previous window
                        prevCDF = np.cumsum(histNorm[winLoc[elem-1, 0]:
                                                     winLoc[elem-1, -1]+(2*idx)])
                    else:
                        # Evaluate CDF of previous window
                        prevCDF = np.cumsum(histNorm[winLoc[elem-1, 0]-idx:
                                                     winLoc[elem-1, -1]+idx])
                    
                    # Safeguard expansion for last slidding window
                    if elem+1 == winLoc.shape[0] - 1:
                        # Evaluate CDF of next window
                        nxtCDF = np.cumsum(histNorm[winLoc[elem+1, 0]-(2*idx):
                                                    winLoc[elem+1, -1]])
                    else:
                        # Evaluate CDF of next window
                        nxtCDF = np.cumsum(histNorm[winLoc[elem+1, 0]-idx:
                                                    winLoc[elem+1, -1]+idx])
                    
                    # Significance test on all windows
                    signCrit = np.max( ( np.abs(currCDF - prevCDF), 
                                            np.abs(currCDF - nxtCDF) ) )
                    
                    # Significance test results on both sets of comparison
                    dirIndex = np.zeros(2, dtype=np.float32)
                    
                    dirIndex[0] = np.max(np.abs(currCDF - prevCDF))
                    
                    dirIndex[1] = np.max(np.abs(currCDF - nxtCDF))
                # Expant towards leftmost values
                elif dirIndex[0] >= 5.0 and dirIndex[1] < 5.0:
                    # Output window
                    out_winVals = histNorm[np.int(winLoc[elem, 0])
                                  -idx: np.int(winLoc[elem, -1])]
                    
                    # Output window locations
                    out_winLoc = np.arange(winLoc[elem, 0]-idx,
                                   winLoc[elem, -1], dtype=np.uint8)
                    
                    # Evaluate CDF of current window
                    currCDF = np.cumsum(out_winVals)
                    
                    # Evaluate CDF of previous window
                    prevCDF = np.cumsum(histNorm[winLoc[elem-1, 0]:
                                                 winLoc[elem-1, -1]+idx])
                    
                    # Significance test on all windows
                    signCrit = np.max(np.abs(currCDF - prevCDF))
                    
                    dirIndex[0] = np.max(np.abs(currCDF - prevCDF))
                # Expand towards rightmost values
                elif dirIndex[1] >= 5.0 and dirIndex[0] < 5.0:
                    # Output window
                    out_winVals = histNorm[np.int(winLoc[elem, 0])
                                  : np.int(winLoc[elem, -1]) + idx]
                    
                    # Output window locations
                    out_winLoc = np.arange(winLoc[elem, 0],
                                   winLoc[elem, -1]+idx, dtype=np.uint8)
                    
                    # Evaluate CDF of current window
                    currCDF = np.cumsum(out_winVals)
                    
                    # Evaluate CDF of next window
                    nxtCDF = np.cumsum(histNorm[winLoc[elem+1, 0]-idx:
                                                winLoc[elem+1, -1]])
                    
                    # Significance test on all windows
                    signCrit = np.max(np.abs(currCDF - nxtCDF))
                    
                    dirIndex[1] = np.max(np.abs(currCDF - nxtCDF))
    # Return resulting windows
    return out_winVals, out_winLoc
