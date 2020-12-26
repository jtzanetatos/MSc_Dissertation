# -*- coding: utf-8 -*-
"""
"""

import numpy as np


def ClusterEstimator(ksLoc):
    '''
    Estimation of present clusters found in input frame based on cross similarity
    of cross channel first element bin locations.
    
    Parameters
    ----------
    ksLoc : np.array
        Locations of every colour channel's windows.
    
    Returns
    -------
    n_clusters : uint
        Number of estimated clusters.
    
    '''
    # Initialize number of clusters
    n_clusters = 2
    
    # Find present colour channels
    nChns = []
    for i in range(len(ksLoc)):
        if ksLoc[i] is not None:
            nChns.append(i)
    
    # Colour channels present
    if len(nChns) > 0:
        # Initialize array to store starting locations of each window
        locInd = np.zeros(len(nChns), dtype=np.object)
        
        # Initialize array to track number of windows of each colour channel
        maxWin = np.zeros(len(nChns), dtype=np.uint8)
        # Iterate over colour channels
        for i in nChns:
            # Find colour channel with most windows
            maxWin[i] = len(ksLoc[i:i+1])
            # Initialize array to store first index of each window
            currInd = np.zeros(len(ksLoc[i]), dtype=np.object)
            
            # Iterate over windows
            for k in range(len(ksLoc[i])):
                # Track first location element
                currInd[k] = ksLoc[i][k][0]
            # Store resulting indices
            locInd[i] = currInd
        
        # One colour channel present
        if len(nChns) == 1:
            n_clusters += len(locInd[maxWin.argmax()])
            
            return n_clusters
        # Two colour channels present
        elif len(nChns) == 2:
            # Iterate over most dominant colour channel
            for i in range(len(locInd[maxWin.argmax()])):
                # Iterate over other colour channel
                for k in range(len(locInd[maxWin.argmin()])):
                    # Compare location elements
                    if locInd[maxWin.argmax()][i] == locInd[maxWin.argmin()][k]:
                        # Increment number of clusters by one
                        n_clusters += 1
                    # Increment number of clusters by 2
                    else:
                        n_clusters += 2
            return n_clusters
        # All colour channels present
        elif len(nChns) == 3:
            # Keep most dominant colour channel's index
            dIdx = maxWin.argmax()
            
            # Remove dominant colour channel index
            maxWin[maxWin.argmax()] = np.nan
            
            # Apply mask
            maxWin = maxWin[~np.isnan(maxWin)]
            
            # Iterate over most dominant colour channel
            for i in range(len(locInd[dIdx])):
                # Iterate over other colour channels
                for j in maxWin:
                    # Iterate over current other colour channel
                    for k in range(len(locInd[j])):
                        # Compare location elements
                        if locInd[dIdx][i] == locInd[j][k]:
                            # Increment number of clusters by one
                            n_clusters += 1
                        # Increment number of clusters by 2
                        else:
                            n_clusters += 2
    # No colour channels present
    else:
        return n_clusters