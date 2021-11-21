# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from scipy.stats import entropy
from .KS_Tree import KSAdaptiveWindows

# --------------------------------------------------------------------------- #
# Implement Rule Based Tree on RGB/Grayscale Windows
# --------------------------------------------------------------------------- #
def CN2(win_vals, win_binloc, histNorm):
    '''
    Implementation of a rule based decision tree, in order to find regions of
    interest, and determine number of clusters.
    
    Parameters
    ----------
    win_vals : float32 array
        Slidding window values of input frame's histogram.
    win_binloc : uint8 array
        Slidding window locations of input frame's histogram.
    
    Returns
    -------
    wind_vals : np.array
        Resulting winodws values.
    wind_bins : np.array
        Resulting windows locations.
    
    '''
    # Get shape of input windows
    try:
        rows, cols, chns = win_vals.shape
    # Grayscale frame
    except ValueError:
        rows, cols = win_vals.shape
        
        # Initialize list to store indices
        win_ind = []
        
        # Iterate over windows
        for i in range(0, rows-1):
            # Filter out black pixels
            if i == 0:
                # Plain window max value greater than overlapping
                if win_vals[i, 2:].max() > win_vals[i+1, :].max():
                    # Plain contains useful info & over 0.4
                    if ( win_vals[i, 2:].max() >= 0.4 and 
                    4 < win_vals[i, 2:].argmax() and 
                    cols - 4 > win_vals[i, 2:].argmax() ):
                        
                        # Append window index
                        win_ind.append(i)
                        
                    # Overlapping window contains useful info & over 0.4
                    elif ( win_vals[i+1, :].max() >= 0.4 and 
                    4 < win_vals[i+1, :].argmax() and 
                    cols-4 > win_vals[i+1, :].argmax() ):
                        
                        # Append window index
                        win_ind.append(i+1)
                        
                    # No window contains criteria, continue to next set
                    else:
                        continue
                # Overlapping window's max value greater than plain
                elif win_vals[i+1, :].max() > win_vals[i, 2:].max():
                    # Overlapping window contains useful info & over 0.4
                    if ( win_vals[i+1, :].max() >= 0.4 and 
                    4 < win_vals[i+1, :].argmax() and 
                    cols-4 > win_vals[i+1, :].argmax() ):
                        
                        # Append window index
                        win_ind.append(i+1)
                        
                    # Plain window contains useful info & over 0.4
                    elif ( win_vals[i, 2:].max() >= 0.4 and 
                    4 < win_vals[i, 2:].argmax() and 
                    cols-4 > win_vals[i, 2:].argmax() ):
                        
                        # Append window index
                        win_ind.append(i)
                        
                    # No window contains criteria, continue to next set
                    else:
                        continue
            # Other windows
            else:
                # Plain window max value greater than overlapping
                if win_vals[i, :].max() > win_vals[i+1, :].max():
                    # Plain contains useful info & over 0.4
                    if ( win_vals[i, 2:].max() >= 0.4 and 
                    4 < win_vals[i, :].argmax() and 
                    cols-4 > win_vals[i, :].argmax() ):
                        
                        # Append window index
                        win_ind.append(i)
                        
                    # Overlapping window contains useful info & over 0.4
                    elif ( win_vals[i+1, :].max() >= 0.4 and
                    4 < win_vals[i+1, :].argmax() and
                    cols-4 > win_vals[i+1, :].argmax() ):
                        
                        # Append window index
                        win_ind.append(i+1)
                        
                    # No window contains criteria, continue to next set
                    else:
                        continue
                # Overlapping window's max value greater than plain
                elif win_vals[i+1, :].max() > win_vals[i, 2:].max():
                    # Overlapping window contains useful info & over 0.4
                    if ( win_vals[i+1, :].max() >= 0.4 and 
                    4 < win_vals[i+1, :].argmax() and 
                    cols-4 > win_vals[i+1, :].argmax() ):
                        
                        # Append window index
                        win_ind.append(i+1)
                        
                    # Plain window contains useful info & over 0.4
                    elif ( win_vals[i, :].max() >= 0.4 and 
                    4 < win_vals[i, :].argmax() and 
                    cols-4 > win_vals[i, :].argmax() ):
                        
                        # Append window index
                        win_ind.append(i)
                        
                    # No window contains criteria, continue to next set
                    else:
                        continue
        # Initialize pruning indices list
        pruneInd = []
        # Find reduntant windows
        for index in range(len(win_ind[:, i])):
            # Iterate over window indices
            for l in range(index+1, len(win_ind[:, i])):
                # Remove redundancy
                if win_ind[index, i] == win_ind[l, i]:
                    pruneInd.append(l)
        # Prune redundant window
        for i in pruneInd:
            del win_ind[i]
        
        # Initialize final windows
        wind_vals = np.zeros((len(win_ind), cols), dtype=np.float32)
        
        # Initialize final window bin locations
        wind_bins = np.zeros((len(win_ind), cols), dtype=np.uint8)
        
        # Store useful windows
        for i in range(len(win_ind)):
            wind_vals[i, :] = win_vals[win_ind[i], :]
            
            wind_bins[i, :] = win_binloc[win_ind[i], :]
    # Frame is RGB
    else:
        # Initialize array to store window indices
        win_ind = np.full((rows, chns), np.nan)
        
        # Iterate over colour channels
        for c in range(chns):
            # Iterate over each row of windows(plain & overlapping)
            for i in range(0, rows-1):
                # Filter out black pixels
                if i == 0:
                    # Plain window max value greater than overlapping
                    if win_vals[i, 2:, c].max() > win_vals[i+1, :, c].max():
                        # Plain contains useful info & over 0.4
                        if ( win_vals[i, 2:, c].max() >= 0.4 and 
                        4 < win_vals[i, 2:, c].argmax() and 
                        cols-4 > win_vals[i, 2:, c].argmax() ):
                            
                            # Append window index
                            win_ind[i, c] = i
                            
                        # Overlapping window contains useful info & over 0.4
                        elif ( win_vals[i+1, :, c].max() >= 0.4 and 
                        4 < win_vals[i+1, :, c].argmax() and 
                        cols-4 > win_vals[i+1, :, c].argmax() ):
                            
                            # Append window index
                            win_ind[i, c] = i+1
                            
                        # No window contains criteria, continue to next set
                        else:
                            continue
                    # Overlapping window's max value greater than plain
                    elif win_vals[i+1, :, c].max() > win_vals[i, 2:, c].max():
                        # Overlapping window contains useful info & over 0.4
                        if ( win_vals[i+1, :, c].max() >= 0.4 and 
                        4 < win_vals[i+1, :, c].argmax() and 
                        cols-4 > win_vals[i+1, :, c].argmax() ):
                            
                            # Append window index
                            win_ind[i, c] = i+1
                            
                        # Plain window contains useful info & over 0.4
                        elif ( win_vals[i, 2:, c].max() >= 0.4 and 
                        4 < win_vals[i, 2:, c].argmax() and 
                        cols-4 > win_vals[i, 2:, c].argmax() ):
                            
                            # Append window index
                            win_ind[i, c] = i
                            
                        # No window contains criteria, continue to next set
                        else:
                            continue
                # Other windows
                else:
                    # Plain window max value greater than overlapping
                    if win_vals[i, :, c].max() > win_vals[i+1, :, c].max():
                        # Plain contains useful info & over 0.4
                        if ( win_vals[i, :, c].max() >= 0.4 and 
                        4 < win_vals[i, :, c].argmax() and 
                        cols-4 > win_vals[i, :, c].argmax() ):
                            
                            # Append window index
                            win_ind[i, c] = i
                            
                        # Overlapping window contains useful info & over 0.4
                        elif ( win_vals[i+1, :, c].max() >= 0.4 and
                        4 < win_vals[i+1, :, c].argmax() and
                        cols-4 > win_vals[i+1, :, c].argmax() ):
                            
                            # Append window index
                            win_ind[i, c] = i+1
                            
                        # No window contains criteria, continue to next set
                        else:
                            continue
                    # Overlapping window's max value greater than plain
                    elif win_vals[i+1, :, c].max() > win_vals[i, :, c].max():
                        # Overlapping window contains useful info & over 0.4
                        if ( win_vals[i+1, :, c].max() >= 0.4 and 
                        4 < win_vals[i+1, :, c].argmax() and 
                        cols-4 > win_vals[i+1, :, c].argmax() ):
                            
                            # Append window index
                            win_ind[i, c] = i+1
                            
                        # Plain window contains useful info & over 0.4
                        elif ( win_vals[i, :, c].max() >= 0.4 and 
                        4 < win_vals[i, :, c].argmax() and 
                        cols-4 > win_vals[i, :, c].argmax() ):
                            
                            # Append window index
                            win_ind[i, c] = i
                            
                        # No window contains criteria, continue to next set
                        else:
                            continue
        # Initialize arrays to store arrays
        wind_vals = np.empty(chns, dtype=np.object)
        wind_bins = np.empty(chns, dtype=np.object)
        
        # Iterate over channels
        for i in range(chns):
            # Prune reduntant windows
            for index in range(len(win_ind[:, i])):
                # Iterate over valid values
                if ~np.isnan(win_ind[index, i]):
                    # Iterate over window indices
                    for l in range(index+1, len(win_ind[:, i])):
                        # Remove redundancy
                        if win_ind[index, i] == win_ind[l, i]:
                            win_ind[l, i] = np.nan
            # Current channel contains window
            if win_ind[~np.isnan(win_ind[:, i]), i].shape[0] > 0:
                # Initialize current channel's ouput values array
                temp_vals = np.zeros((len(win_ind[~np.isnan(win_ind[:, i]), i]), cols),
                                     dtype=np.float32)
                
                # Initialize current channel's bin values array
                temp_bins = np.zeros((len(win_ind[~np.isnan(win_ind[:, i]), i]), cols),
                                     dtype=np.uint8)
                
                # Filter out invalid elements
                curr_ind = np.uint8(win_ind[~np.isnan(win_ind[:, i]), i])
                
                # Iterate over extracted indices
                for j, k in enumerate(curr_ind):
                    # Store corresponding window
                    temp_vals[j, :] = win_vals[np.uint8(k), :, i]
                    
                    # Corresponding bin locations
                    temp_bins[j, :] = win_binloc[np.uint8(k), :]
                # Append resulting arrays to list
                wind_vals[i] = temp_vals
                
                wind_bins[i] = temp_bins
    # Return resulting arrays
    # return wind_vals, wind_bins
    ksVals, ksLoc = KSAdaptiveWindows(wind_vals, wind_bins, 
                                      histNorm, win_vals, win_binloc)

def CN2Entropy(win_vals, win_binloc, histNorm):
    '''
    Implementation of a rule based decision tree, in order to find regions of
    interest, and determine number of clusters.
    
    Parameters
    ----------
    win_vals : float32 array
        Slidding window values of input frame's histogram.
    win_binloc : uint8 array
        Slidding window locations of input frame's histogram.
    
    Returns
    -------
    wind_vals : np.array
        Resulting winodws values.
    wind_bins : np.array
        Resulting windows locations.
    
    '''
    # Get shape of input windows
    try:
        rows, cols, chns = win_vals.shape
    # Grayscale frame
    except ValueError:
        rows, cols = win_vals.shape
        
        # Initialize list to store indices
        win_ind = []
        
        # Iterate over windows
        for i in range(0, rows-1):
            # Evaluate difference of windows entropy
            diffEntr = ( entropy(win_vals[i, :], base=2) -
                         entropy(win_vals[i+1, :], base=2) )
            # Plain window's entropy greater than overlapping's
            if diffEntr > 0:
                # Append window index
                win_ind.append(i)
                
            # Overlapping window's entropy greater than overlapping's
            elif diffEntr < 0:
                # Append window index
                win_ind.append(i+1)
                
            # TODO: Add other metric; pass when result is nan
            else:
                continue
        # Initialize pruning indices list
        pruneInd = []
        # Find reduntant windows
        for index in range(len(win_ind[:, i])):
            # Iterate over window indices
            for l in range(index+1, len(win_ind[:, i])):
                # Remove redundancy
                if win_ind[index, i] == win_ind[l, i]:
                    pruneInd.append(l)
        # Prune redundant window
        for i in pruneInd:
            del win_ind[i]
        # Initialize final windows
        wind_vals = np.zeros((len(win_ind), cols), dtype=np.float32)
        
        # Initialize final window bin locations
        wind_bins = np.zeros((len(win_ind), cols), dtype=np.uint8)
        
        # Store useful windows
        for i in range(len(win_ind)):
            wind_vals[i, :] = win_vals[win_ind[i], :]
            
            wind_bins[i, :] = win_binloc[win_ind[i], :]
    
    # Frame is RGB
    else:
        # Initialize array to store window indices
        win_ind = np.full((rows, chns), np.nan)
        
        # Iterate over colour channels
        for c in range(chns):
            # Iterate over each row of windows(plain & overlapping)
            for i in range(0, rows-1):
                # Evaluate difference of windows entropy
                diffEntr = ( entropy(win_vals[i, :, c], base=2) -
                             entropy(win_vals[i+1, :, c], base=2) )
                # Current window's entropy greater than overlapping's
                if diffEntr > 0:
                    # Append window index
                    win_ind[i, c] = i
                # Next window's entropy greater than overlapping's
                elif diffEntr < 0:
                    # Append window index
                    win_ind[i, c] = i+1
                    
                # TODO: Add other metric; pass when result is nan
                else:
                    continue
        # Initialize arrays to store arrays
        wind_vals = np.empty(chns, dtype=object)
        wind_bins = np.empty(chns, dtype=object)
        
        # Iterate over channels
        for i in range(chns):
            # Prune reduntant windows
            for index in range(len(win_ind[:, i])):
                # Iterate over valid values
                if ~np.isnan(win_ind[index, i]):
                    # Iterate over window indices
                    for l in range(index+1, len(win_ind[:, i])):
                        # Remove redundancy
                        if win_ind[index, i] == win_ind[l, i]:
                            win_ind[l, i] = np.nan
            # Current channel contains window
            if win_ind[~np.isnan(win_ind[:, i]), i].shape[0] > 0:
                # Initialize current channel's ouput values array
                temp_vals = np.zeros((len(win_ind[~np.isnan(win_ind[:, i]), i]), cols),
                                     dtype=np.float32)
                
                # Initialize current channel's bin values array
                temp_bins = np.zeros((len(win_ind[~np.isnan(win_ind[:, i]), i]), cols),
                                     dtype=np.uint8)
                
                # Filter out invalid elements
                curr_ind = win_ind[~np.isnan(win_ind[:, i]), i]
                
                # Iterate over extracted indices
                for j, k in enumerate(curr_ind):
                    # Store corresponding window
                    temp_vals[j, :] = win_vals[np.uint8(k), :, i]
                    
                    # Corresponding bin locations
                    temp_bins[j, :] = win_binloc[np.uint8(k), :]
                # Append resulting arrays to list
                wind_vals[i] = temp_vals
                
                wind_bins[i] = temp_bins
                
    # Return resulting arrays
    # return wind_vals, wind_bins
    ksVals, ksLoc = KSAdaptiveWindows(wind_vals, wind_bins, 
                                      histNorm, win_vals, win_binloc)
    return ksVals, ksLoc
