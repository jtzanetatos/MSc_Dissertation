# -*- coding: utf-8 -*-
"""
"""

import numpy as np

# --------------------------------------------------------------------------- #
# Implement Rule Based Tree on RGB Windows Function
# --------------------------------------------------------------------------- #
def CN2(win_vals, over_vals, win_binloc, over_binloc):
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

    
    # Return resulting arrays
    return final_win_vals, final_win_loc