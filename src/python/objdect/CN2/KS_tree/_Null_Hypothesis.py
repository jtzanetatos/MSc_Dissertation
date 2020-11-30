# -*- coding: utf-8 -*-
"""
"""

import numpy as np


# TODO: This function RETURNS Adaptive windows!!,
# DO NOT EXPAND IN UNITY! EACH CHANNEL MUST EXPAND INDEPENDENTLY!!!
# Return array of arrays to avoid padding
# Function needs to determine which channels are present
def NullHypothesis(final_win_loc, final_win_vals):
    
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
            # TODO: Implement validation check for both ends of histogram
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
    
    # Return resulting windows
