# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 19:54:29 2020

@author: iason
"""

import numpy as np

def KSTest(sigWind):
    '''
    Kolmogorov-Smirnoff test on sliding windows to determine significance on 
    distribution.
    
    Parameters
    ----------
    sigWind : float32 multidimensional array
        Array containing sliding windows.
    
    Returns
    -------
    signCrit : float32
        Results of KS test/significance.
    
    '''
    
    # Get input shape
    try:
        row, colm = sigWind.shape
    
    except ValueError:
        # Set column to 1
        colm = 1
    
    # Initialize array to store KS test results
    signCrit = np.zeros(colm, dtype=np.float32)
    
    # Evaluate CDF of sliding windows
    for i in range(colm):
        plainCDF = np.cumsum(sigWind[:, ]) / 256
        overCDF = np.cumsum(sigWind[:, ]) / 256
        
        signCrit[i] = np.max(np.abs(plainCDF - overCDF))
    # Return KS test results
    return signCrit