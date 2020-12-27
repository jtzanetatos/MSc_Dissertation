# -*- coding: utf-8 -*-
"""
"""

import cv2 as cv

# --------------------------------------------------------------------------- #
# Determine  available Video (Camera sensors) sources
# --------------------------------------------------------------------------- #
def ListSensors():
    '''
    Returns corresponding visual sensor to utilize.
    
    Returns
    -------
    list
        indices of available visual sensors.
    
    '''
    def detectSources():
        '''
        Probes for all available visual sensors & returns sensor indices
        
        Returns
        -------
        avail_cams : list
            Indices of available visual sensors.
        
        '''
        # Initialize list to store available sensors indices
        avail_cams = []
        
        # Initialize indices flag
        i = 0
        
        # Check for any possible input sources
        while True:
            # Test input source of current index
            cap = cv.VideoCapture(i)
            
            # If current index source is unavaiable, break
            if cap is None or not cap.isOpened():
                break
            
            # Else, append current source's index
            else:
                avail_cams.append(i)
            i += 1
            
        # Return list of available sources
        return avail_cams
    
    # Get available sources
    avail_source = detectSources()
    
    # If only one source has been detected, return its index
    if len(avail_source) == 1:
      return avail_source[0]
      
    # If two sources have been detected, return the index of the second source
    elif len(avail_source) == 2:
        return avail_source[1]