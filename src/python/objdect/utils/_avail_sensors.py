# -*- coding: utf-8 -*-
"""
"""

import cv2 as cv

# --------------------------------------------------------------------------- #
# Determine  available Video (Camera sensors) sources
# --------------------------------------------------------------------------- #
def ListSensors():
    '''
    Function that prompts user to choose camera.
    In its current implementation, it is limited to returing only two possible
    options.
    
    Input: available cameras list
    Output: Valid user choise
    '''
    def detectSources():
        '''
        Function that detects any available camera.
        Input: max value of index integer number
        Output: list of available cameras currently connected.
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