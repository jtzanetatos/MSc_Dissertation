# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from scipy.stats import entropy
from _Null_Hypothesis import NullHypothesis
from _KS_Significance_Test import KSTest

# --------------------------------------------------------------------------- #
# Implement adaptive windows of interest by means of KS statistical test
# --------------------------------------------------------------------------- #
def KSAdaptiveWindows(CN2winVals, CN2winLoc, histNorm, winVals, winLoc):
    '''
    Adaptive windows utilizing the Kolmogorov-Smirnov statistical test 
    containing the resulting histograms of each colour channel.
    In its current implementation, the windows of interest,
    resulting from the Rule Based Tree (rule_tree) function are utilized.
    
    The Null Hypothesis (H0) is defined as the shape of plain & overlapping 
    windows histograms overlap, therefore the current window expands accordingly
    untill they no longer overlap significantly.
    
    The Alternative Hypothesis is defined as the shape of plain & overlapping
    windows do not overlap significantly, therefore no need to expand current 
    window, keep current window as is
    
    It is assumed that the input arrays have dimentions of (8x32x3);
    however the only restriction is the number of rows that the algorithm
    takes into consideration.
    
    '''
    # If all colour channels present
    if ( np.sum(CN2winLoc[:, 0]) > 0 and
         np.sum(CN2winLoc[:, 1]) > 0 and
         np.sum(CN2winLoc[:, 2]) > 0 ):
        # KS Significance test
        signCrit = KSTest(CN2winVals)
        
        # If Alternative Hypothesis (H1) true
        if ( signCrit[0] >= 0.5 and
             signCrit[1] >= 0.5 and
             signCrit[2] >= 0.5):
            # Return input windows
            outWinvals = CN2winVals
            outBinlocs = CN2winLoc
            
        # If Null Hypothesis (H0) true
        else:
            # Expanded windows
            outWinvals,outBinlocs = NullHypothesis(CN2winLoc, CN2winVals)
            
    # Determine if Blue channel is present
    if np.sum(CN2winLoc[:, 0]) > 0:
        # Check if Red & Blue channels present only
        if np.sum(CN2winLoc[:, 2]) > 0:
            # KS Significance test
            signCrit = KSTest(CN2winVals)
            
            # If Alternative Hypothesis (H1) true
            if signCrit.all() >= 0.5:
                # Return input windows
                outWinvals = CN2winVals
                outBinlocs = CN2winLoc
            # If Null Hypothesis (H0) true
            else:
                # Expanded windows
                outWinvals,outBinlocs = NullHypothesis(CN2winLoc, CN2winVals)
        # Check if Blue & Green channels are present
        elif np.sum(CN2winLoc[:, 1]) > 0:
            # KS Significance test
            signCrit = KSTest(CN2winVals)
            
            # If Alternative Hypothesis (H1) true
            if ( signCrit[0] >= 0.5 and
                 signCrit[1] >= 0.5 ):
                
                # Return input windows
                outWinvals = CN2winVals
                outBinlocs = CN2winLoc
            # If Null Hypothesis (H0) true
            else:
                # Expanded windows
                outWinvals,outBinlocs = NullHypothesis(CN2winLoc, CN2winVals)
        # If only Blue Channel present
        else:
            # KS Significance test
            signCrit = KSTest(CN2winVals)
            
            # If Alternative Hypothesis (H1) true
            if signCrit >= 0.5:
                # Return input windows
                outWinvals = CN2winVals
                outBinlocs = CN2winLoc
            # If Null Hypothesis (H0) true
            else:
                # Expanded windows
                outWinvals,outBinlocs = NullHypothesis(CN2winLoc, CN2winVals)
    # If Green Channel present
    elif np.sum(CN2winLoc[:, 1]) > 0:
        # If Green & Blue channels present
        if np.sum(CN2winLoc[:, 0]) > 0:
            # KS Significance test
            signCrit = KSTest(CN2winVals)
            
            # If Alternative Hypothesis (H1) true
            if ( signCrit[0] >= 0.5 and
                 signCrit[1] >= 0.5 ):
                
                # Return input windows
                outWinvals = CN2winVals
                outBinlocs = CN2winLoc
            # If Null Hypothesis (H0) true
            else:
                # Expanded windows
                outWinvals,outBinlocs = NullHypothesis(CN2winLoc, CN2winVals)
        # If Green & Red channels present
        elif np.sum(CN2winLoc[:, 2]) > 0:
            # KS Significance test
            signCrit = KSTest(CN2winVals)
            
            # If Alternative Hypothesis (H1) true
            if ( signCrit[0] >= 0.5 and
                 signCrit[1] >= 0.5 ):
                
                # Return input windows
                outWinvals = CN2winVals
                outBinlocs = CN2winLoc
            # If Null Hypothesis (H0) true
            else:
                # Expanded windows
                outWinvals,outBinlocs = NullHypothesis(CN2winLoc, CN2winVals)
        # If Green Channel only
        else:
            # KS Significance test
            signCrit = KSTest(CN2winVals)
            
            # If Alternative Hypothesis (H1) true
            if signCrit >= 0.5:
                # Return input windows
                outWinvals = CN2winVals
                outBinlocs = CN2winLoc
            # If Null Hypothesis (H0) true
            else:
                # Expanded windows
                outWinvals,outBinlocs = NullHypothesis(CN2winLoc, CN2winVals)
    # If Red Chanel present
    elif np.sum(CN2winLoc[:, 2]) > 0:
        # If Red & Green Channels present
        if np.sum(CN2winLoc[:, 1]) > 0:
            # KS Significance test
            signCrit = KSTest(CN2winVals)
            
            # If Alternative Hypothesis (H1) true
            if ( signCrit[0] >= 0.5 and
                 signCrit[1] >= 0.5 ):
                
                # Return input windows
                outWinvals = CN2winVals
                outBinlocs = CN2winLoc
                
            # If Null Hypothesis (H0) true
            else:
                # Expanded windows
                outWinvals,outBinlocs = NullHypothesis(CN2winLoc, CN2winVals)
        # If Red & Blue channels present
        elif np.sum(CN2winLoc[:, 0]) > 0:
            # KS Significance test
            signCrit = KSTest(CN2winVals)
            
            # If Alternative Hypothesis (H1) true
            if ( signCrit[0] >= 0.5 and
                 signCrit[1] >= 0.5 ):
                
                # Return input windows
                outWinvals = CN2winVals
                outBinlocs = CN2winLoc
                
            # If Null Hypothesis (H0) true
            else:
                # Expanded windows
                outWinvals,outBinlocs = NullHypothesis(CN2winLoc, CN2winVals)
        # If only Red channel present
        else:
            # KS Significance test
            signCrit = KSTest(CN2winVals)
            
            # If Alternative Hypothesis (H1) true
            if signCrit >= 0.5:
                # Return input windows
                outWinvals = CN2winVals
                outBinlocs = CN2winLoc
                
            # If Null Hypothesis (H0) true
            else:
                # Expanded windows
                outWinvals,outBinlocs = NullHypothesis(CN2winLoc, CN2winVals)
    # If no colour channel present return zeros
    else:
        # Initialize arrays
        outBinlocs = np.zeros((32, 3), dtype=np.uint8)
        outWinvals = np.zeros((32, 3), dtype=np.float32)
    
    # Return resulting arrays
    return outBinlocs, outWinvals
