# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans

# --------------------------------------------------------------------------- #
# K-Means Implementation Function - OpenCV
# --------------------------------------------------------------------------- #
def kmeans_cv(frame, n_clusters):
    '''
    K-Means function that utilizes the already built-in function found in the
    opencv library. In the current implementation, the k-means clustering alg-
    orithm is utilized in order to separate potential present objects in the 
    current frame. The algorithm utilizes the Kmeans++ initialization.
    The criteria for the K-Means are defined as, max number of iterations set to
    300, and the acceptable error rate is set to 1e-4.
    
    Inputs:         frame: Current frame; can be any size or colour type
                    
                    n_clusters: Number of clusters for the algorithm to evaluate
                    
    Outputs:         res_frame: Resulting frame from the k-means algorithm
    '''
    # Define criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 300, 1e-4)
    
    # Flatten input frame
    inpt_cv = np.float32(frame.reshape((-1, 3)))
    
    # Fit current frame to the k-means algorithm
    ret,label,center = cv.kmeans(inpt_cv, n_clusters, None, criteria,
                               10, cv.KMEANS_PP_CENTERS)
    
    # Obtain labels
    center = np.uint8(center)
    
    # Evaluate new frame based on resulting labels
    res_frame = center[label.flatten()]
    
    # Reshape frame to its origianl dimmentions
    res_frame = res_frame.reshape((frame.shape))
    
    # Return resulting array
    return res_frame


# --------------------------------------------------------------------------- #
# K-Means Implementation Function - Scikit-Learn
# --------------------------------------------------------------------------- #
def kmeans_sk(frame, n_clusters):
    '''
    K-Means function that utilizes the already built-in function found in the
    scikit-learn library. In the current implementation, the k-means clustering
    algorithm is utilized in order to separate potential present objects in the
    current frame. This function utilizes the Kmeans++ initialization, as well
    as 2 CPU cores for faster processing. The criteria for the k-means are
    kept to their default values, which are 300 max iterations and acceptable
    error rate of 1e-4.
    
    Inputs:         frame: Current frame; can be any size or colour type
                    
                    n_clusters: Number of clusters for the algorithm to evaluate
                    
    Outputs:         res_frame: Resulting frame from the k-means algorithm
    '''
    # Get image dimmentions
    heigh, width, _ =  frame.shape
    
    # Flatten image values for the k-means algorithm
    inpt = np.reshape(frame, (width * heigh, 3))
    
    # Initialize the k-means model
    kmeans = KMeans(n_clusters, init='k-means++', n_jobs=-1)
    
    # Fit the input image into the model
    kmeans.fit(inpt)
    
    # Predict the closest cluster each sample in input image belongs to
    labels = kmeans.predict(inpt)
    
    # Output separated objects into image
    res_frame = np.zeros((heigh, width, 3), dtype=np.uint8)
    
    # Initialize label index
    label_idx = 0
    
    # Loop through image dimentions
    for i in range(heigh):
        for k in range(width):
            # At each iteration, select the corresponding cluster center of each label
            res_frame[i, k] = kmeans.cluster_centers_[labels[label_idx]]
            # Update label index
            label_idx += 1
    
    # Return resulting frame
    return(res_frame)


# --------------------------------------------------------------------------- #
# Image Processing Function - Morphological Operations & Contour capture
# --------------------------------------------------------------------------- #
def frame_proc(frame, fgmask, kernel, contour_size):
    '''
    Function that implements a number of morphological operations in order to 
    capture the contours of the detected objects (i.e. contours of interest)
    
    The function first perfoms an bitwise self addition of the input frame,
    utilizing the evaluated mask from the MOG2 algorithm.
    
    Afterwards, a morphological diation is performed on the frame in order to 
    close potential empty regions inside the contous of interest.
    
    The morphological closing is performed on the frame in order to capture the
    now filled contours of interest.
    
    The entire silhouette of each contour is evaluated and captured, in order
    to draw the detected contours; the contours that are below of a threshold
    value are deleted.
    
    Inputs:             frame: Input frame
    
                        fgmask: Evaluated mask for the current input frame
                        
                        kernel: Kernel of size 9x9
                        
                        contour_size: Threshold of accepted contours size,
                                      defined as 60 pixels.
                                      
    Outputs:            res2: Output frame, masked with the morphological closing
                              of input frame
                              
                        res: Frame that the contours are to be drawn.
                        
                        contours: The detected contous of current frame
    '''
    # Self bitwise operation on current frame
    res = cv.bitwise_and(frame,frame, mask= fgmask)
    
    # Morphologically dilate current frame
    e_im = cv.dilate(fgmask, kernel, iterations = 1)
    
    # Morphologically close current frame
    e_im = cv.morphologyEx(e_im, cv.MORPH_CLOSE, kernel)
    
    # Evaluate & capture each entire silhouettes
    contours, hierarchy = cv.findContours(e_im, cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_SIMPLE)
    
    # Remove contours that are lower than threshold's value
    temp = []
    num = 0
    # Loop through each detected contours
    for i in range(len(contours)):
        # If current contour size less than threshold's value, store contour
        if len(contours[i]) < contour_size:
            temp.append(i)
    # Loop through each contour that is less than threshold's value
    for i in temp:
        # Delete the contours that are less than threshold's value
        del contours[i - num]
        num = num + 1
    
    # Perform bitwise and operation using the morphological processed frame as
    # a mask
    res2 = cv.bitwise_and(frame,frame, mask = e_im)
    
    # Return resulting frames and detected contours
    return res2, res, contours