# -*- coding: utf-8 -*-
"""
"""

from numpy import (float32, uint8, reshape, zeros)
from cv2 import (TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, KMEANS_PP_CENTERS,
                 kmeans, bitwise_and, dilate, MORPH_CLOSE, findContours, 
                 RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
from sklearn.cluster import KMeans

# --------------------------------------------------------------------------- #
# K-Means Implementation Function - OpenCV
# --------------------------------------------------------------------------- #
def kmeans_cv(frame, n_clusters):
    '''
    K-Means function that utilizes the already built-in function found in the
    opencv library. The k-means clustering algorithm is utilized in order to
    separate potential present objects in the current frame. The algorithm 
    utilizes the Kmeans++ initialization. The criteria for the K-Means are 
    defined as, max number of iterations set to 300, and the acceptable error
    rate is set to 1e-4.
    
    Parameters
    ----------
    frame : uint8 array
        Input (background subtracted) frame.
    n_clusters : uint
        Number of clusters to segment input frame.
    
    Returns
    -------
    res_frame : uint8 array
        Clustered frame.
    
    '''
    # Define criteria
    criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 300, 1e-4)
    
    # Flatten input frame
    inpt_cv = float32(frame.reshape((-1, 3)))
    
    # Fit current frame to the k-means algorithm
    ret,label,center = kmeans(inpt_cv, n_clusters, None, criteria,
                               10, KMEANS_PP_CENTERS)
    
    # Obtain labels
    center = uint8(center)
    
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
    scikit-learn library. The k-means clustering algorithm is utilized in 
    order to separate potential present objects in the current frame. 
    This function utilizes the Kmeans++ initialization. The criteria for 
    the k-means are kept to their default values, which are 300 max iterations
    and acceptable error rate of 1e-4.
    
    Parameters
    ----------
    frame : unit8 array
        Input (background subtracted) frame.
    n_clusters : int
        Number of clusters to segment input frame.

    Returns
    -------
    res_frame : uint8 array
        Clustered frame.

    '''
    # Get image dimmentions
    row, colm, chns=  frame.shape
    
    # Flatten image values for the k-means algorithm
    inpt = reshape(frame, (row * colm, chns))
    
    # Initialize the k-means model
    kmeans = KMeans(n_clusters, init='k-means++')
    
    # Fit the input image into the model
    kmeans.fit(inpt)
    
    # Predict the closest cluster each sample in input image belongs to
    labels = kmeans.predict(inpt)
    
    # Output separated objects into image
    res_frame = zeros((row, colm, chns), dtype=uint8)
    
    # Initialize label index
    label_idx = 0
    
    # Loop through image dimentions
    for i in range(row):
        for k in range(colm):
            # At each iteration, select the corresponding cluster center of each label
            res_frame[i, k] = kmeans.cluster_centers_[labels[label_idx]]
            # Update label index
            label_idx += 1
    
    # Return resulting frame
    return res_frame


# --------------------------------------------------------------------------- #
# Image Processing Function - Morphological Operations & Contour capture
# --------------------------------------------------------------------------- #
def frame_proc(frame, fgmask, kernel, contour_size):
    '''
    Implemention of a number of morphological operations in order to 
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
    
    Parameters
    ----------
    frame : uint8 array
        Input frame.
    fgmask : bool array
        Background subtraction mask.
    kernel : uint8
        Morphological operations kernel size.
    contour_size : uint8
        Thershold of contour size to keep.
    
    Returns
    -------
    res2 : uint8 array
        Foreground image.
    res : 
        Foreground image to draw contours.
    contours : np.array
        Contour locations.
    
    '''
    # Self bitwise operation on current frame
    res = bitwise_and(frame,frame, mask=fgmask)
    
    # Morphologically dilate current frame
    e_im = dilate(fgmask, kernel, iterations=1)
    
    # Morphologically close current frame
    e_im = morphologyEx(e_im, MORPH_CLOSE, kernel)
    
    # Evaluate & capture each entire silhouettes
    contours, hierarchy = findContours(e_im, RETR_EXTERNAL,
                                          CHAIN_APPROX_SIMPLE)
    
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
    res2 = bitwise_and(frame,frame, mask=e_im)
    
    # Return resulting frames and detected contours
    return res2, res, contours