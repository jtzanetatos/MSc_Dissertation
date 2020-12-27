# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 12:45:39 2019

@author: iason
"""

import numpy as np
import cv2 as cv
import objdect as obj
import sys

def main():
    
    # Initialize frame stream source
    source = obj.utils.ListSensors()
    cap = cv.VideoCapture(source)
    
    # 3 by 3 rectangular window for morph operators
    kernel = np.ones((9, 9), dtype=np.uint8) 
    # Zivkovic MOG
    fgbg = cv.createBackgroundSubtractorMOG2(detectShadows = False)
    
    # contours px size to accept
    contour_size = 60
    
    # Iteration flag to check if MOG2 needs to stop
    mog_flg = 1
    
    # Initialize frame array
    prev_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res2 = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Initialize state of statistical tests
    mog2_stat = False
    dev_stat = False
    
    # Initialize mask array
    prev_fgmask = np.full((480, 640), 255,dtype=np.uint8)
    # Initialize k-means frame
    res_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Initialize number of clusters
    n_clusters = 0
    
    # Initialize arrays of histograms of current & previous frame
    curr_hist = np.zeros((256, 3), dtype=np.uint32)
    prev_hist = np.zeros((256, 3), dtype=np.uint32)
    
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Get input frame dimmentions
        width, height, col = frame.shape
        
        # If 10th frame
        if mog_flg == 10:
            
            # Evaluate histograms of past & current frame
            for i in range(col):
                curr_hist[:, i] = np.reshape(cv.calcHist(res2, [i], None, [256],
                         [0, 256]), 256)
                prev_hist[:, i] = np.reshape(cv.calcHist(prev_frame, [i], None,
                     [256], [0, 256]), 256)
            
            # Filter current and previous frames histograms
            filt_curr = obj.histogram_ops.HistBpf(curr_hist)
            filt_prev = obj.histogram_ops.HistBpf(prev_hist)
            
            # Normalize current & previous frames filtered histograms
            norm_curr = obj.histogram_ops.HistNorm(filt_curr)
            norm_prev = obj.histogram_ops.HistNorm(filt_prev)
            
            # Evaluate if K-S test indicates the MOG2 has converged
            mog2_stat = obj.BG_converge.KSMoG2(norm_curr, norm_prev)
            
            # Implement plain & overlapping windows
            win_vals, win_binloc = obj.histogram_ops.HistWindows(norm_curr)
            
            # Adaptive windows
            wind_vals, wind_bins = obj.CN2.CN2Entropy(win_vals, win_binloc, norm_curr)
            
            # Evaluate the result of the STD test
            # dev_stat = obj.BG_converge.HistDeviation(out_win_vals)
            # Cluster estimator
            # n_clusters = obj.CN2.KS_Tree.ClusterEstimator(wind_bins)
            
            # If MOG2 has converged
            if mog2_stat and n_clusters > 2:
                
                # Implement K-means clustering to segment detected objects
                res_frame = obj.image_processing.kmeans_sk(frame, n_clusters)
            else:
                #apply MOG2 and get foreground pixels
                fgmask = fgbg.apply(frame)
                res2, res, contours = obj.image_processing.frame_proc(frame,
                                                fgmask, kernel, contour_size)
                
        else:
            # If MOG2 has not converge
            if mog2_stat==False:
                #apply MOG2 and get foreground pixels
                fgmask = fgbg.apply(frame)
                res2, res, contours = obj.image_processing.frame_proc(frame,
                                                fgmask, kernel, contour_size)
            
            # Apply previous mask on the current frame
    #        else:
    #            (res2, res, contours) = f.frame_proc(frame, prev_fgmask, kernel, contour_size)
                # If number of clusters greater than two
    #            if n_clusters > 2:
                    # Implement K-means clustering to segment detected objects
    #                res_frame = f.kmeans_cv(frame, n_clusters)
        
        # Show resulting frames
        cv.imshow('Foreground', res2)
        cv.drawContours(res, contours, -1, (0, 0, 255), 2)
        cv.imshow('Contours', res)
        cv.imshow('K-Means results', res_frame)
        
    #    dev_list.append(dev_stat)
    #    mog2_list.append(mog2_stat)
    #    clst.append(n_clusters)
    #    
        # Store raw frame data
    #    raw_path = main_path +'/Measurements/Temp/raw_data'
    #    try:
    #        os.chdir(raw_path)
    #    except OSError:
    #        os.makedirs(raw_path)
    #    
    #    cv.imwrite(os.path.join(raw_path, str(n_frame) + 'raw_data.jpg'), frame ,
    #               [int(cv.IMWRITE_JPEG_QUALITY), 100])
    #    
    #    # Store foreground frame data
    #    fore_path = main_path + '/Measurements/Temp/fg_data'
    #    try:
    #        os.chdir(fore_path)
    #    except OSError:
    #        os.makedirs(fore_path)
    #    
    #    cv.imwrite(os.path.join(fore_path, str(n_frame) + 'fg_data.jpg'), res2, 
    #               [int(cv.IMWRITE_JPEG_QUALITY), 100])
    #    
    #    # Store Contouts data
    #    contour_path = main_path + '/Measurements/Temp/cont_data'
    #    try:
    #        os.chdir(contour_path)
    #    except OSError:
    #        os.makedirs(contour_path)
    #    
    #    cv.imwrite(os.path.join(contour_path, str(n_frame) + 'cont_data.jpg'), res, 
    #               [int(cv.IMWRITE_JPEG_QUALITY), 100])
        
    #    # Store K-Means data
    #    kmeans_path = main_path + '/Measurements/Temp/kmeans_data'
    #    cv.imwrite(os.path.join(kmeans_path, str(n_frame) + 'kmeans_data.jpg'), res_frame,
    #               [int(cv.IMWRITE_JPEG_QUALITY), 100])
        
        # n_frame += 1
        # Store current frame for next iteration
        if mog_flg == 1:
            prev_frame = res2
            
            # Increment flag for next iteration
            mog_flg += 1
            
            # If MOG2 has converged keep current mask
    #        if mog2_stat:
    #            prev_fgmask = fgmask
        
        # Reset flag for next iteration
        elif mog_flg == 10:
            mog_flg = 1
        
        else:
            # Increment by 1 the flag
            mog_flg += 1
        
        # To break, press the q key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        # To save the current frame press 's' key
        elif cv.waitKey(1) & 0xFF == ord('s'):
            cv.imwrite('result.jpg', res2, [int(cv.IMWRITE_JPEG_QUALITY), 100])
            print('Screenshot caputred successfully.')
        
    # Release capture & close all windows
    cap.release()
    cv.destroyAllWindows()
    
    #os.chdir(os.path.join(main_path + '/Measurements/Temp/'))
    ## Store clusters
    #with open('clst.txt', 'w') as file:
    #    for s in clst:
    #        file.write(str(s)+'\n')
    #
    ## Store dev
    #with open('dev.txt', 'w') as file:
    #    for s in dev_list:
    #        file.write(str(s)+'\n')
    #
    ## Store mog2
    #with open('mog.txt', 'w') as file:
    #    for s in mog2_list:
    #        file.write(str(s)+'\n')

if __name__ == "__main__":
    sys.exit(main())